"""Adversarial and end-to-end tests for multi-bank Latent Lattice state isolation (arXiv:2611.21608)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from warp_modeld.canary import CanarySuite
from warp_modeld.errors import (
    CheckpointMismatchError,
    LatticeFlushError,
    RuntimeFlagDriftError,
)
from warp_modeld.fingerprint import (
    CheckpointFingerprint,
    RuntimeFlags,
    compute_zone_manifest_id,
)
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.server import WarpUnixRpcClient, WarpUnixRpcServer
from warp_modeld.state_banks import STATE_BANK_KV, STATE_BANK_LATTICE, STATE_BANK_SSM
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    fake_sha,
    make_adapter_bundle_dict,
    make_joint_bounds,
    make_lineage,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer


def _make_lattice_checkpoint(*, has_ssm: bool = True, layers: int = 24, width: int = 256) -> CheckpointFingerprint:
    return CheckpointFingerprint(
        model_id="Lattice-GDN-406M",
        layout_sha256=fake_sha("layout-lattice-gdn"),
        weights_sha256=fake_sha("weights-lattice-gdn"),
        config_sha256=fake_sha("config-lattice-gdn"),
        tokenizer_sha256=fake_sha("tokenizer-lattice-gdn"),
        chat_template_sha256=fake_sha("chat-template-lattice-gdn"),
        safetensors_provenance_sha256=fake_sha("safetensors-lattice-gdn"),
        has_ssm_recurrent=has_ssm,
        has_horizontal_lattice=True,
        lattice_layers=layers,
        lattice_width=width,
    )


def _make_lattice_runtime_flags(*, lattice_flush_supported: bool = True) -> RuntimeFlags:
    return RuntimeFlags(
        llama_server_commit="7fe450e19",
        quantization="Q4_K_M",
        ctx_size=16384,
        kv_cache_type="q8_0",
        ctx_checkpoints=0,
        swa_checkpoints=0,
        lora_init_without_apply=True,
        rope_args="",
        chat_template_args="thinking=off",
        has_horizontal_lattice=True,
        lattice_flush_supported=lattice_flush_supported,
    )


def _make_lattice_bundle(
    ckpt: CheckpointFingerprint,
    flags: RuntimeFlags,
    *,
    zone_manifest_id: str | None = None,
) -> dict:
    raw = make_adapter_bundle_dict(
        adapter_id="mde:corpus",
        domain="corpus",
        llama_adapter_id=0,
        checkpoint=ckpt,
        lineage=make_lineage(),
        runtime_flags=flags,
    )
    raw["checkpoint"]["has_ssm_recurrent"] = ckpt.has_ssm_recurrent
    raw["checkpoint"]["has_horizontal_lattice"] = ckpt.has_horizontal_lattice
    raw["checkpoint"]["lattice_layers"] = ckpt.lattice_layers
    raw["checkpoint"]["lattice_width"] = ckpt.lattice_width
    if zone_manifest_id is not None:
        raw["zone_manifest_id"] = zone_manifest_id
    return raw


def _make_zone_manifest() -> dict:
    body = {
        "schema": "zone-manifest/v0.1",
        "mde_version": "0.1.0",
        "precedence_order": ["formal_proof", "condensed_json", "ui_module", "code", "prose"],
        "zones": [
            {"zone": "formal_proof", "l1_bank_tag": 3, "default_l2_scales": {"formal-verification": 1.0}},
            {"zone": "condensed_json", "l1_bank_tag": 4, "default_l2_scales": {"corpus": 0.0}},
            {"zone": "ui_module", "l1_bank_tag": 2, "default_l2_scales": {"weft-eido": 1.0}},
            {"zone": "code", "l1_bank_tag": 1, "default_l2_scales": {"corpus": 0.4}},
            {"zone": "prose", "l1_bank_tag": 0, "default_l2_scales": {"corpus": 1.0}},
        ],
    }
    manifest = dict(body)
    manifest["manifest_id"] = compute_zone_manifest_id(body)
    return manifest


class TestLatentLatticeStateIsolation(unittest.TestCase):
    def test_contrast_kv_only_flush_allows_horizontal_lattice_to_recontaminate_fresh_kv_writes(self) -> None:
        """Prove why KV-only erase is insufficient on Latent Lattice models (arXiv:2611.21608)."""
        mock = MockLlamaServer(has_ssm_recurrent=True, lattice_layers=24)
        url = mock.start()
        try:
            raw_client = LlamaServerClient(url)
            # Step t: adapter runs and taints BOTH kv_cache and 24-layer horizontal lattice streams.
            raw_client.chat_completion(
                [{"role": "user", "content": "step t under adapter"}],
                lora=[{"id": 0, "scale": 1.0}],
                id_slot=0,
            )
            self.assertTrue(mock.state.slots[0].recurrent_state_tainted)
            self.assertTrue(mock.state.slots[0].lattice_state_tainted)
            self.assertEqual(mock.state.slots[0].lattice_layers_tainted, 24)

            # Naive KV-only flush clears kv_cache while leaving horizontal lattice latents poisoned.
            resp = raw_client.erase_slot(0, banks=[STATE_BANK_KV])
            self.assertEqual(resp["banks_flushed"], [STATE_BANK_KV])
            self.assertFalse(mock.state.slots[0].recurrent_state_tainted)
            self.assertTrue(mock.state.slots[0].lattice_state_tainted)

            # Step t+1 under Base W0 ("lora": []): surviving horizontal latents condition KV writes
            # and re-contaminate the freshly flushed KV cache immediately!
            recontam = raw_client.chat_completion(
                [{"role": "user", "content": "step t+1 base W0"}],
                lora=[],
                id_slot=0,
            )
            content = recontam["choices"][0]["message"]["content"]
            self.assertIn("[LATTICE_KV_RECONTAMINATION:layers=24,lora=0]", content)
            self.assertTrue(mock.state.slots[0].recurrent_state_tainted)
        finally:
            mock.stop()

    def test_stage1_refuses_lattice_checkpoint_when_backend_lacks_lattice_flush(self) -> None:
        mock = MockLlamaServer(has_ssm_recurrent=False, lattice_layers=24)
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                ckpt = _make_lattice_checkpoint(has_ssm=False, layers=24)
                unsafe_flags = _make_lattice_runtime_flags(lattice_flush_supported=False)
                bundle = _make_lattice_bundle(ckpt, unsafe_flags)

                signer = HmacSha256Signer(b"lattice-stage1-key-32-bytes-0001", key_id="warp-lat-v1")
                modeld = WarpModeld(
                    require_adapter_file=False,
                    pinned_checkpoint=ckpt,
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=unsafe_flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(Path(td) / "ledger.jsonl", signer),
                )
                with self.assertRaises(RuntimeFlagDriftError) as ctx:
                    modeld.mount_stack([bundle])
                self.assertEqual(ctx.exception.code, "ERR_UNSAFE_RUNTIME_FLAGS")
                self.assertEqual(len(mock.state.request_log), 0)
        finally:
            mock.stop()

    def test_stage1_rejects_spoofed_non_lattice_bundle_on_lattice_host(self) -> None:
        mock = MockLlamaServer(has_ssm_recurrent=True, lattice_layers=24)
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                ckpt = _make_lattice_checkpoint(has_ssm=True, layers=24)
                flags = _make_lattice_runtime_flags(lattice_flush_supported=True)
                spoofed = _make_lattice_bundle(ckpt, flags)
                spoofed["checkpoint"]["has_horizontal_lattice"] = False
                spoofed["checkpoint"]["lattice_layers"] = 0
                spoofed["checkpoint"]["lattice_width"] = 0

                signer = HmacSha256Signer(b"lattice-spoof-key-32-bytes-00001", key_id="warp-lat-v1")
                modeld = WarpModeld(
                    require_adapter_file=False,
                    pinned_checkpoint=ckpt,
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(Path(td) / "ledger.jsonl", signer),
                )
                with self.assertRaises(CheckpointMismatchError):
                    modeld.mount_stack([spoofed])
                self.assertEqual(len(mock.state.request_log), 0)
        finally:
            mock.stop()

    def test_kv_only_rollback_fails_closed_with_err_lattice_flush_failed_and_quarantines(self) -> None:
        mock = MockLlamaServer(has_ssm_recurrent=True, lattice_layers=24)
        url = mock.start()
        key = b"lattice-quarantine-key-32-bytes1"
        signer = HmacSha256Signer(key, key_id="warp-lat-v1")
        try:
            with tempfile.TemporaryDirectory() as td:
                ledger_path = Path(td) / "ledger.jsonl"
                ckpt = _make_lattice_checkpoint(has_ssm=True, layers=24)
                flags = _make_lattice_runtime_flags(lattice_flush_supported=True)
                bundle = _make_lattice_bundle(ckpt, flags)

                modeld = WarpModeld(
                    require_adapter_file=False,
                    pinned_checkpoint=ckpt,
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(ledger_path, signer),
                )
                modeld.mount_stack([bundle])
                modeld.infer([{"role": "user", "content": "populate all 3 state banks"}])
                self.assertTrue(mock.state.slots[0].lattice_state_tainted)

                # Fault injection: backend only flushes kv_cache on slot 0, omitting horizontal lattice!
                mock.state.kv_only_erase_slot_ids.add(0)
                with self.assertRaises(LatticeFlushError) as ctx:
                    modeld.rollback(reason="emergency_kill", engage_kill_switch=True)
                self.assertEqual(ctx.exception.code, "ERR_LATTICE_FLUSH_FAILED")

                status = modeld.status()
                self.assertTrue(status["state_quarantined"])
                self.assertEqual(status["quarantine_code"], "ERR_LATTICE_FLUSH_FAILED")

                # Subsequent inference and completion MUST fail closed with ERR_LATTICE_FLUSH_FAILED.
                with self.assertRaises(LatticeFlushError) as inf_ctx:
                    modeld.infer([{"role": "user", "content": "must be blocked"}])
                self.assertEqual(inf_ctx.exception.code, "ERR_LATTICE_FLUSH_FAILED")

                with self.assertRaises(LatticeFlushError) as comp_ctx:
                    modeld.complete("must also be blocked")
                self.assertEqual(comp_ctx.exception.code, "ERR_LATTICE_FLUSH_FAILED")

                receipts = LedgerVerifier.verify_file(ledger_path, signer)
                self.assertEqual([r.event for r in receipts], ["mount", "reject"])
                self.assertEqual(receipts[1].error_code, "ERR_LATTICE_FLUSH_FAILED")
        finally:
            mock.stop()

    def test_lying_backend_claiming_lattice_flush_while_retaining_taint_is_quarantined(self) -> None:
        mock = MockLlamaServer(has_ssm_recurrent=False, lattice_layers=24)
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                ckpt = _make_lattice_checkpoint(has_ssm=False, layers=24)
                flags = _make_lattice_runtime_flags(lattice_flush_supported=True)
                bundle = _make_lattice_bundle(ckpt, flags)

                signer = HmacSha256Signer(b"lattice-lie-test-key-32-bytes-01", key_id="warp-lat-v1")
                modeld = WarpModeld(
                    require_adapter_file=False,
                    pinned_checkpoint=ckpt,
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(Path(td) / "ledger.jsonl", signer),
                )
                modeld.mount_stack([bundle])
                modeld.infer([{"role": "user", "content": "taint horizontal lattice"}])

                # Fault injection: POST response lies ("banks_flushed": [...]) while GET /slots still reports taint!
                mock.state.lie_about_lattice_flush_slot_ids.add(0)
                with self.assertRaises(LatticeFlushError) as ctx:
                    modeld.rollback(reason="catch_lying_backend")
                self.assertEqual(ctx.exception.code, "ERR_LATTICE_FLUSH_FAILED")
                self.assertTrue(modeld.status()["state_quarantined"])
        finally:
            mock.stop()

    def test_atomic_multibank_rollback_restores_exact_base_w0_with_signed_per_bank_receipts(self) -> None:
        mock = MockLlamaServer(has_ssm_recurrent=True, lattice_layers=24)
        url = mock.start()
        key = b"lattice-atomic-key-32-bytes-0001"
        signer = HmacSha256Signer(key, key_id="warp-lat-v1")
        zone_manifest = _make_zone_manifest()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-lattice.sock"
                ledger_path = tmp / "ledger.jsonl"

                ckpt = _make_lattice_checkpoint(has_ssm=True, layers=24, width=256)
                flags = _make_lattice_runtime_flags(lattice_flush_supported=True)
                bundle = _make_lattice_bundle(ckpt, flags, zone_manifest_id=zone_manifest["manifest_id"])

                modeld = WarpModeld(
                    require_adapter_file=False,
                    pinned_checkpoint=ckpt,
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(ledger_path, signer),
                    pinned_zone_manifest=zone_manifest,
                )
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    rpc = WarpUnixRpcClient(sock_path)
                    ident = rpc.call("wmcp_checkpoint_identify")
                    self.assertTrue(ident["result"]["has_horizontal_lattice"])
                    self.assertEqual(
                        ident["result"]["required_state_banks"],
                        [STATE_BANK_KV, STATE_BANK_SSM, STATE_BANK_LATTICE],
                    )

                    mounted = rpc.call(
                        "wmcp_slot_mount",
                        {"bundles": [bundle], "zone_manifest_id": zone_manifest["manifest_id"]},
                    )
                    self.assertIn("result", mounted)
                    self.assertEqual(len(mounted["result"]["bank_flush_receipts"]), 2)
                    self.assertEqual(
                        mounted["result"]["bank_flush_receipts"][0]["banks_confirmed"],
                        [STATE_BANK_KV, STATE_BANK_SSM, STATE_BANK_LATTICE],
                    )

                    active_turn = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "active lattice turn"}],
                            "zone_manifest_id": zone_manifest["manifest_id"],
                        },
                    )
                    self.assertIn("[ACTIVE_LORA:0]", active_turn["result"]["response"]["choices"][0]["message"]["content"])
                    self.assertTrue(mock.state.slots[0].lattice_state_tainted)
                    self.assertTrue(mock.state.slots[0].ssm_state_tainted)

                    # Atomic multi-bank kill switch clears kv_cache + ssm_recurrent + lattice_horizontal_streams!
                    ks = rpc.call("warp_kill_switch", {"reason": "atomic_lattice_rollback"})
                    self.assertIn("result", ks)
                    self.assertEqual(ks["result"]["slots_flushed"], [0, 1])
                    for brec in ks["result"]["bank_flush_receipts"]:
                        self.assertTrue(brec["verified_clean"])
                        self.assertEqual(brec["zone_manifest_id"], zone_manifest["manifest_id"])
                        self.assertEqual(
                            brec["banks_confirmed"],
                            [STATE_BANK_KV, STATE_BANK_SSM, STATE_BANK_LATTICE],
                        )
                    self.assertFalse(mock.state.slots[0].lattice_state_tainted)
                    self.assertFalse(mock.state.slots[0].ssm_state_tainted)
                    self.assertFalse(mock.state.slots[0].recurrent_state_tainted)

                    # Post-rollback turn restores exact clean Base W0 with zero KV re-contamination!
                    post_turn = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "clean post-rollback base"}],
                            "zone_manifest_id": zone_manifest["manifest_id"],
                        },
                    )
                    text = post_turn["result"]["response"]["choices"][0]["message"]["content"]
                    self.assertIn("[BASE_W0]", text)
                    self.assertNotIn("LATTICE_KV_RECONTAMINATION", text)
                    self.assertNotIn("RESIDUAL_RECURRENT_LEAK", text)

                    # Signed HMAC-SHA256 ledger preserves exact per-bank flush evidence.
                    receipts = LedgerVerifier.verify_file(ledger_path, signer)
                    self.assertEqual([r.event for r in receipts], ["mount", "kill_switch"])
                    self.assertEqual(
                        receipts[1].details["required_state_banks"],
                        [STATE_BANK_KV, STATE_BANK_SSM, STATE_BANK_LATTICE],
                    )
                    self.assertEqual(len(receipts[1].details["bank_flush_receipts"]), 2)
                finally:
                    server.stop()
        finally:
            mock.stop()


if __name__ == "__main__":
    unittest.main()

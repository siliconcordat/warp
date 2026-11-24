"""Unit and adversarial tests for native L2 weft ingestion and L5 zone-manifest/v0.1 enforcement."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from warp_modeld.certificate import FixedRadixCertificate, verify_division_free
from warp_modeld.errors import (
    CertificateError,
    CheckpointMismatchError,
    ForgedCanaryError,
    RuntimeFlagDriftError,
    ZoneManifestDriftError,
)
from warp_modeld.fingerprint import (
    AdapterBundle,
    CheckpointFingerprint,
    compute_zone_manifest_id,
    sha256_json,
    verify_zone_manifest_dict,
)
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier
from warp_modeld.server import WarpUnixRpcClient, WarpUnixRpcServer
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    fake_sha,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_runtime_flags,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer
from warp_modeld.canary import CanarySuite
from warp_modeld.ledger import SignedLedger
from warp_modeld.llama_client import LlamaServerClient


def _make_weft_cert_dict(*, exp2: int = 24, frob: int | None = None, spec: int | None = None) -> tuple[dict, str]:
    scale = 1 << exp2
    cert = {
        "radix_scale_exp2": exp2,
        "frobenius_sq_scaled": frob if frob is not None else scale,
        "spectral_norm_sq_scaled": spec if spec is not None else (scale // 4),
        "leakage_energy_scaled": scale // 1000,
        "max_safety_dot_sq_scaled": scale // 10000,
        "projector_epoch": 1,
        "factor_digest": fake_sha("weft-factor-corpus-v1"),
        "quant_dtype": "bf16",
    }
    raw = json.dumps(cert, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return cert, hashlib.sha256(raw).hexdigest()


def _make_weft_manifest(cert_dict: dict, cert_sha: str) -> dict:
    wf_runtime = {
        "quantization_mode": "bf16",
        "mamba_cache_mode": "none",
        "enable_thinking": False,
        "tool_call_parser": "none",
        "rope_scaling": "default",
        "ctx_size": 16384,
    }
    return {
        "schema": "weft-binding/v0.1",
        "adapter_name": "mde:corpus",
        "adapter_class": "class1_rederivable",
        "layout_sha256": fake_sha("weft-layout-qwen"),
        "checkpoint_fingerprint": fake_sha("weft-weights-qwen"),
        "n_tensors": 124,
        "config_hashes": {"config.json": fake_sha("weft-config-json")},
        "tokenizer_hashes": {
            "tokenizer.json": fake_sha("weft-tokenizer-json"),
            "chat_template.jinja": fake_sha("weft-chat-template"),
        },
        "plane1_hashes": {"clean-writing.md": fake_sha("plane1-cw")},
        "derivation_recipe": {"compiler": "betaedit-f64", "rank": 16, "seed": 42},
        "runtime_flags": wf_runtime,
        "runtime_flags_sha256": sha256_json(wf_runtime),
        "factor_digest": cert_dict["factor_digest"],
        "certificate_sha256": cert_sha,
    }


def _make_zone_manifest(*, formal_tag: int = 3) -> dict:
    body = {
        "schema": "zone-manifest/v0.1",
        "mde_version": "0.1.0",
        "precedence_order": ["formal_proof", "condensed_json", "ui_module", "code", "prose"],
        "zones": [
            {"zone": "formal_proof", "l1_bank_tag": formal_tag, "default_l2_scales": {"formal-verification": 1.0}},
            {"zone": "condensed_json", "l1_bank_tag": 4, "default_l2_scales": {"corpus": 0.0}},
            {"zone": "ui_module", "l1_bank_tag": 2, "default_l2_scales": {"weft-eido": 1.0}},
            {"zone": "code", "l1_bank_tag": 1, "default_l2_scales": {"corpus": 0.4}},
            {"zone": "prose", "l1_bank_tag": 0, "default_l2_scales": {"corpus": 1.0}},
        ],
    }
    manifest = dict(body)
    manifest["manifest_id"] = compute_zone_manifest_id(body)
    return manifest


class TestWeftNativeIngestion(unittest.TestCase):
    def test_from_weft_dict_exact_integer_conversion_and_digest_verification(self) -> None:
        weft_cert, weft_sha = _make_weft_cert_dict(exp2=24)
        base_sha = fake_sha("base-w0")
        adapter_sha = fake_sha("gguf-lora-bytes")

        cert = FixedRadixCertificate.from_weft_dict(
            weft_cert,
            adapter_id="mde:corpus",
            adapter_sha256=adapter_sha,
            base_weights_sha256=base_sha,
            expected_weft_sha256=weft_sha,
        )
        self.assertEqual(cert.radix_scale, 1 << 24)
        self.assertEqual(cert.frobenius_sq_scaled, 1 << 24)
        res = verify_division_free(
            cert,
            make_single_bounds(),
            expected_adapter_id="mde:corpus",
            expected_adapter_sha256=adapter_sha,
            expected_base_weights_sha256=base_sha,
        )
        self.assertTrue(res.passed)

    def test_from_weft_dict_rejects_bool_float_out_of_range_impossible_norms_and_tampered_digest(self) -> None:
        base_cert, base_sha = _make_weft_cert_dict(exp2=24)
        common = {
            "adapter_id": "mde:corpus",
            "adapter_sha256": fake_sha("a"),
            "base_weights_sha256": fake_sha("b"),
        }

        bad_mutations = [
            {"radix_scale_exp2": True},
            {"radix_scale_exp2": 24.0},
            {"radix_scale_exp2": 7},
            {"radix_scale_exp2": 49},
            {"frobenius_sq_scaled": 100.5},
            {"frobenius_sq_scaled": False},
            {"frobenius_sq_scaled": -1},
            {"spectral_norm_sq_scaled": True},
            {"leakage_energy_scaled": 0.0},
            {"max_safety_dot_sq_scaled": False},
            {"projector_epoch": True},
            {"projector_epoch": -1},
            {"factor_digest": "not-a-sha256"},
            {"quant_dtype": ""},
            # Impossible matrix norms: spectral > frobenius.
            {"frobenius_sq_scaled": 100, "spectral_norm_sq_scaled": 101},
        ]
        for patch in bad_mutations:
            mutated = dict(base_cert)
            mutated.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(CertificateError):
                    FixedRadixCertificate.from_weft_dict(mutated, **common)

        # Tampered expected_weft_sha256 fails closed.
        with self.assertRaises(CertificateError):
            FixedRadixCertificate.from_weft_dict(
                base_cert,
                expected_weft_sha256=fake_sha("wrong-weft-cert-digest"),
                **common,
            )

        # Valid digest matches cleanly.
        ok = FixedRadixCertificate.from_weft_dict(base_cert, expected_weft_sha256=base_sha, **common)
        self.assertEqual(ok.radix_scale, 1 << 24)

    def test_from_weft_binding_end_to_end_native_mount_without_bridge(self) -> None:
        weft_cert, weft_sha = _make_weft_cert_dict(exp2=24)
        weft_manifest = _make_weft_manifest(weft_cert, weft_sha)
        pinned_ckpt = CheckpointFingerprint.from_weft_binding(weft_manifest, model_id="TinyStories-1M")
        lineage = make_lineage()
        flags = make_runtime_flags()

        mock = MockLlamaServer(model_bytes=b"weft-weights-qwen")
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                gguf_file = tmp / "corpus-f16.gguf"
                # Minimal valid GGUF v3 F16 LoRA (rank=2, alpha=2.0, B=0.5*I_2, A=I_2 -> frob^2=0.5<=1.0, spec^2=0.25<=0.25).
                def _pstr(s: str) -> bytes:
                    raw = s.encode("utf-8")
                    return struct.pack("<Q", len(raw)) + raw

                kv_items = [
                    ("adapter.lora.alpha", 6, struct.pack("<f", 2.0)),
                    ("adapter.type", 8, _pstr("lora")),
                    ("general.alignment", 4, struct.pack("<I", 32)),
                    ("general.architecture", 8, _pstr("qwen2")),
                    ("general.type", 8, _pstr("adapter")),
                ]
                a_raw = struct.pack("<4e", 1.0, 0.0, 0.0, 1.0)
                b_raw = struct.pack("<4e", 0.5, 0.0, 0.0, 0.5)
                hdr = b"".join(
                    [
                        struct.pack("<4sIQQ", b"GGUF", 3, 2, len(kv_items)),
                        *(_pstr(k) + struct.pack("<I", vt) + vb for k, vt, vb in kv_items),
                        _pstr("blk.0.attn_q.weight.lora_a") + struct.pack("<IQQIQ", 2, 2, 2, 1, 0),
                        _pstr("blk.0.attn_q.weight.lora_b") + struct.pack("<IQQIQ", 2, 2, 2, 1, 32),
                    ]
                )
                pad_hdr = (-len(hdr)) % 32
                gguf_bytes = hdr + b"\x00" * pad_hdr + a_raw + b"\x00" * (32 - len(a_raw)) + b_raw
                gguf_file.write_bytes(gguf_bytes)
                gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()
                mock.state.adapters[0]["path"] = str(gguf_file)

                bundle = AdapterBundle.from_weft_binding(
                    weft_manifest,
                    weft_cert,
                    adapter_id="mde:corpus",
                    domain="corpus",
                    llama_adapter_id=0,
                    adapter_sha256=gguf_sha,
                    model_id="TinyStories-1M",
                    source_lineage={
                        "corpus_tree_sha256": lineage.corpus_tree_sha256,
                        "domain_commits": {"corpus": lineage.domain_commits["corpus"]},
                        "contested_heads": [],
                    },
                    runtime_flags_sha256=flags.canonical_digest(),
                    scale_num=1000,
                    scale_den=1000,
                    adapter_path=str(gguf_file),
                )

                signer = HmacSha256Signer(b"native-weft-test-key-32-bytes-01", key_id="warp-weft-v1")
                modeld = WarpModeld(
                    pinned_checkpoint=pinned_ckpt,
                    pinned_lineage=lineage,
                    live_runtime_flags=flags,
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(tmp / "ledger.jsonl", signer),
                )
                mounted = modeld.mount_stack([bundle.to_dict()])
                self.assertEqual(len(mounted.active_adapters), 1)
                self.assertEqual(mounted.stage1_reports[0].adapter_file_verified, True)

                out = modeld.infer([{"role": "user", "content": "native weft mount check"}])
                self.assertIn("[ACTIVE_LORA:0]", out["response"]["choices"][0]["message"]["content"])
                self.assertEqual(len(LedgerVerifier.verify_file(tmp / "ledger.jsonl", signer)), 1)
        finally:
            mock.stop()

    def test_from_weft_binding_adversarial_rejections(self) -> None:
        weft_cert, weft_sha = _make_weft_cert_dict(exp2=24)
        weft_manifest = _make_weft_manifest(weft_cert, weft_sha)
        lineage = make_lineage()
        flags_sha = make_runtime_flags().canonical_digest()
        common = dict(
            adapter_id="mde:corpus",
            domain="corpus",
            llama_adapter_id=0,
            adapter_sha256=fake_sha("gguf"),
            model_id="TinyStories-1M",
            source_lineage=lineage,
            runtime_flags_sha256=flags_sha,
        )

        # Reject float/bool rational scales and llama_adapter_id.
        with self.assertRaises(CheckpointMismatchError):
            AdapterBundle.from_weft_binding(weft_manifest, weft_cert, scale_num=1.0, scale_den=1, **common)  # type: ignore[arg-type]
        with self.assertRaises(CheckpointMismatchError):
            AdapterBundle.from_weft_binding(weft_manifest, weft_cert, scale_num=True, scale_den=1, **common)  # type: ignore[arg-type]

        # Reject Hard Veto 1 breach: Class 1 manifest missing derivation_recipe.
        no_recipe = copy.deepcopy(weft_manifest)
        no_recipe["derivation_recipe"] = None
        with self.assertRaises(CheckpointMismatchError):
            AdapterBundle.from_weft_binding(no_recipe, weft_cert, **common)

        # Reject internally tampered weft runtime_flags vs runtime_flags_sha256.
        bad_flags = copy.deepcopy(weft_manifest)
        bad_flags["runtime_flags"]["ctx_size"] = 999999
        with self.assertRaises(RuntimeFlagDriftError):
            AdapterBundle.from_weft_binding(bad_flags, weft_cert, **common)

        # Reject mismatched factor_digest between manifest and certificate.
        wrong_factor = copy.deepcopy(weft_manifest)
        wrong_factor["factor_digest"] = fake_sha("other-factor")
        with self.assertRaises(CheckpointMismatchError):
            AdapterBundle.from_weft_binding(wrong_factor, weft_cert, **common)

        # Reject forged inline canary smuggling inside weft_manifest.
        smuggled = copy.deepcopy(weft_manifest)
        smuggled["canaries"] = [{"prompt": "ok"}]
        with self.assertRaises(ForgedCanaryError):
            AdapterBundle.from_weft_binding(smuggled, weft_cert, **common)


class TestZoneManifestRotationEnforcement(unittest.TestCase):
    def test_zone_manifest_canonical_hash_and_1bit_tamper_rejection(self) -> None:
        m = _make_zone_manifest(formal_tag=3)
        mid, tags = verify_zone_manifest_dict(m)
        self.assertEqual(mid, m["manifest_id"])
        self.assertEqual(tags["formal_proof"], 3)
        self.assertEqual(tags["prose"], 0)

        # 1-bit tamper on a bank tag without rotating manifest_id fails closed.
        tampered = copy.deepcopy(m)
        tampered["zones"][0]["l1_bank_tag"] = 7
        with self.assertRaises(ZoneManifestDriftError) as ctx:
            verify_zone_manifest_dict(tampered)
        self.assertEqual(ctx.exception.code, "ERR_ZONE_MANIFEST_DRIFT")

        # Duplicate l1_bank_tag fails closed even if manifest_id is recomputed.
        dup_tag = copy.deepcopy(m)
        dup_tag["zones"][0]["l1_bank_tag"] = 0
        dup_tag["manifest_id"] = compute_zone_manifest_id(dup_tag)
        with self.assertRaises(ZoneManifestDriftError):
            verify_zone_manifest_dict(dup_tag)

        # Boolean l1_bank_tag fails closed.
        bool_tag = copy.deepcopy(m)
        bool_tag["zones"][0]["l1_bank_tag"] = True
        bool_tag["manifest_id"] = compute_zone_manifest_id(bool_tag)
        with self.assertRaises(ZoneManifestDriftError):
            verify_zone_manifest_dict(bool_tag)

    def test_zone_manifest_bank_tag_rotation_auto_unloads_flushes_and_fails_closed_over_rpc(self) -> None:
        m_v1 = _make_zone_manifest(formal_tag=3)
        m_v2 = _make_zone_manifest(formal_tag=9)
        self.assertNotEqual(m_v1["manifest_id"], m_v2["manifest_id"])

        mock = MockLlamaServer()
        url = mock.start()
        key = b"zone-manifest-test-key-32-bytes1"
        signer = HmacSha256Signer(key, key_id="warp-zone-v1")
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-zone.sock"
                ledger_path = tmp / "ledger.jsonl"

                modeld = WarpModeld(
                    pinned_checkpoint=make_checkpoint(),
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=make_runtime_flags(),
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                    llama_client=LlamaServerClient(url),
                    ledger=SignedLedger(ledger_path, signer),
                    pinned_zone_manifest=m_v1,
                )
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    rpc = WarpUnixRpcClient(sock_path)
                    ident = rpc.call("wmcp_checkpoint_identify")
                    self.assertEqual(ident["result"]["pinned_zone_manifest_id"], m_v1["manifest_id"])
                    self.assertEqual(ident["result"]["zone_bank_tags"]["formal_proof"], 3)

                    # 1. Bundle omitting zone_manifest_id fails closed BEFORE touching llama-server.
                    missing_zm = make_adapter_bundle_dict()
                    r_missing = rpc.call("wmcp_slot_mount", {"bundles": [missing_zm]})
                    self.assertEqual(r_missing["error"]["data"]["code"], "ERR_ZONE_MANIFEST_DRIFT")
                    self.assertEqual(len(mock.state.request_log), 0)

                    # 2. Bundle with wrong zone_manifest_id fails closed BEFORE touching llama-server.
                    wrong_zm = make_adapter_bundle_dict()
                    wrong_zm["zone_manifest_id"] = m_v2["manifest_id"]
                    r_wrong = rpc.call("wmcp_slot_mount", {"bundles": [wrong_zm]})
                    self.assertEqual(r_wrong["error"]["data"]["code"], "ERR_ZONE_MANIFEST_DRIFT")
                    self.assertEqual(len(mock.state.request_log), 0)

                    # 3. Bundle pinned to m_v1 mounts cleanly and serves traffic.
                    b_v1 = make_adapter_bundle_dict()
                    b_v1["zone_manifest_id"] = m_v1["manifest_id"]
                    r_ok = rpc.call(
                        "wmcp_slot_mount",
                        {"bundles": [b_v1], "zone_manifest_id": m_v1["manifest_id"]},
                    )
                    self.assertEqual(r_ok["result"]["zone_manifest_id"], m_v1["manifest_id"])

                    inf_ok = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "zone v1 turn"}],
                            "zone_manifest_id": m_v1["manifest_id"],
                        },
                    )
                    self.assertIn("[ACTIVE_LORA:0]", inf_ok["result"]["response"]["choices"][0]["message"]["content"])
                    self.assertTrue(mock.state.slots[0].recurrent_state_tainted)

                    # 4. Infer with drifted zone_manifest_id fails closed immediately.
                    inf_drift = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "drifted zone turn"}],
                            "zone_manifest_id": m_v2["manifest_id"],
                        },
                    )
                    self.assertEqual(inf_drift["error"]["data"]["code"], "ERR_ZONE_MANIFEST_DRIFT")

                    # 5. Tampered zone manifest update over RPC is rejected without unloading valid m_v1!
                    tampered_v2 = copy.deepcopy(m_v2)
                    tampered_v2["zones"][0]["l1_bank_tag"] = 42
                    r_tamper = rpc.call("warp_update_zone_manifest", {"zone_manifest": tampered_v2})
                    self.assertEqual(r_tamper["error"]["data"]["code"], "ERR_ZONE_MANIFEST_DRIFT")

                    # 6. Valid bank-tag rotation to m_v2 auto-unloads m_v1 and flushes KV/DeltaNet slots!
                    r_rot = rpc.call("warp_update_zone_manifest", {"zone_manifest": m_v2})
                    self.assertEqual(r_rot["result"]["status"]["pinned_zone_manifest_id"], m_v2["manifest_id"])
                    self.assertEqual(r_rot["result"]["status"]["zone_bank_tags"]["formal_proof"], 9)
                    self.assertEqual(r_rot["result"]["status"]["active_adapters"], [])
                    self.assertEqual(
                        r_rot["result"]["rollback_receipt"]["reason"],
                        "zone_manifest_drift_auto_unload",
                    )
                    self.assertEqual(r_rot["result"]["rollback_receipt"]["slots_flushed"], [0, 1])
                    self.assertFalse(mock.state.slots[0].recurrent_state_tainted)

                    # Base inference after rotation is clean W0 with zero residual recurrent-state leak.
                    inf_base = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "post rotation base"}],
                            "zone_manifest_id": m_v2["manifest_id"],
                        },
                    )
                    text = inf_base["result"]["response"]["choices"][0]["message"]["content"]
                    self.assertIn("[BASE_W0]", text)
                    self.assertNotIn("RESIDUAL_RECURRENT_LEAK", text)

                    # 7. Old m_v1 bundle is rejected now; freshly re-pinned m_v2 bundle mounts cleanly.
                    r_stale = rpc.call("wmcp_slot_mount", {"bundles": [b_v1]})
                    self.assertEqual(r_stale["error"]["data"]["code"], "ERR_ZONE_MANIFEST_DRIFT")

                    b_v2 = make_adapter_bundle_dict()
                    b_v2["zone_manifest_id"] = m_v2["manifest_id"]
                    r_remount = rpc.call(
                        "wmcp_slot_mount",
                        {"bundles": [b_v2], "zone_manifest_id": m_v2["manifest_id"]},
                    )
                    self.assertEqual(r_remount["result"]["zone_manifest_id"], m_v2["manifest_id"])

                    # Verify complete signed ledger chain across all rejections, mounts, and rotation rollback.
                    receipts = LedgerVerifier.verify_file(ledger_path, signer)
                    self.assertEqual(
                        [r.event for r in receipts],
                        ["reject", "reject", "mount", "rollback", "reject", "mount"],
                    )
                finally:
                    server.stop()
        finally:
            mock.stop()


if __name__ == "__main__":
    unittest.main()

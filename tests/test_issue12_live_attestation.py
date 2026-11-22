"""Adversarial regression tests for Issue #12: live /props and base model digest attestation."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import unittest

from warp_modeld.errors import CheckpointMismatchError, RuntimeFlagDriftError
from warp_modeld.fingerprint import CheckpointFingerprint, RuntimeFlags
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier
from warp_modeld.live_probe import extract_build_commit, verify_live_server_attestation
from warp_modeld.llama_client import LlamaClientError, LlamaServerClient, LlamaServerProps

from tests.helpers import (
    fake_sha,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_modeld,
    make_runtime_flags,
)
from tests.mock_llama_server import MockLlamaServer


class _RawPropsServer:
    """Minimal localhost HTTP server serving configurable raw payloads on GET /props."""

    def __init__(self, body: bytes = b"{}", *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[tuple[str, str]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                outer.requests.append(("GET", self.path))
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

            def do_POST(self) -> None:
                outer.requests.append(("POST", self.path))
                self.send_response(500)
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class TestIssue12LiveServerAttestation(unittest.TestCase):
    def test_issue12_reproducer_queries_props_and_signs_measured_vs_declared_fields(self) -> None:
        """Issue #12 reproducer: mount queries GET /props, hashes base model, signs measured vs declared."""
        key = b"issue12-repro-key-32-bytes-00001"
        server = MockLlamaServer()
        url = server.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                modeld = make_modeld(url, tmp, signer_key=key)
                res = modeld.mount_stack([make_adapter_bundle_dict()])

                endpoints = sorted({(r["method"], r["path"].split("?")[0]) for r in server.state.request_log})
                self.assertIn(("GET", "/props"), endpoints)
                # /props MUST be queried before any Stage 3/4 POST occurs.
                self.assertEqual(server.state.request_log[0]["method"], "GET")
                self.assertEqual(server.state.request_log[0]["path"], "/props")

                self.assertIsNotNone(res.live_attestation)
                att = res.receipt.details["live_attestation"]
                self.assertEqual(att["props_endpoint"], "/props")
                self.assertEqual(att["server_n_ctx"], 16384)
                self.assertEqual(att["server_commit"], "7fe450e19")
                self.assertEqual(att["measured_weights_sha256"], make_checkpoint().weights_sha256)
                self.assertEqual(
                    att["measured_fields"],
                    [
                        "checkpoint.weights_sha256",
                        "runtime_flags.ctx_size",
                        "runtime_flags.llama_server_commit",
                    ],
                )
                self.assertIn("runtime_flags.kv_cache_type", att["declared_fields"])
                self.assertIn("checkpoint.layout_sha256", att["declared_fields"])
                self.assertIn("checkpoint.safetensors_provenance_sha256", att["declared_fields"])
                # Measured and declared partitions must be strictly disjoint.
                self.assertEqual(set(att["measured_fields"]) & set(att["declared_fields"]), set())

                receipts = LedgerVerifier.verify_file(
                    tmp / "ledger.jsonl",
                    HmacSha256Signer(key, key_id="warp-test-hmac-v1"),
                )
                self.assertEqual(len(receipts), 1)
                self.assertEqual(receipts[0].details["live_attestation"], att)
        finally:
            server.stop()

    def test_rejects_live_ctx_size_and_commit_drift_before_stage3(self) -> None:
        srv = MockLlamaServer()
        url = srv.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                # 1. Context size drift between daemon config (16384) and running server (8192).
                srv.state.ctx_size = 8192
                modeld = make_modeld(url, tmp / "ctx_drift")
                with self.assertRaises(RuntimeFlagDriftError) as ctx:
                    modeld.mount_stack([make_adapter_bundle_dict()])
                self.assertEqual(ctx.exception.code, "ERR_RUNTIME_FLAG_DRIFT")
                self.assertEqual([r for r in srv.state.request_log if r["method"] == "POST"], [])
                self.assertEqual(modeld.status()["active_adapters"], [])
                self.assertEqual(modeld.ledger.count, 1)
                srv.state.ctx_size = 16384

                # 2. Binary build commit drift, dirty build, unknown build, shorter prefix, and smuggling.
                bad_builds = (
                    "b11146-deadbeef0",
                    "b0-unknown",
                    "b11146-7fe450e19-dirty",
                    "b11146-deadbeef-7fe450e19",
                    "b11146-7fe450e",
                    "b11146-7fe450e19\n",
                )
                for idx, bad_build in enumerate(bad_builds):
                    with self.subTest(build_info=bad_build):
                        srv.state.request_log.clear()
                        srv.state.build_info = bad_build
                        md = make_modeld(url, tmp / f"bad_build_{idx}")
                        with self.assertRaises(RuntimeFlagDriftError):
                            md.mount_stack([make_adapter_bundle_dict()])
                        self.assertEqual([r for r in srv.state.request_log if r["method"] == "POST"], [])

                # 3. Positive controls: full 40-char commit matching 9-char pin, and bare commit hash.
                for idx, good_build in enumerate(("b11146-7fe450e19a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "7fe450e19")):
                    with self.subTest(good_build=good_build):
                        srv.state.build_info = good_build
                        md = make_modeld(url, tmp / f"good_build_{idx}")
                        res = md.mount_stack([make_adapter_bundle_dict()])
                        self.assertEqual(len(res.active_adapters), 1)
                        md.rollback(reason="reset_good_build")
        finally:
            srv.stop()

    def test_base_model_file_digest_verification_and_symlink_controls(self) -> None:
        srv = MockLlamaServer()
        url = srv.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                real_base = tmp / "real-base.gguf"
                real_base.write_bytes(b"weights-qwen3-8b-q4km")
                wrong_base = tmp / "wrong-base.gguf"
                wrong_base.write_bytes(b"different-base-model-bytes")
                link_to_real = tmp / "link-real.gguf"
                os.symlink(real_base, link_to_real)
                link_to_wrong = tmp / "link-wrong.gguf"
                os.symlink(wrong_base, link_to_wrong)
                subdir = tmp / "model-dir"
                subdir.mkdir()

                # Positive control: benign symlink to the real base model file passes and records canonical path.
                srv.state.model_path = str(link_to_real)
                md_ok = make_modeld(url, tmp / "ok")
                res = md_ok.mount_stack([make_adapter_bundle_dict()])
                self.assertEqual(
                    res.receipt.details["live_attestation"]["server_model_path"],
                    os.path.realpath(str(real_base)),
                )
                md_ok.rollback(reason="reset_ok_base")

                # Negative controls: wrong bytes, symlink to wrong bytes, missing file, directory.
                neg_cases = (
                    ("wrong_bytes", str(wrong_base)),
                    ("symlink_wrong", str(link_to_wrong)),
                    ("missing_file", str(tmp / "nonexistent.gguf")),
                    ("directory", str(subdir)),
                )
                for label, mpath in neg_cases:
                    with self.subTest(case=label):
                        srv.state.request_log.clear()
                        srv.state.model_path = mpath
                        modeld = make_modeld(url, tmp / f"neg_{label}")
                        with self.assertRaises(CheckpointMismatchError) as ctx:
                            modeld.mount_stack([make_adapter_bundle_dict()])
                        self.assertEqual(ctx.exception.code, "ERR_CHECKPOINT_MISMATCH")
                        self.assertEqual([r for r in srv.state.request_log if r["method"] == "POST"], [])
                        self.assertAlmostEqual(srv.state.adapters[0]["scale"], 0.0)
        finally:
            srv.stop()

    def test_optional_extended_props_promote_to_measured_and_fail_closed_on_drift(self) -> None:
        srv = MockLlamaServer()
        url = srv.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                # Positive control: when server exposes extended properties matching config, all promote to measured_fields.
                srv.state.chat_template = "chat-template-qwen3-8b"
                srv.state.extra_props = {
                    "quantization": "Q4_K_M",
                    "kv_cache_type": "q8_0",
                    "ctx_checkpoints": 0,
                    "swa_checkpoints": 0,
                }
                md_ok = make_modeld(url, tmp / "ext_ok")
                res = md_ok.mount_stack([make_adapter_bundle_dict()])
                att = res.receipt.details["live_attestation"]
                for promoted in (
                    "checkpoint.chat_template_sha256",
                    "runtime_flags.quantization",
                    "runtime_flags.kv_cache_type",
                    "runtime_flags.ctx_checkpoints",
                    "runtime_flags.swa_checkpoints",
                ):
                    self.assertIn(promoted, att["measured_fields"])
                    self.assertNotIn(promoted, att["declared_fields"])
                md_ok.rollback(reason="reset_ext_ok")

                # Negative controls: drift on any extended field fails closed before Stage 3.
                drift_cases: list[tuple[str, dict[str, Any], str | None, type[Exception]]] = [
                    ("chat_template_drift", {}, "tampered-template", CheckpointMismatchError),
                    ("quantization_drift", {"quantization": "Q8_0"}, None, RuntimeFlagDriftError),
                    ("kv_cache_type_drift", {"kv_cache_type": "f16"}, None, RuntimeFlagDriftError),
                    ("ctx_checkpoints_drift", {"ctx_checkpoints": 4}, None, RuntimeFlagDriftError),
                    ("swa_checkpoints_drift", {"swa_checkpoints": 2}, None, RuntimeFlagDriftError),
                ]
                for label, extra, tpl, exc_cls in drift_cases:
                    with self.subTest(case=label):
                        srv.state.request_log.clear()
                        srv.state.chat_template = tpl
                        srv.state.extra_props = dict(extra)
                        modeld = make_modeld(url, tmp / f"ext_{label}")
                        with self.assertRaises(exc_cls):
                            modeld.mount_stack([make_adapter_bundle_dict()])
                        self.assertEqual([r for r in srv.state.request_log if r["method"] == "POST"], [])
        finally:
            srv.stop()

    def test_llama_client_get_props_rejects_malformed_payloads_and_http_errors(self) -> None:
        base_ok: dict[str, Any] = {
            "default_generation_settings": {"n_ctx": 16384},
            "total_slots": 2,
            "model_path": "/tmp/model.gguf",
            "build_info": "b11146-7fe450e19",
        }
        bad_payloads: list[tuple[str, bytes]] = [
            ("non_dict_list", b"[]"),
            ("non_dict_int", b"123"),
            ("missing_dgs", json.dumps({"model_path": "/tmp/m.gguf", "build_info": "b1-7fe450e19"}).encode()),
            ("non_dict_dgs", json.dumps({**base_ok, "default_generation_settings": []}).encode()),
            ("missing_n_ctx", json.dumps({**base_ok, "default_generation_settings": {}}).encode()),
            ("bool_n_ctx", json.dumps({**base_ok, "default_generation_settings": {"n_ctx": True}}).encode()),
            ("zero_n_ctx", json.dumps({**base_ok, "default_generation_settings": {"n_ctx": 0}}).encode()),
            ("neg_n_ctx", json.dumps({**base_ok, "default_generation_settings": {"n_ctx": -16}}).encode()),
            ("float_n_ctx", json.dumps({**base_ok, "default_generation_settings": {"n_ctx": 16384.0}}).encode()),
            ("missing_model_path", json.dumps({k: v for k, v in base_ok.items() if k != "model_path"}).encode()),
            ("int_model_path", json.dumps({**base_ok, "model_path": 123}).encode()),
            ("empty_model_path", json.dumps({**base_ok, "model_path": ""}).encode()),
            ("ws_model_path", json.dumps({**base_ok, "model_path": "   "}).encode()),
            ("nul_model_path", json.dumps({**base_ok, "model_path": "/tmp/m\x00.gguf"}).encode()),
            ("surrogate_model_path", b'{"default_generation_settings":{"n_ctx":16384},"model_path":"\\ud800","build_info":"b1-7fe450e19"}'),
            ("missing_build_info", json.dumps({k: v for k, v in base_ok.items() if k != "build_info"}).encode()),
            ("int_build_info", json.dumps({**base_ok, "build_info": 999}).encode()),
            ("empty_build_info", json.dumps({**base_ok, "build_info": ""}).encode()),
            ("ws_build_info", json.dumps({**base_ok, "build_info": "   "}).encode()),
            ("nul_build_info", json.dumps({**base_ok, "build_info": "b1-7fe450e\x00"}).encode()),
            ("surrogate_build_info", b'{"default_generation_settings":{"n_ctx":16384},"model_path":"/tmp/m.gguf","build_info":"\\ud800"}'),
            ("bool_total_slots", json.dumps({**base_ok, "total_slots": True}).encode()),
            ("zero_total_slots", json.dumps({**base_ok, "total_slots": 0}).encode()),
            ("str_total_slots", json.dumps({**base_ok, "total_slots": "2"}).encode()),
            ("int_chat_template", json.dumps({**base_ok, "chat_template": 123}).encode()),
            ("surrogate_chat_template", b'{"default_generation_settings":{"n_ctx":16384},"model_path":"/tmp/m.gguf","build_info":"b1-7fe450e19","chat_template":"\\ud800"}'),
            ("int_quantization", json.dumps({**base_ok, "quantization": 4}).encode()),
            ("ws_quantization", json.dumps({**base_ok, "quantization": "  "}).encode()),
            ("nul_quantization", json.dumps({**base_ok, "quantization": "Q4\x00K"}).encode()),
            ("surrogate_quantization", b'{"default_generation_settings":{"n_ctx":16384},"model_path":"/tmp/m.gguf","build_info":"b1-7fe450e19","quantization":"\\ud800"}'),
            ("int_kv_cache", json.dumps({**base_ok, "kv_cache_type": 8}).encode()),
            ("ws_kv_cache", json.dumps({**base_ok, "kv_cache_type": " "}).encode()),
            ("nul_kv_cache", json.dumps({**base_ok, "kv_cache_type": "q8\x000"}).encode()),
            ("bool_ctx_cp", json.dumps({**base_ok, "ctx_checkpoints": False}).encode()),
            ("neg_ctx_cp", json.dumps({**base_ok, "ctx_checkpoints": -1}).encode()),
            ("bool_swa_cp", json.dumps({**base_ok, "swa_checkpoints": True}).encode()),
            ("neg_swa_cp", json.dumps({**base_ok, "swa_checkpoints": -1}).encode()),
        ]
        srv = _RawPropsServer()
        url = srv.start()
        try:
            client = LlamaServerClient(url, require_localhost=True)
            for label, payload in bad_payloads:
                with self.subTest(payload=label):
                    srv.status = 200
                    srv.body = payload
                    with self.assertRaises(LlamaClientError):
                        client.get_props()

            # Valid payload with optional zero checkpoints and empty chat_template -> None.
            srv.status = 200
            srv.body = json.dumps(
                {
                    **base_ok,
                    "total_slots": 1,
                    "chat_template": "",
                    "quantization": "Q4_K_M",
                    "kv_cache_type": "q8_0",
                    "ctx_checkpoints": 0,
                    "swa_checkpoints": 0,
                }
            ).encode()
            parsed = client.get_props()
            self.assertEqual(parsed.n_ctx, 16384)
            self.assertEqual(parsed.total_slots, 1)
            self.assertIsNone(parsed.chat_template)
            self.assertEqual(parsed.quantization, "Q4_K_M")
            self.assertEqual(parsed.kv_cache_type, "q8_0")
            self.assertEqual(parsed.ctx_checkpoints, 0)
            self.assertEqual(parsed.swa_checkpoints, 0)

            # HTTP 404 and malformed wire responses (invalid UTF-8, 5000-digit int, deep JSON recursion)
            # fail closed during mount_stack with LlamaClientError and sign a reject ledger receipt.
            for label, status, body in (
                ("http_404", 404, b'{"error":"not found"}'),
                ("invalid_utf8", 200, b"\xff\xfe\x00"),
                (
                    "oversized_5000_digit_int",
                    200,
                    (
                        '{"default_generation_settings":{"n_ctx":'
                        + ("9" * 5000)
                        + '},"model_path":"/tmp/m.gguf","build_info":"b1-7fe450e19"}'
                    ).encode("ascii"),
                ),
                ("deep_json_recursion", 200, (b"[" * 2000) + (b"]" * 2000)),
            ):
                with self.subTest(wire_error=label):
                    srv.status = status
                    srv.body = body
                    with self.assertRaises(LlamaClientError):
                        client.get_props()
                    with tempfile.TemporaryDirectory() as td:
                        modeld = make_modeld(url, Path(td))
                        with self.assertRaises(LlamaClientError):
                            modeld.mount_stack([make_adapter_bundle_dict()])
                        self.assertEqual(modeld.ledger.count, 1)
        finally:
            srv.stop()

    def test_verify_live_server_attestation_direct_defense_in_depth(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = Path(td) / "base.gguf"
            model_file.write_bytes(b"weights-qwen3-8b-q4km")
            ckpt = make_checkpoint()
            flags = make_runtime_flags()

            # Non-LlamaServerProps rejected.
            with self.assertRaises(RuntimeFlagDriftError):
                verify_live_server_attestation({"n_ctx": 16384}, ckpt, flags)  # type: ignore[arg-type]

            # Non-string build_info rejected in extract_build_commit.
            with self.assertRaises(RuntimeFlagDriftError):
                extract_build_commit(123)  # type: ignore[arg-type]

            # Bool n_ctx=True against ctx_size=1 rejected (True == 1 in Python).
            flags_ctx1 = RuntimeFlags(
                llama_server_commit="7fe450e19",
                quantization="Q4_K_M",
                ctx_size=1,
                kv_cache_type="q8_0",
                ctx_checkpoints=0,
                swa_checkpoints=0,
                lora_init_without_apply=True,
            )
            with self.assertRaises(RuntimeFlagDriftError):
                verify_live_server_attestation(
                    LlamaServerProps(n_ctx=True, model_path=str(model_file), build_info="b1-7fe450e19"),
                    ckpt,
                    flags_ctx1,
                )

            # Bool ctx_checkpoints=False and swa_checkpoints=False against 0 rejected (False == 0 in Python).
            with self.assertRaises(RuntimeFlagDriftError):
                verify_live_server_attestation(
                    LlamaServerProps(
                        n_ctx=16384,
                        model_path=str(model_file),
                        build_info="b1-7fe450e19",
                        ctx_checkpoints=False,  # type: ignore[arg-type]
                    ),
                    ckpt,
                    flags,
                )
            with self.assertRaises(RuntimeFlagDriftError):
                verify_live_server_attestation(
                    LlamaServerProps(
                        n_ctx=16384,
                        model_path=str(model_file),
                        build_info="b1-7fe450e19",
                        swa_checkpoints=False,  # type: ignore[arg-type]
                    ),
                    ckpt,
                    flags,
                )

            # Huge 5000-digit ints in direct Python calls must cleanly raise RuntimeFlagDriftError (never ValueError).
            huge = 10**5000
            for huge_props in (
                LlamaServerProps(n_ctx=huge, model_path=str(model_file), build_info="b1-7fe450e19"),
                LlamaServerProps(n_ctx=16384, model_path=str(model_file), build_info="b1-7fe450e19", ctx_checkpoints=huge),
                LlamaServerProps(n_ctx=16384, model_path=str(model_file), build_info="b1-7fe450e19", swa_checkpoints=huge),
            ):
                with self.assertRaises(RuntimeFlagDriftError):
                    verify_live_server_attestation(huge_props, ckpt, flags)

            # Invalid model_path (non-str, empty, whitespace, NUL) rejected with explicit diagnostic.
            for bad_mpath in (None, 123, "", "   ", f"{model_file}\x00evil"):
                with self.subTest(bad_mpath=bad_mpath):
                    with self.assertRaises(CheckpointMismatchError) as ctx:
                        verify_live_server_attestation(
                            LlamaServerProps(
                                n_ctx=16384,
                                model_path=bad_mpath,  # type: ignore[arg-type]
                                build_info="b1-7fe450e19",
                            ),
                            ckpt,
                            flags,
                        )
                    self.assertIn("missing or invalid model_path", str(ctx.exception))

            # Unpaired surrogate in model_path and chat_template rejected cleanly without UnicodeEncodeError.
            with self.assertRaises(CheckpointMismatchError):
                verify_live_server_attestation(
                    LlamaServerProps(n_ctx=16384, model_path="\ud800", build_info="b1-7fe450e19"),
                    ckpt,
                    flags,
                )
            with self.assertRaises(CheckpointMismatchError):
                verify_live_server_attestation(
                    LlamaServerProps(
                        n_ctx=16384,
                        model_path=str(model_file),
                        build_info="b1-7fe450e19",
                        chat_template="\ud800",
                    ),
                    ckpt,
                    flags,
                )

            # Unreadable base model file (os.stat succeeds, compute_file_sha256 hits PermissionError) fails closed.
            unreadable = Path(td) / "unreadable-base.gguf"
            unreadable.write_bytes(b"weights-qwen3-8b-q4km")
            os.chmod(unreadable, 0o000)
            try:
                with self.assertRaises(CheckpointMismatchError):
                    verify_live_server_attestation(
                        LlamaServerProps(n_ctx=16384, model_path=str(unreadable), build_info="b1-7fe450e19"),
                        ckpt,
                        flags,
                    )
            finally:
                os.chmod(unreadable, 0o600)

            # Non-regular readable character device (/dev/null) with matching empty SHA-256 MUST fail S_ISREG.
            if Path("/dev/null").exists():
                ckpt_empty = CheckpointFingerprint(
                    model_id=ckpt.model_id,
                    layout_sha256=ckpt.layout_sha256,
                    weights_sha256=hashlib.sha256(b"").hexdigest(),
                    config_sha256=ckpt.config_sha256,
                    tokenizer_sha256=ckpt.tokenizer_sha256,
                    chat_template_sha256=ckpt.chat_template_sha256,
                    safetensors_provenance_sha256=None,
                )
                with self.assertRaises(CheckpointMismatchError) as dev_ctx:
                    verify_live_server_attestation(
                        LlamaServerProps(n_ctx=16384, model_path="/dev/null", build_info="b1-7fe450e19"),
                        ckpt_empty,
                        flags,
                    )
                self.assertIn("not a regular file", str(dev_ctx.exception))

            # Non-string chat_template rejected cleanly.
            with self.assertRaises(CheckpointMismatchError):
                verify_live_server_attestation(
                    LlamaServerProps(
                        n_ctx=16384,
                        model_path=str(model_file),
                        build_info="b1-7fe450e19",
                        chat_template=123,  # type: ignore[arg-type]
                    ),
                    ckpt,
                    flags,
                )

            # Checkpoint without safetensors_provenance_sha256 omits it cleanly from declared_fields.
            ckpt_no_prov = CheckpointFingerprint(
                model_id=ckpt.model_id,
                layout_sha256=ckpt.layout_sha256,
                weights_sha256=ckpt.weights_sha256,
                config_sha256=ckpt.config_sha256,
                tokenizer_sha256=ckpt.tokenizer_sha256,
                chat_template_sha256=ckpt.chat_template_sha256,
                safetensors_provenance_sha256=None,
            )
            rep = verify_live_server_attestation(
                LlamaServerProps(n_ctx=16384, model_path=str(model_file), build_info="b1-7fe450e19"),
                ckpt_no_prov,
                flags,
            )
            self.assertNotIn("checkpoint.safetensors_provenance_sha256", rep.declared_fields)


if __name__ == "__main__":
    unittest.main()

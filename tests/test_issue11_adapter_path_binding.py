"""Adversarial regression tests for Issue #11: Stage 1/4 adapter path binding."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Callable
import unittest

from warp_modeld.canary import CanaryRunner, CanarySuite
from warp_modeld.errors import CheckpointMismatchError
from warp_modeld.fingerprint import AdapterBundle
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger
from warp_modeld.llama_client import LlamaClientError, LlamaServerClient
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_modeld,
    make_runtime_flags,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer


class _RawPayloadServer:
    """Minimal localhost HTTP server serving raw bytes on GET /lora-adapters."""

    def __init__(self, body: bytes, *, status: int = 200) -> None:
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


class _HookedCanaryRunner:
    """Wraps CanaryRunner and invokes a hook right before Stage 4 runs."""

    def __init__(self, inner: CanaryRunner, hook: Callable[[], None]) -> None:
        self._inner = inner
        self._hook = hook

    def run(self, bundles: Any, **kwargs: Any) -> Any:
        result = self._inner.run(bundles, **kwargs)
        self._hook()
        return result


def _write_adapter(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class TestIssue11AdapterPathBinding(unittest.TestCase):
    def test_issue11_exact_reproducer_fails_closed_before_stage2(self) -> None:
        """Issue #11 reproducer: decoy file hashed while server lists a different path."""
        server = MockLlamaServer()
        url = server.start()
        key = b"issue11-repro-key-32-bytes-00001"
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                decoy = tmp / "decoy.gguf"
                decoy_sha = _write_adapter(decoy, b"bytes-the-verifier-hashed")
                raw = make_adapter_bundle_dict(
                    adapter_path=str(decoy),
                    adapter_sha256=decoy_sha,
                    llama_adapter_id=0,
                )
                modeld = make_modeld(url, tmp, signer_key=key)

                with self.assertRaises(CheckpointMismatchError) as ctx:
                    modeld.mount_stack([raw])
                self.assertEqual(ctx.exception.code, "ERR_CHECKPOINT_MISMATCH")

                # Must fail closed before Stage 2/3/4: only 1 read-only GET /lora-adapters, 0 POSTs.
                self.assertEqual(len(server.state.request_log), 1)
                self.assertEqual(server.state.request_log[0]["method"], "GET")
                self.assertEqual(server.state.request_log[0]["path"], "/lora-adapters")
                self.assertAlmostEqual(server.state.adapters[0]["scale"], 0.0)
                self.assertEqual(modeld.status()["active_adapters"], [])

                receipts = LedgerVerifier.verify_file(
                    tmp / "ledger.jsonl",
                    HmacSha256Signer(key, key_id="warp-test-hmac-v1"),
                )
                self.assertEqual([r.event for r in receipts], ["reject"])
                self.assertEqual(receipts[0].error_code, "ERR_CHECKPOINT_MISMATCH")
        finally:
            server.stop()

    def test_positive_controls_exact_relative_symlink_hardlink_and_multistack(self) -> None:
        server = MockLlamaServer()
        url = server.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sub = tmp / "nested"
                sub.mkdir()
                real0 = tmp / "corpus-f16.gguf"
                real1 = tmp / "weft-eido-f16.gguf"
                sha0 = _write_adapter(real0, b"GGUF-corpus-adapter-bytes")
                sha1 = _write_adapter(real1, b"GGUF-weft-eido-adapter-bytes")

                link_to_0 = tmp / "link-corpus.gguf"
                os.symlink(real0, link_to_0)
                hardlink_to_0 = tmp / "hardlink-corpus.gguf"
                os.link(real0, hardlink_to_0)
                dotdot_to_0 = str(sub / ".." / "corpus-f16.gguf")

                # 1. Exact path match
                server.state.adapters[0]["path"] = str(real0)
                modeld = make_modeld(url, tmp / "run_exact")
                res = modeld.mount_stack(
                    [make_adapter_bundle_dict(adapter_path=str(real0), adapter_sha256=sha0, llama_adapter_id=0)]
                )
                self.assertTrue(res.stage1_reports[0].adapter_file_verified)
                self.assertAlmostEqual(server.state.adapters[0]["scale"], 1.0)
                modeld.rollback(reason="reset_exact")

                # 2. Benign symlink in bundle -> real file on server
                modeld_sym1 = make_modeld(url, tmp / "run_sym1")
                res_sym1 = modeld_sym1.mount_stack(
                    [make_adapter_bundle_dict(adapter_path=str(link_to_0), adapter_sha256=sha0, llama_adapter_id=0)]
                )
                self.assertTrue(res_sym1.stage1_reports[0].adapter_file_verified)
                modeld_sym1.rollback(reason="reset_sym1")

                # 3. Real file in bundle -> benign symlink on server
                server.state.adapters[0]["path"] = str(link_to_0)
                modeld_sym2 = make_modeld(url, tmp / "run_sym2")
                res_sym2 = modeld_sym2.mount_stack(
                    [make_adapter_bundle_dict(adapter_path=str(real0), adapter_sha256=sha0, llama_adapter_id=0)]
                )
                self.assertTrue(res_sym2.stage1_reports[0].adapter_file_verified)
                modeld_sym2.rollback(reason="reset_sym2")

                # 4. Relative dotdot path equivalence
                server.state.adapters[0]["path"] = str(real0)
                modeld_rel = make_modeld(url, tmp / "run_rel")
                res_rel = modeld_rel.mount_stack(
                    [make_adapter_bundle_dict(adapter_path=dotdot_to_0, adapter_sha256=sha0, llama_adapter_id=0)]
                )
                self.assertTrue(res_rel.stage1_reports[0].adapter_file_verified)
                modeld_rel.rollback(reason="reset_rel")

                # 5. Hardlink to identical kernel inode
                server.state.adapters[0]["path"] = str(hardlink_to_0)
                modeld_hl = make_modeld(url, tmp / "run_hl")
                res_hl = modeld_hl.mount_stack(
                    [make_adapter_bundle_dict(adapter_path=str(real0), adapter_sha256=sha0, llama_adapter_id=0)]
                )
                self.assertTrue(res_hl.stage1_reports[0].adapter_file_verified)
                modeld_hl.rollback(reason="reset_hl")

                # 6. Multi-adapter stack where all paths match
                server.state.adapters[0]["path"] = str(real0)
                server.state.adapters[1]["path"] = str(real1)
                modeld_multi = make_modeld(url, tmp / "run_multi")
                res_multi = modeld_multi.mount_stack(
                    [
                        make_adapter_bundle_dict(
                            adapter_id="mde:corpus",
                            domain="corpus",
                            llama_adapter_id=0,
                            adapter_path=str(real0),
                            adapter_sha256=sha0,
                        ),
                        make_adapter_bundle_dict(
                            adapter_id="mde:weft-eido",
                            domain="weft-eido",
                            llama_adapter_id=1,
                            adapter_path=str(real1),
                            adapter_sha256=sha1,
                        ),
                    ]
                )
                self.assertEqual(len(res_multi.active_adapters), 2)
                self.assertAlmostEqual(server.state.adapters[0]["scale"], 1.0)
                self.assertAlmostEqual(server.state.adapters[1]["scale"], 1.0)
        finally:
            server.stop()

    def test_negative_controls_swapped_files_symlinks_dirs_missing_id_and_partial_stack(self) -> None:
        server = MockLlamaServer()
        url = server.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                bundle_file = tmp / "verified.gguf"
                other_diff = tmp / "other-different.gguf"
                other_same_bytes = tmp / "other-same-bytes.gguf"
                payload = b"GGUF-identical-bytes-different-file"
                bundle_sha = _write_adapter(bundle_file, payload)
                _write_adapter(other_diff, b"GGUF-completely-different-bytes")
                _write_adapter(other_same_bytes, payload)

                symlink_other = tmp / "symlink-to-other.gguf"
                os.symlink(other_diff, symlink_other)
                broken_symlink = tmp / "broken-symlink.gguf"
                os.symlink(tmp / "does-not-exist.gguf", broken_symlink)
                subdir = tmp / "adapter-dir"
                subdir.mkdir()

                cases: list[tuple[str, str]] = [
                    ("swapped_different_file", str(other_diff)),
                    ("different_file_with_identical_bytes", str(other_same_bytes)),
                    ("symlink_to_different_target", str(symlink_other)),
                    ("broken_symlink", str(broken_symlink)),
                    ("directory_path", str(subdir)),
                    ("nonexistent_server_path", str(tmp / "missing.gguf")),
                ]
                for label, server_path in cases:
                    with self.subTest(case=label):
                        server.state.request_log.clear()
                        server.state.adapters[0]["path"] = server_path
                        modeld = make_modeld(url, tmp / f"neg_{label}")
                        raw = make_adapter_bundle_dict(
                            adapter_path=str(bundle_file),
                            adapter_sha256=bundle_sha,
                            llama_adapter_id=0,
                        )
                        with self.assertRaises(CheckpointMismatchError):
                            modeld.mount_stack([raw])
                        posts = [r for r in server.state.request_log if r["method"] == "POST"]
                        self.assertEqual(posts, [])
                        self.assertAlmostEqual(server.state.adapters[0]["scale"], 0.0)

                # Requested llama_adapter_id absent from server's /lora-adapters list fails before Stage 2.
                server.state.request_log.clear()
                modeld_missing_id = make_modeld(url, tmp / "neg_missing_id")
                missing_id_bundle = make_adapter_bundle_dict(
                    adapter_path=str(bundle_file),
                    adapter_sha256=bundle_sha,
                    llama_adapter_id=9,
                )
                with self.assertRaises(CheckpointMismatchError):
                    modeld_missing_id.mount_stack([missing_id_bundle])
                self.assertEqual([r for r in server.state.request_log if r["method"] == "POST"], [])

                # Multi-adapter stack where adapter 0 matches but adapter 1 mismatches fails closed.
                server.state.request_log.clear()
                server.state.adapters[0]["path"] = str(bundle_file)
                server.state.adapters[1]["path"] = str(other_diff)
                modeld_partial = make_modeld(url, tmp / "neg_partial")
                with self.assertRaises(CheckpointMismatchError):
                    modeld_partial.mount_stack(
                        [
                            make_adapter_bundle_dict(
                                adapter_id="mde:corpus",
                                domain="corpus",
                                llama_adapter_id=0,
                                adapter_path=str(bundle_file),
                                adapter_sha256=bundle_sha,
                            ),
                            make_adapter_bundle_dict(
                                adapter_id="mde:weft-eido",
                                domain="weft-eido",
                                llama_adapter_id=1,
                                adapter_path=str(bundle_file),
                                adapter_sha256=bundle_sha,
                            ),
                        ]
                    )
                self.assertEqual([r for r in server.state.request_log if r["method"] == "POST"], [])
                self.assertAlmostEqual(server.state.adapters[0]["scale"], 0.0)
        finally:
            server.stop()

    def test_bundle_rejects_malformed_adapter_path_before_http(self) -> None:
        server = MockLlamaServer()
        url = server.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                for bad_path in (123, "", "   ", "/tmp/evil\x00.gguf", "/tmp/surrogate-\ud800.gguf"):
                    with self.subTest(bad_path=repr(bad_path)):
                        raw = make_adapter_bundle_dict()
                        raw["adapter_path"] = bad_path
                        with self.assertRaises(CheckpointMismatchError):
                            AdapterBundle.from_dict(raw)

                        server.state.request_log.clear()
                        modeld = make_modeld(url, tmp / f"bad_bundle_{abs(hash(repr(bad_path)))}")
                        with self.assertRaises(CheckpointMismatchError):
                            modeld.mount_stack([raw])
                        self.assertEqual(server.state.request_log, [])
                        self.assertEqual(modeld.ledger.count, 1)
        finally:
            server.stop()

    def test_llama_client_rejects_malformed_lora_adapters_payloads(self) -> None:
        bad_payloads: list[tuple[str, bytes]] = [
            ("non_list_empty_object", b"{}"),
            ("non_list_object", b'{"adapters":[]}'),
            ("non_list_int", b"123"),
            ("non_dict_entry", b'["not-an-object"]'),
            ("non_dict_entry_containing_id", b'["id"]'),
            ("missing_id", b'[{"path":"/tmp/a.gguf","scale":0.0}]'),
            ("bool_id_true", b'[{"id":true,"path":"/tmp/a.gguf","scale":0.0}]'),
            ("bool_id_false", b'[{"id":false,"path":"/tmp/a.gguf","scale":0.0}]'),
            ("float_id", b'[{"id":1.5,"path":"/tmp/a.gguf","scale":0.0}]'),
            ("string_id", b'[{"id":"0","path":"/tmp/a.gguf","scale":0.0}]'),
            ("negative_id", b'[{"id":-1,"path":"/tmp/a.gguf","scale":0.0}]'),
            ("null_id", b'[{"id":null,"path":"/tmp/a.gguf","scale":0.0}]'),
            (
                "duplicate_id",
                b'[{"id":0,"path":"/tmp/a.gguf","scale":0.0},{"id":0,"path":"/tmp/b.gguf","scale":0.0}]',
            ),
            ("missing_path", b'[{"id":0,"scale":0.0}]'),
            ("null_path", b'[{"id":0,"path":null,"scale":0.0}]'),
            ("int_path", b'[{"id":0,"path":123,"scale":0.0}]'),
            ("empty_path", b'[{"id":0,"path":"","scale":0.0}]'),
            ("whitespace_path", b'[{"id":0,"path":"   ","scale":0.0}]'),
            ("nul_path", b'[{"id":0,"path":"/tmp/a\\u0000b.gguf","scale":0.0}]'),
            ("surrogate_path", b'[{"id":0,"path":"\\ud800","scale":0.0}]'),
            ("bool_scale", b'[{"id":0,"path":"/tmp/a.gguf","scale":true}]'),
            ("string_scale", b'[{"id":0,"path":"/tmp/a.gguf","scale":"1.0"}]'),
            ("overflow_int_scale", f'[{{"id":0,"path":"/tmp/a.gguf","scale":{"1" + "0" * 400}}}]'.encode("ascii")),
            ("nan_scale", b'[{"id":0,"path":"/tmp/a.gguf","scale":NaN}]'),
            ("inf_scale", b'[{"id":0,"path":"/tmp/a.gguf","scale":Infinity}]'),
        ]
        srv = _RawPayloadServer(b"[]")
        url = srv.start()
        try:
            client = LlamaServerClient(url, require_localhost=True)
            for label, payload in bad_payloads:
                with self.subTest(payload=label):
                    srv.body = payload
                    with self.assertRaises(LlamaClientError):
                        client.list_lora_adapters()

            # Valid payloads: id=0 with omitted scale defaults to exact 0.0, id=1 with explicit scale passes.
            srv.body = json.dumps(
                [
                    {"id": 0, "path": "/tmp/a.gguf"},
                    {"id": 1, "path": "/tmp/b.gguf", "scale": 0.5},
                ]
            ).encode("utf-8")
            adapters = client.list_lora_adapters()
            self.assertEqual(len(adapters), 2)
            self.assertEqual(adapters[0].id, 0)
            self.assertEqual(adapters[0].path, "/tmp/a.gguf")
            self.assertEqual(adapters[0].scale, 0.0)
            self.assertEqual(adapters[1].id, 1)
            self.assertEqual(adapters[1].scale, 0.5)

            # End-to-end mount_stack fails closed on malformed /lora-adapters (bool id, surrogate path, huge scale) and logs reject.
            for end_label, wire_body in (
                ("bool_id", b'[{"id":false,"path":"/tmp/a.gguf","scale":0.0}]'),
                ("surrogate_path", b'[{"id":0,"path":"\\ud800","scale":0.0}]'),
                ("overflow_scale", f'[{{"id":0,"path":"/tmp/a.gguf","scale":{"1" + "0" * 400}}}]'.encode("ascii")),
            ):
                with self.subTest(mount_bad_wire=end_label):
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td)
                        f = tmp / "a.gguf"
                        f_sha = _write_adapter(f, b"adapter-bytes")
                        srv.requests.clear()
                        srv.body = wire_body
                        modeld = make_modeld(url, tmp)
                        raw = make_adapter_bundle_dict(adapter_path=str(f), adapter_sha256=f_sha, llama_adapter_id=0)
                        with self.assertRaises(LlamaClientError):
                            modeld.mount_stack([raw])
                        self.assertEqual([m for m, _ in srv.requests], ["GET"])
                        self.assertEqual(modeld.ledger.count, 1)
        finally:
            srv.stop()

    def test_verify_loaded_adapter_bindings_defense_in_depth_against_stub_clients(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            f = tmp / "a.gguf"
            f_sha = _write_adapter(f, b"stub-test-bytes")
            bundle = AdapterBundle.from_dict(
                make_adapter_bundle_dict(adapter_path=str(f), adapter_sha256=f_sha, llama_adapter_id=0)
            )
            bundle_id1 = AdapterBundle.from_dict(
                make_adapter_bundle_dict(adapter_path=str(f), adapter_sha256=f_sha, llama_adapter_id=1)
            )

            # Positive control with id=0 and id=1.
            mapping = WarpModeld._verify_loaded_adapter_bindings(
                [bundle, bundle_id1],
                [
                    SimpleNamespace(id=0, path=str(f), scale=0.0),
                    SimpleNamespace(id=1, path=str(f), scale=0.0),
                ],
            )
            self.assertEqual(sorted(mapping.keys()), [0, 1])

            # Non-sequence, non-iterable, empty-dict, str, and bytes payloads rejected (even with empty bundles).
            for bad_loaded in ({}, {"id": 0}, 123, "", "not-a-list", b"", b"not-a-list"):
                with self.subTest(bad_loaded=repr(bad_loaded)):
                    with self.assertRaises(CheckpointMismatchError):
                        WarpModeld._verify_loaded_adapter_bindings([], bad_loaded)  # type: ignore[arg-type]

            # Invalid adapter IDs (bool False==0, True==1, negative alongside valid id=0, string, missing) and duplicates.
            bad_items: list[tuple[str, list[Any], str]] = [
                ("bool_id_false", [SimpleNamespace(id=False, path=str(f))], "invalid loaded adapter id"),
                ("bool_id_true", [SimpleNamespace(id=True, path=str(f))], "invalid loaded adapter id"),
                (
                    "negative_id_with_valid",
                    [SimpleNamespace(id=0, path=str(f)), SimpleNamespace(id=-1, path=str(f))],
                    "invalid loaded adapter id",
                ),
                ("str_id", [SimpleNamespace(id="0", path=str(f))], "invalid loaded adapter id"),
                ("missing_id_attr", [SimpleNamespace(path=str(f))], "invalid loaded adapter id"),
                ("duplicate_id", [SimpleNamespace(id=0, path=str(f)), SimpleNamespace(id=0, path=str(f))], "duplicate"),
                ("missing_path_attr", [SimpleNamespace(id=0)], "missing or invalid server-reported adapter path"),
                ("none_path", [SimpleNamespace(id=0, path=None)], "missing or invalid server-reported adapter path"),
                ("int_path", [SimpleNamespace(id=0, path=42)], "missing or invalid server-reported adapter path"),
                ("empty_path", [SimpleNamespace(id=0, path="")], "missing or invalid server-reported adapter path"),
                ("whitespace_path", [SimpleNamespace(id=0, path="   ")], "missing or invalid server-reported adapter path"),
                ("nul_path", [SimpleNamespace(id=0, path=f"{f}\x00evil")], "missing or invalid server-reported adapter path"),
                ("surrogate_path", [SimpleNamespace(id=0, path="\ud800")], "non-UTF-8 server-reported adapter path"),
            ]
            for label, items, expected_msg in bad_items:
                with self.subTest(case=label):
                    with self.assertRaises(CheckpointMismatchError) as ctx:
                        WarpModeld._verify_loaded_adapter_bindings([bundle], items)
                    self.assertIn(expected_msg, str(ctx.exception))

            # Non-regular file (directory on both sides -> samestat is True, S_ISREG is False) MUST fail closed.
            bundle_dir = AdapterBundle.from_dict(
                make_adapter_bundle_dict(adapter_path=str(tmp), adapter_sha256=f_sha, llama_adapter_id=0)
            )
            with self.assertRaises(CheckpointMismatchError):
                WarpModeld._verify_loaded_adapter_bindings(
                    [bundle_dir],
                    [SimpleNamespace(id=0, path=str(tmp))],
                )

    def test_stage4_catches_mid_mount_path_swap_digest_tamper_dir_swap_and_missing_id(self) -> None:
        scenarios = (
            "server_path_swap",
            "file_digest_tamper",
            "file_replaced_by_dir",
            "file_unreadable_stage4",
            "adapter_dropped_stage4",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                server = MockLlamaServer()
                url = server.start()
                try:
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td)
                        adapter_file = tmp / "live-adapter.gguf"
                        decoy_file = tmp / "late-decoy.gguf"
                        sha = _write_adapter(adapter_file, b"GGUF-stage1-verified-bytes")
                        _write_adapter(decoy_file, b"GGUF-late-swapped-bytes")
                        server.state.adapters[0]["path"] = str(adapter_file)

                        def _mutate_before_stage4() -> None:
                            if scenario == "server_path_swap":
                                server.state.adapters[0]["path"] = str(decoy_file)
                            elif scenario == "file_digest_tamper":
                                adapter_file.write_bytes(b"GGUF-overwritten-before-stage4")
                            elif scenario == "file_replaced_by_dir":
                                adapter_file.unlink()
                                adapter_file.mkdir()
                            elif scenario == "file_unreadable_stage4":
                                # chmod 0o000 lets os.stat succeed while compute_file_sha256 hits PermissionError.
                                os.chmod(adapter_file, 0o000)
                            elif scenario == "adapter_dropped_stage4":
                                server.state.adapters.pop(0, None)

                        client = LlamaServerClient(url, require_localhost=True)
                        suite = CanarySuite.from_verifier_file(CANARY_SUITE_PATH)
                        signer = HmacSha256Signer(b"stage4-swap-key-32-bytes-0000001", key_id="warp-test-hmac-v1")
                        modeld = WarpModeld(
                            pinned_checkpoint=make_checkpoint(),
                            pinned_lineage=make_lineage(),
                            live_runtime_flags=make_runtime_flags(),
                            single_bounds=make_single_bounds(),
                            joint_bounds=make_joint_bounds(),
                            canary_suite=suite,
                            llama_client=client,
                            ledger=SignedLedger(tmp / "ledger.jsonl", signer),
                            canary_runner_override=_HookedCanaryRunner(
                                CanaryRunner(client, suite, canary_slot_id=0),
                                _mutate_before_stage4,
                            ),
                        )

                        if scenario == "adapter_dropped_stage4":
                            # Also proves Stage 4 checks missing adapter IDs even when adapter_path is None.
                            raw = make_adapter_bundle_dict(adapter_path=None, llama_adapter_id=0)
                        else:
                            raw = make_adapter_bundle_dict(
                                adapter_path=str(adapter_file),
                                adapter_sha256=sha,
                                llama_adapter_id=0,
                            )

                        try:
                            with self.assertRaises(CheckpointMismatchError) as ctx:
                                modeld.mount_stack([raw])
                            self.assertEqual(ctx.exception.code, "ERR_CHECKPOINT_MISMATCH")

                            # Scale must NEVER have been activated!
                            if 0 in server.state.adapters:
                                self.assertAlmostEqual(server.state.adapters[0]["scale"], 0.0)
                            self.assertEqual(modeld.status()["active_adapters"], [])
                            self.assertEqual(modeld.ledger.count, 1)
                        finally:
                            if adapter_file.is_file():
                                os.chmod(adapter_file, 0o600)
                            elif adapter_file.is_dir():
                                shutil.rmtree(adapter_file)
                finally:
                    server.stop()


if __name__ == "__main__":
    unittest.main()

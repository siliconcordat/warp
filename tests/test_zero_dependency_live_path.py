"""Regression tests proving zero-dependency stdlib execution across offline and live Stage 1-4 paths (Issue #7)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.helpers import CANARY_SUITE_PATH
from tests.live_gguf_stdlib import (
    GGML_TYPE_F16,
    GGUF_MAGIC,
    GGUF_VERSION,
    build_deterministic_f16_lora_and_cert,
    build_llama_server_lora_flags_stdlib,
    read_gguf_model_geometry,
    write_gguf_lora_stdlib,
)
from tests.mock_llama_server import MockLlamaServer
from warp_modeld.canary import CanarySuite
from warp_modeld.certificate import CertificateBounds, FixedRadixCertificate, RationalBound
from warp_modeld.fingerprint import (
    AdapterBundle,
    CheckpointFingerprint,
    Plane1Lineage,
    RuntimeFlags,
    compute_file_sha256,
    sha256_json,
)
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.server import WarpUnixRpcClient, WarpUnixRpcServer
from warp_modeld.verifier import WarpModeld


def _pack_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _build_synthetic_base_gguf(
    *,
    meta_entries: list[tuple[str, int, bytes]],
    tensor_entries: list[tuple[str, tuple[int, ...], int, int]],
    magic: bytes = GGUF_MAGIC,
    version: int = GGUF_VERSION,
    n_tensors_override: int | None = None,
    n_kv_override: int | None = None,
) -> bytes:
    n_tensors = len(tensor_entries) if n_tensors_override is None else n_tensors_override
    n_kv = len(meta_entries) if n_kv_override is None else n_kv_override
    parts = [struct.pack("<4sIQQ", magic, version, n_tensors, n_kv)]
    for key, vtype, vbytes in meta_entries:
        parts.append(_pack_str(key) + struct.pack("<I", vtype) + vbytes)
    for name, ggml_ne, ggml_type, rel_off in tensor_entries:
        parts.append(_pack_str(name) + struct.pack("<I", len(ggml_ne)))
        for d in ggml_ne:
            parts.append(struct.pack("<Q", d))
        parts.append(struct.pack("<IQ", ggml_type, rel_off))
    return b"".join(parts)


class TestZeroDependencyLivePath(unittest.TestCase):
    def test_no_numpy_or_weft_imports_in_src_or_tests(self) -> None:
        """AST guard: neither src/ nor tests/ imports numpy or weft anywhere (Issue #7)."""
        for root_name in ("src", "tests"):
            for py_path in sorted((_REPO_ROOT / root_name).rglob("*.py")):
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            self.assertNotIn(top, ("numpy", "weft"), msg=f"forbidden import in {py_path.name}")
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top = node.module.split(".")[0]
                        self.assertNotIn(top, ("numpy", "weft"), msg=f"forbidden from-import in {py_path.name}")

    def test_live_stage1_to_stage4_pipeline_runs_with_numpy_and_weft_hard_blocked(self) -> None:
        """Full Stage 1-4 live-path lifecycle succeeds with sys.modules['numpy']=None and ['weft']=None."""
        with mock.patch.dict(sys.modules, {"numpy": None, "weft": None}):
            with self.assertRaises(ModuleNotFoundError):
                __import__("numpy")
            with self.assertRaises(ModuleNotFoundError):
                __import__("weft")

            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                base_gguf = tmp / "synthetic-qwen2.gguf"
                # Include tokenizer.* string array, scalar array, nested array, and non-tokenizer scalars.
                nested_arr = struct.pack("<IQ", 9, 1) + struct.pack("<IQBB", 0, 2, 1, 2)
                base_meta = [
                    ("general.architecture", 8, _pack_str("qwen2")),
                    ("general.name", 8, _pack_str("Qwen2-Synthetic-Tiny")),
                    ("qwen2.block_count", 4, struct.pack("<I", 2)),
                    ("qwen2.embedding_length", 4, struct.pack("<I", 16)),
                    ("qwen2.feed_forward_length", 4, struct.pack("<I", 32)),
                    ("tokenizer.ggml.model", 8, _pack_str("gpt2")),
                    ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
                    ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _pack_str("a") + _pack_str("b")),
                    ("tokenizer.ggml.scores", 9, struct.pack("<IQff", 6, 2, 0.1, 0.2)),
                    ("custom.u32_arr", 9, struct.pack("<IQII", 4, 2, 7, 8)),
                    ("custom.str_arr", 9, struct.pack("<IQ", 8, 1) + _pack_str("ok")),
                    ("custom.nested_arr", 9, nested_arr),
                ]
                base_bytes = _build_synthetic_base_gguf(
                    meta_entries=base_meta,
                    tensor_entries=[
                        ("blk.0.ffn_up.weight", (16, 32), GGML_TYPE_F16, 0),
                        ("blk.1.ffn_up.weight", (16, 32), GGML_TYPE_F16, 1024),
                    ],
                )
                base_gguf.write_bytes(base_bytes)

                geom = read_gguf_model_geometry(base_gguf)
                self.assertEqual(geom["architecture"], "qwen2")
                self.assertEqual(geom["block_count"], 2)
                self.assertEqual(geom["embedding_length"], 16)
                self.assertEqual(geom["feed_forward_length"], 32)
                target_tensor = "blk.1.ffn_up.weight"
                self.assertEqual(geom["tensors"][target_tensor]["shape"], (32, 16))
                filtered = read_gguf_model_geometry(base_gguf, tensor_names=[target_tensor])
                self.assertEqual(list(filtered["tensors"].keys()), [target_tensor])

                d_out, d_in = geom["tensors"][target_tensor]["shape"]
                rank = 4
                factor_b, factor_a, weft_cert_dict, weft_cert_sha = build_deterministic_f16_lora_and_cert(
                    d_out=d_out,
                    d_in=d_in,
                    rank=rank,
                    seed=20261119,
                    radix_scale_exp2=24,
                    projector_epoch=0,
                )
                self.assertGreater(weft_cert_dict["frobenius_sq_scaled"], 0)
                self.assertEqual(weft_cert_dict["spectral_norm_sq_scaled"], weft_cert_dict["frobenius_sq_scaled"])

                lora_path = tmp / "corpus-f16.gguf"
                lora_rebuild = tmp / "corpus-f16-rebuild.gguf"
                _, lora_sha = write_gguf_lora_stdlib(
                    lora_path,
                    architecture="qwen2",
                    lora_alpha=float(rank),
                    layer_factors={target_tensor: (factor_b, factor_a)},
                    extra_metadata={
                        "weft.slot": "mde:corpus",
                        "weft.certificate_sha256": weft_cert_sha,
                        "weft.deterministic": True,
                        "weft.rank": rank,
                        "weft.epoch_u64": 1 << 40,
                        "weft.scale_hint": 1.0,
                    },
                )
                _, lora_rebuild_sha = write_gguf_lora_stdlib(
                    lora_rebuild,
                    architecture="qwen2",
                    lora_alpha=float(rank),
                    layer_factors={target_tensor: (factor_b, factor_a)},
                    extra_metadata={
                        "weft.slot": "mde:corpus",
                        "weft.certificate_sha256": weft_cert_sha,
                        "weft.deterministic": True,
                        "weft.rank": rank,
                        "weft.epoch_u64": 1 << 40,
                        "weft.scale_hint": 1.0,
                    },
                )
                self.assertEqual(lora_sha, lora_rebuild_sha)

                flags_cli = build_llama_server_lora_flags_stdlib([lora_path], slot_save_path=tmp / "slots")
                self.assertIn("--lora-init-without-apply", flags_cli)
                self.assertIn("--slot-save-path", flags_cli)
                flags_no_slot = build_llama_server_lora_flags_stdlib([lora_path])
                self.assertNotIn("--slot-save-path", flags_no_slot)
                flags_str_path = build_llama_server_lora_flags_stdlib(
                    [str(lora_path)], initial_scale=1, slot_save_path=str(tmp / "slots_str")
                )
                self.assertEqual(flags_str_path[1], f"{lora_path}:1.0")
                self.assertEqual(flags_str_path[-1], str(tmp / "slots_str"))

                # Run full Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> kill-switch lifecycle over Unix RPC!
                mock_srv = MockLlamaServer()
                base_url = mock_srv.start()
                try:
                    base_weights_sha = compute_file_sha256(base_gguf)
                    ckpt = CheckpointFingerprint(
                        model_id="qwen2-live-stdlib",
                        layout_sha256=sha256_json({"arch": "qwen2", "shape": [d_out, d_in]}),
                        weights_sha256=base_weights_sha,
                        config_sha256=sha256_json({"block_count": 2}),
                        tokenizer_sha256=sha256_json({"tokenizer": "gguf-embedded"}),
                        chat_template_sha256=sha256_json({"chat_template": "gguf-embedded"}),
                        safetensors_provenance_sha256=None,
                    )
                    lineage = Plane1Lineage(
                        corpus_tree_sha256=hashlib.sha256(b"stdlib-corpus-v1").hexdigest(),
                        domain_commits={"corpus": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"},
                        contested_heads=(),
                    )
                    runtime_flags = RuntimeFlags(
                        llama_server_commit="7fe450e19",
                        quantization="Q4_K_M",
                        ctx_size=512,
                        kv_cache_type="f16",
                        ctx_checkpoints=0,
                        swa_checkpoints=0,
                        lora_init_without_apply=True,
                    )
                    warp_cert = FixedRadixCertificate.from_weft_dict(
                        weft_cert_dict,
                        adapter_id="mde:corpus",
                        adapter_sha256=lora_sha,
                        base_weights_sha256=base_weights_sha,
                        expected_weft_sha256=weft_cert_sha,
                    )
                    bundle = AdapterBundle(
                        adapter_id="mde:corpus",
                        domain="corpus",
                        llama_adapter_id=0,
                        adapter_sha256=lora_sha,
                        adapter_path=str(lora_path),
                        scale_num=1000,
                        scale_den=1000,
                        checkpoint=ckpt,
                        source_lineage=lineage,
                        runtime_flags_sha256=runtime_flags.canonical_digest(),
                        derivation_digest=sha256_json({"compiler": "stdlib-f16", "rank": rank}),
                        certificate=warp_cert,
                    )
                    signer = HmacSha256Signer(b"stdlib-live-key-32-bytes-0000001", key_id="warp-stdlib")
                    ledger_path = tmp / "ledger.jsonl"
                    modeld = WarpModeld(
                        pinned_checkpoint=ckpt,
                        pinned_lineage=lineage,
                        live_runtime_flags=runtime_flags,
                        single_bounds=CertificateBounds(
                            max_frobenius_sq=RationalBound(50, 1),
                            max_spectral_norm_sq=RationalBound(50, 1),
                            max_leakage_energy=RationalBound(1, 1),
                            max_safety_dot_sq=RationalBound(1, 1),
                        ),
                        joint_bounds=CertificateBounds(
                            max_frobenius_sq=RationalBound(100, 1),
                            max_spectral_norm_sq=RationalBound(100, 1),
                            max_leakage_energy=RationalBound(2, 1),
                            max_safety_dot_sq=RationalBound(2, 1),
                        ),
                        canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                        llama_client=LlamaServerClient(base_url, require_localhost=True),
                        ledger=SignedLedger(ledger_path, signer),
                    )
                    sock_path = tmp / "warp.sock"
                    rpc_server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                    rpc_server.start()
                    try:
                        rpc = WarpUnixRpcClient(sock_path, timeout_seconds=10.0)
                        r_mnt = rpc.call("wmcp_slot_mount", {"bundles": [bundle.to_dict()]})
                        self.assertIn("result", r_mnt, msg=f"mount failed: {r_mnt}")
                        r_ks = rpc.call("warp_kill_switch", {"reason": "stdlib_test_done"})
                        self.assertTrue(r_ks["result"]["kill_switch_engaged"])
                        receipts = LedgerVerifier.verify_file(ledger_path, signer)
                        self.assertEqual([r.event for r in receipts], ["mount", "kill_switch"])
                    finally:
                        rpc_server.stop()
                finally:
                    mock_srv.stop()

    def test_stdlib_gguf_helpers_fail_closed_on_malformed_inputs(self) -> None:
        """Cover every error and boundary guard in tests/live_gguf_stdlib.py."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            def check_bad_gguf(raw: bytes, pattern: str) -> None:
                p = tmp / "bad.gguf"
                p.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, pattern):
                    read_gguf_model_geometry(p)

            check_bad_gguf(b"GGUF\x03\x00", "truncated GGUF stream")
            check_bad_gguf(_build_synthetic_base_gguf(meta_entries=[], tensor_entries=[], magic=b"NOPE"), "invalid GGUF magic")
            check_bad_gguf(_build_synthetic_base_gguf(meta_entries=[], tensor_entries=[], version=2), "unsupported GGUF version")
            check_bad_gguf(
                _build_synthetic_base_gguf(meta_entries=[], tensor_entries=[], n_kv_override=100_001),
                "exceed safety cap",
            )
            check_bad_gguf(
                _build_synthetic_base_gguf(meta_entries=[], tensor_entries=[], n_tensors_override=100_001),
                "exceed safety cap",
            )
            # Boundary n_kv=0, n_tensors=0 passes cleanly; exact-EOF skipped string and exact-EOF scalar pass cleanly!
            empty_ok = tmp / "empty_ok.gguf"
            empty_ok.write_bytes(_build_synthetic_base_gguf(meta_entries=[], tensor_entries=[]))
            self.assertEqual(read_gguf_model_geometry(empty_ok)["n_tensors"], 0)
            eof_skip_str = tmp / "eof_skip_str.gguf"
            eof_skip_str.write_bytes(
                _build_synthetic_base_gguf(meta_entries=[("tokenizer.eof", 8, _pack_str("z"))], tensor_entries=[])
            )
            self.assertEqual(read_gguf_model_geometry(eof_skip_str)["n_tensors"], 0)
            eof_scalar = tmp / "eof_scalar.gguf"
            eof_scalar.write_bytes(
                _build_synthetic_base_gguf(meta_entries=[("general.u32", 4, struct.pack("<I", 7))], tensor_entries=[])
            )
            self.assertEqual(read_gguf_model_geometry(eof_scalar)["n_tensors"], 0)

            # Truncated / invalid UTF-8 / bad value type / bad array / depth exceeded / bad ndim
            check_bad_gguf(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + struct.pack("<Q", 50) + b"abc", "string length exceeds")
            check_bad_gguf(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + struct.pack("<Q", 2) + b"\xff\xfe", "invalid UTF-8")
            check_bad_gguf(
                struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + _pack_str("tokenizer.x") + struct.pack("<IQ", 8, 50) + b"abc",
                "skipped string extends",
            )
            check_bad_gguf(
                struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + _pack_str("k") + struct.pack("<I", 99),
                "unsupported GGUF metadata value type",
            )
            check_bad_gguf(
                struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + _pack_str("k") + struct.pack("<I", 4) + b"\x00\x00",
                "scalar metadata value extends",
            )
            check_bad_gguf(
                struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + _pack_str("k") + struct.pack("<IIQ", 9, 4, 5) + b"\x00",
                "scalar array extends",
            )
            check_bad_gguf(
                struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + _pack_str("k") + struct.pack("<IIQ", 9, 99, 1),
                "unsupported GGUF array element type",
            )
            # Nested array at max valid depth (7) passes, depth 8 fails
            nested_ok = struct.pack("<IQB", 0, 1, 1)
            for _ in range(7):
                nested_ok = struct.pack("<IQ", 9, 1) + nested_ok
            ok_depth_file = tmp / "depth_ok.gguf"
            ok_depth_file.write_bytes(_build_synthetic_base_gguf(meta_entries=[("k", 9, nested_ok)], tensor_entries=[]))
            self.assertEqual(read_gguf_model_geometry(ok_depth_file)["n_tensors"], 0)
            nested_deep = struct.pack("<IQ", 9, 1) + nested_ok
            check_bad_gguf(
                _build_synthetic_base_gguf(meta_entries=[("k", 9, nested_deep)], tensor_entries=[]),
                "nesting depth exceeded",
            )
            # ndim == 0, ndim == 8 (passes), ndim == 9 (fails)
            check_bad_gguf(
                _build_synthetic_base_gguf(meta_entries=[], tensor_entries=[("t", (), GGML_TYPE_F16, 0)]),
                "invalid GGUF tensor ndim",
            )
            ok_ndim8 = tmp / "ndim8.gguf"
            ok_ndim8.write_bytes(
                _build_synthetic_base_gguf(meta_entries=[], tensor_entries=[("t", (1,) * 8, GGML_TYPE_F16, 0)])
            )
            self.assertEqual(len(read_gguf_model_geometry(ok_ndim8)["tensors"]["t"]["shape"]), 8)
            check_bad_gguf(
                _build_synthetic_base_gguf(meta_entries=[], tensor_entries=[("t", (1,) * 9, GGML_TYPE_F16, 0)]),
                "invalid GGUF tensor ndim",
            )

            # Writer validation guards
            out_gguf = tmp / "out.gguf"
            with self.assertRaisesRegex(ValueError, "layer_factors must be non-empty"):
                write_gguf_lora_stdlib(out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={})
            for bad_arch in ("", "   ", 123, "qwen" + chr(0xD800)):
                with self.assertRaises(ValueError):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture=bad_arch,  # type: ignore[arg-type]
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                    )
            for bad_alpha in (True, float("nan"), float("inf"), "1.0", 10**400, 1e100):
                with self.assertRaises(ValueError):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=bad_alpha,  # type: ignore[arg-type]
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                    )
            with self.assertRaisesRegex(ValueError, "at least one row"):
                write_gguf_lora_stdlib(out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={"w": ([], [[1.0]])})
            with self.assertRaisesRegex(ValueError, "at least one column"):
                write_gguf_lora_stdlib(out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={"w": ([[]], [[1.0]])})
            with self.assertRaisesRegex(ValueError, "ragged row"):
                write_gguf_lora_stdlib(
                    out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={"w": ([[1.0], [1.0, 2.0]], [[1.0]])}
                )
            for bad_entry in (True, float("nan"), float("inf"), "x", 10**400):
                with self.assertRaises(ValueError):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[bad_entry]], [[1.0]])},  # type: ignore[list-item]
                    )
            with self.assertRaisesRegex(ValueError, "F16 overflow"):
                write_gguf_lora_stdlib(
                    out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={"w": ([[1e30]], [[1.0]])}
                )
            with self.assertRaisesRegex(ValueError, "incompatible LoRA"):
                write_gguf_lora_stdlib(
                    out_gguf, architecture="qwen2", lora_alpha=1.0, layer_factors={"w": ([[1.0, 2.0]], [[1.0]])}
                )
            # Default extra_metadata=None works cleanly and matches explicit None bit-for-bit!
            default_gguf = tmp / "default_meta.gguf"
            explicit_none_gguf = tmp / "none_meta.gguf"
            _, default_sha = write_gguf_lora_stdlib(
                default_gguf, architecture="qwen2", lora_alpha=2, layer_factors={"w": ([[1.0]], [[1.0]])}
            )
            _, none_sha = write_gguf_lora_stdlib(
                explicit_none_gguf,
                architecture="qwen2",
                lora_alpha=2,
                layer_factors={"w": ([[1.0]], [[1.0]])},
                extra_metadata=None,
            )
            self.assertEqual(default_sha, none_sha)
            self.assertEqual(read_gguf_model_geometry(default_gguf)["n_tensors"], 2)

            # Non-mapping extra_metadata, bad keys, surrogates, and reserved header overrides fail closed!
            for bad_meta_container in ((), [("k", 1)], "not-a-mapping"):
                with self.assertRaisesRegex(TypeError, "extra_metadata must be a mapping"):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                        extra_metadata=bad_meta_container,  # type: ignore[arg-type]
                    )
            for bad_meta in ({123: 1}, {"": 1}, {"surr": chr(0xD800)}, {chr(0xD800): "v"}):
                with self.assertRaises(ValueError):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                        extra_metadata=bad_meta,  # type: ignore[arg-type]
                    )
            for reserved_key, reserved_val in (
                ("general.alignment", 64),
                ("general.alignment", 0),
                ("general.alignment", "x"),
                ("general.alignment", 32),
                ("adapter.lora.alpha", 99.0),
                ("adapter.type", "evil"),
                ("general.architecture", "llama"),
                ("general.type", "model"),
                ("weft.schema", "evil/v0"),
            ):
                with self.assertRaisesRegex(ValueError, "reserved GGUF LoRA metadata key"):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                        extra_metadata={reserved_key: reserved_val},
                    )
            # Metadata KV packing bounds & types
            for bad_int in (-1, 1 << 64):
                with self.assertRaisesRegex(ValueError, "unsupported GGUF integer"):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                        extra_metadata={"bad_i": bad_int},
                    )
            for bad_float in (float("nan"), 1e100):
                with self.assertRaises(ValueError):
                    write_gguf_lora_stdlib(
                        out_gguf,
                        architecture="qwen2",
                        lora_alpha=1.0,
                        layer_factors={"w": ([[1.0]], [[1.0]])},
                        extra_metadata={"bad_f": bad_float},
                    )
            with self.assertRaisesRegex(TypeError, "unsupported GGUF metadata type"):
                write_gguf_lora_stdlib(
                    out_gguf,
                    architecture="qwen2",
                    lora_alpha=1.0,
                    layer_factors={"w": ([[1.0]], [[1.0]])},
                    extra_metadata={"bad_t": [1, 2]},
                )
            # Boundary uint32 max and uint64 max pass cleanly!
            write_gguf_lora_stdlib(
                out_gguf,
                architecture="qwen2",
                lora_alpha=1.0,
                layer_factors={"w": ([[1.0]], [[1.0]])},
                extra_metadata={"u32_zero": 0, "u32_max": 0xFFFFFFFF, "u64_max": 0xFFFFFFFFFFFFFFFF},
            )

            # Cert builder & CLI flags validation guards
            for bad_kw in (
                {"d_out": 0, "d_in": 4, "rank": 2},
                {"d_out": True, "d_in": 4, "rank": 2},
                {"d_out": 4, "d_in": 0, "rank": 2},
                {"d_out": 4, "d_in": True, "rank": 2},
                {"d_out": 4, "d_in": 4, "rank": 0},
                {"d_out": 4, "d_in": 4, "rank": True},
                {"d_out": 4, "d_in": 4, "rank": 2, "radix_scale_exp2": 7},
                {"d_out": 4, "d_in": 4, "rank": 2, "radix_scale_exp2": 49},
                {"d_out": 4, "d_in": 4, "rank": 2, "radix_scale_exp2": True},
            ):
                with self.assertRaises(ValueError):
                    build_deterministic_f16_lora_and_cert(**bad_kw)  # type: ignore[arg-type]
            # Valid radix boundaries 8 and 48 pass cleanly
            build_deterministic_f16_lora_and_cert(d_out=4, d_in=4, rank=2, radix_scale_exp2=8)
            build_deterministic_f16_lora_and_cert(d_out=4, d_in=4, rank=2, radix_scale_exp2=48)

            with self.assertRaisesRegex(ValueError, "adapter_paths must be non-empty"):
                build_llama_server_lora_flags_stdlib([])
            for bad_container in ("/tmp/ok.gguf", b"/tmp/ok.gguf", bytearray(b"/tmp/ok.gguf"), {out_gguf}, 123):
                with self.assertRaisesRegex(ValueError, "non-string sequence"):
                    build_llama_server_lora_flags_stdlib(bad_container)  # type: ignore[arg-type]
            for bad_scale in (True, float("nan"), float("inf"), "0.0", 10**400):
                with self.assertRaises(ValueError):
                    build_llama_server_lora_flags_stdlib([out_gguf], initial_scale=bad_scale)  # type: ignore[arg-type]
            for bad_adapter_path in (
                "",
                "   ",
                ".",
                "./",
                Path(""),
                Path("."),
                "/tmp/ok.gguf,/tmp/evil.gguf:1.0",
                "/tmp/ok:1.gguf",
                Path("/tmp/ok,bad.gguf"),
                Path("/tmp/ok:bad.gguf"),
                "/tmp/ok\x00.gguf",
                Path("/tmp/ok\x00.gguf"),
                "/tmp/bad" + chr(0xD800) + ".gguf",
                Path("/tmp/bad" + chr(0xD800) + ".gguf"),
                b"/tmp/ok.gguf",
                123,
                None,
            ):
                with self.assertRaises(ValueError):
                    build_llama_server_lora_flags_stdlib([bad_adapter_path])  # type: ignore[list-item]
            for bad_slot_path in (
                "",
                "   ",
                ".",
                Path(""),
                "/tmp/slots\x00x",
                "/tmp/slots" + chr(0xD800),
                b"/tmp/slots",
                123,
            ):
                with self.assertRaises(ValueError):
                    build_llama_server_lora_flags_stdlib([out_gguf], slot_save_path=bad_slot_path)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

"""Stage 2 regression and adversarial tests binding GGUF v3 LoRA binaries to FixedRadixCertificate (Issue #13)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from warp_modeld.canary import CanarySuite
from warp_modeld.certificate import CertificateBounds, FixedRadixCertificate, RationalBound
from warp_modeld.errors import CertificateError, CheckpointMismatchError
from warp_modeld.fingerprint import AdapterBundle
from warp_modeld.gguf_lora_cert import (
    _decode_f16_bits,
    _decode_f32_bits,
    _is_symmetric_psd_bareiss,
    verify_adapter_bytes_against_certificate,
    verify_adapter_file_against_certificate,
)
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_runtime_flags,
)
from tests.mock_llama_server import MockLlamaServer

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def _pack_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _align_up(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_matrix_f16(rows: list[list[float]]) -> bytes:
    return b"".join(struct.pack(f"<{len(row)}e", *row) for row in rows)


def _pack_matrix_f32(rows: list[list[float]]) -> bytes:
    return b"".join(struct.pack(f"<{len(row)}f", *row) for row in rows)


def _build_gguf_raw(
    *,
    meta_entries: list[tuple[str, int, bytes]],
    tensor_entries: list[tuple[str, tuple[int, ...], int, bytes, int | None]],
    magic: bytes = b"GGUF",
    version: int = 3,
    alignment: int = 32,
) -> bytes:
    """Build deterministic GGUF binary bytes; tensor_entries are (name, shape, ggml_type, raw, rel_off_override)."""
    parts = [struct.pack("<4sIQQ", magic, version, len(tensor_entries), len(meta_entries))]
    for key, vtype, vbytes in meta_entries:
        parts.append(_pack_str(key) + struct.pack("<I", vtype) + vbytes)

    rel = 0
    blobs: list[tuple[int, bytes]] = []
    for name, shape, ggml_type, raw, rel_override in tensor_entries:
        use_rel = rel if rel_override is None else rel_override
        # GGUF stores dims reversed: ne[0] = inner dim = cols, ne[1] = rows (for 2D).
        dims_rev = tuple(reversed(shape))
        parts.append(_pack_str(name) + struct.pack("<I", len(dims_rev)))
        for d in dims_rev:
            parts.append(struct.pack("<Q", d))
        parts.append(struct.pack("<IQ", ggml_type, use_rel))
        blobs.append((use_rel, raw))
        rel = _align_up(use_rel + len(raw), alignment)

    header = b"".join(parts)
    data_start = _align_up(len(header), alignment)
    out = [header, b"\x00" * (data_start - len(header))]
    written = 0
    for off, raw in blobs:
        if off > written:
            out.append(b"\x00" * (off - written))
            written = off
        if off == written:
            out.append(raw)
            written += len(raw)
    return b"".join(out)


def _build_lora_gguf(
    layers: dict[str, tuple[list[list[float]], list[list[float]]]],
    *,
    alpha: float,
    dtype: int = GGML_TYPE_F16,
    layer_dtypes: dict[str, tuple[int, int]] | None = None,
    alignment: int | None = 32,
    extra_meta: list[tuple[str, int, bytes]] | None = None,
) -> bytes:
    meta: list[tuple[str, int, bytes]] = [
        ("adapter.lora.alpha", GGUF_TYPE_FLOAT32, struct.pack("<f", alpha)),
        ("adapter.type", GGUF_TYPE_STRING, _pack_str("lora")),
        ("general.architecture", GGUF_TYPE_STRING, _pack_str("qwen2")),
        ("general.type", GGUF_TYPE_STRING, _pack_str("adapter")),
    ]
    if alignment is not None:
        meta.append(("general.alignment", GGUF_TYPE_UINT32, struct.pack("<I", alignment)))
    if extra_meta:
        meta.extend(extra_meta)

    tensors: list[tuple[str, tuple[int, ...], int, bytes, int | None]] = []
    for base in sorted(layers):
        b_mat, a_mat = layers[base]
        b_type, a_type = layer_dtypes[base] if layer_dtypes and base in layer_dtypes else (dtype, dtype)
        a_raw = _pack_matrix_f16(a_mat) if a_type == GGML_TYPE_F16 else _pack_matrix_f32(a_mat)
        b_raw = _pack_matrix_f16(b_mat) if b_type == GGML_TYPE_F16 else _pack_matrix_f32(b_mat)
        tensors.append((f"{base}.lora_a", (len(a_mat), len(a_mat[0])), a_type, a_raw, None))
        tensors.append((f"{base}.lora_b", (len(b_mat), len(b_mat[0])), b_type, b_raw, None))
    return _build_gguf_raw(meta_entries=meta, tensor_entries=tensors, alignment=alignment or 32)


def _make_cert(
    adapter_bytes: bytes,
    *,
    frob_scaled: int,
    spec_scaled: int,
    radix_scale: int = 1 << 24,
    adapter_id: str = "mde:test",
    base_weights_sha256: str | None = None,
    adapter_sha256_override: str | None = None,
) -> FixedRadixCertificate:
    sha = adapter_sha256_override or hashlib.sha256(adapter_bytes).hexdigest()
    ckpt_sha = base_weights_sha256 or make_checkpoint().weights_sha256
    return FixedRadixCertificate.create(
        adapter_id=adapter_id,
        adapter_sha256=sha,
        base_weights_sha256=ckpt_sha,
        radix_scale=radix_scale,
        frobenius_sq_scaled=frob_scaled,
        spectral_norm_sq_scaled=spec_scaled,
        leakage_energy_scaled=100,
        max_safety_dot_sq_scaled=10,
    )


def _make_mount_bundle(
    adapter_path: Path,
    adapter_bytes: bytes,
    cert: FixedRadixCertificate,
    *,
    ckpt=None,
    lineage=None,
    flags=None,
) -> dict:
    ckpt = ckpt or make_checkpoint()
    lineage = lineage or make_lineage()
    flags = flags or make_runtime_flags()
    sha = hashlib.sha256(adapter_bytes).hexdigest()
    bundle = AdapterBundle(
        adapter_id=cert.adapter_id,
        domain="corpus",
        llama_adapter_id=0,
        adapter_sha256=sha,
        adapter_path=str(adapter_path),
        scale_num=1000,
        scale_den=1000,
        checkpoint=ckpt,
        source_lineage=lineage,
        runtime_flags_sha256=flags.canonical_digest(),
        derivation_digest=hashlib.sha256(b"issue13-test-derivation").hexdigest(),
        certificate=cert,
    )
    return bundle.to_dict()


class TestIssue13Stage2GgufBinding(unittest.TestCase):
    def test_issue13_reproducer_honest_mounts_while_b_times_8_and_alpha_32_rejected_before_stage3(self) -> None:
        """Exact Issue #13 reproducer: honest adapter mounts; b-times-8.gguf and alpha-32.gguf fail Stage 2 with 0 llama-server calls."""
        rng = random.Random(2026)
        d_out, d_in, rank = 64, 64, 4
        b_honest = [[round(rng.uniform(-0.2, 0.2), 3) for _ in range(rank)] for _ in range(d_out)]
        a_honest = [[round(rng.uniform(-0.2, 0.2), 3) for _ in range(d_in)] for _ in range(rank)]
        b_times_8 = [[x * 8.0 for x in row] for row in b_honest]

        honest_bytes = _build_lora_gguf({"blk.0.attn_q.weight": (b_honest, a_honest)}, alpha=float(rank))
        bx8_bytes = _build_lora_gguf({"blk.0.attn_q.weight": (b_times_8, a_honest)}, alpha=float(rank))
        alpha32_bytes = _build_lora_gguf({"blk.0.attn_q.weight": (b_honest, a_honest)}, alpha=32.0)

        ckpt = make_checkpoint()
        lineage = make_lineage()
        flags = make_runtime_flags()
        scale = 1 << 24
        bounds = CertificateBounds(
            max_frobenius_sq=RationalBound(20, 1),
            max_spectral_norm_sq=RationalBound(20, 1),
            max_leakage_energy=RationalBound(1, 1),
            max_safety_dot_sq=RationalBound(1, 1),
        )

        # Compute tight conservative honest certificate values (true ||BA||_F^2 ~ 7.5 < 12 < 20).
        honest_frob_scaled = 12 * scale
        honest_spec_scaled = 12 * scale

        for label, gbytes, expect_mount, expected_err in (
            ("b-times-8.gguf", bx8_bytes, False, "Frobenius|spectral"),
            ("alpha-32.gguf", alpha32_bytes, False, "alpha"),
            ("honest.gguf", honest_bytes, True, ""),
        ):
            mock = MockLlamaServer()
            url = mock.start()
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td)
                    path = tmp / label
                    path.write_bytes(gbytes)
                    cert = _make_cert(
                        gbytes,
                        frob_scaled=honest_frob_scaled,
                        spec_scaled=honest_spec_scaled,
                        radix_scale=scale,
                        base_weights_sha256=ckpt.weights_sha256,
                    )
                    bundle = _make_mount_bundle(path, gbytes, cert, ckpt=ckpt, lineage=lineage, flags=flags)
                    signer = HmacSha256Signer(b"issue13-verifier-key-32-bytes-01", key_id="warp-i13")
                    ledger_path = tmp / "ledger.jsonl"
                    modeld = WarpModeld(
                        pinned_checkpoint=ckpt,
                        pinned_lineage=lineage,
                        live_runtime_flags=flags,
                        single_bounds=bounds,
                        joint_bounds=make_joint_bounds(),
                        canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                        llama_client=LlamaServerClient(url),
                        ledger=SignedLedger(ledger_path, signer),
                    )
                    if expect_mount:
                        mounted = modeld.mount_stack([bundle])
                        self.assertEqual(len(mounted.active_adapters), 1)
                        self.assertTrue(mounted.stage1_reports[0].adapter_file_verified)
                        self.assertGreater(len(mock.state.request_log), 0)
                    else:
                        with self.assertRaisesRegex(CertificateError, expected_err):
                            modeld.mount_stack([bundle])
                        # Stage 2 MUST reject BEFORE any HTTP request reaches llama-server.
                        self.assertEqual(mock.state.request_log, [])
                        self.assertEqual(modeld.status()["active_adapters"], [])
                        verified = LedgerVerifier.verify_file(ledger_path, signer)
                        self.assertEqual(len(verified), 1)
                        self.assertEqual(verified[0].event, "reject")
                        self.assertEqual(verified[0].stage_failed, "stage2_certificate")
                        self.assertEqual(verified[0].error_code, "ERR_CERT_INVALID")
            finally:
                mock.stop()

    def test_exact_frobenius_and_spectral_boundaries_and_off_by_one_forgeries(self) -> None:
        """Verify exact rational boundary pass and 1-ULP understatement rejection for Frobenius and spectral norms."""
        # Rank-2 spiked adapter: B = [[1, 0], [0, 0.5]], A = [[1, 0], [0, 1]]
        # BA = diag(1, 0.5) -> ||BA||_F^2 = 1.25 = 5/4, sigma_max(BA)^2 = 1.0.
        b_mat = [[1.0, 0.0], [0.0, 0.5]]
        a_mat = [[1.0, 0.0], [0.0, 1.0]]
        raw = _build_lora_gguf({"blk.0.w": (b_mat, a_mat)}, alpha=2.0)
        radix = 1 << 24
        exact_frob = (5 * radix) // 4
        exact_spec = radix

        rep = verify_adapter_bytes_against_certificate(
            raw, _make_cert(raw, frob_scaled=exact_frob, spec_scaled=exact_spec, radix_scale=radix)
        )
        self.assertEqual(rep.layer_count, 1)
        self.assertEqual(rep.lora_rank, 2)
        self.assertEqual(rep.frobenius_sq_num * 4, 5 << rep.frobenius_sq_shift)
        self.assertGreaterEqual(rep.elapsed_ns, 0)

        # Understate Frobenius by 1 ULP (2^-24) while keeping spectral exact -> rejected!
        with self.assertRaisesRegex(CertificateError, "Frobenius"):
            verify_adapter_bytes_against_certificate(
                raw, _make_cert(raw, frob_scaled=exact_frob - 1, spec_scaled=exact_spec, radix_scale=radix)
            )

        # Keep Frobenius honest (or even inflated!) while understating spectral by 1 ULP -> rejected!
        with self.assertRaisesRegex(CertificateError, "spectral"):
            verify_adapter_bytes_against_certificate(
                raw, _make_cert(raw, frob_scaled=exact_frob * 10, spec_scaled=exact_spec - 1, radix_scale=radix)
            )

    def test_spectral_zero_diagonal_off_diagonal_and_rank_deficient_bareiss_cases(self) -> None:
        """Exercise all Bareiss PSD branches: zero-diag indefinite, diagonal pivoting, rank-3 Schur step, and zero product."""
        radix = 1 << 24

        # 1. Zero-diagonal indefinite M = [[0, -1], [-1, 0]] at tau=1:
        #    A = I_2, B = [[1, 1], [0, 0]] -> BA has frob^2 = 2, spec^2 = 2.
        #    Claiming frob=2, spec=1 yields tau*G_A - G_A G_B G_A = [[0, -1], [-1, 0]] (all zero diags, nonzero off-diag).
        raw_indef = _build_lora_gguf(
            {"blk.0.w": ([[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])},
            alpha=2.0,
        )
        verify_adapter_bytes_against_certificate(
            raw_indef, _make_cert(raw_indef, frob_scaled=2 * radix, spec_scaled=2 * radix, radix_scale=radix)
        )
        with self.assertRaisesRegex(CertificateError, "spectral"):
            verify_adapter_bytes_against_certificate(
                raw_indef, _make_cert(raw_indef, frob_scaled=2 * radix, spec_scaled=1 * radix, radix_scale=radix)
            )

        # 2. Rank-deficient A forcing pivot swap (k=0 diag is 0, k=1 diag > 0):
        #    A = [[0, 0], [1, 0]], B = [[0, 1], [0, 0]] -> BA = [[1, 0], [0, 0]], frob^2 = 1, spec^2 = 1.
        raw_pivot = _build_lora_gguf(
            {"blk.0.w": ([[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]])},
            alpha=2.0,
        )
        verify_adapter_bytes_against_certificate(
            raw_pivot, _make_cert(raw_pivot, frob_scaled=2 * radix, spec_scaled=2 * radix, radix_scale=radix)
        )
        verify_adapter_bytes_against_certificate(
            raw_pivot, _make_cert(raw_pivot, frob_scaled=1 * radix, spec_scaled=1 * radix, radix_scale=radix)
        )
        with self.assertRaisesRegex(CertificateError, "spectral"):
            verify_adapter_bytes_against_certificate(
                raw_pivot, _make_cert(raw_pivot, frob_scaled=2 * radix, spec_scaled=radix - 1, radix_scale=radix)
            )

        # 3. Rank-3 non-diagonal LoRA exercising Bareiss step k >= 1 (exact division by prev > 1):
        #    B = [[1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1]], A = [[1, 0.25, 0], [0, 1, 0.25], [0, 0, 1]]
        #    Exact ||BA||_F^2 = 4.40625, sigma_max(BA)^2 ~= 3.1057525485 -> tight 1-ULP boundary around 3.1057525485!
        b3 = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]]
        a3 = [[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]]
        raw_r3 = _build_lora_gguf({"blk.0.w": (b3, a3)}, alpha=3.0)
        tight_spec_ceil = math.ceil(3.105752548509279 * radix)
        tight_spec_floor = tight_spec_ceil - 1
        verify_adapter_bytes_against_certificate(
            raw_r3, _make_cert(raw_r3, frob_scaled=5 * radix, spec_scaled=tight_spec_ceil, radix_scale=radix)
        )
        with self.assertRaisesRegex(CertificateError, "spectral"):
            verify_adapter_bytes_against_certificate(
                raw_r3, _make_cert(raw_r3, frob_scaled=5 * radix, spec_scaled=tight_spec_floor, radix_scale=radix)
            )

        # 4. Zero product (B != 0, A != 0, B*A == 0) with cert spectral = 0 and frob = 0 passes,
        #    while nonzero B*A with cert spectral = 0 fails.
        raw_zero_prod = _build_lora_gguf(
            {"blk.0.w": ([[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]])},
            alpha=2.0,
        )
        verify_adapter_bytes_against_certificate(
            raw_zero_prod, _make_cert(raw_zero_prod, frob_scaled=0, spec_scaled=0, radix_scale=radix)
        )
        with self.assertRaisesRegex(CertificateError, "spectral_norm_sq_scaled is 0"):
            verify_adapter_bytes_against_certificate(
                raw_pivot, _make_cert(raw_pivot, frob_scaled=2 * radix, spec_scaled=0, radix_scale=radix)
            )

        # 5. Direct unit test of _is_symmetric_psd_bareiss on 3x3 positive-definite, semidefinite, and indefinite matrices.
        self.assertTrue(_is_symmetric_psd_bareiss([[2, 1, 0], [1, 2, 1], [0, 1, 2]]))
        self.assertTrue(_is_symmetric_psd_bareiss([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))
        self.assertFalse(_is_symmetric_psd_bareiss([[2, 1, 0], [1, 2, 3], [0, 3, 2]]))
        self.assertFalse(_is_symmetric_psd_bareiss([[0, 0, 1], [0, 0, 0], [1, 0, 0]]))

    def test_multilayer_sum_frob_and_mixed_f16_f32_and_mixed_rank_rejection(self) -> None:
        """Multi-layer Frobenius sums across layers; mixed F16/F32 supported; mixed rank rejected."""
        radix = 1 << 24
        # Layer 0 (F16): BA = diag(1, 0) -> frob^2 = 1, spec^2 = 1.
        # Layer 1 (F32): BA = diag(1, 0) -> frob^2 = 1, spec^2 = 1.
        # Combined: total frob^2 = 2, max spec^2 = 1.
        layers = {
            "blk.0.attn_q.weight": ([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]),
            "blk.1.attn_k.weight": ([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]),
        }
        raw_mixed = _build_lora_gguf(
            layers,
            alpha=2.0,
            layer_dtypes={
                "blk.0.attn_q.weight": (GGML_TYPE_F16, GGML_TYPE_F32),
                "blk.1.attn_k.weight": (GGML_TYPE_F32, GGML_TYPE_F32),
            },
            alignment=None,  # omit general.alignment to test default=32
        )
        rep = verify_adapter_bytes_against_certificate(
            raw_mixed, _make_cert(raw_mixed, frob_scaled=2 * radix, spec_scaled=1 * radix, radix_scale=radix)
        )
        self.assertEqual(rep.layer_count, 2)
        self.assertEqual(rep.lora_rank, 2)

        # Each layer has frob^2 = 1 <= 1.5, but SUM is 2 > 1.5 -> rejected!
        with self.assertRaisesRegex(CertificateError, "Frobenius"):
            verify_adapter_bytes_against_certificate(
                raw_mixed,
                _make_cert(raw_mixed, frob_scaled=(3 * radix) // 2, spec_scaled=1 * radix, radix_scale=radix),
            )

        # Mixed ranks across layers (rank 2 + rank 1 with alpha=2.0) -> rejected by alpha==rank gate!
        mixed_rank_raw = _build_lora_gguf(
            {
                "blk.0.attn_q.weight": ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]),
                "blk.1.attn_q.weight": ([[1.0], [0.0]], [[1.0, 0.0]]),
            },
            alpha=2.0,
        )
        with self.assertRaisesRegex(CertificateError, "alpha"):
            verify_adapter_bytes_against_certificate(
                mixed_rank_raw,
                _make_cert(mixed_rank_raw, frob_scaled=10 * radix, spec_scaled=10 * radix, radix_scale=radix),
            )

    def test_exact_ieee754_f16_and_f32_bit_decoders_and_nonfinite_rejection(self) -> None:
        """Cover normal, subnormal, signed-zero, max, Inf, and NaN decoding for both F16 and F32."""
        # F16 exact checks (multiples of 2^-24)
        self.assertEqual(_decode_f16_bits(0x0000), 0)
        self.assertEqual(_decode_f16_bits(0x8000), 0)
        self.assertEqual(_decode_f16_bits(0x0001), 1)
        self.assertEqual(_decode_f16_bits(0x8001), -1)
        self.assertEqual(_decode_f16_bits(0x03FF), 1023)
        self.assertEqual(_decode_f16_bits(0x0400), 1024)
        self.assertEqual(_decode_f16_bits(0x3C00), 1 << 24)
        self.assertEqual(_decode_f16_bits(0xB800), -(1 << 23))
        self.assertEqual(_decode_f16_bits(0x7BFF), 65504 << 24)
        self.assertIsNone(_decode_f16_bits(0x7C00))  # +Inf
        self.assertIsNone(_decode_f16_bits(0xFC00))  # -Inf
        self.assertIsNone(_decode_f16_bits(0x7E00))  # qNaN

        # F32 exact checks (multiples of 2^-149)
        self.assertEqual(_decode_f32_bits(0x00000000), 0)
        self.assertEqual(_decode_f32_bits(0x80000000), 0)
        self.assertEqual(_decode_f32_bits(0x00000001), 1)
        self.assertEqual(_decode_f32_bits(0x80000001), -1)
        self.assertEqual(_decode_f32_bits(0x007FFFFF), 0x7FFFFF)
        self.assertEqual(_decode_f32_bits(0x00800000), 0x800000)
        self.assertEqual(_decode_f32_bits(0x3F800000), 1 << 149)
        self.assertEqual(_decode_f32_bits(0xBF000000), -(1 << 148))
        self.assertIsNone(_decode_f32_bits(0x7F800000))  # +Inf
        self.assertIsNone(_decode_f32_bits(0xFF800000))  # -Inf
        self.assertIsNone(_decode_f32_bits(0x7FC00000))  # qNaN

        radix = 1 << 24
        # Subnormal F16 tensor weight works cleanly without underflow.
        subnorm_meta = [
            ("adapter.lora.alpha", GGUF_TYPE_FLOAT32, struct.pack("<f", 1.0)),
            ("adapter.type", GGUF_TYPE_STRING, _pack_str("lora")),
            ("general.type", GGUF_TYPE_STRING, _pack_str("adapter")),
        ]
        raw_subnorm = _build_gguf_raw(
            meta_entries=subnorm_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<H", 0x0001), None),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<H", 0x3C00), None),
            ],
        )
        rep = verify_adapter_bytes_against_certificate(
            raw_subnorm, _make_cert(raw_subnorm, frob_scaled=1, spec_scaled=1, radix_scale=radix)
        )
        self.assertGreater(rep.frobenius_sq_num, 0)

        # NaN and Inf weights in both F16 and F32 rejected fail-closed!
        for dtype, bad_word, label in (
            (GGML_TYPE_F16, struct.pack("<H", 0x7E00), "non-finite F16"),
            (GGML_TYPE_F16, struct.pack("<H", 0x7C00), "non-finite F16"),
            (GGML_TYPE_F32, struct.pack("<I", 0x7FC00000), "non-finite F32"),
            (GGML_TYPE_F32, struct.pack("<I", 0xFF800000), "non-finite F32"),
        ):
            good_word = struct.pack("<H", 0x3C00) if dtype == GGML_TYPE_F16 else struct.pack("<I", 0x3F800000)
            raw_bad = _build_gguf_raw(
                meta_entries=subnorm_meta,
                tensor_entries=[
                    ("blk.0.w.lora_a", (1, 1), dtype, bad_word, None),
                    ("blk.0.w.lora_b", (1, 1), dtype, good_word, None),
                ],
            )
            with self.assertRaisesRegex(CertificateError, label):
                verify_adapter_bytes_against_certificate(
                    raw_bad, _make_cert(raw_bad, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)
                )

    def test_metadata_arrays_custom_alignment_and_strict_parser_rejections(self) -> None:
        """Cover valid GGUF metadata arrays/custom alignment and every malformed/crafted GGUF rejection."""
        radix = 1 << 24

        # Valid extra metadata: scalar array, string array, nested array at max valid depth (7), alignment=64.
        # Build depth-7 nested array of UINT8 (depths 0..7 -> valid!).
        nested_ok = struct.pack("<IQB", GGUF_TYPE_UINT8, 1, 7)
        for _ in range(7):
            nested_ok = struct.pack("<IQ", GGUF_TYPE_ARRAY, 1) + nested_ok
        extra = [
            ("custom.bool", GGUF_TYPE_BOOL, b"\x01"),
            ("custom.u64", GGUF_TYPE_UINT64, struct.pack("<Q", 42)),
            ("custom.u32_arr", GGUF_TYPE_ARRAY, struct.pack("<IQII", GGUF_TYPE_UINT32, 2, 10, 20)),
            ("custom.str_arr", GGUF_TYPE_ARRAY, struct.pack("<IQ", GGUF_TYPE_STRING, 2) + _pack_str("a") + _pack_str("b")),
            ("custom.nested_ok", GGUF_TYPE_ARRAY, nested_ok),
        ]
        valid_raw = _build_lora_gguf(
            {"blk.0.w": ([[1.0]], [[1.0]])}, alpha=1.0, alignment=64, extra_meta=extra
        )
        verify_adapter_bytes_against_certificate(
            valid_raw, _make_cert(valid_raw, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)
        )

        # Depth-8 nested array -> rejected!
        nested_too_deep = struct.pack("<IQ", GGUF_TYPE_ARRAY, 1) + nested_ok
        raw_deep = _build_lora_gguf(
            {"blk.0.w": ([[1.0]], [[1.0]])},
            alpha=1.0,
            extra_meta=[("custom.too_deep", GGUF_TYPE_ARRAY, nested_too_deep)],
        )
        with self.assertRaisesRegex(CertificateError, "nesting depth"):
            verify_adapter_bytes_against_certificate(
                raw_deep, _make_cert(raw_deep, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)
            )

        def check_bad(raw: bytes, pattern: str) -> None:
            cert = _make_cert(raw, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)
            with self.assertRaisesRegex(CertificateError, pattern):
                verify_adapter_bytes_against_certificate(raw, cert)

        base_meta = [
            ("adapter.lora.alpha", GGUF_TYPE_FLOAT32, struct.pack("<f", 1.0)),
            ("adapter.type", GGUF_TYPE_STRING, _pack_str("lora")),
            ("general.type", GGUF_TYPE_STRING, _pack_str("adapter")),
        ]
        one_a = ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), None)
        one_b = ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), None)

        # Header & magic & version & empty tensors
        check_bad(b"GGUF\x03\x00\x00\x00" + b"\x00" * 15, "truncated GGUF header")
        check_bad(struct.pack("<4sIQQ", b"NOPE", 3, 2, 0), "invalid GGUF magic")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 2, 2, 0), "unsupported GGUF version")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0), "no tensors")

        # Truncated strings, non-UTF8 strings, truncated KV value type, unknown vtype, truncated scalar/array
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + b"\x00" * 7, "truncated GGUF string length")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + struct.pack("<Q", 5) + b"abcd", "truncated GGUF string payload")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + struct.pack("<Q", 2) + b"\xff\xfe", "invalid UTF-8")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + b"\x00\x00\x00", "truncated GGUF metadata value type")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<I", 99), "unsupported GGUF metadata value type")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<I", GGUF_TYPE_UINT32) + b"\x00\x00\x00", "truncated GGUF metadata scalar")
        check_bad(struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<I", GGUF_TYPE_ARRAY) + b"\x00" * 11, "truncated GGUF array header")
        check_bad(
            struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_UINT32, 2) + b"\x00" * 7,
            "truncated GGUF scalar array",
        )
        check_bad(
            struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<IIQ", GGUF_TYPE_ARRAY, 99, 1),
            "unsupported GGUF array element type",
        )
        check_bad(
            struct.pack("<4sIQQ", b"GGUF", 3, 2, 1) + _pack_str("k") + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING, 10**18),
            "truncated GGUF string length",
        )

        # Duplicate KV key (smuggling two adapter.lora.alpha values!)
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta + [("adapter.lora.alpha", GGUF_TYPE_FLOAT32, struct.pack("<f", 32.0))],
                tensor_entries=[one_a, one_b],
            ),
            "duplicate GGUF metadata key",
        )

        # Required metadata checks: general.type, adapter.type, alpha missing/wrong-type/0/-1/NaN/Inf, alignment
        check_bad(
            _build_gguf_raw(meta_entries=[base_meta[0], base_meta[1]], tensor_entries=[one_a, one_b]),
            "general.type",
        )
        check_bad(
            _build_gguf_raw(meta_entries=[base_meta[0], base_meta[2]], tensor_entries=[one_a, one_b]),
            "adapter.type",
        )
        check_bad(
            _build_gguf_raw(meta_entries=[base_meta[1], base_meta[2]], tensor_entries=[one_a, one_b]),
            "adapter.lora.alpha must be present",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=[("adapter.lora.alpha", GGUF_TYPE_UINT32, struct.pack("<I", 1)), base_meta[1], base_meta[2]],
                tensor_entries=[one_a, one_b],
            ),
            "adapter.lora.alpha must be present as FLOAT32",
        )
        for bad_alpha in (0.0, -0.0, -1.0, float("nan"), float("inf")):
            check_bad(
                _build_gguf_raw(
                    meta_entries=[("adapter.lora.alpha", GGUF_TYPE_FLOAT32, struct.pack("<f", bad_alpha)), base_meta[1], base_meta[2]],
                    tensor_entries=[one_a, one_b],
                ),
                "finite and strictly positive",
            )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta + [("general.alignment", GGUF_TYPE_STRING, _pack_str("32"))],
                tensor_entries=[one_a, one_b],
            ),
            "general.alignment must be UINT32",
        )
        for bad_align in (0, 3, 48):
            check_bad(
                _build_gguf_raw(
                    meta_entries=base_meta + [("general.alignment", GGUF_TYPE_UINT32, struct.pack("<I", bad_align))],
                    tensor_entries=[one_a, one_b],
                ),
                "invalid GGUF alignment",
            )

        # Tensor descriptor checks: duplicate name, truncated ndim/desc, ndim!=2, zero dim, unsupported ggml_type,
        # empty base name, unexpected non-LoRA tensor, unpaired lora_a/lora_b, rank mismatch, truncated payload (F16 & F32).
        check_bad(_build_gguf_raw(meta_entries=base_meta, tensor_entries=[one_a, one_a]), "duplicate GGUF tensor name")
        meta_hdr = struct.pack("<4sIQQ", b"GGUF", 3, 1, len(base_meta)) + b"".join(
            _pack_str(k) + struct.pack("<I", vt) + vb for k, vt, vb in base_meta
        )
        check_bad(meta_hdr + _pack_str("blk.0.w.lora_a") + b"\x00\x00\x00", "truncated GGUF tensor ndim")
        check_bad(meta_hdr + _pack_str("blk.0.w.lora_a") + struct.pack("<I", 2) + b"\x00" * 27, "truncated GGUF tensor descriptor")
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[("blk.0.w.lora_a", (1,), GGML_TYPE_F16, b"\x00\x00", None), one_b],
            ),
            "expected 2D",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[("blk.0.w.lora_a", (0, 1), GGML_TYPE_F16, b"", None), one_b],
            ),
            "zero-sized",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[("blk.0.w.lora_a", (1, 0), GGML_TYPE_F16, b"", None), one_b],
            ),
            "zero-sized",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[("blk.0.w.lora_a", (1, 1), 2, b"\x00\x00", None), one_b],
            ),
            "unsupported ggml_type",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[(".lora_a", (1, 1), GGML_TYPE_F16, b"\x00\x00", None), one_b],
            ),
            "empty base name for .lora_a",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[one_a, (".lora_b", (1, 1), GGML_TYPE_F16, b"\x00\x00", None)],
            ),
            "empty base name for .lora_b",
        )
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[one_a, one_b, ("output.weight", (1, 1), GGML_TYPE_F16, b"\x00\x00", None)],
            ),
            "unexpected non-LoRA tensor",
        )
        check_bad(_build_gguf_raw(meta_entries=base_meta, tensor_entries=[one_a]), "unpaired")
        check_bad(
            _build_gguf_raw(
                meta_entries=base_meta,
                tensor_entries=[
                    ("blk.0.w.lora_a", (1, 2), GGML_TYPE_F16, struct.pack("<2e", 1.0, 0.0), None),
                    ("blk.0.w.lora_b", (2, 2), GGML_TYPE_F16, struct.pack("<4e", 1.0, 0.0, 0.0, 1.0), None),
                ],
            ),
            "LoRA rank mismatch",
        )
        # Truncated tensor payloads (exact 1-byte truncation for F16 and F32)
        valid_f16 = _build_lora_gguf({"blk.0.w": ([[1.0]], [[1.0]])}, alpha=1.0, dtype=GGML_TYPE_F16)
        check_bad(valid_f16[:-1], "truncated F16 tensor payload")
        valid_f32 = _build_lora_gguf({"blk.0.w": ([[1.0]], [[1.0]])}, alpha=1.0, dtype=GGML_TYPE_F32)
        check_bad(valid_f32[:-1], "truncated F32 tensor payload")

        # Structural GGUF smuggling checks (R2): exact alignment boundary pass, trailing zero-pad boundary pass,
        # overlapping offsets, unaligned offsets, unreferenced gaps (leading & interior), non-zero header/inter/trailing
        # padding, and extra trailing bytes beyond alignment padding.
        # 1. Exact alignment-multiple tensor size (4x4 F16 = 32 bytes == alignment=32 -> rel_start == prev_end_rel) PASSES!
        eye4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        exact_aligned_raw = _build_lora_gguf({"blk.0.w": (eye4, eye4)}, alpha=4.0, dtype=GGML_TYPE_F16, alignment=32)
        verify_adapter_bytes_against_certificate(
            exact_aligned_raw,
            _make_cert(exact_aligned_raw, frob_scaled=4 * radix, spec_scaled=radix, radix_scale=radix),
        )
        # 2. Exact trailing zero-padding up to alignment boundary PASSES; 1 byte beyond alignment FAILS; non-zero trailing FAILS!
        pad_tail = _align_up(len(valid_f16), 32) - len(valid_f16)
        self.assertGreater(pad_tail, 0)
        padded_ok = valid_f16 + (b"\x00" * pad_tail)
        verify_adapter_bytes_against_certificate(
            padded_ok, _make_cert(padded_ok, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)
        )
        check_bad(padded_ok + b"\x00", "extra trailing bytes")
        check_bad(valid_f16 + b"\x01", "non-zero trailing bytes")
        check_bad(valid_f16 + b"SMUGGLED" * 8, "extra trailing bytes")

        # 3. Overlapping tensor byte ranges (both identical offset 0 and partial overlap [0,64) vs [32,34))
        overlap_same = _build_gguf_raw(
            meta_entries=base_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 0),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 0),
            ],
        )
        check_bad(overlap_same, "overlapping GGUF tensor byte ranges")
        big_a = struct.pack("<32e", *([1.0] * 32))  # 64 bytes -> [0, 64)
        overlap_partial = _build_gguf_raw(
            meta_entries=base_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 32), GGML_TYPE_F16, big_a, 0),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 32),
            ],
        )
        check_bad(overlap_partial, "overlapping GGUF tensor byte ranges")

        # 4. Unaligned tensor offset (rel_off=2 with alignment=32)
        unaligned_raw = _build_gguf_raw(
            meta_entries=base_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 0),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 2),
            ],
        )
        check_bad(unaligned_raw, "unaligned GGUF tensor offset")

        # 5. Unreferenced payload gaps (both leading gap rel_off=32 and interior gap rel_off=64)
        gap_interior = _build_gguf_raw(
            meta_entries=base_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 0),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 64),
            ],
        )
        check_bad(gap_interior, "unreferenced payload gap")
        gap_leading = _build_gguf_raw(
            meta_entries=base_meta,
            tensor_entries=[
                ("blk.0.w.lora_a", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 32),
                ("blk.0.w.lora_b", (1, 1), GGML_TYPE_F16, struct.pack("<e", 1.0), 64),
            ],
        )
        check_bad(gap_leading, "unreferenced payload gap")

        # 6. Non-zero header padding (undeclared hidden bytes before data_start) & non-zero inter-tensor padding
        # Compute exact header length and data_start of valid_f16 (alignment=32, last tensor is 2 bytes at end).
        hdr_pad_test = bytearray(valid_f16)
        data_start_est = len(valid_f16) - 34  # lora_a(2) + inter_pad(30) + lora_b(2) = 34 bytes
        # Corrupt 1 byte inside inter-tensor padding [data_start + 2 : data_start + 32]
        inter_corrupt = bytearray(valid_f16)
        inter_corrupt[data_start_est + 5] = 0xAA
        check_bad(bytes(inter_corrupt), "non-zero inter-tensor padding")
        # Corrupt 1 byte inside header padding [pos : data_start] (ensure valid_f16 has non-empty header pad with alignment=64)
        valid_a64 = _build_lora_gguf({"blk.0.w": ([[1.0]], [[1.0]])}, alpha=1.0, dtype=GGML_TYPE_F16, alignment=64)
        data_start_64 = len(valid_a64) - 66  # lora_a(2) + inter_pad(62) + lora_b(2) = 66 bytes
        hdr_corrupt = bytearray(valid_a64)
        hdr_corrupt[data_start_64 - 1] = 0xFF
        check_bad(bytes(hdr_corrupt), "non-zero header padding")

    def test_zone_b_type_safety_toctou_digest_and_file_errors(self) -> None:
        """Cover strict type checks, non-ASCII/newline/mismatched SHA-256 guards, and file I/O errors."""
        radix = 1 << 24
        raw = _build_lora_gguf({"blk.0.w": ([[1.0]], [[1.0]])}, alpha=1.0)
        sha = hashlib.sha256(raw).hexdigest()
        cert = _make_cert(raw, frob_scaled=radix, spec_scaled=radix, radix_scale=radix)

        class EvilBytes(bytes):
            pass

        class EvilStr(str):
            pass

        class EvilCert(FixedRadixCertificate):
            pass

        # Non-exact bytes / certificate types
        with self.assertRaisesRegex(CertificateError, "exact bytes"):
            verify_adapter_bytes_against_certificate(bytearray(raw), cert)  # type: ignore[arg-type]
        with self.assertRaisesRegex(CertificateError, "exact bytes"):
            verify_adapter_bytes_against_certificate(EvilBytes(raw), cert)
        evil_cert = EvilCert(**cert.to_dict())
        with self.assertRaisesRegex(CertificateError, "exact FixedRadixCertificate"):
            verify_adapter_bytes_against_certificate(raw, evil_cert)

        # Certificate digest mismatch vs file bytes (TOCTOU swap guard)
        wrong_sha = hashlib.sha256(b"other-bytes").hexdigest()
        wrong_cert = _make_cert(
            raw, frob_scaled=radix, spec_scaled=radix, radix_scale=radix, adapter_sha256_override=wrong_sha
        )
        with self.assertRaisesRegex(CertificateError, "certificate adapter_sha256"):
            verify_adapter_bytes_against_certificate(raw, wrong_cert)

        # Expected digest mismatch + non-ASCII / surrogate / trailing-newline / EvilStr before hmac.compare_digest
        with self.assertRaisesRegex(CertificateError, "expected adapter_sha256"):
            verify_adapter_bytes_against_certificate(raw, cert, expected_adapter_sha256=wrong_sha)
        for bad_sha in ("\u00e9" * 64, chr(0xD800) * 64, sha + "\n", sha.upper(), EvilStr(sha), 123):
            with self.assertRaisesRegex(CertificateError, "64-char lowercase ASCII hex"):
                verify_adapter_bytes_against_certificate(raw, cert, expected_adapter_sha256=bad_sha)  # type: ignore[arg-type]

        # Valid expected_adapter_sha256 passes cleanly via file API
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            good_path = tmp / "adapter.gguf"
            good_path.write_bytes(raw)
            rep = verify_adapter_file_against_certificate(good_path, cert, expected_adapter_sha256=sha)
            self.assertEqual(rep.adapter_sha256, sha)

            # Missing file, directory path, NUL bytes, unpaired UTF-16 surrogates, bytes-PathLike, empty & invalid path types fail closed.
            class BytesPath:
                def __fspath__(self) -> bytes:
                    return b"/tmp/bytes-path.gguf"

            with self.assertRaisesRegex(CertificateError, "failed to read adapter file"):
                verify_adapter_file_against_certificate(tmp / "missing.gguf", cert)
            with self.assertRaisesRegex(CertificateError, "failed to read adapter file"):
                verify_adapter_file_against_certificate(tmp, cert)
            with self.assertRaisesRegex(CertificateError, "failed to read adapter file"):
                verify_adapter_file_against_certificate(chr(0xD800), cert)
            with self.assertRaisesRegex(CertificateError, "without NUL bytes"):
                verify_adapter_file_against_certificate("/tmp/a\x00b", cert)
            with self.assertRaisesRegex(CertificateError, "without NUL bytes"):
                verify_adapter_file_against_certificate("   ", cert)
            with self.assertRaisesRegex(CertificateError, "without NUL bytes"):
                verify_adapter_file_against_certificate(BytesPath(), cert)  # type: ignore[arg-type]
            with self.assertRaisesRegex(CertificateError, "adapter_path must be a string or path"):
                verify_adapter_file_against_certificate(123, cert)  # type: ignore[arg-type]

            # End-to-end mount_stack with surrogate or NUL adapter_path fails closed with signed reject ledger receipt (W13-BUG-01).
            mock = MockLlamaServer()
            url = mock.start()
            try:
                ckpt = make_checkpoint()
                lineage = make_lineage()
                flags = make_runtime_flags()
                bounds = CertificateBounds(
                    max_frobenius_sq=RationalBound(20, 1),
                    max_spectral_norm_sq=RationalBound(20, 1),
                    max_leakage_energy=RationalBound(1, 1),
                    max_safety_dot_sq=RationalBound(1, 1),
                )
                for idx, hostile_path in enumerate((chr(0xD800), "/tmp/a\x00b")):
                    signer = HmacSha256Signer(b"w13-bug01-key-32-bytes-000000001", key_id="warp-w13")
                    ledger_path = tmp / f"ledger_w13_{idx}.jsonl"
                    modeld = WarpModeld(
                        pinned_checkpoint=ckpt,
                        pinned_lineage=lineage,
                        live_runtime_flags=flags,
                        single_bounds=bounds,
                        joint_bounds=make_joint_bounds(),
                        canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
                        llama_client=LlamaServerClient(url),
                        ledger=SignedLedger(ledger_path, signer),
                    )
                    bundle_dict = _make_mount_bundle(good_path, raw, cert, ckpt=ckpt, lineage=lineage, flags=flags)
                    bundle_dict["adapter_path"] = hostile_path
                    with self.assertRaises(CheckpointMismatchError):
                        modeld.mount_stack([bundle_dict])
                    self.assertEqual(mock.state.request_log, [])
                    receipts = LedgerVerifier.verify_file(ledger_path, signer)
                    self.assertEqual(len(receipts), 1)
                    self.assertEqual(receipts[0].event, "reject")
            finally:
                mock.stop()


if __name__ == "__main__":
    unittest.main()

"""Stage 2 exact integer GGUF v3 LoRA adapter-to-certificate binding verifier.

Reads a GGUF v3 LoRA binary once, checks its SHA-256 against the certificate
and bundle digest in constant time, parses its metadata and tensor payloads
strictly, enforces unit LoRA scaling (`adapter.lora.alpha == rank` for every
layer), and verifies the certificate's `frobenius_sq_scaled` and
`spectral_norm_sq_scaled` bounds directly against the stored tensor weights
using exact integer arithmetic with zero floating-point operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import operator
import os
from pathlib import Path
import re
import struct
import time
from typing import Any

from .certificate import FixedRadixCertificate
from .errors import CertificateError

_SHA256_FULL_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_DEFAULT_ALIGNMENT = 32
_MAX_ARRAY_DEPTH = 8

_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_SCALAR_WIDTHS: dict[int, int] = {
    _GGUF_TYPE_UINT8: 1,
    _GGUF_TYPE_INT8: 1,
    _GGUF_TYPE_UINT16: 2,
    _GGUF_TYPE_INT16: 2,
    _GGUF_TYPE_UINT32: 4,
    _GGUF_TYPE_INT32: 4,
    _GGUF_TYPE_FLOAT32: 4,
    _GGUF_TYPE_BOOL: 1,
    _GGUF_TYPE_UINT64: 8,
    _GGUF_TYPE_INT64: 8,
    _GGUF_TYPE_FLOAT64: 8,
}

_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1

_F16_SHIFT = 24
_F32_SHIFT = 149
_MAX_FROB_SHIFT = 4 * _F32_SHIFT


def _decode_f16_bits(bits: int) -> int | None:
    """Decode IEEE-754 binary16 bits to an exact integer multiple of 2^-24, or None if non-finite."""
    exp = (bits >> 10) & 0x1F
    if exp == 0x1F:
        return None
    frac = bits & 0x3FF
    mag = frac if exp == 0 else ((0x400 | frac) << (exp - 1))
    return -mag if (bits & 0x8000) else mag


def _decode_f32_bits(bits: int) -> int | None:
    """Decode IEEE-754 binary32 bits to an exact integer multiple of 2^-149, or None if non-finite."""
    exp = (bits >> 23) & 0xFF
    if exp == 0xFF:
        return None
    frac = bits & 0x7FFFFF
    mag = frac if exp == 0 else ((0x800000 | frac) << (exp - 1))
    return -mag if (bits & 0x80000000) else mag


_F16_EXACT_TABLE: tuple[int | None, ...] = tuple(_decode_f16_bits(bits) for bits in range(65536))


@dataclass(frozen=True)
class AdapterFileCertificateReport:
    """Exact verification witness binding a GGUF v3 LoRA binary to a FixedRadixCertificate."""

    adapter_sha256: str
    layer_count: int
    lora_rank: int
    frobenius_sq_num: int
    frobenius_sq_shift: int
    elapsed_ns: int


def _require_sha256_hex(value: Any, name: str) -> str:
    if type(value) is not str or not _SHA256_FULL_RE.match(value):
        raise CertificateError(f"{name} must be a 64-char lowercase ASCII hex digest")
    return value


def _align_up(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _read_exact(buf: memoryview, pos: int, n: int, label: str) -> tuple[memoryview, int]:
    end = pos + n
    if end > len(buf):
        raise CertificateError(f"truncated {label}")
    return buf[pos:end], end


def _read_gguf_string(buf: memoryview, pos: int) -> tuple[str, int]:
    len_slice, pos = _read_exact(buf, pos, 8, "GGUF string length")
    (slen,) = struct.unpack("<Q", len_slice)
    raw_slice, pos = _read_exact(buf, pos, slen, "GGUF string payload")
    try:
        text = bytes(raw_slice).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CertificateError(f"invalid UTF-8 in GGUF string: {exc}") from exc
    return text, pos


def _read_gguf_value(buf: memoryview, pos: int, vtype: int, depth: int) -> tuple[Any, int]:
    if vtype == _GGUF_TYPE_STRING:
        return _read_gguf_string(buf, pos)
    if vtype == _GGUF_TYPE_ARRAY:
        if depth >= _MAX_ARRAY_DEPTH:
            raise CertificateError("GGUF metadata array nesting depth exceeded")
        hdr_slice, pos = _read_exact(buf, pos, 12, "GGUF array header")
        elem_type, count = struct.unpack("<IQ", hdr_slice)
        elem_width = _GGUF_SCALAR_WIDTHS.get(elem_type)
        if elem_width is not None:
            _, pos = _read_exact(buf, pos, count * elem_width, "GGUF scalar array payload")
            return None, pos
        if elem_type in (_GGUF_TYPE_STRING, _GGUF_TYPE_ARRAY):
            for _ in range(count):
                _, pos = _read_gguf_value(buf, pos, elem_type, depth + 1)
            return None, pos
        raise CertificateError(f"unsupported GGUF array element type: {elem_type}")
    width = _GGUF_SCALAR_WIDTHS.get(vtype)
    if width is None:
        raise CertificateError(f"unsupported GGUF metadata value type: {vtype}")
    val_slice, pos = _read_exact(buf, pos, width, "GGUF metadata scalar value")
    return bytes(val_slice), pos


def _decode_tensor_values(
    buf: memoryview,
    abs_off: int,
    rows: int,
    cols: int,
    ggml_type: int,
    tensor_name: str,
) -> tuple[list[int], int]:
    count = rows * cols
    if ggml_type == _GGML_TYPE_F16:
        payload, _ = _read_exact(buf, abs_off, count * 2, f"F16 tensor payload for {tensor_name!r}")
        raw_words = struct.unpack(f"<{count}H", payload)
        decoded = list(map(_F16_EXACT_TABLE.__getitem__, raw_words))
        if None in decoded:
            raise CertificateError(f"non-finite F16 weight in GGUF tensor {tensor_name!r}")
        return decoded, _F16_SHIFT  # type: ignore[return-value]
    payload, _ = _read_exact(buf, abs_off, count * 4, f"F32 tensor payload for {tensor_name!r}")
    raw_words = struct.unpack(f"<{count}I", payload)
    decoded_f32: list[int] = []
    for word in raw_words:
        val = _decode_f32_bits(word)
        if val is None:
            raise CertificateError(f"non-finite F32 weight in GGUF tensor {tensor_name!r}")
        decoded_f32.append(val)
    return decoded_f32, _F32_SHIFT


def _gram_from_rows(flat: list[int], rows: int, cols: int) -> list[list[int]]:
    row_vecs = [flat[i * cols : (i + 1) * cols] for i in range(rows)]
    gram = [[0] * rows for _ in range(rows)]
    for i in range(rows):
        ri = row_vecs[i]
        for j in range(i, rows):
            dot = sum(map(operator.mul, ri, row_vecs[j]))
            gram[i][j] = dot
            gram[j][i] = dot
    return gram


def _gram_from_cols(flat: list[int], rows: int, cols: int) -> list[list[int]]:
    col_vecs = [flat[c::cols] for c in range(cols)]
    gram = [[0] * cols for _ in range(cols)]
    for i in range(cols):
        ci = col_vecs[i]
        for j in range(i, cols):
            dot = sum(map(operator.mul, ci, col_vecs[j]))
            gram[i][j] = dot
            gram[j][i] = dot
    return gram


def _matmul_int(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    n = len(left)
    r_cols = [[right[k][j] for k in range(n)] for j in range(n)]
    return [[sum(map(operator.mul, left[i], r_cols[j])) for j in range(n)] for i in range(n)]


def _is_symmetric_psd_bareiss(matrix: list[list[int]]) -> bool:
    """Return True iff symmetric integer matrix is positive semidefinite (exact Bareiss).

    Uses symmetric diagonal pivoting + Sylvester's determinant identity so every
    division by `prev` is exact in Z with zero floating-point operations.
    """
    n = len(matrix)
    s = [list(row) for row in matrix]
    prev = 1
    for k in range(n):
        pivot_idx = -1
        for i in range(k, n):
            diag = s[i][i]
            if diag < 0:
                return False
            if diag > 0:
                pivot_idx = i
        if pivot_idx < 0:
            for i in range(k, n):
                for j in range(i + 1, n):
                    if s[i][j] != 0:
                        return False
            return True
        s[k], s[pivot_idx] = s[pivot_idx], s[k]
        for row in s:
            row[k], row[pivot_idx] = row[pivot_idx], row[k]
        piv = s[k][k]
        for i in range(k + 1, n):
            sik = s[i][k]
            for j in range(i, n):
                val = (piv * s[i][j] - sik * s[j][k]) // prev
                s[i][j] = val
                s[j][i] = val
        prev = piv
    return True


def verify_adapter_bytes_against_certificate(
    raw_bytes: bytes,
    certificate: FixedRadixCertificate,
    *,
    expected_adapter_sha256: str | None = None,
) -> AdapterFileCertificateReport:
    """Verify GGUF v3 LoRA bytes directly against a FixedRadixCertificate using exact integer math.

    Checks enforced fail-closed:
      1. Constant-time SHA-256 equality with `certificate.adapter_sha256` and
         (if provided) `expected_adapter_sha256`.
      2. Strict GGUF v3 LoRA framing, metadata (`general.type == "adapter"`,
         `adapter.type == "lora"`, finite positive float32 `adapter.lora.alpha`),
         valid power-of-two alignment, unique keys/tensors, 2D F16/F32 tensors,
         and strictly paired `<base>.lora_a` / `<base>.lora_b` layers.
      3. Exact `adapter.lora.alpha == rank` for every layer so `llama.cpp`
         applies unit scaling (`alpha / rank == 1`) and cannot amplify any of
         the four certified metrics (`frobenius_sq`, `spectral_norm_sq`,
         `leakage_energy`, `max_safety_dot_sq`).
      4. Exact squared spectral norm `sigma_max(B_l A_l)^2 <= spectral_norm_sq_scaled / radix_scale`
         for every layer via exact integer Bareiss PSD test on
         `tau * G_A - G_A G_B G_A >= 0`.
      5. Exact total squared Frobenius norm `sum_l ||B_l A_l||_F^2 <= frobenius_sq_scaled / radix_scale`
         via exact integer Gram trace `Tr(G_B G_A)`.
    """
    start_ns = time.perf_counter_ns()
    if type(raw_bytes) is not bytes:
        raise CertificateError(f"adapter payload must be exact bytes (got {type(raw_bytes).__name__})")
    if type(certificate) is not FixedRadixCertificate:
        raise CertificateError(
            f"certificate must be an exact FixedRadixCertificate (got {type(certificate).__name__})"
        )

    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    cert_sha = _require_sha256_hex(certificate.adapter_sha256, "certificate.adapter_sha256")
    if not hmac.compare_digest(actual_sha, cert_sha):
        raise CertificateError(
            f"adapter file digest ({actual_sha}) != certificate adapter_sha256 ({cert_sha})"
        )
    if expected_adapter_sha256 is not None:
        exp_sha = _require_sha256_hex(expected_adapter_sha256, "expected_adapter_sha256")
        if not hmac.compare_digest(actual_sha, exp_sha):
            raise CertificateError(
                f"adapter file digest ({actual_sha}) != expected adapter_sha256 ({exp_sha})"
            )

    buf = memoryview(raw_bytes)
    hdr_slice, pos = _read_exact(buf, 0, 24, "GGUF header")
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", hdr_slice)
    if magic != _GGUF_MAGIC:
        raise CertificateError(f"invalid GGUF magic: {magic!r}")
    if version != _GGUF_VERSION:
        raise CertificateError(f"unsupported GGUF version: {version}")
    if n_tensors == 0:
        raise CertificateError("GGUF adapter contains no tensors")

    meta: dict[str, tuple[int, Any]] = {}
    for _ in range(n_kv):
        key, pos = _read_gguf_string(buf, pos)
        if key in meta:
            raise CertificateError(f"duplicate GGUF metadata key: {key!r}")
        vt_slice, pos = _read_exact(buf, pos, 4, f"GGUF metadata value type for {key!r}")
        (vtype,) = struct.unpack("<I", vt_slice)
        val, pos = _read_gguf_value(buf, pos, vtype, 0)
        meta[key] = (vtype, val)

    if meta.get("general.type") != (_GGUF_TYPE_STRING, "adapter"):
        raise CertificateError("GGUF general.type must be 'adapter'")
    if meta.get("adapter.type") != (_GGUF_TYPE_STRING, "lora"):
        raise CertificateError("GGUF adapter.type must be 'lora'")

    alpha_entry = meta.get("adapter.lora.alpha")
    if alpha_entry is None or alpha_entry[0] != _GGUF_TYPE_FLOAT32:
        raise CertificateError("GGUF adapter.lora.alpha must be present as FLOAT32")
    (alpha_bits,) = struct.unpack("<I", alpha_entry[1])
    alpha_exact = _decode_f32_bits(alpha_bits)
    if alpha_exact is None or alpha_exact <= 0:
        raise CertificateError("GGUF adapter.lora.alpha must be finite and strictly positive")

    if "general.alignment" in meta:
        align_type, align_raw = meta["general.alignment"]
        if align_type != _GGUF_TYPE_UINT32:
            raise CertificateError("GGUF general.alignment must be UINT32")
        (alignment,) = struct.unpack("<I", align_raw)
        if alignment == 0 or (alignment & (alignment - 1)) != 0:
            raise CertificateError(f"invalid GGUF alignment: {alignment}")
    else:
        alignment = _GGUF_DEFAULT_ALIGNMENT

    seen_tensors: set[str] = set()
    lora_a_specs: dict[str, tuple[int, int, int, int]] = {}
    lora_b_specs: dict[str, tuple[int, int, int, int]] = {}
    tensor_spans: list[tuple[int, int, str]] = []
    for _ in range(n_tensors):
        tname, pos = _read_gguf_string(buf, pos)
        if tname in seen_tensors:
            raise CertificateError(f"duplicate GGUF tensor name: {tname!r}")
        seen_tensors.add(tname)
        ndim_slice, pos = _read_exact(buf, pos, 4, f"GGUF tensor ndim for {tname!r}")
        (ndim,) = struct.unpack("<I", ndim_slice)
        if ndim != 2:
            raise CertificateError(f"expected 2D LoRA tensor for {tname!r}, got ndim={ndim}")
        desc_slice, pos = _read_exact(buf, pos, 28, f"GGUF tensor descriptor for {tname!r}")
        dim0, dim1, ggml_type, rel_off = struct.unpack("<QQIQ", desc_slice)
        if dim0 == 0 or dim1 == 0:
            raise CertificateError(f"zero-sized GGUF tensor dimension for {tname!r}")
        if ggml_type not in (_GGML_TYPE_F16, _GGML_TYPE_F32):
            raise CertificateError(
                f"unsupported ggml_type {ggml_type} for {tname!r}; expected F16 or F32"
            )
        rows, cols = int(dim1), int(dim0)
        elem_width = 2 if ggml_type == _GGML_TYPE_F16 else 4
        tensor_spans.append((int(rel_off), int(rel_off) + rows * cols * elem_width, tname))
        if tname.endswith(".lora_a"):
            base = tname[:-7]
            if not base:
                raise CertificateError("empty base name for .lora_a tensor")
            lora_a_specs[base] = (rows, cols, int(ggml_type), int(rel_off))
        elif tname.endswith(".lora_b"):
            base = tname[:-7]
            if not base:
                raise CertificateError("empty base name for .lora_b tensor")
            lora_b_specs[base] = (rows, cols, int(ggml_type), int(rel_off))
        else:
            raise CertificateError(f"unexpected non-LoRA tensor in GGUF adapter: {tname!r}")

    if lora_a_specs.keys() != lora_b_specs.keys():
        raise CertificateError("unpaired .lora_a/.lora_b tensor(s) in GGUF adapter")

    data_start = _align_up(pos, alignment)
    if any(buf[pos:data_start]):
        raise CertificateError("non-zero header padding before GGUF tensor data")
    tensor_spans.sort(key=lambda item: (item[0], item[1]))
    expected_rel = 0
    prev_end_rel = 0
    for rel_start, rel_end, tname in tensor_spans:
        if rel_start % alignment != 0:
            raise CertificateError(f"unaligned GGUF tensor offset {rel_start} for {tname!r}")
        if rel_start < prev_end_rel:
            raise CertificateError(f"overlapping GGUF tensor byte ranges at {tname!r}")
        if rel_start > expected_rel:
            raise CertificateError(f"unreferenced payload gap before GGUF tensor {tname!r}")
        if any(buf[data_start + prev_end_rel : data_start + rel_start]):
            raise CertificateError(f"non-zero inter-tensor padding before GGUF tensor {tname!r}")
        prev_end_rel = rel_end
        expected_rel = _align_up(rel_end, alignment)

    total_frob_num = 0
    layer_rank = 0
    spec_num = certificate.spectral_norm_sq_scaled
    spec_den = certificate.radix_scale

    for base in sorted(lora_a_specs):
        a_rows, a_cols, a_type, a_rel = lora_a_specs[base]
        b_rows, b_cols, b_type, b_rel = lora_b_specs[base]
        if b_cols != a_rows:
            raise CertificateError(
                f"LoRA rank mismatch for {base!r}: B cols ({b_cols}) != A rows ({a_rows})"
            )
        rank = a_rows
        if alpha_exact != (rank << _F32_SHIFT):
            raise CertificateError(
                f"GGUF adapter.lora.alpha does not equal rank {rank} for {base!r}; "
                "non-unit alpha/rank scaling is rejected"
            )
        layer_rank = rank

        a_flat, shift_a = _decode_tensor_values(
            buf, data_start + a_rel, a_rows, a_cols, a_type, f"{base}.lora_a"
        )
        b_flat, shift_b = _decode_tensor_values(
            buf, data_start + b_rel, b_rows, b_cols, b_type, f"{base}.lora_b"
        )

        gram_a = _gram_from_rows(a_flat, a_rows, a_cols)
        gram_b = _gram_from_cols(b_flat, b_rows, b_cols)

        layer_frob = sum(
            gram_b[i][j] * gram_a[i][j]
            for i in range(rank)
            for j in range(rank)
        )
        layer_shift = 2 * (shift_b + shift_a)
        total_frob_num += layer_frob << (_MAX_FROB_SHIFT - layer_shift)

        if spec_num == 0:
            if layer_frob > 0:
                raise CertificateError(
                    f"GGUF layer {base!r} is non-zero but certificate spectral_norm_sq_scaled is 0"
                )
        else:
            ga_gb = _matmul_int(gram_a, gram_b)
            ga_gb_ga = _matmul_int(ga_gb, gram_a)
            tau_scale = spec_num << layer_shift
            psd_mat = [
                [tau_scale * gram_a[i][j] - spec_den * ga_gb_ga[i][j] for j in range(rank)]
                for i in range(rank)
            ]
            if not _is_symmetric_psd_bareiss(psd_mat):
                raise CertificateError(
                    f"GGUF layer {base!r} squared spectral norm exceeds certificate spectral_norm_sq_scaled"
                )

    payload_end = data_start + prev_end_rel
    max_aligned_end = _align_up(payload_end, alignment)
    if len(buf) > max_aligned_end:
        raise CertificateError("extra trailing bytes beyond GGUF tensor payload")
    if any(buf[payload_end:]):
        raise CertificateError("non-zero trailing bytes beyond GGUF tensor payload")

    if total_frob_num * certificate.radix_scale > (certificate.frobenius_sq_scaled << _MAX_FROB_SHIFT):
        raise CertificateError(
            f"GGUF adapter squared Frobenius norm exceeds certificate frobenius_sq_scaled "
            f"for {certificate.adapter_id}"
        )

    elapsed_ns = time.perf_counter_ns() - start_ns
    return AdapterFileCertificateReport(
        adapter_sha256=actual_sha,
        layer_count=len(lora_a_specs),
        lora_rank=layer_rank,
        frobenius_sq_num=total_frob_num,
        frobenius_sq_shift=_MAX_FROB_SHIFT,
        elapsed_ns=elapsed_ns,
    )


def verify_adapter_file_against_certificate(
    adapter_path: Path | str,
    certificate: FixedRadixCertificate,
    *,
    expected_adapter_sha256: str | None = None,
) -> AdapterFileCertificateReport:
    """Read a GGUF v3 LoRA file once and verify it against `certificate` in exact integer math."""
    if not isinstance(adapter_path, (str, os.PathLike)):
        raise CertificateError(
            f"adapter_path must be a string or path (got {type(adapter_path).__name__})"
        )
    path_str = os.fspath(adapter_path)
    if not isinstance(path_str, str) or not path_str.strip() or "\x00" in path_str:
        raise CertificateError("adapter_path must be a non-empty string without NUL bytes")
    try:
        path_str.encode("utf-8")
        raw_bytes = Path(path_str).read_bytes()
    except (OSError, UnicodeEncodeError, ValueError) as exc:
        raise CertificateError(f"failed to read adapter file: {exc}") from exc
    return verify_adapter_bytes_against_certificate(
        raw_bytes,
        certificate,
        expected_adapter_sha256=expected_adapter_sha256,
    )

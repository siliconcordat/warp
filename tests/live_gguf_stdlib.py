"""Pure-stdlib GGUF v3 geometry reader, deterministic F16 LoRA writer, and exact certificate builder.

Replaces the optional ``numpy``/``weft`` imports previously required by the live
Stage 1-4 ``llama-server`` smoke test so ``warp-modeld`` and its entire offline
and live test suite remain strictly zero-dependency Python 3 standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path
import random
import struct
from typing import Any, BinaryIO, Mapping, Sequence

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
_MAX_GGUF_HEADER_COUNT = 100_000
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

_GGUF_SCALAR_FMTS: dict[int, tuple[str, int]] = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8: ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16: ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32: ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL: ("<?", 1),
    _GGUF_TYPE_UINT64: ("<Q", 8),
    _GGUF_TYPE_INT64: ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}

GGML_TYPE_F16 = 1
_F16_SHIFT = 24
_FROB_SHIFT = 4 * _F16_SHIFT

_RESERVED_LORA_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "adapter.lora.alpha",
        "adapter.type",
        "general.alignment",
        "general.architecture",
        "general.type",
        "weft.schema",
    }
)


def _require_finite_float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite numeric value")
    try:
        num = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} overflows float: {exc}") from exc
    if not math.isfinite(num):
        raise ValueError(f"{label} must be a finite numeric value")
    return num


def _validate_cli_path(path: Path | str, *, label: str) -> str:
    if type(path) is str:
        raw = path
    elif isinstance(path, Path):
        raw = str(path)
    else:
        raise ValueError(f"{label} must be a str or Path")
    if not raw.strip():
        raise ValueError(f"{label} must be non-empty")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8: {exc}") from exc
    if "\x00" in raw:
        raise ValueError(f"{label} must not contain NUL bytes")
    norm = str(Path(raw))
    if norm == ".":
        raise ValueError(f"{label} must not be empty or '.'")
    return norm


def _pack_string(text: str) -> bytes:
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"invalid UTF-8 string for GGUF packing: {exc}") from exc
    return struct.pack("<Q", len(raw)) + raw


def _pack_kv(key: str, value: Any) -> bytes:
    head = _pack_string(key)
    if type(value) is bool:
        return head + struct.pack("<I?", _GGUF_TYPE_BOOL, value)
    if type(value) is int:
        if 0 <= value <= 0xFFFFFFFF:
            return head + struct.pack("<II", _GGUF_TYPE_UINT32, value)
        if 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            return head + struct.pack("<IQ", _GGUF_TYPE_UINT64, value)
        raise ValueError(f"unsupported GGUF integer metadata value for {key!r}")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite GGUF float metadata value for {key!r}")
        try:
            return head + struct.pack("<If", _GGUF_TYPE_FLOAT32, value)
        except OverflowError as exc:
            raise ValueError(f"float32 overflow in GGUF metadata value for {key!r}: {exc}") from exc
    if type(value) is str:
        return head + struct.pack("<I", _GGUF_TYPE_STRING) + _pack_string(value)
    raise TypeError(f"unsupported GGUF metadata type for {key!r}: {type(value).__name__}")


def _align_up(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _read_exact(fh: BinaryIO, nbytes: int, label: str) -> bytes:
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"truncated GGUF stream while reading {label}")
    return data


def _read_gguf_str(fh: BinaryIO, file_size: int) -> str:
    (slen,) = struct.unpack("<Q", _read_exact(fh, 8, "string length"))
    if slen > file_size - fh.tell():
        raise ValueError("GGUF string length exceeds file size")
    try:
        return _read_exact(fh, slen, "string bytes").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in GGUF string: {exc}") from exc


def _skip_or_read_val(
    fh: BinaryIO,
    vtype: int,
    *,
    keep: bool,
    file_size: int,
    depth: int = 0,
) -> Any:
    if vtype == _GGUF_TYPE_STRING:
        if keep:
            return _read_gguf_str(fh, file_size)
        (slen,) = struct.unpack("<Q", _read_exact(fh, 8, "skipped string length"))
        if slen > file_size - fh.tell():
            raise ValueError("GGUF skipped string extends past end of file")
        fh.seek(slen, 1)
        return None
    if vtype == _GGUF_TYPE_ARRAY:
        if depth >= _MAX_ARRAY_DEPTH:
            raise ValueError("GGUF metadata array nesting depth exceeded")
        elem_type, count = struct.unpack("<IQ", _read_exact(fh, 12, "array header"))
        scalar_spec = _GGUF_SCALAR_FMTS.get(elem_type)
        if scalar_spec is not None:
            skip = count * scalar_spec[1]
            if skip > file_size - fh.tell():
                raise ValueError("GGUF scalar array extends past end of file")
            fh.seek(skip, 1)
            return f"<array:{count}>" if keep else None
        if elem_type in (_GGUF_TYPE_STRING, _GGUF_TYPE_ARRAY):
            for _ in range(count):
                _skip_or_read_val(fh, elem_type, keep=False, file_size=file_size, depth=depth + 1)
            return f"<array:{count}>" if keep else None
        raise ValueError(f"unsupported GGUF array element type: {elem_type}")
    scalar_spec = _GGUF_SCALAR_FMTS.get(vtype)
    if scalar_spec is None:
        raise ValueError(f"unsupported GGUF metadata value type: {vtype}")
    fmt, width = scalar_spec
    if width > file_size - fh.tell():
        raise ValueError("GGUF scalar metadata value extends past end of file")
    if keep:
        return struct.unpack(fmt, _read_exact(fh, width, "scalar value"))[0]
    fh.seek(width, 1)
    return None


def read_gguf_model_geometry(
    path: Path | str,
    *,
    tensor_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read GGUF v3 model metadata and tensor shapes without loading weight bytes."""
    wanted = set(tensor_names) if tensor_names is not None else None
    gguf_path = Path(path)
    file_size = gguf_path.stat().st_size
    with gguf_path.open("rb") as fh:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", _read_exact(fh, 24, "GGUF header"))
        if magic != GGUF_MAGIC:
            raise ValueError(f"invalid GGUF magic: {magic!r}")
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version: {version}")
        if n_kv > _MAX_GGUF_HEADER_COUNT or n_tensors > _MAX_GGUF_HEADER_COUNT:
            raise ValueError(f"GGUF header counts exceed safety cap: tensors={n_tensors}, kv={n_kv}")

        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_gguf_str(fh, file_size)
            (vtype,) = struct.unpack("<I", _read_exact(fh, 4, "metadata value type"))
            keep = not key.startswith("tokenizer.")
            val = _skip_or_read_val(fh, vtype, keep=keep, file_size=file_size)
            if keep:
                meta[key] = val

        tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            tname = _read_gguf_str(fh, file_size)
            (ndim,) = struct.unpack("<I", _read_exact(fh, 4, "tensor ndim"))
            if ndim == 0 or ndim > 8:
                raise ValueError(f"invalid GGUF tensor ndim={ndim} for {tname!r}")
            ggml_ne = struct.unpack(f"<{ndim}Q", _read_exact(fh, 8 * ndim, "tensor dims"))
            ggml_type, rel_off = struct.unpack("<IQ", _read_exact(fh, 12, "tensor type/offset"))
            if wanted is None or tname in wanted:
                row_major_shape = tuple(int(d) for d in reversed(ggml_ne))
                tensors[tname] = {
                    "ggml_ne": tuple(int(d) for d in ggml_ne),
                    "shape": row_major_shape,
                    "ggml_type": int(ggml_type),
                    "rel_offset": int(rel_off),
                }

    arch = str(meta.get("general.architecture", "unknown"))
    return {
        "architecture": arch,
        "model_name": str(meta.get("general.name", "")),
        "n_tensors": int(n_tensors),
        "block_count": int(meta.get(f"{arch}.block_count", 0)),
        "embedding_length": int(meta.get(f"{arch}.embedding_length", 0)),
        "feed_forward_length": int(meta.get(f"{arch}.feed_forward_length", 0)),
        "tensors": tensors,
    }


def _decode_f16_bits(bits: int) -> int:
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    mag = frac if exp == 0 else ((0x400 | frac) << (exp - 1))
    return -mag if (bits & 0x8000) else mag


def _pack_f16_matrix(rows: Sequence[Sequence[float]], name: str) -> tuple[tuple[int, int], bytes]:
    if not rows:
        raise ValueError(f"LoRA tensor {name!r} must have at least one row")
    cols = len(rows[0])
    if cols == 0:
        raise ValueError(f"LoRA tensor {name!r} must have at least one column")
    chunks: list[bytes] = []
    for row in rows:
        if len(row) != cols:
            raise ValueError(f"ragged row in LoRA tensor {name!r}")
        fvals = [_require_finite_float(val, f"entry in LoRA tensor {name!r}") for val in row]
        try:
            chunks.append(struct.pack(f"<{cols}e", *fvals))
        except OverflowError as exc:
            raise ValueError(f"F16 overflow in LoRA tensor {name!r}: {exc}") from exc
    return (len(rows), cols), b"".join(chunks)


def write_gguf_lora_stdlib(
    path: Path | str,
    *,
    architecture: str,
    lora_alpha: float,
    layer_factors: Mapping[str, tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]],
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    """Write a deterministic 32-byte aligned GGUF v3 F16 LoRA binary using pure stdlib."""
    if not layer_factors:
        raise ValueError("layer_factors must be non-empty")
    if type(architecture) is not str or not architecture.strip():
        raise ValueError("architecture must be a non-empty string")
    alpha_f = _require_finite_float(lora_alpha, "lora_alpha")

    tensor_blobs: list[tuple[str, tuple[int, int], bytes, int]] = []
    rel_offset = 0
    for base_name in sorted(layer_factors):
        b_mat, a_mat = layer_factors[base_name]
        a_shape, a_raw = _pack_f16_matrix(a_mat, f"{base_name}.lora_a")
        b_shape, b_raw = _pack_f16_matrix(b_mat, f"{base_name}.lora_b")
        if b_shape[1] != a_shape[0]:
            raise ValueError(f"incompatible LoRA (B, A) shapes for {base_name!r}: {b_shape} vs {a_shape}")
        tensor_blobs.append((f"{base_name}.lora_a", a_shape, a_raw, rel_offset))
        rel_offset = _align_up(rel_offset + len(a_raw))
        tensor_blobs.append((f"{base_name}.lora_b", b_shape, b_raw, rel_offset))
        rel_offset = _align_up(rel_offset + len(b_raw))

    meta: dict[str, Any] = {
        "adapter.lora.alpha": alpha_f,
        "adapter.type": "lora",
        "general.alignment": int(GGUF_DEFAULT_ALIGNMENT),
        "general.architecture": architecture,
        "general.type": "adapter",
        "weft.schema": "weft-gguf-lora/v0.1",
    }
    if extra_metadata is not None:
        if not isinstance(extra_metadata, Mapping):
            raise TypeError("extra_metadata must be a mapping or None")
        for k, v in extra_metadata.items():
            if type(k) is not str or not k:
                raise ValueError("extra_metadata keys must be non-empty strings")
            if k in _RESERVED_LORA_METADATA_KEYS:
                raise ValueError(f"reserved GGUF LoRA metadata key cannot be overridden: {k!r}")
            meta[k] = v

    sorted_keys = sorted(meta)
    parts: list[bytes] = [struct.pack("<4sIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensor_blobs), len(sorted_keys))]
    for key in sorted_keys:
        parts.append(_pack_kv(key, meta[key]))
    for name, (rows, cols), _raw, t_off in tensor_blobs:
        parts.append(_pack_string(name))
        parts.append(struct.pack("<IQQIQ", 2, cols, rows, GGML_TYPE_F16, t_off))

    header = b"".join(parts)
    data_start = _align_up(len(header))
    out_parts = [header, b"\x00" * (data_start - len(header))]
    written = 0
    for _name, _shape, raw, t_off in tensor_blobs:
        if t_off > written:
            out_parts.append(b"\x00" * (t_off - written))
            written = t_off
        out_parts.append(raw)
        written += len(raw)

    payload = b"".join(out_parts)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    return out_path, hashlib.sha256(payload).hexdigest()


def build_deterministic_f16_lora_and_cert(
    *,
    d_out: int,
    d_in: int,
    rank: int = 4,
    seed: int = 20261119,
    radix_scale_exp2: int = 24,
    projector_epoch: int = 0,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any], str]:
    """Generate a bounded deterministic F16 LoRA pair (B, A) and exact integer weft certificate."""
    if type(d_out) is not int or d_out <= 0:
        raise ValueError("d_out must be a positive int")
    if type(d_in) is not int or d_in <= 0:
        raise ValueError("d_in must be a positive int")
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be a positive int")
    if type(radix_scale_exp2) is not int or not (8 <= radix_scale_exp2 <= 48):
        raise ValueError("radix_scale_exp2 must be an int in [8, 48]")

    rng = random.Random(seed)
    scale_b = 0.05 / math.sqrt(d_out * rank)
    scale_a = 0.05 / math.sqrt(d_in * rank)

    b_rows: list[list[float]] = []
    b_raw_chunks: list[bytes] = []
    b_cols_int: list[list[int]] = [[] for _ in range(rank)]
    for _ in range(d_out):
        vals = [rng.gauss(0.0, scale_b) for _ in range(rank)]
        raw_row = struct.pack(f"<{rank}e", *vals)
        b_raw_chunks.append(raw_row)
        b_rows.append(list(struct.unpack(f"<{rank}e", raw_row)))
        for c, bits in enumerate(struct.unpack(f"<{rank}H", raw_row)):
            b_cols_int[c].append(_decode_f16_bits(bits))

    a_rows: list[list[float]] = []
    a_raw_chunks: list[bytes] = []
    a_rows_int: list[list[int]] = []
    for _ in range(rank):
        vals = [rng.gauss(0.0, scale_a) for _ in range(d_in)]
        raw_row = struct.pack(f"<{d_in}e", *vals)
        a_raw_chunks.append(raw_row)
        a_rows.append(list(struct.unpack(f"<{d_in}e", raw_row)))
        a_rows_int.append([_decode_f16_bits(bits) for bits in struct.unpack(f"<{d_in}H", raw_row)])

    # Exact integer squared Frobenius norm: ||B A||_F^2 = Tr((B^T B)(A A^T)) / 2^96.
    gram_b = [[0] * rank for _ in range(rank)]
    gram_a = [[0] * rank for _ in range(rank)]
    for i in range(rank):
        for j in range(i, rank):
            gb_ij = sum(map(operator.mul, b_cols_int[i], b_cols_int[j]))
            ga_ij = sum(map(operator.mul, a_rows_int[i], a_rows_int[j]))
            gram_b[i][j] = gram_b[j][i] = gb_ij
            gram_a[i][j] = gram_a[j][i] = ga_ij

    frob_int = sum(gram_b[i][j] * gram_a[i][j] for i in range(rank) for j in range(rank))
    # Exact integer ceiling of (frob_int * 2^radix_scale_exp2) / 2^96.
    frob_scaled = ((frob_int << radix_scale_exp2) + ((1 << _FROB_SHIFT) - 1)) >> _FROB_SHIFT
    # Conservative exact spectral upper bound: sigma_max(BA)^2 <= ||BA||_F^2.
    spec_scaled = frob_scaled

    factor_digest = hashlib.sha256(b"".join(b_raw_chunks) + b"".join(a_raw_chunks)).hexdigest()
    cert_dict: dict[str, Any] = {
        "radix_scale_exp2": radix_scale_exp2,
        "frobenius_sq_scaled": frob_scaled,
        "spectral_norm_sq_scaled": spec_scaled,
        "leakage_energy_scaled": 0,
        "max_safety_dot_sq_scaled": 0,
        "projector_epoch": projector_epoch,
        "factor_digest": factor_digest,
        "quant_dtype": "f16",
    }
    cert_sha = hashlib.sha256(
        json.dumps(cert_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return b_rows, a_rows, cert_dict, cert_sha


def build_llama_server_lora_flags_stdlib(
    adapter_paths: Sequence[Path | str],
    *,
    initial_scale: float = 0.0,
    slot_save_path: Path | str | None = None,
) -> list[str]:
    """Build fail-closed llama-server CLI flags for multi-LoRA serving using pure stdlib."""
    if isinstance(adapter_paths, (str, bytes, bytearray)) or not isinstance(adapter_paths, Sequence):
        raise ValueError("adapter_paths must be a non-string sequence of paths")
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty")
    scale_f = _require_finite_float(initial_scale, "initial_scale")
    validated_paths: list[str] = []
    for p in adapter_paths:
        norm = _validate_cli_path(p, label="adapter_path")
        if "," in norm or ":" in norm:
            raise ValueError("adapter_path must not contain ',' or ':'")
        validated_paths.append(norm)
    scaled_spec = ",".join(f"{norm}:{scale_f}" for norm in validated_paths)
    flags = [
        "--lora-scaled",
        scaled_spec,
        "--lora-init-without-apply",
        "--slots",
        "--no-cache-idle-slots",
        "--slot-prompt-similarity",
        "0.0",
        "--ctx-checkpoints",
        "0",
    ]
    if slot_save_path is not None:
        flags.extend(["--slot-save-path", _validate_cli_path(slot_save_path, label="slot_save_path")])
    return flags

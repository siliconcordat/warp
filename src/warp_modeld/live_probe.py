"""Stage 1 live llama-server attestation: verify /props and base model digest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat
from typing import Any

from .errors import CheckpointMismatchError, RuntimeFlagDriftError
from .fingerprint import CheckpointFingerprint, RuntimeFlags, compute_file_sha256
from .llama_client import LlamaServerProps

BUILD_INFO_COMMIT_RE = re.compile(r"\A(?:b[0-9]+-)?([0-9a-f]{7,64})\Z")


def extract_build_commit(build_info: str) -> str:
    """Extract a canonical lowercase hex git commit from llama-server build_info."""
    if not isinstance(build_info, str):
        raise RuntimeFlagDriftError("live llama-server build_info must be a string")
    match = BUILD_INFO_COMMIT_RE.match(build_info)
    if match is None:
        raise RuntimeFlagDriftError(f"unparseable live llama-server build_info: {build_info!r}")
    return match.group(1)


@dataclass(frozen=True)
class LiveAttestationReport:
    """Signed evidence record distinguishing measured live server fields from config declarations."""

    server_build_info: str
    server_commit: str
    server_n_ctx: int
    server_model_path: str
    measured_weights_sha256: str
    measured_fields: tuple[str, ...]
    declared_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "props_endpoint": "/props",
            "server_build_info": self.server_build_info,
            "server_commit": self.server_commit,
            "server_n_ctx": self.server_n_ctx,
            "server_model_path": self.server_model_path,
            "measured_weights_sha256": self.measured_weights_sha256,
            "measured_fields": list(self.measured_fields),
            "declared_fields": list(self.declared_fields),
        }


def verify_live_server_attestation(
    props: LlamaServerProps,
    pinned_checkpoint: CheckpointFingerprint,
    live_runtime_flags: RuntimeFlags,
) -> LiveAttestationReport:
    """Verify running llama-server /props and on-disk base model digest against daemon pins.

    Fails closed on any runtime-flag or base-model mismatch, and returns an explicit
    measured vs declared field inventory for signing into the ledger receipt.
    """
    if not isinstance(props, LlamaServerProps):
        raise RuntimeFlagDriftError("live server properties must be a LlamaServerProps instance")

    # 1. Measure live context size (default_generation_settings.n_ctx).
    if type(props.n_ctx) is not int or props.n_ctx != live_runtime_flags.ctx_size:
        raise RuntimeFlagDriftError(
            f"live llama-server n_ctx (type={type(props.n_ctx).__name__}) "
            f"!= configured ctx_size={live_runtime_flags.ctx_size}"
        )

    # 2. Measure live binary commit from build_info (allowing a longer live commit matching the pinned prefix).
    live_commit = extract_build_commit(props.build_info)
    expected_commit = live_runtime_flags.llama_server_commit
    if not hmac.compare_digest(live_commit[: len(expected_commit)], expected_commit):
        raise RuntimeFlagDriftError(
            f"live llama-server commit {live_commit!r} (from {props.build_info!r}) "
            f"!= configured llama_server_commit {expected_commit!r}"
        )

    # 3. Measure live base model weights SHA-256 directly from props.model_path on disk.
    if not isinstance(props.model_path, str) or not props.model_path.strip() or "\x00" in props.model_path:
        raise CheckpointMismatchError("live llama-server reported missing or invalid model_path")
    try:
        props.model_path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointMismatchError("live llama-server reported non-UTF-8 model_path") from exc
    resolved_model = os.path.realpath(props.model_path)
    try:
        model_stat = os.stat(resolved_model)
    except OSError as exc:
        raise CheckpointMismatchError(
            f"live llama-server base model path inaccessible: {props.model_path!r}: {exc}"
        ) from exc
    if not stat.S_ISREG(model_stat.st_mode):
        raise CheckpointMismatchError(
            f"live llama-server base model path is not a regular file: {resolved_model!r}"
        )
    try:
        measured_weights_sha = compute_file_sha256(Path(resolved_model))
    except OSError as exc:
        raise CheckpointMismatchError(
            f"live llama-server base model file unreadable: {resolved_model!r}: {exc}"
        ) from exc
    if not hmac.compare_digest(measured_weights_sha, pinned_checkpoint.weights_sha256):
        raise CheckpointMismatchError(
            f"live base model digest mismatch for {resolved_model!r}: "
            f"disk {measured_weights_sha} != pinned {pinned_checkpoint.weights_sha256}"
        )

    measured: list[str] = [
        "checkpoint.weights_sha256",
        "runtime_flags.ctx_size",
        "runtime_flags.llama_server_commit",
    ]
    declared: list[str] = [
        "checkpoint.config_sha256",
        "checkpoint.layout_sha256",
        "checkpoint.tokenizer_sha256",
        "runtime_flags.chat_template_args",
        "runtime_flags.lora_init_without_apply",
        "runtime_flags.rope_args",
    ]
    if pinned_checkpoint.safetensors_provenance_sha256 is not None:
        declared.append("checkpoint.safetensors_provenance_sha256")

    # 4. Verify optional extended /props fields whenever the running server reports them;
    #    mark unexposed fields honestly as declared.
    if props.chat_template is not None:
        if not isinstance(props.chat_template, str):
            raise CheckpointMismatchError("live llama-server chat_template must be a string")
        try:
            tpl_bytes = props.chat_template.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CheckpointMismatchError("live llama-server chat_template is not valid UTF-8") from exc
        tpl_sha = hashlib.sha256(tpl_bytes).hexdigest()
        if not hmac.compare_digest(tpl_sha, pinned_checkpoint.chat_template_sha256):
            raise CheckpointMismatchError(
                f"live llama-server chat_template digest {tpl_sha} "
                f"!= pinned {pinned_checkpoint.chat_template_sha256}"
            )
        measured.append("checkpoint.chat_template_sha256")
    else:
        declared.append("checkpoint.chat_template_sha256")

    if props.quantization is not None:
        if props.quantization != live_runtime_flags.quantization:
            raise RuntimeFlagDriftError(
                f"live llama-server quantization {props.quantization!r} "
                f"!= configured {live_runtime_flags.quantization!r}"
            )
        measured.append("runtime_flags.quantization")
    else:
        declared.append("runtime_flags.quantization")

    if props.kv_cache_type is not None:
        if props.kv_cache_type != live_runtime_flags.kv_cache_type:
            raise RuntimeFlagDriftError(
                f"live llama-server kv_cache_type {props.kv_cache_type!r} "
                f"!= configured {live_runtime_flags.kv_cache_type!r}"
            )
        measured.append("runtime_flags.kv_cache_type")
    else:
        declared.append("runtime_flags.kv_cache_type")

    if props.ctx_checkpoints is not None:
        if type(props.ctx_checkpoints) is not int or props.ctx_checkpoints != live_runtime_flags.ctx_checkpoints:
            raise RuntimeFlagDriftError(
                f"live llama-server ctx_checkpoints (type={type(props.ctx_checkpoints).__name__}) "
                f"!= configured {live_runtime_flags.ctx_checkpoints}"
            )
        measured.append("runtime_flags.ctx_checkpoints")
    else:
        declared.append("runtime_flags.ctx_checkpoints")

    if props.swa_checkpoints is not None:
        if type(props.swa_checkpoints) is not int or props.swa_checkpoints != live_runtime_flags.swa_checkpoints:
            raise RuntimeFlagDriftError(
                f"live llama-server swa_checkpoints (type={type(props.swa_checkpoints).__name__}) "
                f"!= configured {live_runtime_flags.swa_checkpoints}"
            )
        measured.append("runtime_flags.swa_checkpoints")
    else:
        declared.append("runtime_flags.swa_checkpoints")

    return LiveAttestationReport(
        server_build_info=props.build_info,
        server_commit=live_commit,
        server_n_ctx=props.n_ctx,
        server_model_path=resolved_model,
        measured_weights_sha256=measured_weights_sha,
        measured_fields=tuple(sorted(measured)),
        declared_fields=tuple(sorted(declared)),
    )

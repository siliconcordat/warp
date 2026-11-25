"""Shared test fixture builders for warp-modeld unit and adversarial tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from warp_modeld.canary import CanarySuite
from warp_modeld.certificate import CertificateBounds, FixedRadixCertificate, RationalBound
from warp_modeld.fingerprint import CheckpointFingerprint, Plane1Lineage, RuntimeFlags
from warp_modeld.ledger import HmacSha256Signer, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.verifier import WarpModeld

REPO_ROOT = Path(__file__).resolve().parent.parent
CANARY_SUITE_PATH = REPO_ROOT / "schemas" / "verifier-canaries.json"


def fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_checkpoint() -> CheckpointFingerprint:
    return CheckpointFingerprint(
        model_id="Qwen3-8B-Q4_K_M",
        layout_sha256=fake_sha("layout-qwen3-8b"),
        weights_sha256=fake_sha("weights-qwen3-8b-q4km"),
        config_sha256=fake_sha("config-qwen3-8b"),
        tokenizer_sha256=fake_sha("tokenizer-qwen3-8b"),
        chat_template_sha256=fake_sha("chat-template-qwen3-8b"),
        safetensors_provenance_sha256=fake_sha("safetensors-qwen3-8b-bf16"),
    )


def make_lineage() -> Plane1Lineage:
    return Plane1Lineage(
        corpus_tree_sha256=fake_sha("corpus-tree-2026-11-19"),
        domain_commits={
            "corpus": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            "weft-eido": "be2766525500112233445566778899aabbccddee",
            "formal-verification": "3165e20e7a00112233445566778899aabbccddee",
        },
        contested_heads=(),
    )


def make_runtime_flags() -> RuntimeFlags:
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
    )


def make_single_bounds() -> CertificateBounds:
    return CertificateBounds(
        max_frobenius_sq=RationalBound(num=5, den=1),
        max_spectral_norm_sq=RationalBound(num=2, den=1),
        max_leakage_energy=RationalBound(num=1, den=100),
        max_safety_dot_sq=RationalBound(num=1, den=1000),
    )


def make_joint_bounds() -> CertificateBounds:
    return CertificateBounds(
        max_frobenius_sq=RationalBound(num=20, den=1),
        max_spectral_norm_sq=RationalBound(num=8, den=1),
        max_leakage_energy=RationalBound(num=1, den=20),
        max_safety_dot_sq=RationalBound(num=1, den=200),
    )


def make_adapter_bundle_dict(
    *,
    adapter_id: str = "mde:corpus",
    domain: str = "corpus",
    llama_adapter_id: int = 0,
    scale_num: int = 1000,
    scale_den: int = 1000,
    checkpoint: CheckpointFingerprint | None = None,
    lineage: Plane1Lineage | None = None,
    runtime_flags: RuntimeFlags | None = None,
    radix_scale: int = 1 << 24,
    frobenius_sq_scaled: int | None = None,
    spectral_norm_sq_scaled: int | None = None,
    leakage_energy_scaled: int | None = None,
    max_safety_dot_sq_scaled: int | None = None,
    adapter_path: str | None = None,
    adapter_sha256: str | None = None,
) -> dict[str, Any]:
    ckpt = checkpoint or make_checkpoint()
    lin = lineage or make_lineage()
    flags = runtime_flags or make_runtime_flags()
    asha = adapter_sha256 or fake_sha(f"adapter-{adapter_id}")

    frob = frobenius_sq_scaled if frobenius_sq_scaled is not None else radix_scale  # 1.0
    spec = spectral_norm_sq_scaled if spectral_norm_sq_scaled is not None else (radix_scale // 4)  # 0.25
    leak = leakage_energy_scaled if leakage_energy_scaled is not None else (radix_scale // 1000)  # 0.001
    safety = max_safety_dot_sq_scaled if max_safety_dot_sq_scaled is not None else (radix_scale // 10000)  # 0.0001

    cert = FixedRadixCertificate.create(
        adapter_id=adapter_id,
        adapter_sha256=asha,
        base_weights_sha256=ckpt.weights_sha256,
        radix_scale=radix_scale,
        frobenius_sq_scaled=frob,
        spectral_norm_sq_scaled=spec,
        leakage_energy_scaled=leak,
        max_safety_dot_sq_scaled=safety,
    )

    return {
        "adapter_id": adapter_id,
        "domain": domain,
        "llama_adapter_id": llama_adapter_id,
        "adapter_sha256": asha,
        "adapter_path": adapter_path,
        "scale_num": scale_num,
        "scale_den": scale_den,
        "checkpoint": {
            "model_id": ckpt.model_id,
            "layout_sha256": ckpt.layout_sha256,
            "weights_sha256": ckpt.weights_sha256,
            "config_sha256": ckpt.config_sha256,
            "tokenizer_sha256": ckpt.tokenizer_sha256,
            "chat_template_sha256": ckpt.chat_template_sha256,
            "safetensors_provenance_sha256": ckpt.safetensors_provenance_sha256,
        },
        "source_lineage": {
            "corpus_tree_sha256": lin.corpus_tree_sha256,
            "domain_commits": {domain: lin.domain_commits[domain]},
            "contested_heads": list(lin.contested_heads),
        },
        "runtime_flags_sha256": flags.canonical_digest(),
        "derivation_digest": fake_sha(f"derivation-{adapter_id}"),
        "certificate": cert.to_dict(),
    }


def make_modeld(base_url: str, tmp_dir: Path, *, signer_key: bytes = b"test-verifier-key-32-bytes-00001") -> WarpModeld:
    signer = HmacSha256Signer(signer_key, key_id="warp-test-hmac-v1")
    ledger = SignedLedger(tmp_dir / "ledger.jsonl", signer)
    suite = CanarySuite.from_verifier_file(CANARY_SUITE_PATH)
    client = LlamaServerClient(base_url, require_localhost=True)
    return WarpModeld(
        require_adapter_file=False,
        pinned_checkpoint=make_checkpoint(),
        pinned_lineage=make_lineage(),
        live_runtime_flags=make_runtime_flags(),
        single_bounds=make_single_bounds(),
        joint_bounds=make_joint_bounds(),
        canary_suite=suite,
        llama_client=client,
        ledger=ledger,
        canary_slot_id=0,
    )

"""Stage 1 fail-closed verification: 4-layer checkpoint, Plane 1 lineage, and 5th runtime-flag digest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .certificate import FixedRadixCertificate
from .errors import (
    CheckpointMismatchError,
    ContestedLineageError,
    RuntimeFlagDriftError,
    StaleLineageError,
    ZoneManifestDriftError,
)
from .state_banks import STATE_BANK_KV, STATE_BANK_LATTICE, STATE_BANK_SSM

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
ZONE_MANIFEST_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ZONE_MANIFEST_SCHEMA = "zone-manifest/v0.1"
WEFT_BINDING_SCHEMA = "weft-binding/v0.1"
WEFT_CLASS_1 = "class1_rederivable"
WEFT_CLASS_2 = "class2_open_recipe_base"


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize payload into deterministic canonical UTF-8 JSON bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256(value: str, field_name: str, exc_cls: type[Exception]) -> str:
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise exc_cls(f"{field_name} must be a 64-char lowercase hex sha256 digest")
    return value


def _validate_commit(value: str, field_name: str, exc_cls: type[Exception]) -> str:
    if not isinstance(value, str) or not COMMIT_RE.match(value):
        raise exc_cls(f"{field_name} must be a lowercase hex commit or digest (7 to 64 hex chars)")
    return value


def _require_strict_int(value: Any, field_name: str, exc_cls: type[Exception]) -> int:
    if type(value) is not int:
        raise exc_cls(f"{field_name} must be an exact integer")
    return value


def validate_zone_manifest_id(value: str) -> str:
    if not isinstance(value, str) or not ZONE_MANIFEST_ID_RE.match(value):
        raise ZoneManifestDriftError(
            f"zone_manifest_id must match 'sha256:<64 lowercase hex>' (got {value!r})"
        )
    return value


def compute_zone_manifest_id(manifest: Mapping[str, Any]) -> str:
    """Compute canonical L5 zone-manifest/v0.1 SHA-256 digest matching siliconcordat/mde."""
    if not isinstance(manifest, Mapping):
        raise ZoneManifestDriftError("zone manifest must be a JSON object")
    body = {k: v for k, v in manifest.items() if k != "manifest_id"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def verify_zone_manifest_dict(manifest: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    """Verify an L5 zone-manifest/v0.1 object and extract its canonical bank-tag mapping.

    Fails closed on schema mismatch, invalid or tampered manifest_id, duplicate or
    non-integer l1_bank_tag values, or missing zones.
    """
    if not isinstance(manifest, Mapping):
        raise ZoneManifestDriftError("zone manifest must be a JSON object")
    if manifest.get("schema") != ZONE_MANIFEST_SCHEMA:
        raise ZoneManifestDriftError(
            f"unsupported zone manifest schema {manifest.get('schema')!r} (expected {ZONE_MANIFEST_SCHEMA!r})"
        )
    stored_id = validate_zone_manifest_id(manifest.get("manifest_id", ""))
    computed_id = compute_zone_manifest_id(manifest)
    if not hmac.compare_digest(stored_id, computed_id):
        raise ZoneManifestDriftError(
            f"zone manifest tampered: stored {stored_id} != computed {computed_id}"
        )

    zones = manifest.get("zones")
    if not isinstance(zones, list) or not zones:
        raise ZoneManifestDriftError("zone manifest must define a non-empty 'zones' list")

    bank_tags: dict[str, int] = {}
    seen_tags: set[int] = set()
    for entry in zones:
        if not isinstance(entry, Mapping):
            raise ZoneManifestDriftError("zone manifest entry must be a mapping")
        zone_name = entry.get("zone")
        tag = entry.get("l1_bank_tag")
        if not isinstance(zone_name, str) or not zone_name.strip():
            raise ZoneManifestDriftError("zone manifest entry missing valid 'zone' name")
        if zone_name in bank_tags:
            raise ZoneManifestDriftError(f"duplicate zone '{zone_name}' in zone manifest")
        if type(tag) is not int or tag < 0 or tag > 255:
            raise ZoneManifestDriftError(f"zone '{zone_name}' l1_bank_tag must be an int in [0, 255]")
        if tag in seen_tags:
            raise ZoneManifestDriftError(f"duplicate l1_bank_tag {tag} in zone manifest")
        seen_tags.add(tag)
        bank_tags[zone_name] = tag

    return stored_id, bank_tags


@dataclass(frozen=True)
class CheckpointFingerprint:
    """Layers 1 to 3 of the base checkpoint identity (layout, weights, config/tokenizer/template)."""

    model_id: str
    layout_sha256: str
    weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    safetensors_provenance_sha256: str | None = None
    has_ssm_recurrent: bool = False
    has_horizontal_lattice: bool = False
    lattice_layers: int = 0
    lattice_width: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise CheckpointMismatchError("model_id must be a non-empty string")
        _validate_sha256(self.layout_sha256, "layout_sha256", CheckpointMismatchError)
        _validate_sha256(self.weights_sha256, "weights_sha256", CheckpointMismatchError)
        _validate_sha256(self.config_sha256, "config_sha256", CheckpointMismatchError)
        _validate_sha256(self.tokenizer_sha256, "tokenizer_sha256", CheckpointMismatchError)
        _validate_sha256(self.chat_template_sha256, "chat_template_sha256", CheckpointMismatchError)
        if self.safetensors_provenance_sha256 is not None:
            _validate_sha256(
                self.safetensors_provenance_sha256,
                "safetensors_provenance_sha256",
                CheckpointMismatchError,
            )
        if type(self.has_ssm_recurrent) is not bool:
            raise CheckpointMismatchError("has_ssm_recurrent must be a boolean")
        if type(self.has_horizontal_lattice) is not bool:
            raise CheckpointMismatchError("has_horizontal_lattice must be a boolean")
        layers = _require_strict_int(self.lattice_layers, "lattice_layers", CheckpointMismatchError)
        width = _require_strict_int(self.lattice_width, "lattice_width", CheckpointMismatchError)
        if layers < 0 or width < 0:
            raise CheckpointMismatchError("lattice_layers and lattice_width must be non-negative")
        if self.has_horizontal_lattice and layers <= 0:
            raise CheckpointMismatchError("lattice_layers must be positive when has_horizontal_lattice is true")
        if not self.has_horizontal_lattice and (layers != 0 or width != 0):
            raise CheckpointMismatchError("lattice_layers and lattice_width must be 0 when has_horizontal_lattice is false")

    def required_state_banks(self) -> tuple[str, ...]:
        banks = [STATE_BANK_KV]
        if self.has_ssm_recurrent:
            banks.append(STATE_BANK_SSM)
        if self.has_horizontal_lattice:
            banks.append(STATE_BANK_LATTICE)
        return tuple(banks)

    def config_bundle_sha256(self) -> str:
        return sha256_json(
            {
                "config_sha256": self.config_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "chat_template_sha256": self.chat_template_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "layout_sha256": self.layout_sha256,
            "weights_sha256": self.weights_sha256,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "config_bundle_sha256": self.config_bundle_sha256(),
            "safetensors_provenance_sha256": self.safetensors_provenance_sha256,
        }
        if self.has_ssm_recurrent or self.has_horizontal_lattice:
            out["has_ssm_recurrent"] = self.has_ssm_recurrent
            out["has_horizontal_lattice"] = self.has_horizontal_lattice
            out["lattice_layers"] = self.lattice_layers
            out["lattice_width"] = self.lattice_width
        return out

    def canonical_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointFingerprint:
        if not isinstance(data, Mapping):
            raise CheckpointMismatchError("checkpoint fingerprint must be a mapping")
        try:
            return cls(
                model_id=data["model_id"],
                layout_sha256=data["layout_sha256"],
                weights_sha256=data["weights_sha256"],
                config_sha256=data["config_sha256"],
                tokenizer_sha256=data["tokenizer_sha256"],
                chat_template_sha256=data["chat_template_sha256"],
                safetensors_provenance_sha256=data.get("safetensors_provenance_sha256"),
                has_ssm_recurrent=data.get("has_ssm_recurrent", False),
                has_horizontal_lattice=data.get("has_horizontal_lattice", False),
                lattice_layers=data.get("lattice_layers", 0),
                lattice_width=data.get("lattice_width", 0),
            )
        except KeyError as exc:
            raise CheckpointMismatchError(f"missing checkpoint fingerprint field: {exc}") from exc

    @classmethod
    def from_weft_binding(cls, weft_manifest: Mapping[str, Any], *, model_id: str) -> CheckpointFingerprint:
        """Build a CheckpointFingerprint natively from an L2 weft BindingManifest dict."""
        if not isinstance(weft_manifest, Mapping):
            raise CheckpointMismatchError("weft_manifest must be a mapping")
        schema = weft_manifest.get("schema")
        if schema is not None and schema != WEFT_BINDING_SCHEMA:
            raise CheckpointMismatchError(
                f"unsupported weft binding schema {schema!r} (expected {WEFT_BINDING_SCHEMA!r})"
            )

        if "layout_sha256" not in weft_manifest or "checkpoint_fingerprint" not in weft_manifest:
            raise CheckpointMismatchError("weft_manifest missing layout_sha256 or checkpoint_fingerprint")

        raw_cfg = weft_manifest.get("config_hashes")
        raw_tok = weft_manifest.get("tokenizer_hashes")
        if not isinstance(raw_cfg, Mapping) or not isinstance(raw_tok, Mapping):
            raise CheckpointMismatchError("weft_manifest config_hashes and tokenizer_hashes must be mappings")

        config_hashes: dict[str, str] = {}
        for k, v in raw_cfg.items():
            config_hashes[str(k)] = _validate_sha256(v, f"config_hashes[{k}]", CheckpointMismatchError)
        tokenizer_hashes: dict[str, str] = {}
        for k, v in raw_tok.items():
            tokenizer_hashes[str(k)] = _validate_sha256(v, f"tokenizer_hashes[{k}]", CheckpointMismatchError)

        config_sha = config_hashes.get("config.json") or sha256_json(config_hashes)
        tokenizer_sha = tokenizer_hashes.get("tokenizer.json") or sha256_json(tokenizer_hashes)
        chat_template_sha = (
            tokenizer_hashes.get("chat_template.jinja")
            or tokenizer_hashes.get("chat_template.json")
            or sha256_json({"chat_template": "absent", "tokenizer": tokenizer_hashes})
        )
        weights_sha = _validate_sha256(
            weft_manifest["checkpoint_fingerprint"],
            "checkpoint_fingerprint",
            CheckpointMismatchError,
        )
        return cls(
            model_id=model_id,
            layout_sha256=weft_manifest["layout_sha256"],
            weights_sha256=weights_sha,
            config_sha256=config_sha,
            tokenizer_sha256=tokenizer_sha,
            chat_template_sha256=chat_template_sha,
            safetensors_provenance_sha256=weights_sha,
        )


@dataclass(frozen=True)
class Plane1Lineage:
    """Layer 4: pinned Plane 1 source policy lineage and contested-head state."""

    corpus_tree_sha256: str
    domain_commits: dict[str, str]
    contested_heads: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_sha256(self.corpus_tree_sha256, "corpus_tree_sha256", StaleLineageError)
        if not isinstance(self.domain_commits, dict) or not self.domain_commits:
            raise StaleLineageError("domain_commits must be a non-empty mapping")
        for domain, commit in self.domain_commits.items():
            if not isinstance(domain, str) or not domain.strip():
                raise StaleLineageError("domain name must be non-empty")
            _validate_commit(commit, f"domain_commits[{domain}]", StaleLineageError)
        if not isinstance(self.contested_heads, tuple):
            object.__setattr__(self, "contested_heads", tuple(self.contested_heads))
        for head in self.contested_heads:
            if not isinstance(head, str) or not head.strip():
                raise ContestedLineageError("contested_heads entries must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_tree_sha256": self.corpus_tree_sha256,
            "domain_commits": dict(sorted(self.domain_commits.items())),
            "contested_heads": list(self.contested_heads),
        }

    def canonical_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plane1Lineage:
        if not isinstance(data, dict):
            raise StaleLineageError("source_lineage must be a mapping")
        try:
            contested = data.get("contested_heads", ())
            if not isinstance(contested, (list, tuple)):
                raise ContestedLineageError("contested_heads must be a list")
            return cls(
                corpus_tree_sha256=data["corpus_tree_sha256"],
                domain_commits=dict(data["domain_commits"]),
                contested_heads=tuple(contested),
            )
        except KeyError as exc:
            raise StaleLineageError(f"missing source_lineage field: {exc}") from exc


@dataclass(frozen=True)
class RuntimeFlags:
    """Layer 5: serving runtime flags bound to adapter verification."""

    llama_server_commit: str
    quantization: str
    ctx_size: int
    kv_cache_type: str
    ctx_checkpoints: int
    swa_checkpoints: int
    lora_init_without_apply: bool = True
    rope_args: str = ""
    chat_template_args: str = ""
    has_horizontal_lattice: bool = False
    lattice_flush_supported: bool = False

    def __post_init__(self) -> None:
        _validate_commit(self.llama_server_commit, "llama_server_commit", RuntimeFlagDriftError)
        if not isinstance(self.quantization, str) or not self.quantization.strip():
            raise RuntimeFlagDriftError("quantization must be a non-empty string")
        ctx = _require_strict_int(self.ctx_size, "ctx_size", RuntimeFlagDriftError)
        if ctx <= 0:
            raise RuntimeFlagDriftError("ctx_size must be positive")
        if not isinstance(self.kv_cache_type, str) or not self.kv_cache_type.strip():
            raise RuntimeFlagDriftError("kv_cache_type must be a non-empty string")
        ctx_cp = _require_strict_int(self.ctx_checkpoints, "ctx_checkpoints", RuntimeFlagDriftError)
        swa_cp = _require_strict_int(self.swa_checkpoints, "swa_checkpoints", RuntimeFlagDriftError)
        if ctx_cp < 0 or swa_cp < 0:
            raise RuntimeFlagDriftError("checkpoint counts must be non-negative")
        if type(self.lora_init_without_apply) is not bool:
            raise RuntimeFlagDriftError("lora_init_without_apply must be a boolean")
        if type(self.has_horizontal_lattice) is not bool:
            raise RuntimeFlagDriftError("has_horizontal_lattice must be a boolean")
        if type(self.lattice_flush_supported) is not bool:
            raise RuntimeFlagDriftError("lattice_flush_supported must be a boolean")
        if not isinstance(self.rope_args, str) or not isinstance(self.chat_template_args, str):
            raise RuntimeFlagDriftError("rope_args and chat_template_args must be strings")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "llama_server_commit": self.llama_server_commit,
            "quantization": self.quantization,
            "ctx_size": self.ctx_size,
            "kv_cache_type": self.kv_cache_type,
            "ctx_checkpoints": self.ctx_checkpoints,
            "swa_checkpoints": self.swa_checkpoints,
            "lora_init_without_apply": self.lora_init_without_apply,
            "rope_args": self.rope_args,
            "chat_template_args": self.chat_template_args,
        }
        if self.has_horizontal_lattice or self.lattice_flush_supported:
            out["has_horizontal_lattice"] = self.has_horizontal_lattice
            out["lattice_flush_supported"] = self.lattice_flush_supported
        return out

    def canonical_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeFlags:
        if not isinstance(data, Mapping):
            raise RuntimeFlagDriftError("runtime_flags must be a mapping")
        try:
            return cls(
                llama_server_commit=data["llama_server_commit"],
                quantization=data["quantization"],
                ctx_size=data["ctx_size"],
                kv_cache_type=data["kv_cache_type"],
                ctx_checkpoints=data["ctx_checkpoints"],
                swa_checkpoints=data["swa_checkpoints"],
                lora_init_without_apply=data["lora_init_without_apply"],
                rope_args=data.get("rope_args", ""),
                chat_template_args=data.get("chat_template_args", ""),
                has_horizontal_lattice=data.get("has_horizontal_lattice", False),
                lattice_flush_supported=data.get("lattice_flush_supported", False),
            )
        except KeyError as exc:
            raise RuntimeFlagDriftError(f"missing runtime_flags field: {exc}") from exc


@dataclass(frozen=True)
class AdapterBundle:
    """Verified input manifest describing one candidate GGUF LoRA adapter."""

    adapter_id: str
    domain: str
    llama_adapter_id: int
    adapter_sha256: str
    scale_num: int
    scale_den: int
    checkpoint: CheckpointFingerprint
    source_lineage: Plane1Lineage
    runtime_flags_sha256: str
    derivation_digest: str
    certificate: FixedRadixCertificate
    adapter_path: str | None = None
    zone_manifest_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id.strip():
            raise CheckpointMismatchError("adapter_id must be a non-empty string")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise CheckpointMismatchError("domain must be a non-empty string")
        lid = _require_strict_int(self.llama_adapter_id, "llama_adapter_id", CheckpointMismatchError)
        if lid < 0:
            raise CheckpointMismatchError("llama_adapter_id must be non-negative")
        _validate_sha256(self.adapter_sha256, "adapter_sha256", CheckpointMismatchError)
        s_num = _require_strict_int(self.scale_num, "scale_num", CheckpointMismatchError)
        s_den = _require_strict_int(self.scale_den, "scale_den", CheckpointMismatchError)
        if s_num <= 0 or s_den <= 0:
            raise CheckpointMismatchError("scale_num and scale_den must be positive integers")
        _validate_sha256(self.runtime_flags_sha256, "runtime_flags_sha256", RuntimeFlagDriftError)
        _validate_sha256(self.derivation_digest, "derivation_digest", CheckpointMismatchError)
        if self.zone_manifest_id is not None:
            validate_zone_manifest_id(self.zone_manifest_id)

    @property
    def scale_float(self) -> float:
        return self.scale_num / self.scale_den

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "domain": self.domain,
            "llama_adapter_id": self.llama_adapter_id,
            "adapter_sha256": self.adapter_sha256,
            "scale_num": self.scale_num,
            "scale_den": self.scale_den,
            "derivation_digest": self.derivation_digest,
        }
        if self.zone_manifest_id is not None:
            entry["zone_manifest_id"] = self.zone_manifest_id
        return entry

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "domain": self.domain,
            "llama_adapter_id": self.llama_adapter_id,
            "adapter_sha256": self.adapter_sha256,
            "adapter_path": self.adapter_path,
            "scale_num": self.scale_num,
            "scale_den": self.scale_den,
            "checkpoint": {
                "model_id": self.checkpoint.model_id,
                "layout_sha256": self.checkpoint.layout_sha256,
                "weights_sha256": self.checkpoint.weights_sha256,
                "config_sha256": self.checkpoint.config_sha256,
                "tokenizer_sha256": self.checkpoint.tokenizer_sha256,
                "chat_template_sha256": self.checkpoint.chat_template_sha256,
                "safetensors_provenance_sha256": self.checkpoint.safetensors_provenance_sha256,
                **(
                    {
                        "has_ssm_recurrent": self.checkpoint.has_ssm_recurrent,
                        "has_horizontal_lattice": self.checkpoint.has_horizontal_lattice,
                        "lattice_layers": self.checkpoint.lattice_layers,
                        "lattice_width": self.checkpoint.lattice_width,
                    }
                    if (self.checkpoint.has_ssm_recurrent or self.checkpoint.has_horizontal_lattice)
                    else {}
                ),
            },
            "source_lineage": self.source_lineage.to_dict(),
            "runtime_flags_sha256": self.runtime_flags_sha256,
            "derivation_digest": self.derivation_digest,
            "certificate": self.certificate.to_dict(),
        }
        if self.zone_manifest_id is not None:
            out["zone_manifest_id"] = self.zone_manifest_id
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdapterBundle:
        if not isinstance(data, Mapping):
            raise CheckpointMismatchError("adapter bundle must be a mapping")
        try:
            return cls(
                adapter_id=data["adapter_id"],
                domain=data["domain"],
                llama_adapter_id=data["llama_adapter_id"],
                adapter_sha256=data["adapter_sha256"],
                scale_num=data["scale_num"],
                scale_den=data["scale_den"],
                checkpoint=CheckpointFingerprint.from_dict(data["checkpoint"]),
                source_lineage=Plane1Lineage.from_dict(data["source_lineage"]),
                runtime_flags_sha256=data["runtime_flags_sha256"],
                derivation_digest=data["derivation_digest"],
                certificate=FixedRadixCertificate.from_dict(data["certificate"]),
                adapter_path=data.get("adapter_path"),
                zone_manifest_id=data.get("zone_manifest_id"),
            )
        except KeyError as exc:
            raise CheckpointMismatchError(f"missing adapter bundle field: {exc}") from exc

    @classmethod
    def from_weft_binding(
        cls,
        weft_manifest: Mapping[str, Any],
        weft_certificate: Mapping[str, Any],
        *,
        adapter_id: str,
        domain: str,
        llama_adapter_id: int,
        adapter_sha256: str,
        model_id: str,
        source_lineage: Plane1Lineage | Mapping[str, Any],
        runtime_flags_sha256: str,
        scale_num: int = 1000,
        scale_den: int = 1000,
        adapter_path: str | None = None,
        zone_manifest_id: str | None = None,
    ) -> AdapterBundle:
        """Construct a verified AdapterBundle directly from L2 weft BindingManifest + certificate."""
        from .canary import reject_adapter_supplied_canaries

        if not isinstance(weft_manifest, Mapping):
            raise CheckpointMismatchError("weft_manifest must be a mapping")
        if not isinstance(weft_certificate, Mapping):
            raise CheckpointMismatchError("weft_certificate must be a mapping")

        reject_adapter_supplied_canaries(dict(weft_manifest), path="weft_manifest")
        reject_adapter_supplied_canaries(dict(weft_certificate), path="weft_certificate")

        adapter_class = weft_manifest.get("adapter_class")
        if adapter_class not in (WEFT_CLASS_1, WEFT_CLASS_2):
            raise CheckpointMismatchError(f"unknown weft adapter_class: {adapter_class!r}")
        if adapter_class == WEFT_CLASS_1:
            recipe = weft_manifest.get("derivation_recipe")
            if not isinstance(recipe, Mapping) or not recipe:
                raise CheckpointMismatchError(
                    "Hard Veto 1: class1_rederivable weft manifest requires a non-empty derivation_recipe"
                )

        # Verify internal weft runtime_flags match weft runtime_flags_sha256 (fail closed on internal tamper).
        if "runtime_flags" in weft_manifest and "runtime_flags_sha256" in weft_manifest:
            wf_flags = weft_manifest["runtime_flags"]
            wf_flags_sha = _validate_sha256(
                weft_manifest["runtime_flags_sha256"],
                "weft_manifest.runtime_flags_sha256",
                RuntimeFlagDriftError,
            )
            if not isinstance(wf_flags, Mapping) or not hmac.compare_digest(sha256_json(dict(wf_flags)), wf_flags_sha):
                raise RuntimeFlagDriftError(
                    "weft_manifest runtime_flags does not match its declared runtime_flags_sha256"
                )

        checkpoint = CheckpointFingerprint.from_weft_binding(weft_manifest, model_id=model_id)

        expected_cert_sha = weft_manifest.get("certificate_sha256")
        if expected_cert_sha is None:
            raise CheckpointMismatchError("weft_manifest missing certificate_sha256")
        expected_factor_digest = _validate_sha256(
            weft_manifest.get("factor_digest", ""),
            "weft_manifest.factor_digest",
            CheckpointMismatchError,
        )

        cert = FixedRadixCertificate.from_weft_dict(
            weft_certificate,
            adapter_id=adapter_id,
            adapter_sha256=adapter_sha256,
            base_weights_sha256=checkpoint.weights_sha256,
            expected_weft_sha256=expected_cert_sha,
        )
        if not hmac.compare_digest(str(weft_certificate["factor_digest"]), expected_factor_digest):
            raise CheckpointMismatchError(
                f"weft factor_digest mismatch between manifest and certificate for {adapter_id}"
            )

        lineage_obj = (
            source_lineage
            if isinstance(source_lineage, Plane1Lineage)
            else Plane1Lineage.from_dict(dict(source_lineage))
        )
        derivation_digest = sha256_json(dict(weft_manifest))
        return cls(
            adapter_id=adapter_id,
            domain=domain,
            llama_adapter_id=llama_adapter_id,
            adapter_sha256=adapter_sha256,
            scale_num=scale_num,
            scale_den=scale_den,
            checkpoint=checkpoint,
            source_lineage=lineage_obj,
            runtime_flags_sha256=runtime_flags_sha256,
            derivation_digest=derivation_digest,
            certificate=cert,
            adapter_path=adapter_path,
            zone_manifest_id=zone_manifest_id,
        )


@dataclass(frozen=True)
class Stage1Report:
    adapter_id: str
    checkpoint_digest: str
    lineage_digest: str
    runtime_flags_digest: str
    adapter_file_verified: bool
    zone_manifest_id: str | None = None


def verify_stage1(
    bundle: AdapterBundle,
    pinned_checkpoint: CheckpointFingerprint,
    pinned_lineage: Plane1Lineage,
    live_runtime_flags: RuntimeFlags,
    *,
    pinned_zone_manifest_id: str | None = None,
) -> Stage1Report:
    """Verify all 4 checkpoint/lineage layers plus the 5th runtime-flag digest.

    Fails closed before any GPU memory or llama-server HTTP call is touched.
    """
    if not live_runtime_flags.lora_init_without_apply:
        raise RuntimeFlagDriftError(
            "llama-server must run with --lora-init-without-apply enabled",
            code="ERR_UNSAFE_RUNTIME_FLAGS",
        )

    if pinned_checkpoint.has_horizontal_lattice or live_runtime_flags.has_horizontal_lattice:
        if pinned_checkpoint.has_horizontal_lattice != live_runtime_flags.has_horizontal_lattice:
            raise RuntimeFlagDriftError(
                "has_horizontal_lattice mismatch between pinned checkpoint and live runtime flags",
                code="ERR_UNSAFE_RUNTIME_FLAGS",
            )
        if not live_runtime_flags.lattice_flush_supported:
            raise RuntimeFlagDriftError(
                "checkpoint declares horizontal lattice streams but backend lacks atomic lattice_flush_supported",
                code="ERR_UNSAFE_RUNTIME_FLAGS",
            )

    if pinned_lineage.contested_heads:
        raise ContestedLineageError(
            f"pinned Plane 1 lineage has contested heads: {list(pinned_lineage.contested_heads)}"
        )
    if bundle.source_lineage.contested_heads:
        raise ContestedLineageError(
            f"adapter {bundle.adapter_id} carries contested heads: {list(bundle.source_lineage.contested_heads)}"
        )

    # Layers 1-3: checkpoint fingerprint
    checks = (
        ("model_id", bundle.checkpoint.model_id, pinned_checkpoint.model_id),
        ("layout_sha256", bundle.checkpoint.layout_sha256, pinned_checkpoint.layout_sha256),
        ("weights_sha256", bundle.checkpoint.weights_sha256, pinned_checkpoint.weights_sha256),
        ("config_sha256", bundle.checkpoint.config_sha256, pinned_checkpoint.config_sha256),
        ("tokenizer_sha256", bundle.checkpoint.tokenizer_sha256, pinned_checkpoint.tokenizer_sha256),
        ("chat_template_sha256", bundle.checkpoint.chat_template_sha256, pinned_checkpoint.chat_template_sha256),
    )
    for field_name, got, expected in checks:
        if not hmac.compare_digest(got, expected):
            raise CheckpointMismatchError(
                f"adapter {bundle.adapter_id} {field_name} mismatch: got {got}, expected {expected}"
            )

    if pinned_checkpoint.safetensors_provenance_sha256 is not None:
        got_prov = bundle.checkpoint.safetensors_provenance_sha256
        if got_prov is None or not hmac.compare_digest(got_prov, pinned_checkpoint.safetensors_provenance_sha256):
            raise CheckpointMismatchError(
                f"adapter {bundle.adapter_id} safetensors_provenance_sha256 mismatch"
            )

    # State-bank architecture agreement (prevents spoofing non-lattice bundles onto a Latent Lattice host).
    if (
        bundle.checkpoint.has_ssm_recurrent != pinned_checkpoint.has_ssm_recurrent
        or bundle.checkpoint.has_horizontal_lattice != pinned_checkpoint.has_horizontal_lattice
        or bundle.checkpoint.lattice_layers != pinned_checkpoint.lattice_layers
        or bundle.checkpoint.lattice_width != pinned_checkpoint.lattice_width
    ):
        raise CheckpointMismatchError(
            f"adapter {bundle.adapter_id} state-bank architecture mismatch "
            f"(bundle ssm={bundle.checkpoint.has_ssm_recurrent}, lattice={bundle.checkpoint.has_horizontal_lattice}/"
            f"{bundle.checkpoint.lattice_layers}x{bundle.checkpoint.lattice_width} vs pinned "
            f"ssm={pinned_checkpoint.has_ssm_recurrent}, lattice={pinned_checkpoint.has_horizontal_lattice}/"
            f"{pinned_checkpoint.lattice_layers}x{pinned_checkpoint.lattice_width})"
        )

    # Layer 4: Plane 1 lineage (stale_lineage check)
    if not hmac.compare_digest(bundle.source_lineage.corpus_tree_sha256, pinned_lineage.corpus_tree_sha256):
        raise StaleLineageError(
            f"stale_lineage: adapter {bundle.adapter_id} corpus_tree_sha256 "
            f"{bundle.source_lineage.corpus_tree_sha256} != pinned {pinned_lineage.corpus_tree_sha256}"
        )

    if bundle.domain not in pinned_lineage.domain_commits:
        raise StaleLineageError(
            f"stale_lineage: domain '{bundle.domain}' is not pinned in active Plane 1 lineage"
        )
    if bundle.domain not in bundle.source_lineage.domain_commits:
        raise StaleLineageError(
            f"stale_lineage: adapter {bundle.adapter_id} omits commit pin for its own domain '{bundle.domain}'"
        )

    for domain, bundle_commit in bundle.source_lineage.domain_commits.items():
        if domain not in pinned_lineage.domain_commits:
            raise StaleLineageError(
                f"stale_lineage: adapter {bundle.adapter_id} references unpinned domain '{domain}'"
            )
        pinned_commit = pinned_lineage.domain_commits[domain]
        if not hmac.compare_digest(bundle_commit, pinned_commit):
            raise StaleLineageError(
                f"stale_lineage: domain '{domain}' commit {bundle_commit} != pinned {pinned_commit}"
            )

    # Layer 5: serving runtime-flag digest
    live_flags_digest = live_runtime_flags.canonical_digest()
    if not hmac.compare_digest(bundle.runtime_flags_sha256, live_flags_digest):
        raise RuntimeFlagDriftError(
            f"runtime flag drift for adapter {bundle.adapter_id}: "
            f"bundle expects {bundle.runtime_flags_sha256}, live host is {live_flags_digest}"
        )

    # Optional L5 zone-manifest/v0.1 bank-tag rotation binding
    if pinned_zone_manifest_id is not None:
        validate_zone_manifest_id(pinned_zone_manifest_id)
        if bundle.zone_manifest_id is None:
            raise ZoneManifestDriftError(
                f"adapter {bundle.adapter_id} missing zone_manifest_id required by host ({pinned_zone_manifest_id})"
            )
        if not hmac.compare_digest(bundle.zone_manifest_id, pinned_zone_manifest_id):
            raise ZoneManifestDriftError(
                f"zone_manifest_id drift for adapter {bundle.adapter_id}: "
                f"bundle has {bundle.zone_manifest_id}, host pins {pinned_zone_manifest_id}"
            )
    elif bundle.zone_manifest_id is not None:
        raise ZoneManifestDriftError(
            f"adapter {bundle.adapter_id} pins zone_manifest_id {bundle.zone_manifest_id} "
            f"but host has no pinned zone manifest"
        )

    # Optional on-disk adapter GGUF integrity verification
    adapter_file_verified = False
    if bundle.adapter_path is not None:
        if not isinstance(bundle.adapter_path, str) or not bundle.adapter_path.strip() or "\x00" in bundle.adapter_path:
            raise CheckpointMismatchError("adapter_path must be a non-empty string without NUL bytes")
        try:
            bundle.adapter_path.encode("utf-8")
            path = Path(bundle.adapter_path)
            if not path.is_file():
                raise CheckpointMismatchError(f"adapter file missing on disk: {bundle.adapter_path}")
            actual_sha = compute_file_sha256(path)
        except (OSError, UnicodeEncodeError, ValueError) as exc:
            raise CheckpointMismatchError(f"adapter file unreadable on disk: {exc}") from exc
        if not hmac.compare_digest(actual_sha, bundle.adapter_sha256):
            raise CheckpointMismatchError(
                f"adapter {bundle.adapter_id} file digest mismatch: disk {actual_sha} != manifest {bundle.adapter_sha256}"
            )
        adapter_file_verified = True

    return Stage1Report(
        adapter_id=bundle.adapter_id,
        checkpoint_digest=pinned_checkpoint.canonical_digest(),
        lineage_digest=pinned_lineage.canonical_digest(),
        runtime_flags_digest=live_flags_digest,
        adapter_file_verified=adapter_file_verified,
        zone_manifest_id=pinned_zone_manifest_id,
    )

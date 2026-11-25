"""Stage 3 fail-closed gate: verifier-owned shadow-load canary runner."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import subprocess
import time
from typing import Any, Sequence

from .errors import CanaryFailedError, ForgedCanaryError, WarpError
from .fingerprint import AdapterBundle
from .llama_client import LlamaClientError, LlamaServerClient

CANARY_MODE_FIXED_SUITE = "fixed_suite_regression"
CANARY_MODE_SAMPLED_POOL = "verifier_sampled_pool"
CANARY_MODE_SEALED_EXTERNAL = "sealed_external_command"

EVIDENCE_SCOPE_BY_MODE: dict[str, str] = {
    CANARY_MODE_FIXED_SUITE: (
        "Deterministic regression and wiring check against a fixed canary suite; "
        "does not prove adversarial resistance against an adapter tuned to known prompts."
    ),
    CANARY_MODE_SAMPLED_POOL: (
        "Per-mount randomized subset drawn from a verifier-owned canary pool; "
        "bounds static prompt memorization only insofar as the pool stays secret from training."
    ),
    CANARY_MODE_SEALED_EXTERNAL: (
        "Sealed external verifier command executed outside warp-modeld plaintext prompt custody; "
        "strength depends on the external verifier's private pool secrecy and grading oracle."
    ),
}

_LOWER_HEX = frozenset("0123456789abcdef")


def _is_sha256_hex(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in _LOWER_HEX for ch in value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_exact_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise CanaryConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _require_positive_finite_float(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0.0:
        raise CanaryConfigError(f"{name} must be a finite number > 0")
    return float(value)

FORBIDDEN_ADAPTER_CANARY_KEYS = frozenset(
    {
        "canaries",
        "canary",
        "canary_suite",
        "canary_results",
        "canary_pass",
        "canary_passed",
        "canary_override",
        "skip_canaries",
        "self_reported_pass",
        "verifier_override",
        "expected_outputs",
    }
)


class CanaryConfigError(WarpError):
    code = "ERR_CANARY_CONFIG"
    stage = "stage3_canary"


def reject_adapter_supplied_canaries(payload: Any, path: str = "root") -> None:
    """Recursively reject any adapter manifest attempting to supply or override canaries."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            norm = str(key).strip().lower()
            if norm in FORBIDDEN_ADAPTER_CANARY_KEYS:
                raise ForgedCanaryError(
                    f"adapter payload attempted to supply forbidden canary field '{key}' at {path}"
                )
            reject_adapter_supplied_canaries(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            reject_adapter_supplied_canaries(item, f"{path}[{idx}]")


def _strip_fenced_code_blocks(text: str) -> str:
    """Return only prose lines outside triple-backtick fenced code blocks."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


@dataclass(frozen=True)
class CanarySpec:
    canary_id: str
    category: str  # "preservation" | "domain" | "cross_adapter"
    prompt: str
    domains: tuple[str, ...] = ()
    system_prompt: str | None = None
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    prose_only_forbid: tuple[str, ...] = ()
    regex_must_match: tuple[str, ...] = ()
    regex_must_not_match: tuple[str, ...] = ()
    min_chars: int = 1
    max_chars: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.category not in ("preservation", "domain", "cross_adapter"):
            raise CanaryConfigError(f"invalid canary category: {self.category}")
        if not self.canary_id.strip() or not self.prompt.strip():
            raise CanaryConfigError("canary_id and prompt must be non-empty")
        if self.category == "domain" and len(self.domains) != 1:
            raise CanaryConfigError(f"domain canary {self.canary_id} must specify exactly 1 domain")
        if self.category == "cross_adapter" and len(self.domains) < 2:
            raise CanaryConfigError(f"cross_adapter canary {self.canary_id} must specify at least 2 domains")

    def evaluate(self, response_text: str) -> list[str]:
        """Return a list of deterministic rule failure reasons (empty means pass)."""
        failures: list[str] = []
        if len(response_text) < self.min_chars:
            failures.append(f"output length {len(response_text)} < min_chars {self.min_chars}")
        if self.max_chars is not None and len(response_text) > self.max_chars:
            failures.append(f"output length {len(response_text)} > max_chars {self.max_chars}")

        for needle in self.must_contain:
            if needle not in response_text:
                failures.append(f"missing required substring {needle!r}")

        for forbidden in self.must_not_contain:
            if forbidden in response_text:
                failures.append(f"found forbidden substring {forbidden!r}")

        if self.prose_only_forbid:
            prose = _strip_fenced_code_blocks(response_text)
            for forbidden in self.prose_only_forbid:
                if forbidden in prose:
                    failures.append(f"found forbidden prose substring {forbidden!r}")

        for pattern in self.regex_must_match:
            if re.search(pattern, response_text, re.MULTILINE) is None:
                failures.append(f"did not match required regex {pattern!r}")

        for pattern in self.regex_must_not_match:
            if re.search(pattern, response_text, re.MULTILINE) is not None:
                failures.append(f"matched forbidden regex {pattern!r}")

        return failures

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanarySpec:
        if not isinstance(data, dict):
            raise CanaryConfigError("canary entry must be a mapping")
        try:
            return cls(
                canary_id=str(data["canary_id"]),
                category=str(data["category"]),
                prompt=str(data["prompt"]),
                domains=tuple(str(d) for d in data.get("domains", ())),
                system_prompt=data.get("system_prompt"),
                must_contain=tuple(str(x) for x in data.get("must_contain", ())),
                must_not_contain=tuple(str(x) for x in data.get("must_not_contain", ())),
                prose_only_forbid=tuple(str(x) for x in data.get("prose_only_forbid", ())),
                regex_must_match=tuple(str(x) for x in data.get("regex_must_match", ())),
                regex_must_not_match=tuple(str(x) for x in data.get("regex_must_not_match", ())),
                min_chars=int(data.get("min_chars", 1)),
                max_chars=int(data["max_chars"]) if data.get("max_chars") is not None else None,
                max_tokens=int(data["max_tokens"]) if data.get("max_tokens") is not None else None,
            )
        except KeyError as exc:
            raise CanaryConfigError(f"missing canary field: {exc}") from exc


# Explicit JSONL categories, and the explicit `family` values that mark a safety /
# general-capability preservation canary. Classification never looks at the canary ID.
_JSONL_CATEGORIES = frozenset({"preservation", "domain", "cross_adapter"})
_JSONL_PRESERVATION_FAMILIES = frozenset({"safety", "general", "preservation", "s", "g"})


@dataclass(frozen=True)
class CanarySuite:
    """OS-owned read-only suite of preservation, domain, and cross-adapter canaries."""

    suite_sha256: str
    canaries: tuple[CanarySpec, ...]
    source_path: str
    suite_role: str = "fixed_regression_suite"
    example_only: bool = False

    @classmethod
    def sealed_placeholder(
        cls,
        suite_sha256: str,
        *,
        source_path: str = "sealed-external-command",
    ) -> CanarySuite:
        if not _is_sha256_hex(suite_sha256):
            raise CanaryConfigError("sealed canary suite_sha256 must be a 64-char lowercase hex digest")
        return cls(
            suite_sha256=suite_sha256,
            canaries=(),
            source_path=source_path,
            suite_role="sealed_external_suite",
            example_only=False,
        )

    @classmethod
    def from_verifier_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        min_preservation: int = 4,
        reject_world_writable: bool = True,
    ) -> CanarySuite:
        file_path = Path(path)
        if not file_path.is_file():
            raise CanaryConfigError(f"verifier canary file not found: {file_path}")

        if reject_world_writable and os.name == "posix":
            mode = file_path.stat().st_mode
            if mode & stat.S_IWOTH:
                raise CanaryConfigError(f"verifier canary file must not be world-writable: {file_path}")

        raw_bytes = file_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(actual_sha, expected_sha256):
            raise CanaryConfigError(
                f"verifier canary suite digest mismatch: got {actual_sha}, expected {expected_sha256}"
            )

        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CanaryConfigError(f"invalid verifier canary JSON: {exc}") from exc

        if not isinstance(parsed, dict) or not isinstance(parsed.get("canaries"), list):
            raise CanaryConfigError("verifier canary file must contain a 'canaries' list")

        raw_role = parsed.get("suite_role", "fixed_regression_suite")
        if type(raw_role) is not str or not raw_role.strip():
            raise CanaryConfigError("verifier canary 'suite_role' must be a non-empty string")
        raw_example = parsed.get("example_only", False)
        if type(raw_example) is not bool:
            raise CanaryConfigError("verifier canary 'example_only' must be a boolean")

        specs = tuple(CanarySpec.from_dict(item) for item in parsed["canaries"])
        preservation_count = sum(1 for spec in specs if spec.category == "preservation")
        if preservation_count < min_preservation:
            raise CanaryConfigError(
                f"verifier canary suite requires at least {min_preservation} preservation canaries "
                f"(found {preservation_count})"
            )

        return cls(
            suite_sha256=actual_sha,
            canaries=specs,
            source_path=str(file_path),
            suite_role=raw_role.strip(),
            example_only=raw_example,
        )

    @classmethod
    def from_jsonl_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        min_preservation: int = 4,
        reject_world_writable: bool = True,
        default_domain: str = "formal-verification",
    ) -> CanarySuite:
        """Load a verifier-owned JSONL canary suite (e.g. Phase 1.3 canaries_16.jsonl)."""
        file_path = Path(path)
        if not file_path.is_file():
            raise CanaryConfigError(f"verifier canary JSONL file not found: {file_path}")
        if reject_world_writable and os.name == "posix":
            mode = file_path.stat().st_mode
            if mode & stat.S_IWOTH:
                raise CanaryConfigError(f"verifier canary file must not be world-writable: {file_path}")

        raw_bytes = file_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(actual_sha, expected_sha256):
            raise CanaryConfigError(
                f"verifier canary suite digest mismatch: got {actual_sha}, expected {expected_sha256}"
            )

        specs: list[CanarySpec] = []
        seen_ids: set[str] = set()
        for idx, raw_line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanaryConfigError(f"invalid JSON on line {idx} of {file_path}: {exc}") from exc
            if not isinstance(row, dict):
                raise CanaryConfigError(f"line {idx} of {file_path} must be a JSON object")

            # Canary identity and category must be explicit. The preservation floor is the
            # safety gate, so a row is never classified from substrings of its ID
            # (e.g. "lean-express-1" must not count as a preservation canary).
            raw_id = row.get("canary_id", row.get("id"))
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise CanaryConfigError(
                    f"line {idx} of {file_path} must carry an explicit non-empty 'canary_id' or 'id'"
                )
            cid = raw_id.strip()
            if cid in seen_ids:
                raise CanaryConfigError(f"duplicate canary_id {cid!r} on line {idx} of {file_path}")
            seen_ids.add(cid)

            raw_cat = str(row.get("category") or "").strip().lower()
            raw_fam = str(row.get("family") or "").strip().lower()
            if raw_cat:
                if raw_cat not in _JSONL_CATEGORIES:
                    raise CanaryConfigError(
                        f"canary {cid!r} on line {idx} of {file_path} has unknown category {raw_cat!r}; "
                        f"expected one of {sorted(_JSONL_CATEGORIES)}"
                    )
                cat = raw_cat
            elif raw_fam:
                cat = "preservation" if raw_fam in _JSONL_PRESERVATION_FAMILIES else "domain"
            else:
                raise CanaryConfigError(
                    f"canary {cid!r} on line {idx} of {file_path} must declare an explicit "
                    f"'category' {sorted(_JSONL_CATEGORIES)} or 'family'"
                )

            raw_domains = row.get("domains")
            if isinstance(raw_domains, list) and raw_domains:
                domains = tuple(str(d) for d in raw_domains)
            elif cat == "domain":
                domains = (str(row.get("domain") or default_domain),)
            else:
                domains = ()

            specs.append(
                CanarySpec(
                    canary_id=cid,
                    category=cat,
                    prompt=str(row["prompt"]),
                    domains=domains,
                    system_prompt=row.get("system_prompt"),
                    must_contain=tuple(str(x) for x in row.get("must_contain", ())),
                    must_not_contain=tuple(str(x) for x in row.get("must_not_contain", ())),
                    prose_only_forbid=tuple(str(x) for x in row.get("prose_only_forbid", ())),
                    regex_must_match=tuple(str(x) for x in row.get("regex_must_match", ())),
                    regex_must_not_match=tuple(str(x) for x in row.get("regex_must_not_match", ())),
                    min_chars=int(row.get("min_chars", 1)),
                    max_chars=int(row["max_chars"]) if row.get("max_chars") is not None else None,
                    max_tokens=int(row["max_tokens"]) if row.get("max_tokens") is not None else None,
                )
            )

        preservation_count = sum(1 for spec in specs if spec.category == "preservation")
        if preservation_count < min_preservation:
            raise CanaryConfigError(
                f"verifier canary JSONL requires at least {min_preservation} preservation canaries "
                f"(found {preservation_count})"
            )
        return cls(suite_sha256=actual_sha, canaries=tuple(specs), source_path=str(file_path))

    def select_for_domains(self, domains: Sequence[str]) -> list[CanarySpec]:
        domain_set = set(domains)
        preservation = [c for c in self.canaries if c.category == "preservation"]
        domain_matches = [c for c in self.canaries if c.category == "domain" and c.domains[0] in domain_set]
        cross_matches = [
            c
            for c in self.canaries
            if c.category == "cross_adapter" and len(domain_set) > 1 and set(c.domains).issubset(domain_set)
        ]

        # Every requested adapter domain must have at least one verifier-owned domain canary.
        covered_domains = {c.domains[0] for c in domain_matches}
        missing_domains = sorted(domain_set - covered_domains)
        if missing_domains:
            raise CanaryFailedError(
                f"no verifier-owned domain canaries configured for domains: {missing_domains}"
            )

        preservation.sort(key=lambda c: c.canary_id)
        domain_matches.sort(key=lambda c: c.canary_id)
        cross_matches.sort(key=lambda c: c.canary_id)
        return preservation + domain_matches + cross_matches


@dataclass(frozen=True)
class CanaryCheckOutcome:
    canary_id: str
    category: str
    passed: bool
    failures: tuple[str, ...]
    prompt_sha256: str | None = None
    response_sha256: str | None = None


def _compute_transcript_sha256(
    suite_sha256: str,
    evidence_mode: str,
    outcomes: Sequence[CanaryCheckOutcome],
) -> str:
    return _canonical_sha256(
        {
            "suite_sha256": suite_sha256,
            "evidence_mode": evidence_mode,
            "outcomes": [
                {
                    "canary_id": o.canary_id,
                    "category": o.category,
                    "passed": o.passed,
                    "prompt_sha256": o.prompt_sha256,
                    "response_sha256": o.response_sha256,
                }
                for o in outcomes
            ],
        }
    )


def _make_outcome(spec: CanarySpec, response_text: str) -> CanaryCheckOutcome:
    failures = tuple(spec.evaluate(response_text))
    return CanaryCheckOutcome(
        canary_id=spec.canary_id,
        category=spec.category,
        passed=(len(failures) == 0),
        failures=failures,
        prompt_sha256=_canonical_sha256({"system_prompt": spec.system_prompt, "prompt": spec.prompt}),
        response_sha256=_sha256_text(response_text),
    )


@dataclass(frozen=True)
class CanaryResult:
    suite_sha256: str
    outcomes: tuple[CanaryCheckOutcome, ...]
    preservation_passed: int
    preservation_total: int
    canary_slot_flushed: bool
    elapsed_ms: float = 0.0
    evidence_mode: str = CANARY_MODE_FIXED_SUITE
    evidence_scope: str = EVIDENCE_SCOPE_BY_MODE[CANARY_MODE_FIXED_SUITE]
    pool_total: int | None = None
    sampled_per_mount: bool = False
    example_suite_used: bool = False
    transcript_sha256: str | None = None

    def attestation_dict(self) -> dict[str, Any]:
        pool_size = self.pool_total if self.pool_total is not None else len(self.outcomes)
        transcript = (
            self.transcript_sha256
            if self.transcript_sha256 is not None
            else _compute_transcript_sha256(self.suite_sha256, self.evidence_mode, self.outcomes)
        )
        return {
            "schema_version": "warp.canary_attestation/v1",
            "evidence_mode": self.evidence_mode,
            "evidence_scope": self.evidence_scope,
            "suite_sha256": self.suite_sha256,
            "example_suite_used": self.example_suite_used,
            "sampled_per_mount": self.sampled_per_mount,
            "pool_total": pool_size,
            "canaries_passed": len(self.outcomes),
            "preservation_passed": self.preservation_passed,
            "preservation_total": self.preservation_total,
            "evaluated_canary_ids": [o.canary_id for o in self.outcomes],
            "transcript_sha256": transcript,
        }


def verify_canary_attestation(
    attestation: Mapping[str, Any],
    *,
    expected_suite_sha256: str | None = None,
    require_non_example: bool = False,
    min_preservation: int = 4,
) -> dict[str, Any]:
    """Fail-closed verification of a signed mount receipt's canary_attestation payload."""
    _require_exact_int(min_preservation, "min_preservation", minimum=1)
    if expected_suite_sha256 is not None and not _is_sha256_hex(expected_suite_sha256):
        raise CanaryConfigError("expected_suite_sha256 must be a 64-char lowercase hex digest")
    if type(require_non_example) is not bool:
        raise CanaryConfigError("require_non_example must be a boolean")
    if not isinstance(attestation, Mapping):
        raise CanaryFailedError("canary_attestation must be a JSON object")

    schema_ver = attestation.get("schema_version")
    if type(schema_ver) is not str or schema_ver != "warp.canary_attestation/v1":
        raise CanaryFailedError("unsupported or missing canary_attestation schema_version")

    mode = attestation.get("evidence_mode")
    if type(mode) is not str or mode not in EVIDENCE_SCOPE_BY_MODE:
        raise CanaryFailedError(f"unknown canary evidence_mode: {mode!r}")

    scope = attestation.get("evidence_scope")
    if type(scope) is not str or scope != EVIDENCE_SCOPE_BY_MODE[mode]:
        raise CanaryFailedError("canary evidence_scope does not match declared evidence_mode")

    suite_sha = attestation.get("suite_sha256")
    if not _is_sha256_hex(suite_sha):
        raise CanaryFailedError("canary_attestation suite_sha256 must be a 64-char lowercase hex digest")
    if expected_suite_sha256 is not None and not hmac.compare_digest(suite_sha, expected_suite_sha256):
        raise CanaryFailedError("canary_attestation suite_sha256 does not match expected digest")

    transcript_sha = attestation.get("transcript_sha256")
    if not _is_sha256_hex(transcript_sha):
        raise CanaryFailedError("canary_attestation transcript_sha256 must be a 64-char lowercase hex digest")

    example_used = attestation.get("example_suite_used")
    if type(example_used) is not bool:
        raise CanaryFailedError("canary_attestation example_suite_used must be a boolean")
    if require_non_example and example_used:
        raise CanaryFailedError("canary_attestation used an example-only suite when non-example evidence is required")

    sampled = attestation.get("sampled_per_mount")
    if type(sampled) is not bool:
        raise CanaryFailedError("canary_attestation sampled_per_mount must be a boolean")

    evaluated_ids = attestation.get("evaluated_canary_ids")
    if not isinstance(evaluated_ids, list) or len(evaluated_ids) == 0:
        raise CanaryFailedError("canary_attestation evaluated_canary_ids must be a non-empty list")
    seen_ids: set[str] = set()
    for cid in evaluated_ids:
        if type(cid) is not str or not cid.strip():
            raise CanaryFailedError("canary_attestation evaluated_canary_ids must contain non-empty strings")
        if cid in seen_ids:
            raise CanaryFailedError(f"duplicate canary_id in canary_attestation: {cid!r}")
        seen_ids.add(cid)

    canaries_passed = attestation.get("canaries_passed")
    pool_total = attestation.get("pool_total")
    pres_passed = attestation.get("preservation_passed")
    pres_total = attestation.get("preservation_total")
    for field_name, val in (
        ("canaries_passed", canaries_passed),
        ("pool_total", pool_total),
        ("preservation_passed", pres_passed),
        ("preservation_total", pres_total),
    ):
        if type(val) is not int:
            raise CanaryFailedError(f"canary_attestation {field_name} must be an integer")

    if canaries_passed != len(evaluated_ids):
        raise CanaryFailedError("canary_attestation canaries_passed must equal len(evaluated_canary_ids)")
    if pool_total < canaries_passed:
        raise CanaryFailedError("canary_attestation pool_total cannot be smaller than canaries_passed")
    if mode == CANARY_MODE_FIXED_SUITE and (sampled or pool_total != canaries_passed):
        raise CanaryFailedError(
            "fixed_suite_regression attestation requires sampled_per_mount=false and pool_total == canaries_passed"
        )
    if mode == CANARY_MODE_SAMPLED_POOL and (not sampled or pool_total <= canaries_passed):
        raise CanaryFailedError(
            "verifier_sampled_pool attestation requires sampled_per_mount=true and pool_total > canaries_passed"
        )
    if pres_total < min_preservation:
        raise CanaryFailedError(
            f"canary_attestation preservation_total ({pres_total}) < min_preservation ({min_preservation})"
        )
    if pres_total > canaries_passed:
        raise CanaryFailedError("canary_attestation preservation_total cannot exceed canaries_passed")
    if pres_passed != pres_total:
        raise CanaryFailedError(
            f"canary_attestation preservation gate incomplete ({pres_passed}/{pres_total})"
        )

    return dict(attestation)


def _extract_chat_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(response.get("content"), str):
        return response["content"]
    raise CanaryFailedError("llama-server returned response without assistant text content")


class CanaryRunner:
    """Executes shadow-loaded verifier-owned canaries against llama-server."""

    def __init__(
        self,
        llama_client: LlamaServerClient,
        suite: CanarySuite,
        *,
        canary_slot_id: int = 0,
        seed: int = 42,
        max_tokens: int = 256,
        top_p: float = 1.0,
        sample_preservation: int | None = None,
        sample_per_domain: int | None = None,
        sample_cross_adapter: int | None = None,
        sample_rng: random.Random | None = None,
        disallow_example_suite: bool = False,
    ) -> None:
        if type(disallow_example_suite) is not bool:
            raise CanaryConfigError("disallow_example_suite must be a boolean")
        if disallow_example_suite and suite.example_only:
            raise CanaryConfigError(
                "refusing to run example-only canary suite when disallow_example_suite is enabled"
            )
        if sample_preservation is not None:
            _require_exact_int(sample_preservation, "sample_preservation", minimum=4)
        if sample_per_domain is not None:
            _require_exact_int(sample_per_domain, "sample_per_domain", minimum=1)
        if sample_cross_adapter is not None:
            _require_exact_int(sample_cross_adapter, "sample_cross_adapter", minimum=1)
        self.llama_client = llama_client
        self.suite = suite
        self.canary_slot_id = canary_slot_id
        self.seed = seed
        self.max_tokens = max_tokens
        self.top_p = float(top_p)
        self.sample_preservation = sample_preservation
        self.sample_per_domain = sample_per_domain
        self.sample_cross_adapter = sample_cross_adapter
        self.sample_rng = sample_rng
        self.disallow_example_suite = disallow_example_suite
        self._sampling_enabled = (
            sample_preservation is not None
            or sample_per_domain is not None
            or sample_cross_adapter is not None
        )

    def _select_canaries(self, domains: Sequence[str]) -> tuple[list[CanarySpec], int]:
        eligible = self.suite.select_for_domains(domains)
        pool_total = len(eligible)
        if not self._sampling_enabled:
            return eligible, pool_total

        rng = self.sample_rng if self.sample_rng is not None else secrets.SystemRandom()
        preservation: list[CanarySpec] = []
        by_domain: dict[str, list[CanarySpec]] = {}
        cross: list[CanarySpec] = []
        for spec in eligible:
            if spec.category == "preservation":
                preservation.append(spec)
            elif spec.category == "domain":
                by_domain.setdefault(spec.domains[0], []).append(spec)
            else:
                cross.append(spec)

        if self.sample_preservation is not None:
            if len(preservation) < self.sample_preservation:
                raise CanaryConfigError(
                    f"preservation canary pool ({len(preservation)}) < sample_preservation ({self.sample_preservation})"
                )
            preservation = sorted(rng.sample(preservation, self.sample_preservation), key=lambda c: c.canary_id)

        sampled_domain: list[CanarySpec] = []
        for dom in sorted(by_domain):
            dom_specs = by_domain[dom]
            if self.sample_per_domain is not None:
                if len(dom_specs) < self.sample_per_domain:
                    raise CanaryConfigError(
                        f"domain {dom!r} canary pool ({len(dom_specs)}) < sample_per_domain ({self.sample_per_domain})"
                    )
                dom_specs = rng.sample(dom_specs, self.sample_per_domain)
            sampled_domain.extend(dom_specs)
        sampled_domain.sort(key=lambda c: c.canary_id)

        if self.sample_cross_adapter is not None and cross:
            if len(cross) < self.sample_cross_adapter:
                raise CanaryConfigError(
                    f"cross_adapter canary pool ({len(cross)}) < sample_cross_adapter ({self.sample_cross_adapter})"
                )
            cross = sorted(rng.sample(cross, self.sample_cross_adapter), key=lambda c: c.canary_id)

        selected = preservation + sampled_domain + cross
        if len(selected) == pool_total:
            raise CanaryConfigError(
                f"canary pool sampling requires eligible pool ({pool_total}) strictly larger than "
                f"sample size ({len(selected)})"
            )
        return selected, pool_total

    def run(
        self,
        bundles: Sequence[AdapterBundle],
        *,
        allowed_active_ids: frozenset[int] = frozenset(),
        required_state_banks: Sequence[str] | None = None,
    ) -> CanaryResult:
        t0 = time.perf_counter()
        if not bundles:
            raise CanaryFailedError("cannot run shadow canaries on an empty adapter list")

        # Verify all candidate llama_adapter_ids are loaded on llama-server and
        # no unverified adapter has a non-zero global scale (--lora-init-without-apply invariant).
        try:
            loaded = self.llama_client.list_lora_adapters()
        except LlamaClientError as exc:
            raise CanaryFailedError(f"failed to inspect loaded llama-server adapters: {exc}") from exc

        for item in loaded:
            if item.id not in allowed_active_ids and abs(item.scale) > 0.0:
                raise CanaryFailedError(
                    f"unverified adapter id {item.id} already has non-zero scale {item.scale}; "
                    f"llama-server must start with --lora-init-without-apply",
                    code="ERR_UNSAFE_RUNTIME_FLAGS",
                )

        loaded_ids = {item.id for item in loaded}
        requested_ids = {b.llama_adapter_id for b in bundles}
        missing = sorted(requested_ids - loaded_ids)
        if missing:
            raise CanaryFailedError(f"candidate adapter ids not loaded in llama-server: {missing}")

        # Ensure zero global scales are left active during shadow testing if not already mounted.
        lora_payload = [{"id": b.llama_adapter_id, "scale": b.scale_float} for b in bundles]
        selected, pool_total = self._select_canaries([b.domain for b in bundles])

        outcomes: list[CanaryCheckOutcome] = []
        preservation_total = 0
        preservation_passed = 0
        slot_flushed = False

        erase_banks = tuple(required_state_banks) if required_state_banks and len(required_state_banks) > 1 else None

        slot_ids: tuple[int, ...] = (self.canary_slot_id,)
        if hasattr(self.llama_client, "list_slots"):
            try:
                listed_slots = self.llama_client.list_slots()
                if len(listed_slots) > 1:
                    slot_ids = tuple(int(s.id) for s in listed_slots)
            except LlamaClientError:
                slot_ids = (self.canary_slot_id,)

        try:
            if len(slot_ids) == 1:
                for spec in selected:
                    self.llama_client.erase_slot(self.canary_slot_id, banks=erase_banks)
                    messages: list[dict[str, str]] = []
                    if spec.system_prompt:
                        messages.append({"role": "system", "content": spec.system_prompt})
                    messages.append({"role": "user", "content": spec.prompt})

                    resp = self.llama_client.chat_completion(
                        messages,
                        lora=lora_payload,
                        temperature=0.0,
                        top_p=self.top_p,
                        seed=self.seed,
                        max_tokens=spec.max_tokens if spec.max_tokens is not None else self.max_tokens,
                        cache_prompt=False,
                        id_slot=self.canary_slot_id,
                    )
                    text = _extract_chat_content(resp)
                    outcome = _make_outcome(spec, text)
                    if spec.category == "preservation":
                        preservation_total += 1
                        if outcome.passed:
                            preservation_passed += 1
                    outcomes.append(outcome)
            else:
                # LPT load-balanced deterministic partitioning across parallel server slots for <3s Stage 3 execution
                buckets: dict[int, list[tuple[int, CanarySpec]]] = {sid: [] for sid in slot_ids}
                slot_loads: dict[int, int] = {sid: 0 for sid in slot_ids}
                ordered_specs = sorted(
                    enumerate(selected),
                    key=lambda pair: (
                        -(pair[1].max_tokens if pair[1].max_tokens is not None else self.max_tokens),
                        0 if pair[1].category != "preservation" else 1,
                        pair[0],
                    ),
                )
                for orig_idx, spec in ordered_specs:
                    best_sid = min(slot_ids, key=lambda s: (slot_loads[s], s))
                    buckets[best_sid].append((orig_idx, spec))
                    slot_loads[best_sid] += spec.max_tokens if spec.max_tokens is not None else self.max_tokens

                def _run_slot_bucket(sid: int, items: list[tuple[int, CanarySpec]]) -> list[tuple[int, CanaryCheckOutcome]]:
                    if not items:
                        return []
                    bucket_out: list[tuple[int, CanaryCheckOutcome]] = []
                    for orig_idx, spec in items:
                        self.llama_client.erase_slot(sid, banks=erase_banks)
                        msgs: list[dict[str, str]] = []
                        if spec.system_prompt:
                            msgs.append({"role": "system", "content": spec.system_prompt})
                        msgs.append({"role": "user", "content": spec.prompt})
                        r = self.llama_client.chat_completion(
                            msgs,
                            lora=lora_payload,
                            temperature=0.0,
                            top_p=self.top_p,
                            seed=self.seed,
                            max_tokens=spec.max_tokens if spec.max_tokens is not None else self.max_tokens,
                            cache_prompt=False,
                            id_slot=sid,
                        )
                        txt = _extract_chat_content(r)
                        bucket_out.append((orig_idx, _make_outcome(spec, txt)))
                    return bucket_out

                indexed_outcomes: list[tuple[int, CanaryCheckOutcome]] = []
                with ThreadPoolExecutor(max_workers=len(slot_ids)) as pool:
                    futs = [pool.submit(_run_slot_bucket, sid, buckets[sid]) for sid in slot_ids]
                    for fut in futs:
                        indexed_outcomes.extend(fut.result())

                indexed_outcomes.sort(key=lambda pair: pair[0])
                for _, out_item in indexed_outcomes:
                    if out_item.category == "preservation":
                        preservation_total += 1
                        if out_item.passed:
                            preservation_passed += 1
                    outcomes.append(out_item)
        except LlamaClientError as exc:
            raise CanaryFailedError(f"shadow canary execution failed: {exc}") from exc
        finally:
            all_flushed = True
            if len(slot_ids) <= 1:
                for sid in slot_ids:
                    try:
                        self.llama_client.erase_slot(sid, banks=erase_banks)
                    except LlamaClientError:
                        all_flushed = False
            else:
                def _safe_erase(sid: int) -> bool:
                    try:
                        self.llama_client.erase_slot(sid, banks=erase_banks)
                        return True
                    except LlamaClientError:
                        return False

                with ThreadPoolExecutor(max_workers=len(slot_ids)) as pool:
                    all_flushed = all(pool.map(_safe_erase, slot_ids))
            slot_flushed = all_flushed

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        failed = [o for o in outcomes if not o.passed]
        if preservation_passed != preservation_total:
            detail = "; ".join(f"{o.canary_id}: {list(o.failures)}" for o in failed if o.category == "preservation")
            raise CanaryFailedError(
                f"preservation canary gate failed ({preservation_passed}/{preservation_total}): {detail}"
            )
        if failed:
            detail = "; ".join(f"{o.canary_id}({o.category}): {list(o.failures)}" for o in failed)
            raise CanaryFailedError(f"verifier canary gate failed ({len(failed)}/{len(outcomes)}): {detail}")
        if not slot_flushed:
            raise CanaryFailedError("failed to flush canary slot after shadow evaluation")

        evidence_mode = CANARY_MODE_SAMPLED_POOL if self._sampling_enabled else CANARY_MODE_FIXED_SUITE
        outcomes_tuple = tuple(outcomes)
        return CanaryResult(
            suite_sha256=self.suite.suite_sha256,
            outcomes=outcomes_tuple,
            preservation_passed=preservation_passed,
            preservation_total=preservation_total,
            canary_slot_flushed=slot_flushed,
            elapsed_ms=elapsed_ms,
            evidence_mode=evidence_mode,
            evidence_scope=EVIDENCE_SCOPE_BY_MODE[evidence_mode],
            pool_total=pool_total,
            sampled_per_mount=self._sampling_enabled,
            example_suite_used=self.suite.example_only,
            transcript_sha256=_compute_transcript_sha256(self.suite.suite_sha256, evidence_mode, outcomes_tuple),
        )


class ExternalCommandCanaryRunner:
    """Executes an opaque verifier-owned canary command without reading private canary plaintext.

    The command receives `--llama-url <url> --lora-json <json> --domains-json <json> --slot-id <id>`
    and must exit 0 with a JSON object on stdout containing:
      - `suite_sha256`: 64-char hex digest of the sealed canary file
      - `preservation_passed`: int
      - `preservation_total`: int (>= 4, must equal `preservation_passed`)
      - `outcomes`: list of `{"canary_id": str, "category": str, "passed": bool, "failures": [...]}`
    """

    def __init__(
        self,
        llama_client: LlamaServerClient,
        command: Sequence[str],
        *,
        suite_sha256_hint: str | None = None,
        expected_suite_sha256: str | None = None,
        canary_slot_id: int = 0,
        timeout_seconds: float = 30.0,
        min_preservation: int = 4,
    ) -> None:
        if isinstance(command, (str, bytes)) or not command:
            raise CanaryConfigError("external canary command must be a non-empty sequence of strings")
        cmd_parts: list[str] = []
        for part in command:
            if type(part) is not str or not part.strip():
                raise CanaryConfigError("external canary command entries must be non-empty strings")
            cmd_parts.append(part)
        _require_exact_int(min_preservation, "min_preservation", minimum=1)
        _require_exact_int(canary_slot_id, "canary_slot_id", minimum=0)
        timeout_val = _require_positive_finite_float(timeout_seconds, "timeout_seconds")
        if suite_sha256_hint is not None and not _is_sha256_hex(suite_sha256_hint):
            raise CanaryConfigError("suite_sha256_hint must be a 64-char lowercase hex digest")
        if expected_suite_sha256 is not None and not _is_sha256_hex(expected_suite_sha256):
            raise CanaryConfigError("expected_suite_sha256 must be a 64-char lowercase hex digest")

        self.llama_client = llama_client
        self.command = tuple(cmd_parts)
        self.expected_suite_sha256 = expected_suite_sha256
        if expected_suite_sha256 is not None:
            self.suite_sha256 = expected_suite_sha256
        elif suite_sha256_hint is not None:
            self.suite_sha256 = suite_sha256_hint
        else:
            self.suite_sha256 = hashlib.sha256(" ".join(self.command).encode("utf-8")).hexdigest()
        self.canary_slot_id = canary_slot_id
        self.timeout_seconds = timeout_val
        self.min_preservation = min_preservation

    def run(
        self,
        bundles: Sequence[AdapterBundle],
        *,
        allowed_active_ids: frozenset[int] = frozenset(),
        required_state_banks: Sequence[str] | None = None,
    ) -> CanaryResult:
        t0 = time.perf_counter()
        if not bundles:
            raise CanaryFailedError("cannot run shadow canaries on an empty adapter list")

        try:
            loaded = self.llama_client.list_lora_adapters()
        except LlamaClientError as exc:
            raise CanaryFailedError(f"failed to inspect loaded llama-server adapters: {exc}") from exc

        for item in loaded:
            if item.id not in allowed_active_ids and abs(item.scale) > 0.0:
                raise CanaryFailedError(
                    f"unverified adapter id {item.id} already has non-zero scale {item.scale}; "
                    f"llama-server must start with --lora-init-without-apply",
                    code="ERR_UNSAFE_RUNTIME_FLAGS",
                )

        loaded_ids = {item.id for item in loaded}
        requested_ids = {b.llama_adapter_id for b in bundles}
        missing = sorted(requested_ids - loaded_ids)
        if missing:
            raise CanaryFailedError(f"candidate adapter ids not loaded in llama-server: {missing}")

        lora_payload = [{"id": b.llama_adapter_id, "scale": b.scale_float} for b in bundles]
        domains = [b.domain for b in bundles]
        erase_banks = tuple(required_state_banks) if required_state_banks and len(required_state_banks) > 1 else None

        lora_json_str = json.dumps(lora_payload, separators=(",", ":"))
        domains_json_str = json.dumps(domains, separators=(",", ":"))
        has_placeholders = any(
            tok_p in arg
            for arg in self.command
            for tok_p in ("{server_url}", "{llama_url}", "{lora_json}", "{domains_json}", "{slot_id}")
        )
        if has_placeholders:
            cmd = [
                arg.replace("{server_url}", self.llama_client.base_url)
                .replace("{llama_url}", self.llama_client.base_url)
                .replace("{lora_json}", lora_json_str)
                .replace("{domains_json}", domains_json_str)
                .replace("{slot_id}", str(self.canary_slot_id))
                for arg in self.command
            ]
        else:
            cmd = [
                *self.command,
                "--llama-url",
                self.llama_client.base_url,
                "--lora-json",
                lora_json_str,
                "--domains-json",
                domains_json_str,
                "--slot-id",
                str(self.canary_slot_id),
            ]
        run_env = dict(os.environ)
        run_env["WARP_SERVER_URL"] = self.llama_client.base_url
        run_env["WARP_LORA_JSON"] = lora_json_str

        slot_flushed = False
        try:
            self.llama_client.erase_slot(self.canary_slot_id, banks=erase_banks)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=run_env,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, LlamaClientError) as exc:
            raise CanaryFailedError(f"opaque verifier canary command failed to execute: {exc}") from exc
        finally:
            try:
                self.llama_client.erase_slot(self.canary_slot_id, banks=erase_banks)
                slot_flushed = True
            except LlamaClientError:
                slot_flushed = False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not slot_flushed:
            raise CanaryFailedError("failed to flush canary slot after opaque canary command")
        if proc.returncode != 0:
            raise CanaryFailedError(
                f"opaque verifier canary command rejected candidate (exit={proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[:300]}"
            )
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise CanaryFailedError(f"opaque verifier canary command returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise CanaryFailedError("opaque verifier canary command output must be a JSON object")

        raw_suite_sha = parsed.get("suite_sha256")
        if raw_suite_sha is None:
            raw_suite_sha = parsed.get("canaries_sha256")
        if raw_suite_sha is not None and not _is_sha256_hex(raw_suite_sha):
            raise CanaryFailedError("opaque verifier suite_sha256 must be a 64-char lowercase hex digest")
        if self.expected_suite_sha256 is not None and (
            raw_suite_sha is None or not hmac.compare_digest(raw_suite_sha, self.expected_suite_sha256)
        ):
            raise CanaryFailedError("opaque verifier suite_sha256 missing or mismatched against pinned digest")
        suite_sha = raw_suite_sha if raw_suite_sha is not None else self.suite_sha256

        raw_outcomes = parsed.get("outcomes")
        if raw_outcomes is None:
            raw_outcomes = parsed.get("results")
        if raw_outcomes is None:
            raw_outcomes = parsed.get("canaries")

        if not isinstance(raw_outcomes, list) or len(raw_outcomes) == 0:
            raise CanaryFailedError(
                "opaque verifier canary command must return a non-empty per-canary 'outcomes' list; "
                "aggregate-only reports are rejected"
            )

        parsed_outcomes: list[CanaryCheckOutcome] = []
        seen_ids: set[str] = set()
        for idx, item in enumerate(raw_outcomes):
            if not isinstance(item, dict) or not isinstance(item.get("passed"), bool):
                raise CanaryFailedError(
                    f"opaque verifier canary outcome #{idx} must be an object with boolean 'passed'"
                )
            raw_id = item.get("canary_id") if item.get("canary_id") is not None else item.get("id")
            cid = str(raw_id or "").strip()
            if not cid:
                raise CanaryFailedError(f"opaque verifier canary outcome #{idx} is missing a non-empty 'canary_id'")
            if cid in seen_ids:
                raise CanaryFailedError(f"duplicate canary_id in opaque verifier outcomes: {cid!r}")
            seen_ids.add(cid)

            raw_cat = str(item.get("category") or "").strip().lower()
            if raw_cat == "preservation":
                cat = "preservation"
            elif raw_cat in ("domain", "formal"):
                cat = "domain"
            elif raw_cat == "cross_adapter":
                cat = "cross_adapter"
            else:
                raise CanaryFailedError(
                    f"invalid category {item.get('category')!r} for canary {cid!r}; "
                    f"expected preservation, domain, or cross_adapter"
                )

            raw_failures = item.get("failures")
            if not isinstance(raw_failures, list):
                raw_failures = item.get("failure_codes", ())
            prompt_sha = item.get("prompt_sha256")
            if prompt_sha is not None and not _is_sha256_hex(prompt_sha):
                raise CanaryFailedError(f"invalid prompt_sha256 for canary {cid!r}")
            response_sha = item.get("response_sha256")
            if response_sha is not None and not _is_sha256_hex(response_sha):
                raise CanaryFailedError(f"invalid response_sha256 for canary {cid!r}")
            parsed_outcomes.append(
                CanaryCheckOutcome(
                    canary_id=cid,
                    category=cat,
                    passed=bool(item["passed"]),
                    failures=tuple(str(f) for f in raw_failures),
                    prompt_sha256=prompt_sha,
                    response_sha256=response_sha,
                )
            )

        outcomes = tuple(parsed_outcomes)
        pres_items = [o for o in outcomes if o.category == "preservation"]
        pres_total = len(pres_items)
        pres_passed = sum(1 for o in pres_items if o.passed)

        raw_pres_total = parsed.get("preservation_total")
        if raw_pres_total is not None and (type(raw_pres_total) is not int or raw_pres_total != pres_total):
            raise CanaryFailedError(
                f"opaque verifier preservation_total ({raw_pres_total}) disagrees with "
                f"outcomes count ({pres_total})"
            )
        raw_pres_passed = parsed.get("preservation_passed")
        if raw_pres_passed is not None and (type(raw_pres_passed) is not int or raw_pres_passed != pres_passed):
            raise CanaryFailedError(
                f"opaque verifier preservation_passed ({raw_pres_passed}) disagrees with "
                f"outcomes count ({pres_passed})"
            )
        if pres_total < self.min_preservation:
            raise CanaryFailedError(
                f"opaque verifier canary gate requires at least {self.min_preservation} preservation canaries "
                f"(found {pres_total})"
            )

        raw_sampled = parsed.get("sampled_per_mount", False)
        if type(raw_sampled) is not bool:
            raise CanaryFailedError("opaque verifier sampled_per_mount must be a boolean")
        raw_pool_total = parsed.get("pool_total")
        if raw_pool_total is None:
            pool_total = len(outcomes)
        elif type(raw_pool_total) is not int or raw_pool_total < len(outcomes):
            raise CanaryFailedError("opaque verifier pool_total must be an integer >= len(outcomes)")
        else:
            pool_total = raw_pool_total
        raw_transcript = parsed.get("transcript_sha256")
        if raw_transcript is not None and not _is_sha256_hex(raw_transcript):
            raise CanaryFailedError("opaque verifier transcript_sha256 must be a 64-char lowercase hex digest")

        failed = [o for o in outcomes if not o.passed]
        verdict_str = str(parsed.get("verdict") or "").strip().lower()
        if (
            pres_passed != pres_total
            or failed
            or parsed.get("passed") is False
            or parsed.get("all_passed") is False
            or verdict_str in ("reject", "fail", "failed")
        ):
            raise CanaryFailedError(
                f"opaque verifier canary gate failed (preservation {pres_passed}/{pres_total}, failed={len(failed)})"
            )
        transcript_sha = (
            raw_transcript
            if raw_transcript is not None
            else _compute_transcript_sha256(suite_sha, CANARY_MODE_SEALED_EXTERNAL, outcomes)
        )
        return CanaryResult(
            suite_sha256=suite_sha,
            outcomes=outcomes,
            preservation_passed=pres_passed,
            preservation_total=pres_total,
            canary_slot_flushed=True,
            elapsed_ms=elapsed_ms,
            evidence_mode=CANARY_MODE_SEALED_EXTERNAL,
            evidence_scope=EVIDENCE_SCOPE_BY_MODE[CANARY_MODE_SEALED_EXTERNAL],
            pool_total=pool_total,
            sampled_per_mount=raw_sampled,
            example_suite_used=False,
            transcript_sha256=transcript_sha,
        )


def _load_canary_suite_path(path: Any, *, expected_sha256: str | None) -> CanarySuite:
    if type(path) is not str or not path.strip():
        raise CanaryConfigError("canary_suite_path must be a non-empty string")
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        return CanarySuite.from_jsonl_file(file_path, expected_sha256=expected_sha256)
    return CanarySuite.from_verifier_file(file_path, expected_sha256=expected_sha256)


def build_canary_runner_from_config(
    cfg: Mapping[str, Any],
    llama_client: LlamaServerClient,
) -> tuple[CanarySuite, CanaryRunner | ExternalCommandCanaryRunner]:
    """Construct Stage 3 suite metadata and runner from a daemon JSON config mapping."""
    if not isinstance(cfg, Mapping):
        raise CanaryConfigError("daemon config must be a JSON object")

    canary_slot_id = _require_exact_int(cfg.get("canary_slot_id", 0), "canary_slot_id", minimum=0)
    canary_max_tokens = _require_exact_int(cfg.get("canary_max_tokens", 256), "canary_max_tokens", minimum=1)
    expected_sha = cfg.get("canary_suite_sha256")
    if expected_sha is not None and not _is_sha256_hex(expected_sha):
        raise CanaryConfigError("canary_suite_sha256 must be a 64-char lowercase hex digest")

    raw_cmd = cfg.get("canary_runner_command")
    raw_path = cfg.get("canary_suite_path")
    if raw_cmd is not None:
        timeout_raw = cfg.get("canary_runner_timeout_seconds", 30.0)
        min_pres_raw = cfg.get("canary_min_preservation", 4)
        if raw_path is not None:
            suite = _load_canary_suite_path(raw_path, expected_sha256=expected_sha)
            runner = ExternalCommandCanaryRunner(
                llama_client,
                raw_cmd,
                suite_sha256_hint=suite.suite_sha256,
                expected_suite_sha256=expected_sha,
                canary_slot_id=canary_slot_id,
                timeout_seconds=timeout_raw,
                min_preservation=min_pres_raw,
            )
            return suite, runner
        if expected_sha is None:
            raise CanaryConfigError(
                "canary_runner_command without canary_suite_path requires pinned canary_suite_sha256"
            )
        runner = ExternalCommandCanaryRunner(
            llama_client,
            raw_cmd,
            expected_suite_sha256=expected_sha,
            canary_slot_id=canary_slot_id,
            timeout_seconds=timeout_raw,
            min_preservation=min_pres_raw,
        )
        return CanarySuite.sealed_placeholder(expected_sha), runner

    if raw_path is None:
        raise CanaryConfigError("config must specify either 'canary_suite_path' or 'canary_runner_command'")

    suite = _load_canary_suite_path(raw_path, expected_sha256=expected_sha)
    runner = CanaryRunner(
        llama_client,
        suite,
        canary_slot_id=canary_slot_id,
        max_tokens=canary_max_tokens,
        sample_preservation=cfg.get("canary_sample_preservation"),
        sample_per_domain=cfg.get("canary_sample_per_domain"),
        sample_cross_adapter=cfg.get("canary_sample_cross_adapter"),
        disallow_example_suite=cfg.get("disallow_example_canary_suite", False),
    )
    return suite, runner

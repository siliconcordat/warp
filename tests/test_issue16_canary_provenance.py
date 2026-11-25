"""Tests for Stage 3 canary provenance, per-mount sampling, sealed CLI wiring, and receipt attestation (Issue #16)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from warp_modeld import cli as warp_cli
from warp_modeld.canary import (
    CANARY_MODE_FIXED_SUITE,
    CANARY_MODE_SAMPLED_POOL,
    CANARY_MODE_SEALED_EXTERNAL,
    EVIDENCE_SCOPE_BY_MODE,
    CanaryCheckOutcome,
    CanaryConfigError,
    CanaryResult,
    CanaryRunner,
    CanarySuite,
    ExternalCommandCanaryRunner,
    _compute_transcript_sha256,
    _is_sha256_hex,
    build_canary_runner_from_config,
    verify_canary_attestation,
)
from warp_modeld.errors import CanaryFailedError
from warp_modeld.fingerprint import AdapterBundle
from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    fake_sha,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_modeld,
    make_runtime_flags,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer


class _StrSub(str):
    pass


class _IntSub(int):
    pass


class _InMemoryLlamaClient:
    """Fast deterministic mock llama-server client supporting single-slot and multi-slot execution."""

    def __init__(self, *, slot_count: int = 1, response_text: str = "refuse No 391 export textContent theorem Nat") -> None:
        self.base_url = "http://127.0.0.1:18080"
        self.slot_count = slot_count
        self.response_text = response_text
        self.prompts_seen: list[str] = []

    def list_lora_adapters(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=0, scale=0.0), SimpleNamespace(id=1, scale=0.0), SimpleNamespace(id=2, scale=0.0)]

    def list_slots(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=i) for i in range(self.slot_count)]

    def erase_slot(self, sid: int, banks: object = None) -> dict[str, int]:
        return {"id_slot": sid}

    def chat_completion(self, messages: list[dict[str, str]], **kwargs: object) -> dict[str, object]:
        self.prompts_seen.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": self.response_text}}]}


def _write_private_pool_json(path: Path, *, example_only: bool = False, suite_role: str = "private_verifier_pool") -> Path:
    canaries = [
        {
            "canary_id": f"pres-{i:02d}",
            "category": "preservation",
            "prompt": f"Preservation prompt {i}",
            "system_prompt": "Be strict." if i == 0 else None,
            "must_contain": ["391"],
        }
        for i in range(8)
    ]
    for dom in ("corpus", "formal-verification"):
        for j in range(3):
            canaries.append(
                {
                    "canary_id": f"dom-{dom}-{j:02d}",
                    "category": "domain",
                    "domains": [dom],
                    "prompt": f"Domain {dom} prompt {j}",
                    "must_contain": ["theorem"],
                }
            )
    for k in range(3):
        canaries.append(
            {
                "canary_id": f"cross-cf-{k:02d}",
                "category": "cross_adapter",
                "domains": ["corpus", "formal-verification"],
                "prompt": f"Cross prompt {k}",
                "must_contain": ["theorem"],
            }
        )
    payload = {
        "suite_version": "1.0.0",
        "owner": "private-verifier",
        "suite_role": suite_role,
        "example_only": example_only,
        "canaries": canaries,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _write_jsonl_suite(path: Path) -> Path:
    rows = [
        {"id": f"p-{i}", "category": "preservation", "prompt": f"Pres {i}", "must_contain": ["391"]}
        for i in range(4)
    ] + [
        {
            "id": "d-corpus-1",
            "category": "domain",
            "domain": "corpus",
            "prompt": "Corpus check",
            "must_contain": ["theorem"],
        }
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _valid_attestation(
    *,
    mode: str = CANARY_MODE_FIXED_SUITE,
    sampled: bool = False,
    pool_total: int = 5,
    evaluated_ids: list[str] | None = None,
    pres_passed: int = 4,
    pres_total: int = 4,
    example_used: bool = False,
    suite_sha: str | None = None,
    transcript_sha: str | None = None,
) -> dict[str, object]:
    ids = evaluated_ids if evaluated_ids is not None else ["p1", "p2", "p3", "p4", "d1"]
    return {
        "schema_version": "warp.canary_attestation/v1",
        "evidence_mode": mode,
        "evidence_scope": EVIDENCE_SCOPE_BY_MODE[mode],
        "suite_sha256": suite_sha if suite_sha is not None else "a" * 64,
        "example_suite_used": example_used,
        "sampled_per_mount": sampled,
        "pool_total": pool_total,
        "canaries_passed": len(ids),
        "preservation_passed": pres_passed,
        "preservation_total": pres_total,
        "evaluated_canary_ids": ids,
        "transcript_sha256": transcript_sha if transcript_sha is not None else "b" * 64,
    }


def _external_ok_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "suite_sha256": "a" * 64,
        "preservation_passed": 4,
        "preservation_total": 4,
        "outcomes": [
            {"canary_id": f"p{i}", "category": "preservation", "passed": True, "failures": []}
            for i in range(1, 5)
        ]
        + [{"canary_id": "d1", "category": "domain", "passed": True, "failures": []}],
    }
    payload.update(overrides)
    return payload


class TestIssue16CanaryProvenance(unittest.TestCase):
    def test_hex_helper_and_suite_metadata_validation(self) -> None:
        self.assertTrue(_is_sha256_hex("0123456789abcdef" * 4))
        for bad in (
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            ("a" * 63) + "\n",
            123,
            None,
            _StrSub("a" * 64),
        ):
            self.assertFalse(_is_sha256_hex(bad), msg=f"expected invalid hex for {bad!r}")

        sealed = CanarySuite.sealed_placeholder("f" * 64)
        self.assertEqual(sealed.suite_sha256, "f" * 64)
        self.assertEqual(sealed.suite_role, "sealed_external_suite")
        self.assertFalse(sealed.example_only)
        self.assertEqual(sealed.canaries, ())
        with self.assertRaises(CanaryConfigError):
            CanarySuite.sealed_placeholder("not-a-sha")

        with tempfile.TemporaryDirectory() as td:
            base = json.loads(CANARY_SUITE_PATH.read_text(encoding="utf-8"))

            # Default metadata when omitted, plus whitespace-trimmed suite_role.
            trimmed = dict(base)
            trimmed["suite_role"] = "  custom_role  "
            trimmed["example_only"] = False
            p_ok = Path(td) / "ok.json"
            p_ok.write_text(json.dumps(trimmed), encoding="utf-8")
            os.chmod(p_ok, 0o600)
            loaded = CanarySuite.from_verifier_file(p_ok)
            self.assertEqual(loaded.suite_role, "custom_role")
            self.assertFalse(loaded.example_only)

            for bad_role in ("", "   ", 123, None, True):
                bad_doc = dict(base, suite_role=bad_role)
                p_bad = Path(td) / "bad_role.json"
                p_bad.write_text(json.dumps(bad_doc), encoding="utf-8")
                os.chmod(p_bad, 0o600)
                with self.assertRaises(CanaryConfigError):
                    CanarySuite.from_verifier_file(p_bad)

            for bad_ex in (1, 0, "true", None):
                bad_doc = dict(base, example_only=bad_ex)
                p_bad = Path(td) / "bad_ex.json"
                p_bad.write_text(json.dumps(bad_doc), encoding="utf-8")
                os.chmod(p_bad, 0o600)
                with self.assertRaises(CanaryConfigError):
                    CanarySuite.from_verifier_file(p_bad)

    def test_verify_canary_attestation_valid_and_fail_closed_matrix(self) -> None:
        # Minimal CanaryResult fallback computes pool_total and transcript_sha256 automatically.
        minimal_outcomes = tuple(
            CanaryCheckOutcome(canary_id=f"p{i}", category="preservation", passed=True, failures=())
            for i in range(4)
        )
        minimal = CanaryResult(
            suite_sha256="a" * 64,
            outcomes=minimal_outcomes,
            preservation_passed=4,
            preservation_total=4,
            canary_slot_flushed=True,
        )
        att_min = minimal.attestation_dict()
        self.assertEqual(att_min["pool_total"], 4)
        self.assertEqual(
            att_min["transcript_sha256"],
            _compute_transcript_sha256("a" * 64, CANARY_MODE_FIXED_SUITE, minimal_outcomes),
        )
        verified_min = verify_canary_attestation(att_min, expected_suite_sha256="a" * 64, min_preservation=4)
        self.assertEqual(verified_min, att_min)

        # Valid sampled pool and sealed external (with pres_total == canaries_passed boundary and both sampled flags).
        att_sampled = _valid_attestation(mode=CANARY_MODE_SAMPLED_POOL, sampled=True, pool_total=9)
        self.assertEqual(
            verify_canary_attestation(att_sampled, require_non_example=True, min_preservation=4),
            att_sampled,
        )
        for sampled_flag in (False, True):
            att_sealed = _valid_attestation(
                mode=CANARY_MODE_SEALED_EXTERNAL,
                sampled=sampled_flag,
                pool_total=4,
                evaluated_ids=["p1", "p2", "p3", "p4"],
                pres_passed=4,
                pres_total=4,
            )
            self.assertEqual(
                verify_canary_attestation(att_sealed, require_non_example=True, min_preservation=1),
                att_sealed,
            )

        # Invalid verifier configuration arguments fail closed with CanaryConfigError.
        with self.assertRaises(CanaryConfigError):
            verify_canary_attestation(att_min, min_preservation=0)
        with self.assertRaises(CanaryConfigError):
            verify_canary_attestation(att_min, min_preservation=True)  # type: ignore[arg-type]
        with self.assertRaises(CanaryConfigError):
            verify_canary_attestation(att_min, min_preservation=_IntSub(4))
        with self.assertRaises(CanaryConfigError):
            verify_canary_attestation(att_min, expected_suite_sha256="bad-hex")
        with self.assertRaises(CanaryConfigError):
            verify_canary_attestation(att_min, require_non_example=1)  # type: ignore[arg-type]

        # Non-mapping attestation fails closed.
        with self.assertRaises(CanaryFailedError):
            verify_canary_attestation(["not", "a", "mapping"])  # type: ignore[arg-type]

        # Example-only accepted when require_non_example=False, rejected when True.
        att_example = _valid_attestation(example_used=True)
        verify_canary_attestation(att_example, require_non_example=False)
        with self.assertRaises(CanaryFailedError):
            verify_canary_attestation(att_example, require_non_example=True)

        # Expected suite digest mismatch rejected.
        with self.assertRaises(CanaryFailedError):
            verify_canary_attestation(_valid_attestation(suite_sha="a" * 64), expected_suite_sha256="c" * 64)

        # Exhaustive tampered-field cases.
        bad_cases: list[tuple[str, dict[str, object]]] = [
            ("wrong_schema", {**_valid_attestation(), "schema_version": "warp.canary_attestation/v2"}),
            ("sub_schema", {**_valid_attestation(), "schema_version": _StrSub("warp.canary_attestation/v1")}),
            ("unknown_mode", {**_valid_attestation(), "evidence_mode": "mystery_mode"}),
            ("sub_mode", {**_valid_attestation(), "evidence_mode": _StrSub(CANARY_MODE_FIXED_SUITE)}),
            ("wrong_scope", {**_valid_attestation(), "evidence_scope": "proves universal safety"}),
            (
                "sub_scope",
                {**_valid_attestation(), "evidence_scope": _StrSub(EVIDENCE_SCOPE_BY_MODE[CANARY_MODE_FIXED_SUITE])},
            ),
            ("bad_suite_sha", {**_valid_attestation(), "suite_sha256": "Z" * 64}),
            ("bad_transcript_sha", {**_valid_attestation(), "transcript_sha256": "1234"}),
            ("non_bool_example", {**_valid_attestation(), "example_suite_used": 0}),
            ("non_bool_sampled", {**_valid_attestation(), "sampled_per_mount": 0}),
            ("ids_not_list", {**_valid_attestation(), "evaluated_canary_ids": ("p1", "p2", "p3", "p4", "d1")}),
            ("ids_empty", {**_valid_attestation(), "evaluated_canary_ids": []}),
            ("ids_non_str", {**_valid_attestation(), "evaluated_canary_ids": ["p1", "p2", "p3", "p4", 5]}),
            ("ids_blank", {**_valid_attestation(), "evaluated_canary_ids": ["p1", "p2", "p3", "p4", "   "]}),
            ("ids_sub_str", {**_valid_attestation(), "evaluated_canary_ids": ["p1", "p2", "p3", "p4", _StrSub("d1")]}),
            ("ids_dup", {**_valid_attestation(), "evaluated_canary_ids": ["p1", "p1", "p3", "p4", "d1"]}),
            ("canaries_passed_bool", {**_valid_attestation(), "canaries_passed": True}),
            ("pool_total_sub_int", {**_valid_attestation(), "pool_total": _IntSub(5)}),
            ("pres_passed_float", {**_valid_attestation(), "preservation_passed": 4.0}),
            ("pres_total_str", {**_valid_attestation(), "preservation_total": "4"}),
            ("count_mismatch", {**_valid_attestation(), "canaries_passed": 4}),
            ("pool_smaller", {**_valid_attestation(), "pool_total": 4}),
            ("fixed_claims_sampled", _valid_attestation(mode=CANARY_MODE_FIXED_SUITE, sampled=True, pool_total=5)),
            ("fixed_pool_gt_passed", _valid_attestation(mode=CANARY_MODE_FIXED_SUITE, sampled=False, pool_total=9)),
            ("sampled_claims_false", _valid_attestation(mode=CANARY_MODE_SAMPLED_POOL, sampled=False, pool_total=9)),
            ("sampled_pool_eq_passed", _valid_attestation(mode=CANARY_MODE_SAMPLED_POOL, sampled=True, pool_total=5)),
            ("pres_below_floor", _valid_attestation(pres_passed=3, pres_total=3)),
            (
                "pres_exceeds_total",
                _valid_attestation(
                    evaluated_ids=["p1", "p2", "p3", "p4"],
                    pool_total=4,
                    pres_passed=5,
                    pres_total=5,
                ),
            ),
            ("pres_incomplete", _valid_attestation(pres_passed=3, pres_total=4)),
        ]
        for label, tampered in bad_cases:
            with self.subTest(case=label):
                with self.assertRaises(CanaryFailedError):
                    verify_canary_attestation(tampered, min_preservation=4)

    def test_canary_runner_sampling_transcript_and_multislot_parity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pool_path = _write_private_pool_json(Path(td) / "pool.json")
            pool_suite = CanarySuite.from_verifier_file(pool_path)
            example_suite = CanarySuite.from_verifier_file(CANARY_SUITE_PATH, reject_world_writable=False)
            client = _InMemoryLlamaClient(slot_count=1)

            # Constructor guards.
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, disallow_example_suite=1)  # type: ignore[arg-type]
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, example_suite, disallow_example_suite=True)
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_preservation=3)
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_preservation=True)  # type: ignore[arg-type]
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_per_domain=0)
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_cross_adapter=0)

            b_corpus = AdapterBundle.from_dict(
                make_adapter_bundle_dict(adapter_id="mde:corpus", domain="corpus", llama_adapter_id=0)
            )
            b_formal = AdapterBundle.from_dict(
                make_adapter_bundle_dict(
                    adapter_id="mde:formal-verification",
                    domain="formal-verification",
                    llama_adapter_id=2,
                )
            )

            # 1. Fixed mode (no sampling) on private pool.
            fixed_runner = CanaryRunner(client, pool_suite, disallow_example_suite=True)
            res_fixed = fixed_runner.run([b_corpus])
            self.assertEqual(res_fixed.evidence_mode, CANARY_MODE_FIXED_SUITE)
            self.assertFalse(res_fixed.sampled_per_mount)
            self.assertEqual(res_fixed.pool_total, 11)
            self.assertEqual(len(res_fixed.outcomes), 11)
            self.assertFalse(res_fixed.example_suite_used)
            for out in res_fixed.outcomes:
                self.assertTrue(_is_sha256_hex(out.prompt_sha256))
                self.assertTrue(_is_sha256_hex(out.response_sha256))
            # system_prompt alters prompt_sha256, and response alters transcript_sha256.
            self.assertNotEqual(res_fixed.outcomes[0].prompt_sha256, res_fixed.outcomes[1].prompt_sha256)
            client_alt = _InMemoryLlamaClient(slot_count=1, response_text="391 theorem alt")
            res_fixed_alt = CanaryRunner(client_alt, pool_suite).run([b_corpus])
            self.assertNotEqual(res_fixed.transcript_sha256, res_fixed_alt.transcript_sha256)
            verify_canary_attestation(res_fixed.attestation_dict(), require_non_example=True, min_preservation=4)

            # 2. Multi-slot execution produces exact transcript & attestation parity with single-slot.
            multi_client = _InMemoryLlamaClient(slot_count=4)
            res_multi = CanaryRunner(multi_client, pool_suite).run([b_corpus])
            self.assertEqual(res_multi.attestation_dict(), res_fixed.attestation_dict())

            # 3. Sample preservation only (4 of 8) -> deterministic with seeded RNG, distinct across seeds.
            r_pres_a = CanaryRunner(client, pool_suite, sample_preservation=4, sample_rng=random.Random(11))
            r_pres_b = CanaryRunner(client, pool_suite, sample_preservation=4, sample_rng=random.Random(11))
            r_pres_c = CanaryRunner(client, pool_suite, sample_preservation=4, sample_rng=random.Random(99))
            out_a = r_pres_a.run([b_corpus])
            out_b = r_pres_b.run([b_corpus])
            out_c = r_pres_c.run([b_corpus])
            self.assertEqual(out_a.evidence_mode, CANARY_MODE_SAMPLED_POOL)
            self.assertTrue(out_a.sampled_per_mount)
            self.assertEqual(out_a.pool_total, 11)
            self.assertEqual(len(out_a.outcomes), 7)  # 4 pres + 3 domain
            self.assertEqual(out_a.preservation_total, 4)
            self.assertEqual(out_a.attestation_dict(), out_b.attestation_dict())
            self.assertNotEqual(
                out_a.attestation_dict()["evaluated_canary_ids"],
                out_c.attestation_dict()["evaluated_canary_ids"],
            )
            self.assertEqual(
                out_a.attestation_dict()["evaluated_canary_ids"][:4],
                sorted(out_a.attestation_dict()["evaluated_canary_ids"][:4]),
            )
            verify_canary_attestation(out_a.attestation_dict(), require_non_example=True, min_preservation=4)

            # 4. Sample per-domain only (1 of 3).
            r_dom = CanaryRunner(client, pool_suite, sample_per_domain=1, sample_rng=random.Random(7))
            out_dom = r_dom.run([b_corpus])
            self.assertEqual(out_dom.evidence_mode, CANARY_MODE_SAMPLED_POOL)
            self.assertEqual(out_dom.pool_total, 11)
            self.assertEqual(len(out_dom.outcomes), 9)  # 8 pres + 1 domain
            verify_canary_attestation(out_dom.attestation_dict(), require_non_example=True)

            # 5. Sample cross-adapter only (1 of 3) on 2-domain mount, and with empty cross on 1-domain mount.
            r_cross = CanaryRunner(client, pool_suite, sample_cross_adapter=1, sample_rng=random.Random(5))
            out_cross = r_cross.run([b_corpus, b_formal])
            self.assertEqual(out_cross.evidence_mode, CANARY_MODE_SAMPLED_POOL)
            self.assertEqual(out_cross.pool_total, 17)  # 8 + 3 + 3 + 3
            self.assertEqual(len(out_cross.outcomes), 15)  # 8 + 3 + 3 + 1
            verify_canary_attestation(out_cross.attestation_dict(), require_non_example=True)

            r_pres_and_cross = CanaryRunner(
                client,
                pool_suite,
                sample_preservation=4,
                sample_cross_adapter=1,
                sample_rng=random.Random(5),
            )
            out_single_with_cross = r_pres_and_cross.run([b_corpus])
            self.assertEqual(len(out_single_with_cross.outcomes), 7)

            # 6. Default CSPRNG path (sample_rng=None).
            r_csprng = CanaryRunner(client, pool_suite, sample_preservation=4, sample_per_domain=1)
            out_csprng = r_csprng.run([b_corpus])
            self.assertEqual(out_csprng.evidence_mode, CANARY_MODE_SAMPLED_POOL)
            self.assertEqual(len(out_csprng.outcomes), 5)

            # 7. Fail-closed sampling errors: pool smaller than requested OR zero subset entropy.
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_preservation=9).run([b_corpus])
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_per_domain=4).run([b_corpus])
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_cross_adapter=4).run([b_corpus, b_formal])
            with self.assertRaises(CanaryConfigError):
                CanaryRunner(client, pool_suite, sample_preservation=8).run([b_corpus])

    def test_external_command_runner_hardening_and_pinned_digest(self) -> None:
        client = _InMemoryLlamaClient()
        bundle = AdapterBundle.from_dict(
            make_adapter_bundle_dict(adapter_id="mde:corpus", domain="corpus", llama_adapter_id=0)
        )

        # Constructor validation.
        for bad_cmd in ("python3", b"python3", [], ["python3", ""], ["python3", "  "], ["python3", 1], [_StrSub("python3")]):
            with self.assertRaises(CanaryConfigError):
                ExternalCommandCanaryRunner(client, bad_cmd)  # type: ignore[arg-type]
        for bad_min in (0, -1, True, _IntSub(4)):
            with self.assertRaises(CanaryConfigError):
                ExternalCommandCanaryRunner(client, ["python3"], min_preservation=bad_min)  # type: ignore[arg-type]
        for bad_slot in (-1, True, _IntSub(0)):
            with self.assertRaises(CanaryConfigError):
                ExternalCommandCanaryRunner(client, ["python3"], canary_slot_id=bad_slot)  # type: ignore[arg-type]
        for bad_to in (0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "5", _IntSub(5)):
            with self.assertRaises(CanaryConfigError):
                ExternalCommandCanaryRunner(client, ["python3"], timeout_seconds=bad_to)  # type: ignore[arg-type]
        with self.assertRaises(CanaryConfigError):
            ExternalCommandCanaryRunner(client, ["python3"], suite_sha256_hint="bad")
        with self.assertRaises(CanaryConfigError):
            ExternalCommandCanaryRunner(client, ["python3"], expected_suite_sha256="bad")

        # Valid fractional timeout (0.5) and digest precedence (expected > hint > command hash).
        r_prec = ExternalCommandCanaryRunner(
            client,
            ["python3", "-c", "pass"],
            suite_sha256_hint="b" * 64,
            expected_suite_sha256="a" * 64,
            timeout_seconds=0.5,
            min_preservation=1,
            canary_slot_id=0,
        )
        self.assertEqual(r_prec.suite_sha256, "a" * 64)
        self.assertEqual(r_prec.timeout_seconds, 0.5)
        r_hint = ExternalCommandCanaryRunner(client, ["python3"], suite_sha256_hint="b" * 64, timeout_seconds=1)
        self.assertEqual(r_hint.suite_sha256, "b" * 64)
        r_hash = ExternalCommandCanaryRunner(client, ["python3", "verify.py"])
        self.assertEqual(r_hash.suite_sha256, hashlib.sha256(b"python3 verify.py").hexdigest())

        def _run_with_stdout(runner: ExternalCommandCanaryRunner, data: dict[str, object]) -> CanaryResult:
            completed = subprocess.CompletedProcess(args=list(runner.command), returncode=0, stdout=json.dumps(data), stderr="")
            with mock.patch("warp_modeld.canary.subprocess.run", return_value=completed):
                return runner.run([bundle])

        # Valid pinned run with prompt/response digests, sampled_per_mount, pool_total, and explicit transcript.
        pinned = ExternalCommandCanaryRunner(client, ["verifier-bin"], expected_suite_sha256="a" * 64)
        ok_data = _external_ok_payload(
            sampled_per_mount=True,
            pool_total=12,
            transcript_sha256="d" * 64,
        )
        ok_data["outcomes"][0]["prompt_sha256"] = "1" * 64  # type: ignore[index]
        ok_data["outcomes"][0]["response_sha256"] = "2" * 64  # type: ignore[index]
        res_ok = _run_with_stdout(pinned, ok_data)
        self.assertEqual(res_ok.evidence_mode, CANARY_MODE_SEALED_EXTERNAL)
        self.assertEqual(res_ok.suite_sha256, "a" * 64)
        self.assertTrue(res_ok.sampled_per_mount)
        self.assertEqual(res_ok.pool_total, 12)
        self.assertEqual(res_ok.transcript_sha256, "d" * 64)
        self.assertEqual(res_ok.outcomes[0].prompt_sha256, "1" * 64)
        self.assertEqual(res_ok.outcomes[0].response_sha256, "2" * 64)
        verify_canary_attestation(res_ok.attestation_dict(), expected_suite_sha256="a" * 64, require_non_example=True)

        # Alias canaries_sha256 + default transcript computation + unpinned fallback.
        unpinned = ExternalCommandCanaryRunner(client, ["verifier-bin"], suite_sha256_hint="e" * 64)
        alias_data = _external_ok_payload()
        del alias_data["suite_sha256"]
        alias_data["canaries_sha256"] = "c" * 64
        res_alias = _run_with_stdout(unpinned, alias_data)
        self.assertEqual(res_alias.suite_sha256, "c" * 64)
        self.assertEqual(res_alias.pool_total, 5)
        self.assertFalse(res_alias.sampled_per_mount)
        self.assertEqual(
            res_alias.transcript_sha256,
            _compute_transcript_sha256("c" * 64, CANARY_MODE_SEALED_EXTERNAL, res_alias.outcomes),
        )

        # Missing suite_sha256 uses hint when unpinned, but FAILS when expected_suite_sha256 is pinned.
        no_sha = _external_ok_payload()
        del no_sha["suite_sha256"]
        res_fallback = _run_with_stdout(unpinned, no_sha)
        self.assertEqual(res_fallback.suite_sha256, "e" * 64)
        with self.assertRaises(CanaryFailedError):
            _run_with_stdout(pinned, no_sha)

        # Fail-closed rejection cases for malformed external command outputs.
        bad_payloads: list[tuple[str, ExternalCommandCanaryRunner, dict[str, object]]] = [
            ("bad_suite_sha", unpinned, _external_ok_payload(suite_sha256="not-hex")),
            ("wrong_pinned_sha", pinned, _external_ok_payload(suite_sha256="b" * 64)),
            ("pres_total_bool", unpinned, _external_ok_payload(preservation_total=True)),
            ("pres_passed_bool", unpinned, _external_ok_payload(preservation_passed=True)),
            ("pres_passed_mismatch", unpinned, _external_ok_payload(preservation_passed=3)),
            ("sampled_non_bool", unpinned, _external_ok_payload(sampled_per_mount=1)),
            ("pool_total_bool", unpinned, _external_ok_payload(pool_total=True)),
            ("pool_total_too_small", unpinned, _external_ok_payload(pool_total=4)),
            ("bad_transcript_sha", unpinned, _external_ok_payload(transcript_sha256="BAD")),
        ]
        bad_prompt_payload = _external_ok_payload()
        bad_prompt_payload["outcomes"][0]["prompt_sha256"] = "short"  # type: ignore[index]
        bad_payloads.append(("bad_prompt_sha", unpinned, bad_prompt_payload))
        bad_resp_payload = _external_ok_payload()
        bad_resp_payload["outcomes"][0]["response_sha256"] = "short"  # type: ignore[index]
        bad_payloads.append(("bad_response_sha", unpinned, bad_resp_payload))

        for label, runner_obj, payload in bad_payloads:
            with self.subTest(case=label):
                with self.assertRaises(CanaryFailedError):
                    _run_with_stdout(runner_obj, payload)

    def test_config_builder_verifier_receipt_and_cli_wiring(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:18080")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pool_path = _write_private_pool_json(tmp / "pool.json")
            jsonl_path = _write_jsonl_suite(tmp / "suite.JSONL")
            pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()

            # Invalid top-level config / fields.
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config([], client)  # type: ignore[arg-type]
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config({"canary_slot_id": -1, "canary_suite_path": str(pool_path)}, client)
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config({"canary_max_tokens": 0, "canary_suite_path": str(pool_path)}, client)
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config(
                    {"canary_suite_path": str(pool_path), "canary_suite_sha256": "invalid"},
                    client,
                )
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config({}, client)
            for bad_path in ("", "   ", 123, _StrSub(str(pool_path))):
                with self.assertRaises(CanaryConfigError):
                    build_canary_runner_from_config({"canary_suite_path": bad_path}, client)

            # Sampled JSON suite config + disallow_example_canary_suite enforcement.
            suite_s, runner_s = build_canary_runner_from_config(
                {
                    "canary_suite_path": str(pool_path),
                    "canary_suite_sha256": pool_sha,
                    "canary_slot_id": 1,
                    "canary_max_tokens": 128,
                    "canary_sample_preservation": 4,
                    "canary_sample_per_domain": 1,
                    "canary_sample_cross_adapter": 1,
                    "disallow_example_canary_suite": True,
                },
                client,
            )
            self.assertIsInstance(runner_s, CanaryRunner)
            self.assertEqual(suite_s.suite_sha256, pool_sha)
            self.assertEqual(runner_s.canary_slot_id, 1)
            self.assertEqual(runner_s.max_tokens, 128)
            self.assertEqual(runner_s.sample_preservation, 4)
            self.assertEqual(runner_s.sample_per_domain, 1)
            self.assertEqual(runner_s.sample_cross_adapter, 1)

            ex_copy = tmp / "example-canaries.json"
            ex_copy.write_text(CANARY_SUITE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            os.chmod(ex_copy, 0o600)
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config(
                    {"canary_suite_path": str(ex_copy), "disallow_example_canary_suite": True},
                    client,
                )

            # JSONL suite path (case-insensitive .JSONL suffix).
            suite_j, runner_j = build_canary_runner_from_config({"canary_suite_path": str(jsonl_path)}, client)
            self.assertIsInstance(runner_j, CanaryRunner)
            self.assertEqual(len(suite_j.canaries), 5)

            # External command WITH canary_suite_path.
            suite_ec, runner_ec = build_canary_runner_from_config(
                {
                    "canary_suite_path": str(pool_path),
                    "canary_runner_command": [sys.executable, "verify.py"],
                    "canary_runner_timeout_seconds": 12.5,
                    "canary_min_preservation": 5,
                },
                client,
            )
            self.assertIsInstance(runner_ec, ExternalCommandCanaryRunner)
            self.assertEqual(suite_ec.suite_sha256, pool_sha)
            self.assertEqual(runner_ec.suite_sha256, pool_sha)
            self.assertIsNone(runner_ec.expected_suite_sha256)
            self.assertEqual(runner_ec.timeout_seconds, 12.5)
            self.assertEqual(runner_ec.min_preservation, 5)

            # External command WITHOUT canary_suite_path MUST pin canary_suite_sha256.
            with self.assertRaises(CanaryConfigError):
                build_canary_runner_from_config({"canary_runner_command": [sys.executable, "verify.py"]}, client)

            sealed_sha = fake_sha("sealed-verifier-vault-v1")
            suite_sealed, runner_sealed = build_canary_runner_from_config(
                {
                    "canary_runner_command": [sys.executable, "sealed.py"],
                    "canary_suite_sha256": sealed_sha,
                    "canary_slot_id": 2,
                },
                client,
            )
            self.assertIsInstance(runner_sealed, ExternalCommandCanaryRunner)
            self.assertEqual(suite_sealed.suite_role, "sealed_external_suite")
            self.assertFalse(suite_sealed.example_only)
            self.assertEqual(suite_sealed.suite_sha256, sealed_sha)
            self.assertEqual(runner_sealed.expected_suite_sha256, sealed_sha)
            self.assertEqual(runner_sealed.canary_slot_id, 2)

            # End-to-end WarpModeld mount receipt & signed ledger verification across example + sealed external modes.
            mock_srv = MockLlamaServer()
            url = mock_srv.start()
            try:
                modeld_ex = make_modeld(url, tmp / "ex_ledger")
                ident_ex = modeld_ex.identify_checkpoint()
                self.assertEqual(ident_ex["canary_suite_role"], "example_fixed_regression_suite")
                self.assertTrue(ident_ex["canary_suite_example_only"])

                res_ex = modeld_ex.mount_stack([make_adapter_bundle_dict(domain="corpus", llama_adapter_id=0)])
                att_ex = res_ex.receipt.details["canary_attestation"]
                self.assertEqual(att_ex["evidence_mode"], CANARY_MODE_FIXED_SUITE)
                self.assertTrue(att_ex["example_suite_used"])
                self.assertFalse(att_ex["sampled_per_mount"])
                verify_canary_attestation(att_ex, expected_suite_sha256=modeld_ex.canary_suite.suite_sha256)
                with self.assertRaises(CanaryFailedError):
                    verify_canary_attestation(att_ex, require_non_example=True)
                receipts = LedgerVerifier.verify_file(
                    tmp / "ex_ledger" / "ledger.jsonl",
                    HmacSha256Signer(b"test-verifier-key-32-bytes-00001", key_id="warp-test-hmac-v1"),
                )
                self.assertEqual(receipts[-1].details["canary_attestation"], att_ex)

                # Reset mock adapter scale for sealed external mount.
                mock_srv.state.adapters[0]["scale"] = 0.0
                ext_script = tmp / "sealed_verifier.py"
                ext_script.write_text(
                    "import json\n"
                    f"print(json.dumps({{\n"
                    f"    'suite_sha256': {sealed_sha!r},\n"
                    f"    'sampled_per_mount': True,\n"
                    f"    'pool_total': 16,\n"
                    f"    'preservation_passed': 4,\n"
                    f"    'preservation_total': 4,\n"
                    f"    'outcomes': [\n"
                    f"        {{'canary_id': f'sp-{{i}}', 'category': 'preservation', 'passed': True, 'failures': []}}\n"
                    f"        for i in range(4)\n"
                    f"    ] + [{{'canary_id': 'sd-1', 'category': 'domain', 'passed': True, 'failures': []}}],\n"
                    f"}}))\n",
                    encoding="utf-8",
                )
                sealed_client = LlamaServerClient(url, require_localhost=True)
                sealed_suite, sealed_runner = build_canary_runner_from_config(
                    {
                        "canary_runner_command": [sys.executable, str(ext_script)],
                        "canary_suite_sha256": sealed_sha,
                    },
                    sealed_client,
                )
                signer = HmacSha256Signer(b"test-verifier-key-32-bytes-00002", key_id="warp-test-hmac-v2")
                ledger_sealed = SignedLedger(tmp / "sealed_ledger" / "ledger.jsonl", signer)
                modeld_sealed = WarpModeld(
                    pinned_checkpoint=make_checkpoint(),
                    pinned_lineage=make_lineage(),
                    live_runtime_flags=make_runtime_flags(),
                    single_bounds=make_single_bounds(),
                    joint_bounds=make_joint_bounds(),
                    canary_suite=sealed_suite,
                    canary_runner_override=sealed_runner,
                    llama_client=sealed_client,
                    ledger=ledger_sealed,
                )
                ident_sealed = modeld_sealed.identify_checkpoint()
                self.assertEqual(ident_sealed["canary_suite_role"], "sealed_external_suite")
                self.assertFalse(ident_sealed["canary_suite_example_only"])
                res_sealed = modeld_sealed.mount_stack([make_adapter_bundle_dict(domain="corpus", llama_adapter_id=0)])
                att_sealed = res_sealed.receipt.details["canary_attestation"]
                self.assertEqual(att_sealed["evidence_mode"], CANARY_MODE_SEALED_EXTERNAL)
                self.assertTrue(att_sealed["sampled_per_mount"])
                self.assertEqual(att_sealed["pool_total"], 16)
                verify_canary_attestation(att_sealed, expected_suite_sha256=sealed_sha, require_non_example=True)
            finally:
                mock_srv.stop()

            # Verify cli._cmd_serve wires build_canary_runner_from_config into WarpModeld.canary_runner.
            key_file = tmp / "ledger.key"
            key_file.write_bytes(b"cli-wiring-test-key-32-bytes-001")
            cfg_path = tmp / "modeld-sealed.json"
            cfg_doc = {
                "socket_path": str(tmp / "control.sock"),
                "llama_server_url": "http://127.0.0.1:18080",
                "require_localhost": True,
                "canary_slot_id": 0,
                "canary_runner_command": [sys.executable, "sealed.py"],
                "canary_suite_sha256": sealed_sha,
                "ledger_path": str(tmp / "cli_ledger.jsonl"),
                "ledger_key_path": str(key_file),
                "checkpoint": make_checkpoint().to_dict(),
                "pinned_lineage": make_lineage().to_dict(),
                "runtime_flags": make_runtime_flags().to_dict(),
                "single_bounds": {
                    "max_frobenius_sq": {"num": 5, "den": 1},
                    "max_spectral_norm_sq": {"num": 2, "den": 1},
                    "max_leakage_energy": {"num": 1, "den": 100},
                    "max_safety_dot_sq": {"num": 1, "den": 1000},
                },
                "joint_bounds": {
                    "max_frobenius_sq": {"num": 20, "den": 1},
                    "max_spectral_norm_sq": {"num": 8, "den": 1},
                    "max_leakage_energy": {"num": 1, "den": 20},
                    "max_safety_dot_sq": {"num": 1, "den": 200},
                },
            }
            cfg_path.write_text(json.dumps(cfg_doc), encoding="utf-8")

            captured_modeld: list[WarpModeld] = []

            class _StubServer:
                def __init__(self, _sock: str, modeld: WarpModeld, socket_mode: int = 0o600) -> None:
                    captured_modeld.append(modeld)

                def start(self) -> None:
                    pass

                def stop(self) -> None:
                    pass

            class _DoneEvent:
                def is_set(self) -> bool:
                    return True

                def set(self) -> None:
                    pass

                def wait(self, timeout: float = 0.0) -> bool:
                    return True

            with (
                mock.patch.object(warp_cli, "WarpUnixRpcServer", _StubServer),
                mock.patch.object(warp_cli.signal, "signal"),
                mock.patch.object(warp_cli.threading, "Event", _DoneEvent),
            ):
                rc = warp_cli._cmd_serve(argparse.Namespace(config=str(cfg_path)))
            self.assertEqual(rc, 0)
            self.assertEqual(len(captured_modeld), 1)
            self.assertIsInstance(captured_modeld[0].canary_runner, ExternalCommandCanaryRunner)
            self.assertEqual(captured_modeld[0].canary_suite.suite_sha256, sealed_sha)


if __name__ == "__main__":
    unittest.main()

"""warp-modeld orchestrator uniting Stages 1 to 4 with O(1) rollback and hybrid state flush."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hmac
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Mapping, Sequence

from .canary import CanaryResult, CanaryRunner, CanarySuite, reject_adapter_supplied_canaries
from .certificate import (
    CertificateBounds,
    FixedRadixCertificate,
    StackCertificateResult,
    verify_stack_division_free,
)
from .errors import (
    CheckpointMismatchError,
    KillSwitchEngagedError,
    LatticeFlushError,
    StateFlushError,
    UnverifiedLoraOverrideError,
    WarpError,
    ZoneManifestDriftError,
)
from .fingerprint import (
    AdapterBundle,
    CheckpointFingerprint,
    Plane1Lineage,
    RuntimeFlags,
    Stage1Report,
    compute_file_sha256,
    sha256_json,
    validate_zone_manifest_id,
    verify_stage1,
    verify_zone_manifest_dict,
)
from .gguf_lora_cert import verify_adapter_file_against_certificate
from .ledger import LedgerReceipt, SignedLedger
from .llama_client import LlamaClientError, LlamaServerClient
from .live_probe import LiveAttestationReport, verify_live_server_attestation
from .state_banks import BankFlushReceipt, STATE_BANK_KV, STATE_BANK_LATTICE, STATE_BANK_SSM


@dataclass(frozen=True)
class MountResult:
    slot_manifest_digest: str
    active_adapters: tuple[dict[str, Any], ...]
    stage1_reports: tuple[Stage1Report, ...]
    stage2_result: StackCertificateResult
    stage3_result: CanaryResult
    slots_flushed: tuple[int, ...]
    receipt: LedgerReceipt
    bank_flush_receipts: tuple[BankFlushReceipt, ...] = ()
    live_attestation: LiveAttestationReport | None = None


@dataclass(frozen=True)
class RollbackResult:
    reason: str
    kill_switch_engaged: bool
    slots_flushed: tuple[int, ...]
    elapsed_ms: float
    receipt: LedgerReceipt
    bank_flush_receipts: tuple[BankFlushReceipt, ...] = ()


class WarpModeld:
    """L3 zero-trust inference gatekeeper between L4 exo and L2 llama-server."""

    def __init__(
        self,
        *,
        pinned_checkpoint: CheckpointFingerprint,
        pinned_lineage: Plane1Lineage,
        live_runtime_flags: RuntimeFlags,
        single_bounds: CertificateBounds,
        joint_bounds: CertificateBounds,
        canary_suite: CanarySuite,
        llama_client: LlamaServerClient,
        ledger: SignedLedger,
        canary_slot_id: int = 0,
        canary_max_tokens: int = 256,
        canary_runner_override: Any | None = None,
        pinned_zone_manifest_id: str | None = None,
        pinned_zone_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.pinned_checkpoint = pinned_checkpoint
        self.pinned_lineage = pinned_lineage
        self.live_runtime_flags = live_runtime_flags
        self.single_bounds = single_bounds
        self.joint_bounds = joint_bounds
        self.canary_suite = canary_suite
        self.llama_client = llama_client
        self.ledger = ledger
        self.canary_runner = (
            canary_runner_override
            if canary_runner_override is not None
            else CanaryRunner(
                llama_client,
                canary_suite,
                canary_slot_id=canary_slot_id,
                max_tokens=canary_max_tokens,
            )
        )

        self._zone_bank_tags: dict[str, int] = {}
        if pinned_zone_manifest is not None:
            verified_id, tags = verify_zone_manifest_dict(pinned_zone_manifest)
            if pinned_zone_manifest_id is not None:
                validate_zone_manifest_id(pinned_zone_manifest_id)
                if not hmac.compare_digest(verified_id, pinned_zone_manifest_id):
                    raise ZoneManifestDriftError(
                        f"pinned_zone_manifest_id ({pinned_zone_manifest_id}) != manifest_id ({verified_id})"
                    )
            self.pinned_zone_manifest_id: str | None = verified_id
            self._zone_bank_tags = tags
        elif pinned_zone_manifest_id is not None:
            self.pinned_zone_manifest_id = validate_zone_manifest_id(pinned_zone_manifest_id)
        else:
            self.pinned_zone_manifest_id = None

        self._active_bundles: list[AdapterBundle] = []
        self._slot_manifest_digest: str | None = None
        self._kill_switch_engaged: bool = False
        self._learning_enabled: bool = True
        self._state_quarantined: bool = False
        self._quarantine_error_cls: type[StateFlushError] = StateFlushError

    def identify_checkpoint(self) -> dict[str, Any]:
        with self._lock:
            return {
                "checkpoint": self.pinned_checkpoint.to_dict(),
                "checkpoint_digest": self.pinned_checkpoint.canonical_digest(),
                "pinned_lineage": self.pinned_lineage.to_dict(),
                "lineage_digest": self.pinned_lineage.canonical_digest(),
                "runtime_flags": self.live_runtime_flags.to_dict(),
                "runtime_flags_digest": self.live_runtime_flags.canonical_digest(),
                "canary_suite_sha256": self.canary_suite.suite_sha256,
                "pinned_zone_manifest_id": self.pinned_zone_manifest_id,
                "zone_bank_tags": dict(self._zone_bank_tags),
                "has_horizontal_lattice": self.pinned_checkpoint.has_horizontal_lattice,
                "required_state_banks": list(self.pinned_checkpoint.required_state_banks()),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            stale = self._is_active_stack_stale()
            zone_drift = self._is_active_zone_manifest_drifted()
            return {
                "active_adapters": [b.manifest_entry() for b in self._active_bundles],
                "slot_manifest_digest": self._slot_manifest_digest,
                "kill_switch_engaged": self._kill_switch_engaged,
                "learning_enabled": self._learning_enabled,
                "state_quarantined": self._state_quarantined,
                "quarantine_code": self._quarantine_error_cls.code if self._state_quarantined else None,
                "stale_lineage": stale,
                "zone_manifest_drift": zone_drift,
                "pinned_zone_manifest_id": self.pinned_zone_manifest_id,
                "zone_bank_tags": dict(self._zone_bank_tags),
                "has_horizontal_lattice": self.pinned_checkpoint.has_horizontal_lattice,
                "required_state_banks": list(self.pinned_checkpoint.required_state_banks()),
                "checkpoint_digest": self.pinned_checkpoint.canonical_digest(),
                "lineage_digest": self.pinned_lineage.canonical_digest(),
                "runtime_flags_digest": self.live_runtime_flags.canonical_digest(),
                "ledger_head_sha256": self.ledger.head_sha256,
                "ledger_count": self.ledger.count,
            }

    def _is_active_zone_manifest_drifted(self) -> bool:
        if not self._active_bundles:
            return False
        for bundle in self._active_bundles:
            if bundle.zone_manifest_id != self.pinned_zone_manifest_id:
                return True
        return False

    def _is_active_stack_stale(self) -> bool:
        if not self._active_bundles:
            return False
        if self.pinned_lineage.contested_heads:
            return True
        for bundle in self._active_bundles:
            if bundle.source_lineage.corpus_tree_sha256 != self.pinned_lineage.corpus_tree_sha256:
                return True
            for domain, commit in bundle.source_lineage.domain_commits.items():
                if self.pinned_lineage.domain_commits.get(domain) != commit:
                    return True
        return False

    def _compute_slot_manifest_digest(self, bundles: Sequence[AdapterBundle]) -> str:
        payload: dict[str, Any] = {
            "checkpoint_digest": self.pinned_checkpoint.canonical_digest(),
            "lineage_digest": self.pinned_lineage.canonical_digest(),
            "runtime_flags_digest": self.live_runtime_flags.canonical_digest(),
            "canary_suite_sha256": self.canary_suite.suite_sha256,
            "adapters": [b.manifest_entry() for b in bundles],
        }
        if self.pinned_zone_manifest_id is not None:
            payload["zone_manifest_id"] = self.pinned_zone_manifest_id
        return sha256_json(payload)

    def _zero_all_lora_scales(self) -> None:
        loaded = self.llama_client.list_lora_adapters()
        if loaded:
            self.llama_client.set_lora_adapters([{"id": item.id, "scale": 0.0} for item in loaded])

    def _flush_all_slots(self) -> tuple[tuple[int, ...], tuple[BankFlushReceipt, ...]]:
        """Erase every slot across all required state banks (KV, SSM, horizontal lattice).

        When `pinned_checkpoint.has_horizontal_lattice` is true, requires explicit atomic
        backend confirmation of BOTH `kv_cache` and `lattice_horizontal_streams` (plus
        `ssm_recurrent` if present), and verifies post-erase cleanliness across every bank.
        Any missing or failed horizontal-lattice confirmation raises LatticeFlushError
        (`ERR_LATTICE_FLUSH_FAILED`).
        """
        required_banks = self.pinned_checkpoint.required_state_banks()
        needs_lattice = STATE_BANK_LATTICE in required_banks
        needs_ssm = STATE_BANK_SSM in required_banks
        req_banks_arg = required_banks if len(required_banks) > 1 else None

        try:
            slots = self.llama_client.list_slots()
            flushed: list[int] = []
            confirmed_by_slot: dict[int, tuple[str, ...]] = {}

            if len(slots) <= 1:
                slot_resps = [(slot, self.llama_client.erase_slot(slot.id, banks=req_banks_arg)) for slot in slots]
            else:
                with ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    futs = [(slot, pool.submit(self.llama_client.erase_slot, slot.id, banks=req_banks_arg)) for slot in slots]
                    slot_resps = [(slot, fut.result()) for slot, fut in futs]

            for slot, resp in slot_resps:
                flushed.append(slot.id)
                raw_confirmed = resp.get("banks_flushed")
                if isinstance(raw_confirmed, list):
                    confirmed = tuple(str(b) for b in raw_confirmed)
                elif not needs_lattice and not needs_ssm:
                    confirmed = (STATE_BANK_KV,)
                else:
                    confirmed = ()

                if needs_lattice:
                    missing_banks = [b for b in required_banks if b not in confirmed]
                    if missing_banks:
                        raise LatticeFlushError(
                            f"slot {slot.id} erase failed to confirm required state banks "
                            f"{missing_banks} (confirmed={list(confirmed)})"
                        )
                confirmed_by_slot[slot.id] = confirmed

            # Re-verify zero residual recurrent state, KV cache tokens, or horizontal lattice taint.
            after = self.llama_client.list_slots()
            receipts: list[BankFlushReceipt] = []
            for slot in after:
                if needs_lattice:
                    lattice_bank = slot.state_banks.get(STATE_BANK_LATTICE)
                    if (
                        slot.lattice_state_tainted
                        or slot.lattice_layers_tainted > 0
                        or lattice_bank is None
                        or not lattice_bank.is_clean
                    ):
                        raise LatticeFlushError(
                            f"slot {slot.id} retained horizontal lattice state after erase "
                            f"(lattice_tainted={slot.lattice_state_tainted}, "
                            f"layers_tainted={slot.lattice_layers_tainted}, "
                            f"bank={lattice_bank.to_dict() if lattice_bank else None})"
                        )
                if needs_ssm:
                    ssm_bank = slot.state_banks.get(STATE_BANK_SSM)
                    if slot.ssm_state_tainted or (ssm_bank is not None and not ssm_bank.is_clean):
                        raise StateFlushError(
                            f"slot {slot.id} retained SSM recurrent state after erase "
                            f"(ssm_tainted={slot.ssm_state_tainted})"
                        )
                if (
                    slot.recurrent_state_tainted
                    or slot.active_lora_ids
                    or slot.n_decoded > 0
                    or slot.n_prompt_tokens_cache > 0
                ):
                    raise StateFlushError(
                        f"slot {slot.id} retained residual state after erase "
                        f"(tainted={slot.recurrent_state_tainted}, active={slot.active_lora_ids}, "
                        f"n_decoded={slot.n_decoded}, n_prompt_tokens_cache={slot.n_prompt_tokens_cache})"
                    )
                receipts.append(
                    BankFlushReceipt(
                        slot_id=slot.id,
                        banks_requested=required_banks,
                        banks_confirmed=confirmed_by_slot.get(slot.id, required_banks),
                        zone_manifest_id=self.pinned_zone_manifest_id,
                        verified_clean=True,
                    )
                )
            return tuple(flushed), tuple(receipts)
        except LlamaClientError as exc:
            if needs_lattice:
                raise LatticeFlushError(f"failed to flush horizontal lattice and KV slots: {exc}") from exc
            raise StateFlushError(f"failed to flush inference slots: {exc}") from exc

    def _record_reject(
        self,
        err: WarpError,
        *,
        adapters: Sequence[dict[str, Any]] = (),
        details: dict[str, Any] | None = None,
    ) -> LedgerReceipt:
        return self.ledger.append(
            "reject",
            checkpoint_fingerprint_sha256=self.pinned_checkpoint.canonical_digest(),
            runtime_flags_sha256=self.live_runtime_flags.canonical_digest(),
            slot_manifest_digest=self._slot_manifest_digest,
            adapters=adapters,
            stage_failed=err.stage,
            error_code=err.code,
            reason=str(err),
            details=details,
        )

    def mount_stack(
        self,
        raw_bundles: Sequence[dict[str, Any]],
        *,
        raw_composed_certificate: dict[str, Any] | None = None,
        expected_zone_manifest_id: str | None = None,
    ) -> MountResult:
        """Execute all 4 fail-closed stages before activating a single or multi-adapter stack."""
        with self._lock:
            ckpt_digest = self.pinned_checkpoint.canonical_digest()
            flags_digest = self.live_runtime_flags.canonical_digest()

            if self._state_quarantined:
                err = self._quarantine_error_cls("daemon is quarantined after prior slot flush failure")
                self._record_reject(err)
                raise err

            if self._kill_switch_engaged:
                err = KillSwitchEngagedError("kill switch is engaged; refusing adapter mount")
                self._record_reject(err)
                raise err

            if not self._learning_enabled:
                err = KillSwitchEngagedError("learning is disabled; refusing adapter mount")
                self._record_reject(err)
                raise err

            if not raw_bundles:
                err = WarpError("mount stack must contain at least one adapter", code="ERR_EMPTY_STACK", stage="stage1_fingerprint")
                self._record_reject(err)
                raise err

            if expected_zone_manifest_id is not None:
                try:
                    validate_zone_manifest_id(expected_zone_manifest_id)
                    if self.pinned_zone_manifest_id is None or not hmac.compare_digest(
                        expected_zone_manifest_id, self.pinned_zone_manifest_id
                    ):
                        raise ZoneManifestDriftError(
                            f"mount request zone_manifest_id {expected_zone_manifest_id} "
                            f"!= host pinned {self.pinned_zone_manifest_id}"
                        )
                except WarpError as err:
                    self._record_reject(err)
                    raise

            # Stage 0 / anti-forgery pre-check: forbid adapter-supplied canaries before parsing.
            try:
                reject_adapter_supplied_canaries(list(raw_bundles), path="bundles")
                if raw_composed_certificate is not None:
                    reject_adapter_supplied_canaries(raw_composed_certificate, path="composed_certificate")
            except WarpError as err:
                self._record_reject(err)
                raise

            # Parse bundles + Stage 1 (4-layer checkpoint + lineage + 5th runtime-flag digest + optional zone manifest).
            bundles: list[AdapterBundle] = []
            stage1_reports: list[Stage1Report] = []
            seen_llama_ids: set[int] = set()
            try:
                for raw in raw_bundles:
                    bundle = AdapterBundle.from_dict(raw)
                    if bundle.llama_adapter_id in seen_llama_ids:
                        raise WarpError(
                            f"duplicate llama_adapter_id {bundle.llama_adapter_id} in mount stack",
                            code="ERR_DUPLICATE_ADAPTER_ID",
                            stage="stage1_fingerprint",
                        )
                    seen_llama_ids.add(bundle.llama_adapter_id)
                    bundles.append(bundle)
                for bundle in bundles:
                    stage1_reports.append(
                        verify_stage1(
                            bundle,
                            self.pinned_checkpoint,
                            self.pinned_lineage,
                            self.live_runtime_flags,
                            pinned_zone_manifest_id=self.pinned_zone_manifest_id,
                        )
                    )
            except WarpError as err:
                self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                raise

            # Stage 1b: bind every Stage-1 verified adapter_path to the exact file loaded by
            # llama-server for that llama_adapter_id BEFORE Stage 2/3/4 can run.
            if any(b.adapter_path is not None for b in bundles):
                try:
                    loaded_stage1 = self.llama_client.list_lora_adapters()
                    self._verify_loaded_adapter_bindings(bundles, loaded_stage1)
                except WarpError as err:
                    self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                    raise

            # Stage 2: division-free exact integer FixedRadixCertificate gate (<1ms).
            try:
                composed_cert = (
                    FixedRadixCertificate.from_dict(raw_composed_certificate)
                    if raw_composed_certificate is not None
                    else None
                )
                cert_items = [
                    (b.certificate, b.adapter_id, b.adapter_sha256, b.scale_num, b.scale_den)
                    for b in bundles
                ]
                stage2_res = verify_stack_division_free(
                    cert_items,
                    self.single_bounds,
                    self.joint_bounds,
                    expected_base_weights_sha256=self.pinned_checkpoint.weights_sha256,
                    composed_certificate=composed_cert,
                )
                for b in bundles:
                    if b.adapter_path is not None:
                        verify_adapter_file_against_certificate(
                            b.adapter_path,
                            b.certificate,
                            expected_adapter_sha256=b.adapter_sha256,
                        )
            except WarpError as err:
                self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                raise

            # Live llama-server attestation: measure /props and on-disk base model digest
            # against daemon pins before running Stage 3 canaries or Stage 4 activation.
            try:
                live_props = self.llama_client.get_props()
                live_attestation = verify_live_server_attestation(
                    live_props,
                    self.pinned_checkpoint,
                    self.live_runtime_flags,
                )
            except WarpError as err:
                self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                raise

            # Stage 3: shadow-load and verifier-owned deterministic canary checks.
            try:
                currently_verified_ids = frozenset(b.llama_adapter_id for b in self._active_bundles)
                stage3_res = self.canary_runner.run(
                    bundles,
                    allowed_active_ids=currently_verified_ids,
                    required_state_banks=self.pinned_checkpoint.required_state_banks(),
                )
            except WarpError as err:
                self._record_reject(
                    err,
                    adapters=[b.manifest_entry() for b in bundles],
                    details={"canary_suite_sha256": self.canary_suite.suite_sha256},
                )
                raise

            # Stage 4 activation: flush all live slots across every required state bank so previous
            # KV, SSM, or horizontal lattice states cannot mix, then apply verified scales.
            try:
                flushed, bank_receipts = self._flush_all_slots()
                loaded = self.llama_client.list_lora_adapters()
                loaded_by_id = self._verify_loaded_adapter_bindings(bundles, loaded)
                for bundle in bundles:
                    if bundle.adapter_path is not None:
                        loaded_path = os.path.realpath(loaded_by_id[bundle.llama_adapter_id].path)
                        try:
                            actual_sha = compute_file_sha256(Path(loaded_path))
                        except OSError as exc:
                            raise CheckpointMismatchError(
                                f"adapter {bundle.adapter_id} file unreadable before Stage 4 activation: {exc}"
                            ) from exc
                        if not hmac.compare_digest(actual_sha, bundle.adapter_sha256):
                            raise CheckpointMismatchError(
                                f"adapter {bundle.adapter_id} file digest drifted before Stage 4 activation: "
                                f"disk {actual_sha} != manifest {bundle.adapter_sha256}"
                            )
                scale_by_id = {b.llama_adapter_id: b.scale_float for b in bundles}
                scale_payload = [
                    {"id": item.id, "scale": scale_by_id.get(item.id, 0.0)}
                    for item in loaded
                ]
                self.llama_client.set_lora_adapters(scale_payload)
            except CheckpointMismatchError as err:
                self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                raise
            except (LlamaClientError, StateFlushError) as exc:
                err = exc if isinstance(exc, StateFlushError) else (
                    LatticeFlushError(f"activation failed: {exc}")
                    if self.pinned_checkpoint.has_horizontal_lattice
                    else StateFlushError(f"activation failed: {exc}")
                )
                self._state_quarantined = True
                self._quarantine_error_cls = type(err)
                self._active_bundles = []
                self._slot_manifest_digest = None
                self._record_reject(err, adapters=[b.manifest_entry() for b in bundles])
                raise err from exc

            manifest_digest = self._compute_slot_manifest_digest(bundles)
            self._active_bundles = list(bundles)
            self._slot_manifest_digest = manifest_digest

            receipt = self.ledger.append(
                "mount",
                checkpoint_fingerprint_sha256=ckpt_digest,
                runtime_flags_sha256=flags_digest,
                slot_manifest_digest=manifest_digest,
                adapters=[b.manifest_entry() for b in bundles],
                reason="verified_mount",
                slots_flushed=flushed,
                details={
                    "stage2_elapsed_ns": stage2_res.elapsed_ns,
                    "stage3_elapsed_ms": round(stage3_res.elapsed_ms, 3),
                    "composed_bound_checked": stage2_res.composed_bound_checked,
                    "composed_cert_verified": stage2_res.composed_cert_verified,
                    "canary_suite_sha256": stage3_res.suite_sha256,
                    "preservation_passed": stage3_res.preservation_passed,
                    "preservation_total": stage3_res.preservation_total,
                    "canaries_passed": len(stage3_res.outcomes),
                    "zone_manifest_id": self.pinned_zone_manifest_id,
                    "required_state_banks": list(self.pinned_checkpoint.required_state_banks()),
                    "bank_flush_receipts": [r.to_dict() for r in bank_receipts],
                    "live_attestation": live_attestation.to_dict(),
                },
            )
            return MountResult(
                slot_manifest_digest=manifest_digest,
                active_adapters=tuple(b.manifest_entry() for b in bundles),
                stage1_reports=tuple(stage1_reports),
                live_attestation=live_attestation,
                stage2_result=stage2_res,
                stage3_result=stage3_res,
                slots_flushed=flushed,
                receipt=receipt,
                bank_flush_receipts=bank_receipts,
            )

    @staticmethod
    def _verify_loaded_adapter_bindings(
        bundles: Sequence[AdapterBundle],
        loaded: Sequence[Any],
    ) -> dict[int, Any]:
        """Fail closed unless llama-server loaded the exact Stage-1 verified adapter file(s)."""
        if not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes)):
            raise CheckpointMismatchError("llama-server returned non-sequence adapter listing")

        loaded_by_id: dict[int, Any] = {}
        for item in loaded:
            item_id = getattr(item, "id", None)
            if type(item_id) is not int or item_id < 0:
                raise CheckpointMismatchError(f"invalid loaded adapter id from llama-server: {item_id!r}")
            if item_id in loaded_by_id:
                raise CheckpointMismatchError(f"duplicate adapter id {item_id} reported by llama-server")
            loaded_by_id[item_id] = item

        for bundle in bundles:
            if bundle.llama_adapter_id not in loaded_by_id:
                raise CheckpointMismatchError(
                    f"adapter {bundle.adapter_id} (llama_adapter_id={bundle.llama_adapter_id}) "
                    "is not loaded in llama-server"
                )
            if bundle.adapter_path is None:
                continue

            loaded_path = getattr(loaded_by_id[bundle.llama_adapter_id], "path", None)
            if not isinstance(loaded_path, str) or not loaded_path.strip() or "\x00" in loaded_path:
                raise CheckpointMismatchError(
                    f"adapter {bundle.adapter_id} (llama_adapter_id={bundle.llama_adapter_id}) "
                    "has missing or invalid server-reported adapter path"
                )
            try:
                loaded_path.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CheckpointMismatchError(
                    f"adapter {bundle.adapter_id} (llama_adapter_id={bundle.llama_adapter_id}) "
                    "has non-UTF-8 server-reported adapter path"
                ) from exc

            resolved_bundle = os.path.realpath(bundle.adapter_path)
            resolved_server = os.path.realpath(loaded_path)
            try:
                bundle_stat = os.stat(resolved_bundle)
                server_stat = os.stat(resolved_server)
            except OSError as exc:
                raise CheckpointMismatchError(
                    f"adapter {bundle.adapter_id} (llama_adapter_id={bundle.llama_adapter_id}) "
                    f"server path verification failed: {exc}"
                ) from exc

            if not stat.S_ISREG(server_stat.st_mode) or not os.path.samestat(bundle_stat, server_stat):
                raise CheckpointMismatchError(
                    f"adapter {bundle.adapter_id} (llama_adapter_id={bundle.llama_adapter_id}) "
                    f"path mismatch: Stage 1 hashed {resolved_bundle!r}, llama-server loaded {resolved_server!r}"
                )

        return loaded_by_id

    def rollback(self, reason: str = "operator_rollback", *, engage_kill_switch: bool = False) -> RollbackResult:
        """O(1) kill-switch and explicit multi-bank (KV, SSM, horizontal lattice) state flush."""
        with self._lock:
            start = time.perf_counter()
            prior_adapters = [b.manifest_entry() for b in self._active_bundles]
            self._active_bundles = []
            self._slot_manifest_digest = None
            if engage_kill_switch:
                self._kill_switch_engaged = True

            event_name = "kill_switch" if engage_kill_switch else "rollback"
            ckpt_digest = self.pinned_checkpoint.canonical_digest()
            flags_digest = self.live_runtime_flags.canonical_digest()

            try:
                self._zero_all_lora_scales()
                flushed, bank_receipts = self._flush_all_slots()
            except (LlamaClientError, StateFlushError) as exc:
                err = exc if isinstance(exc, StateFlushError) else (
                    LatticeFlushError(f"rollback scale reset or horizontal lattice flush failed: {exc}")
                    if self.pinned_checkpoint.has_horizontal_lattice
                    else StateFlushError(f"rollback scale reset or slot flush failed: {exc}")
                )
                self._state_quarantined = True
                self._quarantine_error_cls = type(err)
                self._record_reject(
                    err,
                    adapters=prior_adapters,
                    details={
                        "rollback_reason": reason,
                        "kill_switch_engaged": engage_kill_switch,
                        "required_state_banks": list(self.pinned_checkpoint.required_state_banks()),
                    },
                )
                raise err from exc

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            receipt = self.ledger.append(
                event_name,
                checkpoint_fingerprint_sha256=ckpt_digest,
                runtime_flags_sha256=flags_digest,
                slot_manifest_digest=None,
                adapters=[],
                reason=reason,
                slots_flushed=flushed,
                details={
                    "unmounted_adapters": [a["adapter_id"] for a in prior_adapters],
                    "kill_switch_engaged": self._kill_switch_engaged,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "required_state_banks": list(self.pinned_checkpoint.required_state_banks()),
                    "bank_flush_receipts": [r.to_dict() for r in bank_receipts],
                },
            )
            return RollbackResult(
                reason=reason,
                kill_switch_engaged=self._kill_switch_engaged,
                slots_flushed=flushed,
                elapsed_ms=elapsed_ms,
                receipt=receipt,
                bank_flush_receipts=bank_receipts,
            )

    def reset_kill_switch(self, operator_confirmation: str) -> dict[str, Any]:
        with self._lock:
            if self._state_quarantined:
                raise self._quarantine_error_cls("cannot reset kill switch while daemon is quarantined")
            if not isinstance(operator_confirmation, str) or len(operator_confirmation.strip()) < 4:
                raise KillSwitchEngagedError("explicit operator confirmation string is required")
            self._kill_switch_engaged = False
            # Never auto-remount previous adapters; stay on clean W0 until fresh verification.
            return self.status()

    def update_pinned_lineage(self, new_lineage: Plane1Lineage) -> RollbackResult | None:
        """Update Plane 1 lineage pin; automatically unload stale or contested active stacks."""
        with self._lock:
            self.pinned_lineage = new_lineage
            if self._is_active_stack_stale():
                return self.rollback(reason="stale_lineage_auto_unload", engage_kill_switch=False)
            return None

    def update_zone_manifest(self, manifest_or_id: str | Mapping[str, Any]) -> RollbackResult | None:
        """Rotate pinned L5 zone-manifest/v0.1 digest; auto-unload and flush on bank-tag drift."""
        with self._lock:
            if isinstance(manifest_or_id, Mapping):
                new_id, new_tags = verify_zone_manifest_dict(manifest_or_id)
            else:
                new_id = validate_zone_manifest_id(manifest_or_id)
                new_tags = {}
            self.pinned_zone_manifest_id = new_id
            self._zone_bank_tags = new_tags
            if self._is_active_zone_manifest_drifted():
                return self.rollback(reason="zone_manifest_drift_auto_unload", engage_kill_switch=False)
            return None

    def set_learning_enabled(self, enabled: bool) -> RollbackResult | None:
        """Toggle parametric adapter mounting (mirrors .mep/learning off switch)."""
        with self._lock:
            self._learning_enabled = bool(enabled)
            if not self._learning_enabled and self._active_bundles:
                return self.rollback(reason="learning_disabled", engage_kill_switch=False)
            return None

    def infer(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        max_tokens: int = 256,
        raw_params: dict[str, Any] | None = None,
        expected_zone_manifest_id: str | None = None,
    ) -> dict[str, Any]:
        """Proxy chat completion through warp-modeld using only verified active LoRA state."""
        with self._lock:
            if self._state_quarantined:
                raise self._quarantine_error_cls("refusing inference while slot state is quarantined")
            if raw_params is not None and "lora" in raw_params:
                raise UnverifiedLoraOverrideError(
                    "callers may not pass raw 'lora' overrides to warp_infer; mount through Stage 1-4 verifier"
                )
            if expected_zone_manifest_id is not None:
                validate_zone_manifest_id(expected_zone_manifest_id)
                if self.pinned_zone_manifest_id is None or not hmac.compare_digest(
                    expected_zone_manifest_id, self.pinned_zone_manifest_id
                ):
                    raise ZoneManifestDriftError(
                        f"infer zone_manifest_id {expected_zone_manifest_id} "
                        f"!= host pinned {self.pinned_zone_manifest_id}"
                    )
            if self._is_active_zone_manifest_drifted():
                raise ZoneManifestDriftError(
                    "active adapter stack drifted from pinned zone_manifest_id; remount required"
                )
            active_lora = [
                {"id": b.llama_adapter_id, "scale": b.scale_float}
                for b in self._active_bundles
            ]
            manifest = self._slot_manifest_digest
            response = self.llama_client.chat_completion(
                messages,
                lora=active_lora,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
                cache_prompt=False,
            )
            return {
                "slot_manifest_digest": manifest,
                "zone_manifest_id": self.pinned_zone_manifest_id,
                "active_lora": active_lora,
                "response": response,
            }

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        n_predict: int = 64,
        n_probs: int = 0,
        raw_params: dict[str, Any] | None = None,
        expected_zone_manifest_id: str | None = None,
    ) -> dict[str, Any]:
        """Proxy raw /completion through warp-modeld using only verified active LoRA state."""
        with self._lock:
            if self._state_quarantined:
                raise self._quarantine_error_cls("refusing completion while slot state is quarantined")
            if raw_params is not None and "lora" in raw_params:
                raise UnverifiedLoraOverrideError(
                    "callers may not pass raw 'lora' overrides to warp_complete; mount through Stage 1-4 verifier"
                )
            if expected_zone_manifest_id is not None:
                validate_zone_manifest_id(expected_zone_manifest_id)
                if self.pinned_zone_manifest_id is None or not hmac.compare_digest(
                    expected_zone_manifest_id, self.pinned_zone_manifest_id
                ):
                    raise ZoneManifestDriftError(
                        f"complete zone_manifest_id {expected_zone_manifest_id} "
                        f"!= host pinned {self.pinned_zone_manifest_id}"
                    )
            if self._is_active_zone_manifest_drifted():
                raise ZoneManifestDriftError(
                    "active adapter stack drifted from pinned zone_manifest_id; remount required"
                )
            active_lora = [
                {"id": b.llama_adapter_id, "scale": b.scale_float}
                for b in self._active_bundles
            ]
            manifest = self._slot_manifest_digest
            response = self.llama_client.completion(
                prompt,
                lora=active_lora,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                n_predict=n_predict,
                n_probs=n_probs,
                cache_prompt=False,
            )
            return {
                "slot_manifest_digest": manifest,
                "zone_manifest_id": self.pinned_zone_manifest_id,
                "active_lora": active_lora,
                "response": response,
            }

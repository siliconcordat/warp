"""Stdlib HTTP client speaking the llama-server LoRA, slots, and chat completions API."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import shutil
import subprocess
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request

from .errors import WarpError
from .state_banks import STATE_BANK_KV, STATE_BANK_LATTICE, STATE_BANK_SSM, StateBankStatus

REQUIRED_LLAMA_SERVER_FLAGS = (
    "--lora",
    "--lora-scaled",
    "--lora-init-without-apply",
    "--slots",
    "--slot-save-path",
    "--no-cache-idle-slots",
    "--slot-prompt-similarity",
    "--ctx-checkpoints",
)


def find_missing_llama_server_flags(help_text: str) -> list[str]:
    """Return any required L3 safety/slot flags missing from `llama-server --help` output."""
    return [flag for flag in REQUIRED_LLAMA_SERVER_FLAGS if flag not in help_text]


def probe_llama_server_binary(binary: str = "llama-server", *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Run `<binary> --help` read-only and verify all required L3 flags are supported."""
    resolved = shutil.which(binary) if "/" not in binary else binary
    if not resolved:
        raise LlamaClientError(f"llama-server binary not found: {binary}")
    try:
        proc = subprocess.run(
            [resolved, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LlamaClientError(f"failed to execute {resolved} --help: {exc}") from exc

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    missing = find_missing_llama_server_flags(combined)
    if missing:
        raise LlamaClientError(
            f"llama-server binary at {resolved} lacks required L3 flags: {missing}"
        )
    return {
        "binary": resolved,
        "returncode": proc.returncode,
        "verified_flags": list(REQUIRED_LLAMA_SERVER_FLAGS),
    }


class LlamaClientError(WarpError):
    code = "ERR_LLAMA_BACKEND"
    stage = "backend"


@dataclass(frozen=True)
class LlamaAdapterInfo:
    id: int
    path: str
    scale: float


@dataclass(frozen=True)
class LlamaSlotInfo:
    id: int
    is_processing: bool
    n_decoded: int
    active_lora_ids: tuple[int, ...] = ()
    recurrent_state_tainted: bool = False
    n_prompt_tokens_cache: int = 0
    ssm_state_tainted: bool = False
    lattice_state_tainted: bool = False
    lattice_layers_tainted: int = 0
    state_banks: dict[str, StateBankStatus] = field(default_factory=dict)


class LlamaServerClient:
    """HTTP client targeting a local llama-server started with --lora-init-without-apply."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        require_localhost: bool = True,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise LlamaClientError(f"invalid llama-server URL: {base_url}")
        if require_localhost and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise LlamaClientError(
                f"refusing non-localhost llama-server endpoint {parsed.hostname} when require_localhost=True"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, body: Any | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            finally:
                exc.close()
            raise LlamaClientError(f"llama-server HTTP {exc.code} on {method} {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LlamaClientError(f"llama-server request failed on {method} {path}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        res = self._request("GET", "/health")
        if not isinstance(res, dict):
            raise LlamaClientError("unexpected /health payload from llama-server")
        return res

    def list_lora_adapters(self) -> list[LlamaAdapterInfo]:
        res = self._request("GET", "/lora-adapters")
        if not isinstance(res, list):
            raise LlamaClientError("unexpected /lora-adapters payload from llama-server")
        adapters: list[LlamaAdapterInfo] = []
        seen_ids: set[int] = set()
        for entry in res:
            if not isinstance(entry, dict) or "id" not in entry:
                raise LlamaClientError("malformed adapter item in /lora-adapters")
            raw_id = entry["id"]
            if type(raw_id) is not int or raw_id < 0:
                raise LlamaClientError(f"invalid adapter id in /lora-adapters: {raw_id!r}")
            if raw_id in seen_ids:
                raise LlamaClientError(f"duplicate adapter id in /lora-adapters: {raw_id}")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
                raise LlamaClientError(f"invalid adapter path in /lora-adapters for id {raw_id}")
            try:
                raw_path.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise LlamaClientError(f"invalid UTF-8 adapter path in /lora-adapters for id {raw_id}") from exc
            raw_scale = entry.get("scale", 0.0)
            if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
                raise LlamaClientError(f"invalid adapter scale in /lora-adapters for id {raw_id}")
            try:
                scale_val = float(raw_scale)
            except OverflowError as exc:
                raise LlamaClientError(f"out-of-range adapter scale in /lora-adapters for id {raw_id}") from exc
            if not math.isfinite(scale_val):
                raise LlamaClientError(f"non-finite adapter scale in /lora-adapters for id {raw_id}")
            seen_ids.add(raw_id)
            adapters.append(
                LlamaAdapterInfo(
                    id=raw_id,
                    path=raw_path,
                    scale=scale_val,
                )
            )
        return adapters

    def set_lora_adapters(self, adapters: list[dict[str, Any]]) -> dict[str, Any]:
        payload = [{"id": int(item["id"]), "scale": float(item["scale"])} for item in adapters]
        res = self._request("POST", "/lora-adapters", payload)
        return res if isinstance(res, dict) else {"result": res}

    def list_slots(self) -> list[LlamaSlotInfo]:
        res = self._request("GET", "/slots")
        if not isinstance(res, list):
            raise LlamaClientError("unexpected /slots payload from llama-server")
        slots: list[LlamaSlotInfo] = []
        for entry in res:
            if not isinstance(entry, dict) or "id" not in entry:
                raise LlamaClientError("malformed slot entry in /slots")
            active_ids = tuple(int(x) for x in entry.get("active_lora_ids", []))
            raw_n_decoded = entry.get("n_decoded")
            if raw_n_decoded is None:
                next_tok = entry.get("next_token")
                if isinstance(next_tok, list) and next_tok and isinstance(next_tok[0], dict):
                    raw_n_decoded = next_tok[0].get("n_decoded", 0)
                else:
                    raw_n_decoded = 0
            n_decoded_val = int(raw_n_decoded)
            n_cache_val = int(entry.get("n_prompt_tokens_cache", 0))
            rec_tainted = bool(entry.get("recurrent_state_tainted", False))
            ssm_tainted = bool(entry.get("ssm_state_tainted", False))
            lat_tainted = bool(entry.get("lattice_state_tainted", False))
            lat_layers = int(entry.get("lattice_layers_tainted", 0))

            parsed_banks: dict[str, StateBankStatus] = {}
            raw_banks = entry.get("state_banks")
            if isinstance(raw_banks, dict):
                for bank_name, bdata in raw_banks.items():
                    if not isinstance(bdata, dict):
                        continue
                    cap_tag = bdata.get("capability_tag")
                    parsed_banks[str(bank_name)] = StateBankStatus(
                        bank=str(bank_name),
                        tainted=bool(bdata.get("tainted", False)),
                        active_lora_ids=tuple(int(x) for x in bdata.get("active_lora_ids", [])),
                        tokens_or_layers=int(bdata.get("tokens_or_layers", 0)),
                        capability_tag=int(cap_tag) if cap_tag is not None and type(cap_tag) is int else None,
                        zone_manifest_id=(
                            str(bdata["zone_manifest_id"]) if bdata.get("zone_manifest_id") is not None else None
                        ),
                    )
            else:
                parsed_banks[STATE_BANK_KV] = StateBankStatus(
                    bank=STATE_BANK_KV,
                    tainted=rec_tainted,
                    active_lora_ids=active_ids,
                    tokens_or_layers=n_cache_val if n_cache_val > 0 else n_decoded_val,
                )
                if ssm_tainted:
                    parsed_banks[STATE_BANK_SSM] = StateBankStatus(
                        bank=STATE_BANK_SSM,
                        tainted=True,
                        active_lora_ids=active_ids,
                    )
                if lat_tainted or lat_layers > 0:
                    parsed_banks[STATE_BANK_LATTICE] = StateBankStatus(
                        bank=STATE_BANK_LATTICE,
                        tainted=lat_tainted or lat_layers > 0,
                        active_lora_ids=active_ids,
                        tokens_or_layers=lat_layers,
                    )

            slots.append(
                LlamaSlotInfo(
                    id=int(entry["id"]),
                    is_processing=bool(entry.get("is_processing", False)),
                    n_decoded=n_decoded_val,
                    active_lora_ids=active_ids,
                    recurrent_state_tainted=rec_tainted,
                    n_prompt_tokens_cache=n_cache_val,
                    ssm_state_tainted=ssm_tainted,
                    lattice_state_tainted=lat_tainted,
                    lattice_layers_tainted=lat_layers,
                    state_banks=parsed_banks,
                )
            )
        return slots

    def erase_slot(
        self,
        slot_id: int,
        *,
        banks: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if banks is not None:
            body = {"banks": [str(b) for b in banks], "atomic": True}
        res = self._request("POST", f"/slots/{int(slot_id)}?action=erase", body)
        return res if isinstance(res, dict) else {"result": res}

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        lora: list[dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        max_tokens: int = 256,
        cache_prompt: bool = False,
        id_slot: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": float(top_p),
            "seed": seed,
            "max_tokens": max_tokens,
            "cache_prompt": cache_prompt,
            "lora": [{"id": int(a["id"]), "scale": float(a["scale"])} for a in lora],
        }
        if id_slot is not None:
            payload["id_slot"] = int(id_slot)
        res = self._request("POST", "/v1/chat/completions", payload)
        if not isinstance(res, dict):
            raise LlamaClientError("unexpected /v1/chat/completions payload from llama-server")
        return res

    def completion(
        self,
        prompt: str,
        *,
        lora: list[dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        n_predict: int = 64,
        n_probs: int = 0,
        cache_prompt: bool = False,
        id_slot: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": float(top_p),
            "seed": seed,
            "n_predict": int(n_predict),
            "cache_prompt": cache_prompt,
            "lora": [{"id": int(a["id"]), "scale": float(a["scale"])} for a in lora],
        }
        if int(n_probs) > 0:
            payload["n_probs"] = int(n_probs)
        if id_slot is not None:
            payload["id_slot"] = int(id_slot)
        res = self._request("POST", "/completion", payload)
        if not isinstance(res, dict):
            raise LlamaClientError("unexpected /completion payload from llama-server")
        return res

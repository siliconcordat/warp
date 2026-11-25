# warp

`warp` is the Layer 3 operating system layer of the Siliconcordat stack.
It isolates AI agent tools inside Linux sandboxes, verifies local model adapters before they touch GPU memory, and gives Linux hosts single-reboot disk recovery.
It is built for operators running local open-weight models and coding agents who want hard system limits instead of prompt-only rules.

## Position in the Siliconcordat Stack

| Layer | Repository | Role |
| --- | --- | --- |
| L5 Improved MCP | `siliconcordat/mde` | Structured tool protocol and code editing seams |
| L4 Harness | `siliconcordat/exo` | Agent loop, memory log, and human approval gate |
| L3 OS | `siliconcordat/warp` | Sandboxed execution, verified model loading, and OS recovery |
| L2 Model | `siliconcordat/weft` | Open-weight adapters and deterministic adapter compilers |
| L1 Hardware | `siliconcordat/loom` | Hardware architecture research and theory |

## What Warp Provides

- **Verified model gatekeeper (`warp-modeld`):** A zero-dependency Python 3 standard-library service sitting between L4 `exo` and L2 `llama-server`. It checks model fingerprints, lineage hashes, runtime flags, optional L5 `zone-manifest/v0.1` bank-tag pins, exact integer math certificates (including native `siliconcordat/weft` manifest ingestion), and system-owned canary tests before activating any adapter.
- **Instant safe rollback:** Base weights stay on read-only storage. A kill switch sets adapter scales to zero, flushes every required slot state bank (`kv_cache`, `ssm_recurrent`, and `lattice_horizontal_streams` for Latent Lattice architectures), quarantines the slot on incomplete flush (`ERR_LATTICE_FLUSH_FAILED`), and writes per-bank signed receipts to an append-only JSONL ledger.
- **Mandatory tool sandboxes:** Declarative `Landlock`, `seccomp`, and `bubblewrap` blueprints confine agent tools inside an empty network namespace with read-only access to base weights and explicit denial of `warp-modeld` sockets, signing keys, and verifier canaries.
- **Recoverable Linux images (`mkosi` + `btrfs` + UKI):** Declarative Arch Linux profiles (`vmtest`, `compute-rtx3080`, `client-omarchy`) pair read-only `btrfs` root snapshots with archived Unified Kernel Images (UKI) for single-reboot recovery.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/warp_modeld/` | Four-stage fail-closed verifier (`fingerprint.py`, `certificate.py`, `canary.py`, `llama_client.py`, `state_banks.py`, `ledger.py`, `verifier.py`, `server.py`, `cli.py`) |
| `schemas/` | Adapter bundle JSON schema (`adapter-bundle.schema.json`), example fixed regression canary suite (`verifier-canaries.json`), and daemon config template (`modeld-config.example.json`) |
| `blueprints/mkosi/` | Base Arch `btrfs` + UKI config (zero `sudo`), post-install and ancestor mode-hardening scripts, and `vmtest`, `compute-rtx3080`, and `client-omarchy` profiles |
| `blueprints/systemd/` | Hardened service units for `llama-server` (`--lora-init-without-apply`), `warp-modeld`, and pre-upgrade checkpoints |
| `blueprints/sandbox/` | `Landlock` rules, default-deny `seccomp` filter, and fail-closed `bubblewrap` tool executor (`WARP_TOOL_EXECUTOR_REQUIRED=1`) |
| `blueprints/recovery/` | `btrfs` snapshot and archived UKI checkpoint/rollback helper (`warp-recovery.sh`) |
| `tests/` | Unit, end-to-end Unix socket daemon, blueprint validation, and adversarial tests against a local HTTP mock of `llama-server` |

## Running Tests and CLI Checks

Requires Python 3.11 or newer with no external runtime packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Verify a signed ledger file, archive and reseed a damaged ledger explicitly, run a division-free certificate check, probe an installed `llama-server` binary, or run the opt-in live GGUF Stage 1 to Stage 4 test:

```bash
PYTHONPATH=src python3 -m warp_modeld.cli verify-ledger --ledger <ledger.jsonl> --key-file <ledger.key>
PYTHONPATH=src python3 -m warp_modeld.cli repair-ledger --ledger <ledger.jsonl> --archive-dir <archive> --key-file <ledger.key> --confirm <phrase>
PYTHONPATH=src python3 -m warp_modeld.cli check-certificate --cert <cert.json> --bounds <bounds.json>
WARP_LIVE_LLAMA_TEST=1 PYTHONPATH=src python3 -m unittest tests/test_live_llama_smoke.py -v
WARP_LIVE_LLAMA_TEST=1 WARP_LIVE_BASE_GGUF=<model.gguf> WEFT_SRC=<weft/src> PYTHONPATH=src python3 -m unittest tests/test_live_llama_smoke.py -v
```

## Three-Machine Deployment Model

`warp` separates human control from experimental Linux compute so a failing driver or model test cannot break a primary workstation:

1. **Clean macOS Base Hub (64 GiB unified memory):** Runs stock macOS with zero OS changes. Holds human keys, source policies, and approval prompts. Can run read-only user-space `llama-server` on demand for 27B model checks.
2. **RTX 3080 Compute Host (8 GiB VRAM, 32 GiB RAM):** Runs a reversible dual-boot `warp` Linux image alongside Windows 11. Serves 4B to 8B open-weight models with multiple adapters and compiles adapters on CPU.
3. **Omarchy Linux Client (lightweight x64 laptop):** Runs the desktop shell, L4 `exo`, L5 `mde`, and local sandboxes on a dedicated test laptop (or inside a VM on shared hardware), sending large model requests over a private link to the compute host.

## Verification Boundary

- **Verified offline now (53 passing unit, RPC, adversarial, and blueprint tests plus 1 opt-in live GGUF test):** Covers all four `warp-modeld` stages, Unix socket RPC (`warp_infer`, `warp_complete`), and CLI subprocesses. Adversarial cases verify forged certificates, inflated scales, stale lineage, flag and zone-manifest drift, forged canaries, ledger tampering, and blueprint invariants.
- **Verified against simulated multi-bank Latent Lattice backend:** `tests/test_lattice_state_isolation.py` proves KV-only flush lets poisoned horizontal latents re-contaminate fresh KV writes, verifies `ERR_LATTICE_FLUSH_FAILED` quarantine on KV-only or lying backends, and confirms atomic multi-bank (`kv_cache` + `ssm_recurrent` + `lattice_horizontal_streams`) rollback with per-bank signed receipts.
- **Verified live on `llama-server` 0.5.0 (build 11146, commit `7fe450e19`, CPU mode on `Qwen2.5-1.5B-Instruct`):** Full Stage 1 to Stage 4 `warp-modeld` lifecycle over a `0600` Unix socket mounted a real `weft` rank-4 F16 GGUF v3 LoRA (`84,384` bytes on `blk.27.ffn_up.weight`, 0-byte re-derivation). Stage 2 ran in `16.71 us`, 5 live shadow canaries cleared in `4.90 s`, and kill-switch plus `POST /slots/0?action=erase` ran in `4.22 ms`, restoring bit-exact base `W0` output with verified HMAC receipts.
- **Requires Linux target hardware or real Lattice serving engine (not yet executed):** Building and booting `mkosi` disk images in QEMU, flashing dual-boot `compute-rtx3080` or `client-omarchy` laptops, live CUDA multi-adapter benchmarks on the 8 GiB RTX 3080 host, and live hardware execution on a real Latent Lattice checkpoint once a GGUF/C++ Lattice backend ships.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the zero-trust inference boundary, the four verification stages, the three-machine layout, and recovery rules.
- [PLAN.md](PLAN.md) lists milestones, acceptance gates, and non-goals.

## License

Code in this repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
Documentation is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

# Warp Architecture

`warp` enforces operating system boundaries beneath an AI agent harness (`siliconcordat/exo`) and above local model serving (`llama-server` running adapters from `siliconcordat/weft`).
Every boundary fails closed. When a check cannot run or returns an unexpected digest, `warp` blocks execution rather than falling back to an unsafe path.

## Trust Model

`warp` preserves a three-tier trust split across every local interface:

1. **Human authority stays outside model weights.** Approval signatures, source policies, and kill switches live on the control host and in OS-protected files. Model weights never acquire permission to approve actions or edit policies.
2. **Deterministic OS code enforces safety.** Sandboxes (`bubblewrap`, `Landlock`, `seccomp`) and the load verifier (`warp-modeld`) run as ordinary host software outside the model. Mounting an adapter never disables a sandbox, ledger check, or prose gate.
3. **Model adapters stay revocable caches.** Base model weights (`W0`) live on read-only storage. Adapters apply on top of `W0` at runtime and unload cleanly in constant time.

## Zero-Trust Inference Trusted Computing Base (TCB)

The L4 harness (`exo`) never writes to model weight files and never talks directly to the admin endpoints of `llama-server`.
Instead, `exo` connects over a local kernel-authenticated Unix domain socket to `warp-modeld`.

```text
exo (L4 Harness)
  |
  |  local Unix socket (W-MCP slot commands + inference proxy)
  v
warp-modeld (L3 Gatekeeper)
  |--> read-only base weights (W0) and verified F16 GGUF LoRA files
  |--> read-only OS-owned verifier canary suite
  |--> append-only signed JSONL receipt ledger
  |
  |  localhost HTTP only
  v
llama-server (L2 Serving Engine, started with --lora-init-without-apply)
```

Low-rank adapters export to F16 GGUF LoRA files so `llama-server` can compose multiple adapters in one forward pass using `--lora` and per-request `"lora": [{"id": ..., "scale": ...}]` arrays.
Sparse full-weight Safetensors diffs do not hot-swap directly inside `llama-server` without low-rank conversion or a separate Safetensors runtime.

## Four-Stage Fail-Closed Load Verifier (`warp-modeld`)

Before `warp-modeld` permits any single adapter or multi-adapter stack to serve traffic, the bundle must clear four ordered stages. Failure at any stage stops the load immediately and records a signed `reject` receipt.

### Stage 1: Checkpoint, Lineage, and Runtime-Flag Binding

Stage 1 checks five content digests on CPU before touching GPU memory:

1. **Weight layout digest (`layout_sha256`):** Confirms tensor names, shapes, and quantization layout match the expected base checkpoint.
2. **Weight bytes fingerprint:** Confirms the GGUF file SHA-256 (and base checkpoint provenance link) matches the pinned base model `W0`.
3. **Config and template digest:** Confirms `config`, tokenizer files, and `chat_template.jinja` SHA-256 values match the pinned base model.
4. **Source lineage digest (`source.lineage`):** Confirms the adapter was compiled from the currently pinned source policy tree (`corpus_tree_sha256` and domain source commits) with zero contested heads. Any newer source policy commit marks the adapter `stale_lineage` and rejects the mount.
5. **Runtime-flag digest:** Confirms the running `llama-server` binary commit, quantization format, context size (`--ctx-size`), KV cache quantization, checkpoint flags (`--ctx-checkpoints`, `--swa-checkpoints`), and RoPE or chat-template arguments match the adapter's verified serving profile.
6. **Optional L5 zone-manifest pinning (`zone-manifest/v0.1`):** When the host pins a canonical `zone_manifest_id` (`sha256:<64 hex>`) from `siliconcordat/mde`, every mounted adapter must carry that matching digest. Any 1-bit change to zone rules or L1 bank tags rotates `zone_manifest_id`, auto-unloads active adapters with an explicit slot flush, and rejects stale mounts or inference calls with `ERR_ZONE_MANIFEST_DRIFT`.

`warp-modeld` also provides native L2 `siliconcordat/weft` ingestion constructors (`FixedRadixCertificate.from_weft_dict` and `AdapterBundle.from_weft_binding`) that expand `radix_scale_exp2` into exact integer `1 << exp2` scales, verify internal manifest digests, and reject `bool` or `float` coercion before Stage 2 runs.

### Stage 2: Exact Integer `FixedRadixCertificate` Gate

Stage 2 runs in under 1 millisecond on CPU using exact integer arithmetic with no floating-point division.
Each adapter bundle carries a `FixedRadixCertificate` with scaled integer measurements produced after quantization:

- `frobenius_sq_scaled`: squared Frobenius norm bound of the quantized delta.
- `spectral_norm_sq_scaled`: squared spectral norm bound of the quantized delta.
- `leakage_energy_scaled`: squared leakage energy against the preserved base subspace (`K0`).
- `max_safety_dot_sq_scaled`: maximum squared projection onto safety preservation directions.

`warp-modeld` checks each scaled integer numerator against its allowed rational bound using integer cross-multiplication (`value * limit_den <= limit_num * radix_scale`).
Negative numbers, non-integers, invalid radix scales, or composed multi-adapter sums that breach joint bounds fail closed with zero calls to `llama-server`.

### Stage 3: Shadow Load and Verifier-Owned Canary Checks

`llama-server` starts with `--lora-scaled <path>:0.0` paired with `--lora-init-without-apply` and `--slot-save-path <dir>`.
Using explicit `:0.0` scales prevents `llama-server` 0.5.0 from activating unverified adapters at scale `1.0` on startup, and `--slot-save-path` enables `POST /slots/{id}?action=erase` to return HTTP 200 instead of HTTP 501.
`warp-modeld` loads canary prompts strictly from a read-only directory owned by the OS verifier. Any adapter manifest that attempts to supply inline canary prompts, expected outputs, or self-reported pass flags is rejected immediately.

On an isolated canary slot with freshly cleared KV and recurrent state, `warp-modeld` sends deterministic requests (`temperature=0`, fixed seed, prompt caching off) with the exact candidate `"lora"` scale array:

- **Preservation and safety canaries:** Must pass at 100 percent with no relaxation.
- **Per-adapter behavioral canaries:** Deterministic checks verifying the target domain habit (such as plain prose checks, UI module rules, or formal proof syntax).
- **Cross-adapter conflict canaries:** Prompts that mix domains when multiple adapters mount together (such as a formal proof block inside a plain prose explanation) to catch interference.

Only when every canary passes does `warp-modeld` publish the active `slot_manifest_digest` and write a signed `mount` receipt containing an explicit `canary_attestation` block (`warp.canary_attestation/v1`).
That receipt states the Stage 3 evidence mode (`fixed_suite_regression`, `verifier_sampled_pool`, or `sealed_external_command`), an explicit `evidence_scope` describing what a pass does and does not show, whether an example suite was used (`example_suite_used`), the exact evaluated canary IDs, pool size, and a canonical `transcript_sha256` binding evaluated prompt and response digests without storing prompt plaintext.
The checked-in `schemas/verifier-canaries.json` file is marked `"example_only": true` as a fixed regression and wiring suite; production deployments should supply a private verifier pool sampled per mount (`canary_sample_preservation` / `canary_sample_per_domain`) or wire an opaque `canary_runner_command` with a pinned `canary_suite_sha256` and `disallow_example_canary_suite: true`.

### Stage 4: Constant-Time Kill Switch and Multi-Bank State Flush

Modern model architectures carry up to three distinct cross-token state banks inside an active inference slot:

- **`kv_cache`:** Append-only key-value entries for full softmax attention layers.
- **`ssm_recurrent`:** Sequence-mixing recurrent states in hybrid models such as `Qwen3.8-27B` (16 full GQA layers plus 48 Gated DeltaNet layers) or Mamba-2 hybrids.
- **`lattice_horizontal_streams`:** Per-layer horizontal computational latent streams in Latent Lattice Language Models (Duggal et al., `arXiv:2611.21608`), where horizontal latents persist across timesteps alongside the vertical residual stream and directly condition attention queries, KV-cache writes, MLP layers, and decoding.

Setting adapter scales back to `0.0` stops new weight contributions, but stale recurrent state can still leak adapter trajectories.
On Latent Lattice checkpoints, flushing only `kv_cache` is also insufficient because surviving poisoned horizontal latents condition the next KV write and re-contaminate the fresh KV cache on the next token.

When a user triggers the kill switch, disables learning, rotates a zone manifest, or hits a runtime anomaly, `warp-modeld` executes a fail-closed multi-bank rollback sequence:

1. Marks the active adapter stack empty (`"lora": []`) so new requests route only to base weights `W0`.
2. Sends scale `0.0` updates to `POST /lora-adapters`.
3. Queries active slots (`GET /slots`) and sends `POST /slots/{id}?action=erase` across every required state bank (`kv_cache`, plus `ssm_recurrent` and `lattice_horizontal_streams` whenever declared on the pinned checkpoint).
4. Re-queries `GET /slots` to confirm every required bank is clean. On checkpoints with `has_horizontal_lattice: true`, Stage 1 refuses mounting unless `lattice_flush_supported: true` is present, and any missing backend confirmation or residual horizontal-lattice taint during rollback quarantines the daemon with `ERR_LATTICE_FLUSH_FAILED` instead of reporting success.
5. Appends a hash-chained, cryptographically signed `rollback` or `kill_switch` receipt carrying per-slot, per-bank flush receipts (`bank_flush_receipts`) to the JSONL ledger.

If a ledger file is truncated or tampered with on disk, `warp-modeld` refuses to start and never resets history automatically. An explicit root command (`warp-modeld repair-ledger`) preserves the broken ledger read-only with a SHA-256 checksum in an archive directory before initializing a fresh chain.

On the local Unix RPC socket, `warp-modeld` rejects duplicate JSON keys on every request frame and rejects any caller attempt to pass raw `"lora"` overrides during inference.

## Three-Machine Hardware Topology

All memory figures below use binary units (`GiB` = 1024^3 bytes, `MiB` = 1024^2 bytes).

| Machine | Hardware | OS and Runtime Role | Hard Boundary |
| --- | --- | --- | --- |
| Clean macOS Hub | Apple M1 Max, 64 GiB unified RAM | Stock macOS, human keys, policy custody, approval UI, and optional on-demand read-only `llama-server` for 27B checks (`Q4_K_M` ~15.24 GiB + three F16 LoRAs ~0.89 GiB + 32k context ~18.5 GiB total, never concurrent with 20 GiB Ollama weights) | Zero OS modifications, zero experimental daemons, zero untrusted sandboxes |
| Compute Host | ZenBook Pro Duo, Intel i9-11900H, 32 GiB RAM, RTX 3080 Laptop (8 GiB VRAM), 1 TB NVMe | Reversible dual-boot (`Windows 11` untouched on partition 1, `warp` Arch Linux `mkosi` image + CUDA on partition 2). Serves 4B to 8B models (`Qwen3-4B` or `Qwen3-8B` Q4_K_M + three LoRAs + Q8 KV cache within ~5.4 to 6.7 GiB VRAM) and runs CPU `float64` adapter builds in 32 GiB RAM | Holds zero org admin tokens, personal backups, or unapproved paid API keys |
| Omarchy Client | Lightweight x64 Arch + Omarchy/Hyprland laptop | Full `warp` `mkosi` image on a dedicated spare laptop, or QEMU/KVM VM / dual-boot partition if testing on shared hardware. Runs desktop shell, L4 `exo`, L5 `mde`, mandatory tool sandboxes, and optional small local CPU model | Never runs heavy GPU builds and never modifies live workstation services |

Trust flows strictly asymmetrically across the private authenticated link (Tailscale or WireGuard with mutual key authentication):

- **Mac hub to Linux hosts (downward):** Signed policy tree pins, exact-byte approval signatures, verifier canary digests, and kill-switch commands.
- **Linux hosts to Mac hub (upward):** Content-addressed evidence bundles, adapter manifests, and signed ledger heads. Linux hosts hold zero write access to the Mac hub.
- **Capability grants above transport:** Transport encryption alone (Tailscale, WireGuard, or future Iroh peers) authenticates endpoints but never grants broad host control. Cross-machine calls require scoped, expiring per-capability grants, durable `(principal, request_id)` replay rejection, and strict duplicate-key JSON validation.

## Declarative Imaging (`mkosi`) and `btrfs` + UKI Recovery

`warp` builds reproducible Arch Linux disk images with `mkosi` across three profiles: `vmtest` (QEMU validation first), `compute-rtx3080` (dual-boot CUDA host), and `client-omarchy` (Wayland desktop client).
Every profile omits `sudo` completely so privileged system actions run only through dedicated root helper sockets or a physical recovery console.
During image assembly, `harden-image-modes.sh` walks every ancestor directory along trusted policy, verifier, model, and unit paths and strips group and world write permissions (`root:root`, `go-w`) so an unprivileged user cannot swap verifier files via a writable parent directory.

Disk state splits across dedicated `btrfs` subvolumes:

- `@`: Root filesystem, snapshotted read-only before every kernel, driver, or image change.
- `@models`: Read-only base model weights (`W0`) and approved F16 GGUF LoRA bundles.
- `@store`, `@ledger`, `@egress`, `@sysd`: Persistent daemon state, append-only signed ledger, egress proxy config, and privileged action receipts.
- `@home`: User workspace data, kept untouched during root rollbacks.

Before any OS or driver upgrade, `warp-recovery` snapshots `@` read-only and archives the current Unified Kernel Image (a single signed EFI binary bundling kernel, initramfs, and command line). If an upgrade fails to boot or breaks GPU drivers, selecting the archived UKI entry restores the previous root snapshot in one reboot while leaving `@models`, `@ledger`, and `@home` intact.

## Mandatory Tool Sandbox Executor

Agent tool commands and verification jobs never run directly on the host user environment.
Every tool invocation passes through the mandatory executor (`WARP_TOOL_EXECUTOR_REQUIRED=1`), which refuses to run if kernel isolation is unavailable:

- **Namespace isolation (`bubblewrap`):** Runs inside an unprivileged user namespace with an empty network namespace (`CLONE_NEWNET`) by default, private `/tmp`, and scrubbed environment variables.
- **Filesystem rules (`Landlock`):** Grants read-write access only to an ephemeral `btrfs` copy-on-write worktree. Mounts base model weights strictly read-only. Explicitly denies access to verifier canary suites (`/etc/warp/canaries`), `warp-modeld` control sockets, ledger files, and host credentials.
- **Syscall filter (`seccomp`):** Applies a default-deny BPF filter allowing only the system calls needed by build and test tools.
- **Zone A vs Zone B split:** Deterministic builds, conformance checks, and CPU `float64` adapter compilation run in Zone A (no network, pinned toolchain digest, fixed `SOURCE_DATE_EPOCH`). Live GPU inference and external API calls run in Zone B with secret scanning and per-task log size caps.

#!/usr/bin/env bash
set -euo pipefail

# Mandatory fail-closed bubblewrap tool executor for siliconcordat/warp.
# Enforces empty network namespace (Zone A default), read-only base model mounts,
# and zero access to warp-modeld control sockets, signing keys, or verifier canaries.

if [[ "${WARP_TOOL_EXECUTOR_REQUIRED:-1}" != "1" ]]; then
    echo "ERR_SANDBOX_POLICY: WARP_TOOL_EXECUTOR_REQUIRED must remain 1" >&2
    exit 126
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ephemeral-worktree-dir> <command> [args...]" >&2
    exit 2
fi

WORKTREE="$1"
shift

if [[ ! -d "${WORKTREE}" ]]; then
    echo "ERR_SANDBOX_WORKTREE: worktree directory does not exist: ${WORKTREE}" >&2
    exit 126
fi

if ! command -v bwrap >/dev/null 2>&1; then
    echo "ERR_SANDBOX_UNAVAILABLE: bubblewrap (bwrap) is required; refusing unsandboxed execution" >&2
    exit 126
fi

BWRAP_ARGS=(
    --unshare-all
    --unshare-net
    --die-with-parent
    --new-session
    --clearenv
    --setenv PATH "/usr/local/bin:/usr/bin:/bin"
    --setenv LANG "C.UTF-8"
    --setenv SOURCE_DATE_EPOCH "${SOURCE_DATE_EPOCH:-1700000000}"
    --setenv WARP_SANDBOXED "1"
    --ro-bind /usr /usr
    --ro-bind /bin /bin
    --ro-bind /lib /lib
    --proc /proc
    --dev /dev
    --tmpfs /tmp
    --bind "${WORKTREE}" /workspace
    --chdir /workspace
)

if [[ -d /lib64 ]]; then
    BWRAP_ARGS+=(--ro-bind /lib64 /lib64)
fi

# Base weights stay strictly read-only if present.
# Verifier canary suites (/etc/warp/canaries), signing keys (/etc/warp/keys),
# and warp-modeld control sockets are intentionally excluded from agent tool sandboxes.
if [[ -d /var/lib/warp/models ]]; then
    BWRAP_ARGS+=(--ro-bind /var/lib/warp/models /var/lib/warp/models)
fi

exec bwrap "${BWRAP_ARGS[@]}" -- "$@"

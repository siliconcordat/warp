"""Static, syntax, and security invariant tests for mkosi, systemd, Landlock/seccomp, and recovery blueprints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from warp_modeld.canary import CanarySuite

from tests.helpers import REPO_ROOT


class TestBlueprintsAndHygiene(unittest.TestCase):
    def test_schemas_and_verifier_canary_suite(self) -> None:
        bundle_schema = json.loads((REPO_ROOT / "schemas" / "adapter-bundle.schema.json").read_text(encoding="utf-8"))
        self.assertFalse(bundle_schema["additionalProperties"])
        self.assertNotIn("canaries", bundle_schema["properties"])

        suite = CanarySuite.from_verifier_file(
            REPO_ROOT / "schemas" / "verifier-canaries.json",
            min_preservation=4,
            reject_world_writable=False,
        )
        self.assertEqual(suite.suite_role, "example_fixed_regression_suite")
        self.assertTrue(suite.example_only)
        self.assertGreaterEqual(sum(1 for c in suite.canaries if c.category == "preservation"), 4)
        self.assertGreaterEqual(sum(1 for c in suite.canaries if c.category == "cross_adapter"), 2)

        example_cfg = json.loads((REPO_ROOT / "schemas" / "modeld-config.example.json").read_text(encoding="utf-8"))
        self.assertTrue(example_cfg["require_localhost"])
        self.assertTrue(example_cfg["runtime_flags"]["lora_init_without_apply"])

    def test_landlock_and_seccomp_policies_fail_closed(self) -> None:
        landlock = json.loads((REPO_ROOT / "blueprints" / "sandbox" / "landlock-policy.json").read_text(encoding="utf-8"))
        self.assertEqual(landlock["default_action"], "deny")
        self.assertTrue(landlock["fail_closed_if_unsupported"])
        self.assertEqual(landlock["env_gate"], "WARP_TOOL_EXECUTOR_REQUIRED=1")
        self.assertNotIn("/etc/warp/canaries", landlock["rules"]["read_only"])
        self.assertIn("/var/lib/warp/models", landlock["rules"]["read_only"])
        self.assertIn("/run/warp-modeld", landlock["rules"]["explicit_deny"])
        self.assertIn("/var/lib/warp-ledger", landlock["rules"]["explicit_deny"])
        self.assertIn("/etc/warp/keys", landlock["rules"]["explicit_deny"])
        self.assertIn("/etc/warp/canaries", landlock["rules"]["explicit_deny"])
        self.assertFalse(landlock["network"]["zone_a_allow_network"])
        self.assertTrue(landlock["network"]["unshare_net_default"])

        sandbox_script = (REPO_ROOT / "blueprints" / "sandbox" / "warp-sandbox-exec.sh").read_text(encoding="utf-8")
        self.assertNotIn("--ro-bind /etc/warp/canaries", sandbox_script)
        postinst = (REPO_ROOT / "blueprints" / "mkosi" / "mkosi.postinst").read_text(encoding="utf-8")
        self.assertIn("chown root:warp-modeld /etc/warp/canaries /etc/warp/keys", postinst)
        self.assertIn("chmod 0750 /etc/warp/canaries /etc/warp/keys", postinst)
        harden = (REPO_ROOT / "blueprints" / "mkosi" / "harden-image-modes.sh").read_text(encoding="utf-8")
        self.assertIn("normalize_verifier_private_tree", harden)

        seccomp = json.loads(
            (REPO_ROOT / "blueprints" / "sandbox" / "seccomp-tool-filter.json").read_text(encoding="utf-8")
        )
        self.assertEqual(seccomp["default_action"], "SCMP_ACT_ERRNO")
        for dangerous in ("ptrace", "mount", "bpf", "kexec_load", "init_module"):
            self.assertIn(dangerous, seccomp["deny_explicit"])

    def test_systemd_units_enforce_zero_trust_invariants(self) -> None:
        llama_unit = (REPO_ROOT / "blueprints" / "systemd" / "warp-llama-server.service").read_text(encoding="utf-8")
        for required in (
            "--host 127.0.0.1",
            "--lora-scaled",
            "--lora-init-without-apply",
            "--slots",
            "--slot-save-path",
            "--no-cache-idle-slots",
            "ProtectSystem=strict",
            "ReadOnlyPaths=/var/lib/warp/models",
            "IPAddressDeny=any",
            "IPAddressAllow=localhost",
        ):
            self.assertIn(required, llama_unit)

        # --ctx-checkpoints and --swa-checkpoints are aliases in llama-server; pass only one active flag.
        active_lines = [line for line in llama_unit.splitlines() if not line.lstrip().startswith("#")]
        active_text = "\n".join(active_lines)
        self.assertIn("--ctx-checkpoints", active_text)
        self.assertNotIn("--swa-checkpoints", active_text)

        modeld_unit = (REPO_ROOT / "blueprints" / "systemd" / "warp-modeld.service").read_text(encoding="utf-8")
        for required in (
            "NoNewPrivileges=yes",
            "ProtectSystem=strict",
            "ReadOnlyPaths=/var/lib/warp/models /etc/warp/canaries",
            "IPAddressDeny=any",
            "IPAddressAllow=localhost",
            "MemoryDenyWriteExecute=yes",
        ):
            self.assertIn(required, modeld_unit)

    def test_mkosi_profiles_exist_and_enable_uki_btrfs_landlock(self) -> None:
        base_cfg = (REPO_ROOT / "blueprints" / "mkosi" / "mkosi.conf").read_text(encoding="utf-8")
        for required in ("UnifiedKernelImages=yes", "btrfs-progs", "bubblewrap", "lsm=landlock", "harden-image-modes.sh"):
            self.assertIn(required, base_cfg)

        # Hardening invariant: sudo must never be installed as a package in any mkosi profile.
        all_mkosi = [REPO_ROOT / "blueprints" / "mkosi" / "mkosi.conf"]
        for profile in ("vmtest", "compute-rtx3080", "client-omarchy"):
            prof_path = REPO_ROOT / "blueprints" / "mkosi" / "mkosi.profiles" / profile / "mkosi.conf"
            self.assertTrue(prof_path.is_file(), f"missing profile {profile}")
            all_mkosi.append(prof_path)

        for cfg_path in all_mkosi:
            for line in cfg_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                self.assertNotEqual(stripped, "sudo", msg=f"sudo must not be installed in {cfg_path}")

        rtx_cfg = (REPO_ROOT / "blueprints" / "mkosi" / "mkosi.profiles" / "compute-rtx3080" / "mkosi.conf").read_text(
            encoding="utf-8"
        )
        self.assertIn("nvidia-open", rtx_cfg)
        self.assertIn("cuda", rtx_cfg)

    def test_shell_scripts_syntax_and_fail_closed_sandbox_behavior(self) -> None:
        scripts = (
            REPO_ROOT / "blueprints" / "mkosi" / "mkosi.postinst",
            REPO_ROOT / "blueprints" / "mkosi" / "harden-image-modes.sh",
            REPO_ROOT / "blueprints" / "sandbox" / "warp-sandbox-exec.sh",
            REPO_ROOT / "blueprints" / "recovery" / "warp-recovery.sh",
        )
        for script in scripts:
            content = script.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", content)
            res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, msg=f"bash -n failed on {script}: {res.stderr}")

        # Verify warp-sandbox-exec.sh refuses to run when WARP_TOOL_EXECUTOR_REQUIRED=0.
        sandbox_script = REPO_ROOT / "blueprints" / "sandbox" / "warp-sandbox-exec.sh"
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ)
            env["WARP_TOOL_EXECUTOR_REQUIRED"] = "0"
            denied = subprocess.run(
                ["bash", str(sandbox_script), td, "echo", "should-not-run"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(denied.returncode, 126)
            self.assertIn("ERR_SANDBOX_POLICY", denied.stderr)

            # Verify warp-sandbox-exec.sh fails closed when bwrap is unavailable on PATH.
            env2 = {"PATH": "/usr/bin:/bin", "WARP_TOOL_EXECUTOR_REQUIRED": "1"}
            if subprocess.run(["which", "bwrap"], capture_output=True, check=False).returncode != 0:
                missing_bwrap = subprocess.run(
                    ["bash", str(sandbox_script), td, "echo", "should-not-run"],
                    capture_output=True,
                    text=True,
                    env=env2,
                    check=False,
                )
                self.assertEqual(missing_bwrap.returncode, 126)
                self.assertIn("ERR_SANDBOX_UNAVAILABLE", missing_bwrap.stderr)

    def test_public_repo_hygiene_no_local_paths_or_coauthored_trailers(self) -> None:
        forbidden = (
            "/" + "Users/",
            "/" + "home/",
            "Co-authored-" + "by:",
            "haf" + "iz",
        )
        for path in REPO_ROOT.rglob("*"):
            if ".git" in path.parts or "__pycache__" in path.parts or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in forbidden:
                self.assertNotIn(needle, text, msg=f"forbidden string {needle!r} found in {path}")


if __name__ == "__main__":
    unittest.main()

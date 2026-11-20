"""End-to-end daemon tests over Unix domain socket + CLI verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from warp_modeld.server import WarpUnixRpcClient, WarpUnixRpcServer

from tests.helpers import (
    CANARY_SUITE_PATH,
    REPO_ROOT,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_modeld,
    make_runtime_flags,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer


class TestEndToEndDaemon(unittest.TestCase):
    def test_unix_socket_lifecycle_and_adversarial_rpc(self) -> None:
        mock = MockLlamaServer()
        url = mock.start()
        key = b"test-verifier-key-32-bytes-00001"
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-modeld.sock"
                modeld = make_modeld(url, tmp, signer_key=key)
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    self.assertEqual(os.stat(sock_path).st_mode & 0o777, 0o600)
                    rpc = WarpUnixRpcClient(sock_path)

                    ident = rpc.call("wmcp_checkpoint_identify")
                    self.assertEqual(ident["result"]["checkpoint"]["model_id"], "Qwen3-8B-Q4_K_M")

                    # Mount 2-adapter stack over Unix socket.
                    bundles = [
                        make_adapter_bundle_dict(adapter_id="mde:corpus", domain="corpus", llama_adapter_id=0),
                        make_adapter_bundle_dict(
                            adapter_id="mde:formal-verification",
                            domain="formal-verification",
                            llama_adapter_id=2,
                        ),
                    ]
                    mounted = rpc.call("wmcp_slot_mount", {"bundles": bundles})
                    self.assertIn("result", mounted)
                    self.assertEqual(len(mounted["result"]["active_adapters"]), 2)

                    # Proxied inference uses verified active stack.
                    inf = rpc.call("warp_infer", {"messages": [{"role": "user", "content": "hello"}]})
                    self.assertEqual(
                        inf["result"]["slot_manifest_digest"],
                        mounted["result"]["slot_manifest_digest"],
                    )
                    self.assertIn("[ACTIVE_LORA:0,2]", inf["result"]["response"]["choices"][0]["message"]["content"])

                    # Proxied /completion uses verified active stack and rejects raw lora overrides.
                    comp = rpc.call("warp_complete", {"prompt": "raw completion check", "n_predict": 16})
                    self.assertIn("[ACTIVE_LORA:0,2]", comp["result"]["response"]["content"])
                    sneaky_comp = rpc.call(
                        "warp_complete",
                        {"prompt": "bypass", "lora": [{"id": 1, "scale": 1.0}]},
                    )
                    self.assertEqual(sneaky_comp["error"]["data"]["code"], "ERR_UNVERIFIED_LORA_OVERRIDE")

                    # Adversarial RPC: raw lora override rejected!
                    sneaky = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "bypass"}],
                            "lora": [{"id": 1, "scale": 1.0}],
                        },
                    )
                    self.assertIn("error", sneaky)
                    self.assertEqual(sneaky["error"]["data"]["code"], "ERR_UNVERIFIED_LORA_OVERRIDE")

                    # Adversarial RPC: duplicate JSON keys rejected before dispatch!
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw_sock:
                        raw_sock.settimeout(5.0)
                        raw_sock.connect(str(sock_path))
                        dup_wire = (
                            b'{"jsonrpc":"2.0","id":99,"method":"wmcp_slot_status",'
                            b'"params":{"a":1,"a":2}}\n'
                        )
                        raw_sock.sendall(dup_wire)
                        dup_resp = json.loads(raw_sock.recv(65536).decode("utf-8"))
                    self.assertIn("error", dup_resp)
                    self.assertIn("duplicate JSON key", dup_resp["error"]["message"])

                    # Adversarial RPC: forged canary rejected without disturbing active mount!
                    forged = make_adapter_bundle_dict()
                    forged["canaries"] = [{"prompt": "bypass"}]
                    bad_mount = rpc.call("wmcp_slot_mount", {"bundles": [forged]})
                    self.assertEqual(bad_mount["error"]["data"]["code"], "ERR_FORGED_CANARY_REJECTED")

                    # Kill switch over RPC flushes slots and reverts to clean W0.
                    ks = rpc.call("warp_kill_switch", {"reason": "rpc_emergency"})
                    self.assertTrue(ks["result"]["kill_switch_engaged"])
                    self.assertEqual(ks["result"]["slots_flushed"], [0, 1])

                    after_ks = rpc.call("warp_infer", {"messages": [{"role": "user", "content": "base check"}]})
                    text = after_ks["result"]["response"]["choices"][0]["message"]["content"]
                    self.assertIn("[BASE_W0]", text)
                    self.assertNotIn("RESIDUAL_RECURRENT_LEAK", text)

                    # Verify ledger via real CLI subprocess.
                    key_file = tmp / "ledger.key"
                    key_file.write_bytes(key)
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(REPO_ROOT / "src")
                    cli_res = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "warp_modeld.cli",
                            "verify-ledger",
                            "--ledger",
                            str(tmp / "ledger.jsonl"),
                            "--key-file",
                            str(key_file),
                            "--key-id",
                            "warp-test-hmac-v1",
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                    )
                    self.assertEqual(cli_res.returncode, 0, msg=cli_res.stderr)
                    cli_json = json.loads(cli_res.stdout)
                    self.assertTrue(cli_json["ok"])
                    self.assertEqual(cli_json["receipts_verified"], 3)  # mount, reject, kill_switch

                    # Verify check-certificate CLI subprocess too.
                    cert_file = tmp / "cert.json"
                    bounds_file = tmp / "bounds.json"
                    cert_file.write_text(json.dumps(make_adapter_bundle_dict()["certificate"]), encoding="utf-8")
                    b = make_single_bounds()
                    bounds_file.write_text(
                        json.dumps(
                            {
                                "max_frobenius_sq": b.max_frobenius_sq.to_dict(),
                                "max_spectral_norm_sq": b.max_spectral_norm_sq.to_dict(),
                                "max_leakage_energy": b.max_leakage_energy.to_dict(),
                                "max_safety_dot_sq": b.max_safety_dot_sq.to_dict(),
                            }
                        ),
                        encoding="utf-8",
                    )
                    cert_cli = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "warp_modeld.cli",
                            "check-certificate",
                            "--cert",
                            str(cert_file),
                            "--bounds",
                            str(bounds_file),
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                    )
                    self.assertEqual(cert_cli.returncode, 0, msg=cert_cli.stderr)
                    cert_json = json.loads(cert_cli.stdout)
                    self.assertTrue(cert_json["ok"])
                    self.assertLess(cert_json["elapsed_us"], 1000.0)
                finally:
                    server.stop()
        finally:
            mock.stop()

    def test_real_subprocess_daemon_serve_command(self) -> None:
        mock = MockLlamaServer()
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                key_file = tmp / "ledger.key"
                key_file.write_bytes(b"subprocess-test-key-32-bytes-001")
                os.chmod(key_file, 0o600)

                sock_path = tmp / "daemon.sock"
                ledger_path = tmp / "daemon-ledger.jsonl"
                sb = make_single_bounds()
                jb = make_joint_bounds()
                cfg = {
                    "socket_path": str(sock_path),
                    "socket_mode": "0600",
                    "llama_server_url": url,
                    "require_localhost": True,
                    "canary_slot_id": 0,
                    "require_adapter_file": False,
                    "canary_suite_path": str(CANARY_SUITE_PATH),
                    "ledger_path": str(ledger_path),
                    "ledger_key_path": str(key_file),
                    "ledger_key_id": "warp-subproc-v1",
                    "checkpoint": make_checkpoint().to_dict(),
                    "pinned_lineage": make_lineage().to_dict(),
                    "runtime_flags": make_runtime_flags().to_dict(),
                    "single_bounds": {
                        "max_frobenius_sq": sb.max_frobenius_sq.to_dict(),
                        "max_spectral_norm_sq": sb.max_spectral_norm_sq.to_dict(),
                        "max_leakage_energy": sb.max_leakage_energy.to_dict(),
                        "max_safety_dot_sq": sb.max_safety_dot_sq.to_dict(),
                    },
                    "joint_bounds": {
                        "max_frobenius_sq": jb.max_frobenius_sq.to_dict(),
                        "max_spectral_norm_sq": jb.max_spectral_norm_sq.to_dict(),
                        "max_leakage_energy": jb.max_leakage_energy.to_dict(),
                        "max_safety_dot_sq": jb.max_safety_dot_sq.to_dict(),
                    },
                }
                cfg_path = tmp / "modeld.json"
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

                env = dict(os.environ)
                env["PYTHONPATH"] = str(REPO_ROOT / "src")
                proc = subprocess.Popen(
                    [sys.executable, "-m", "warp_modeld.cli", "serve", "--config", str(cfg_path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    deadline = time.monotonic() + 5.0
                    while not sock_path.exists() and time.monotonic() < deadline:
                        self.assertIsNone(proc.poll(), "warp-modeld serve exited early")
                        time.sleep(0.05)
                    self.assertTrue(sock_path.exists(), "daemon socket was not created")

                    rpc = WarpUnixRpcClient(sock_path)
                    st = rpc.call("wmcp_slot_status")
                    self.assertEqual(st["result"]["active_adapters"], [])

                    mnt = rpc.call("wmcp_slot_mount", {"bundles": [make_adapter_bundle_dict()]})
                    self.assertEqual(len(mnt["result"]["active_adapters"]), 1)

                    unm = rpc.call("wmcp_slot_unmount", {"reason": "subproc_test_done"})
                    self.assertEqual(unm["result"]["slots_flushed"], [0, 1])
                finally:
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
                        try:
                            proc.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=2.0)
                            raise
                    if proc.stdout:
                        proc.stdout.close()
                    if proc.stderr:
                        proc.stderr.close()

                verify_res = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "warp_modeld.cli",
                        "verify-ledger",
                        "--ledger",
                        str(ledger_path),
                        "--key-file",
                        str(key_file),
                        "--key-id",
                        "warp-subproc-v1",
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )
                self.assertEqual(verify_res.returncode, 0, msg=verify_res.stderr)
                self.assertEqual(json.loads(verify_res.stdout)["receipts_verified"], 2)
        finally:
            mock.stop()


if __name__ == "__main__":
    unittest.main()

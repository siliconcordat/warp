"""Regression tests for strict JSON-RPC inference parameter validation in warp_modeld.server.

Covers:
  - Rejection of non-finite floats (temperature="NaN", top_p=1e999, raw JSON NaN/Infinity),
    out-of-range floats, booleans, strings, and unbounded integers (max_tokens=10**18, n_predict=-1,
    seed=-1 / 2**31, n_probs=-1 / 129) with JSON-RPC error -32602 (or -32700 for non-standard JSON)
    BEFORE any HTTP request reaches llama-server.
  - Preservation of valid inference/completion dispatches across exact inclusive boundaries.
  - Zero bare asserts; runs cleanly across -I -B, -I -O -B, and -I -B -W error.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from warp_modeld.server import (
    MAX_MAX_TOKENS,
    MAX_N_PREDICT,
    MAX_N_PROBS,
    MAX_SEED,
    MAX_TEMPERATURE,
    WarpUnixRpcClient,
    WarpUnixRpcServer,
    loads_strict_json,
)

from tests.helpers import make_modeld
from tests.mock_llama_server import MockLlamaServer


def _send_raw_line(sock_path: Path, raw_wire: bytes) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(str(sock_path))
        sock.sendall(raw_wire)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
    return json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))


class TestWarpRpcParamHardening(unittest.TestCase):
    def test_loads_strict_json_rejects_nan_and_infinity_constants(self) -> None:
        for raw in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
            with self.assertRaises(ValueError):
                loads_strict_json(raw)

    def test_warp_infer_and_complete_reject_non_finite_and_unbounded_params_without_llama_call(self) -> None:
        mock = MockLlamaServer()
        url = mock.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-modeld.sock"
                modeld = make_modeld(url, tmp)
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    rpc = WarpUnixRpcClient(sock_path)
                    # Record exact number of llama-server requests before adversarial probes
                    reqs_before = len(mock.state.request_log)

                    # 1. Non-finite / string / out-of-range / bool / huge-int probes on warp_infer
                    bad_infer_cases = [
                        {"messages": [{"role": "user", "content": "x"}], "temperature": "NaN"},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": -0.01},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": MAX_TEMPERATURE + 0.1},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": True},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": 0.0},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": -0.5},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": 1.01},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": "1.0"},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10**18},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": MAX_MAX_TOKENS + 1},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": -1},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": True},
                        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 256.5},
                        {"messages": [{"role": "user", "content": "x"}], "seed": -1},
                        {"messages": [{"role": "user", "content": "x"}], "seed": MAX_SEED + 1},
                        {"messages": [{"role": "user", "content": "x"}], "seed": False},
                    ]
                    for params in bad_infer_cases:
                        resp = rpc.call("warp_infer", params)
                        self.assertIn("error", resp, msg=f"expected error for {params!r}")
                        self.assertEqual(resp["error"]["code"], -32602, msg=f"params={params!r} resp={resp!r}")

                    # Raw JSON 1e999 overflow to +Inf -> -32602; non-standard NaN token -> -32700
                    inf_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":101,"method":"warp_infer","params":{"messages":[{"role":"user","content":"x"}],"top_p":1e999}}\n',
                    )
                    self.assertEqual(inf_resp["error"]["code"], -32602, msg=repr(inf_resp))

                    nan_wire_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":102,"method":"warp_infer","params":{"messages":[{"role":"user","content":"x"}],"temperature":NaN}}\n',
                    )
                    self.assertEqual(nan_wire_resp["error"]["code"], -32700, msg=repr(nan_wire_resp))

                    # 2. Adversarial probes on warp_complete
                    bad_complete_cases = [
                        {"prompt": "hi", "temperature": "Inf"},
                        {"prompt": "hi", "top_p": 0.0},
                        {"prompt": "hi", "n_predict": 10**18},
                        {"prompt": "hi", "n_predict": MAX_N_PREDICT + 1},
                        {"prompt": "hi", "n_predict": 0},
                        {"prompt": "hi", "n_predict": -1},
                        {"prompt": "hi", "n_predict": True},
                        {"prompt": "hi", "n_probs": -1},
                        {"prompt": "hi", "n_probs": MAX_N_PROBS + 1},
                        {"prompt": "hi", "n_probs": True},
                        {"prompt": "hi", "seed": -1},
                        {"prompt": "hi", "seed": MAX_SEED + 1},
                    ]
                    for params in bad_complete_cases:
                        resp = rpc.call("warp_complete", params)
                        self.assertIn("error", resp, msg=f"expected error for {params!r}")
                        self.assertEqual(resp["error"]["code"], -32602, msg=f"params={params!r} resp={resp!r}")

                    comp_inf_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":103,"method":"warp_complete","params":{"prompt":"hi","top_p":1e999}}\n',
                    )
                    self.assertEqual(comp_inf_resp["error"]["code"], -32602, msg=repr(comp_inf_resp))

                    # Confirm ZERO requests reached llama-server across all invalid probes
                    self.assertEqual(len(mock.state.request_log), reqs_before)

                    # 3. Valid dispatches at standard & boundary values must succeed
                    ok_inf = rpc.call(
                        "warp_infer",
                        {
                            "messages": [{"role": "user", "content": "valid"}],
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": MAX_SEED,
                            "max_tokens": 256,
                        },
                    )
                    self.assertIn("result", ok_inf, msg=repr(ok_inf))
                    reqs_after_inf = len(mock.state.request_log)
                    self.assertGreater(reqs_after_inf, reqs_before)

                    ok_comp = rpc.call(
                        "warp_complete",
                        {
                            "prompt": "valid",
                            "temperature": MAX_TEMPERATURE,
                            "top_p": 0.5,
                            "seed": 0,
                            "n_predict": MAX_N_PREDICT,
                            "n_probs": MAX_N_PROBS,
                        },
                    )
                    self.assertIn("result", ok_comp, msg=repr(ok_comp))
                    self.assertGreater(len(mock.state.request_log), reqs_after_inf)
                finally:
                    server.stop()
        finally:
            mock.stop()

    def test_zero_bare_asserts_in_server_and_tests(self) -> None:
        for rel in ("src/warp_modeld/server.py", "tests/test_rpc_param_hardening.py"):
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)], [])


if __name__ == "__main__":
    unittest.main()

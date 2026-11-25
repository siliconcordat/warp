"""Regression tests for strict JSON-RPC inference parameter validation in warp_modeld.server.

Covers:
  - Rejection of non-standard JSON constants (NaN/Infinity/-Infinity) and overflowing non-finite
    float literals (1e999/-1e999) anywhere in the JSON request (including id, params, and
    state-changing methods such as warp_kill_switch) at decode time with JSON-RPC error -32700
    BEFORE dispatch while keeping the Unix socket connection alive for pipelined requests.
  - Rejection of non-finite floats, 400-digit int float64 overflows, out-of-range floats, booleans,
    strings, and unbounded integers (max_tokens=10**18, n_predict=-1, seed=-1 / 2**31,
    n_probs=-1 / 129) with JSON-RPC error -32602 BEFORE any HTTP request reaches llama-server.
  - Defense-in-depth _send() fallback returning RFC-8259 JSON-RPC -32603 (never NaN/Infinity or
    0-byte drop) if an upstream response payload contains a non-finite float.
  - Preservation of valid inference/completion dispatches across exact inclusive boundaries.
  - Zero bare asserts in touched files; runs cleanly across -I -B, -I -O -B, and -I -B -W error.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

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
    _InvalidRpcParamsError,
    _parse_bounded_float,
    loads_strict_json,
)

from tests.helpers import make_modeld
from tests.mock_llama_server import MockLlamaServer


def _send_raw_lines(sock_path: Path, raw_wire: bytes, expected_replies: int = 1) -> list[dict]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(str(sock_path))
        sock.sendall(raw_wire)
        buf = b""
        while buf.count(b"\n") < expected_replies:
            data = sock.recv(65536)
            if not data:
                break
            buf += data
    lines = buf.splitlines()
    return [loads_strict_json(line.decode("utf-8")) for line in lines]


def _send_raw_line(sock_path: Path, raw_wire: bytes) -> dict:
    replies = _send_raw_lines(sock_path, raw_wire, expected_replies=1)
    if not replies:
        raise AssertionError("server closed connection without sending a JSON-RPC reply")
    return replies[0]


class TestWarpRpcParamHardening(unittest.TestCase):
    def test_loads_strict_json_rejects_nan_infinity_overflow_floats_and_duplicate_keys(self) -> None:
        for raw in (
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": -Infinity}',
            '{"x": 1e999}',
            '{"x": -1e999}',
            '{"id": 1e999, "method": "wmcp_slot_status"}',
            '{"a": 1, "a": 2}',
        ):
            with self.assertRaises(ValueError, msg=f"expected ValueError for {raw!r}"):
                loads_strict_json(raw)

        parsed = loads_strict_json('{"zero": 0.0, "subnormal": 5e-324, "large": 1e308}')
        self.assertEqual(parsed["zero"], 0.0)
        self.assertEqual(parsed["subnormal"], 5e-324)
        self.assertEqual(parsed["large"], 1e308)

    def test_parse_bounded_float_rejects_non_finite_and_int_overflow_directly(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf"), 10**400, -(10**400)):
            with self.assertRaises(_InvalidRpcParamsError, msg=f"expected rejection for {bad!r}"):
                _parse_bounded_float(bad, "temperature", lo=0.0, hi=MAX_TEMPERATURE)

    def test_non_finite_id_rejected_before_dispatch_and_connection_stays_open(self) -> None:
        mock_srv = MockLlamaServer()
        url = mock_srv.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-modeld.sock"
                modeld = make_modeld(url, tmp)
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    reqs_before = len(mock_srv.state.request_log)

                    # 1. Read-only method with overflowing float id gets -32700 reply (never 0 bytes)
                    status_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":1e999,"method":"wmcp_slot_status","params":{}}\n',
                    )
                    self.assertIn("error", status_resp, msg=repr(status_resp))
                    self.assertEqual(status_resp["error"]["code"], -32700, msg=repr(status_resp))
                    self.assertIsNone(status_resp["id"])

                    # 2. Inference method with -1e999 id rejected before dispatch (0 llama requests)
                    infer_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":-1e999,"method":"warp_infer","params":{"messages":[{"role":"user","content":"x"}]}}\n',
                    )
                    self.assertIn("error", infer_resp, msg=repr(infer_resp))
                    self.assertEqual(infer_resp["error"]["code"], -32700, msg=repr(infer_resp))
                    self.assertIsNone(infer_resp["id"])

                    # 3. State-changing warp_kill_switch with id=1e999 pipelined before valid wmcp_slot_status:
                    #    kill switch MUST NOT engage, reply 1 MUST be -32700, and pipelined reply 2 MUST arrive.
                    pipelined = (
                        b'{"jsonrpc":"2.0","id":1e999,"method":"warp_kill_switch","params":{"reason":"m1_probe"}}\n'
                        b'{"jsonrpc":"2.0","id":42,"method":"wmcp_slot_status","params":{}}\n'
                    )
                    replies = _send_raw_lines(sock_path, pipelined, expected_replies=2)
                    self.assertEqual(len(replies), 2, msg=repr(replies))
                    self.assertIn("error", replies[0], msg=repr(replies[0]))
                    self.assertEqual(replies[0]["error"]["code"], -32700, msg=repr(replies[0]))
                    self.assertIsNone(replies[0]["id"])
                    self.assertEqual(replies[1]["id"], 42, msg=repr(replies[1]))
                    self.assertIn("result", replies[1], msg=repr(replies[1]))
                    self.assertFalse(replies[1]["result"]["kill_switch_engaged"])

                    # Confirm ZERO requests reached llama-server across all non-finite id probes
                    self.assertEqual(len(mock_srv.state.request_log), reqs_before)
                finally:
                    server.stop()
        finally:
            mock_srv.stop()

    def test_send_fails_closed_on_non_finite_upstream_response_and_preserves_connection(self) -> None:
        mock_srv = MockLlamaServer()
        url = mock_srv.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                sock_path = tmp / "warp-modeld.sock"
                modeld = make_modeld(url, tmp)
                server = WarpUnixRpcServer(sock_path, modeld, socket_mode=0o600)
                server.start()
                try:
                    # Inject a non-finite float into a server result payload; _send must NOT emit NaN
                    # and must NOT drop the connection, instead replying -32603 and serving pipelined follow-up.
                    with mock.patch.object(modeld, "status", return_value={"poison": float("nan")}):
                        pipelined = (
                            b'{"jsonrpc":"2.0","id":7,"method":"wmcp_slot_status","params":{}}\n'
                            b'{"jsonrpc":"2.0","id":8,"method":"wmcp_checkpoint_identify","params":{}}\n'
                        )
                        replies = _send_raw_lines(sock_path, pipelined, expected_replies=2)
                    self.assertEqual(len(replies), 2, msg=repr(replies))
                    self.assertEqual(replies[0]["id"], 7, msg=repr(replies[0]))
                    self.assertIn("error", replies[0], msg=repr(replies[0]))
                    self.assertEqual(replies[0]["error"]["code"], -32603, msg=repr(replies[0]))
                    self.assertEqual(replies[1]["id"], 8, msg=repr(replies[1]))
                    self.assertIn("result", replies[1], msg=repr(replies[1]))
                finally:
                    server.stop()
        finally:
            mock_srv.stop()

    def test_warp_infer_and_complete_reject_non_finite_and_unbounded_params_without_llama_call(self) -> None:
        mock_srv = MockLlamaServer()
        url = mock_srv.start()
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
                    reqs_before = len(mock_srv.state.request_log)

                    # 1. Non-finite / string / out-of-range / bool / huge-int probes on warp_infer
                    bad_infer_cases = [
                        {"messages": [{"role": "user", "content": "x"}], "temperature": "NaN"},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": -0.01},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": MAX_TEMPERATURE + 0.1},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": 10**400},
                        {"messages": [{"role": "user", "content": "x"}], "temperature": True},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": 0.0},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": -0.5},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": 1.01},
                        {"messages": [{"role": "user", "content": "x"}], "top_p": 10**400},
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

                    # Raw JSON 1e999 overflow and non-standard NaN token both rejected at parse layer -> -32700
                    inf_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":101,"method":"warp_infer","params":{"messages":[{"role":"user","content":"x"}],"top_p":1e999}}\n',
                    )
                    self.assertEqual(inf_resp["error"]["code"], -32700, msg=repr(inf_resp))

                    nan_wire_resp = _send_raw_line(
                        sock_path,
                        b'{"jsonrpc":"2.0","id":102,"method":"warp_infer","params":{"messages":[{"role":"user","content":"x"}],"temperature":NaN}}\n',
                    )
                    self.assertEqual(nan_wire_resp["error"]["code"], -32700, msg=repr(nan_wire_resp))

                    # 2. Adversarial probes on warp_complete
                    bad_complete_cases = [
                        {"prompt": "hi", "temperature": "Inf"},
                        {"prompt": "hi", "temperature": -(10**400)},
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
                    self.assertEqual(comp_inf_resp["error"]["code"], -32700, msg=repr(comp_inf_resp))

                    # Confirm ZERO requests reached llama-server across all invalid probes
                    self.assertEqual(len(mock_srv.state.request_log), reqs_before)

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
                    reqs_after_inf = len(mock_srv.state.request_log)
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
                    self.assertGreater(len(mock_srv.state.request_log), reqs_after_inf)
                finally:
                    server.stop()
        finally:
            mock_srv.stop()

    def test_zero_bare_asserts_in_touched_files(self) -> None:
        for rel in ("src/warp_modeld/server.py", "tests/test_rpc_param_hardening.py"):
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)], [])


if __name__ == "__main__":
    unittest.main()

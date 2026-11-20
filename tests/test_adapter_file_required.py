"""A mount must fail closed when the adapter bytes were never hashed (Stage 1)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from warp_modeld.canary import CanarySuite
from warp_modeld.errors import CheckpointMismatchError
from warp_modeld.ledger import HmacSha256Signer, SignedLedger
from warp_modeld.llama_client import LlamaServerClient
from warp_modeld.verifier import WarpModeld

from tests.helpers import (
    CANARY_SUITE_PATH,
    make_adapter_bundle_dict,
    make_checkpoint,
    make_joint_bounds,
    make_lineage,
    make_runtime_flags,
    make_single_bounds,
)
from tests.mock_llama_server import MockLlamaServer


def _modeld(base_url: str, tmp: Path, **extra: object) -> WarpModeld:
    signer = HmacSha256Signer(b"test-verifier-key-32-bytes-00001", key_id="warp-test-hmac-v1")
    return WarpModeld(
        pinned_checkpoint=make_checkpoint(),
        pinned_lineage=make_lineage(),
        live_runtime_flags=make_runtime_flags(),
        single_bounds=make_single_bounds(),
        joint_bounds=make_joint_bounds(),
        canary_suite=CanarySuite.from_verifier_file(CANARY_SUITE_PATH),
        llama_client=LlamaServerClient(base_url, require_localhost=True),
        ledger=SignedLedger(tmp / "ledger.jsonl", signer),
        canary_slot_id=0,
        **extra,  # type: ignore[arg-type]
    )


class TestAdapterFileRequired(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.server = MockLlamaServer()
        self.url = self.server.start()

    def tearDown(self) -> None:
        self.server.stop()
        self._tmp.cleanup()

    def _bundle_with_file(self) -> dict:
        path = self.tmp / "adapter.gguf"
        path.write_bytes(b"adapter-bytes")
        return make_adapter_bundle_dict(
            adapter_path=str(path), adapter_sha256=hashlib.sha256(b"adapter-bytes").hexdigest()
        )

    def test_default_rejects_bundle_without_adapter_path_before_touching_server(self) -> None:
        modeld = _modeld(self.url, self.tmp)
        with self.assertRaises(CheckpointMismatchError):
            modeld.mount_stack([make_adapter_bundle_dict()])
        posts = [r for r in self.server.state.request_log if r["method"] == "POST"]
        self.assertEqual(posts, [])

    def test_default_accepts_bundle_whose_file_was_hashed(self) -> None:
        modeld = _modeld(self.url, self.tmp)
        res = modeld.mount_stack([self._bundle_with_file()])
        self.assertTrue(res.stage1_reports[0].adapter_file_verified)

    def test_explicit_opt_out_keeps_the_old_behaviour(self) -> None:
        modeld = _modeld(self.url, self.tmp, require_adapter_file=False)
        res = modeld.mount_stack([make_adapter_bundle_dict()])
        self.assertFalse(res.stage1_reports[0].adapter_file_verified)


if __name__ == "__main__":
    unittest.main()

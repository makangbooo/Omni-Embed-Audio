from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SshHostKeyVerificationTests(unittest.TestCase):
    def test_server_script_is_read_only_and_requires_sha256(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/print_ssh_host_fingerprints.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("/etc/ssh/ssh_host_*_key.pub", source)
        self.assertIn("ssh-keygen -E sha256 -lf", source)
        self.assertIn("VERIFY_RESULT=ED25519_MATCH", source)
        self.assertIn("VERIFY_RESULT=ED25519_MISMATCH", source)
        for forbidden in (
            "known_hosts",
            "ssh-keyscan",
            "StrictHostKeyChecking=no",
            "UserKnownHostsFile=/dev/null",
            "sudo",
            "> /etc/",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotRegex(source, r"(?m)^\s*rm(?:\s|$)")

    def test_approval_receipt_records_approval_without_execution(self) -> None:
        receipt = json.loads(
            (
                REPOSITORY_ROOT
                / "results/audits/takeover_approvals_20260729.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "recorded")
        self.assertFalse(receipt["remote_access"]["connect_in_this_turn"])
        self.assertEqual(len(receipt["approvals"]), 4)
        for approval in receipt["approvals"]:
            self.assertTrue(approval["approved"])
            self.assertFalse(approval["started"])


if __name__ == "__main__":
    unittest.main()

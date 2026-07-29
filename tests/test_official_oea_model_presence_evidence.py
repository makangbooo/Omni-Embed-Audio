from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/official_oea_model_presence_scan_20260729.json"
)
STATUS_PATH = REPOSITORY_ROOT / "docs/reproduction_status.md"


class OfficialOeaModelPresenceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_remote_run_identity_and_return_codes_are_fixed(self) -> None:
        run = self.audit["run"]
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(
            run["git_commit"],
            "513e078b195356d1d46f97828cc2e19a04640d7d",
        )
        self.assertEqual(
            run["metrics_sha256"],
            "123cd06970716619fa07e5f0ff36f37e53919893c38fec40f57b5a994ce5e9c1",
        )
        self.assertEqual(
            [run["scan_exit_code"], run["wrapper_exit_code"], run["caller_exit_code"]],
            [0, 0, 0],
        )
        self.assertTrue(run["gpu_disabled"])
        self.assertFalse(run["content_hashing_performed"])

    def test_all_nine_assets_are_only_presence_candidates(self) -> None:
        assets = self.audit["assets"]
        self.assertEqual(len(assets), 9)
        self.assertEqual(len({asset["name"] for asset in assets}), 9)
        self.assertTrue(all(asset["presence_candidate"] for asset in assets))
        self.assertTrue(all(asset["revision_marker_matches"] for asset in assets))
        self.assertEqual(self.audit["summary"]["presence_candidate_count"], 9)
        self.assertEqual(self.audit["summary"]["incomplete_file_count"], 0)
        self.assertEqual(self.audit["summary"]["symlink_count"], 0)

    def test_claim_boundary_does_not_promote_metadata_to_integrity(self) -> None:
        boundary = self.audit["claim_boundary"]
        self.assertEqual(boundary["source_label"], "OBSERVED")
        self.assertIn("content SHA256 correctness", boundary["unsupported_claims"])
        self.assertIn("authorization to run evaluation", boundary["unsupported_claims"])
        status = STATUS_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "results/audits/official_oea_model_presence_scan_20260729.json",
            status,
        )
        self.assertIn("metadata-only presence candidate", status)


if __name__ == "__main__":
    unittest.main()

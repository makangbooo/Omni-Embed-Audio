from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MECAT_AUDIT = ROOT / "results/audits/data04_data05_mecat_remote_20260729.json"
AUDIOCAPS_AUDIT = (
    ROOT / "results/audits/data06_data07_audiocaps_remote_20260729.json"
)
AUDIOCAPS_RESOURCE = ROOT / "configs/resources/data06_audiocaps_v2_metadata.json"


class TakeoverRemoteDataEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mecat = json.loads(MECAT_AUDIT.read_text(encoding="utf-8"))
        cls.audiocaps = json.loads(AUDIOCAPS_AUDIT.read_text(encoding="utf-8"))
        cls.audiocaps_resource = json.loads(
            AUDIOCAPS_RESOURCE.read_text(encoding="utf-8")
        )

    def test_mecat_remote_run_and_all_child_stages_completed(self) -> None:
        self.assertEqual(self.mecat["status"], "complete")
        self.assertEqual(self.mecat["execution"]["final_run_rc"], 0)
        self.assertEqual(self.mecat["data04"]["download_exit_code"], 0)
        self.assertEqual(self.mecat["data04"]["wrapper_exit_code"], 0)
        self.assertEqual(self.mecat["data05"]["validation_exit_code"], 0)
        self.assertEqual(self.mecat["data05"]["wrapper_exit_code"], 0)

    def test_mecat_848_samples_are_not_confused_with_marker_json(self) -> None:
        data05 = self.mecat["data05"]
        self.assertEqual(data05["manifest"]["rows"], 848)
        self.assertEqual(data05["archive_examples"], 848)
        self.assertEqual(data05["decoded_files"], 848)
        self.assertEqual(data05["recursive_flac_file_count"], 848)
        self.assertEqual(data05["recursive_json_file_count"], 849)
        self.assertIn("hidden .data05_mecat_extraction_complete.json marker", data05["recursive_json_count_reconciliation"])
        self.assertEqual(
            data05["manifest"]["sha256"],
            "b4c4d8c1c5928ba5b23a25b3169519871cb2438dfed05823610aebeccb8594a6",
        )

    def test_mecat_strict_paper_claim_remains_blocked(self) -> None:
        boundary = self.mecat["claim_boundary"]
        self.assertEqual(boundary["strict_paper_retrieval_status"], "BLOCKED")
        self.assertIn("[PAPER][MISSING]", boundary["paper_subset"])
        self.assertIn("[MISSING]", boundary["retrieval_caption"])

    def test_audiocaps_wrapper_is_complete_but_hash_gap_is_explicit(self) -> None:
        self.assertEqual(self.audiocaps["execution"]["final_run_rc"], 0)
        self.assertEqual(self.audiocaps["data06"]["status"], "complete")
        self.assertEqual(self.audiocaps["data07"]["status"], "complete")
        self.assertEqual(
            self.audiocaps["status"],
            "remote_execution_complete_artifact_hashes_pending",
        )
        pending = self.audiocaps["pending_evidence_registration"]
        self.assertIn("but not the child exit-code files", pending["reason"])
        self.assertEqual(len(pending["not_claimed_observed"]), 5)

    def test_audiocaps_expected_identities_match_resource_manifest(self) -> None:
        expected = self.audiocaps["data06"]["expected_source_identities"]
        for file_specification in self.audiocaps_resource["files"]:
            name = file_specification["name"]
            self.assertEqual(
                expected[name]["size_bytes"], file_specification["size_bytes"]
            )
            self.assertEqual(expected[name]["sha256"], file_specification["sha256"])


if __name__ == "__main__":
    unittest.main()

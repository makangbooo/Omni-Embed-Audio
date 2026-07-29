from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MECAT_AUDIT = ROOT / "results/audits/data04_data05_mecat_remote_20260729.json"
AUDIOCAPS_AUDIT = (
    ROOT / "results/audits/data06_data07_audiocaps_remote_20260729.json"
)
WAVCAPS_AUDIT = ROOT / "results/audits/data08_data09_wavcaps_remote_20260729.json"
MECAT_WAVCAPS_AUDIT = (
    ROOT / "results/audits/data11_mecat_wavcaps_remote_20260729.json"
)
NEMO_UIQ_PREFLIGHT = (
    ROOT / "results/audits/nemo3b_cl_uiq_gpu_preflight_20260729.json"
)
AUDIOCAPS_RESOURCE = ROOT / "configs/resources/data06_audiocaps_v2_metadata.json"


class TakeoverRemoteDataEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mecat = json.loads(MECAT_AUDIT.read_text(encoding="utf-8"))
        cls.audiocaps = json.loads(AUDIOCAPS_AUDIT.read_text(encoding="utf-8"))
        cls.wavcaps = json.loads(WAVCAPS_AUDIT.read_text(encoding="utf-8"))
        cls.mecat_wavcaps = json.loads(
            MECAT_WAVCAPS_AUDIT.read_text(encoding="utf-8")
        )
        cls.nemo_uiq_preflight = json.loads(
            NEMO_UIQ_PREFLIGHT.read_text(encoding="utf-8")
        )
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

    def test_audiocaps_remote_evidence_registration_is_complete(self) -> None:
        self.assertEqual(self.audiocaps["execution"]["final_run_rc"], 0)
        self.assertEqual(self.audiocaps["data06"]["status"], "complete")
        self.assertEqual(self.audiocaps["data07"]["status"], "complete")
        self.assertEqual(self.audiocaps["status"], "complete")
        self.assertEqual(self.audiocaps["data06"]["download_exit_code"], 0)
        self.assertEqual(self.audiocaps["data06"]["wrapper_exit_code"], 0)
        self.assertEqual(self.audiocaps["data07"]["validation_exit_code"], 0)
        self.assertEqual(self.audiocaps["data07"]["wrapper_exit_code"], 0)
        self.assertEqual(
            self.audiocaps["evidence_registration"]["status"], "complete"
        )

    def test_audiocaps_expected_identities_match_resource_manifest(self) -> None:
        expected = self.audiocaps["data06"]["source_identities"]
        for file_specification in self.audiocaps_resource["files"]:
            name = file_specification["name"]
            self.assertEqual(
                expected[name]["size_bytes"], file_specification["size_bytes"]
            )
            self.assertEqual(expected[name]["sha256"], file_specification["sha256"])

    def test_audiocaps_remote_manifests_are_fixed(self) -> None:
        manifests = self.audiocaps["data07"]["manifests"]
        self.assertEqual(manifests["test"]["rows"], 975)
        self.assertEqual(manifests["validation"]["rows"], 495)
        self.assertEqual(manifests["train_public_loader"]["rows"], 91254)
        self.assertEqual(manifests["train_repaired_bare_cr"]["rows"], 91254)
        for artifact in manifests.values():
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertGreater(artifact["size_bytes"], 0)

    def test_wavcaps_remote_evidence_registration_is_complete(self) -> None:
        self.assertEqual(self.wavcaps["status"], "complete")
        self.assertEqual(self.wavcaps["execution"]["final_run_rc"], 0)
        self.assertEqual(self.wavcaps["execution"]["elapsed_seconds"], 130)
        self.assertEqual(self.wavcaps["data08"]["download_exit_code"], 0)
        self.assertEqual(self.wavcaps["data08"]["wrapper_exit_code"], 0)
        self.assertEqual(self.wavcaps["data09"]["validation_exit_code"], 0)
        self.assertEqual(self.wavcaps["data09"]["wrapper_exit_code"], 0)
        self.assertEqual(self.wavcaps["data08"]["selected_files"], 8)
        self.assertEqual(self.wavcaps["data08"]["expected_total_bytes"], 176863095)
        self.assertEqual(self.wavcaps["data09"]["manifest"]["rows"], 403050)
        self.assertEqual(
            self.wavcaps["evidence_registration"]["status"], "complete"
        )

    def test_wavcaps_generated_artifact_identities_are_fixed(self) -> None:
        artifacts = {
            "manifest": (
                self.wavcaps["data09"]["manifest"],
                403050,
                "ad1c1ce7e6294c985398ab8dd1a4e66ac3021f392e6fc39fefa6c48de4bb3ba7",
            ),
            "audiocaps_test": (
                self.wavcaps["data09"]["blocklist_artifacts"]["audiocaps_test"],
                173,
                "77da39bcfa6283cff1fa59fdb0c012d16067e0dd6bfc3efdd2423ed4c1921a1d",
            ),
            "clotho_filename_matches": (
                self.wavcaps["data09"]["blocklist_artifacts"][
                    "clotho_filename_matches"
                ],
                638,
                "c40700763ce2094c909ee424cf3fa57b7057c2186bc2c0137f6313e7958c7cc4",
            ),
            "clotho_sound_id_filename_confirmed": (
                self.wavcaps["data09"]["blocklist_artifacts"][
                    "clotho_sound_id_filename_confirmed"
                ],
                611,
                "d6c51f269e8a81ef7b470d5f266595c6949573c6819928f930abc34fb6e7afb1",
            ),
            "clotho_conservative_candidates": (
                self.wavcaps["data09"]["blocklist_artifacts"][
                    "clotho_conservative_candidates"
                ],
                1017,
                "11c2a1d1f6e97f98f117dfb7c453419f554dd4bff4bdc724ee1c30ab88adb805",
            ),
        }
        for name, (artifact, expected_rows, expected_sha256) in artifacts.items():
            with self.subTest(name=name):
                self.assertEqual(artifact["rows"], expected_rows)
                self.assertEqual(artifact["sha256"], expected_sha256)
                self.assertGreater(artifact["size_bytes"], 0)

    def test_wavcaps_paper_protocol_conflict_remains_explicit(self) -> None:
        boundary = self.wavcaps["data09"]["claim_boundary"]
        self.assertIn("[INFERRED]", boundary["count_equivalent_duration_protocol"])
        self.assertIn("[PAPER]", boundary["paper_written_duration_protocol"])
        self.assertIn("[MISSING]", boundary["exact_post_blocklist_manifest"])
        self.assertFalse(
            boundary["metadata_only_conservative_reconstruction_is_paper_exact"]
        )

    def test_data11_canonical_join_is_complete_with_fixed_candidates(self) -> None:
        audit = self.mecat_wavcaps
        self.assertEqual(audit["status"], "complete")
        self.assertEqual(audit["execution"]["call_exit_code"], 0)
        self.assertEqual(audit["execution"]["audit_exit_code"], 0)
        self.assertEqual(audit["execution"]["wrapper_exit_code"], 0)
        self.assertEqual(audit["results"]["mecat_examples"], 848)
        self.assertEqual(audit["results"]["mecat_unique_source_videos"], 807)
        self.assertEqual(audit["results"]["source_video_overlap_count"], 4)
        self.assertEqual(
            audit["results"]["source_video_overlap_ids"],
            ["FQIZHO6l0IY", "Nw2EarZypA0", "qEfTLLEpojc", "vzt3AXNeKIQ"],
        )
        self.assertEqual(
            audit["artifacts"]["statistics"]["sha256"],
            "1e89bf6031c7899e095884e713a890c067d56056318f4f194d897b73fdda630a",
        )
        candidates = audit["artifacts"]["source_video_candidates"]
        self.assertEqual(candidates["rows"], 4)
        self.assertEqual(candidates["size_bytes"], 1873)
        self.assertEqual(
            candidates["sha256"],
            "d0f50a93c9b05f6bcaaaa94959c2c5c3a8e2b17aa8107960d1f49990ab7a5b85",
        )

    def test_data11_strict_claim_boundary_remains_in_progress(self) -> None:
        boundary = self.mecat_wavcaps["claim_boundary"]
        self.assertIn("does not prove", boundary["not_proven"])
        self.assertIn("[MISSING]", boundary["paper_protocol"])
        self.assertEqual(boundary["strict_exp09_status"], "IN_PROGRESS")

    def test_nemo_uiq_gpu_task_was_not_started_without_visible_gpu(self) -> None:
        preflight = self.nemo_uiq_preflight
        self.assertEqual(preflight["status"], "blocked_no_visible_gpu")
        self.assertFalse(preflight["torch_cuda_available"])
        self.assertEqual(preflight["torch_cuda_device_count"], 0)
        self.assertFalse(preflight["torch_bf16_supported"])
        self.assertEqual(preflight["launch_status"], "NOT_STARTED")


if __name__ == "__main__":
    unittest.main()

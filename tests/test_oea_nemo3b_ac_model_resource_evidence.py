from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/oea_nemo3b_ac_model_resource_audit_20260729.json"
)
STATUS_PATH = REPOSITORY_ROOT / "docs/reproduction_status.md"


class OeaNemo3bAcModelResourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_run_identity_hash_and_exit_codes_are_fixed(self) -> None:
        run = self.audit["run"]
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(
            run["git_commit"],
            "708cf9ce5d4babf9533d4f43af94d0746cfabbc4",
        )
        self.assertEqual(
            run["metrics_sha256"],
            "2985b739877ad5c5f2cad07f0bddc509b6b3e773ec5dfc1a91f5d6fee888f4f5",
        )
        self.assertEqual(run["metrics_size_bytes"], 20874)
        self.assertEqual(
            [run["final_run_rc"], run["audit_exit_code"], run["wrapper_exit_code"]],
            [0, 0, 0],
        )
        self.assertFalse(run["gpu_used"])
        self.assertFalse(run["downloads_performed"])

    def test_both_assets_are_exactly_complete(self) -> None:
        summary = self.audit["summary"]
        self.assertEqual(summary["complete"], 2)
        self.assertEqual(summary["verified_file_count"], 25)
        self.assertEqual(summary["verified_file_count"], summary["expected_file_count"])
        self.assertEqual(summary["verified_bytes"], 18889955098)
        self.assertEqual(summary["verified_bytes"], summary["expected_bytes"])
        for field in (
            "missing_files",
            "extra_files",
            "incomplete_files",
            "errors",
            "size_mismatches",
            "lfs_sha256_mismatches",
            "git_blob_mismatches",
        ):
            with self.subTest(field=field):
                self.assertEqual(summary[field], 0)
        self.assertEqual(
            {asset["name"]: asset["status"] for asset in self.audit["assets"]},
            {"omni_embed_nemotron_3b": "complete", "oea_nemo3b_ac": "complete"},
        )

    def test_checkpoint_identity_and_claim_boundary_are_explicit(self) -> None:
        checkpoint = self.audit["assets"][1]["source_checkpoint"]
        self.assertEqual(checkpoint["size_bytes"], 9466826153)
        self.assertEqual(
            checkpoint["sha256"],
            "55579dfbd4f6621b5d842c5e731d6a1d37dbfd04b26b1c55bf8cdea980e67d25",
        )
        self.assertTrue(checkpoint["matches_remote_lfs_sha256"])
        self.assertIn(
            "checkpoint internal LoRA and projection-head structure",
            self.audit["claim_boundary"]["not_yet_proven"],
        )
        status = STATUS_PATH.read_text(encoding="utf-8")
        self.assertIn(AUDIT_PATH.relative_to(REPOSITORY_ROOT).as_posix(), status)
        self.assertIn("OEA-Nemo3B-AC 按变体内容 hash/provenance 审计 | COMPLETED", status)


if __name__ == "__main__":
    unittest.main()

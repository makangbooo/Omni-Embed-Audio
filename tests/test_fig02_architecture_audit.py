from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/fig02_architecture_verification_20260728.json"


class Fig02ArchitectureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_all_bound_evidence_identities_match(self) -> None:
        for identity in self.audit["evidence_files"]:
            path = REPOSITORY_ROOT / identity["path"]
            self.assertTrue(path.is_file(), identity["path"])
            self.assertEqual(path.stat().st_size, identity["size_bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), identity["sha256"]
            )

    def test_required_architecture_claims_are_verified(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        expected = {
            "shared_multimodal_backbone",
            "lora_attachment",
            "separate_projection_heads",
            "projection_dimension_512",
            "l2_normalization",
            "five_example_smoke",
        }
        self.assertEqual(set(self.audit["claims"]), expected)
        self.assertEqual(
            {claim["status"] for claim in self.audit["claims"].values()},
            {"verified"},
        )

    def test_qwen7b_and_prefix_limitations_are_not_hidden(self) -> None:
        boundary = json.dumps(self.audit["completion_boundary"])
        self.assertIn("Qwen7B", boundary)
        self.assertIn("passage", boundary)
        self.assertIn("[MISSING]", self.audit["source_labels"]["missing"])


if __name__ == "__main__":
    unittest.main()

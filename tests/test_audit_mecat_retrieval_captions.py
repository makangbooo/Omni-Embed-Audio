from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_mecat_retrieval_captions import audit_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditMecatRetrievalCaptionsTests(unittest.TestCase):
    def create_manifest(self, root: Path) -> Path:
        rows = [
            {
                "sample_id": "a",
                "caption_fields": {
                    "long": ["long a"],
                    "short": ["short a1", "short a2"],
                    "speech": ["None"],
                    "music": [],
                    "sound": ["sound a"],
                    "environment": ["outside"],
                },
            },
            {
                "sample_id": "b",
                "caption_fields": {
                    "long": ["long b"],
                    "short": ["short b"],
                    "speech": None,
                    "music": ["music b"],
                    "sound": ["None"],
                    "environment": [],
                },
            },
        ]
        path = root / "manifest.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_protocol_viability_is_reported_without_guessing_paper_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory))
            report = audit_manifest(manifest, sha256(manifest), 2)
        short_first = report["candidate_protocols"]["short_first"]
        short_all = report["candidate_protocols"]["short_all"]
        all_fields = report["candidate_protocols"]["all_six_fields_flat"]
        self.assertTrue(short_first["complete_t2a_candidate"])
        self.assertFalse(short_first["complete_t2t_self_exclusion_candidate"])
        self.assertEqual(short_all["total_captions"], 3)
        self.assertFalse(short_all["complete_t2t_self_exclusion_candidate"])
        self.assertEqual(all_fields["total_captions"], 8)
        self.assertEqual(report["claim_boundary"]["paper_caption_protocol"], "[MISSING]")

    def test_manifest_hash_and_order_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                audit_manifest(manifest, "0" * 64, 2)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical sample_id order"):
                audit_manifest(manifest, sha256(manifest), 2)

    def test_non_string_caption_items_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory))
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["caption_fields"]["short"] = [123]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(TypeError, "non-string short"):
                audit_manifest(manifest, sha256(manifest), 2)


if __name__ == "__main__":
    unittest.main()

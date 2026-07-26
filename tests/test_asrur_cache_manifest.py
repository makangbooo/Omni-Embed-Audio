import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (
    CacheManifestMismatchError,
    CacheManifestValidationError,
    assert_cache_compatible,
    build_cache_manifest,
    file_record,
    load_cache_manifest,
    manifest_fingerprint,
    validate_cache_manifest,
    verify_file_records,
    write_cache_manifest_once,
)


class CacheManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.jsonl"
        self.output_path = self.root / "embeddings.bin"
        self.input_path.write_text('{"id":"q1"}\n', encoding="utf-8")
        self.output_path.write_bytes(b"\x00\x01\x02")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def build(self, *, git_commit: str = "a" * 40) -> dict:
        return build_cache_manifest(
            artifact_type="document_embeddings",
            dataset="FiQA",
            split="corpus",
            inputs=[file_record(self.input_path, relative_to=self.root)],
            model_name="BAAI/bge-base-en-v1.5",
            model_revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            model_checkpoint=None,
            tokenizer={
                "name": "BAAI/bge-base-en-v1.5",
                "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            },
            pooling="cls",
            embedding_dim=768,
            max_length=512,
            dtype="float32",
            normalization="l2",
            seed=42,
            command=["python", "scripts/encode.py", "--split", "corpus"],
            git_commit=git_commit,
            outputs=[file_record(self.output_path, relative_to=self.root)],
            extra_identity={"document_template": 'title + "\\n" + text'},
            created_at="2026-07-26T08:00:00+00:00",
        )

    def test_complete_manifest_is_valid_and_fingerprinted(self) -> None:
        manifest = self.build()
        validate_cache_manifest(manifest)
        self.assertEqual(len(manifest_fingerprint(manifest)), 64)
        self.assertEqual(manifest["identity"]["embedding_dim"], 768)
        self.assertIn("sha256", manifest["identity"]["inputs"][0])

    def test_creation_time_and_outputs_do_not_change_compatibility_identity(self) -> None:
        first = self.build()
        second = deepcopy(first)
        second["provenance"]["created_at"] = "2026-07-27T08:00:00+00:00"
        second["outputs"][0]["path"] = "moved.bin"
        self.assertEqual(manifest_fingerprint(first), manifest_fingerprint(second))
        assert_cache_compatible(first, second)

    def test_identity_mismatch_is_rejected_with_field_path(self) -> None:
        expected = self.build()
        actual = deepcopy(expected)
        actual["identity"]["pooling"] = "mean"
        with self.assertRaises(CacheManifestMismatchError) as context:
            assert_cache_compatible(expected, actual)
        self.assertIn("$.identity.pooling", str(context.exception))

    def test_git_commit_mismatch_requires_explicit_override(self) -> None:
        expected = self.build(git_commit="a" * 40)
        actual = self.build(git_commit="b" * 40)
        with self.assertRaises(CacheManifestMismatchError):
            assert_cache_compatible(expected, actual)
        assert_cache_compatible(
            expected,
            actual,
            allow_git_commit_mismatch=True,
        )

    def test_manifest_is_created_once_and_never_replaced(self) -> None:
        manifest = self.build()
        path = self.root / "manifest.json"
        self.assertTrue(write_cache_manifest_once(path, manifest))
        self.assertFalse(write_cache_manifest_once(path, manifest))
        self.assertEqual(load_cache_manifest(path), manifest)

        changed = deepcopy(manifest)
        changed["identity"]["dtype"] = "bfloat16"
        with self.assertRaises(FileExistsError):
            write_cache_manifest_once(path, changed)
        self.assertEqual(load_cache_manifest(path), manifest)

    def test_missing_protocol_field_and_bad_checksum_are_rejected(self) -> None:
        missing = self.build()
        del missing["identity"]["normalization"]
        with self.assertRaises(CacheManifestValidationError):
            validate_cache_manifest(missing)

        bad_hash = self.build()
        bad_hash["outputs"][0]["sha256"] = "ABC"
        with self.assertRaises(CacheManifestValidationError):
            validate_cache_manifest(bad_hash)

    def test_file_verification_detects_mutation(self) -> None:
        records = [file_record(self.output_path, relative_to=self.root)]
        self.assertEqual(verify_file_records(records, root=self.root), [])
        self.output_path.write_bytes(b"changed")
        mismatches = verify_file_records(records, root=self.root)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("size expected", mismatches[0])

    def test_json_manifest_rejects_nan(self) -> None:
        manifest = self.build()
        manifest["identity"]["extra"] = float("nan")
        with self.assertRaises(CacheManifestValidationError):
            validate_cache_manifest(manifest)

    def test_written_json_is_strict_utf8_with_final_newline(self) -> None:
        path = self.root / "manifest.json"
        write_cache_manifest_once(path, self.build())
        raw = path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        json.loads(raw.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()

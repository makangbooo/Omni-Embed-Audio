import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "verify_asrur_model_assets.py"
SPEC = importlib.util.spec_from_file_location("verify_asrur_model_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyASRURModelAssetsTest(unittest.TestCase):
    def manifest(self, payload: bytes) -> dict:
        return {
            "resource_id": "TEST",
            "expected_selected_bytes": len(payload),
            "assets": [
                {
                    "name": "model",
                    "repo_id": "org/model",
                    "revision": "a" * 40,
                    "local_subdir": "model",
                    "required_files": ["model.safetensors"],
                    "expected_files": [
                        {
                            "path": "model.safetensors",
                            "size_bytes": len(payload),
                            "lfs_sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ],
                    "expected_selected_bytes": len(payload),
                }
            ],
        }

    def test_complete_file_passes_without_network(self) -> None:
        payload = b"real model bytes"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "model"
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(payload)
            report = MODULE.audit_assets(
                self.manifest(payload),
                root,
                require_git_revision=False,
            )
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["network_access_performed"])
        self.assertEqual(report["observed_selected_bytes"], len(payload))

    def test_unresolved_lfs_pointer_fails(self) -> None:
        payload = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + b"a" * 64 + b"\nsize 123\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "model"
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(payload)
            report = MODULE.audit_assets(
                self.manifest(payload),
                root,
                require_git_revision=False,
            )
        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "unresolved_git_lfs_pointer",
            report["assets"][0]["files"][0]["errors"],
        )

    def test_wrong_size_and_digest_fail(self) -> None:
        expected = b"expected"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "model"
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(b"wrong")
            report = MODULE.audit_assets(
                self.manifest(expected),
                root,
                require_git_revision=False,
            )
        errors = report["assets"][0]["files"][0]["errors"]
        self.assertIn("size_mismatch", errors)
        self.assertIn("sha256_mismatch", errors)

    def test_unsafe_subdirectory_is_rejected(self) -> None:
        manifest = self.manifest(b"x")
        manifest["assets"][0]["local_subdir"] = "../escape"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                MODULE.audit_assets(
                    manifest,
                    Path(temporary),
                    require_git_revision=False,
                )

    def test_wrapper_is_cpu_only_and_uses_new_model_root(self) -> None:
        text = (
            REPOSITORY_ROOT / "scripts" / "run_asrur_model_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("verify_asrur_model_assets.py", text)
        self.assertNotIn("/home/jg525/model_cache", text)
        self.assertNotIn("rm -", text)

    def test_no_operational_shell_script_uses_deleted_model_cache(self) -> None:
        offenders = []
        for path in (REPOSITORY_ROOT / "scripts").glob("*.sh"):
            if "/home/jg525/model_cache" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_d2_repair_is_single_file_resumable_and_non_overwriting(self) -> None:
        text = (
            REPOSITORY_ROOT / "scripts" / "run_asrur_d2_whisper_repair.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn('MODEL_DIR="${MODELS_ROOT}/whisper-large-v3"', text)
        self.assertIn('REVISION="06f233fe06e710322aca913c1bc4249a0d71fce1"', text)
        self.assertIn('EXPECTED_SIZE="3087130976"', text)
        self.assertIn(
            'EXPECTED_SHA256="a8e94b85976e5864ba3e9525c7e6c83'
            'b2a1eca42d4b797a0c7c24d778e40fd95"',
            text,
        )
        self.assertIn('"${FINAL_FILE}.part"', text)
        self.assertIn("--continue-at -", text)
        self.assertIn("--http1.1", text)
        self.assertIn("refusing to overwrite it", text)
        self.assertIn("bash scripts/run_asrur_model_audit.sh", text)
        self.assertNotIn("bge-base-en-v1.5", text)
        self.assertNotIn("bge-reranker-v2-m3", text)
        self.assertNotIn("rm -", text)


if __name__ == "__main__":
    unittest.main()

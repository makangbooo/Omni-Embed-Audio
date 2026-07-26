import json
import unittest
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = (
    REPOSITORY_ROOT / "configs" / "asr_uncertainty_reranking" / "resources"
)


class ASRURResourceManifestTest(unittest.TestCase):
    def load(self, name: str) -> dict:
        return json.loads((RESOURCE_ROOT / name).read_text(encoding="utf-8"))

    def test_fiqa_is_pinned_to_the_approved_jsonl_subset(self) -> None:
        manifest = self.load("fiqa.json")
        self.assertEqual(manifest["resource_id"], "ASRUR-D1")
        self.assertEqual(manifest["approval"]["status"], "approved_by_user")
        self.assertEqual(manifest["license"]["hf_card_value"], "unknown")
        self.assertEqual(len(manifest["assets"]), 1)

        asset = manifest["assets"][0]
        self.assertEqual(asset["repo_id"], "mteb/fiqa")
        self.assertEqual(asset["repo_type"], "dataset")
        self.assertEqual(
            asset["revision"], "5e59eeb3a7df6b85882112b747008547c21587ea"
        )
        self.assertEqual(asset["allow_patterns"], asset["required_files"])
        self.assertEqual(
            set(asset["allow_patterns"]),
            {
                "corpus.jsonl",
                "qrels/dev.jsonl",
                "qrels/test.jsonl",
                "qrels/train.jsonl",
                "queries.jsonl",
            },
        )
        self.assert_inventory_is_exact(asset)
        self.assertEqual(asset["expected_selected_bytes"], 48_616_245)

    def test_models_use_only_the_approved_minimal_safetensors_inventories(
        self,
    ) -> None:
        manifest = self.load("models.json")
        self.assertEqual(manifest["resource_id"], "ASRUR-D2-D4")
        self.assertEqual(manifest["approval"]["status"], "approved_by_user")
        assets = {asset["name"]: asset for asset in manifest["assets"]}
        self.assertEqual(
            set(assets),
            {"whisper_large_v3", "bge_base_en_v1_5", "bge_reranker_v2_m3"},
        )
        expected = {
            "whisper_large_v3": (
                "openai/whisper-large-v3",
                "06f233fe06e710322aca913c1bc4249a0d71fce1",
                3_091_519_764,
            ),
            "bge_base_en_v1_5": (
                "BAAI/bge-base-en-v1.5",
                "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                438_900_399,
            ),
            "bge_reranker_v2_m3": (
                "BAAI/bge-reranker-v2-m3",
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                2_293_242_108,
            ),
        }
        for name, (repo_id, revision, byte_count) in expected.items():
            asset = assets[name]
            self.assertEqual(asset["repo_id"], repo_id)
            self.assertEqual(asset["revision"], revision)
            self.assertEqual(asset["allow_patterns"], asset["required_files"])
            self.assert_inventory_is_exact(asset)
            self.assertEqual(asset["expected_selected_bytes"], byte_count)
            self.assertIn("model.safetensors", asset["allow_patterns"])
            self.assertFalse(
                any(
                    path.endswith((".bin", ".onnx", ".msgpack"))
                    or ".fp32-" in path
                    for path in asset["allow_patterns"]
                )
            )
        self.assertEqual(
            manifest["expected_selected_bytes"],
            sum(asset["expected_selected_bytes"] for asset in assets.values()),
        )
        self.assertEqual(manifest["expected_selected_bytes"], 5_823_662_271)
        self.assertEqual(
            {asset["local_subdir"] for asset in assets.values()},
            {
                "whisper-large-v3",
                "bge-base-en-v1.5",
                "bge-reranker-v2-m3",
            },
        )

    def test_wrapper_is_cpu_only_locked_and_uses_external_roots(self) -> None:
        text = (
            REPOSITORY_ROOT / "scripts" / "run_asrur_resource_download.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("flock -n 9", text)
        self.assertIn("git status --short --untracked-files=all", text)
        self.assertIn("download_model_assets.py", text)
        self.assertIn("ASRUR_DOWNLOAD_MAX_WORKERS:-1", text)
        self.assertIn("HF_HUB_DISABLE_XET", text)
        self.assertIn('MODE=${2:-}', text)
        self.assertIn('--dry-run', text)
        self.assertIn('"network_access_performed": False', text)
        self.assertIn('"download_performed": False', text)
        self.assertIn("/home/jg525/datasets/oea", text)
        self.assertIn("/home/jg525/models", text)
        self.assertIn("/home/jg525/.cache/huggingface", text)
        self.assertNotIn("/home/jg525/model_cache", text)
        self.assertNotIn("rm -", text)
        self.assertNotIn("git reset", text)

    def assert_inventory_is_exact(self, asset: dict) -> None:
        expected = asset["expected_files"]
        paths = [entry["path"] for entry in expected]
        self.assertEqual(paths, asset["allow_patterns"])
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            sum(entry["size_bytes"] for entry in expected),
            asset["expected_selected_bytes"],
        )
        for entry in expected:
            path = PurePosixPath(entry["path"])
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertEqual(len(entry.get("sha256", "")) or 64, 64)
            self.assertEqual(len(entry.get("lfs_sha256", "")) or 64, 64)


if __name__ == "__main__":
    unittest.main()

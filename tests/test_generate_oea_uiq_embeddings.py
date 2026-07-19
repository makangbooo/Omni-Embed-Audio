from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.generate_oea_uiq_embeddings import (
    RELEASED_QUERY_TYPES,
    build_uiq_query_metadata,
    load_uiq_config,
    verify_repository_resource,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GenerateOEAUIQEmbeddingsTest(unittest.TestCase):
    def test_direct_script_entrypoint_resolves_repository_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/generate_oea_uiq_embeddings.py"),
                    "--help",
                ],
                cwd=directory,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--base-embedding-config", completed.stdout)

    def test_fixed_config_pins_all_released_positive_files(self) -> None:
        configs = {
            "qwen3b_cl_clotho_positive_uiq_embeddings.json": "OEA-Qwen3B (+Cl)",
            "qwen3b_clotho_positive_uiq_embeddings.json": "OEA-Qwen3B",
        }
        for filename, expected_model in configs.items():
            with self.subTest(config=filename):
                config = load_uiq_config(
                    REPOSITORY_ROOT / "configs/eval" / filename
                )
                self.assertEqual(config["model"], expected_model)
                self.assertEqual(config["expected_examples"], 1045)
                self.assertEqual(config["expected_total_queries"], 4180)
                self.assertEqual(config["expected_embedding_dim"], 512)
                self.assertEqual(
                    tuple(
                        row["released_query_type"]
                        for row in config["query_sources"]
                    ),
                    RELEASED_QUERY_TYPES,
                )
                self.assertEqual(
                    config["query_sources"][-1]["paper_query_type"], "Keyphrase"
                )
                self.assertEqual(
                    config["query_sources"][-1]["released_query_type"], "tagging"
                )
                verify_repository_resource(config["base_embedding_config"])
                for source in config["query_sources"]:
                    verify_repository_resource(source)

    def make_fixture(self, root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
        manifest = [
            {
                "sample_id": f"sample-{index}.wav",
                "captions": [f"caption {index}-{caption}" for caption in range(1, 6)],
            }
            for index in range(2)
        ]
        sources = []
        for query_type in RELEASED_QUERY_TYPES:
            path = root / f"{query_type}.jsonl"
            rows = []
            for index, item in enumerate(manifest):
                rows.append(
                    {
                        "audio_id": item["sample_id"],
                        "dataset": "clotho",
                        "dataset_slug": "clotho_evaluation",
                        "generated_query": f"{query_type} query {index}",
                        "metadata": {"num_captions": 5, "split": "evaluation"},
                        "original_captions": item["captions"],
                        "query_type": query_type,
                        "regen_model": "gpt-5.1",
                        "source_model": "gpt-5.1",
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            sources.append(
                {
                    "released_query_type": query_type,
                    "paper_query_type": (
                        "Keyphrase" if query_type == "tagging" else query_type.title()
                    ),
                    "path": path.name,
                    "expected_rows": 2,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        config = {"expected_total_queries": 8, "query_sources": sources}
        return manifest, config

    def run_fixture(
        self, root: Path, manifest: list[dict[str, object]], config: dict[str, object]
    ) -> list[dict[str, object]]:
        queries, _ = build_uiq_query_metadata(
            manifest, config, resource_root=root
        )
        return queries

    def test_query_order_is_type_major_then_canonical_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self.make_fixture(root)
            queries = self.run_fixture(root, manifest, config)
        self.assertEqual(
            [row["query_id"] for row in queries],
            [
                "sample-0.wav#uiq_question",
                "sample-1.wav#uiq_question",
                "sample-0.wav#uiq_imperative",
                "sample-1.wav#uiq_imperative",
                "sample-0.wav#uiq_paraphrase",
                "sample-1.wav#uiq_paraphrase",
                "sample-0.wav#uiq_tagging",
                "sample-1.wav#uiq_tagging",
            ],
        )
        self.assertEqual([row["query_index"] for row in queries], list(range(8)))

    def test_missing_candidate_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self.make_fixture(root)
            manifest.append(
                {
                    "sample_id": "missing.wav",
                    "captions": [f"missing {index}" for index in range(5)],
                }
            )
            with self.assertRaisesRegex(ValueError, "UIQ/canonical manifest ID mismatch"):
                self.run_fixture(root, manifest, config)

    def test_caption_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self.make_fixture(root)
            manifest[0]["captions"] = ["changed"] * 5
            with self.assertRaisesRegex(ValueError, "original_captions differ"):
                self.run_fixture(root, manifest, config)

    def test_source_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self.make_fixture(root)
            source = config["query_sources"][0]
            source["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                self.run_fixture(root, manifest, config)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.export_candidate_audio_npz import export
from scripts.generate_oea_negative_uiq_embeddings import generate
from scripts.collect_oea_negative_uiq_results import collect_variant


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    instances = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        FakeEncoder.instances += 1

    def precompute(self, uiq_jsonl: Path, output_dir: Path, dataset: str):
        rows = [json.loads(line) for line in uiq_jsonl.read_text().splitlines()]
        embeddings = np.ones((len(rows), 512), dtype=np.float32)
        np.savez_compressed(
            output_dir / "uiq_negative_embeddings.npz",
            embeddings=embeddings,
            clip_ids=np.asarray([row["audio_id"] for row in rows], dtype=object),
        )
        return {"negative": {"num_queries": len(rows)}}


class OEANegativeUIQPipelineTest(unittest.TestCase):
    def test_candidate_export_preserves_ids_and_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "generation_metrics.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            rows = [
                {"candidate_index": 0, "candidate_id": "a", "audio_path": "/x/a.wav"},
                {"candidate_index": 1, "candidate_id": "b", "audio_path": "/x/b.wav"},
            ]
            (source / "candidate_metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            np.save(source / "candidate_embeddings.npy", np.eye(2, dtype=np.float32))
            report = export(
                Namespace(
                    generation_dir=source,
                    output_dir=output,
                    expected_candidates=2,
                    dataset_name="fixture",
                )
            )
            self.assertEqual(report["status"], "complete")
            with np.load(output / "audio_embeddings.npz", allow_pickle=True) as data:
                self.assertEqual(data["clip_ids"].tolist(), ["a", "b"])
                self.assertEqual(data["filenames"].tolist(), ["a.wav", "b.wav"])

    def test_three_datasets_share_one_encoder_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            local_model = root / "model"
            checkpoint.write_bytes(b"checkpoint")
            local_model.mkdir()
            specs = []
            counts = {"clotho": 2, "audiocaps": 3, "mecat": 1}
            for name, count in counts.items():
                path = root / f"{name}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "audio_id": f"{name}-{index}",
                                "query_type": "negative",
                                "negative_query": "query",
                            }
                        )
                        + "\n"
                        for index in range(count)
                    ),
                    encoding="utf-8",
                )
                specs.append((name, str(path), str(root / f"out-{name}"), str(count)))
            FakeEncoder.instances = 0
            report = generate(
                Namespace(
                    checkpoint=checkpoint,
                    repo_id="fixture/model",
                    local_path=local_model,
                    device="cpu",
                    batch_size_text=2,
                    dataset_spec=specs,
                    summary=root / "summary.json",
                ),
                encoder_factory=FakeEncoder,
            )
            self.assertEqual(FakeEncoder.instances, 1)
            self.assertEqual(report["model_load_count"], 1)
            self.assertEqual(report["total_query_count"], 6)

    def test_runner_pins_models_pairings_and_existing_audio_sources(self) -> None:
        runner = (REPOSITORY_ROOT / "scripts/run_oea_negative_uiq.sh").read_text()
        for variant in (
            "oea_nemo3b",
            "oea_nemo3b_cl",
            "oea_qwen3b",
            "oea_qwen3b_cl",
            "oea_qwen7b",
            "oea_qwen7b_cl",
        ):
            self.assertIn(f"  {variant})", runner)
        for digest in (
            "c05101e76d6343a1446a0132bb16b041c03dcca7e9d96a62d61710177c9fb177",
            "8a773c5cd8519214baf1ebd715df89dd1b75ee685e563ba4bf9c0c8cf1e7effa",
            "e7c7311281681190f544c5d9d0eed348fd3378bbc4ce9c31358096da36bab8aa",
        ):
            self.assertIn(digest, runner)
        self.assertIn("scripts/generate_oea_negative_uiq_embeddings.py", runner)
        self.assertIn("scripts/evaluate_negative_uiq_npz.py", runner)
        self.assertIn("DOWNLOADS_REQUIRED=no", runner)
        self.assertNotIn("rm ", runner)
        matrix = (
            REPOSITORY_ROOT / "scripts/run_oea_negative_uiq_matrix.sh"
        ).read_text()
        self.assertIn("scripts/collect_oea_negative_uiq_results.py", matrix)
        self.assertEqual(matrix.count("oea_qwen7b_cl"), 1)
        self.assertLess(
            matrix.index("scripts/collect_oea_negative_uiq_results.py"),
            matrix.index('echo "MATRIX_STATUS=complete"'),
        )

    def test_compact_collector_recomputes_three_dataset_means(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "oea_fixture_negative_uiq_three_dataset_20260803_000000"
            run.mkdir()
            (run / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (run / "artifact_sha256.txt").write_text("fixture\n", encoding="utf-8")
            for index, name in enumerate(("clotho", "audiocaps", "mecat"), 1):
                metrics_dir = run / "metrics" / name
                metrics_dir.mkdir(parents=True)
                metrics = {
                    metric: float(index)
                    for metric in (
                        "R@5",
                        "R@10",
                        "Delta-Rank",
                        "HNSR",
                        "HNSR@10",
                        "TFR",
                        "TFR-HN@10",
                    )
                }
                report = {
                    "git_commit": "a" * 40,
                    "model": "OEA fixture",
                    "candidate_count": index,
                    "evaluated_query_count": index,
                    "metrics": metrics,
                    "inputs": {
                        key: {"sha256": str(index) * 64}
                        for key in ("audio_npz", "query_npz", "pairing_jsonl")
                    },
                }
                (metrics_dir / "metrics.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
            result = collect_variant(root, "oea_fixture", "a" * 40)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["three_dataset_mean"]["R@5"], 2.0)


if __name__ == "__main__":
    unittest.main()

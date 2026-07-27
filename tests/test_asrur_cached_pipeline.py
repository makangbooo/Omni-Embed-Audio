import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "asr_uncertainty_reranking"
    / "main_experiment.json"
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_cached_split(root: Path, split: str, query_ids: list[str]) -> dict[str, Path]:
    paths = {
        "top100": root / f"{split}.top100.jsonl",
        "nbest": root / f"{split}.nbest.jsonl",
        "cross_encoder": root / f"{split}.ce.jsonl",
        "qrels": root / f"{split}.qrels.jsonl",
    }
    top100_rows = []
    nbest_rows = []
    ce_rows = []
    qrel_rows = []
    for query_index, query_id in enumerate(query_ids):
        candidate_ids = [f"{query_id}_d{index}" for index in range(6)]
        positive_index = query_index % len(candidate_ids)
        oea_scores = [0.9 - 0.1 * index for index in range(6)]
        oea_scores[positive_index] += 0.15
        top100_rows.append(
            {
                "query_id": query_id,
                "candidate_ids": candidate_ids,
                "scores": oea_scores,
            }
        )
        nbest_rows.append(
            {
                "query_id": query_id,
                "no_speech_probability": 0.01 * query_index,
                "hypotheses": [
                    {
                        "rank": rank,
                        "text": f"financial query {query_id} variant {rank}",
                        "sequence_score": -0.2 * rank,
                        "average_token_logprob": -0.1 * rank,
                        "valid_token_count": 5,
                    }
                    for rank in range(1, 5)
                ],
            }
        )
        score_matrix = []
        for rank in range(1, 5):
            scores = [-0.4 - 0.05 * index for index in range(6)]
            scores[positive_index] = 1.2 - 0.1 * rank
            score_matrix.append(scores)
        ce_rows.append(
            {
                "query_id": query_id,
                "candidate_ids": candidate_ids,
                "scores": score_matrix,
            }
        )
        qrel_rows.append(
            {
                "query-id": query_id,
                "corpus-id": candidate_ids[positive_index],
                "score": 1,
            }
        )
    write_jsonl(paths["top100"], top100_rows)
    write_jsonl(paths["nbest"], nbest_rows)
    write_jsonl(paths["cross_encoder"], ce_rows)
    write_jsonl(paths["qrels"], qrel_rows)
    return paths


class CachedPipelineTest(unittest.TestCase):
    def run_script(self, script: str, *arguments: str) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            self.fail(
                f"{script} exited {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return completed

    def test_build_select_train_and_evaluate_cached_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(MAIN_CONFIG.read_text(encoding="utf-8"))
            config["gate"]["max_epochs"] = 3
            config["gate"]["patience"] = 2
            config["selection_grids"]["fixed_asr_weight"] = [0.0, 0.5, 1.0]
            config["selection_grids"]["rrf_rank_constant"] = [20, 60]
            config["selection_grids"]["proxy_temperature"] = [0.5, 1.0]
            config["selection_grids"]["cross_encoder_temperature"] = [1.0]
            config["metrics"]["paired_bootstrap_iterations"] = 50
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            train = build_cached_split(root, "train", ["train1", "train2"])
            dev = build_cached_split(root, "dev", ["dev1", "dev2"])
            test = build_cached_split(root, "test", ["test1", "test2"])

            train_features = root / "train_features"
            dev_features = root / "dev_features"
            for split, paths, output in (
                ("train", train, train_features),
                ("dev", dev, dev_features),
            ):
                self.run_script(
                    "build_asrur_features.py",
                    "--config",
                    str(config_path),
                    "--split",
                    split,
                    "--top100",
                    str(paths["top100"]),
                    "--nbest",
                    str(paths["nbest"]),
                    "--cross-encoder",
                    str(paths["cross_encoder"]),
                    "--qrels",
                    str(paths["qrels"]),
                    "--output-dir",
                    str(output),
                )

            selection_dir = root / "selection"
            self.run_script(
                "select_asrur_dev_parameters.py",
                "--config",
                str(config_path),
                "--top100",
                str(dev["top100"]),
                "--nbest",
                str(dev["nbest"]),
                "--cross-encoder",
                str(dev["cross_encoder"]),
                "--qrels",
                str(dev["qrels"]),
                "--output-dir",
                str(selection_dir),
            )
            selection_path = selection_dir / "frozen_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["selected_on_split"], "fiqa_dev")
            self.assertFalse(selection["test_qrels_used"])

            test_features = root / "test_features"
            self.run_script(
                "build_asrur_features.py",
                "--config",
                str(config_path),
                "--split",
                "test",
                "--top100",
                str(test["top100"]),
                "--nbest",
                str(test["nbest"]),
                "--cross-encoder",
                str(test["cross_encoder"]),
                "--qrels",
                str(test["qrels"]),
                "--selection",
                str(selection_path),
                "--output-dir",
                str(test_features),
            )

            dense_evaluation_dir = root / "dense_evaluation"
            self.run_script(
                "evaluate_asrur_dense_rankings.py",
                "--ranking",
                "B3_oea=" + str(test["top100"]),
                "--ranking",
                "B2_original_omni=" + str(test["top100"]),
                "--qrels",
                str(test["qrels"]),
                "--output-dir",
                str(dense_evaluation_dir),
            )
            dense_metrics = json.loads(
                (dense_evaluation_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(dense_metrics["evaluations"]),
                {"B2_original_omni", "B3_oea"},
            )
            self.assertIsNotNone(dense_metrics["oea_candidate_oracle"])
            self.assertEqual(
                dense_metrics["oea_candidate_oracle"]["num_queries"],
                2,
            )

            gate_dir = root / "gates"
            self.run_script(
                "train_asrur_gate.py",
                "--config",
                str(config_path),
                "--train-features",
                str(train_features / "features.jsonl"),
                "--dev-features",
                str(dev_features / "features.jsonl"),
                "--output-dir",
                str(gate_dir),
            )
            training_summary = json.loads(
                (gate_dir / "training_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(training_summary["train_query_count"], 2)
            self.assertEqual(len(training_summary["seeds"]), 3)

            evaluation_dir = root / "evaluation"
            gold_ce_path = root / "test.gold_ce.jsonl"
            gold_rows = []
            for line in test["cross_encoder"].read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["scores"] = [row["scores"][0]]
                gold_rows.append(row)
            write_jsonl(gold_ce_path, gold_rows)
            self.run_script(
                "evaluate_asrur_reranking.py",
                "--config",
                str(config_path),
                "--selection",
                str(selection_path),
                "--top100",
                str(test["top100"]),
                "--nbest",
                str(test["nbest"]),
                "--cross-encoder",
                str(test["cross_encoder"]),
                "--qrels",
                str(test["qrels"]),
                "--query-gate",
                str(gate_dir / "seed_17" / "query_gate.model.json"),
                "--candidate-gate",
                str(gate_dir / "seed_17" / "candidate_gate.model.json"),
                "--ablation-gate",
                "A4_no_asr_confidence="
                + str(gate_dir / "seed_17" / "A4_no_asr_confidence.model.json"),
                "--ablation-gate",
                "A5_no_nbest_entropy="
                + str(gate_dir / "seed_17" / "A5_no_nbest_entropy.model.json"),
                "--ablation-gate",
                "A6_no_rank_disagreement="
                + str(gate_dir / "seed_17" / "A6_no_rank_disagreement.model.json"),
                "--gold-cross-encoder",
                str(gold_ce_path),
                "--output-dir",
                str(evaluation_dir),
            )
            metrics = json.loads(
                (evaluation_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("Ours_candidate_gate", metrics["evaluations"])
            self.assertIn("A4_no_asr_confidence", metrics["evaluations"])
            self.assertIn("A9_candidate_gate_top50", metrics["evaluations"])
            self.assertIn("U2_gold_ce", metrics["evaluations"])
            self.assertIn("B5_fixed_fusion", metrics["significance"])
            self.assertEqual(metrics["evaluations"]["B3_oea"]["num_queries"], 2)

            cell_index_path = root / "fiqa_cells.json"
            cell_index_path.write_text(
                json.dumps(
                    {
                        "dataset": "fiqa",
                        "cells": [
                            {
                                "condition": condition,
                                "seed": seed,
                                "dense_metrics": str(
                                    dense_evaluation_dir / "metrics.json"
                                ),
                                "rerank_metrics": str(
                                    evaluation_dir / "metrics.json"
                                ),
                            }
                            for condition in ("clean", "snr_20", "snr_10", "snr_0")
                            for seed in (17, 42, 73)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary_dir = root / "summary"
            self.run_script(
                "summarize_asrur_results.py",
                "--config",
                str(config_path),
                "--cell-index",
                str(cell_index_path),
                "--output-dir",
                str(summary_dir),
            )
            self.assertTrue((summary_dir / "main_results.csv").is_file())
            self.assertTrue((summary_dir / "noise_degradation.csv").is_file())
            self.assertTrue((summary_dir / "paired_bootstrap.csv").is_file())


if __name__ == "__main__":
    unittest.main()

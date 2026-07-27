import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_asrur_phase2_results import (
    REQUIRED_METHODS,
    audit_phase2,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def evaluation(ndcg: float, recall: float, *, queries: int = 648) -> dict:
    return {
        "num_queries": queries,
        "scale": "fraction",
        "mean": {
            "nDCG@10": ndcg,
            "MRR@10": ndcg,
            "Recall@10": recall,
            "Recall@20": recall,
            "Recall@50": recall,
            "Recall@100": recall,
        },
        "per_query": {},
    }


class AuditAsrurPhase2ResultsTest(unittest.TestCase):
    def make_fixture(self, root: Path, *, failing_condition: str | None = None):
        run_dir = root / "run"
        result_root = root / "results"
        run_dir.mkdir()
        (run_dir / "wrapper_exit_code.txt").write_text("0\n", encoding="utf-8")
        condition_metrics = {}
        for condition in ("clean", "snr_20", "snr_10", "snr_0"):
            metrics_path = result_root / condition / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            recall = 0.79 if condition == failing_condition else 0.90
            payload = {
                "schema_version": 1,
                "scale": "fraction",
                "evaluations": {
                    "B1_whisper_1best_bge_dense": evaluation(0.35, 0.80),
                    "B2_original_omni": evaluation(0.30, 0.82),
                    "B3_oea": evaluation(0.40, recall),
                    "U1_gold_bge_dense": evaluation(0.50, 0.92),
                },
                "oea_candidate_oracle": evaluation(0.50, recall),
                "rankings": {},
                "provenance": {},
            }
            self.assertEqual(set(payload["evaluations"]), REQUIRED_METHODS)
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            condition_metrics[condition] = str(metrics_path.resolve())
        (run_dir / "completion_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "condition_metrics": condition_metrics,
                }
            ),
            encoding="utf-8",
        )
        return run_dir, result_root

    def test_all_conditions_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, result_root = self.make_fixture(Path(temporary))
            result = audit_phase2(
                config_path=REPOSITORY_ROOT
                / "configs/asr_uncertainty_reranking/main_experiment.json",
                run_dir=run_dir,
                result_root=result_root,
            )
        self.assertEqual(result["overall_decision"], "GO")
        self.assertFalse(result["candidate_generation_change_authorized"])
        self.assertTrue(
            all(
                item["decision"]["decision"] == "GO"
                for item in result["per_condition"].values()
            )
        )

    def test_one_condition_failure_requires_user_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, result_root = self.make_fixture(
                Path(temporary),
                failing_condition="snr_0",
            )
            result = audit_phase2(
                config_path=REPOSITORY_ROOT
                / "configs/asr_uncertainty_reranking/main_experiment.json",
                run_dir=run_dir,
                result_root=result_root,
            )
        self.assertEqual(
            result["overall_decision"],
            "NO_GO_REQUIRES_USER_DECISION",
        )
        self.assertEqual(
            result["per_condition"]["snr_0"]["decision"]["decision"],
            "NO_GO_REQUIRES_USER_DECISION",
        )

    def test_rejects_nonzero_wrapper_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, result_root = self.make_fixture(Path(temporary))
            (run_dir / "wrapper_exit_code.txt").write_text("1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exit code 0"):
                audit_phase2(
                    config_path=REPOSITORY_ROOT
                    / "configs/asr_uncertainty_reranking/main_experiment.json",
                    run_dir=run_dir,
                    result_root=result_root,
                )

    def test_rejects_wrong_query_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, result_root = self.make_fixture(Path(temporary))
            metrics_path = result_root / "clean/metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["evaluations"]["B3_oea"]["num_queries"] = 647
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query count"):
                audit_phase2(
                    config_path=REPOSITORY_ROOT
                    / "configs/asr_uncertainty_reranking/main_experiment.json",
                    run_dir=run_dir,
                    result_root=result_root,
                )


if __name__ == "__main__":
    unittest.main()

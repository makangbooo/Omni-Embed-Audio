import unittest
from pathlib import Path

from AudioRetrieval.asr_uncertainty_reranking.configuration import (
    experiment_inventory,
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.experiment import (
    RerankQueryInput,
    evaluate_cached_reranking,
    phase2_go_no_go,
    truncate_query_input,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationExperimentTest(unittest.TestCase):
    def test_main_config_covers_every_required_method_and_ablation(self) -> None:
        config = load_main_experiment_config(
            REPOSITORY_ROOT
            / "configs"
            / "asr_uncertainty_reranking"
            / "main_experiment.json"
        )
        inventory = experiment_inventory(config)
        self.assertEqual(inventory["baseline_and_upper_bound_method_count"], 15)
        self.assertEqual(inventory["ablation_count"], 9)
        self.assertEqual(inventory["fiqa_formal_gate_cells"], 12)
        self.assertEqual(
            config["candidate_protocol"]["fusion_asr_source"],
            "proxy_posterior",
        )
        self.assertEqual(
            config["candidate_protocol"]["fusion_asr_source_status"],
            "INFERRED_user_confirmed_2026-07-26",
        )

    def test_cached_reranking_never_changes_candidate_membership(self) -> None:
        query = RerankQueryInput(
            query_id="q",
            candidate_ids=("a", "b", "c"),
            oea_scores=(0.8, 0.7, 0.1),
            hypotheses=("query", "queries", "query text", "text query"),
            proxy_logits=(-0.1, -0.2, -0.4, -0.8),
            cross_encoder_scores=(
                (0.1, 1.0, 0.0),
                (0.2, 0.9, 0.0),
                (0.0, 0.8, 0.1),
                (0.1, 0.7, 0.2),
            ),
        )
        result = evaluate_cached_reranking({"q": query}, {"q": {"b": 1}})
        expected = set(query.candidate_ids)
        for rankings in result["rankings"].values():
            self.assertEqual(set(rankings["q"]), expected)
        self.assertIn("B7c_4best_proxy", result["evaluations"])
        self.assertIn("U4_candidate_recall", result["evaluations"])
        truncated = truncate_query_input(query, k=2)
        self.assertEqual(truncated.candidate_ids, ("a", "b"))
        self.assertTrue(
            all(len(row) == 2 for row in truncated.cross_encoder_scores)
        )

    def test_go_no_go_requires_all_preregistered_checks(self) -> None:
        go = phase2_go_no_go(
            oea_metrics={"Recall@100": 0.9, "nDCG@10": 0.4},
            original_omni_metrics={"Recall@100": 0.8, "nDCG@10": 0.3},
            oracle_metrics={"Recall@100": 0.9, "nDCG@10": 0.5},
        )
        self.assertEqual(go["decision"], "GO")
        no_go = phase2_go_no_go(
            oea_metrics={"Recall@100": 0.7, "nDCG@10": 0.4},
            original_omni_metrics={"Recall@100": 0.8, "nDCG@10": 0.3},
            oracle_metrics={"Recall@100": 0.9, "nDCG@10": 0.5},
        )
        self.assertEqual(no_go["decision"], "NO_GO_REQUIRES_USER_DECISION")
        self.assertFalse(no_go["candidate_generation_change_authorized"])


if __name__ == "__main__":
    unittest.main()

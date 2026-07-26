import math
import unittest

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (
    build_asr_uncertainty_features,
)
from AudioRetrieval.asr_uncertainty_reranking.features import (
    FEATURE_NAMES,
    QUERY_ONLY_FEATURE_NAMES,
    ablation_feature_names,
    build_candidate_feature_rows,
)
from AudioRetrieval.asr_uncertainty_reranking.gate import (
    CandidateGateMLP,
    build_training_groups,
    multi_positive_listwise_loss,
    retain_positive_candidate_queries,
    sample_group_candidates,
    train_candidate_gate,
)
from AudioRetrieval.asr_uncertainty_reranking.schema import CandidateFeatureRow


class FeatureBuilderTest(unittest.TestCase):
    def test_candidate_features_have_locked_schema(self) -> None:
        uncertainty = build_asr_uncertainty_features(
            ["query", "queries", "query text", "text query"],
            [-0.1, -0.2, -0.4, -0.8],
        )
        rows = build_candidate_feature_rows(
            query_id="q",
            candidate_ids=["d1", "d2", "d3"],
            oea_scores=[0.9, 0.7, 0.1],
            asr_scores=[0.2, 0.8, 0.0],
            qrels={"d2": 1},
            uncertainty=uncertainty,
        )
        self.assertEqual(rows[0].feature_names, FEATURE_NAMES)
        self.assertEqual(rows[1].relevance, 1.0)
        query_indices = [FEATURE_NAMES.index(name) for name in QUERY_ONLY_FEATURE_NAMES]
        self.assertEqual(
            [rows[0].features[index] for index in query_indices],
            [rows[1].features[index] for index in query_indices],
        )
        self.assertNotIn("rank_gap_signed", ablation_feature_names("no_rank_disagreement"))


class GateTest(unittest.TestCase):
    @staticmethod
    def rows(query_id: str):
        names = ("signal",)
        return [
            CandidateFeatureRow(
                query_id=query_id,
                document_id=f"{query_id}_positive",
                relevance=1.0,
                oea_score=0.0,
                asr_score=1.0,
                features=(1.0,),
                feature_names=names,
            ),
            CandidateFeatureRow(
                query_id=query_id,
                document_id=f"{query_id}_negative",
                relevance=0.0,
                oea_score=0.0,
                asr_score=-1.0,
                features=(-1.0,),
                feature_names=names,
            ),
        ]

    def test_listwise_loss_and_positive_preserving_sampling(self) -> None:
        self.assertAlmostEqual(
            multi_positive_listwise_loss([1.0, 0.0], [1.0, 0.0]),
            math.log1p(math.exp(-1.0)),
        )
        sampled = sample_group_candidates(self.rows("q"), group_size=1)
        self.assertEqual(len(sampled), 1)
        self.assertGreater(sampled[0].relevance, 0.0)

    def test_zero_positive_queries_are_audited_before_gate_training(self) -> None:
        zero_positive = [
            CandidateFeatureRow(
                query_id="q0",
                document_id="q0_negative",
                relevance=0.0,
                oea_score=0.0,
                asr_score=0.0,
                features=(0.0,),
                feature_names=("signal",),
            )
        ]
        retained, excluded = retain_positive_candidate_queries(
            zero_positive + self.rows("q1")
        )
        self.assertEqual(excluded, ("q0",))
        self.assertEqual({row.query_id for row in retained}, {"q1"})

    def test_numpy_gate_trains_and_round_trips(self) -> None:
        train = build_training_groups(
            self.rows("q1") + self.rows("q2"),
            group_size=None,
        )
        dev = build_training_groups(self.rows("q3"), group_size=None)
        model, history = train_candidate_gate(
            train,
            dev,
            feature_names=("signal",),
            hidden_dim=4,
            learning_rate=0.05,
            max_epochs=80,
            patience=15,
            seed=7,
        )
        self.assertLess(history["best_dev_listwise_loss"], math.log(2.0))
        gates = model.predict_gates(dev[0].rows)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in gates))
        restored = CandidateGateMLP.from_dict(model.as_dict())
        self.assertEqual(restored.predict_gates(dev[0].rows), gates)


if __name__ == "__main__":
    unittest.main()

import unittest

from AudioRetrieval.asr_uncertainty_reranking.candidates import (
    build_top_k,
    fixed_weight_fusion,
    reciprocal_rank_fusion,
    top_k_overlap,
    validate_fixed_candidate_sets,
)


class ASRURCandidatesTest(unittest.TestCase):
    def test_top_k_and_ties_are_deterministic(self) -> None:
        result = build_top_k({"q": {"b": 1.0, "a": 1.0, "c": 0.0}}, k=2)
        self.assertEqual(result["q"], [("a", 1.0), ("b", 1.0)])

    def test_fusion_and_rrf_preserve_candidate_order(self) -> None:
        candidates = ["a", "b"]
        fused = fixed_weight_fusion(
            candidates,
            [1.0, 0.0],
            [0.0, 1.0],
            right_weight=0.25,
        )
        self.assertEqual(fused, [0.75, 0.25])
        rrf = reciprocal_rank_fusion(
            candidates,
            [1.0, 0.0],
            [0.0, 1.0],
            rank_constant=10,
        )
        self.assertEqual(len(rrf), 2)

    def test_candidate_contract_and_overlap_modes(self) -> None:
        validate_fixed_candidate_sets({"q": ["a", "b"]}, {"q": ["a", "b"]})
        with self.assertRaises(ValueError):
            validate_fixed_candidate_sets({"q": ["a", "b"]}, {"q": ["b", "a"]})
        self.assertEqual(
            top_k_overlap(
                ["a", "b", "c"],
                ["a", "c", "x"],
                k=2,
                mode="overlap_coefficient",
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()

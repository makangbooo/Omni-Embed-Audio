import unittest

import numpy as np

from AudioRetrieval.asr_uncertainty_reranking.dense import (
    exact_chunked_topk,
    l2_normalize_rows,
)


class DenseRetrievalTest(unittest.TestCase):
    def test_chunked_topk_is_exact_and_has_deterministic_ties(self) -> None:
        documents = l2_normalize_rows(
            np.asarray(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        queries = l2_normalize_rows(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        result = exact_chunked_topk(
            queries,
            documents,
            query_ids=["q1", "q2"],
            document_ids=["b", "a", "c", "d"],
            k=3,
            query_batch_size=1,
            document_chunk_size=2,
        )
        self.assertEqual(result["q1"]["candidate_ids"], ["a", "b", "c"])
        self.assertEqual(result["q2"]["candidate_ids"][0], "c")

    def test_invalid_or_unnormalized_embeddings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            l2_normalize_rows(np.zeros((1, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "not L2 normalized"):
            exact_chunked_topk(
                np.asarray([[2.0, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                query_ids=["q"],
                document_ids=["d"],
                k=1,
            )


if __name__ == "__main__":
    unittest.main()

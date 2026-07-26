"""Auditable spoken-query retrieval and ASR metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

QueryId = Hashable
DocumentId = Hashable


def _validated_relevance(
    qrels: Mapping[DocumentId, Real],
    *,
    query_id: QueryId,
) -> Dict[DocumentId, float]:
    relevance: Dict[DocumentId, float] = {}
    for document_id, raw_value in qrels.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(
                f"qrels[{query_id!r}][{document_id!r}] must be a real number"
            )
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"qrels[{query_id!r}][{document_id!r}] must be finite and non-negative"
            )
        relevance[document_id] = value
    if not any(value > 0.0 for value in relevance.values()):
        raise ValueError(f"query {query_id!r} has no positive-relevance document")
    return relevance


def _validated_ranking(
    ranking: Sequence[DocumentId],
    *,
    query_id: QueryId,
) -> List[DocumentId]:
    documents = list(ranking)
    if not documents:
        raise ValueError(f"ranking for query {query_id!r} must not be empty")
    seen = set()
    for rank, document_id in enumerate(documents, start=1):
        if document_id in seen:
            raise ValueError(
                f"ranking for query {query_id!r} contains duplicate document "
                f"{document_id!r} at rank {rank}"
            )
        seen.add(document_id)
    return documents


def dcg_at_k(
    ranking: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
    *,
    k: int,
) -> float:
    """Compute graded DCG@k using ``2**relevance - 1`` gains."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return math.fsum(
        (2.0 ** float(qrels.get(document_id, 0.0)) - 1.0) / math.log2(rank + 1.0)
        for rank, document_id in enumerate(ranking[:k], start=1)
    )


def ndcg_at_k(
    ranking: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
    *,
    k: int,
) -> float:
    """Compute nDCG@k against the global ideal qrels ranking."""

    actual = dcg_at_k(ranking, qrels, k=k)
    ideal_relevances = sorted(
        (float(value) for value in qrels.values() if float(value) > 0.0),
        reverse=True,
    )
    ideal = math.fsum(
        (2.0**relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(ideal_relevances[:k], start=1)
    )
    if ideal <= 0.0:
        raise ValueError("nDCG is undefined when a query has no positive relevance")
    return actual / ideal


def reciprocal_rank_at_k(
    ranking: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant document within k."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    for rank, document_id in enumerate(ranking[:k], start=1):
        if float(qrels.get(document_id, 0.0)) > 0.0:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranking: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
    *,
    k: int,
) -> float:
    """Return the fraction of all positive qrels retrieved within k."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = {document_id for document_id, value in qrels.items() if float(value) > 0.0}
    if not relevant:
        raise ValueError("Recall is undefined when a query has no positive relevance")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def per_query_retrieval_metrics(
    ranking: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_ks: Sequence[int] = (10, 20, 50, 100),
) -> Dict[str, float]:
    """Compute the preregistered retrieval metrics for one query."""

    metrics = {
        f"nDCG@{ndcg_k}": ndcg_at_k(ranking, qrels, k=ndcg_k),
        f"MRR@{mrr_k}": reciprocal_rank_at_k(ranking, qrels, k=mrr_k),
    }
    for k in recall_ks:
        key = f"Recall@{k}"
        if key in metrics:
            raise ValueError(f"duplicate metric cutoff: {key}")
        metrics[key] = recall_at_k(ranking, qrels, k=k)
    return metrics


def evaluate_rankings(
    rankings: Mapping[QueryId, Sequence[DocumentId]],
    qrels: Mapping[QueryId, Mapping[DocumentId, Real]],
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_ks: Sequence[int] = (10, 20, 50, 100),
    require_exact_query_set: bool = True,
) -> Dict[str, object]:
    """Evaluate rankings with explicit query-set and duplicate checks.

    Returned metric values are fractions in ``[0, 1]``, not percentages.
    """

    if not rankings:
        raise ValueError("rankings must not be empty")
    ranking_queries = set(rankings)
    qrel_queries = set(qrels)
    if require_exact_query_set and ranking_queries != qrel_queries:
        missing_rankings = sorted(map(repr, qrel_queries - ranking_queries))
        missing_qrels = sorted(map(repr, ranking_queries - qrel_queries))
        raise ValueError(
            "ranking/qrels query sets differ: "
            f"missing_rankings={missing_rankings}, missing_qrels={missing_qrels}"
        )
    missing_qrels = ranking_queries - qrel_queries
    if missing_qrels:
        raise ValueError(f"missing qrels for queries: {sorted(map(repr, missing_qrels))}")

    per_query: Dict[str, Dict[str, float]] = {}
    for query_id in sorted(rankings, key=lambda value: (type(value).__name__, repr(value))):
        ranking = _validated_ranking(rankings[query_id], query_id=query_id)
        relevance = _validated_relevance(qrels[query_id], query_id=query_id)
        per_query[str(query_id)] = per_query_retrieval_metrics(
            ranking,
            relevance,
            ndcg_k=ndcg_k,
            mrr_k=mrr_k,
            recall_ks=recall_ks,
        )

    metric_names = list(next(iter(per_query.values())))
    mean = {
        metric_name: math.fsum(
            query_metrics[metric_name] for query_metrics in per_query.values()
        )
        / len(per_query)
        for metric_name in metric_names
    }
    return {
        "num_queries": len(per_query),
        "scale": "fraction",
        "mean": mean,
        "per_query": per_query,
    }


def oracle_ranking(
    candidates: Sequence[DocumentId],
    qrels: Mapping[DocumentId, Real],
) -> List[DocumentId]:
    """Return the best possible ordering of a fixed candidate set.

    Relevance is sorted descending. Original candidate rank is the
    deterministic tie-break, so the oracle never changes candidate membership.
    """

    documents = _validated_ranking(candidates, query_id="<oracle>")
    return [
        document_id
        for _, document_id in sorted(
            enumerate(documents),
            key=lambda pair: (-float(qrels.get(pair[1], 0.0)), pair[0]),
        )
    ]


def evaluate_candidate_oracle(
    candidates: Mapping[QueryId, Sequence[DocumentId]],
    qrels: Mapping[QueryId, Mapping[DocumentId, Real]],
    *,
    ndcg_k: int = 10,
    recall_ks: Sequence[int] = (20, 50, 100),
) -> Dict[str, object]:
    """Evaluate the nDCG oracle and recall ceiling of fixed candidate lists."""

    oracle_rankings = {
        query_id: oracle_ranking(ranking, qrels[query_id])
        for query_id, ranking in candidates.items()
        if query_id in qrels
    }
    return evaluate_rankings(
        oracle_rankings,
        qrels,
        ndcg_k=ndcg_k,
        mrr_k=10,
        recall_ks=recall_ks,
        require_exact_query_set=True,
    )


def noise_degradation(
    clean_metrics: Mapping[str, Real],
    noisy_metrics: Mapping[str, Real],
    *,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute signed absolute and relative clean-to-noisy performance drops."""

    names = list(metric_names) if metric_names is not None else sorted(clean_metrics)
    if not names:
        raise ValueError("metric_names must not be empty")
    output: Dict[str, Dict[str, Optional[float]]] = {}
    for name in names:
        if name not in clean_metrics or name not in noisy_metrics:
            raise KeyError(f"metric {name!r} must exist in both mappings")
        clean = float(clean_metrics[name])
        noisy = float(noisy_metrics[name])
        if not math.isfinite(clean) or not math.isfinite(noisy):
            raise ValueError(f"metric {name!r} must be finite")
        absolute_drop = clean - noisy
        relative_drop = absolute_drop / clean if clean != 0.0 else None
        output[name] = {
            "clean": clean,
            "noisy": noisy,
            "absolute_drop": absolute_drop,
            "relative_drop": relative_drop,
        }
    return output


@dataclass(frozen=True)
class WordErrorCounts:
    """Levenshtein error counts for one reference/hypothesis pair."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> Optional[float]:
        return self.errors / self.reference_words if self.reference_words else None

    def as_dict(self) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = dict(asdict(self))
        result["errors"] = self.errors
        result["wer"] = self.wer
        return result


def word_error_counts(reference: str, hypothesis: str) -> WordErrorCounts:
    """Compute deterministic word-level substitution/deletion/insertion counts."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    source = reference.casefold().split()
    target = hypothesis.casefold().split()
    # Each cell stores (total errors, substitutions, deletions, insertions).
    previous: List[Tuple[int, int, int, int]] = [
        (index, 0, 0, index) for index in range(len(target) + 1)
    ]
    for source_index, source_token in enumerate(source, start=1):
        current: List[Tuple[int, int, int, int]] = [(source_index, 0, source_index, 0)]
        for target_index, target_token in enumerate(target, start=1):
            if source_token == target_token:
                current.append(previous[target_index - 1])
                continue
            substitution = previous[target_index - 1]
            deletion = previous[target_index]
            insertion = current[target_index - 1]
            candidates = [
                (
                    substitution[0] + 1,
                    substitution[1] + 1,
                    substitution[2],
                    substitution[3],
                ),
                (
                    deletion[0] + 1,
                    deletion[1],
                    deletion[2] + 1,
                    deletion[3],
                ),
                (
                    insertion[0] + 1,
                    insertion[1],
                    insertion[2],
                    insertion[3] + 1,
                ),
            ]
            current.append(min(candidates))
        previous = current
    total, substitutions, deletions, insertions = previous[-1]
    assert total == substitutions + deletions + insertions
    return WordErrorCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(source),
    )


def corpus_wer(pairs: Iterable[Tuple[str, str]]) -> Dict[str, float | int]:
    """Compute micro-averaged corpus WER and aggregate error counts."""

    counts = [word_error_counts(reference, hypothesis) for reference, hypothesis in pairs]
    if not counts:
        raise ValueError("pairs must not be empty")
    substitutions = sum(count.substitutions for count in counts)
    deletions = sum(count.deletions for count in counts)
    insertions = sum(count.insertions for count in counts)
    reference_words = sum(count.reference_words for count in counts)
    if reference_words == 0:
        raise ValueError("corpus WER is undefined when all references are empty")
    errors = substitutions + deletions + insertions
    return {
        "num_utterances": len(counts),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "reference_words": reference_words,
        "WER": errors / reference_words,
    }

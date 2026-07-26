"""Candidate- and query-level features for uncertainty-aware fusion."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Mapping, Sequence, Tuple

from .aggregation import ASRUncertaintyFeatures
from .candidates import rank_scores, top_k_overlap
from .normalization import rank_normalize_scores, zscore_scores
from .schema import CandidateFeatureRow

FEATURE_NAMES: Tuple[str, ...] = (
    "asr_top1_average_token_logprob",
    "asr_normalized_nbest_entropy",
    "asr_top1_top2_proxy_margin",
    "asr_hypothesis_edit_distance_mean",
    "asr_hypothesis_edit_distance_max",
    "asr_no_speech_probability",
    "asr_no_speech_probability_missing",
    "oea_score_normalized",
    "asr_score_normalized",
    "oea_rank_normalized",
    "asr_rank_normalized",
    "rank_gap_signed",
    "rank_gap_absolute",
    "top10_overlap",
    "top20_overlap",
    "top50_overlap",
    "oea_score_degenerate",
    "asr_score_degenerate",
)

QUERY_ONLY_FEATURE_NAMES: Tuple[str, ...] = (
    "asr_top1_average_token_logprob",
    "asr_normalized_nbest_entropy",
    "asr_top1_top2_proxy_margin",
    "asr_hypothesis_edit_distance_mean",
    "asr_hypothesis_edit_distance_max",
    "asr_no_speech_probability",
    "asr_no_speech_probability_missing",
    "top10_overlap",
    "top20_overlap",
    "top50_overlap",
    "oea_score_degenerate",
    "asr_score_degenerate",
)


def _finite_scores(values: Sequence[Real], *, label: str) -> list[float]:
    result = []
    for index, raw_value in enumerate(values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"{label}[{index}] must be a real number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(value)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _rank_positions(ranking: Sequence[str]) -> Dict[str, int]:
    return {
        document_id: rank
        for rank, document_id in enumerate(ranking, start=1)
    }


def build_candidate_feature_rows(
    *,
    query_id: str,
    candidate_ids: Sequence[str],
    oea_scores: Sequence[Real],
    asr_scores: Sequence[Real],
    qrels: Mapping[str, Real],
    uncertainty: ASRUncertaintyFeatures,
    normalization: str = "zscore",
    overlap_mode: str = "overlap_coefficient",
) -> list[CandidateFeatureRow]:
    """Build the preregistered feature vector for one frozen Top-K list."""

    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id must be a non-empty string")
    if len(candidate_ids) != len(set(candidate_ids)) or not candidate_ids:
        raise ValueError("candidate_ids must be unique and non-empty")
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise ValueError("candidate IDs must be non-empty strings")
    oea = _finite_scores(oea_scores, label="oea_scores")
    asr = _finite_scores(asr_scores, label="asr_scores")
    if len(oea) != len(candidate_ids) or len(asr) != len(candidate_ids):
        raise ValueError("candidate IDs and both score routes must have equal length")

    if normalization == "zscore":
        normalized_oea = zscore_scores(oea)
        normalized_asr = zscore_scores(asr)
    elif normalization == "rank":
        normalized_oea = rank_normalize_scores(oea)
        normalized_asr = rank_normalize_scores(asr)
    else:
        raise ValueError("normalization must be zscore or rank")

    oea_degenerate = float(max(oea) - min(oea) <= 1e-12)
    asr_degenerate = float(max(asr) - min(asr) <= 1e-12)
    oea_ranking = rank_scores(candidate_ids, oea)
    asr_ranking = rank_scores(candidate_ids, asr)
    oea_positions = _rank_positions(oea_ranking)
    asr_positions = _rank_positions(asr_ranking)
    denominator = max(len(candidate_ids) - 1, 1)
    overlaps = {
        cutoff: top_k_overlap(
            oea_ranking,
            asr_ranking,
            k=cutoff,
            mode=overlap_mode,
        )
        for cutoff in (10, 20, 50)
    }
    no_speech = (
        0.0
        if uncertainty.no_speech_probability is None
        else float(uncertainty.no_speech_probability)
    )

    rows = []
    for index, document_id in enumerate(candidate_ids):
        oea_rank_normalized = (oea_positions[document_id] - 1) / denominator
        asr_rank_normalized = (asr_positions[document_id] - 1) / denominator
        rank_gap = asr_rank_normalized - oea_rank_normalized
        features = (
            float(uncertainty.top1_average_token_logprob),
            float(uncertainty.normalized_nbest_entropy),
            float(uncertainty.top1_top2_proxy_margin),
            float(uncertainty.hypothesis_edit_distance_mean),
            float(uncertainty.hypothesis_edit_distance_max),
            no_speech,
            float(uncertainty.no_speech_probability_missing),
            float(normalized_oea[index]),
            float(normalized_asr[index]),
            oea_rank_normalized,
            asr_rank_normalized,
            rank_gap,
            abs(rank_gap),
            overlaps[10],
            overlaps[20],
            overlaps[50],
            oea_degenerate,
            asr_degenerate,
        )
        relevance_raw = qrels.get(document_id, 0.0)
        if isinstance(relevance_raw, bool) or not isinstance(relevance_raw, Real):
            raise TypeError(f"qrels[{document_id!r}] must be a real number")
        relevance = float(relevance_raw)
        if not math.isfinite(relevance) or relevance < 0.0:
            raise ValueError(f"qrels[{document_id!r}] must be finite and non-negative")
        rows.append(
            CandidateFeatureRow(
                query_id=query_id,
                document_id=document_id,
                relevance=relevance,
                oea_score=float(normalized_oea[index]),
                asr_score=float(normalized_asr[index]),
                features=features,
                feature_names=FEATURE_NAMES,
            )
        )
    return rows


def select_feature_columns(
    rows: Sequence[CandidateFeatureRow],
    feature_names: Sequence[str],
) -> list[list[float]]:
    if not rows:
        raise ValueError("rows must not be empty")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique and non-empty")
    result = []
    for row in rows:
        values = row.feature_dict()
        missing = [name for name in feature_names if name not in values]
        if missing:
            raise KeyError(f"unknown feature names: {missing}")
        result.append([values[name] for name in feature_names])
    return result


def ablation_feature_names(ablation: str) -> Tuple[str, ...]:
    """Return the feature schema for the preregistered gate ablations."""

    excluded: Dict[str, set[str]] = {
        "full": set(),
        "no_asr_confidence": {
            "asr_top1_average_token_logprob",
            "asr_top1_top2_proxy_margin",
        },
        "no_nbest_entropy": {"asr_normalized_nbest_entropy"},
        "no_rank_disagreement": {
            "oea_rank_normalized",
            "asr_rank_normalized",
            "rank_gap_signed",
            "rank_gap_absolute",
            "top10_overlap",
            "top20_overlap",
            "top50_overlap",
        },
    }
    if ablation not in excluded:
        raise ValueError(f"unsupported feature ablation: {ablation}")
    return tuple(name for name in FEATURE_NAMES if name not in excluded[ablation])

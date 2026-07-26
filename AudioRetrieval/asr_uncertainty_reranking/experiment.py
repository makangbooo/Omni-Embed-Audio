"""CPU reranking matrix over one immutable OEA candidate file."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .aggregation import (
    aggregate_nbest_scores,
    build_asr_uncertainty_features,
    softmax_proxy_posteriors,
)
from .candidates import (
    fixed_weight_fusion,
    rank_scores,
    reciprocal_rank_fusion,
)
from .features import build_candidate_feature_rows
from .gate import CandidateGateMLP
from .metrics import evaluate_candidate_oracle, evaluate_rankings, oracle_ranking
from .normalization import rank_normalize_scores, zscore_scores


@dataclass(frozen=True)
class RerankQueryInput:
    query_id: str
    candidate_ids: Tuple[str, ...]
    oea_scores: Tuple[float, ...]
    hypotheses: Tuple[str, ...]
    proxy_logits: Tuple[float, ...]
    cross_encoder_scores: Tuple[Tuple[float, ...], ...]
    top1_average_token_logprob: Optional[float] = None
    no_speech_probability: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string")
        if not self.candidate_ids or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError("candidate_ids must be unique and non-empty")
        if len(self.oea_scores) != len(self.candidate_ids):
            raise ValueError("oea_scores length mismatch")
        if not self.hypotheses or len(self.hypotheses) != len(self.proxy_logits):
            raise ValueError("hypotheses/proxy_logits length mismatch")
        if len(self.cross_encoder_scores) != len(self.hypotheses):
            raise ValueError("cross_encoder_scores hypothesis count mismatch")
        for row in self.cross_encoder_scores:
            if len(row) != len(self.candidate_ids):
                raise ValueError("cross_encoder_scores candidate count mismatch")
        numeric_values = list(self.oea_scores) + list(self.proxy_logits)
        numeric_values.extend(
            score for row in self.cross_encoder_scores for score in row
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError("all query input scores must be finite real numbers")


def truncate_query_input(
    query: RerankQueryInput,
    *,
    k: int,
) -> RerankQueryInput:
    """Truncate one frozen candidate list without regenerating membership."""

    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
        or k > len(query.candidate_ids)
    ):
        raise ValueError("k must be a positive integer within the candidate list")
    return RerankQueryInput(
        query_id=query.query_id,
        candidate_ids=query.candidate_ids[:k],
        oea_scores=query.oea_scores[:k],
        hypotheses=query.hypotheses,
        proxy_logits=query.proxy_logits,
        cross_encoder_scores=tuple(row[:k] for row in query.cross_encoder_scores),
        top1_average_token_logprob=query.top1_average_token_logprob,
        no_speech_probability=query.no_speech_probability,
    )


def _normalize(scores: Sequence[Real], mode: str) -> list[float]:
    if mode == "zscore":
        return zscore_scores(scores)
    if mode == "rank":
        return rank_normalize_scores(scores)
    raise ValueError("normalization must be zscore or rank")


def score_query_methods(
    query: RerankQueryInput,
    *,
    qrels: Mapping[str, Real],
    normalization: str,
    overlap_mode: str,
    fixed_asr_weight: float,
    rrf_rank_constant: float,
    proxy_temperature: float,
    cross_encoder_temperature: float,
    fusion_asr_source: str = "proxy_posterior",
    query_gate: Optional[CandidateGateMLP] = None,
    candidate_gate: Optional[CandidateGateMLP] = None,
) -> tuple[Dict[str, list[str]], Dict[str, object]]:
    """Score B3-B7, optional gates, and U3 without changing membership."""

    candidate_ids = list(query.candidate_ids)
    proxy_posteriors = softmax_proxy_posteriors(
        query.proxy_logits,
        temperature=proxy_temperature,
    )
    aggregated = {
        "one_best": aggregate_nbest_scores(
            query.cross_encoder_scores,
            mode="one_best",
        ),
        "equal_mean": aggregate_nbest_scores(
            query.cross_encoder_scores,
            mode="equal_mean",
        ),
        "max": aggregate_nbest_scores(
            query.cross_encoder_scores,
            mode="max",
        ),
        "proxy_posterior": aggregate_nbest_scores(
            query.cross_encoder_scores,
            mode="proxy_posterior_logsumexp",
            proxy_posteriors=proxy_posteriors,
            cross_encoder_temperature=cross_encoder_temperature,
        ),
    }
    normalized_oea = _normalize(query.oea_scores, normalization)
    normalized_routes = {
        name: _normalize(scores, normalization)
        for name, scores in aggregated.items()
    }
    if fusion_asr_source not in {"one_best", "proxy_posterior"}:
        raise ValueError("fusion_asr_source must be one_best or proxy_posterior")
    fusion_route = normalized_routes[fusion_asr_source]
    method_scores: Dict[str, list[float]] = {
        "B3_oea": normalized_oea,
        "B4_1best_ce": normalized_routes["one_best"],
        "B5_fixed_fusion": fixed_weight_fusion(
            candidate_ids,
            normalized_oea,
            fusion_route,
            right_weight=fixed_asr_weight,
        ),
        "B6_rrf": reciprocal_rank_fusion(
            candidate_ids,
            normalized_oea,
            fusion_route,
            rank_constant=rrf_rank_constant,
        ),
        "B5_aux_1best_fixed_fusion": fixed_weight_fusion(
            candidate_ids,
            normalized_oea,
            normalized_routes["one_best"],
            right_weight=fixed_asr_weight,
        ),
        "B6_aux_1best_rrf": reciprocal_rank_fusion(
            candidate_ids,
            normalized_oea,
            normalized_routes["one_best"],
            rank_constant=rrf_rank_constant,
        ),
        "B7a_4best_equal": normalized_routes["equal_mean"],
        "B7b_4best_max": normalized_routes["max"],
        "B7c_4best_proxy": normalized_routes["proxy_posterior"],
        "A7_oea_only": normalized_oea,
        "A8_asr_only": normalized_routes["proxy_posterior"],
    }
    uncertainty = build_asr_uncertainty_features(
        query.hypotheses,
        query.proxy_logits,
        proxy_temperature=proxy_temperature,
        top1_average_token_logprob=query.top1_average_token_logprob,
        no_speech_probability=query.no_speech_probability,
    )
    feature_rows = build_candidate_feature_rows(
        query_id=query.query_id,
        candidate_ids=candidate_ids,
        oea_scores=query.oea_scores,
        asr_scores=aggregated["proxy_posterior"],
        qrels=qrels,
        uncertainty=uncertainty,
        normalization=normalization,
        overlap_mode=overlap_mode,
    )
    gate_outputs: Dict[str, object] = {}
    for method_name, model in (
        ("QG_query_gate", query_gate),
        ("Ours_candidate_gate", candidate_gate),
    ):
        if model is None:
            continue
        gates = model.predict_gates(feature_rows)
        scores = model.predict_scores(feature_rows)
        method_scores[method_name] = scores
        gate_outputs[method_name] = {
            "gates": gates,
            "mean_gate": math.fsum(gates) / len(gates),
            "min_gate": min(gates),
            "max_gate": max(gates),
        }

    rankings = {
        method: rank_scores(candidate_ids, scores)
        for method, scores in method_scores.items()
    }
    rankings["U3_candidate_oracle"] = oracle_ranking(candidate_ids, qrels)
    return rankings, {
        "proxy_posteriors": proxy_posteriors,
        "uncertainty": uncertainty.as_dict(),
        "gate_outputs": gate_outputs,
        "candidate_count": len(candidate_ids),
    }


def evaluate_cached_reranking(
    queries: Mapping[str, RerankQueryInput],
    qrels: Mapping[str, Mapping[str, Real]],
    *,
    normalization: str = "zscore",
    overlap_mode: str = "overlap_coefficient",
    fixed_asr_weight: float = 0.5,
    rrf_rank_constant: float = 60.0,
    proxy_temperature: float = 1.0,
    cross_encoder_temperature: float = 1.0,
    fusion_asr_source: str = "proxy_posterior",
    query_gate: Optional[CandidateGateMLP] = None,
    candidate_gate: Optional[CandidateGateMLP] = None,
) -> Dict[str, object]:
    if set(queries) != set(qrels):
        raise ValueError("queries and qrels query sets must match exactly")
    rankings_by_method: Dict[str, Dict[str, list[str]]] = {}
    diagnostics: Dict[str, object] = {}
    candidate_reference: Dict[str, list[str]] = {}
    for query_id in sorted(queries):
        query = queries[query_id]
        if query.query_id != query_id:
            raise ValueError("query mapping key and record query_id differ")
        rankings, query_diagnostics = score_query_methods(
            query,
            qrels=qrels[query_id],
            normalization=normalization,
            overlap_mode=overlap_mode,
            fixed_asr_weight=fixed_asr_weight,
            rrf_rank_constant=rrf_rank_constant,
            proxy_temperature=proxy_temperature,
            cross_encoder_temperature=cross_encoder_temperature,
            fusion_asr_source=fusion_asr_source,
            query_gate=query_gate,
            candidate_gate=candidate_gate,
        )
        candidate_reference[query_id] = list(query.candidate_ids)
        diagnostics[query_id] = query_diagnostics
        for method, ranking in rankings.items():
            if set(ranking) != set(query.candidate_ids):
                raise AssertionError(f"{method} changed candidate membership")
            rankings_by_method.setdefault(method, {})[query_id] = ranking

    evaluations = {
        method: evaluate_rankings(rankings, qrels)
        for method, rankings in sorted(rankings_by_method.items())
    }
    evaluations["U4_candidate_recall"] = evaluate_candidate_oracle(
        candidate_reference,
        qrels,
    )
    return {
        "schema_version": 1,
        "scale": "fraction",
        "normalization": normalization,
        "overlap_mode": overlap_mode,
        "fixed_asr_weight": fixed_asr_weight,
        "rrf_rank_constant": rrf_rank_constant,
        "proxy_temperature": proxy_temperature,
        "cross_encoder_temperature": cross_encoder_temperature,
        "fusion_asr_source": fusion_asr_source,
        "evaluations": evaluations,
        "rankings": rankings_by_method,
        "diagnostics": diagnostics,
    }


def phase2_go_no_go(
    *,
    oea_metrics: Mapping[str, Real],
    original_omni_metrics: Mapping[str, Real],
    oracle_metrics: Mapping[str, Real],
    minimum_recall_at_100: float = 0.80,
    minimum_oracle_ndcg_gain: float = 0.08,
) -> Dict[str, object]:
    required = ("Recall@100", "nDCG@10")
    for label, metrics in (
        ("oea", oea_metrics),
        ("original_omni", original_omni_metrics),
        ("oracle", oracle_metrics),
    ):
        missing = [key for key in required if key not in metrics]
        if missing:
            raise KeyError(f"{label} metrics missing: {missing}")
    recall_pass = float(oea_metrics["Recall@100"]) >= minimum_recall_at_100
    oracle_gain = float(oracle_metrics["nDCG@10"]) - float(oea_metrics["nDCG@10"])
    oracle_pass = oracle_gain >= minimum_oracle_ndcg_gain
    omni_pass = float(oea_metrics["nDCG@10"]) >= float(
        original_omni_metrics["nDCG@10"]
    )
    decision = recall_pass and oracle_pass and omni_pass
    return {
        "decision": "GO" if decision else "NO_GO_REQUIRES_USER_DECISION",
        "checks": {
            "oea_recall_at_100": {
                "value": float(oea_metrics["Recall@100"]),
                "threshold": minimum_recall_at_100,
                "passed": recall_pass,
            },
            "oracle_ndcg_gain": {
                "value": oracle_gain,
                "threshold": minimum_oracle_ndcg_gain,
                "passed": oracle_pass,
            },
            "oea_not_weaker_than_original_omni_ndcg": {
                "oea": float(oea_metrics["nDCG@10"]),
                "original_omni": float(original_omni_metrics["nDCG@10"]),
                "passed": omni_pass,
            },
        },
        "candidate_generation_change_authorized": False,
    }

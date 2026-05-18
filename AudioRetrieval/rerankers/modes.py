"""Concrete reranker implementations covering audio/listwise modes."""

from __future__ import annotations

from time import perf_counter
import os
from typing import Dict, List, Tuple

from .base import BaseReranker, CandidateItem, RerankResult


class AudioPointwiseReranker(BaseReranker):
    """Audio-native pointwise reranking."""

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        subset = self.limit_candidates(candidates)
        start = perf_counter()
        original_order = [cand.candidate_idx for cand in subset]
        initial_scores = {cand.candidate_idx: float(cand.initial_score) for cand in subset}
        scored: List[Tuple[float, CandidateItem]] = []
        supports_batch = hasattr(self.client, "score_audio_batch")
        batch_size = getattr(self.client, "batch_size", 1) if supports_batch else 1
        batch_size = max(1, int(batch_size))
        idx = 0
        total = len(subset)
        lalms_scores: Dict[int, float] = {}

        # Optional inner progress per query (shows candidate scoring progress)
        inner = None
        try:
            if os.environ.get("LALM_INNER_PROGRESS", "").lower() in {"1", "true", "yes"}:
                try:
                    from tqdm import tqdm  # type: ignore
                    inner = tqdm(total=total, desc=f"{self.name} candidates", leave=False, ascii=True)
                except Exception:
                    inner = None
        except Exception:
            inner = None
        while idx < total:
            chunk = subset[idx : idx + (batch_size if supports_batch else 1)]
            if supports_batch and len(chunk) > 1:
                scores = self.client.score_audio_batch(
                    query,
                    [cand.audio_path for cand in chunk],
                )
            else:
                scores = [self.client.score_audio(query, cand.audio_path) for cand in chunk]
            for cand, score in zip(chunk, scores):
                scored.append((score, cand))
                lalms_scores[cand.candidate_idx] = float(score)
            idx += len(chunk)
            if inner:
                try:
                    inner.update(len(chunk))
                except Exception:
                    pass

        scored.sort(key=lambda x: x[0], reverse=True)
        latency = perf_counter() - start
        ordered = [cand for _, cand in scored]
        reranked_order = [cand.candidate_idx for cand in ordered]
        diagnostics = {
            "type": "pointwise",
            "original_order": original_order,
            "reranked_order": reranked_order,
            "initial_scores": initial_scores,
            "scores": lalms_scores,
        }
        try:
            if inner:
                inner.close()
        except Exception:
            pass

        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=0.0,
            extra={
                "mode": "audio_native_pointwise",
                "diagnostics": diagnostics,
            },
        )


class AudioListwiseReranker(BaseReranker):
    """Audio-native listwise reranking."""

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        subset = self.limit_candidates(candidates)
        start = perf_counter()
        original_order = [cand.candidate_idx for cand in subset]
        initial_scores = {cand.candidate_idx: float(cand.initial_score) for cand in subset}
        order = self.client.rank_audio(query, subset)
        latency = perf_counter() - start
        ordered = [subset[idx] for idx in order]
        reranked_order = [cand.candidate_idx for cand in ordered]
        diagnostics = {
            "type": "listwise",
            "original_order": original_order,
            "reranked_order": reranked_order,
            "initial_scores": initial_scores,
        }
        info_getter = getattr(self.client, "get_last_rank_info", None)
        if callable(info_getter):
            info = info_getter()
            if info is not None:
                diagnostics["listwise_info"] = info
        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=0.0,
            extra={
                "mode": "audio_native_listwise",
                "diagnostics": diagnostics,
            },
        )


class CaptionPointwiseReranker(BaseReranker):
    """Caption-proxy pointwise reranking (caption generation + scoring)."""

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        subset = self.limit_candidates(candidates)
        start = perf_counter()
        scored: List[Dict] = []
        for cand in subset:
            if not cand.caption:
                cand.caption = self.client.generate_caption(cand.audio_path)
            score = self.client.score_caption(query, cand.caption)
            scored.append({"score": score, "candidate": cand})
        scored.sort(key=lambda x: x["score"], reverse=True)
        latency = perf_counter() - start
        ordered = [entry["candidate"] for entry in scored]
        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=0.0,
            extra={"mode": "caption_proxy_pointwise"},
        )


class CaptionListwiseReranker(BaseReranker):
    """Caption-proxy listwise reranking."""

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        subset = self.limit_candidates(candidates)
        start = perf_counter()
        captions: List[str] = []
        for cand in subset:
            if not cand.caption:
                cand.caption = self.client.generate_caption(cand.audio_path)
            captions.append(cand.caption)
        original_order = [cand.candidate_idx for cand in subset]
        initial_scores = {cand.candidate_idx: float(cand.initial_score) for cand in subset}
        order = self.client.rank_captions(query, captions)
        latency = perf_counter() - start
        ordered = [subset[idx] for idx in order]
        reranked_order = [cand.candidate_idx for cand in ordered]
        diagnostics = {
            "type": "caption_listwise",
            "original_order": original_order,
            "reranked_order": reranked_order,
            "initial_scores": initial_scores,
        }
        info_getter = getattr(self.client, "get_last_rank_info", None)
        if callable(info_getter):
            info = info_getter()
            if info is not None:
                diagnostics["listwise_info"] = info
        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=0.0,
            extra={
                "mode": "caption_proxy_listwise",
                "diagnostics": diagnostics,
            },
        )

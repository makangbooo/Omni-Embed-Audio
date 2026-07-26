"""Strict schemas for frozen Top-100, Whisper N-best, CE scores, and features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .data import parse_integral_relevance, read_jsonl
from .experiment import RerankQueryInput
from .schema import CandidateFeatureRow, NBestHypothesis, nonempty_string


@dataclass(frozen=True)
class FrozenCandidateRecord:
    query_id: str
    candidate_ids: Tuple[str, ...]
    scores: Tuple[float, ...]

    def __post_init__(self) -> None:
        nonempty_string(self.query_id, field="query_id")
        if not self.candidate_ids or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError("candidate_ids must be unique and non-empty")
        if len(self.scores) != len(self.candidate_ids):
            raise ValueError("candidate_ids/scores length mismatch")
        if any(not math.isfinite(score) for score in self.scores):
            raise ValueError("candidate scores must be finite")


@dataclass(frozen=True)
class NBestRecord:
    query_id: str
    hypotheses: Tuple[NBestHypothesis, ...]
    no_speech_probability: float | None

    def __post_init__(self) -> None:
        nonempty_string(self.query_id, field="query_id")
        if not self.hypotheses:
            raise ValueError("hypotheses must not be empty")
        expected_ranks = tuple(range(1, len(self.hypotheses) + 1))
        if tuple(value.rank for value in self.hypotheses) != expected_ranks:
            raise ValueError("hypothesis ranks must be contiguous and one-indexed")
        if self.no_speech_probability is not None and not (
            math.isfinite(self.no_speech_probability)
            and 0.0 <= self.no_speech_probability <= 1.0
        ):
            raise ValueError("no_speech_probability must be null or in [0, 1]")


@dataclass(frozen=True)
class CrossEncoderRecord:
    query_id: str
    candidate_ids: Tuple[str, ...]
    scores: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        nonempty_string(self.query_id, field="query_id")
        if not self.candidate_ids or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError("candidate_ids must be unique and non-empty")
        if not self.scores:
            raise ValueError("cross-encoder scores must not be empty")
        if any(len(row) != len(self.candidate_ids) for row in self.scores):
            raise ValueError("cross-encoder score rows must match candidate count")
        if any(not math.isfinite(value) for row in self.scores for value in row):
            raise ValueError("cross-encoder scores must be finite")


def _float_list(value: object, *, field: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{field}[{index}] must be numeric")
        converted = float(raw)
        if not math.isfinite(converted):
            raise ValueError(f"{field}[{index}] must be finite")
        result.append(converted)
    return tuple(result)


def _string_list(value: object, *, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(nonempty_string(item, field=field) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must be unique")
    return result


def load_frozen_candidates(path: Path | str) -> Dict[str, FrozenCandidateRecord]:
    result = {}
    for line_number, row in read_jsonl(path):
        record = FrozenCandidateRecord(
            query_id=nonempty_string(
                row.get("query_id"),
                field=f"{path}:{line_number}:query_id",
            ),
            candidate_ids=_string_list(
                row.get("candidate_ids"),
                field=f"{path}:{line_number}:candidate_ids",
            ),
            scores=_float_list(
                row.get("scores"),
                field=f"{path}:{line_number}:scores",
            ),
        )
        if record.query_id in result:
            raise ValueError(f"duplicate candidate query {record.query_id!r}")
        result[record.query_id] = record
    if not result:
        raise ValueError("frozen candidate artifact is empty")
    return result


def load_nbest(path: Path | str, *, expected_size: int = 4) -> Dict[str, NBestRecord]:
    result = {}
    for line_number, row in read_jsonl(path):
        raw_hypotheses = row.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raise TypeError(f"hypotheses must be a list: {path}:{line_number}")
        hypotheses = []
        for index, raw in enumerate(raw_hypotheses):
            if not isinstance(raw, dict):
                raise TypeError("each hypothesis must be an object")
            hypotheses.append(
                NBestHypothesis(
                    rank=raw.get("rank"),
                    text=raw.get("text"),
                    sequence_score=raw.get("sequence_score"),
                    average_token_logprob=raw.get("average_token_logprob"),
                    valid_token_count=raw.get("valid_token_count"),
                )
            )
        if len(hypotheses) != expected_size:
            raise ValueError(
                f"expected {expected_size} hypotheses: {path}:{line_number}"
            )
        no_speech_raw = row.get("no_speech_probability")
        record = NBestRecord(
            query_id=nonempty_string(
                row.get("query_id"),
                field=f"{path}:{line_number}:query_id",
            ),
            hypotheses=tuple(hypotheses),
            no_speech_probability=(
                None if no_speech_raw is None else float(no_speech_raw)
            ),
        )
        if record.query_id in result:
            raise ValueError(f"duplicate N-best query {record.query_id!r}")
        result[record.query_id] = record
    if not result:
        raise ValueError("N-best artifact is empty")
    return result


def load_cross_encoder_scores(
    path: Path | str,
) -> Dict[str, CrossEncoderRecord]:
    result = {}
    for line_number, row in read_jsonl(path):
        raw_matrix = row.get("scores")
        if not isinstance(raw_matrix, list) or not raw_matrix:
            raise ValueError(f"scores must be a non-empty matrix: {path}:{line_number}")
        matrix = tuple(
            _float_list(values, field=f"{path}:{line_number}:scores")
            for values in raw_matrix
        )
        record = CrossEncoderRecord(
            query_id=nonempty_string(
                row.get("query_id"),
                field=f"{path}:{line_number}:query_id",
            ),
            candidate_ids=_string_list(
                row.get("candidate_ids"),
                field=f"{path}:{line_number}:candidate_ids",
            ),
            scores=matrix,
        )
        if record.query_id in result:
            raise ValueError(f"duplicate CE query {record.query_id!r}")
        result[record.query_id] = record
    if not result:
        raise ValueError("cross-encoder artifact is empty")
    return result


def load_unbounded_qrels(path: Path | str) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    seen = set()
    for line_number, row in read_jsonl(path):
        query_raw = row.get("query-id", row.get("query_id"))
        document_raw = row.get("corpus-id", row.get("corpus_id"))
        query_id = nonempty_string(
            str(query_raw) if query_raw is not None else None,
            field=f"{path}:{line_number}:query_id",
        )
        document_id = nonempty_string(
            str(document_raw) if document_raw is not None else None,
            field=f"{path}:{line_number}:document_id",
        )
        if "score" not in row:
            raise KeyError(f"missing score: {path}:{line_number}")
        relevance = parse_integral_relevance(
            row["score"],
            location=f"{path}:{line_number}:score",
        )
        pair = (query_id, document_id)
        if pair in seen:
            raise ValueError(f"duplicate qrels pair {pair!r}")
        seen.add(pair)
        qrels.setdefault(query_id, {})[document_id] = float(relevance)
    if not qrels or any(
        not any(relevance > 0.0 for relevance in documents.values())
        for documents in qrels.values()
    ):
        raise ValueError("qrels must contain a positive document for every query")
    return qrels


def assemble_rerank_inputs(
    candidates: Mapping[str, FrozenCandidateRecord],
    nbest: Mapping[str, NBestRecord],
    cross_encoder: Mapping[str, CrossEncoderRecord],
) -> Dict[str, RerankQueryInput]:
    if set(candidates) != set(nbest) or set(candidates) != set(cross_encoder):
        raise ValueError("candidate, N-best, and CE query sets must match exactly")
    result = {}
    for query_id in sorted(candidates):
        candidate = candidates[query_id]
        hypotheses = nbest[query_id]
        ce = cross_encoder[query_id]
        if ce.candidate_ids != candidate.candidate_ids:
            raise ValueError(f"CE candidate order differs for query {query_id!r}")
        if len(ce.scores) != len(hypotheses.hypotheses):
            raise ValueError(f"CE/N-best hypothesis count differs for query {query_id!r}")
        result[query_id] = RerankQueryInput(
            query_id=query_id,
            candidate_ids=candidate.candidate_ids,
            oea_scores=candidate.scores,
            hypotheses=tuple(value.text for value in hypotheses.hypotheses),
            proxy_logits=tuple(
                value.average_token_logprob for value in hypotheses.hypotheses
            ),
            cross_encoder_scores=ce.scores,
            top1_average_token_logprob=hypotheses.hypotheses[
                0
            ].average_token_logprob,
            no_speech_probability=hypotheses.no_speech_probability,
        )
    return result


def write_feature_rows_once(
    path: Path | str,
    rows: Iterable[CandidateFeatureRow],
) -> None:
    destination = Path(path)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        count = 0
        for row in rows:
            stream.write(
                json.dumps(
                    row.as_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
        if count == 0:
            raise ValueError("feature rows must not be empty")


def load_feature_rows(path: Path | str) -> list[CandidateFeatureRow]:
    rows = []
    for line_number, row in read_jsonl(path):
        try:
            feature_names = tuple(row["feature_names"])
            features = tuple(float(value) for value in row["features"])
            record = CandidateFeatureRow(
                query_id=row["query_id"],
                document_id=row["document_id"],
                relevance=float(row["relevance"]),
                oea_score=float(row["oea_score"]),
                asr_score=float(row["asr_score"]),
                features=features,
                feature_names=feature_names,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid feature row: {path}:{line_number}") from exc
        rows.append(record)
    if not rows:
        raise ValueError("feature artifact is empty")
    return rows

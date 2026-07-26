"""FiQA/NQ/SQuTR JSONL loading and split-leakage auditing."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

from .schema import AudioQuery, CorpusDocument, TextQuery, nonempty_string

SQuTR_CONDITIONS = ("clean", "snr_20", "snr_10", "snr_0")


def read_jsonl(path: Path | str) -> Iterator[tuple[int, Dict[str, Any]]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                raise ValueError(f"blank JSONL row: {source}:{line_number}")
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object: {source}:{line_number}")
            yield line_number, value


def construct_document_text(title: str, text: str) -> str:
    """Match the pinned SQuTR/MTEB construction exactly."""

    if not isinstance(title, str) or not isinstance(text, str):
        raise TypeError("title and text must be strings")
    return f"{title}\n{text}" if title else text


def normalized_query_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("query text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def load_corpus(
    path: Path | str,
    *,
    id_field: str = "_id",
    title_field: str = "title",
    text_field: str = "text",
) -> Dict[str, CorpusDocument]:
    documents: Dict[str, CorpusDocument] = {}
    for line_number, row in read_jsonl(path):
        document_id = nonempty_string(
            row.get(id_field),
            field=f"{path}:{line_number}:{id_field}",
        )
        title = row.get(title_field, "")
        text = row.get(text_field, "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise TypeError(f"corpus title/text must be strings: {path}:{line_number}")
        if document_id in documents:
            raise ValueError(f"duplicate corpus ID {document_id!r}: {path}:{line_number}")
        documents[document_id] = CorpusDocument(
            document_id=document_id,
            title=title,
            text=text,
            constructed_text=construct_document_text(title, text),
        )
    if not documents:
        raise ValueError(f"corpus is empty: {path}")
    return documents


def load_text_queries(
    path: Path | str,
    *,
    id_field: str = "_id",
    text_field: str = "text",
) -> Dict[str, TextQuery]:
    queries: Dict[str, TextQuery] = {}
    for line_number, row in read_jsonl(path):
        query_id = nonempty_string(
            row.get(id_field),
            field=f"{path}:{line_number}:{id_field}",
        )
        text = row.get(text_field)
        if not isinstance(text, str):
            raise TypeError(f"query text must be a string: {path}:{line_number}")
        if query_id in queries:
            raise ValueError(f"duplicate query ID {query_id!r}: {path}:{line_number}")
        queries[query_id] = TextQuery(query_id=query_id, text=text)
    if not queries:
        raise ValueError(f"query file is empty: {path}")
    return queries


def parse_integral_relevance(raw: object, *, location: str) -> int:
    """Match the pinned SQuTR loader's integer relevance semantics safely."""

    if isinstance(raw, bool):
        raise ValueError(f"boolean relevance is invalid: {location}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"non-integral relevance is invalid: {location}")
        return int(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not re.fullmatch(r"[+-]?\d+", stripped):
            raise ValueError(f"invalid integer-string relevance: {location}")
        return int(stripped, 10)
    raise TypeError(f"unsupported relevance type at {location}: {type(raw).__name__}")


def load_qrels(
    path: Path | str,
    *,
    query_ids: Iterable[str],
    document_ids: Iterable[str],
    query_fields: Sequence[str] = ("query-id", "query_id"),
    document_fields: Sequence[str] = ("corpus-id", "corpus_id"),
    score_field: str = "score",
) -> Dict[str, Dict[str, float]]:
    query_set = set(query_ids)
    document_set = set(document_ids)
    qrels: Dict[str, Dict[str, float]] = {}
    seen_pairs = set()
    for line_number, row in read_jsonl(path):
        query_raw = next((row[field] for field in query_fields if field in row), None)
        document_raw = next((row[field] for field in document_fields if field in row), None)
        query_id = nonempty_string(
            str(query_raw) if query_raw is not None else None,
            field=f"{path}:{line_number}:query_id",
        )
        document_id = nonempty_string(
            str(document_raw) if document_raw is not None else None,
            field=f"{path}:{line_number}:document_id",
        )
        if score_field not in row:
            raise KeyError(f"missing {score_field}: {path}:{line_number}")
        relevance = parse_integral_relevance(
            row[score_field],
            location=f"{path}:{line_number}:{score_field}",
        )
        if relevance < 0:
            raise ValueError(f"negative relevance is invalid: {path}:{line_number}")
        if query_id not in query_set:
            raise KeyError(f"qrels references unknown query {query_id!r}")
        if document_id not in document_set:
            raise KeyError(f"qrels references unknown document {document_id!r}")
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate qrels pair {pair!r}")
        seen_pairs.add(pair)
        qrels.setdefault(query_id, {})[document_id] = float(relevance)
    missing = sorted(query_set - set(qrels))
    if missing:
        raise ValueError(f"queries without qrels: {missing[:20]}")
    return qrels


def load_squtr_audio_manifest(
    path: Path | str,
    *,
    require_audio_files: bool = False,
) -> Dict[str, AudioQuery]:
    records: Dict[str, AudioQuery] = {}
    seen_query_condition = set()
    for line_number, row in read_jsonl(path):
        condition = nonempty_string(
            row.get("condition"),
            field=f"{path}:{line_number}:condition",
        )
        if condition not in SQuTR_CONDITIONS:
            raise ValueError(f"unsupported SQuTR condition {condition!r}")
        record = AudioQuery(
            record_id=nonempty_string(
                row.get("record_id"),
                field=f"{path}:{line_number}:record_id",
            ),
            query_id=nonempty_string(
                row.get("query_id"),
                field=f"{path}:{line_number}:query_id",
            ),
            dataset=str(row.get("dataset", "squtr")),
            subset=nonempty_string(
                row.get("subset"),
                field=f"{path}:{line_number}:subset",
            ),
            condition=condition,
            audio_path=nonempty_string(
                row.get("audio_path"),
                field=f"{path}:{line_number}:audio_path",
            ),
            original_query_text=row.get("original_query_text", ""),
            sample_rate=row.get("sample_rate"),
            duration_seconds=row.get("duration_seconds"),
        )
        if record.record_id in records:
            raise ValueError(f"duplicate record_id {record.record_id!r}")
        key = (record.subset, record.condition, record.query_id)
        if key in seen_query_condition:
            raise ValueError(f"duplicate subset/condition/query tuple {key!r}")
        seen_query_condition.add(key)
        if require_audio_files and not Path(record.audio_path).is_file():
            raise FileNotFoundError(f"missing audio file: {record.audio_path}")
        records[record.record_id] = record
    if not records:
        raise ValueError(f"SQuTR audio manifest is empty: {path}")
    return records


def audit_query_split_leakage(
    splits: Mapping[str, Mapping[str, TextQuery] | Mapping[str, str]],
    *,
    example_limit: int = 20,
) -> Dict[str, object]:
    if len(splits) < 2:
        raise ValueError("at least two splits are required")
    normalized: Dict[str, Dict[str, str]] = {}
    for split, queries in splits.items():
        split_name = nonempty_string(split, field="split")
        normalized[split_name] = {}
        for query_id, query in queries.items():
            text = query.text if isinstance(query, TextQuery) else query
            if not isinstance(text, str):
                raise TypeError("split query values must be TextQuery or string")
            normalized[split_name][str(query_id)] = normalized_query_text(text)

    pairs = []
    split_names = sorted(normalized)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            id_overlap = sorted(set(normalized[left]).intersection(normalized[right]))
            left_by_text: Dict[str, list[str]] = {}
            right_by_text: Dict[str, list[str]] = {}
            for query_id, text in normalized[left].items():
                left_by_text.setdefault(text, []).append(query_id)
            for query_id, text in normalized[right].items():
                right_by_text.setdefault(text, []).append(query_id)
            shared_texts = sorted(
                text
                for text in set(left_by_text).intersection(right_by_text)
                if text
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "query_id_overlap_count": len(id_overlap),
                    "query_id_overlap_examples": id_overlap[:example_limit],
                    "normalized_text_overlap_count": len(shared_texts),
                    "normalized_text_overlap_examples": [
                        {
                            "text": text,
                            "left_ids": sorted(left_by_text[text])[:example_limit],
                            "right_ids": sorted(right_by_text[text])[:example_limit],
                        }
                        for text in shared_texts[:example_limit]
                    ],
                }
            )
    return {
        "split_counts": {
            split: len(queries) for split, queries in sorted(normalized.items())
        },
        "pairs": pairs,
        "has_query_id_overlap": any(
            pair["query_id_overlap_count"] > 0 for pair in pairs
        ),
        "has_normalized_text_overlap": any(
            pair["normalized_text_overlap_count"] > 0 for pair in pairs
        ),
    }

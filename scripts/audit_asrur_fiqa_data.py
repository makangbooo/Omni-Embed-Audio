#!/usr/bin/env python3
"""Audit FiQA train/dev and SQuTR-FiQA test identity before any encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    SQuTR_CONDITIONS,
    audit_query_split_leakage,
    load_corpus,
    load_qrels,
    load_squtr_audio_manifest,
    load_text_queries,
    normalized_query_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fiqa-root", type=Path, required=True)
    parser.add_argument("--squtr-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-audio-files", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    corpus_path = args.fiqa_root / "corpus.jsonl"
    queries_path = args.fiqa_root / "queries.jsonl"
    qrel_paths = {
        split: args.fiqa_root / "qrels" / f"{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    input_paths = [
        args.config,
        corpus_path,
        queries_path,
        *qrel_paths.values(),
        args.squtr_manifest,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    corpus = load_corpus(corpus_path)
    all_queries = load_text_queries(queries_path)
    qrels = {}
    split_queries = {}
    for split, path in qrel_paths.items():
        raw = load_unbounded_qrels(path)
        split_query_ids = set(raw)
        qrels[split] = load_qrels(
            path,
            query_ids=split_query_ids,
            document_ids=corpus,
        )
        missing_queries = sorted(split_query_ids - set(all_queries))
        if missing_queries:
            raise ValueError(f"{split} qrels reference missing queries: {missing_queries[:20]}")
        split_queries[split] = {
            query_id: all_queries[query_id] for query_id in split_query_ids
        }

    squtr_records = load_squtr_audio_manifest(
        args.squtr_manifest,
        require_audio_files=args.require_audio_files,
    )
    fiqa_records = [
        record
        for record in squtr_records.values()
        if record.subset.casefold() == "fiqa"
    ]
    if not fiqa_records:
        raise ValueError("SQuTR manifest contains no FiQA records")
    squtr_by_condition = {}
    for condition in SQuTR_CONDITIONS:
        records = [record for record in fiqa_records if record.condition == condition]
        squtr_by_condition[condition] = {
            record.query_id: record.original_query_text for record in records
        }
        if len(squtr_by_condition[condition]) != len(records):
            raise ValueError(f"duplicate FiQA query IDs under {condition}")

    expected = config["datasets"]["fiqa"]
    violations = []
    observed_counts = {
        "corpus": len(corpus),
        "train_queries": len(split_queries["train"]),
        "dev_queries": len(split_queries["dev"]),
        "test_queries": len(split_queries["test"]),
        **{
            f"squtr_{condition}_queries": len(values)
            for condition, values in squtr_by_condition.items()
        },
    }
    expected_counts = {
        "corpus": expected["documents_expected"],
        "train_queries": expected["train_queries_expected"],
        "dev_queries": expected["dev_queries_expected"],
        "test_queries": expected["test_queries_expected"],
        **{
            f"squtr_{condition}_queries": expected["test_queries_expected"]
            for condition in SQuTR_CONDITIONS
        },
    }
    for name, expected_count in expected_counts.items():
        if observed_counts[name] != expected_count:
            violations.append(
                f"{name}: observed {observed_counts[name]}, expected {expected_count}"
            )

    test_ids = set(split_queries["test"])
    for condition, values in squtr_by_condition.items():
        if set(values) != test_ids:
            violations.append(
                f"{condition}: SQuTR/MTEB test query ID sets differ "
                f"(missing={len(test_ids - set(values))}, extra={len(set(values) - test_ids)})"
            )
    condition_text_mismatches = {}
    for condition, values in squtr_by_condition.items():
        mismatches = [
            query_id
            for query_id, text in values.items()
            if query_id in all_queries
            and normalized_query_text(text)
            != normalized_query_text(all_queries[query_id].text)
        ]
        condition_text_mismatches[condition] = mismatches[:20]
        if mismatches:
            violations.append(
                f"{condition}: {len(mismatches)} normalized transcript/query mismatches"
            )

    leakage = audit_query_split_leakage(
        {
            "fiqa_train": split_queries["train"],
            "fiqa_dev": split_queries["dev"],
            "squtr_fiqa_test": squtr_by_condition["clean"],
        }
    )
    if leakage["has_query_id_overlap"]:
        violations.append("train/dev/test query ID leakage detected")
    if leakage["has_normalized_text_overlap"]:
        violations.append("train/dev/test normalized query-text leakage detected")

    report = {
        "schema_version": 1,
        "status": "complete" if not violations else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "qrel_pair_counts": {
            split: sum(len(documents) for documents in values.values())
            for split, values in qrels.items()
        },
        "empty_constructed_documents": sum(
            not document.constructed_text for document in corpus.values()
        ),
        "condition_text_mismatch_examples": condition_text_mismatches,
        "split_leakage": leakage,
        "violations": violations,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "command": sys.argv,
        "input_sha256": {
            str(path.resolve()): sha256(path) for path in input_paths
        },
    }
    with (args.output_dir / "fiqa_data_audit.json").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if not violations else 1)

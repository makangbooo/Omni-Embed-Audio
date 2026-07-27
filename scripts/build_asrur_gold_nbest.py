#!/usr/bin/env python3
"""Build the single-hypothesis gold-transcript artifact for the U2 upper bound."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (  # noqa: E402
    assert_cache_compatible,
    build_cache_manifest,
    file_record,
    load_cache_manifest,
    write_cache_manifest_once,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    load_text_queries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-query-count", type=int, required=True)
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def build_rows(
    *,
    queries_path: Path,
    qrels_path: Path,
    expected_query_count: int,
) -> list[dict]:
    if expected_query_count <= 0:
        raise ValueError("expected query count must be positive")
    queries = load_text_queries(queries_path)
    qrels = load_unbounded_qrels(qrels_path)
    query_ids = sorted(qrels)
    missing = sorted(set(query_ids) - set(queries))
    if missing:
        raise ValueError(f"qrels query IDs are absent from query input: {missing[:20]}")
    if len(query_ids) != expected_query_count:
        raise ValueError(
            f"expected {expected_query_count} qrels queries, found {len(query_ids)}"
        )
    return [
        {
            "query_id": query_id,
            "no_speech_probability": None,
            "hypotheses": [
                {
                    "rank": 1,
                    "text": queries[query_id].text,
                    # Gold text has no ASR posterior. Null prevents it from
                    # masquerading as a measured sequence probability.
                    "sequence_score": None,
                    "average_token_logprob": 0.0,
                    "valid_token_count": max(
                        1,
                        len(queries[query_id].text.split()),
                    ),
                }
            ],
            "source": "gold_text_upper_bound_not_inference_input",
        }
        for query_id in query_ids
    ]


def deterministic_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )


def write_once_or_verify(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rows = build_rows(
        queries_path=args.queries,
        qrels_path=args.qrels,
        expected_query_count=args.expected_query_count,
    )
    output = args.output_dir / "nbest.jsonl"
    write_once_or_verify(output, deterministic_jsonl(rows))
    manifest = build_cache_manifest(
        artifact_type="gold_transcript_single_hypothesis",
        dataset=args.dataset,
        split=args.split,
        inputs=[file_record(args.queries), file_record(args.qrels)],
        model_name="gold_text_upper_bound",
        model_revision="not_applicable",
        model_checkpoint=None,
        tokenizer=None,
        pooling=None,
        embedding_dim=None,
        max_length=None,
        dtype=None,
        normalization="none",
        seed=42,
        command=sys.argv,
        git_commit=git_output("rev-parse", "HEAD"),
        outputs=[file_record(output)],
        extra_identity={
            "query_count": len(rows),
            "hypothesis_count": 1,
            "gold_text_used_only_for_upper_bound": True,
            "asr_posterior_available": False,
        },
    )
    manifest_path = args.output_dir / "cache_manifest.json"
    if manifest_path.exists():
        existing = load_cache_manifest(manifest_path)
        assert_cache_compatible(manifest, existing)
        if existing["outputs"] != manifest["outputs"]:
            raise RuntimeError("existing gold N-best manifest output differs")
    else:
        write_cache_manifest_once(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "query_count": len(rows),
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

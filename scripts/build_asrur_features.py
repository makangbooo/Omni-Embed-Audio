#!/usr/bin/env python3
"""Build immutable gate features from cached OEA, Whisper, and CE artifacts."""

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

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (  # noqa: E402
    aggregate_nbest_scores,
    build_asr_uncertainty_features,
    softmax_proxy_posteriors,
)
from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    assemble_rerank_inputs,
    load_cross_encoder_scores,
    load_frozen_candidates,
    load_nbest,
    load_unbounded_qrels,
    write_feature_rows_once,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.features import (  # noqa: E402
    build_candidate_feature_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--cross-encoder", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy-temperature", type=float, default=1.0)
    parser.add_argument("--cross-encoder-temperature", type=float, default=1.0)
    parser.add_argument("--selection", type=Path)
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


def write_json_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def resolve_temperatures(args: argparse.Namespace) -> tuple[float, float, str]:
    if args.split != "test":
        return args.proxy_temperature, args.cross_encoder_temperature, "explicit_non_test"
    if args.selection is None:
        raise ValueError("test feature generation requires --selection from FiQA dev")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("selected_on_split") != "fiqa_dev":
        raise ValueError("test selection must be frozen on fiqa_dev")
    return (
        float(selection["proxy_temperature"]),
        float(selection["cross_encoder_temperature"]),
        "frozen_fiqa_dev_selection",
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    proxy_temperature, ce_temperature, temperature_source = resolve_temperatures(args)

    candidates = load_frozen_candidates(args.top100)
    nbest = load_nbest(args.nbest, expected_size=4)
    cross_encoder = load_cross_encoder_scores(args.cross_encoder)
    qrels = load_unbounded_qrels(args.qrels)
    inputs = assemble_rerank_inputs(candidates, nbest, cross_encoder)
    if set(inputs) != set(qrels):
        raise ValueError("cached inputs and qrels query sets must match exactly")

    rows = []
    for query_id in sorted(inputs):
        query = inputs[query_id]
        posteriors = softmax_proxy_posteriors(
            query.proxy_logits,
            temperature=proxy_temperature,
        )
        asr_scores = aggregate_nbest_scores(
            query.cross_encoder_scores,
            mode="proxy_posterior_logsumexp",
            proxy_posteriors=posteriors,
            cross_encoder_temperature=ce_temperature,
        )
        uncertainty = build_asr_uncertainty_features(
            query.hypotheses,
            query.proxy_logits,
            proxy_temperature=proxy_temperature,
            top1_average_token_logprob=query.top1_average_token_logprob,
            no_speech_probability=query.no_speech_probability,
        )
        rows.extend(
            build_candidate_feature_rows(
                query_id=query_id,
                candidate_ids=query.candidate_ids,
                oea_scores=query.oea_scores,
                asr_scores=asr_scores,
                qrels=qrels[query_id],
                uncertainty=uncertainty,
                normalization=config["candidate_protocol"]["normalization"],
                overlap_mode=config["candidate_protocol"]["overlap_mode"],
            )
        )

    feature_path = args.output_dir / "features.jsonl"
    write_feature_rows_once(feature_path, rows)
    write_json_once(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "asrur_gate_features",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "query_count": len(inputs),
            "candidate_row_count": len(rows),
            "proxy_temperature": proxy_temperature,
            "cross_encoder_temperature": ce_temperature,
            "temperature_source": temperature_source,
            "normalization": config["candidate_protocol"]["normalization"],
            "overlap_mode": config["candidate_protocol"]["overlap_mode"],
            "inputs": {
                str(path.resolve()): sha256(path)
                for path in (args.config, args.top100, args.nbest, args.cross_encoder, args.qrels)
            },
            "output": {
                "path": str(feature_path.resolve()),
                "sha256": sha256(feature_path),
            },
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short", "--untracked-files=all"),
            "command": sys.argv,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "query_count": len(inputs),
                "candidate_row_count": len(rows),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )

#!/usr/bin/env python3
"""Train query- and candidate-level gates from frozen FiQA train/dev features."""

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
    load_feature_rows,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.features import (  # noqa: E402
    FEATURE_NAMES,
    QUERY_ONLY_FEATURE_NAMES,
    ablation_feature_names,
)
from AudioRetrieval.asr_uncertainty_reranking.gate import (  # noqa: E402
    build_training_groups,
    retain_positive_candidate_queries,
    train_candidate_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


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

    train_rows = load_feature_rows(args.train_features)
    dev_rows = load_feature_rows(args.dev_features)
    train_ids = {row.query_id for row in train_rows}
    dev_ids = {row.query_id for row in dev_rows}
    overlap = sorted(train_ids.intersection(dev_ids))
    if overlap:
        raise ValueError(f"train/dev query ID leakage: {overlap[:20]}")
    if any(
        row.feature_names != FEATURE_NAMES
        for row in train_rows + dev_rows
    ):
        raise ValueError("feature schema does not match the locked main protocol")
    train_rows, excluded_train_ids = retain_positive_candidate_queries(train_rows)
    dev_rows, excluded_dev_ids = retain_positive_candidate_queries(dev_rows)

    gate_config = config["gate"]
    train_groups = build_training_groups(
        train_rows,
        group_size=gate_config["group_size"],
    )
    dev_groups = build_training_groups(dev_rows, group_size=None)
    jobs = {
        "query_gate": QUERY_ONLY_FEATURE_NAMES,
        "candidate_gate": FEATURE_NAMES,
        "A4_no_asr_confidence": ablation_feature_names("no_asr_confidence"),
        "A5_no_nbest_entropy": ablation_feature_names("no_nbest_entropy"),
        "A6_no_rank_disagreement": ablation_feature_names("no_rank_disagreement"),
    }
    summary = {}
    for seed in gate_config["seeds"]:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir()
        summary[str(seed)] = {}
        for job_name, feature_names in jobs.items():
            model, history = train_candidate_gate(
                train_groups,
                dev_groups,
                feature_names=feature_names,
                hidden_dim=gate_config["hidden_dim"],
                learning_rate=gate_config["learning_rate"],
                weight_decay=gate_config["weight_decay"],
                max_epochs=gate_config["max_epochs"],
                patience=gate_config["patience"],
                seed=seed,
            )
            write_json_once(seed_dir / f"{job_name}.model.json", model.as_dict())
            write_json_once(seed_dir / f"{job_name}.history.json", history)
            summary[str(seed)][job_name] = {
                "best_epoch": history["best_epoch"],
                "best_dev_listwise_loss": history["best_dev_listwise_loss"],
            }

    write_json_once(
        args.output_dir / "training_summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "train_query_count": len(train_groups),
            "dev_query_count": len(dev_groups),
            "excluded_zero_positive_candidate_queries": {
                "train": list(excluded_train_ids),
                "dev": list(excluded_dev_ids),
            },
            "seeds": gate_config["seeds"],
            "jobs": summary,
            "input_sha256": {
                "config": sha256(args.config),
                "train_features": sha256(args.train_features),
                "dev_features": sha256(args.dev_features),
            },
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short", "--untracked-files=all"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "scope": "small_gate_only_upstream_models_frozen",
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "seeds": gate_config["seeds"],
                "jobs": sorted(jobs),
            },
            indent=2,
        )
    )

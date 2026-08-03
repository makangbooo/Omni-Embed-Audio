#!/usr/bin/env python3
"""Collect compact evidence from completed OEA negative UIQ model runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METRICS = ("R@5", "R@10", "Delta-Rank", "HNSR", "HNSR@10", "TFR", "TFR-HN@10")
DATASETS = ("clotho", "audiocaps", "mecat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--variant", action="append", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_variant(results_root: Path, variant: str, expected_commit: str) -> dict:
    candidates = sorted(
        results_root.glob(f"{variant}_negative_uiq_three_dataset_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected = None
    reports = None
    for candidate in candidates:
        exit_path = candidate / "exit_code.txt"
        paths = {name: candidate / "metrics" / name / "metrics.json" for name in DATASETS}
        if not exit_path.is_file() or exit_path.read_text().strip() != "0":
            continue
        if not all(path.is_file() for path in paths.values()):
            continue
        loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
        if any(report.get("git_commit") != expected_commit for report in loaded.values()):
            continue
        selected = candidate
        reports = loaded
        break
    if selected is None or reports is None:
        raise FileNotFoundError(
            f"no complete {variant} run found for commit {expected_commit}"
        )

    datasets = {}
    for name in DATASETS:
        report = reports[name]
        path = selected / "metrics" / name / "metrics.json"
        datasets[name] = {
            "candidate_count": report["candidate_count"],
            "query_count": report["evaluated_query_count"],
            "metrics": {metric: report["metrics"][metric] for metric in METRICS},
            "metrics_sha256": sha256(path),
            "audio_npz_sha256": report["inputs"]["audio_npz"]["sha256"],
            "query_npz_sha256": report["inputs"]["query_npz"]["sha256"],
            "pairing_sha256": report["inputs"]["pairing_jsonl"]["sha256"],
        }
    means = {
        metric: sum(datasets[name]["metrics"][metric] for name in DATASETS) / 3
        for metric in METRICS
    }
    return {
        "variant": variant,
        "model": reports["clotho"]["model"],
        "run": str(selected),
        "git_commit": expected_commit,
        "status": "complete",
        "datasets": datasets,
        "three_dataset_mean": means,
        "artifact_manifest_sha256": sha256(selected / "artifact_sha256.txt"),
    }


def main() -> int:
    args = parse_args()
    for variant in args.variant:
        result = collect_variant(
            args.results_root.resolve(), variant, args.expected_git_commit
        )
        print("OEA_NEGATIVE_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

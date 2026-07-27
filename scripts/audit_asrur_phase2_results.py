#!/usr/bin/env python3
"""Audit four-condition Phase-2 outputs and apply the locked Go/No-Go checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.experiment import (  # noqa: E402
    phase2_go_no_go,
)

REQUIRED_METHODS = {
    "B1_whisper_1best_bge_dense",
    "B2_original_omni",
    "B3_oea",
    "U1_gold_bge_dense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_zero_exit(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if raw != "0":
        raise ValueError(f"{path} must contain exit code 0, observed {raw!r}")


def _mean_metrics(
    payload: Mapping[str, Any],
    *,
    method: str,
    expected_queries: int,
) -> Mapping[str, Any]:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != REQUIRED_METHODS:
        raise ValueError("Phase-2 metrics must contain exactly B1/B2/B3/U1")
    method_payload = evaluations[method]
    if not isinstance(method_payload, Mapping):
        raise TypeError(f"{method} evaluation must be an object")
    if method_payload.get("num_queries") != expected_queries:
        raise ValueError(f"{method} query count does not match locked FiQA test")
    if method_payload.get("scale") != "fraction":
        raise ValueError(f"{method} metrics must use fraction scale")
    mean = method_payload.get("mean")
    if not isinstance(mean, Mapping):
        raise TypeError(f"{method} mean metrics must be an object")
    return mean


def audit_phase2(
    *,
    config_path: Path,
    run_dir: Path,
    result_root: Path,
) -> dict[str, Any]:
    config = load_main_experiment_config(config_path)
    conditions = list(config["datasets"]["fiqa"]["conditions"])
    expected_queries = int(config["datasets"]["fiqa"]["test_queries_expected"])
    thresholds = config["go_no_go"]

    require_zero_exit(run_dir / "wrapper_exit_code.txt")
    completion_path = run_dir / "completion_manifest.json"
    completion = load_json_object(completion_path)
    if completion.get("status") != "complete":
        raise ValueError("Phase-2 completion manifest is not complete")

    declared_metrics = completion.get("condition_metrics")
    if not isinstance(declared_metrics, Mapping) or set(declared_metrics) != set(
        conditions
    ):
        raise ValueError("completion manifest condition set is not protocol-locked")

    per_condition: dict[str, Any] = {}
    input_paths = [config_path, completion_path, run_dir / "wrapper_exit_code.txt"]
    for condition in conditions:
        metrics_path = result_root / condition / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        declared_path = Path(str(declared_metrics[condition]))
        if declared_path.resolve() != metrics_path.resolve():
            raise ValueError(
                f"{condition} completion manifest points to unexpected metrics path"
            )
        payload = load_json_object(metrics_path)
        if payload.get("scale") != "fraction":
            raise ValueError(f"{condition} top-level scale must be fraction")
        oea = _mean_metrics(
            payload,
            method="B3_oea",
            expected_queries=expected_queries,
        )
        original_omni = _mean_metrics(
            payload,
            method="B2_original_omni",
            expected_queries=expected_queries,
        )
        _mean_metrics(
            payload,
            method="B1_whisper_1best_bge_dense",
            expected_queries=expected_queries,
        )
        _mean_metrics(
            payload,
            method="U1_gold_bge_dense",
            expected_queries=expected_queries,
        )
        oracle_payload = payload.get("oea_candidate_oracle")
        if not isinstance(oracle_payload, Mapping):
            raise TypeError(f"{condition} OEA candidate oracle must be an object")
        if oracle_payload.get("num_queries") != expected_queries:
            raise ValueError(f"{condition} oracle query count mismatch")
        oracle = oracle_payload.get("mean")
        if not isinstance(oracle, Mapping):
            raise TypeError(f"{condition} oracle mean metrics must be an object")
        per_condition[condition] = {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256(metrics_path),
            "decision": phase2_go_no_go(
                oea_metrics=oea,
                original_omni_metrics=original_omni,
                oracle_metrics=oracle,
                minimum_recall_at_100=float(
                    thresholds["minimum_oea_recall_at_100"]
                ),
                minimum_oracle_ndcg_gain=float(
                    thresholds["minimum_oracle_ndcg_gain"]
                ),
            ),
        }
        input_paths.append(metrics_path)

    all_go = all(
        item["decision"]["decision"] == "GO" for item in per_condition.values()
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "SQuTR-FiQA",
        "conditions": conditions,
        "query_count_per_condition": expected_queries,
        "overall_decision": (
            "GO" if all_go else "NO_GO_REQUIRES_USER_DECISION"
        ),
        "overall_rule": (
            "GO only when every locked acoustic condition passes every "
            "numeric Phase-2 check; any failure requires user review and "
            "does not authorize changing candidate generation."
        ),
        "candidate_generation_change_authorized": False,
        "per_condition": per_condition,
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output(
                "status",
                "--short",
                "--untracked-files=all",
            ),
            "input_sha256": {
                str(path.resolve()): sha256(path) for path in input_paths
            },
            "command": sys.argv,
        },
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output}")
    result = audit_phase2(
        config_path=args.config,
        run_dir=args.run_dir,
        result_root=args.result_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "overall_decision": result["overall_decision"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze the BGE query-template choice using FiQA dev rankings only."""

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
    load_frozen_candidates,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.candidates import (  # noqa: E402
    rank_scores,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    evaluate_rankings,
)


TEMPLATES = ("none", "bge_retrieval")
TIE_BREAK_PRIORITY = {"bge_retrieval": 0, "none": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking",
        action="append",
        required=True,
        metavar="TEMPLATE=PATH",
    )
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
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


def parse_rankings(values: list[str]) -> dict[str, Path]:
    rankings: dict[str, Path] = {}
    for raw in values:
        template, separator, raw_path = raw.partition("=")
        if not separator or template not in TEMPLATES or not raw_path:
            raise ValueError(
                "--ranking must be exactly none=PATH or bge_retrieval=PATH"
            )
        if template in rankings:
            raise ValueError(f"duplicate BGE query template: {template}")
        rankings[template] = Path(raw_path).resolve()
    if set(rankings) != set(TEMPLATES):
        raise ValueError("both BGE query templates must be evaluated on FiQA dev")
    return rankings


def expected_input_sha256(
    qrels_path: Path,
    rankings: dict[str, Path],
) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in [qrels_path, *(rankings[name] for name in TEMPLATES)]
    }


def validate_reusable(
    selection_path: Path,
    *,
    input_sha256: dict[str, str],
    git_commit: str,
) -> bool:
    if not selection_path.is_file():
        return False
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "frozen"
        or value.get("selected_on_split") != "fiqa_dev"
        or value.get("test_qrels_used") is not False
        or value.get("input_sha256") != input_sha256
        or value.get("git_commit") != git_commit
    ):
        raise RuntimeError("existing BGE template selection has incompatible identity")
    print(json.dumps({"status": "reused", "selection": value}, indent=2))
    return True


def select_query_template(
    rankings: dict[str, Path],
    qrels_path: Path,
) -> tuple[str, dict[str, dict]]:
    qrels = load_unbounded_qrels(qrels_path)
    evaluations = {}
    for template in TEMPLATES:
        records = load_frozen_candidates(rankings[template])
        if set(records) != set(qrels):
            raise ValueError(f"{template}: ranking/qrels query sets differ")
        ranked = {
            query_id: rank_scores(record.candidate_ids, record.scores)
            for query_id, record in records.items()
        }
        evaluations[template] = evaluate_rankings(ranked, qrels)

    selected = min(
        TEMPLATES,
        key=lambda template: (
            -evaluations[template]["mean"]["nDCG@10"],
            TIE_BREAK_PRIORITY[template],
        ),
    )
    return selected, evaluations


def main() -> int:
    args = parse_args()
    rankings = parse_rankings(args.ranking)
    qrels_path = args.qrels.resolve()
    output_dir = args.output_dir.resolve()
    selection_path = output_dir / "frozen_selection.json"
    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short", "--untracked-files=all")
    if git_status:
        raise RuntimeError("BGE dev selection requires a clean Git worktree")
    input_sha256 = expected_input_sha256(qrels_path, rankings)
    if validate_reusable(
        selection_path,
        input_sha256=input_sha256,
        git_commit=git_commit,
    ):
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output_dir}")

    selected, evaluations = select_query_template(rankings, qrels_path)
    report = {
        "schema_version": 1,
        "status": "frozen",
        "selected_on_split": "fiqa_dev",
        "test_qrels_used": False,
        "selection_metric": "nDCG@10",
        "selection_mode": "maximum_dev_metric",
        "tie_break": "prefer_bge_retrieval_instruction",
        "selected_query_template": selected,
        "evaluations": evaluations,
        "input_sha256": input_sha256,
        "git_commit": git_commit,
        "git_status_short": git_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = selection_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(selection_path)
    print(json.dumps({"status": "complete", "selection": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

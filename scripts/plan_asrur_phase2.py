#!/usr/bin/env python3
"""Build the auditable Phase-2 FiQA execution plan without loading a model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (  # noqa: E402
    file_record,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    SQuTR_CONDITIONS,
    load_corpus,
    load_squtr_audio_manifest,
    load_text_queries,
    squtr_subset_name,
)


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def phase2_workload(
    *,
    document_count: int,
    dev_query_count: int,
    test_query_count: int,
    condition_count: int,
    candidate_depth: int,
    nbest_size: int,
) -> dict[str, int]:
    values = {
        "document_count": document_count,
        "dev_query_count": dev_query_count,
        "test_query_count": test_query_count,
        "condition_count": condition_count,
        "candidate_depth": candidate_depth,
        "nbest_size": nbest_size,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values.values()
    ):
        raise ValueError("all Phase-2 workload counts must be positive integers")
    audio_queries = test_query_count * condition_count
    return {
        **values,
        "audio_query_count": audio_queries,
        "whisper_hypothesis_count": audio_queries * nbest_size,
        "bge_document_encoding_count": document_count,
        "bge_dev_query_encoding_count_two_templates": dev_query_count * 2,
        "bge_gold_test_query_encoding_count": test_query_count,
        "bge_asr_1best_query_encoding_count": audio_queries,
        "oea_document_encoding_count": document_count,
        "oea_audio_encoding_count": audio_queries,
        "original_omni_document_encoding_count": document_count,
        "original_omni_audio_encoding_count": audio_queries,
        "phase3_one_best_ce_pair_count": audio_queries * candidate_depth,
        "phase3_four_best_ce_pair_count": (
            audio_queries * candidate_depth * nbest_size
        ),
        "oea_dense_dot_product_count": audio_queries * document_count,
        "original_omni_dense_dot_product_count": audio_queries * document_count,
        "bge_asr_dense_dot_product_count": audio_queries * document_count,
        "bge_gold_dense_dot_product_count": test_query_count * document_count,
    }


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe model relative path: {value!r}")
    return path


def fast_model_inventory(
    *,
    manifest: Mapping[str, Any],
    model_root: Path,
) -> dict[str, Any]:
    assets = []
    for specification in manifest["assets"]:
        destination = model_root / safe_relative_path(specification["local_subdir"])
        files = []
        for expected in specification["expected_files"]:
            path = destination / safe_relative_path(expected["path"])
            observed_size = path.stat().st_size if path.is_file() else None
            expected_size = int(expected["size_bytes"])
            files.append(
                {
                    "path": str(path),
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "status": (
                        "size_present"
                        if observed_size == expected_size
                        else "missing_or_wrong_size"
                    ),
                    "strict_sha256_status": "not_checked_by_fast_plan",
                }
            )
        assets.append(
            {
                "name": specification["name"],
                "revision": specification["revision"],
                "destination": str(destination),
                "status": (
                    "size_present"
                    if all(value["status"] == "size_present" for value in files)
                    else "incomplete"
                ),
                "files": files,
            }
        )
    return {
        "status": (
            "size_present"
            if all(value["status"] == "size_present" for value in assets)
            else "incomplete"
        ),
        "claim_boundary": (
            "Fast size gate only. Formal GPU execution additionally requires "
            "the separate D2-D4 strict SHA256 audit with exit code 0."
        ),
        "assets": assets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fiqa-root", type=Path, required=True)
    parser.add_argument("--squtr-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-resource-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [
        args.config,
        args.fiqa_root / "corpus.jsonl",
        args.fiqa_root / "queries.jsonl",
        args.fiqa_root / "qrels" / "dev.jsonl",
        args.fiqa_root / "qrels" / "test.jsonl",
        args.squtr_manifest,
        args.model_resource_manifest,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Phase-2 plan: {args.output}")
    config = load_main_experiment_config(args.config)
    corpus = load_corpus(paths[1])
    all_queries = load_text_queries(paths[2])
    dev_qrels = load_unbounded_qrels(paths[3])
    test_qrels = load_unbounded_qrels(paths[4])
    if not set(dev_qrels).issubset(all_queries) or not set(test_qrels).issubset(all_queries):
        raise ValueError("FiQA qrels reference a query absent from queries.jsonl")
    audio = [
        value
        for value in load_squtr_audio_manifest(
            args.squtr_manifest,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(value.subset) == "fiqa"
    ]
    condition_counts = {
        condition: len(
            {
                value.query_id
                for value in audio
                if value.condition == condition
            }
        )
        for condition in SQuTR_CONDITIONS
    }
    expected = config["datasets"]["fiqa"]
    observed = {
        "documents": len(corpus),
        "dev_queries": len(dev_qrels),
        "test_queries": len(test_qrels),
        "condition_counts": condition_counts,
    }
    violations = []
    if observed["documents"] != expected["documents_expected"]:
        violations.append("FiQA document count mismatch")
    if observed["dev_queries"] != expected["dev_queries_expected"]:
        violations.append("FiQA dev query count mismatch")
    if observed["test_queries"] != expected["test_queries_expected"]:
        violations.append("FiQA test query count mismatch")
    if any(
        value != expected["test_queries_expected"]
        for value in condition_counts.values()
    ):
        violations.append("SQuTR-FiQA condition count mismatch")
    resource_manifest = json.loads(
        args.model_resource_manifest.read_text(encoding="utf-8")
    )
    model_inventory = fast_model_inventory(
        manifest=resource_manifest,
        model_root=args.model_root.resolve(),
    )
    workload = phase2_workload(
        document_count=len(corpus),
        dev_query_count=len(dev_qrels),
        test_query_count=len(test_qrels),
        condition_count=len(SQuTR_CONDITIONS),
        candidate_depth=config["candidate_protocol"]["depth"],
        nbest_size=config["models"]["asr"]["num_return_sequences"],
    )
    jobs = [
        {
            "id": "P2-01",
            "name": "BGE corpus and FiQA-dev two-template encoding",
            "purpose": "select query instruction once on FiQA dev",
            "gpu_required": True,
        },
        {
            "id": "P2-02",
            "name": "Whisper four-beam SQuTR-FiQA transcription",
            "purpose": (
                "produce four independent 648-query condition caches with "
                "FiQA qrels query IDs, 1-best, 4-best, and uncertainty proxies"
            ),
            "gpu_required": True,
        },
        {
            "id": "P2-03",
            "name": "BGE selected-template ASR and gold query encoding",
            "purpose": "B1 and U1 dense baselines",
            "gpu_required": True,
        },
        {
            "id": "P2-04",
            "name": "OEA-Nemo3B(+Cl) FiQA corpus/audio encoding",
            "purpose": (
                "B3 and immutable Top-100 candidates, generated independently "
                "for each acoustic condition"
            ),
            "gpu_required": True,
        },
        {
            "id": "P2-05",
            "name": "original Omni-Embed-Nemotron FiQA corpus/audio encoding",
            "purpose": (
                "B2 original-backbone comparison, generated independently for "
                "each acoustic condition"
            ),
            "gpu_required": True,
        },
        {
            "id": "P2-06",
            "name": "exact cached dense Top-100 generation",
            "purpose": "B1/B2/B3/U1 rankings",
            "gpu_required": False,
        },
        {
            "id": "P2-07",
            "name": "metrics, candidate oracle, and Go/No-Go",
            "purpose": "decide whether reranking may proceed",
            "gpu_required": False,
        },
    ]
    report = {
        "schema_version": 1,
        "status": (
            "ready_for_strict_model_audit"
            if not violations and model_inventory["status"] == "size_present"
            else "blocked"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "gpu_execution_performed": False,
        "model_loaded": False,
        "network_access_performed": False,
        "inputs": [file_record(path) for path in paths],
        "observed": observed,
        "violations": violations,
        "fast_model_inventory": model_inventory,
        "workload": workload,
        "jobs": jobs,
        "go_no_go": config["go_no_go"],
        "required_before_gpu": [
            "clean worktree at one pushed commit",
            "D2-D4 strict SHA256 audit exit code 0",
            "canonical OEA model lock already complete",
            "separate user approval for the exact Phase-2 GPU command",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready_for_strict_model_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())

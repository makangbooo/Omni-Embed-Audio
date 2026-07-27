from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_reproduction_summary import (
    DEFAULT_OBSERVATIONS,
    DEFAULT_OUTPUT,
    DEFAULT_PAPER_REGISTRY,
    REPOSITORY_ROOT,
    SUMMARY_COLUMNS,
    build_summary,
    read_jsonl,
    validate_observations,
    validate_paper_registry,
)


PAPER_PDF_SHA256 = (
    "e76bbd96c82ac78568ed75eaedcbc62e389358c4bb226457ac45b55ab63cfe38"
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_row(**overrides: object) -> dict:
    row = {
        "paper_metric_id": "metric_r1",
        "paper_table": "Table 2",
        "model": "Synthetic model",
        "dataset": "Synthetic dataset",
        "task": "T2A",
        "metric": "R@1",
        "paper_value": "20.00",
        "unit": "percent",
        "sort_order": 1,
        "source": "[PAPER]",
        "paper_pdf_sha256": PAPER_PDF_SHA256,
        "pdf_page": 6,
        "note": "synthetic paper metric",
    }
    row.update(overrides)
    return row


def evidence_identity(path: Path, json_path: list[str | int]) -> dict:
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "json_path": json_path,
    }


def observation_row(evidence: dict, **overrides: object) -> dict:
    row = {
        "observation_id": "observation_r1",
        "paper_metric_id": "metric_r1",
        "experiment": "synthetic_experiment",
        "reproduced_value": "25.00",
        "seed": 42,
        "checkpoint": "synthetic/checkpoint@revision/file.pt#sha256:" + "a" * 64,
        "status": "close",
        "notes": "Explicitly reviewed synthetic comparison.",
        "evidence": evidence,
    }
    row.update(overrides)
    return row


class BuildReproductionSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = REPOSITORY_ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_inputs(
        self,
        *,
        paper_rows: list[dict] | None = None,
        observation_rows: list[dict] | None = None,
    ) -> tuple[Path, Path, Path, Path, Path]:
        evidence_path = self.root / "metrics.json"
        write_json(evidence_path, {"metrics": {"R@1": 25.0}})
        evidence = evidence_identity(evidence_path, ["metrics", "R@1"])
        paper_path = self.root / "paper.jsonl"
        observations_path = self.root / "observations.jsonl"
        output_path = self.root / "summary.csv"
        audit_path = self.root / "summary.audit.json"
        write_jsonl(paper_path, paper_rows or [paper_row()])
        write_jsonl(
            observations_path,
            observation_rows or [observation_row(evidence)],
        )
        return (
            paper_path,
            observations_path,
            output_path,
            audit_path,
            evidence_path,
        )

    def test_fixed_registry_values_and_pdf_pages(self) -> None:
        rows = read_jsonl(DEFAULT_PAPER_REGISTRY, "paper registry")
        registry = validate_paper_registry(rows)
        self.assertEqual(len(registry), 910)
        expected = {
            "table2_oea_qwen3b_cl_clotho_t2a_r1": ("22.87", 6),
            "table2_oea_qwen3b_cl_clotho_t2a_r5": ("49.78", 6),
            "table2_oea_qwen3b_cl_clotho_t2a_r10": ("63.25", 6),
            "table3_oea_qwen3b_cl_clotho_t2t_r1": ("64.52", 7),
            "table3_oea_qwen3b_cl_clotho_t2t_r5": ("75.25", 7),
            "table3_oea_qwen3b_cl_clotho_t2t_r10": ("79.71", 7),
            "table12_oea_qwen3b_cl_clotho_question_r1": ("25.74", 15),
            "table12_oea_qwen3b_cl_clotho_question_r5": ("54.26", 15),
            "table12_oea_qwen3b_cl_clotho_question_r10": ("66.79", 15),
            "table13_oea_qwen3b_cl_clotho_imperative_r1": ("25.45", 16),
            "table13_oea_qwen3b_cl_clotho_imperative_r5": ("55.69", 16),
            "table13_oea_qwen3b_cl_clotho_imperative_r10": ("67.56", 16),
            "table14_oea_qwen3b_cl_clotho_paraphrase_r1": ("27.56", 16),
            "table14_oea_qwen3b_cl_clotho_paraphrase_r5": ("55.50", 16),
            "table14_oea_qwen3b_cl_clotho_paraphrase_r10": ("70.81", 16),
            "table15_oea_qwen3b_cl_clotho_keyphrase_r1": ("27.66", 17),
            "table15_oea_qwen3b_cl_clotho_keyphrase_r5": ("57.99", 17),
            "table15_oea_qwen3b_cl_clotho_keyphrase_r10": ("71.87", 17),
        }
        observed = {
            metric_id: (
                registry[metric_id]["paper_value"],
                registry[metric_id]["pdf_page"],
            )
            for metric_id in expected
        }
        self.assertEqual(observed, expected)
        self.assertEqual(
            {row["paper_pdf_sha256"] for row in registry.values()},
            {PAPER_PDF_SHA256},
        )

    def test_builds_summary_and_signed_relative_delta(self) -> None:
        paper, observations, output, audit, _ = self.make_inputs()
        result = build_summary(paper, observations, output, audit)
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), SUMMARY_COLUMNS)
        self.assertEqual(rows[0]["paper_value"], "20.00")
        self.assertEqual(rows[0]["reproduced_value"], "25")
        self.assertEqual(rows[0]["absolute_delta"], "5")
        self.assertEqual(rows[0]["relative_delta"], "25")
        self.assertEqual(rows[0]["status"], "close")
        self.assertEqual(result["status_counts"], {"close": 1})
        self.assertEqual(result["output"]["sha256"], sha256(output))

    def test_negative_relative_delta_remains_signed(self) -> None:
        evidence_path = self.root / "negative_metrics.json"
        write_json(evidence_path, {"metrics": {"R@1": 15.0}})
        evidence = evidence_identity(evidence_path, ["metrics", "R@1"])
        inputs = self.make_inputs(
            observation_rows=[
                observation_row(evidence, reproduced_value="15.00")
            ]
        )
        build_summary(inputs[0], inputs[1], inputs[2], inputs[3])
        with inputs[2].open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["absolute_delta"], "5")
        self.assertEqual(row["relative_delta"], "-25")

    def test_exact_status_rejects_nonmatching_value(self) -> None:
        paper, observations, _, _, _ = self.make_inputs()
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        rows = read_jsonl(observations, "observations")
        rows[0]["status"] = "exact"
        with self.assertRaisesRegex(ValueError, "exact status requires"):
            validate_observations(rows, registry)

    def test_evidence_hash_drift_is_rejected(self) -> None:
        paper, observations, _, _, evidence = self.make_inputs()
        evidence.write_text("changed\n", encoding="utf-8")
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        with self.assertRaisesRegex(ValueError, "evidence size mismatch"):
            validate_observations(
                read_jsonl(observations, "observations"), registry
            )

    def test_evidence_path_escape_is_rejected(self) -> None:
        paper, observations, _, _, _ = self.make_inputs()
        rows = read_jsonl(observations, "observations")
        rows[0]["evidence"]["path"] = "../outside.json"
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            validate_observations(rows, registry)

    def test_unit_ranges_and_count_integrality_are_enforced(self) -> None:
        invalid_score = paper_row(unit="score_1_to_5", paper_value="0.99")
        with self.assertRaisesRegex(ValueError, "score range"):
            validate_paper_registry([invalid_score])
        invalid_count = paper_row(unit="count", paper_value="1.5")
        with self.assertRaisesRegex(ValueError, "integer count"):
            validate_paper_registry([invalid_count])
        invalid_standard_deviation = paper_row(
            unit="score_standard_deviation", paper_value="-0.01"
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            validate_paper_registry([invalid_standard_deviation])

    def test_exact_status_rejects_qualified_paper_value(self) -> None:
        paper, observations, _, _, evidence = self.make_inputs(
            paper_rows=[
                paper_row(
                    paper_value="25.0",
                    unit="gigabytes",
                    metric="Peak GPU memory (GB)",
                    paper_value_qualifier="approximately",
                )
            ]
        )
        rows = read_jsonl(observations, "observations")
        rows[0].update(status="exact", reproduced_value="25.0")
        rows[0]["evidence"] = evidence_identity(
            evidence, ["metrics", "R@1"]
        )
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        with self.assertRaisesRegex(ValueError, "qualified paper value"):
            validate_observations(rows, registry)

    def test_blocked_status_requires_null_and_needs_no_evidence(self) -> None:
        paper, observations, _, _, _ = self.make_inputs()
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        row = read_jsonl(observations, "observations")[0]
        row.update(status="blocked", reproduced_value=None)
        row.pop("evidence")
        validated = validate_observations([row], registry)
        self.assertIsNone(validated[0]["_reproduced_decimal"])
        row["reproduced_value"] = 20
        with self.assertRaisesRegex(ValueError, "must have null"):
            validate_observations([row], registry)

    def test_duplicate_observation_key_is_rejected(self) -> None:
        paper, observations, _, _, _ = self.make_inputs()
        registry = validate_paper_registry(read_jsonl(paper, "paper"))
        row = read_jsonl(observations, "observations")[0]
        duplicate = dict(row)
        duplicate["observation_id"] = "second_observation"
        with self.assertRaisesRegex(ValueError, "duplicate experiment"):
            validate_observations([row, duplicate], registry)

    def test_committed_summary_matches_default_sources(self) -> None:
        audit_output = DEFAULT_OUTPUT.with_suffix(".audit.json")
        result = build_summary(
            DEFAULT_PAPER_REGISTRY,
            DEFAULT_OBSERVATIONS,
            DEFAULT_OUTPUT,
            audit_output,
        )
        with DEFAULT_OUTPUT.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 77)
        self.assertEqual({row["status"] for row in rows}, {"blocked", "close"})
        blocked = [row for row in rows if row["status"] == "blocked"]
        close = [row for row in rows if row["status"] == "close"]
        self.assertEqual(len(blocked), 23)
        self.assertEqual(len(close), 54)
        self.assertTrue(all(row["reproduced_value"] == "" for row in blocked))
        self.assertTrue(all(row["reproduced_value"] != "" for row in close))
        self.assertEqual(result["paper_metric_count"], 910)
        self.assertEqual(result["unobserved_paper_metric_count"], 863)
        self.assertEqual(result["status_counts"], {"blocked": 23, "close": 54})


if __name__ == "__main__":
    unittest.main()

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/select_asrur_bge_query_template.py"
SPEC = importlib.util.spec_from_file_location("select_bge_template", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class SelectASRURBGEQueryTemplateTest(unittest.TestCase):
    def test_dev_only_selection_and_deterministic_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qrels = root / "dev.qrels.jsonl"
            none = root / "none.jsonl"
            instructed = root / "instructed.jsonl"
            output = root / "selection"
            write_jsonl(
                qrels,
                [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
            )
            ranking = [
                {
                    "query_id": "q1",
                    "candidate_ids": ["d1", "d2"],
                    "scores": [1.0, 0.0],
                }
            ]
            write_jsonl(none, ranking)
            write_jsonl(instructed, ranking)
            selected, evaluations = MODULE.select_query_template(
                {"none": none, "bge_retrieval": instructed},
                qrels,
            )
            self.assertEqual(selected, "bge_retrieval")
            self.assertEqual(
                evaluations["none"]["mean"]["nDCG@10"],
                evaluations["bge_retrieval"]["mean"]["nDCG@10"],
            )

    def test_formal_script_requires_clean_git_and_dev_only_metadata(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("BGE dev selection requires a clean Git worktree", source)
        self.assertIn('"selected_on_split": "fiqa_dev"', source)
        self.assertIn('"test_qrels_used": False', source)
        self.assertIn('"tie_break": "prefer_bge_retrieval_instruction"', source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPOSITORY_ROOT / "docs/paper_experiment_inventory.md"
STATUS_PATH = REPOSITORY_ROOT / "docs/reproduction_status.md"
REPORT_PATH = REPOSITORY_ROOT / "docs/reproduction_report.md"
VALID_STATUSES = {"COMPLETED", "IN_PROGRESS", "TODO", "BLOCKED"}


class ReproductionProgressReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory_text = INVENTORY_PATH.read_text(encoding="utf-8")
        cls.status_text = STATUS_PATH.read_text(encoding="utf-8")
        cls.report_text = REPORT_PATH.read_text(encoding="utf-8")
        table = cls.inventory_text.split("## 全量清单", 1)[1].split("\n## ", 1)[0]
        statuses = []
        for line in table.splitlines():
            if not re.match(r"^\| (?:FIG|EXP|DER)-\d+ \|", line):
                continue
            status = line.rsplit("|", 2)[1].strip()
            if status not in VALID_STATUSES:
                raise AssertionError(f"unexpected inventory status: {status}")
            statuses.append(status)
        cls.counts = Counter(statuses)

    def test_inventory_progress_is_exact(self) -> None:
        self.assertEqual(sum(self.counts.values()), 31)
        self.assertEqual(
            self.counts,
            {
                "COMPLETED": 4,
                "IN_PROGRESS": 10,
                "TODO": 5,
                "BLOCKED": 12,
            },
        )
        self.assertEqual(f"{self.counts['COMPLETED'] / 31 * 100:.1f}%", "12.9%")

    def test_required_user_report_includes_progress_block(self) -> None:
        expected_sections = (
            "【需要你执行的操作】",
            "【执行后请告诉我】",
            "【实验进度】",
            "【我目前正在做什么】",
        )
        positions = [self.status_text.index(section) for section in expected_sections]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("`4/31 = 12.9%`", self.status_text)
        self.assertIn("`是否使用 GPU：是/否`", self.status_text)
        self.assertIn("`是否使用 OEA 论文的源代码文件：是/否`", self.status_text)
        self.assertIn("`预计执行时间：<可审计的时间范围>`", self.status_text)
        self.assertIn("必须在命令前以独立字段显式列出", self.status_text)
        self.assertIn(
            "不得只把 GPU、OEA 官方源码使用情况或预计时间埋在说明段落中",
            self.status_text,
        )

    def test_inventory_and_report_publish_the_same_percentage(self) -> None:
        self.assertIn("`4/31 = 12.9%`", self.inventory_text)
        self.assertIn("`4/31 = 12.9%`", self.report_text)


if __name__ == "__main__":
    unittest.main()

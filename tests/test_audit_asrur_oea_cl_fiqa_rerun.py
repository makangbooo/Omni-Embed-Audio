import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_asrur_oea_cl_fiqa_rerun import (
    classify,
    embedding_comparison,
)


class AuditASRUROEACLFiQARerunTest(unittest.TestCase):
    def test_embedding_comparison_reports_outlier_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = root / "ids.jsonl"
            ids.write_text(
                "".join(
                    json.dumps({"id": value}) + "\n"
                    for value in ("a", "b")
                ),
                encoding="utf-8",
            )
            fresh = root / "fresh.npy"
            reference = root / "reference.npy"
            np.save(
                fresh,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            np.save(
                reference,
                np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            )
            report = embedding_comparison(
                fresh,
                reference,
                fresh_ids_path=ids,
                reference_ids_path=ids,
            )
        self.assertEqual(report["row_count"], 2)
        self.assertFalse(report["byte_identical"])
        self.assertEqual(report["row_cosine"]["minimum"], 0.0)
        self.assertEqual(report["lowest_cosine_rows"][0]["id"], "b")

    def test_classify_reproduced_domain_failure(self) -> None:
        metrics = {
            condition: {
                "fresh": {"metrics": {"Recall@100": 0.002}},
                "reference": {"metrics": {"Recall@100": 0.0025}},
                "delta_fresh_minus_reference": {"Recall@100": -0.0005},
            }
            for condition in ("clean", "snr_20", "snr_10", "snr_0")
        }
        result = classify(metrics)
        self.assertEqual(
            result["decision"],
            "REPRODUCED_OEA_CL_FIQA_TRANSFER_FAILURE",
        )
        self.assertFalse(result["checkpoint_selection_performed"])


class OEACLFiQARerunWrapperTest(unittest.TestCase):
    def test_wrapper_is_fresh_tmux_only_and_offline(self) -> None:
        source = Path(
            "scripts/run_asrur_oea_cl_fiqa_rerun.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ "${MODE}" == "--execute" && -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_oea_cl_fiqa_rerun", source)
        self.assertIn("old_oea_embedding_reuse=0", source)
        self.assertIn("checkpoint_selection=0", source)
        self.assertIn("training=0", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("audio_only_no_text_prefix", source)
        self.assertIn("audit_asrur_oea_cl_fiqa_rerun.py", source)
        self.assertIn("[TIMING]", source)
        self.assertNotIn("nohup", source)
        self.assertNotIn("PHASE2_REUSE_CACHE_ROOT", source)


if __name__ == "__main__":
    unittest.main()

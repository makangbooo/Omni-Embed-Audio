from __future__ import annotations

from pathlib import Path
import unittest

from scripts.audit_official_oea_variant_resources import (
    REPOSITORY_ROOT,
    resolve_variant_audit_plan,
)
from scripts.prepare_official_oea_checkpoint import EXPECTED_VARIANT_IDS


class AuditOfficialOEAVariantResourcesTest(unittest.TestCase):
    def test_all_six_variants_resolve_exact_base_and_checkpoint_assets(self) -> None:
        expected = {
            "oea_nemo3b": (
                ["configs/resources/model03_nemo3b.json"],
                ["omni_embed_nemotron_3b", "oea_nemo3b_ac"],
            ),
            "oea_nemo3b_cl": (
                ["configs/resources/model03_nemo3b.json"],
                ["omni_embed_nemotron_3b", "oea_nemo3b_cl"],
            ),
            "oea_qwen3b": (
                [
                    "configs/resources/model01_qwen3b_cl.json",
                    "configs/resources/model02_qwen3b_ac.json",
                ],
                ["qwen2_5_omni_3b", "oea_qwen3b_ac"],
            ),
            "oea_qwen3b_cl": (
                ["configs/resources/model01_qwen3b_cl.json"],
                ["qwen2_5_omni_3b", "oea_qwen3b_cl"],
            ),
            "oea_qwen7b": (
                ["configs/resources/model04_qwen7b.json"],
                ["qwen2_5_omni_7b", "oea_qwen7b_ac"],
            ),
            "oea_qwen7b_cl": (
                ["configs/resources/model04_qwen7b.json"],
                ["qwen2_5_omni_7b", "oea_qwen7b_cl"],
            ),
        }
        observed = {}
        for variant_id in EXPECTED_VARIANT_IDS:
            plan = resolve_variant_audit_plan(variant_id)
            observed[variant_id] = (plan["manifests"], plan["asset_names"])
            self.assertEqual(plan["schema_version"], 1)
            self.assertEqual(plan["type"], "official_oea_variant")
            self.assertEqual(plan["variant_id"], variant_id)
            self.assertEqual(plan["variant"]["variant_id"], variant_id)
            self.assertEqual(len(plan["checkpoint_registry"]["sha256"]), 64)
        self.assertEqual(observed, expected)

    def test_unknown_variant_is_rejected_before_any_resource_audit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            resolve_variant_audit_plan("oea_unknown")

    def test_cpu_wrapper_is_read_only_clean_and_registry_driven(self) -> None:
        wrapper = (
            REPOSITORY_ROOT
            / "scripts/run_official_oea_model_resource_audit.sh"
        ).read_text(encoding="utf-8")
        implementation = (
            REPOSITORY_ROOT
            / "scripts/audit_official_oea_variant_resources.py"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("requires a clean Git worktree", wrapper)
        self.assertIn("audit_official_oea_variant_resources.py", wrapper)
        self.assertIn("load_checkpoint_registry", implementation)
        self.assertIn("audit_resources", implementation)
        for forbidden in ("snapshot_download", "rm -rf", "Remove-Item"):
            self.assertNotIn(forbidden, wrapper)
            self.assertNotIn(forbidden, implementation)


if __name__ == "__main__":
    unittest.main()

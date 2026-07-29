import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class OfficialOeaPrecomputeLocalResourcesTests(unittest.TestCase):
    def test_cli_exposes_existing_local_model_directory(self):
        source = (REPO_ROOT / "AudioRetrieval" / "cli" / "main.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('embeddings.add_argument("--local-path", type=Path,', source)
        self.assertIn("local_path=str(args.local_path) if args.local_path else None", source)

    def test_precomputer_loads_training_checkpoint_on_cpu(self):
        source = (
            REPO_ROOT
            / "AudioRetrieval"
            / "preprocessing"
            / "embeddings"
            / "oea.py"
        ).read_text(encoding="utf-8")
        self.assertIn("training_module.safe_torch_load(", source)
        self.assertIn('map_location=torch.device("cpu")', source)


if __name__ == "__main__":
    unittest.main()

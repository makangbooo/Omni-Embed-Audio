import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "encode_example.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("oea_encode_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OfficialEncodeExampleLocalResourcesTests(unittest.TestCase):
    def test_local_resource_arguments_are_optional_paths(self):
        module = load_example_module()
        with mock.patch.object(
            sys,
            "argv",
            [
                "encode_example.py",
                "--checkpoint-path",
                "checkpoint.pt",
                "--base-model-path",
                "base-model",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.checkpoint_path, Path("checkpoint.pt"))
        self.assertEqual(args.base_model_path, Path("base-model"))

    def test_help_exposes_local_offline_inputs(self):
        module = load_example_module()
        with mock.patch.object(sys, "argv", ["encode_example.py", "--help"]):
            with self.assertRaises(SystemExit) as raised:
                module.parse_args()
        self.assertEqual(raised.exception.code, 0)

    def test_checkpoint_is_loaded_on_cpu(self):
        source = EXAMPLE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'safe_torch_load(ckpt_path, map_location=torch.device("cpu"))',
            source,
        )


if __name__ == "__main__":
    unittest.main()

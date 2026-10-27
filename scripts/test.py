"""Check installed package/CLI compatibility; no weights, GPU or hardware required.

Run after installing the model package: python scripts/test.py [-v].
These are packaging smoke tests, not model evaluation or numerical validation.
"""
import importlib.util
import json
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallationTests(unittest.TestCase):
    def test_legacy_module_paths(self):
        # Old metadata and commands must work without rewriting saved bundles.
        for name in ("lambda0.models.configuration_lambda0",
                     "lambda0.models.modeling_lambda0", "lambda0.training.engine",
                     "lambda0.inference.server", "lambda0.evaluation.metrics"):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.util.find_spec(name))
        for config in (ROOT / "configs/lambda0").glob("*/metadata.example.json"):
            with self.subTest(config=config):
                from lambda0.data.domain_handler.registry import get_handler_cls
                handler = json.loads(config.read_text())["handler"]
                self.assertEqual(get_handler_cls(handler).__name__, handler.split(":")[1])

    def test_packaged_recipes(self):
        for stage in ("pretrain", "midtrain", "posttrain"):
            with self.subTest(stage=stage):
                recipe = files("lambda0").joinpath(f"recipes/{stage}.json")
                self.assertIsInstance(json.loads(recipe.read_text())["training"], dict)

    def test_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            old = subprocess.check_output(
                [sys.executable, "-m", "lambda0", "train", "--help"],
                cwd=directory, text=True)
            new = subprocess.check_output(
                [sys.executable, str(ROOT / "scripts/train.py"), "--help"],
                cwd=directory, text=True)
            self.assertEqual(old, new)
            info = subprocess.check_output(
                [sys.executable, "-m", "lambda0", "doctor"],
                cwd=directory, text=True)
            self.assertIn("python", json.loads(info))

    def test_config_path_semantics(self):
        from lambda0.runtime.paths import metadata_paths, resolve_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            metadata = root / "bundle/train/episode.json"
            resolved = metadata_paths(
                {"root_path": "dataset", "normalization_path": "../stats/episode.json"},
                metadata, root / "data")
            self.assertEqual(resolved["root_path"], str(root / "data/dataset"))
            self.assertEqual(resolved["normalization_path"], str(root / "bundle/stats/episode.json"))
            self.assertEqual(resolve_path("../weights", root / "configs"), root / "weights")


if __name__ == "__main__":
    unittest.main()

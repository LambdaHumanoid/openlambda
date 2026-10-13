"""Keep generic processing helpers separate from external runtime interfaces."""
import ast
from pathlib import Path
import subprocess
import sys

from setuptools import find_namespace_packages

ROOT = Path(__file__).resolve().parents[1]


def test_no_local_runtime_namespace_or_imports():
    assert not (ROOT / "gear_sonic").exists()
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name == "gear_sonic" or name.startswith("gear_sonic.")
                           for name in names), path


def test_distribution_includes_responsibility_packages():
    packages = set(find_namespace_packages(str(ROOT), include=["egoalign*", "scripts*"]))
    assert {"egoalign.geometry", "egoalign.datasets", "egoalign.robots.g1",
            "egoalign.adapters.sonic"} <= packages
    assert not any(name.startswith("gear_sonic") for name in packages)


def test_pure_hand_mapping_does_not_load_robot_model_dependencies():
    # Hand conversion requires NumPy, not a URDF or Pinocchio installation.
    code = '''
import importlib.abc
import sys
class BlockRobotModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pinocchio" or fullname.startswith("pinocchio."):
            raise AssertionError("pure hand mapping loaded Pinocchio")
sys.meta_path.insert(0, BlockRobotModel())
from egoalign.robots.g1.hand_mapping import retarget_openxr_hand_to_brainco
assert callable(retarget_openxr_hand_to_brainco)
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                   capture_output=True, text=True)

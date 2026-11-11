"""Subprocess bootstrap for eval_simple.py; delegates all episodes to SIMPLE."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys


def native_provenance(module_name: str) -> dict:
    import simple

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Installed SIMPLE has no evaluator {module_name}")
    source = Path(spec.origin).resolve()
    provenance = {
        "version": getattr(simple, "__version__", None),
        "evaluator_source": str(source),
        "evaluator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_revision": None,
        "git_dirty": None,
        "git_status": None,
        "note": "Revision alone does not identify local modifications; source hash and working-tree status are recorded when available.",
    }
    root = next((parent for parent in source.parents if (parent / ".git").exists()), None)
    if root is not None:
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout
            provenance.update(git_revision=revision, git_dirty=bool(status), git_status=status)
        except (OSError, subprocess.SubprocessError) as exc:
            provenance["git_metadata_error"] = str(exc)
    return provenance


def environment_spec(env_id: str):
    import gymnasium as gym
    import simple.envs  # noqa: F401 — register native environments

    return gym.spec(env_id)


def main() -> None:
    protocol_path = Path(sys.argv[1])
    protocol = json.loads(protocol_path.read_text())
    module_name = protocol["eval_module"]
    protocol["native"] = native_provenance(module_name)
    spec = environment_spec(protocol["config"]["env_id"])
    protocol["native"]["environment_entry_point"] = spec.entry_point
    protocol["native"]["environment_kwargs"] = spec.kwargs
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    if spec.entry_point != "simple.envs.loco_manipulation:LocoManipulationEnv":
        raise RuntimeError(
            f"{spec.id} uses {spec.entry_point}, not the native AMO environment. "
            "Use the matching AMO SIMPLE revision; changing the robot or evaluator "
            "does not convert a different task/controller protocol to AMO."
        )

    # Use the public AMO evaluator; policies are discovered by module name.
    evaluator = importlib.import_module(module_name)
    from simple.evals.api import EvalConfig

    protocol["native"]["cli_defaults"] = {
        name: parameter.default
        for name, parameter in inspect.signature(evaluator.main).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    evaluator.run_eval(EvalConfig(**protocol["config"]))


if __name__ == "__main__":
    main()

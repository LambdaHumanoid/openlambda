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


BACKEND_ENTRY_POINTS = {
    "amo": "simple.envs.loco_manipulation:LocoManipulationEnv",
    "decoupled_wbc": "simple.envs.sonic_loco_manip:SonicLocoManipEnv",
}


def validate_environment_backend(env_id: str, entry_point: str, backend: str) -> None:
    if backend not in BACKEND_ENTRY_POINTS:
        raise ValueError(
            f"Unsupported SIMPLE backend {backend!r}; expected one of {sorted(BACKEND_ENTRY_POINTS)}."
        )
    expected = BACKEND_ENTRY_POINTS[backend]
    if entry_point != expected:
        raise RuntimeError(
            f"{env_id} uses {entry_point}, expected {expected} for backend {backend!r}. "
            "Use the matching SIMPLE revision and task backend."
        )


def main() -> None:
    protocol_path = Path(sys.argv[1])
    protocol = json.loads(protocol_path.read_text())
    module_name = protocol["eval_module"]
    backend = protocol.get("backend", "amo")
    protocol["native"] = native_provenance(module_name)
    spec = environment_spec(protocol["config"]["env_id"])
    protocol["native"]["environment_entry_point"] = spec.entry_point
    protocol["native"]["environment_kwargs"] = spec.kwargs
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    validate_environment_backend(spec.id, spec.entry_point, backend)

    # Use the public SIMPLE evaluator; policies are discovered by module name.
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

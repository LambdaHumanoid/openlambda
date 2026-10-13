"""Shared validation and isolation helpers for the simulation runtime."""
import hashlib
from pathlib import Path


def port_set(base, workers):
    if not 1 <= workers <= 16 or base < 20000 or base + (workers-1)*100 + 4 > 65535:
        raise ValueError("Use 1–16 workers and private ports in 20000–65535")
    return [base + worker*100 + offset for worker in range(workers) for offset in range(5)]


def verify_runtime(root, manifest):
    errors = []
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid runtime manifest path")
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {name}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"incompatible hash: {name}")
    if errors:
        raise ValueError("Runtime is not the qualified source snapshot:\n" + "\n".join(errors))


def clean_environment(root, inherited):
    # Do not inherit interactive replay/controller overrides.
    env = {key: value for key, value in inherited.items()
           if not key.startswith(("SONIC_", "SIM_", "PICO_", "ZMQ_", "ROBOT_", "HAND_", "DISABLE_HAND"))}
    env.pop("TMUX", None)
    env.update(PYTHONPATH=str(root), SIM_BRIDGE_MODE="mock",
               SONIC_REPLAY_WRITE_VIDEO="false", HAND_TYPE="brainco")
    if inherited.get("EGOALIGN_BRAINCO_MODEL_DIR"):
        env["SONIC_BRAINCO_MODEL_DIR"] = inherited["EGOALIGN_BRAINCO_MODEL_DIR"]
    return env

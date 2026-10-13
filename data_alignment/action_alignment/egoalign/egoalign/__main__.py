"""Stable, EgoAlign-only command dispatcher."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "endpoint-align": "run_endpoint_retarget_dataset.py",
    "replay-only": None,
    "render-replay": None,
    "candidate": "build_smpl_endpoint_retarget_candidate.py",
    "evaluate": "evaluate_endpoint_retarget_replay.py",
    "audit": "audit_endpoint_retarget_production.py",
    "calibrate": "calibrate_g1_endpoint_morphology.py",
    "promote": "promote_endpoint_retarget_dataset.py",
    "stage-a": "processing/convert_raw_pico_lerobot_to_sonic_gr00t.py",
    "stage-c": "processing/apply_sonic_mujoco_feedback.py",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    # Parse only the subcommand here, so `candidate --help` reaches its parser.
    args = parser.parse_args(sys.argv[1:2])
    extra = sys.argv[2:]
    if args.command == "replay-only":
        from egoalign.replay_only import main as replay_main
        return replay_main(extra)
    # Reject legacy experimental switches before dispatching the bilateral API.
    for flag, required in (("--endpoint-parts", "hands"), ("--hand-endpoint-side", "both")):
        for index, value in enumerate(extra):
            if value == flag and (index + 1 == len(extra) or extra[index + 1] != required):
                parser.error(f"EgoAlign requires {flag} {required}")
            if value.startswith(flag + "=") and value != flag + "=" + required:
                parser.error(f"EgoAlign requires {flag} {required}")
    if any(value.startswith("--experimental-") for value in extra):
        parser.error("Legacy experimental switches are not EgoAlign options")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    environment_key = "EGOALIGN_SIM_PYTHON" if args.command in {"evaluate", "audit", "calibrate", "render-replay"} else "EGOALIGN_DATA_PYTHON"
    executable = env.get(environment_key, sys.executable)
    if args.command == "render-replay":
        return subprocess.call([executable, "-m", "egoalign.render_replay", *extra], cwd=Path.cwd(), env=env)
    return subprocess.call([executable, str(ROOT / "scripts/human_data" / COMMANDS[args.command]), *extra], cwd=Path.cwd(), env=env)

if __name__ == "__main__":
    raise SystemExit(main())

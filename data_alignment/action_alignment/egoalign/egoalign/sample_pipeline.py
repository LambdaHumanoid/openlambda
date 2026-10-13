"""Reproduce the Object Relocation sample using public processing commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from egoalign.io_safety import validate_output_root

ROOT = Path(__file__).resolve().parents[1]


def plan(sample, raw, output, robot_xml, python):
    if sample != "basket":
        raise ValueError("The public sample is Object Relocation (basket)")
    cli = [str(python), "-m", "egoalign"]
    stage = output / "stage_a"
    replay = output / "baseline_replay"
    task = "Move the basket to the table"
    result = [cli + ["stage-a", "--source-root", str(raw), "--output-root", str(stage),
                    "--episodes", "0", "--require-alignment", "--decoder-history-mode", "zero",
                    "--workers", "1", "--task", task]]
    morphology = output / "morphology.json"
    result += [cli + ["calibrate", "--robot-xml", str(robot_xml), "--output", str(morphology)],
               [str(ROOT / "tools/run_sim_replay.sh"), "--preset", "sonic_sim", "--mode", "replay",
                "--hand-type", "brainco", "--replay-dataset", str(stage),
                "--replay-video-dir", str(replay), "--replay-episodes", "all", "--replay-workers", "1"],
               cli + ["endpoint-align", "--raw-dataset", str(raw), "--baseline-dataset", str(stage),
                      "--baseline-contract", "minimal_ab", "--baseline-replay-root", str(replay),
                      "--output-root", str(output / "alignment"), "--mode", "smoke", "--episodes", "0",
                      "--workers", "1", "--state-json", str(output / "state.json"),
                      "--morphology", str(morphology)],
               cli + ["promote", "--baseline", str(stage), "--batch-root", str(output / "alignment"),
                      "--output", str(output / "final")]]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", choices=["basket"])
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--robot-xml", type=Path, help="Exact simulation G1 model XML for FK calibration")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    raw, output = a.raw_root.expanduser().resolve(), a.output_root.expanduser().resolve()
    try:
        validate_output_root(output, [raw])
    except ValueError as exc:
        p.error(str(exc))
    if a.robot_xml is None:
        p.error("basket requires --robot-xml matching the replay scene")
    commands = plan(a.sample, raw, output, a.robot_xml.expanduser().resolve(),
                    os.environ.get("EGOALIGN_DATA_PYTHON", sys.executable))
    if a.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    if not raw.is_dir() or output.exists():
        p.error("raw dataset must exist and output directory must be fresh")
    if os.environ.get("EGOALIGN_ALLOW_SIM_REPLAY") != "1":
        p.error("review runtime/ports and set EGOALIGN_ALLOW_SIM_REPLAY=1")
    output.mkdir(parents=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    def write_state(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = output / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "state.json")
    write_state()
    records = []
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for i, command in enumerate(commands):
        write_state(step=i + 1, total_steps=len(commands), command=command)
        print(f"[{i+1}/{len(commands)}] {command}", flush=True)
        try:
            with (output / f"step_{i:02d}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        except BaseException as exc:
            write_state(status="failed", error=f"{type(exc).__name__}: {exc}",
                        finished_at=datetime.now(timezone.utc).isoformat())
            raise
        records.append({"command": command, "returncode": result.returncode})
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            write_state(status="failed", returncode=result.returncode,
                        finished_at=datetime.now(timezone.utc).isoformat())
            return result.returncode
    write_state(status="complete", returncode=0, finished_at=datetime.now(timezone.utc).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

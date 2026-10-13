"""Real replay and causal export without endpoint optimization."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from egoalign.runtime import validate_replay_workers

ROOT = Path(__file__).resolve().parents[1]

def commands(args):
    replay = [str(ROOT / "tools/run_sim_replay.sh"), "--preset", "sonic_sim", "--mode", "replay",
              "--hand-type", "brainco", "--replay-dataset", str(args.source_root),
              "--replay-video-dir", str(args.replay_root), "--replay-episodes", "all",
              "--replay-workers", str(args.workers)]
    export = [os.environ.get("EGOALIGN_DATA_PYTHON", sys.executable),
              str(ROOT / "scripts/human_data/processing/apply_sonic_mujoco_feedback.py"),
              "--source-root", str(args.source_root), "--output-root", str(args.output_root),
              "--telemetry-root", str(args.replay_root / "telemetry"),
              "--episodes", "all", "--feedback-mode", "captured_cpp",
              "--state-schema", "isaaclab_offsets"]
    return ([replay] if not args.reuse_replay else []) + [export]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Complete Stage-A dataset; never a final training dataset")
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-replay", action="store_true",
                        help="Reuse captured telemetry; exporter still validates source and timing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("source_root", "replay_root", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.reuse_replay:
        try:
            validate_replay_workers(args.workers)
        except ValueError as exc:
            parser.error(str(exc))
    paths = [args.source_root, args.replay_root, args.output_root]
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(paths) for b in paths[i+1:]):
        parser.error("source, replay and output must be separate non-nested directories")
    plan = commands(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.source_root.is_dir():
        parser.error("source dataset does not exist")
    if args.output_root.exists():
        parser.error("output already exists; choose a fresh output directory")
    if args.reuse_replay:
        if not (args.replay_root / "telemetry").is_dir():
            parser.error("reused replay has no telemetry directory")
    elif args.replay_root.exists():
        parser.error("replay directory already exists; choose a fresh one or explicit --reuse-replay")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for command in plan:
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    manifest = {"processing_mode": "replay-only", "endpoint_optimization": False,
                "history": "causal_measured", "motion_token": "same_tick_cpp_capture",
                "source_root": str(args.source_root), "replay_root": str(args.replay_root),
                "commands": plan}
    (args.output_root / "meta" / "processing_mode.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return 0

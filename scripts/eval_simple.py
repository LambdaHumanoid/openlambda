#!/usr/bin/env python3
"""Run one SIMPLE task/level through its installed public evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "configs" / "simple" / "tasks.json"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_args(tasks: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(tasks))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--level", required=True, choices=("level-0", "level-1", "level-2"))
    parser.add_argument("--num-episodes", type=positive_int, default=10)
    parser.add_argument("--episode-start", type=nonnegative_int, default=0)
    parser.add_argument(
        "--data-root", required=True, type=Path,
        help="Actual LeRobot task/level dataset directory containing meta/info.json",
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory; must not exist")
    parser.add_argument(
        "--success-criteria", type=float, default=0.9,
        help="AMO evaluation success threshold (default: 0.9)",
    )
    parser.add_argument(
        "--max-episode-steps", type=positive_int,
        help="Override native task metadata limit; omitted uses task metadata",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--gui", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--num-workers", type=positive_int, default=1, help="Native spawn worker count")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--save-video", dest="save_video", action="store_true")
    video.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=True)
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("--port must be at most 65535")
    if not 0 <= args.success_criteria <= 1:
        parser.error("--success-criteria must be between 0 and 1")
    return args


def check_health(host: str, port: int, dataset: str) -> dict:
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{authority}:{port}/health"
    with urlopen(url, timeout=10) as response:
        health = json.load(response)
    if not isinstance(health, dict):
        raise ValueError(f"{url} did not return a JSON object")
    expected = {
        "status": "ok",
        "normalization_dataset": dataset,
        "normalization_action_space_id": 3,
        "default_action_space_id": 3,
        "default_embodiment_id": 7,
        "inference_steps": 10,
        "rtc_enabled": False,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {health.get(key)!r}"
        for key, value in expected.items()
        if type(health.get(key)) is not type(value) or health.get(key) != value
    ]
    if mismatches:
        raise ValueError("Policy health mismatch: " + "; ".join(mismatches))
    return health


def summarize(output: Path, selection: dict, returncode: int) -> dict:
    expected = [
        f"episode_{index}"
        for index in range(selection["episode_start"], selection["episode_start"] + selection["num_episodes"])
    ]
    stats_path = output / "eval_stats.txt"
    stats: dict[str, bool] = {}
    duplicates = []
    malformed = []
    if stats_path.is_file():
        for line in stats_path.read_text().splitlines():
            line = line.strip()
            match = re.fullmatch(r"(episode_(?:0|[1-9]\d*)):\s*(True|False)", line)
            if match:
                episode, success = match.groups()
                if episode in stats:
                    duplicates.append(episode)
                stats[episode] = success == "True"
            elif line and not (set(line) == {"="} or line.startswith(("run: ", "success rate: "))):
                malformed.append(line)
    missing = sorted(set(expected) - stats.keys())
    unexpected = sorted(stats.keys() - set(expected))
    complete = not (missing or unexpected or duplicates or malformed)
    missing_videos = []
    empty_videos = []
    if selection["save_video"]:
        video_root = output / selection["policy"] / selection["env"].split("/")[1] / selection["level"]
        for episode in expected:
            videos = [path for path in (video_root / episode).glob("*.mp4") if path.is_file()]
            if not videos:
                missing_videos.append(episode)
            empty_videos.extend(str(path) for path in videos if path.stat().st_size == 0)
    videos_complete = not (missing_videos or empty_videos)
    success = complete and videos_complete and returncode == 0
    return {
        "task": selection["task"],
        "level": selection["level"],
        "env": selection["env"],
        "backend": selection["backend"],
        "policy": selection["policy"],
        "status": "complete" if success else "failed",
        "native_returncode": returncode,
        "exit_code": (returncode if returncode > 0 else 128 - returncode) if returncode else (0 if success else 1),
        "expected_episode_ids": expected,
        "completed_episode_ids": sorted(stats, key=lambda item: int(item.split("_")[1])),
        "missing_episode_ids": missing,
        "unexpected_episode_ids": unexpected,
        "duplicate_episode_ids": duplicates,
        "malformed_stats_lines": malformed,
        "episodes_complete": complete,
        "videos_requested": selection["save_video"],
        "videos_complete": videos_complete,
        "missing_video_episode_ids": missing_videos,
        "empty_video_files": empty_videos,
        "num_completed": len(stats),
        "num_successes": sum(stats.values()),
        "success_rate": sum(stats.values()) / len(expected) if success else None,
        "stats": stats,
        "eval_stats": str(stats_path),
        "protocol": str(output / "protocol.json"),
    }


def run(args: argparse.Namespace, tasks: dict) -> int:
    task = tasks[args.task]
    data_root = args.data_root.expanduser().resolve()
    # Check before resolve() so dangling output symlinks also count as collisions.
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}; choose a new directory")
    output = output.resolve()
    if not data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_root}")
    info_path = data_root / "meta" / "info.json"
    with info_path.open() as handle:
        info = json.load(handle)
    total_episodes = info.get("total_episodes")
    if type(total_episodes) is not int or total_episodes < 1:
        raise ValueError(f"{info_path} must declare a positive integer total_episodes")
    if args.episode_start + args.num_episodes > total_episodes:
        raise ValueError(
            f"Requested episodes [{args.episode_start}, {args.episode_start + args.num_episodes}) "
            f"exceed dataset size {total_episodes}"
        )
    health = check_health(args.host, args.port, task["normalization_dataset"])
    selection = {
        **task,
        "task": args.task,
        "level": args.level,
        "data_root": str(data_root),
        "output": str(output),
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "dataset_total_episodes": total_episodes,
        "save_video": args.save_video,
    }
    config = {
        "env_id": task["env"],
        "policy": task["policy"],
        "split": args.level,
        "host": args.host,
        "port": args.port,
        "data_format": "lerobot",
        "sim_mode": "mujoco_isaac",
        "headless": args.headless,
        "eval_dir": str(output),
        "max_episode_steps": args.max_episode_steps,
        "num_episodes": args.num_episodes,
        "episode_start": args.episode_start,
        "data_dir": str(data_root),
        "success_criteria": args.success_criteria,
        "save_video": args.save_video,
        "num_workers": args.num_workers,
    }
    environment = os.environ.copy()
    python_paths = [str(ROOT / "examples" / "simple"), str(ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    protocol = {
        "schema_version": 1,
        "protocol_name": "openlambda_simple_native_evaluator",
        "protocol_note": (
            "Explicit integration settings, not a claim of historical baseline reproduction. "
            "The installed SIMPLE revision and local modifications determine native behavior. "
            "LeRobot input, headless mode, and success threshold 0.9 are "
            "integration defaults for the original AMO tasks. A null step limit "
            "delegates to native task metadata, not EvalConfig's 15000-step default. "
            "The supplied data-root selects the dataset; level labels the native split."
        ),
        "eval_module": task["eval_module"],
        "entrypoint": "run_eval(EvalConfig(**config))",
        "config": config,
        "policy_contract": {"action_space_id": 3, "embodiment_id": 7, "inference_steps": 10, "rtc_enabled": False},
        "health": health,
        "native": None,
        "inherited_runtime_environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID", "VK_ICD_FILENAMES", "SIMPLE_DISABLE_TUI",
            )
            if key in environment
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", selection)
    write_json(output / "protocol.json", protocol)
    command = [sys.executable, str(ROOT / "scripts" / "_simple_eval.py"), str(output / "protocol.json")]
    try:
        returncode = subprocess.run(command, env=environment, check=False).returncode
    except KeyboardInterrupt:
        returncode = 130
    except OSError as exc:
        print(f"Could not launch SIMPLE evaluator: {exc}", file=sys.stderr)
        returncode = 1
    summary = summarize(output, selection, returncode)
    write_json(output / "summary.json", summary)
    print(f"Evaluation {summary['status']}; summary: {output / 'summary.json'}")
    return summary["exit_code"]


def main() -> int:
    with TASKS_PATH.open() as handle:
        tasks = json.load(handle)
    args = parse_args(tasks)
    try:
        return run(args, tasks)
    except (OSError, ValueError, URLError) as exc:
        print(f"Evaluation preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

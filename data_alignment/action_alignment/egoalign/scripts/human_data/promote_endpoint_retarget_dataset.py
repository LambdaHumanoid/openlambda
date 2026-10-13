#!/usr/bin/env python3
"""Promote accepted endpoint-retarget rounds into a trainable LeRobot dataset.

Each accepted round is first passed independently through the canonical
captured-C++ Stage-C writer.  This preserves the exact one-episode staging
parquet hash recorded by the final replay.  The resulting Stage-C episodes are
then compacted into a new immutable dataset while rejected episodes remain
recorded only in the promotion manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.human_data.smpl_endpoint_retargeting import (
    ENDPOINT_DEFINITION,
    ENDPOINT_PARTS_HANDS,
    HAND_ENDPOINT_SIDE_BOTH,
    PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S,
    PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG,
    PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG,
    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
    PICO_WRIST_FILTER_CUTOFF_HZ,
    PICO_WRIST_FILTER_ORDER,
    PICO_WRIST_GRASP_ENTER_THRESHOLD,
    PICO_WRIST_GRASP_EXIT_THRESHOLD,
    PICO_WRIST_GRASP_MAX_DURATION_S,
    PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG,
    PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
    PICO_WRIST_GRASP_PADDING_S,
    PICO_WRIST_MOTION_FILTER_POLICY,
    VR3PT_ORIENTATION_POLICY,
    WRIST_LOCAL_ROTATION_POLICY,
    endpoint_retarget_contract,
    validate_endpoint_candidate_archive,
)

APPLY_FEEDBACK = (
    PROJECT_ROOT / "scripts/human_data/processing/apply_sonic_mujoco_feedback.py"
)
PYTHON = Path(os.environ.get("EGOALIGN_DATA_PYTHON", sys.executable))
CHANGED_RETARGET_COLUMNS = (
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.vr_3pt_position",
    "teleop.vr_3pt_orientation",
    "teleop.left_wrist_joints",
    "teleop.right_wrist_joints",
)


class PromotionError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionError(f"JSON object required: {path}")
    return value


def recompute_global_stats(root: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Recompute LeRobot global numeric statistics from promoted parquet rows."""

    parquet_files = sorted((root / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        raise PromotionError(f"promoted dataset has no parquet files: {root}")
    stats: dict[str, Any] = {}
    for name, feature in info.get("features", {}).items():
        if not isinstance(feature, dict) or feature.get("dtype") == "video":
            continue
        arrays: list[np.ndarray] = []
        for path in parquet_files:
            table = pq.read_table(path, columns=[name])
            column = table[name].combine_chunks()
            if column.null_count:
                raise PromotionError(f"null values in {name}: {path}")
            if pa.types.is_fixed_size_list(column.type):
                values = column.values.to_numpy(zero_copy_only=False).reshape(
                    len(column), -1
                )
            elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                values = np.asarray(column.to_pylist())
            else:
                values = column.to_numpy(zero_copy_only=False).reshape(len(column), 1)
            if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
                continue
            arrays.append(values.astype(np.float64, copy=False))
        if not arrays:
            continue
        values = np.concatenate(arrays, axis=0)
        if not np.isfinite(values).all():
            raise PromotionError(f"non-finite promoted feature: {name}")
        stats[name] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_or_link(source: Path | str, destination: Path | str) -> str:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(len(values))],
    }


def numeric_column(table: pa.Table, name: str) -> np.ndarray:
    column = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
        values = column
        while pa.types.is_fixed_size_list(values.type) or pa.types.is_list(values.type):
            values = values.values
        return np.asarray(values).reshape(len(table), -1)
    return np.asarray(column)


def replace_indices(
    table: pa.Table, episode_index: int, global_start: int
) -> pa.Table:
    replacements = {
        "episode_index": np.full(len(table), episode_index, dtype=np.int64),
        "frame_index": np.arange(len(table), dtype=np.int64),
        "index": np.arange(global_start, global_start + len(table), dtype=np.int64),
    }
    for name, values in replacements.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            raise PromotionError(f"missing required index column: {name}")
        field = table.schema.field(index)
        table = table.set_column(index, field, pa.array(values, type=field.type))
    return table


def rewrite_value(value: Any, old_episode: int, new_episode: int) -> Any:
    if isinstance(value, str):
        return value.replace(
            f"episode_{old_episode:06d}", f"episode_{new_episode:06d}"
        )
    if isinstance(value, list):
        return [rewrite_value(item, old_episode, new_episode) for item in value]
    if isinstance(value, dict):
        result = {
            key: rewrite_value(item, old_episode, new_episode)
            for key, item in value.items()
        }
        if isinstance(result.get("episode_index"), int):
            result["episode_index"] = new_episode
        return result
    return value


def video_templates(info: dict[str, Any]) -> list[str]:
    value = info.get("video_path")
    if isinstance(value, str):
        if "{video_key}" in value:
            features = info.get("features", {})
            keys = [
                key
                for key, feature in features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            ]
            if not keys:
                raise PromotionError("video_path uses {video_key} but no video features exist")
            return [value.replace("{video_key}", key) for key in keys]
        return [value]
    if isinstance(value, dict) and all(isinstance(item, str) for item in value.values()):
        return list(value.values())
    raise PromotionError("baseline video_path must be a string or string mapping")


def format_episode_path(root: Path, template: str, episode: int, chunks: int) -> Path:
    return root / template.format(
        episode_chunk=episode // chunks, episode_index=episode
    )


def validate_promotion_candidate(
    round_dir: Path,
    baseline_parquet: Path,
    staging_parquet: Path,
) -> dict[str, Any]:
    """Prove the accepted artifact still carries the current palm/wrist data."""

    candidate_path = round_dir / "candidate_smpl.npz"
    try:
        with np.load(candidate_path, allow_pickle=False) as candidate:
            contract = validate_endpoint_candidate_archive(candidate)
            archive_source_pose = np.asarray(
                candidate["source_baseline_smpl_pose"], dtype=np.float32
            )
            archive_baseline_pose = np.asarray(
                candidate["baseline_smpl_pose"], dtype=np.float32
            )
            archive_candidate_pose = np.asarray(
                candidate["candidate_smpl_pose"], dtype=np.float32
            )
            archive_source_orientation = np.asarray(
                candidate["source_baseline_vr3pt_orientation"], dtype=np.float32
            )
            archive_candidate_orientation = np.asarray(
                candidate["candidate_vr3pt_orientation"], dtype=np.float32
            )
            hand_endpoint_side = str(
                contract.get("hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH)
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionError(
            f"accepted candidate does not prove current four-MCP/wrist-lock contract: "
            f"{candidate_path}: {exc}"
        ) from exc

    columns = [
        "teleop.smpl_pose",
        "teleop.vr_3pt_orientation",
    ]
    baseline_table = pq.read_table(baseline_parquet, columns=columns)
    staging_table = pq.read_table(staging_parquet, columns=columns)
    baseline_pose = numeric_column(baseline_table, "teleop.smpl_pose").astype(
        np.float32
    )
    candidate_pose = numeric_column(staging_table, "teleop.smpl_pose").astype(
        np.float32
    )
    baseline_orientation = numeric_column(
        baseline_table, "teleop.vr_3pt_orientation"
    ).astype(np.float32)
    candidate_orientation = numeric_column(
        staging_table, "teleop.vr_3pt_orientation"
    ).astype(np.float32)
    if (
        baseline_pose.shape != candidate_pose.shape
        or baseline_pose.shape[1] != 63
        or not np.array_equal(
            archive_baseline_pose[:, 57:63], candidate_pose[:, 57:63]
        )
    ):
        raise PromotionError(
            "accepted staging does not preserve the wrist-denoised "
            f"Pico wrist-local rotation: {staging_parquet}"
        )
    if (
        baseline_orientation.shape != candidate_orientation.shape
        or baseline_orientation.shape[1] != 18
        or not np.array_equal(
            archive_candidate_orientation, candidate_orientation
        )
    ):
        raise PromotionError(
            "accepted staging does not carry the wrist-denoised optical "
            f"vr_3pt_orientation: {staging_parquet}"
        )
    # Link the input baseline and final candidate to the staged parquet exactly.
    archive_input_pose = archive_source_pose
    if not np.array_equal(archive_input_pose, baseline_pose) or not np.array_equal(
        archive_candidate_pose, candidate_pose
    ):
        raise PromotionError(
            "candidate NPZ input/candidate pose arrays do not match "
            "baseline/staging parquet bytes"
        )
    if not np.array_equal(
        archive_source_orientation, baseline_orientation
    ) or not np.array_equal(archive_candidate_orientation, candidate_orientation):
        raise PromotionError(
            "candidate NPZ source/candidate orientation arrays do not match "
            "baseline/staging parquet bytes"
        )

    ik_metrics = load_json(round_dir / "ik_metrics.json")
    wrist_metrics = ik_metrics.get("optimization", {}).get(
        "pico_wrist_local_rotation", {}
    )
    false_grasp_metrics = ik_metrics.get("pico_false_grasp_wrist_repair", {})
    wrist_filter_metrics = ik_metrics.get("pico_wrist_motion_filter", {})
    replay_metrics = load_json(round_dir / "replay_metrics.json")
    production_audit = load_json(round_dir / "production_audit.json")
    archive_endpoint_contract = endpoint_retarget_contract(
        ENDPOINT_PARTS_HANDS,
        str(contract.get("hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH)),
    )
    expected_algorithm = archive_endpoint_contract.algorithm
    expected_target_contract = archive_endpoint_contract.endpoint_target_contract
    if (
        ik_metrics.get("algorithm") != expected_algorithm
        or ik_metrics.get("target_contract") != expected_target_contract
        or wrist_metrics.get("policy") != WRIST_LOCAL_ROTATION_POLICY
        or float(wrist_metrics.get("max_abs_delta_rad", float("inf"))) > 1e-7
        or false_grasp_metrics.get("policy")
        != PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        or wrist_filter_metrics.get("policy") != PICO_WRIST_MOTION_FILTER_POLICY
        or replay_metrics.get("algorithm") != expected_algorithm
        or replay_metrics.get("candidate_contract", {}).get("endpoint_definition")
        != ENDPOINT_DEFINITION
        or production_audit.get("candidate_contract", {}).get("endpoint_definition")
        != ENDPOINT_DEFINITION
    ):
        raise PromotionError(
            f"accepted round metadata does not prove current endpoint contract: {round_dir}"
        )
    return contract


def accepted_decisions(batch_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(batch_root / "summary.json")
    hand_endpoint_side = str(
        summary.get("hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH)
    )
    contract = endpoint_retarget_contract(ENDPOINT_PARTS_HANDS, hand_endpoint_side)
    expected_identity = {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "endpoint_definition": contract.endpoint_definition,
        "endpoint_target_contract": contract.endpoint_target_contract,
        "wrist_local_rotation_policy": WRIST_LOCAL_ROTATION_POLICY,
        "vr3pt_orientation_policy": VR3PT_ORIENTATION_POLICY,
        "wrist_false_grasp_repair_policy": PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
        "wrist_false_grasp_max_island_duration_s": (
            PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S
        ),
        "wrist_false_grasp_min_endpoint_jump_deg": (
            PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG
        ),
        "wrist_false_grasp_max_upstream_step_deg": (
            PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG
        ),
        "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
        "wrist_filter_cutoff_hz": PICO_WRIST_FILTER_CUTOFF_HZ,
        "wrist_filter_order": PICO_WRIST_FILTER_ORDER,
        "wrist_grasp_enter_threshold": PICO_WRIST_GRASP_ENTER_THRESHOLD,
        "wrist_grasp_exit_threshold": PICO_WRIST_GRASP_EXIT_THRESHOLD,
        "wrist_grasp_padding_s": PICO_WRIST_GRASP_PADDING_S,
        "wrist_grasp_max_duration_s": PICO_WRIST_GRASP_MAX_DURATION_S,
        "wrist_grasp_min_residual_deg": PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
        "wrist_grasp_max_upstream_residual_deg": (
            PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
        ),
    }
    for key, expected in expected_identity.items():
        if summary.get(key) != expected:
            raise PromotionError(
                f"batch summary does not satisfy current endpoint contract {key}: "
                f"observed={summary.get(key)!r}, expected={expected!r}"
            )
    decisions = summary.get("episodes")
    if not isinstance(decisions, list) or not decisions:
        raise PromotionError(f"batch summary has no decisions: {batch_root}")
    accepted = [item for item in decisions if item.get("status") == "optimized_pass"]
    rejected = [item for item in decisions if item.get("status") != "optimized_pass"]
    if len(accepted) + len(rejected) != int(summary.get("completed", -1)):
        raise PromotionError("batch summary completion count is inconsistent")
    for item in decisions:
        for key, expected in expected_identity.items():
            if item.get(key) != expected:
                raise PromotionError(
                    f"episode {item.get('compact_episode')} has incompatible {key}"
                )
    return accepted, rejected


def parse_episode_selection(value: str) -> set[int] | None:
    if value.strip().lower() in {"all", "*"}:
        return None
    selected: set[int] = set()
    for token in value.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start, stop = (int(item) for item in token.split("-", 1))
            if stop < start:
                raise PromotionError(f"descending episode selection: {token}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(token))
    if not selected:
        raise PromotionError("episode selection is empty")
    return selected


def augment_staging_stats(
    staging: Path,
    baseline_stats: dict[int, dict[str, Any]],
    compact_episode: int,
) -> None:
    table = pq.read_table(staging / "data/chunk-000/episode_000000.parquet")
    record = json.loads(json.dumps(baseline_stats[compact_episode]))
    record["episode_index"] = 0
    for name in CHANGED_RETARGET_COLUMNS:
        record["stats"][name] = array_stats(numeric_column(table, name))
    write_jsonl(staging / "meta/episodes_stats.jsonl", [record])


def run_stage_c(staging: Path, replay: Path, output: Path) -> None:
    subprocess.run(
        [
            str(PYTHON),
            str(APPLY_FEEDBACK),
            "--source-root",
            str(staging),
            "--output-root",
            str(output),
            "--telemetry-root",
            str(replay),
            "--feedback-mode",
            "captured_cpp",
            "--state-schema",
            "isaaclab_offsets",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )


def select_promotion_replay(
    round_dir: Path, *, expected_algorithm: str
) -> tuple[Path, dict[str, Any] | None]:
    """Prefer a separately re-audited promotion retry over timing-invalid telemetry."""
    for replay in sorted(
        round_dir.glob("replay_promotion_retry_attempt*"), reverse=True
    ):
        metrics_path = replay / "promotion_replay_metrics.json"
        audit_path = replay / "promotion_production_audit.json"
        if not metrics_path.is_file() or not audit_path.is_file():
            continue
        metrics = load_json(metrics_path)
        audit = load_json(audit_path)
        gate = metrics.get("closed_loop_gate", {})
        if (
            gate.get("accepted") is not True
            or gate.get("status") != "optimized_pass"
            or metrics.get("algorithm") != expected_algorithm
            or metrics.get("candidate_contract", {}).get("endpoint_definition")
            != ENDPOINT_DEFINITION
            or audit.get("candidate_contract", {}).get("endpoint_definition")
            != ENDPOINT_DEFINITION
        ):
            continue
        required = [
            replay / "summary.json",
            replay / "manifest.jsonl",
            replay / "telemetry/episode_000000.control.npz",
            replay / "telemetry/episode_000000.sim.npz",
            replay / "telemetry/episode_000000.reference.npz",
            replay / "telemetry/episode_000000.rollout.json",
        ]
        if not all(path.is_file() for path in required):
            continue
        return replay, {
            "reason": "fresh replay replacing telemetry that failed Stage-C timing gate",
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "production_audit": str(audit_path),
            "production_audit_sha256": sha256_file(audit_path),
        }
    return round_dir / "replay", None


def promotion_round_identity(hand_endpoint_side: str) -> dict[str, Any]:
    """Return the exact round contract accepted by promotion."""

    endpoint_contract = endpoint_retarget_contract(
        ENDPOINT_PARTS_HANDS, hand_endpoint_side
    )
    return {
        "algorithm": endpoint_contract.algorithm,
        "candidate_contract_version": endpoint_contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "endpoint_definition": ENDPOINT_DEFINITION,
        "endpoint_target_contract": endpoint_contract.endpoint_target_contract,
        "wrist_local_rotation_policy": WRIST_LOCAL_ROTATION_POLICY,
        "vr3pt_orientation_policy": VR3PT_ORIENTATION_POLICY,
        "wrist_false_grasp_repair_policy": PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
        "wrist_false_grasp_max_island_duration_s": (
            PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S
        ),
        "wrist_false_grasp_min_endpoint_jump_deg": (
            PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG
        ),
        "wrist_false_grasp_max_upstream_step_deg": (
            PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG
        ),
        "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
        "wrist_filter_cutoff_hz": PICO_WRIST_FILTER_CUTOFF_HZ,
        "wrist_filter_order": PICO_WRIST_FILTER_ORDER,
        "wrist_grasp_enter_threshold": PICO_WRIST_GRASP_ENTER_THRESHOLD,
        "wrist_grasp_exit_threshold": PICO_WRIST_GRASP_EXIT_THRESHOLD,
        "wrist_grasp_padding_s": PICO_WRIST_GRASP_PADDING_S,
        "wrist_grasp_max_duration_s": PICO_WRIST_GRASP_MAX_DURATION_S,
        "wrist_grasp_min_residual_deg": PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
        "wrist_grasp_max_upstream_residual_deg": (
            PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
        ),
    }


def promote(args: argparse.Namespace) -> None:
    baseline = args.baseline.resolve()
    batch_root = args.batch_root.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise PromotionError(f"refusing to overwrite {output} or {temporary}")

    accepted, rejected = accepted_decisions(batch_root)
    batch_summary = load_json(batch_root / "summary.json")
    hand_endpoint_side = str(
        batch_summary.get("hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH)
    )
    endpoint_contract = endpoint_retarget_contract(
        ENDPOINT_PARTS_HANDS, hand_endpoint_side
    )
    endpoint_version = "egoalign"
    selected = parse_episode_selection(args.episodes)
    if selected is not None:
        accepted_ids = {int(item["compact_episode"]) for item in accepted}
        unavailable = sorted(selected - accepted_ids)
        if unavailable:
            raise PromotionError(
                f"selected episodes are not optimized_pass: {unavailable}"
            )
        accepted = [
            item for item in accepted if int(item["compact_episode"]) in selected
        ]
        rejected = []
    accepted.sort(key=lambda item: int(item["compact_episode"]))
    baseline_info = load_json(baseline / "meta/info.json")
    baseline_episodes = {
        int(item["episode_index"]): item
        for item in load_jsonl(baseline / "meta/episodes.jsonl")
    }
    baseline_stats = {
        int(item["episode_index"]): item
        for item in load_jsonl(baseline / "meta/episodes_stats.jsonl")
    }
    chunks = int(baseline_info.get("chunks_size", 1000))
    data_template = str(baseline_info["data_path"])
    videos = video_templates(baseline_info)
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    global_start = 0
    last_stage_c_info: dict[str, Any] | None = None

    try:
        temporary.mkdir(parents=True)
        with tempfile.TemporaryDirectory(
            prefix="endpoint-egoalign-promotion-", dir=str(batch_root)
        ) as scratch_text:
            scratch = Path(scratch_text)
            for ordinal, decision in enumerate(accepted):
                compact = int(decision["compact_episode"])
                source = int(decision["source_episode"])
                round_index = decision.get("accepted_round")
                if not isinstance(round_index, int):
                    raise PromotionError(f"accepted episode {compact} has no accepted round")
                round_dir = batch_root / f"episode_{compact:06d}/round_{round_index:02d}"
                staging_source = round_dir / "staging"
                replay, replay_override = select_promotion_replay(
                    round_dir, expected_algorithm=endpoint_contract.algorithm
                )
                contract = load_json(round_dir / "round_contract.json")
                round_identity = promotion_round_identity(hand_endpoint_side)
                for key, expected in round_identity.items():
                    if contract.get(key) != expected:
                        raise PromotionError(
                            f"round contract has incompatible {key}: {round_dir}"
                        )
                artifacts = contract.get("artifacts", {})
                for raw_path, expected_hash in artifacts.items():
                    path = Path(raw_path)
                    if not path.is_file() or sha256_file(path) != expected_hash:
                        raise PromotionError(f"round contract hash mismatch: {path}")
                baseline_parquet = format_episode_path(
                    baseline, data_template, compact, chunks
                )
                candidate_contract = validate_promotion_candidate(
                    round_dir,
                    baseline_parquet,
                    staging_source / "data/chunk-000/episode_000000.parquet",
                )

                staging = scratch / "staging"
                stage_c = scratch / "stage_c"
                shutil.copytree(staging_source, staging, copy_function=copy_or_link)
                augment_staging_stats(staging, baseline_stats, compact)
                run_stage_c(staging, replay, stage_c)

                source_parquet = stage_c / "data/chunk-000/episode_000000.parquet"
                table = replace_indices(pq.read_table(source_parquet), ordinal, global_start)
                baseline_schema = pq.ParquetFile(baseline_parquet).schema_arrow
                try:
                    table = table.cast(baseline_schema, safe=True)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                    raise PromotionError(
                        f"episode {compact}: cannot restore canonical training schema: {exc}"
                    ) from exc
                destination_parquet = format_episode_path(
                    temporary, data_template, ordinal, chunks
                )
                destination_parquet.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, destination_parquet, compression="snappy")

                episode_record = rewrite_value(
                    baseline_episodes[compact], compact, ordinal
                )
                episode_record["episode_index"] = ordinal
                episode_record["length"] = len(table)
                output_episodes.append(episode_record)
                stats_record = load_jsonl(stage_c / "meta/episodes_stats.jsonl")[0]
                stats_record["episode_index"] = ordinal
                output_stats.append(stats_record)

                for template in videos:
                    old_video = format_episode_path(baseline, template, compact, chunks)
                    new_video = format_episode_path(temporary, template, ordinal, chunks)
                    if not old_video.is_file():
                        raise PromotionError(f"baseline video missing: {old_video}")
                    copy_or_link(old_video, new_video)
                    old_sidecar = old_video.with_name(
                        f"{old_video.stem}.gpmf{old_video.suffix}"
                    )
                    if old_sidecar.is_file():
                        copy_or_link(
                            old_sidecar,
                            new_video.with_name(
                                f"{new_video.stem}.gpmf{new_video.suffix}"
                            ),
                        )
                for sync_source in sorted(
                    (baseline / "sync").glob(f"episode_{compact:06d}.*")
                ):
                    sync_target = temporary / "sync" / sync_source.name.replace(
                        f"episode_{compact:06d}", f"episode_{ordinal:06d}", 1
                    )
                    sync_target.parent.mkdir(parents=True, exist_ok=True)
                    if sync_source.suffix == ".json":
                        write_json(
                            sync_target,
                            rewrite_value(load_json(sync_source), compact, ordinal),
                        )
                    else:
                        copy_or_link(sync_source, sync_target)

                mappings.append(
                    {
                        "source_compact_episode": compact,
                        "raw_source_episode": source,
                        "accepted_round": round_index,
                        "promoted_episode": ordinal,
                        "frames": len(table),
                        "round_contract_sha256": sha256_file(round_dir / "round_contract.json"),
                        "promotion_replay": str(replay),
                        "promotion_replay_override": replay_override,
                        "candidate_contract": candidate_contract,
                    }
                )
                global_start += len(table)
                last_stage_c_info = load_json(stage_c / "meta/info.json")
                shutil.rmtree(staging)
                shutil.rmtree(stage_c)
                print(
                    f"[{ordinal + 1}/{len(accepted)}] compact {compact:06d} -> "
                    f"{endpoint_version} {ordinal:06d} ({len(table)} frames)",
                    flush=True,
                )

        if last_stage_c_info is None:
            raise PromotionError("no accepted episode was promoted")
        info = dict(last_stage_c_info)
        info.update(
            {
                "total_episodes": len(accepted),
                "total_frames": global_start,
                "total_videos": len(accepted) * len(videos),
                "total_chunks": (len(accepted) + chunks - 1) // chunks,
                "splits": {"train": f"0:{len(accepted)}"},
            }
        )
        config = info.setdefault("script_config", {})
        config.update(
            {
                "endpoint_retarget_version": endpoint_version,
                "endpoint_retarget_algorithm": endpoint_contract.algorithm,
                "endpoint_retarget_candidate_contract_version": (
                    endpoint_contract.candidate_contract_version
                ),
                "endpoint_retarget_endpoint_parts": ENDPOINT_PARTS_HANDS,
                "endpoint_retarget_endpoint_definition": endpoint_contract.endpoint_definition,
                "endpoint_retarget_hand_endpoint_side": hand_endpoint_side,
                "endpoint_retarget_wrist_local_rotation_policy": (
                    WRIST_LOCAL_ROTATION_POLICY
                ),
                "endpoint_retarget_vr3pt_orientation_policy": (
                    VR3PT_ORIENTATION_POLICY
                ),
                "endpoint_retarget_wrist_false_grasp_repair_policy": (
                    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
                ),
                "endpoint_retarget_wrist_motion_filter_policy": (
                    PICO_WRIST_MOTION_FILTER_POLICY
                ),
                "endpoint_retarget_batch_root": str(batch_root),
                "endpoint_retarget_promoted_only": True,
                "endpoint_retarget_rejected_count": len(rejected),
                "rollout_telemetry_root": "per-episode final accepted round; see endpoint_retarget_promotion_manifest.json",
                "feedback_source_episode_selection": "all promoted episodes",
                "feedback_source_episode_indices": list(range(len(accepted))),
                "feedback_unmodified_source_episode_indices": [],
            }
        )
        write_json(temporary / "meta/info.json", info)
        write_json(temporary / "meta/processing_mode.json", {
            "processing_mode": "endpoint-align",
            "endpoint_optimization": True,
            "history": "causal_measured",
            "motion_token": "same_tick_cpp_capture",
            "algorithm": endpoint_contract.algorithm,
            "batch_root": str(batch_root),
        })
        write_jsonl(temporary / "meta/episodes.jsonl", output_episodes)
        write_jsonl(temporary / "meta/episodes_stats.jsonl", output_stats)
        write_json(
            temporary / "meta/stats.json",
            recompute_global_stats(temporary, info),
        )
        write_json(temporary / "meta/relative_stats.json", {})
        for name in ("tasks.jsonl", "modality.json"):
            source = baseline / "meta" / name
            if source.is_file():
                copy_or_link(source, temporary / "meta" / name)
        write_json(
            temporary / "meta/endpoint_retarget_promotion_manifest.json",
            {
                "format_version": 1,
                "status": f"converted_{endpoint_version}",
                "algorithm": endpoint_contract.algorithm,
                "candidate_contract_version": (
                    endpoint_contract.candidate_contract_version
                ),
                "endpoint_parts": ENDPOINT_PARTS_HANDS,
                "endpoint_definition": endpoint_contract.endpoint_definition,
                "endpoint_target_contract": endpoint_contract.endpoint_target_contract,
                "hand_endpoint_side": hand_endpoint_side,
                "wrist_local_rotation_policy": WRIST_LOCAL_ROTATION_POLICY,
                "vr3pt_orientation_policy": VR3PT_ORIENTATION_POLICY,
                "wrist_false_grasp_repair_policy": (
                    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
                ),
                "wrist_false_grasp_max_island_duration_s": (
                    PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S
                ),
                "wrist_false_grasp_min_endpoint_jump_deg": (
                    PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG
                ),
                "wrist_false_grasp_max_upstream_step_deg": (
                    PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG
                ),
                "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
                "wrist_filter_cutoff_hz": PICO_WRIST_FILTER_CUTOFF_HZ,
                "wrist_filter_order": PICO_WRIST_FILTER_ORDER,
                "wrist_grasp_enter_threshold": PICO_WRIST_GRASP_ENTER_THRESHOLD,
                "wrist_grasp_exit_threshold": PICO_WRIST_GRASP_EXIT_THRESHOLD,
                "wrist_grasp_padding_s": PICO_WRIST_GRASP_PADDING_S,
                "wrist_grasp_max_duration_s": PICO_WRIST_GRASP_MAX_DURATION_S,
                "wrist_grasp_min_residual_deg": PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
                "wrist_grasp_max_upstream_residual_deg": (
                    PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "baseline_dataset": str(baseline),
                "batch_root": str(batch_root),
                "output": str(output),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "accepted_mapping": mappings,
                "rejected": [
                    {
                        "source_compact_episode": int(item["compact_episode"]),
                        "raw_source_episode": int(item["source_episode"]),
                        "status": item["status"],
                    }
                    for item in rejected
                ],
            },
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"promoted {len(accepted)} episodes / {global_start} frames -> {output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Accepted compact episodes to promote (all, 0,2, 4-8).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    promote(parse_args())

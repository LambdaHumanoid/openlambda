#!/usr/bin/env python3
"""Fail-closed dataset orchestrator for SMPL-to-G1 endpoint retargeting.

The orchestrator deliberately does not import MuJoCo. Candidate construction
and replay evaluation use the configured data and simulation interpreters.
SONIC/MuJoCo rollout is delegated to the bundled ``tools/run_sim_replay.sh``
simulation replay adapter.
"""

from __future__ import annotations
from egoalign.runtime import asset_path, human_skeleton_path, validate_replay_workers

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.human_data.smpl_endpoint_retargeting import (
    DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    ENDPOINT_PART_CHOICES,
    ENDPOINT_PARTS_HANDS,
    HAND_ENDPOINT_SIDE_BOTH,
    HAND_ENDPOINT_SIDE_CHOICES,
    ENDPOINT_RETARGET_ALGORITHM,
    LEFT_ENDPOINT_OUTPUT_INDEX,
    RIGHT_ENDPOINT_OUTPUT_INDEX,
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
    canonical_output_joints,
    endpoint_retarget_contract,
    scale_canonical_rest_to_head_height,
    validate_endpoint_candidate_archive,
)
from egoalign.adapters.sonic.telemetry_schema import (  # noqa: E402
    MUJOCO_BODY_JOINT_NAMES,
)
from egoalign.adapters.sonic.rollout_contract import (  # noqa: E402
    INITIAL_BODY_Q_REFERENCE_CONTRACT,
    STRICT_AB_INITIALIZATION_CONTRACT,
)

DATA_PYTHON = Path(os.environ.get("EGOALIGN_DATA_PYTHON", sys.executable))
SIM_PYTHON = Path(os.environ.get("EGOALIGN_SIM_PYTHON", sys.executable))
REPLAY_LAUNCHER = PROJECT_ROOT / "tools/run_sim_replay.sh"
BUILDER = PROJECT_ROOT / "scripts/human_data/build_smpl_endpoint_retarget_candidate.py"
EVALUATOR = PROJECT_ROOT / "scripts/human_data/evaluate_endpoint_retarget_replay.py"
PRODUCTION_AUDITOR = (
    PROJECT_ROOT / "scripts/human_data/audit_endpoint_retarget_production.py"
)
ROBOT_XML = (
    asset_path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
)
DEFAULT_BASELINE_REPLAY_PARENT = PROJECT_ROOT / "outputs/replay_velocity_quality"
MAX_ROUNDS = 3
ALGORITHM = ENDPOINT_RETARGET_ALGORITHM
VALID_DECISIONS = {
    "accepted_early_stop",
    "continue",
    "accepted_at_budget_limit",
    "replay_budget_exhausted",
    "rejected_unstable",
}
TERMINAL_ACCEPT_DECISIONS = {
    "accepted_early_stop",
    "accepted_at_budget_limit",
}
ROLLOUT_CONTRACT = (
    "deploy-exit->sim-reset+bridge-reset+elastic-reset->fresh-deploy->"
    "planner-history-settle->elastic-release+released-physical-settle->"
    "pose-ramp+first-pose-physical-settle+support-integrity-gate->formal-replay:v4"
)
BASELINE_CONTRACTS = {"validated_baseline", "minimal_ab"}
STAGE_A_SOURCE_MAPPING_CONTRACTS = {
    1: "stage_a_source_episode_mapping",
    2: "stage_a_source_episode_mapping_short_grasp_cleaned",
}
SOURCE_EPISODE_VIEW_DIR = "by_source_episode"
STRICT_AB_DIR = "strict_ab"


def endpoint_contract_identity(
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    """Return the explicit cache identity for bilateral-palm alignment."""

    contract = endpoint_retarget_contract(endpoint_parts, hand_endpoint_side)
    wrist_policy = (
        (WRIST_LOCAL_ROTATION_POLICY)
    )
    orientation_policy = (
        (VR3PT_ORIENTATION_POLICY)
    )
    identity = {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": endpoint_parts,
        "hand_endpoint_side": hand_endpoint_side,
        "endpoint_definition": contract.endpoint_definition,
        "endpoint_target_contract": contract.endpoint_target_contract,
        "wrist_local_rotation_policy": wrist_policy,
        "vr3pt_orientation_policy": orientation_policy,
        "canonical_head_height_m": float(canonical_head_height_m),
    }
    identity.update(
        {
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
            "wrist_grasp_enter_threshold": (
                PICO_WRIST_GRASP_ENTER_THRESHOLD
            ),
            "wrist_grasp_exit_threshold": PICO_WRIST_GRASP_EXIT_THRESHOLD,
            "wrist_grasp_padding_s": PICO_WRIST_GRASP_PADDING_S,
            "wrist_grasp_max_duration_s": PICO_WRIST_GRASP_MAX_DURATION_S,
            "wrist_grasp_min_residual_deg": (
                PICO_WRIST_GRASP_MIN_RESIDUAL_DEG
            ),
            "wrist_grasp_max_upstream_residual_deg": (
                PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
            ),
        }
    )
    from egoalign.staged_ik import CONFIG_JSON
    identity["staged_config_json"] = CONFIG_JSON
    return identity


def endpoint_contract_label(endpoint_parts: str) -> str:
    endpoint_retarget_contract(endpoint_parts)
    return "palm"


class ContractError(RuntimeError):
    """Raised when an input or generated artifact violates the batch contract."""


@dataclass(frozen=True)
class EpisodeMapping:
    compact_to_source: tuple[int, ...]
    dropped_sources: tuple[int, ...]
    aligned_rows: tuple[int, ...]
    baseline_rows: tuple[int, ...]
    solution_count: int
    raw_selection_mode: str = "alignment"


@dataclass(frozen=True)
class EpisodePaths:
    compact_episode: int
    source_episode: int
    raw_parquet: Path
    baseline_parquet: Path


@dataclass(frozen=True)
class ParallelReplayItem:
    """One isolated endpoint round admitted to a parallel replay batch."""

    batch_episode: int
    compact_episode: int
    source_episode: int
    round_index: int
    round_dir: Path
    staging_parquet: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON object required: {path}")
    return payload


def validate_baseline_dataset_contract(
    baseline_dataset: Path, baseline_contract: str
) -> dict[str, Any]:
    """Validate whether the baseline dataset is a full baseline or minimal Stage-A input.

    Endpoint correction needs canonical SMPL/real-wrist rows plus the matching
    unretargeted Stage-B control capture.  Stage C and stats are useful for a
    reusable baseline dataset, but are not mathematical inputs to endpoint correction.
    The explicit ``minimal_ab`` contract keeps that distinction auditable.
    """

    if baseline_contract not in BASELINE_CONTRACTS:
        raise ContractError(f"unsupported baseline contract: {baseline_contract!r}")
    info_path = baseline_dataset / "meta/info.json"
    info = load_json(info_path)
    script_config = info.get("script_config")
    if not isinstance(script_config, dict):
        raise ContractError(f"baseline script_config is missing: {info_path}")
    decoder_mode = str(script_config.get("decoder_history_mode", ""))
    pipeline = str(script_config.get("pipeline", ""))
    source_mapping_report: dict[str, Any] | None = None
    if baseline_contract == "minimal_ab":
        if decoder_mode != "zero":
            raise ContractError(
                "minimal_ab baseline must be Stage-A zero history; "
                f"got decoder_history_mode={decoder_mode!r}"
            )
        if "raw_pico_to_smpl_to_sonic_encoder_decoder_to_gr00t" not in pipeline:
            raise ContractError(
                f"minimal_ab baseline has an unexpected pipeline: {pipeline!r}"
            )
        source_mapping_path = baseline_dataset / "meta/stage_a_source_mapping.json"
        source_mapping = load_json(source_mapping_path)
        mapping_version = source_mapping.get("format_version")
        if (
            source_mapping.get("contract")
            != STAGE_A_SOURCE_MAPPING_CONTRACTS.get(mapping_version)
            or not isinstance(source_mapping.get("episodes"), list)
            or not source_mapping["episodes"]
        ):
            raise ContractError(
                "minimal_ab baseline requires a non-empty supported "
                f"Stage-A source mapping: {source_mapping_path}"
            )
        source_mapping_report = {
            "path": str(source_mapping_path),
            "sha256": sha256_file(source_mapping_path),
        }
    else:
        if decoder_mode != "captured_cpp_mujoco_feedback":
            raise ContractError(
                "validated_baseline baseline must contain captured C++ MuJoCo feedback; "
                f"got decoder_history_mode={decoder_mode!r}"
            )
        if not (baseline_dataset / "meta/stats.json").is_file():
            raise ContractError("validated_baseline baseline is missing meta/stats.json")
    return {
        "contract": baseline_contract,
        "dataset": str(baseline_dataset),
        "decoder_history_mode": decoder_mode,
        "pipeline": pipeline,
        "info_sha256": sha256_file(info_path),
        "stats_required": baseline_contract == "validated_baseline",
        "stage_a_source_mapping": source_mapping_report,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSONL {path}:{lineno}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"JSON object required at {path}:{lineno}")
        records.append(payload)
    return records


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_relative_symlink(target: Path, alias: Path) -> None:
    """Publish a relocatable derived view without duplicating large artifacts."""

    if not target.is_file():
        raise ContractError(f"source-view target is missing: {target}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=alias.parent)
    if alias.is_symlink():
        if (
            os.readlink(alias) == relative_target
            and alias.resolve() == target.resolve()
        ):
            return
    elif alias.exists():
        raise ContractError(f"source-view alias is not a symlink: {alias}")

    fd, temp_name = tempfile.mkstemp(prefix=f".{alias.name}.", dir=alias.parent)
    os.close(fd)
    os.unlink(temp_name)
    try:
        os.symlink(relative_target, temp_name)
        os.replace(temp_name, alias)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def publish_source_episode_views(
    output_root: Path,
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose completed MuJoCo replays with raw/source episode identities.

    Canonical endpoint directories intentionally remain compact because their
    paths are part of the LeRobot, telemetry, resume, and promotion contracts.
    Humans must not have to translate that index space, so this derived view
    names both its directory and every MuJoCo MP4 with the immutable source
    episode number.
    """

    views: list[dict[str, Any]] = []
    seen_sources: set[int] = set()
    for decision in decisions:
        try:
            compact_episode = int(decision["compact_episode"])
            source_episode = int(decision["source_episode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"completed decision has invalid episode identity: {decision}"
            ) from exc
        if compact_episode < 0 or source_episode < 0:
            raise ContractError("episode identities must be non-negative")
        if source_episode in seen_sources:
            raise ContractError(
                f"duplicate source episode in human-facing view: {source_episode}"
            )
        seen_sources.add(source_episode)

        canonical_episode_root = output_root / f"episode_{compact_episode:06d}"
        source_episode_root = (
            output_root
            / SOURCE_EPISODE_VIEW_DIR
            / f"episode_{source_episode:06d}"
        )
        final_decision = canonical_episode_root / "final_decision.json"
        atomic_relative_symlink(
            final_decision,
            source_episode_root / "final_decision.json",
        )

        rounds = decision.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise ContractError(
                f"completed source episode {source_episode} has no replay rounds"
            )
        round_views: list[dict[str, Any]] = []
        seen_rounds: set[int] = set()
        for record in rounds:
            if not isinstance(record, dict):
                raise ContractError("completed replay round must be an object")
            try:
                round_index = int(record["round"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError(
                    f"invalid completed replay round: {record}"
                ) from exc
            if round_index < 0 or round_index >= MAX_ROUNDS:
                raise ContractError(f"source-view round is out of range: {round_index}")
            if round_index in seen_rounds:
                raise ContractError(
                    f"duplicate round for source episode {source_episode}: {round_index}"
                )
            seen_rounds.add(round_index)

            canonical_round = canonical_episode_root / f"round_{round_index:02d}"
            canonical_replay = canonical_round / "replay/episode_000000.mp4"
            source_round = source_episode_root / f"round_{round_index:02d}"
            source_replay = (
                source_round
                / f"mujoco_replay_episode_{source_episode:06d}.mp4"
            )
            # Production batches may intentionally run in telemetry-only mode.
            # In that case the replay has already been validated from its
            # control/sim/rollout artifacts, while human-facing videos are
            # rendered only for the final deterministic sample.  Do not turn
            # an intentionally absent intermediate MP4 into a dataset failure.
            video_available = canonical_replay.is_file()
            if video_available:
                atomic_relative_symlink(canonical_replay, source_replay)
            elif source_replay.is_symlink():
                source_replay.unlink()
            atomic_relative_symlink(
                canonical_round / "replay_metrics.json",
                source_round / "replay_metrics.json",
            )
            round_views.append(
                {
                    "round": round_index,
                    "decision": record.get("decision"),
                    "mujoco_replay": (
                        str(source_replay) if video_available else None
                    ),
                    "canonical_compact_replay": (
                        str(canonical_replay) if video_available else None
                    ),
                    "video_status": (
                        "available"
                        if video_available
                        else "deferred_final_sample_only"
                    ),
                }
            )

        mapping = {
            "format_version": 1,
            "contract": "human-facing source episode view",
            "source_episode": source_episode,
            "compact_episode": compact_episode,
            "canonical_compact_episode_root": str(canonical_episode_root),
            "rounds": round_views,
        }
        atomic_write_json(source_episode_root / "source_mapping.json", mapping)
        views.append(mapping)
    return views


def run_checked(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(f"[endpoint-retarget] run: {printable}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False, env=env)
    if completed.returncode != 0:
        raise ContractError(
            f"command failed with exit code {completed.returncode}: {printable}"
        )


def _episode_parquets(root: Path) -> list[Path]:
    paths = sorted((root / "data").rglob("episode_*.parquet"))
    if not paths:
        raise ContractError(f"dataset has no episode parquet: {root}")
    return paths


def alignment_interval(sync_dir: Path, source_episode: int) -> tuple[float, float]:
    intervals: list[tuple[float, float]] = []
    for camera in ("gopro", "level-cam"):
        path = sync_dir / f"episode_{source_episode:06d}.{camera}_alignment.json"
        payload = load_json(path)
        try:
            offset = float(payload["offset_s"])
            duration = float(payload["camera_duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid alignment interval in {path}: {exc}") from exc
        if not np.isfinite([offset, duration]).all() or duration <= 0:
            raise ContractError(f"non-finite/non-positive alignment interval in {path}")
        intervals.append((offset, offset + duration))
    start = max(value[0] for value in intervals)
    end = min(value[1] for value in intervals)
    if end <= start:
        raise ContractError(
            f"camera intervals do not overlap for source episode {source_episode}: {intervals}"
        )
    return start, end


def _count_ordered_length_mappings(
    baseline_rows: list[int], aligned_rows: list[int]
) -> tuple[int, list[int]]:
    """Return capped solution count and the unique compact-to-source mapping."""

    m, n = len(baseline_rows), len(aligned_rows)
    ways = [[0] * (n + 1) for _ in range(m + 1)]
    for column in range(n + 1):
        ways[0][column] = 1
    for row in range(1, m + 1):
        for column in range(1, n + 1):
            count = ways[row][column - 1]
            if baseline_rows[row - 1] == aligned_rows[column - 1]:
                count += ways[row - 1][column - 1]
            ways[row][column] = min(count, 2)
    solution_count = ways[m][n]
    if solution_count != 1:
        return solution_count, []

    mapping: list[int] = []
    row, column = m, n
    while row > 0:
        can_match = (
            column > 0
            and baseline_rows[row - 1] == aligned_rows[column - 1]
            and ways[row - 1][column - 1] > 0
        )
        if can_match:
            mapping.append(column - 1)
            row -= 1
            column -= 1
        elif column > 0 and ways[row][column - 1] > 0:
            column -= 1
        else:
            raise ContractError("ordered mapping backtrack failed")
    mapping.reverse()
    return solution_count, mapping


def derive_mapping(raw_dataset: Path, baseline_dataset: Path) -> EpisodeMapping:
    raw_paths = _episode_parquets(raw_dataset)
    baseline_paths = _episode_parquets(baseline_dataset)
    aligned_rows: list[int] = []
    for source_episode, path in enumerate(raw_paths):
        timestamps = (
            pq.read_table(path, columns=["timestamp"])
            .column(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ContractError(f"raw timestamp is non-finite/non-monotonic: {path}")
        start, end = alignment_interval(raw_dataset / "sync", source_episode)
        aligned_rows.append(int(np.count_nonzero((timestamps >= start) & (timestamps <= end))))
    baseline_rows = [int(pq.ParquetFile(path).metadata.num_rows) for path in baseline_paths]
    count, mapping = _count_ordered_length_mappings(baseline_rows, aligned_rows)
    if count != 1:
        raise ContractError(
            f"raw-to-baseline mapping is not unique: solutions_capped_at_2={count}"
        )
    if len(mapping) != len(baseline_paths) or len(set(mapping)) != len(mapping):
        raise ContractError("raw-to-baseline mapping is incomplete or contains duplicates")
    dropped = tuple(index for index in range(len(raw_paths)) if index not in set(mapping))
    return EpisodeMapping(
        compact_to_source=tuple(mapping),
        dropped_sources=dropped,
        aligned_rows=tuple(aligned_rows),
        baseline_rows=tuple(baseline_rows),
        solution_count=count,
    )


def _indexed_episode_parquets(root: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in _episode_parquets(root):
        suffix = path.stem.removeprefix("episode_")
        if not suffix.isdigit():
            raise ContractError(f"invalid episode parquet name: {path}")
        episode = int(suffix)
        if episode in indexed:
            raise ContractError(
                f"duplicate episode parquet index {episode} under {root}"
            )
        indexed[episode] = path.resolve()
    return indexed


def derive_minimal_ab_mapping(
    raw_dataset: Path, baseline_dataset: Path
) -> tuple[EpisodeMapping, dict[str, Any]]:
    """Resolve raw-to-Stage-A identity only from the conversion-time manifest.

    A minimal Stage A may intentionally ignore an invalid camera alignment.  In
    that case, reconstructing source identity from alignment-cropped row counts
    is both impossible and conceptually wrong.  The converter therefore binds
    each pair of parquets by path, row count, and SHA-256 at conversion time;
    this function validates that immutable mapping without relabeling alignment.
    """

    manifest_path = baseline_dataset / "meta/stage_a_source_mapping.json"
    payload = load_json(manifest_path)
    mapping_version = payload.get("format_version")
    if mapping_version not in STAGE_A_SOURCE_MAPPING_CONTRACTS:
        raise ContractError(f"unsupported Stage-A source mapping version: {manifest_path}")
    if payload.get("contract") != STAGE_A_SOURCE_MAPPING_CONTRACTS[mapping_version]:
        raise ContractError(f"invalid Stage-A source mapping contract: {manifest_path}")
    source_dataset = payload.get("source_dataset")
    if not isinstance(source_dataset, str) or not Path(source_dataset).is_absolute():
        raise ContractError("Stage-A source mapping requires an absolute source_dataset")
    source_dataset_relocated = Path(source_dataset).resolve() != raw_dataset.resolve()
    alignment_mode = payload.get("alignment_mode")
    if alignment_mode not in {"ignore", "require", "apply_if_present"}:
        raise ContractError(
            f"invalid Stage-A source mapping alignment_mode={alignment_mode!r}"
        )
    if payload.get("max_frames_per_episode") is not None:
        raise ContractError(
            "endpoint retargeting requires an uncapped minimal Stage A; "
            "max_frames_per_episode must be null"
        )

    records = payload.get("episodes")
    if not isinstance(records, list) or not records:
        raise ContractError("Stage-A source mapping has no episode records")
    raw_paths = _indexed_episode_parquets(raw_dataset)
    baseline_paths = _indexed_episode_parquets(baseline_dataset)
    expected_raw_ids = set(range(len(raw_paths)))
    expected_baseline_ids = set(range(len(baseline_paths)))
    if set(raw_paths) != expected_raw_ids:
        raise ContractError(
            "minimal_ab raw episode IDs must be compact 0..N-1; "
            f"got {sorted(raw_paths)}"
        )
    if set(baseline_paths) != expected_baseline_ids:
        raise ContractError(
            "minimal_ab Stage-A episode IDs must be compact 0..M-1; "
            f"got {sorted(baseline_paths)}"
        )
    if len(records) != len(baseline_paths):
        raise ContractError(
            "Stage-A source mapping record count does not match baseline parquets: "
            f"records={len(records)}, parquets={len(baseline_paths)}"
        )

    records_by_baseline: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("Stage-A source mapping episode must be an object")
        try:
            baseline_episode = int(record["baseline_episode"])
            source_episode = int(record["source_episode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid Stage-A episode identity: {record}") from exc
        if baseline_episode in records_by_baseline:
            raise ContractError(
                f"duplicate Stage-A baseline episode mapping: {baseline_episode}"
            )
        if source_episode not in raw_paths:
            raise ContractError(
                f"Stage-A source episode is absent from raw dataset: {source_episode}"
            )
        records_by_baseline[baseline_episode] = record
    if set(records_by_baseline) != expected_baseline_ids:
        raise ContractError(
            "Stage-A source mapping baseline IDs are incomplete/non-compact: "
            f"got {sorted(records_by_baseline)}"
        )

    compact_to_source: list[int] = []
    baseline_rows: list[int] = []
    seen_sources: set[int] = set()
    for baseline_episode in range(len(baseline_paths)):
        record = records_by_baseline[baseline_episode]
        source_episode = int(record["source_episode"])
        if source_episode in seen_sources:
            raise ContractError(
                f"duplicate Stage-A source episode mapping: {source_episode}"
            )
        seen_sources.add(source_episode)
        expected_paths = {
            "raw_parquet": raw_paths[source_episode],
            "baseline_parquet": baseline_paths[baseline_episode],
        }
        for field, expected_path in expected_paths.items():
            recorded = record.get(field)
            if not isinstance(recorded, str) or not Path(recorded).is_absolute():
                raise ContractError(f"{field} must be an absolute path in {manifest_path}")
            path_relocated = (
                field == "raw_parquet"
                and source_dataset_relocated
                and Path(recorded).name == expected_path.name
            )
            if Path(recorded).resolve() != expected_path and not path_relocated:
                raise ContractError(
                    f"{field} path mismatch for baseline episode {baseline_episode}: "
                    f"recorded={Path(recorded).resolve()}, expected={expected_path}"
                )
        current_raw_rows = int(
            pq.ParquetFile(raw_paths[source_episode]).metadata.num_rows
        )
        current_baseline_rows = int(
            pq.ParquetFile(baseline_paths[baseline_episode]).metadata.num_rows
        )
        try:
            recorded_raw_rows = int(record["raw_rows"])
            recorded_baseline_rows = int(record["baseline_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"invalid Stage-A row counts for baseline episode {baseline_episode}"
            ) from exc
        if recorded_raw_rows != current_raw_rows:
            raise ContractError(
                f"raw row-count mismatch for source episode {source_episode}: "
                f"recorded={recorded_raw_rows}, current={current_raw_rows}"
            )
        if recorded_baseline_rows != current_baseline_rows:
            raise ContractError(
                f"baseline row-count mismatch for episode {baseline_episode}: "
                f"recorded={recorded_baseline_rows}, current={current_baseline_rows}"
            )
        if alignment_mode == "ignore" and current_raw_rows != current_baseline_rows:
            raise ContractError(
                "uncapped ignore-alignment Stage A must preserve every raw row: "
                f"source={source_episode}, raw={current_raw_rows}, "
                f"baseline={current_baseline_rows}"
            )
        for field, path in (
            ("raw_parquet_sha256", raw_paths[source_episode]),
            ("baseline_parquet_sha256", baseline_paths[baseline_episode]),
        ):
            recorded_hash = record.get(field)
            current_hash = sha256_file(path)
            if recorded_hash != current_hash:
                raise ContractError(
                    f"Stage-A source mapping hash mismatch for {path}: "
                    f"recorded={recorded_hash}, current={current_hash}"
                )
        compact_to_source.append(source_episode)
        baseline_rows.append(current_baseline_rows)

    raw_rows = tuple(
        int(pq.ParquetFile(raw_paths[index]).metadata.num_rows)
        for index in range(len(raw_paths))
    )
    dropped = tuple(index for index in range(len(raw_paths)) if index not in seen_sources)
    preserves_full_source_rows = all(
        raw_rows[source_episode] == baseline_rows[baseline_episode]
        for baseline_episode, source_episode in enumerate(compact_to_source)
    )
    mapping = EpisodeMapping(
        compact_to_source=tuple(compact_to_source),
        dropped_sources=dropped,
        aligned_rows=raw_rows,
        baseline_rows=tuple(baseline_rows),
        solution_count=1,
        # A pre-filtered source dataset already contains exactly the admitted
        # camera-overlap rows.  Reapplying its capture-time alignment interval
        # would crop those rows a second time.  Equal source/baseline row counts
        # are an immutable, hash-validated proof that full-row selection is the
        # correct identity mapping regardless of the Stage-A admission policy.
        raw_selection_mode=(
            "full"
            if alignment_mode == "ignore" or preserves_full_source_rows
            else "alignment"
        ),
    )
    authority = {
        "contract": "conversion-time path+row-count+sha256 source mapping",
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "alignment_mode": alignment_mode,
        "mapping_format_version": int(mapping_version),
        "source_dataset_relocated": source_dataset_relocated,
        "recorded_source_dataset": source_dataset,
        "resolved_source_dataset": str(raw_dataset.resolve()),
        "preserves_full_source_rows": preserves_full_source_rows,
        "max_frames_per_episode": None,
    }
    return mapping, authority


def _alignment_quality(raw_dataset: Path, source_episode: int) -> str:
    path = raw_dataset / "sync" / f"episode_{source_episode:06d}.all_cam_alignment.json"
    quality = str(load_json(path).get("quality", "missing")).lower()
    if quality not in {"pass", "warn", "fail"}:
        raise ContractError(f"unrecognized alignment quality {quality!r}: {path}")
    return quality


def audit_mapping_against_alignment(
    raw_dataset: Path, mapping: EpisodeMapping
) -> dict[str, Any]:
    """Describe current alignment labels without rewriting baseline admission history.

    Older validated baseline datasets used batch-specific visual-motion admission,
    deterministic rollout rejection, or other documented filters.  Their
    membership therefore cannot be reconstructed by assuming that the current
    alignment JSON's ``fail`` label is the sole keep/drop rule.  The unique
    row mapping remains the identity authority; current alignment quality is
    still recorded and is enforced later for smoke eligibility.
    """
    included = set(mapping.compact_to_source)
    if mapping.raw_selection_mode == "full":
        return {
            "contract": (
                "ignore-alignment minimal Stage A preserves full source rows; "
                "camera alignment labels are diagnostic-only and are not required "
                "for dropped or included source episodes"
            ),
            "alignment_required": False,
            "included_source_episodes": sorted(included),
            "dropped_source_episodes": list(mapping.dropped_sources),
        }
    included_by_quality = {quality: [] for quality in ("pass", "warn", "fail")}
    dropped_by_quality = {quality: [] for quality in ("pass", "warn", "fail")}
    for source_episode in range(len(mapping.aligned_rows)):
        quality = _alignment_quality(raw_dataset, source_episode)
        target = included_by_quality if source_episode in included else dropped_by_quality
        target[quality].append(source_episode)
    return {
        "contract": (
            "unique aligned-row source mapping is authoritative; current alignment "
            "quality is audit metadata and a smoke-eligibility check, not a historical "
            "baseline membership reconstruction rule"
        ),
        "included_by_current_alignment_quality": included_by_quality,
        "dropped_by_current_alignment_quality": dropped_by_quality,
    }


def derive_baseline_replay_indices(
    baseline_dataset: Path,
    baseline_replay_root: Path,
    successful_episodes: list[int],
    baseline_rows: tuple[int, ...],
) -> tuple[tuple[int, ...], str | None]:
    """Map compact converted IDs to their immutable Stage-B replay IDs.

    Most datasets use identical compact IDs.  A validated rollout rejection
    may compact Stage C while preserving original Stage-B telemetry IDs; in
    that case the dataset's rejection manifest is the only accepted mapping
    authority.
    """
    total = len(baseline_rows)
    identity = tuple(range(total))
    if successful_episodes == list(identity):
        replay_indices = identity
        manifest_path: str | None = None
    else:
        candidates: list[tuple[tuple[int, ...], Path]] = []
        for path in sorted((baseline_dataset / "rejected_episodes").glob("**/manifest.json")):
            payload = load_json(path)
            raw_mapping = payload.get("old_to_new")
            if not isinstance(raw_mapping, dict):
                continue
            try:
                old_to_new = {int(old): int(new) for old, new in raw_mapping.items()}
            except (TypeError, ValueError):
                continue
            if set(old_to_new.values()) != set(identity):
                continue
            if len(old_to_new) != total:
                continue
            inverse = [None] * total
            for old, new in old_to_new.items():
                if inverse[new] is not None:
                    break
                inverse[new] = old
            if any(index is None for index in inverse):
                continue
            resolved = tuple(int(index) for index in inverse)
            if list(resolved) == successful_episodes:
                candidates.append((resolved, path))
        if len(candidates) != 1:
            raise ContractError(
                "baseline replay IDs are non-compact and do not have exactly one "
                f"matching rejection manifest: candidates={len(candidates)}"
            )
        replay_indices, authority = candidates[0]
        manifest_path = str(authority)

    for compact_episode, replay_episode in enumerate(replay_indices):
        control = (
            baseline_replay_root
            / "telemetry"
            / f"episode_{replay_episode:06d}.control.npz"
        )
        if not control.is_file():
            raise ContractError(f"baseline replay control telemetry missing: {control}")
        try:
            with np.load(control, allow_pickle=False) as payload:
                rows = int(np.asarray(payload["replay_frame_index"]).shape[0])
        except (OSError, KeyError, ValueError) as exc:
            raise ContractError(f"cannot validate baseline control telemetry {control}: {exc}") from exc
        if rows != baseline_rows[compact_episode]:
            raise ContractError(
                "baseline compact/replay row mismatch: "
                f"compact={compact_episode}, replay={replay_episode}, "
                f"parquet_rows={baseline_rows[compact_episode]}, control_rows={rows}"
            )
    return replay_indices, manifest_path


def baseline_control_path(
    baseline_replay_root: Path,
    baseline_replay_indices: tuple[int, ...],
    compact_episode: int,
) -> Path:
    replay_episode = baseline_replay_indices[compact_episode]
    path = (
        baseline_replay_root
        / "telemetry"
        / f"episode_{replay_episode:06d}.control.npz"
    )
    if not path.is_file():
        raise ContractError(f"baseline control telemetry missing: {path}")
    return path


def episode_paths(
    raw_dataset: Path,
    baseline_dataset: Path,
    mapping: EpisodeMapping,
    compact_episode: int,
) -> EpisodePaths:
    if not 0 <= compact_episode < len(mapping.compact_to_source):
        raise ContractError(f"compact episode out of range: {compact_episode}")
    source_episode = mapping.compact_to_source[compact_episode]
    raw_path = raw_dataset / "data" / f"chunk-{source_episode // 1000:03d}" / f"episode_{source_episode:06d}.parquet"
    baseline_path = baseline_dataset / "data" / f"chunk-{compact_episode // 1000:03d}" / f"episode_{compact_episode:06d}.parquet"
    if not raw_path.is_file() or not baseline_path.is_file():
        raise ContractError(f"missing mapped parquet: raw={raw_path}, baseline={baseline_path}")
    return EpisodePaths(compact_episode, source_episode, raw_path, baseline_path)


def parse_episode_selection(expression: str, total: int) -> list[int]:
    text = expression.strip().lower()
    if text in {"", "all", "*"}:
        return list(range(total))
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            fields = token.split(":")
            if len(fields) not in {2, 3}:
                raise ContractError(f"invalid episode slice: {token}")
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else total
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            if step <= 0:
                raise ContractError(f"episode slice step must be positive: {token}")
            selected.update(range(start, stop, step))
        else:
            selected.add(int(token))
    invalid = sorted(index for index in selected if index < 0 or index >= total)
    if invalid:
        raise ContractError(f"episode selection out of range: {invalid}")
    return sorted(selected)


def _stack_nested(values: Iterable[Any], shape: tuple[int, ...]) -> np.ndarray:
    rows = []
    for value in values:
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = np.stack(list(value))
        rows.append(np.asarray(arr, dtype=np.float64).reshape(shape))
    return np.stack(rows)


def _hand_activity(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values).all(axis=1)
    if not finite.all() or len(values) < 2:
        return {
            "finite_frames": int(np.sum(finite)),
            "active_frames": 0,
            "max_displacement_m": 0.0,
            "max_step_m": 0.0,
            "path_length_m": 0.0,
        }
    speed = np.r_[0.0, np.linalg.norm(np.diff(values, axis=0), axis=1)]
    displacement = np.linalg.norm(values - values[0], axis=1)
    threshold = max(0.0015, float(np.percentile(speed[finite], 60)))
    active = finite & ((speed >= threshold) | (displacement >= 0.03))
    return {
        "finite_frames": int(np.sum(finite)),
        "active_frames": int(np.sum(active)),
        "max_displacement_m": float(np.max(displacement)),
        "max_step_m": float(np.max(speed)),
        "path_length_m": float(np.sum(speed)),
        "speed_threshold_m_per_frame": threshold,
    }


def _column_finite(frame: pd.DataFrame, column: str, shape: tuple[int, ...]) -> bool:
    if column not in frame.columns:
        return False
    try:
        values = _stack_nested(frame[column], shape)
    except (TypeError, ValueError):
        return False
    return bool(len(values) == len(frame) and np.isfinite(values).all())


def rank_smoke_episodes(
    raw_dataset: Path,
    baseline_dataset: Path,
    mapping: EpisodeMapping,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> list[dict[str, Any]]:
    contract = endpoint_retarget_contract(endpoint_parts, hand_endpoint_side)
    selected_indices: list[tuple[str, int]] = []
    selected_indices.extend(
        (
            ("hand_left", LEFT_ENDPOINT_OUTPUT_INDEX),
            ("hand_right", RIGHT_ENDPOINT_OUTPUT_INDEX),
        )
    )
    human_model = torch.load(
        human_skeleton_path(),
        map_location="cpu",
        weights_only=False,
    )
    canonical_rest, _ = scale_canonical_rest_to_head_height(
        human_model["J"].cpu().numpy(),
        canonical_head_height_m,
    )
    parents = list(human_model["parents_list"])
    candidates: list[dict[str, Any]] = []
    for compact_episode, source_episode in enumerate(mapping.compact_to_source):
        alignment_quality = (
            "ignored"
            if mapping.raw_selection_mode == "full"
            else _alignment_quality(raw_dataset, source_episode)
        )
        paths = episode_paths(raw_dataset, baseline_dataset, mapping, compact_episode)
        required_columns = {
            "teleop.smpl_joints": (24, 3),
            "teleop.smpl_pose": (63,),
            "teleop.vr_3pt_position": (9,),
            "teleop.vr_3pt_orientation": (18,),
            "teleop.left_wrist_joints": (3,),
            "teleop.right_wrist_joints": (3,),
            "observation.state": (43,),
            "action.motion_token": (64,),
        }
        available = set(pq.ParquetFile(paths.baseline_parquet).schema_arrow.names)
        missing_columns = sorted(set(required_columns) - available)
        columns = ["timestamp", *sorted(set(required_columns) & available)]
        frame = pd.read_parquet(paths.baseline_parquet, columns=columns)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        finite_checks = {
            column: _column_finite(frame, column, shape)
            for column, shape in required_columns.items()
        }
        palm_reconstruction_valid = False
        if (
            finite_checks["teleop.smpl_joints"]
            and finite_checks["teleop.smpl_pose"]
        ):
            try:
                baseline_joints = _stack_nested(
                    frame["teleop.smpl_joints"], (24, 3)
                )
                baseline_pose = _stack_nested(frame["teleop.smpl_pose"], (63,))
                current_joints = canonical_output_joints(
                    baseline_pose,
                    canonical_rest,
                    parents,
                    baseline_joints,
                )
                endpoint_activity = {
                    name: _hand_activity(current_joints[:, index])
                    for name, index in selected_indices
                }
                palm_reconstruction_valid = True
            except (RuntimeError, TypeError, ValueError):
                endpoint_activity = {
                    name: _hand_activity(np.full((len(frame), 3), np.nan))
                    for name, _ in selected_indices
                }
        else:
            endpoint_activity = {
                name: _hand_activity(np.full((len(frame), 3), np.nan))
                for name, _ in selected_indices
            }
        videos = [
            raw_dataset / "videos" / f"chunk-{source_episode // 1000:03d}" / key / f"episode_{source_episode:06d}.mp4"
            for key in ("observation.images.gopro", "observation.images.level-cam")
        ]
        checks = {
            "rows_at_least_200": len(frame) >= 200,
            "timestamp_finite": bool(np.isfinite(timestamps).all()),
            "timestamp_strictly_increasing": bool(np.all(np.diff(timestamps) > 0)),
            "alignment_nonfail": alignment_quality != "fail",
            "both_videos_present_nonempty": all(
                path.is_file() and path.stat().st_size > 0 for path in videos
            ),
            "required_columns_present": not missing_columns,
            "required_arrays_finite": all(finite_checks.values()),
            (
                ("four_mcp_palm_reconstruction_valid")
            ): palm_reconstruction_valid,
        }
        for name, activity in endpoint_activity.items():
            checks[f"{name}_active_frames_at_least_20"] = (
                int(activity["active_frames"]) >= 20
            )
            checks[f"{name}_displacement_at_least_0.03m"] = (
                float(activity["max_displacement_m"]) >= 0.03
            )
        eligible = bool(all(checks.values()))
        activities = list(endpoint_activity.values())
        rank = (
            min(int(activity["active_frames"]) for activity in activities),
            sum(int(activity["active_frames"]) for activity in activities),
            min(float(activity["max_displacement_m"]) for activity in activities),
            sum(float(activity["path_length_m"]) for activity in activities),
            -source_episode,
        )
        candidates.append(
            {
                "compact_episode": compact_episode,
                "source_episode": source_episode,
                "rows": len(frame),
                "alignment_quality": alignment_quality,
                "videos": [str(path) for path in videos],
                "missing_columns": missing_columns,
                "finite_checks": finite_checks,
                "endpoint_parts": endpoint_parts,
                "endpoint_definition": contract.endpoint_definition,
                "eligibility_checks": checks,
                "ineligible_reasons": [name for name, passed in checks.items() if not passed],
                "endpoint_activity": endpoint_activity,
                # Preserve the historical hand-only selection JSON shape.
                **(
                    ({
                        "right": endpoint_activity["hand_right"],
                        **(
                            {"left": endpoint_activity["hand_left"]}
                            if "hand_left" in endpoint_activity
                            else {}
                        ),
                    })
                ),
                "eligible": eligible,
                "rank_key": list(rank),
            }
        )

    candidates.sort(key=lambda item: tuple(item["rank_key"]), reverse=True)
    return candidates


def _baseline_episode_record(baseline_dataset: Path, compact_episode: int) -> dict[str, Any]:
    records = load_jsonl(baseline_dataset / "meta/episodes.jsonl")
    by_index = {int(record["episode_index"]): record for record in records}
    if compact_episode not in by_index:
        raise ContractError(f"baseline episodes.jsonl missing {compact_episode}")
    return dict(by_index[compact_episode])


def _strict_ab_episode_paths(root: Path, episode: int) -> tuple[Path, Path]:
    strict_root = root / STRICT_AB_DIR
    return (
        strict_root / "baseline_ramp" / f"episode_{episode:06d}.parquet",
        strict_root / "initial_body_q" / f"episode_{episode:06d}.npz",
    )


def _baseline_episode_parquet(baseline_dataset: Path, compact_episode: int) -> Path:
    path = (
        baseline_dataset
        / "data"
        / f"chunk-{compact_episode // 1000:03d}"
        / f"episode_{compact_episode:06d}.parquet"
    )
    if not path.is_file():
        raise ContractError(f"strict A/B baseline parquet missing: {path}")
    return path


def _write_initial_body_q_reference(
    baseline_control: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not baseline_control.is_file():
        raise ContractError(
            f"strict A/B baseline control telemetry missing: {baseline_control}"
        )
    try:
        with np.load(baseline_control, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["measured_joint_names"])
            body_q = np.asarray(payload["body_q"], dtype=np.float64)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ContractError(
            f"cannot load strict A/B baseline first body pose: {baseline_control}: {exc}"
        ) from exc
    if names != tuple(MUJOCO_BODY_JOINT_NAMES):
        raise ContractError("strict A/B baseline body_q is not in MuJoCo joint order")
    if body_q.ndim != 2 or body_q.shape[1] != len(names) or not len(body_q):
        raise ContractError(f"strict A/B baseline body_q shape is invalid: {body_q.shape}")
    reference = body_q[0]
    if not np.isfinite(reference).all():
        raise ContractError("strict A/B baseline first body_q is non-finite")
    source_sha = sha256_file(baseline_control)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.savez_compressed(
                file,
                contract=np.asarray(INITIAL_BODY_Q_REFERENCE_CONTRACT),
                joint_names=np.asarray(MUJOCO_BODY_JOINT_NAMES, dtype=np.str_),
                body_q_reference=reference.astype(np.float32),
                baseline_control_sha256=np.asarray(source_sha),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "contract": INITIAL_BODY_Q_REFERENCE_CONTRACT,
        "baseline_control": str(baseline_control.resolve()),
        "baseline_control_sha256": source_sha,
        "initial_body_q_reference_sha256": sha256_file(output_path),
    }


def build_isolated_staging(
    candidate_parquet: Path,
    baseline_dataset: Path,
    compact_episode: int,
    staging: Path,
    *,
    baseline_control: Path | None = None,
    strict_ab: bool = False,
) -> Path:
    if staging.exists():
        raise ContractError(f"refusing to overwrite staging directory: {staging}")
    data_dir = staging / "data/chunk-000"
    meta_dir = staging / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    frame = pd.read_parquet(candidate_parquet)
    rows = len(frame)
    frame["episode_index"] = np.zeros(rows, dtype=np.int64)
    frame["frame_index"] = np.arange(rows, dtype=np.int64)
    frame["index"] = np.arange(rows, dtype=np.int64)
    staging_parquet = data_dir / "episode_000000.parquet"
    frame.to_parquet(staging_parquet, index=False)

    info = load_json(baseline_dataset / "meta/info.json")
    info.update(
        {
            "total_episodes": 1,
            "total_frames": rows,
            "total_videos": 0,
            "splits": {"train": "0:1"},
        }
    )
    atomic_write_json(meta_dir / "info.json", info)
    record = _baseline_episode_record(baseline_dataset, compact_episode)
    record["episode_index"] = 0
    record["length"] = rows
    atomic_write_jsonl(meta_dir / "episodes.jsonl", [record])
    for name in ("tasks.jsonl", "modality.json"):
        source = baseline_dataset / "meta" / name
        if source.is_file():
            shutil.copy2(source, meta_dir / name)
    if strict_ab:
        if baseline_control is None:
            raise ContractError("strict A/B staging requires baseline control telemetry")
        baseline_source = _baseline_episode_parquet(
            baseline_dataset, compact_episode
        )
        _validate_strict_ab_first_pose_wire(baseline_source, staging_parquet)
        baseline_destination, reference_destination = _strict_ab_episode_paths(
            staging, 0
        )
        _copy_or_link(baseline_source, baseline_destination)
        reference = _write_initial_body_q_reference(
            baseline_control, reference_destination
        )
        atomic_write_json(
            staging / STRICT_AB_DIR / "manifest.json",
            {
                "format_version": 1,
                "contract": STRICT_AB_INITIALIZATION_CONTRACT,
                "items": [
                    {
                        "episode_index": 0,
                        "compact_episode": compact_episode,
                        "baseline_ramp_parquet": str(
                            baseline_destination.resolve()
                        ),
                        "baseline_ramp_sha256": sha256_file(
                            baseline_destination
                        ),
                        "initial_body_q_reference": str(
                            reference_destination.resolve()
                        ),
                        **reference,
                    }
                ],
            },
        )
    return staging_parquet


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_parallel_staging(
    items: list[ParallelReplayItem],
    baseline_dataset: Path,
    staging: Path,
) -> list[dict[str, Any]]:
    """Assemble isolated round inputs into one batch-only LeRobot dataset.

    The episode parquet bytes are preserved verbatim.  This matters because the
    rollout contract hashes the exact source file, and lets each canonical
    batch result be split back into the pre-existing isolated round without
    weakening that identity check.  The internal ``episode_index`` column is
    therefore still zero in every file; this staging directory is explicitly
    replay-only and is never promoted as a LeRobot training dataset.
    """

    if not items:
        raise ContractError("parallel staging requires at least one item")
    if staging.exists():
        raise ContractError(f"refusing to overwrite parallel staging: {staging}")
    expected = list(range(len(items)))
    actual = [item.batch_episode for item in items]
    if actual != expected:
        raise ContractError(
            f"parallel batch episode IDs must be contiguous: {actual} != {expected}"
        )

    data_dir = staging / "data/chunk-000"
    meta_dir = staging / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    total_frames = 0
    schema = None
    strict_items = 0
    for item in items:
        if not item.staging_parquet.is_file():
            raise ContractError(f"parallel source parquet missing: {item.staging_parquet}")
        parquet_file = pq.ParquetFile(item.staging_parquet)
        rows = int(parquet_file.metadata.num_rows)
        if rows <= 0:
            raise ContractError(f"parallel source parquet is empty: {item.staging_parquet}")
        if schema is None:
            schema = parquet_file.schema_arrow
        elif not parquet_file.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(
                f"parallel source schema mismatch: {item.staging_parquet}"
            )
        destination = data_dir / f"episode_{item.batch_episode:06d}.parquet"
        _copy_or_link(item.staging_parquet, destination)
        if sha256_file(destination) != sha256_file(item.staging_parquet):
            raise ContractError(f"parallel staging changed parquet bytes: {destination}")

        isolated_root = item.staging_parquet.parents[2]
        isolated_baseline, isolated_reference = _strict_ab_episode_paths(
            isolated_root, 0
        )
        isolated_manifest = isolated_root / STRICT_AB_DIR / "manifest.json"
        strict_presence = (
            isolated_baseline.is_file(),
            isolated_reference.is_file(),
            isolated_manifest.is_file(),
        )
        strict_record: dict[str, Any] = {}
        if any(strict_presence) and not all(strict_presence):
            raise ContractError(
                f"isolated strict A/B staging is incomplete: {isolated_root}"
            )
        if all(strict_presence):
            strict_items += 1
            batch_baseline, batch_reference = _strict_ab_episode_paths(
                staging, item.batch_episode
            )
            _copy_or_link(isolated_baseline, batch_baseline)
            _copy_or_link(isolated_reference, batch_reference)
            strict_record = {
                "baseline_ramp_sha256": sha256_file(batch_baseline),
                "initial_body_q_reference_sha256": sha256_file(batch_reference),
            }
            isolated_strict = load_json(isolated_manifest)
            isolated_items = isolated_strict.get("items")
            if (
                isolated_strict.get("contract")
                != STRICT_AB_INITIALIZATION_CONTRACT
                or not isinstance(isolated_items, list)
                or len(isolated_items) != 1
            ):
                raise ContractError(
                    f"isolated strict A/B manifest is invalid: {isolated_manifest}"
                )
            baseline_control_sha256 = isolated_items[0].get(
                "baseline_control_sha256"
            )
            if not isinstance(baseline_control_sha256, str):
                raise ContractError(
                    "isolated strict A/B manifest lacks baseline control hash"
                )
            strict_record["baseline_control_sha256"] = baseline_control_sha256

        record = _baseline_episode_record(baseline_dataset, item.compact_episode)
        record["episode_index"] = item.batch_episode
        record["length"] = rows
        records.append(record)
        total_frames += rows
        mappings.append(
            {
                "batch_episode": item.batch_episode,
                "compact_episode": item.compact_episode,
                "source_episode": item.source_episode,
                "round": item.round_index,
                "isolated_staging_parquet": str(item.staging_parquet.resolve()),
                "batch_staging_parquet": str(destination.resolve()),
                "rows": rows,
                "source_sha256": sha256_file(item.staging_parquet),
                **strict_record,
            }
        )

    if strict_items not in (0, len(items)):
        raise ContractError(
            "parallel replay cannot mix strict-A/B and ordinary staging episodes"
        )

    info = load_json(baseline_dataset / "meta/info.json")
    info.update(
        {
            "total_episodes": len(items),
            "total_frames": total_frames,
            "total_videos": 0,
            "total_chunks": 1,
            "splits": {"train": f"0:{len(items)}"},
        }
    )
    atomic_write_json(meta_dir / "info.json", info)
    atomic_write_jsonl(meta_dir / "episodes.jsonl", records)
    for name in ("tasks.jsonl", "modality.json"):
        source = baseline_dataset / "meta" / name
        if source.is_file():
            shutil.copy2(source, meta_dir / name)
    atomic_write_json(
        staging / "parallel_staging.json",
        {
            "format_version": 1,
            "contract": "batch-only byte-preserving isolated endpoint staging",
            "created_at": utc_now(),
            "items": mappings,
        },
    )
    if strict_items:
        atomic_write_json(
            staging / STRICT_AB_DIR / "manifest.json",
            {
                "format_version": 1,
                "contract": STRICT_AB_INITIALIZATION_CONTRACT,
                "items": [
                    {
                        "episode_index": record["batch_episode"],
                        "compact_episode": record["compact_episode"],
                        "baseline_ramp_sha256": record[
                            "baseline_ramp_sha256"
                        ],
                        "initial_body_q_reference_sha256": record[
                            "initial_body_q_reference_sha256"
                        ],
                        "baseline_control_sha256": record[
                            "baseline_control_sha256"
                        ],
                    }
                    for record in mappings
                ],
            },
        )
    return mappings


def validate_candidate_bundle(
    round_dir: Path,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    candidate_path = round_dir / "candidate_smpl.npz"
    metrics_path = round_dir / "ik_metrics.json"
    if not candidate_path.is_file() or not metrics_path.is_file():
        raise ContractError(f"candidate bundle is incomplete: {round_dir}")
    diagnostic_only = False
    try:
        with np.load(candidate_path, allow_pickle=False) as archive:
            contract = validate_endpoint_candidate_archive(archive)
            observed_head_height_m = float(
                np.asarray(archive["canonical_target_head_height_m"]).item()
            )
            if "diagnostic_only" in archive.files:
                diagnostic_only = bool(
                    np.asarray(archive["diagnostic_only"]).item()
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"candidate does not satisfy the current endpoint/wrist-lock contract: "
            f"{candidate_path}: {exc}"
        ) from exc
    metrics = load_json(metrics_path)
    if diagnostic_only or metrics.get("diagnostic_support_weight_shift"):
        raise ContractError(
            "diagnostic support-weight-shift candidates are visualization/"
            f"simulation artifacts and cannot enter promotion: {round_dir}"
        )
    wrist = metrics.get("optimization", {}).get("pico_wrist_local_rotation", {})
    false_grasp_repair = metrics.get("pico_false_grasp_wrist_repair", {})
    wrist_motion_filter = metrics.get("pico_wrist_motion_filter", {})
    expected = endpoint_retarget_contract(endpoint_parts, hand_endpoint_side)
    expected_wrist_policy = (
        (WRIST_LOCAL_ROTATION_POLICY)
    )
    observed_parts = contract.get("endpoint_parts", ENDPOINT_PARTS_HANDS)
    if (
        observed_parts != endpoint_parts
        or contract.get("hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH)
        != hand_endpoint_side
        or not np.isclose(
            observed_head_height_m,
            canonical_head_height_m,
            rtol=0.0,
            atol=1e-9,
        )
        or metrics.get("algorithm") != expected.algorithm
        or metrics.get("target_contract") != expected.endpoint_target_contract
        or metrics.get("candidate_contract", {}).get("endpoint_definition")
        != expected.endpoint_definition
        or wrist.get("policy") != expected_wrist_policy
        or float(wrist.get("max_abs_delta_rad", float("inf"))) > 1e-7
        or (
            True
            and false_grasp_repair.get("policy")
            != PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        )
        or (
            True
            and wrist_motion_filter.get("policy")
            != PICO_WRIST_MOTION_FILTER_POLICY
        )
    ):
        raise ContractError(
            f"candidate metrics do not prove the requested endpoint/wrist-lock contract: "
            f"{metrics_path}"
        )
    return contract


def build_candidate(
    paths: EpisodePaths,
    round_index: int,
    round_dir: Path,
    morphology: Path,
    previous_correction: Path | None,
    initial_candidate: Path | None,
    raw_selection_mode: str,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> None:
    command = [
        str(DATA_PYTHON),
        str(BUILDER),
        "--raw-parquet",
        str(paths.raw_parquet),
        "--baseline-parquet",
        str(paths.baseline_parquet),
        "--sync-dir",
        str(paths.raw_parquet.parents[2] / "sync"),
        "--source-episode",
        str(paths.source_episode),
        "--raw-selection-mode",
        raw_selection_mode,
        "--g1-morphology",
        str(morphology),
        "--canonical-head-height-m",
        str(canonical_head_height_m),
        "--output-parquet",
        str(round_dir / "candidate.parquet"),
        "--output-npz",
        str(round_dir / "candidate_smpl.npz"),
        "--metrics",
        str(round_dir / "ik_metrics.json"),
        "--endpoint-parts",
        endpoint_parts,
        "--hand-endpoint-side",
        hand_endpoint_side,
    ]
    if previous_correction is not None:
        command.extend(["--correction-npz", str(previous_correction)])
    if initial_candidate is not None:
        command.extend(["--initial-candidate-npz", str(initial_candidate)])
    run_checked(command)
    validate_candidate_bundle(
        round_dir, endpoint_parts, canonical_head_height_m, hand_endpoint_side
    )


def run_replay(staging: Path, replay_dir: Path) -> None:
    command = [
        str(REPLAY_LAUNCHER),
        "--preset",
        "sonic_sim",
        "--mode",
        "replay",
        "--hand-type",
        "brainco",
        "--replay-dataset",
        str(staging),
        "--replay-video-dir",
        str(replay_dir),
        "--replay-episodes",
        "all",
        "--replay-workers",
        "1",
    ]
    environment = os.environ.copy()
    environment["SONIC_BATCH_REPLAY_HEADLESS"] = "true"
    environment["SONIC_BATCH_REPLAY_LANE"] = "true"
    environment["SONIC_BATCH_REPLAY_LANE_ISOLATED"] = "true"
    environment["SONIC_BATCH_REPLAY_LANE_ID"] = "endpoint-retarget-single"
    environment["SONIC_REPLAY_WRITE_VIDEO"] = "false"
    run_checked(command, env=environment)


def run_parallel_replay(
    staging: Path,
    replay_dir: Path,
    workers: int,
) -> None:
    """Enter the canonical batch dispatcher as its parent, never as a lane."""

    if workers < 2 or workers > 4:
        raise ContractError("endpoint parallel replay requires 2..4 workers")
    validate_replay_workers(workers)
    command = [
        str(REPLAY_LAUNCHER),
        "--preset",
        "sonic_sim",
        "--mode",
        "replay",
        "--hand-type",
        "brainco",
        "--replay-dataset",
        str(staging),
        "--replay-video-dir",
        str(replay_dir),
        "--replay-episodes",
        "all",
        "--replay-workers",
        str(workers),
    ]
    environment = os.environ.copy()
    environment["SONIC_BATCH_REPLAY_HEADLESS"] = "true"
    environment["SONIC_REPLAY_WRITE_VIDEO"] = "false"
    # These markers belong only to children created by the canonical parent.
    # Inheriting one here would bypass the dispatcher and collide on ports.
    for name in (
        "SONIC_BATCH_REPLAY_LANE",
        "SONIC_BATCH_REPLAY_LANE_ISOLATED",
        "SONIC_BATCH_REPLAY_LANE_ID",
        "SONIC_REPLAY_WORKERS",
    ):
        environment.pop(name, None)
    run_checked(command, env=environment)


def _rewrite_parallel_record_value(
    value: Any,
    *,
    batch_replay: Path,
    isolated_replay: Path,
    batch_episode: int,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_parallel_record_value(
                item,
                batch_replay=batch_replay,
                isolated_replay=isolated_replay,
                batch_episode=batch_episode,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_parallel_record_value(
                item,
                batch_replay=batch_replay,
                isolated_replay=isolated_replay,
                batch_episode=batch_episode,
            )
            for item in value
        ]
    if isinstance(value, str):
        rewritten = value.replace(str(batch_replay), str(isolated_replay))
        return rewritten.replace(
            f"episode_{batch_episode:06d}", "episode_000000"
        )
    return value


def materialize_isolated_parallel_replay(
    *,
    batch_replay: Path,
    item: ParallelReplayItem,
) -> dict[str, Path]:
    """Split one validated canonical batch episode back into its round."""

    destination = item.round_dir / "replay"
    if destination.exists() and replay_has_any_artifact(destination):
        raise ContractError(
            f"refusing to mix parallel replay with existing artifacts: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    telemetry = destination / "telemetry"
    logs = destination / "logs"
    telemetry.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    source_stem = f"episode_{item.batch_episode:06d}"
    target_stem = "episode_000000"
    source_video = batch_replay / f"{source_stem}.mp4"
    if source_video.is_file():
        _copy_or_link(source_video, destination / f"{target_stem}.mp4")
    for suffix in ("control.npz", "sim.npz", "reference.npz"):
        _copy_or_link(
            batch_replay / "telemetry" / f"{source_stem}.{suffix}",
            telemetry / f"{target_stem}.{suffix}",
        )

    rollout_source = batch_replay / "telemetry" / f"{source_stem}.rollout.json"
    rollout = load_json(rollout_source)
    expected_hash = sha256_file(item.staging_parquet)
    if int(rollout.get("episode_index", -1)) != item.batch_episode:
        raise ContractError(f"parallel rollout episode mismatch: {rollout_source}")
    if rollout.get("source_sha256") != expected_hash:
        raise ContractError(f"parallel rollout source hash mismatch: {rollout_source}")
    rollout["episode_index"] = 0
    rollout["video_name"] = f"{target_stem}.mp4"
    rollout["parallel_parent"] = {
        "replay": str(batch_replay.resolve()),
        "batch_episode": item.batch_episode,
    }
    atomic_write_json(telemetry / f"{target_stem}.rollout.json", rollout)

    records = [
        record
        for record in load_jsonl(batch_replay / "manifest.jsonl")
        if int(record.get("episode_index", -1)) == item.batch_episode
    ]
    if not records or records[-1].get("status") != "success":
        raise ContractError(
            f"parallel manifest has no terminal success for batch episode {item.batch_episode}"
        )
    isolated_records: list[dict[str, Any]] = []
    for record in records:
        rewritten = _rewrite_parallel_record_value(
            record,
            batch_replay=batch_replay,
            isolated_replay=destination,
            batch_episode=item.batch_episode,
        )
        rewritten["episode_index"] = 0
        rewritten["source"] = str(item.staging_parquet.resolve())
        rewritten["parallel_parent"] = {
            "replay": str(batch_replay.resolve()),
            "batch_episode": item.batch_episode,
            "compact_episode": item.compact_episode,
            "source_episode": item.source_episode,
        }
        if isinstance(rewritten.get("rollout_contract"), dict):
            rewritten["rollout_contract"]["path"] = str(
                (telemetry / f"{target_stem}.rollout.json").resolve()
            )
            rewritten["rollout_contract"]["source_sha256"] = expected_hash
        for key, suffix in (
            ("control_telemetry", "control.npz"),
            ("sim_telemetry", "sim.npz"),
            ("reference_telemetry", "reference.npz"),
        ):
            if isinstance(rewritten.get(key), dict):
                rewritten[key]["path"] = str(
                    (telemetry / f"{target_stem}.{suffix}").resolve()
                )
        if isinstance(rewritten.get("video"), dict):
            rewritten["video"]["path"] = str(
                (destination / f"{target_stem}.mp4").resolve()
            )

        raw_log = record.get("log")
        if isinstance(raw_log, str) and Path(raw_log).is_file():
            attempt = int(record.get("attempt", 0))
            local_log = logs / f"{target_stem}_attempt_{attempt}.log"
            _copy_or_link(Path(raw_log), local_log)
            rewritten["log"] = str(local_log.resolve())
        isolated_records.append(rewritten)
    atomic_write_jsonl(destination / "manifest.jsonl", isolated_records)

    summary = load_json(batch_replay / "summary.json")
    summary.update(
        {
            "dataset": str(item.staging_parquet.parents[2].resolve()),
            "output_dir": str(destination.resolve()),
            "selected_episodes": [0],
            "successful_episodes": [0],
            "failed_episodes": [],
            "skipped_episodes": [],
            "resolved_protocols": {"pose": 1},
            "parallel_parent": {
                "replay": str(batch_replay.resolve()),
                "batch_episode": item.batch_episode,
            },
        }
    )
    config = dict(summary.get("config", {}))
    config.update(
        {
            "dataset_root": str(item.staging_parquet.parents[2].resolve()),
            "output_dir": str(destination.resolve()),
            "episodes": "all",
            "parallel_source_episode": item.batch_episode,
        }
    )
    summary["config"] = config
    atomic_write_json(destination / "summary.json", summary)

    init_prefix = f"init_episode_{item.batch_episode:06d}_attempt_"
    for source in sorted(batch_replay.glob(f"{init_prefix}*.mp4")):
        target_name = source.name.replace(
            f"episode_{item.batch_episode:06d}", "episode_000000"
        )
        _copy_or_link(source, destination / target_name)
    for source in sorted(
        (batch_replay / "telemetry").glob(f"{init_prefix}*.sim.npz")
    ):
        target_name = source.name.replace(
            f"episode_{item.batch_episode:06d}", "episode_000000"
        )
        _copy_or_link(source, telemetry / target_name)
    return validate_replay(destination, item.staging_parquet)


def validate_parallel_replay(
    *,
    batch_replay: Path,
    items: list[ParallelReplayItem],
    workers: int,
) -> dict[str, Any]:
    """Prove completion, source identity and lane namespace separation."""

    expected = list(range(len(items)))
    summary = load_json(batch_replay / "summary.json")
    if summary.get("selected_episodes") != expected:
        raise ContractError("parallel replay selection does not match batch mapping")
    if sorted(summary.get("successful_episodes", [])) != expected:
        raise ContractError("parallel replay did not successfully complete every item")
    if summary.get("failed_episodes"):
        raise ContractError("parallel replay summary contains failed episodes")
    if summary.get("sim_episode_lifecycle_contract") != ROLLOUT_CONTRACT:
        raise ContractError("parallel replay lifecycle contract does not match the required runtime protocol")
    wrist = summary.get("pose_wrist_contract", {})
    if wrist.get("zero_fallback") is not False or wrist.get("version") != 1:
        raise ContractError("parallel replay did not use the real-wrist contract")

    parallel = load_json(batch_replay / "parallel_rollout.json")
    if int(parallel.get("workers", -1)) != min(workers, len(items)):
        raise ContractError("parallel replay worker count mismatch")
    shards = parallel.get("shards")
    if not isinstance(shards, list):
        raise ContractError("parallel replay is missing lane shards")
    flattened = sorted(int(index) for shard in shards for index in shard)
    if flattened != expected:
        raise ContractError("parallel lane shards are incomplete or overlapping")
    lane_ports = parallel.get("lane_ports")
    if not isinstance(lane_ports, list) or len(lane_ports) != min(workers, len(items)):
        raise ContractError("parallel replay lane port manifest is incomplete")
    flat_ports = [int(port) for record in lane_ports for port in record.values()]
    if len(flat_ports) != 5 * len(lane_ports) or len(set(flat_ports)) != len(flat_ports):
        raise ContractError("parallel replay lane ports overlap")

    records = load_jsonl(batch_replay / "manifest.jsonl")
    item_reports: list[dict[str, Any]] = []
    observed_lanes: list[int] = []
    for item in items:
        matching = [
            record
            for record in records
            if int(record.get("episode_index", -1)) == item.batch_episode
        ]
        if not matching or matching[-1].get("status") != "success":
            raise ContractError(
                f"parallel manifest is incomplete for batch episode {item.batch_episode}"
            )
        lane = int(matching[-1].get("parallel_lane", -1))
        if lane < 0 or lane >= len(lane_ports):
            raise ContractError(f"invalid parallel lane ID for batch episode {item.batch_episode}")
        observed_lanes.append(lane)
        source_hash = sha256_file(item.staging_parquet)
        batch_parquet = (
            batch_replay.parent
            / "staging/data/chunk-000"
            / f"episode_{item.batch_episode:06d}.parquet"
        )
        if not batch_parquet.is_file() or sha256_file(batch_parquet) != source_hash:
            raise ContractError(
                f"parallel batch source identity mismatch for {item.batch_episode}"
            )
        rollout_path = (
            batch_replay
            / "telemetry"
            / f"episode_{item.batch_episode:06d}.rollout.json"
        )
        rollout = load_json(rollout_path)
        rows = int(pq.ParquetFile(item.staging_parquet).metadata.num_rows)
        if (
            int(rollout.get("episode_index", -1)) != item.batch_episode
            or rollout.get("source_sha256") != source_hash
            or int(rollout.get("source_rows", -1)) != rows
            or int(rollout.get("replay_rows", -1)) != rows
        ):
            raise ContractError(
                f"parallel rollout contract mismatch for batch episode {item.batch_episode}"
            )
        control_path = (
            batch_replay
            / "telemetry"
            / f"episode_{item.batch_episode:06d}.control.npz"
        )
        try:
            with np.load(control_path, allow_pickle=False) as payload:
                control_rows = int(np.asarray(payload["replay_frame_index"]).shape[0])
        except (OSError, KeyError, ValueError) as exc:
            raise ContractError(f"cannot validate parallel control {control_path}: {exc}") from exc
        if control_rows != rows:
            raise ContractError(
                f"parallel control row mismatch: batch={item.batch_episode}, "
                f"control={control_rows}, source={rows}"
            )
        item_reports.append(
            {
                "batch_episode": item.batch_episode,
                "compact_episode": item.compact_episode,
                "source_episode": item.source_episode,
                "round": item.round_index,
                "parallel_lane": lane,
                "rows": rows,
                "source_sha256": source_hash,
                "rollout_contract": str(rollout_path.resolve()),
            }
        )
    if len(items) == workers and sorted(observed_lanes) != list(range(workers)):
        raise ContractError(
            f"four-item capacity probe did not exercise every lane: {observed_lanes}"
        )
    return {
        "format_version": 1,
        "status": "pass",
        "validated_at": utc_now(),
        "batch_replay": str(batch_replay.resolve()),
        "workers": len(lane_ports),
        "lane_ports": lane_ports,
        "shards": shards,
        "observed_lanes": observed_lanes,
        "items": item_reports,
    }


def run_parallel_round(
    *,
    output_root: Path,
    baseline_dataset: Path,
    items: list[ParallelReplayItem],
    workers: int,
) -> dict[str, Any]:
    if len(items) < 2:
        raise ContractError("parallel round requires at least two pending episodes")
    fingerprint = hashlib.sha256()
    for item in items:
        fingerprint.update(
            f"{item.compact_episode}:{item.source_episode}:{item.round_index}:".encode()
        )
        fingerprint.update(sha256_file(item.staging_parquet).encode())
        isolated_baseline, isolated_reference = _strict_ab_episode_paths(
            item.staging_parquet.parents[2], 0
        )
        for companion in (isolated_baseline, isolated_reference):
            if companion.is_file():
                fingerprint.update(sha256_file(companion).encode())
    batch_root = (
        output_root
        / "_parallel_batches"
        / f"round_{items[0].round_index:02d}_{fingerprint.hexdigest()[:16]}"
    )
    if any(item.round_index != items[0].round_index for item in items):
        raise ContractError("parallel round cannot mix optimizer rounds")
    staging = batch_root / "staging"
    replay = batch_root / "replay"
    if staging.is_dir():
        payload = load_json(staging / "parallel_staging.json")
        expected = [sha256_file(item.staging_parquet) for item in items]
        observed = [str(record.get("source_sha256")) for record in payload.get("items", [])]
        if observed != expected:
            raise ContractError(f"cached parallel staging does not match request: {staging}")
        for item, record in zip(items, payload.get("items", []), strict=True):
            isolated_baseline, isolated_reference = _strict_ab_episode_paths(
                item.staging_parquet.parents[2], 0
            )
            if isolated_baseline.is_file():
                if record.get("baseline_ramp_sha256") != sha256_file(
                    isolated_baseline
                ) or record.get(
                    "initial_body_q_reference_sha256"
                ) != sha256_file(
                    isolated_reference
                ):
                    raise ContractError(
                        f"cached strict A/B staging does not match request: {staging}"
                    )
    else:
        batch_root.mkdir(parents=True, exist_ok=True)
        build_parallel_staging(items, baseline_dataset, staging)

    if (replay / "summary.json").is_file():
        report = validate_parallel_replay(
            batch_replay=replay,
            items=items,
            workers=workers,
        )
    elif replay_has_any_artifact(replay):
        raise ContractError(
            f"parallel replay has artifacts without a completion summary: {replay}"
        )
    else:
        run_parallel_replay(staging, replay, min(workers, len(items)))
        report = validate_parallel_replay(
            batch_replay=replay,
            items=items,
            workers=workers,
        )

    for item in items:
        local_replay = item.round_dir / "replay"
        if (local_replay / "summary.json").is_file():
            validate_replay(local_replay, item.staging_parquet)
        elif replay_has_any_artifact(local_replay):
            raise ContractError(
                f"isolated replay is partial and cannot be replaced: {local_replay}"
            )
        else:
            materialize_isolated_parallel_replay(batch_replay=replay, item=item)
    atomic_write_json(batch_root / "validation.json", report)
    return report


def validate_replay(replay_dir: Path, staging_parquet: Path) -> dict[str, Path]:
    summary = load_json(replay_dir / "summary.json")
    if summary.get("selected_episodes") != [0]:
        raise ContractError("replay summary must select isolated episode 0")
    if summary.get("successful_episodes") != [0] or summary.get("failed_episodes"):
        raise ContractError("isolated replay did not finish successfully")
    if summary.get("sim_episode_lifecycle_contract") != ROLLOUT_CONTRACT:
        raise ContractError("replay lifecycle contract does not match the required runtime protocol")
    wrist = summary.get("pose_wrist_contract", {})
    if wrist.get("zero_fallback") is not False or wrist.get("version") != 1:
        raise ContractError("replay did not use the real-wrist pose contract")

    records = load_jsonl(replay_dir / "manifest.jsonl")
    records = [record for record in records if int(record.get("episode_index", -1)) == 0]
    if not records or records[-1].get("status") != "success":
        raise ContractError("replay manifest has no terminal success for episode 0")
    rollout = records[-1].get("rollout_contract", {})
    if rollout.get("source_sha256") != sha256_file(staging_parquet):
        raise ContractError("replay source hash does not match isolated staging parquet")
    telemetry = replay_dir / "telemetry"
    paths = {
        "control": telemetry / "episode_000000.control.npz",
        "sim": telemetry / "episode_000000.sim.npz",
        "reference": telemetry / "episode_000000.reference.npz",
        "rollout": telemetry / "episode_000000.rollout.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ContractError(f"replay telemetry missing: {missing}")
    rollout_payload = load_json(paths["rollout"])
    expected_rows = int(pq.ParquetFile(staging_parquet).metadata.num_rows)
    if (
        int(rollout_payload.get("episode_index", -1)) != 0
        or rollout_payload.get("source_sha256") != sha256_file(staging_parquet)
        or int(rollout_payload.get("source_rows", -1)) != expected_rows
        or int(rollout_payload.get("replay_rows", -1)) != expected_rows
        or rollout_payload.get("video_name") != "episode_000000.mp4"
    ):
        raise ContractError("isolated rollout JSON does not match staging identity")
    return paths


def evaluate_round(
    *,
    round_index: int,
    round_dir: Path,
    baseline_control: Path,
    previous_evaluation: Path | None,
    next_correction: Path | None,
    gate_budget_limit_improvement_only: bool,
) -> dict[str, Any]:
    telemetry = round_dir / "replay/telemetry"
    command = [
        str(SIM_PYTHON),
        str(EVALUATOR),
        "--round-index",
        str(round_index),
        "--candidate-npz",
        str(round_dir / "candidate_smpl.npz"),
        "--control-npz",
        str(telemetry / "episode_000000.control.npz"),
        "--baseline-control-npz",
        str(baseline_control),
        "--sim-npz",
        str(telemetry / "episode_000000.sim.npz"),
        "--robot-xml",
        str(ROBOT_XML),
        "--output-npz",
        str(round_dir / "evaluation.npz"),
        "--metrics",
        str(round_dir / "replay_metrics.json"),
    ]
    if gate_budget_limit_improvement_only:
        command.append("--gate-budget-limit-improvement-only")
    if previous_evaluation is not None:
        command.extend(["--previous-evaluation-npz", str(previous_evaluation)])
    if next_correction is not None:
        command.extend(["--next-correction-npz", str(next_correction)])
    run_checked(command)
    metrics = load_json(round_dir / "replay_metrics.json")
    gate = metrics.get("closed_loop_gate")
    if not isinstance(gate, dict):
        raise ContractError("replay_metrics.json is missing closed_loop_gate")
    decision = gate.get("decision")
    if decision not in VALID_DECISIONS:
        raise ContractError(f"unrecognized closed-loop decision: {decision!r}")
    return metrics


def audit_round_production(round_dir: Path) -> dict[str, Any]:
    command = [
        str(SIM_PYTHON),
        str(PRODUCTION_AUDITOR),
        "--candidate-npz",
        str(round_dir / "candidate_smpl.npz"),
        "--control-npz",
        str(round_dir / "replay/telemetry/episode_000000.control.npz"),
        "--sim-npz",
        str(round_dir / "replay/telemetry/episode_000000.sim.npz"),
        "--robot-xml",
        str(ROBOT_XML),
        "--output-json",
        str(round_dir / "production_audit.json"),
    ]
    run_checked(command)
    payload = load_json(round_dir / "production_audit.json")
    if payload.get("production_accepted") is not False:
        raise ContractError("uncalibrated smoke production audit must remain fail closed")
    return payload


def validate_decision_transition(round_index: int, decision: str) -> str:
    if decision not in VALID_DECISIONS:
        raise ContractError(f"unrecognized decision: {decision!r}")
    if decision == "continue":
        if round_index >= MAX_ROUNDS - 1:
            raise ContractError("round 02 cannot request a fourth replay")
        return "continue"
    if decision == "rejected_unstable":
        return "rejected"
    if decision == "accepted_early_stop":
        if round_index >= MAX_ROUNDS - 1:
            raise ContractError("round 02 must use accepted_at_budget_limit")
        return "accepted"
    if decision == "accepted_at_budget_limit":
        if round_index != MAX_ROUNDS - 1:
            raise ContractError("accepted_at_budget_limit is only valid at round 02")
        return "accepted"
    if decision == "replay_budget_exhausted":
        if round_index != MAX_ROUNDS - 1:
            raise ContractError("replay_budget_exhausted is only valid at round 02")
        return "rejected"
    raise AssertionError(decision)


def artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise ContractError(f"required artifact missing: {path}")
        result[str(path.resolve())] = sha256_file(path)
    return result


def write_round_contract(
    round_index: int,
    round_dir: Path,
    decision: str,
    staging_parquet: Path,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> None:
    paths = [
        round_dir / "candidate.parquet",
        round_dir / "candidate_smpl.npz",
        round_dir / "ik_metrics.json",
        staging_parquet,
        round_dir / "replay/summary.json",
        round_dir / "replay/manifest.jsonl",
        round_dir / "replay/telemetry/episode_000000.control.npz",
        round_dir / "replay/telemetry/episode_000000.sim.npz",
        round_dir / "replay/telemetry/episode_000000.reference.npz",
        round_dir / "replay/telemetry/episode_000000.rollout.json",
        round_dir / "evaluation.npz",
        round_dir / "replay_metrics.json",
        round_dir / "production_audit.json",
    ]
    atomic_write_json(
        round_dir / "round_contract.json",
        {
            "format_version": 1,
            **endpoint_contract_identity(
                endpoint_parts, canonical_head_height_m, hand_endpoint_side
            ),
            "round": round_index,
            "decision": decision,
            "gate_budget_limit_improvement_only": bool(
                gate_budget_limit_improvement_only
            ),
            "created_at": utc_now(),
            "artifacts": artifact_hashes(paths),
        },
    )


def validate_round_contract(
    path: Path,
    *,
    expected_gate_budget_limit_improvement_only: bool | None = None,
    expected_endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    expected_canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    expected_hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    payload = load_json(path)
    expected_contract = endpoint_contract_identity(
        expected_endpoint_parts,
        expected_canonical_head_height_m,
        expected_hand_endpoint_side,
    )
    for key, expected in expected_contract.items():
        if payload.get(key) != expected:
            raise ContractError(
                f"cached round contract mismatch for {key} in {path}: "
                f"observed={payload.get(key)!r}, expected={expected!r}"
            )
    decision = payload.get("decision")
    if decision not in VALID_DECISIONS:
        raise ContractError(f"invalid cached decision in {path}: {decision!r}")
    if expected_gate_budget_limit_improvement_only is not None:
        observed_gate = bool(payload.get("gate_budget_limit_improvement_only", False))
        if not gate_profile_is_safe_resume_upgrade(
            observed=observed_gate,
            requested=expected_gate_budget_limit_improvement_only,
            round_index=int(payload.get("round", -1)),
            decision=str(decision),
        ):
            raise ContractError(
                "cached round gate profile does not match this run: "
                f"cached={observed_gate}, requested="
                f"{expected_gate_budget_limit_improvement_only}"
            )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ContractError(f"cached contract has no artifacts: {path}")
    for raw_path, expected in artifacts.items():
        artifact = Path(raw_path)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ContractError(f"cached artifact hash mismatch: {artifact}")
    validate_candidate_bundle(
        path.parent,
        expected_endpoint_parts,
        expected_canonical_head_height_m,
        expected_hand_endpoint_side,
    )
    return payload


def gate_profile_is_safe_resume_upgrade(
    *,
    observed: bool,
    requested: bool,
    round_index: int,
    decision: str,
) -> bool:
    """Allow only a proven strict result to enter an improvement-enabled run.

    The opt-in improvement profile changes behavior only at round 02.  Thus a
    strict round 00/01 contract is byte-for-byte stronger/equivalent, and a
    strict round-02 acceptance is stronger than the relaxed acceptance.  A
    relaxed cached result must never enter a strict run, and a strict rejected
    round 02 must be re-evaluated rather than silently promoted.
    """

    if observed == requested:
        return True
    if observed or not requested:
        return False
    if 0 <= round_index < MAX_ROUNDS - 1:
        return True
    return (
        round_index == MAX_ROUNDS - 1
        and decision == "accepted_at_budget_limit"
    )


def replay_has_any_artifact(replay_dir: Path) -> bool:
    """Return whether an incomplete replay left material output behind.

    An empty (or not-yet-created) replay directory is safe to restart. A
    non-empty directory without a valid closure stays fail-closed so resume
    never silently combines artifacts from two simulator attempts.
    """
    return replay_dir.is_dir() and any(path.is_file() for path in replay_dir.rglob("*"))


def process_episode(
    *,
    raw_dataset: Path,
    baseline_dataset: Path,
    output_root: Path,
    mapping: EpisodeMapping,
    compact_episode: int,
    morphology: Path,
    baseline_replay_root: Path,
    baseline_replay_indices: tuple[int, ...],
    resume: bool,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    paths = episode_paths(raw_dataset, baseline_dataset, mapping, compact_episode)
    episode_root = output_root / f"episode_{compact_episode:06d}"
    if episode_root.exists() and not resume:
        raise ContractError(f"episode output exists; pass --resume to inspect it: {episode_root}")
    episode_root.mkdir(parents=True, exist_ok=True)
    baseline_replay_episode = baseline_replay_indices[compact_episode]
    baseline_control = (
        baseline_replay_root
        / "telemetry"
        / f"episode_{baseline_replay_episode:06d}.control.npz"
    )
    if not baseline_control.is_file():
        raise ContractError(f"baseline control telemetry missing: {baseline_control}")

    rounds: list[dict[str, Any]] = []
    initial_candidate: Path | None = None
    previous_evaluation: Path | None = None
    previous_correction: Path | None = None
    accepted_round: int | None = None
    terminal_status = "optimization_fail"

    for round_index in range(MAX_ROUNDS):
        round_dir = episode_root / f"round_{round_index:02d}"
        round_contract_path = round_dir / "round_contract.json"
        if resume and round_contract_path.is_file():
            cached = validate_round_contract(
                round_contract_path,
                expected_gate_budget_limit_improvement_only=(
                    gate_budget_limit_improvement_only
                ),
                expected_endpoint_parts=endpoint_parts,
                expected_canonical_head_height_m=canonical_head_height_m,
                expected_hand_endpoint_side=hand_endpoint_side,
            )
            decision = str(cached["decision"])
            metrics = load_json(round_dir / "replay_metrics.json")
        else:
            if round_dir.exists():
                if not resume:
                    raise ContractError(
                        f"partial round exists without valid resume contract: {round_dir}"
                    )
                candidate_npz = round_dir / "candidate_smpl.npz"
                staging_parquet = round_dir / "staging/data/chunk-000/episode_000000.parquet"
                if not candidate_npz.is_file() or not staging_parquet.is_file():
                    raise ContractError(
                        f"partial round cannot be resumed before candidate/staging closure: {round_dir}"
                    )
                validate_candidate_bundle(
                    round_dir,
                    endpoint_parts,
                    canonical_head_height_m,
                    hand_endpoint_side,
                )
                replay_dir = round_dir / "replay"
                replay_summary = replay_dir / "summary.json"
                if replay_summary.is_file():
                    validate_replay(replay_dir, staging_parquet)
                elif replay_has_any_artifact(replay_dir):
                    raise ContractError(
                        "partial replay has artifacts but no completion summary; "
                        f"isolate or audit the failed attempt before retrying: {replay_dir}"
                    )
                else:
                    run_replay(round_dir / "staging", replay_dir)
                    validate_replay(replay_dir, staging_parquet)
                next_correction = (
                    episode_root / f"round_{round_index + 1:02d}_correction.npz"
                    if round_index < MAX_ROUNDS - 1
                    else None
                )
                audit_round_production(round_dir)
                metrics = evaluate_round(
                    round_index=round_index,
                    round_dir=round_dir,
                    baseline_control=baseline_control,
                    previous_evaluation=previous_evaluation,
                    next_correction=next_correction,
                    gate_budget_limit_improvement_only=(
                        gate_budget_limit_improvement_only
                    ),

                )
                decision = str(metrics["closed_loop_gate"]["decision"])
                write_round_contract(
                    round_index,
                    round_dir,
                    decision,
                    staging_parquet,
                    gate_budget_limit_improvement_only,
                    endpoint_parts,
                    canonical_head_height_m,
                    hand_endpoint_side,
                )
            else:
                round_dir.mkdir(parents=True)
                next_correction = (
                    episode_root / f"round_{round_index + 1:02d}_correction.npz"
                    if round_index < MAX_ROUNDS - 1
                    else None
                )
                build_candidate(
                    paths,
                    round_index,
                    round_dir,
                    morphology,
                    previous_correction,
                    initial_candidate,
                    mapping.raw_selection_mode,
                    endpoint_parts,
                    canonical_head_height_m,
                    hand_endpoint_side,
                )
                staging_parquet = build_isolated_staging(
                    round_dir / "candidate.parquet",
                    baseline_dataset,
                    compact_episode,
                    round_dir / "staging",
                    baseline_control=baseline_control,
                    strict_ab=(False),
                )
                run_replay(round_dir / "staging", round_dir / "replay")
                validate_replay(round_dir / "replay", staging_parquet)
                audit_round_production(round_dir)
                metrics = evaluate_round(
                    round_index=round_index,
                    round_dir=round_dir,
                    baseline_control=baseline_control,
                    previous_evaluation=previous_evaluation,
                    next_correction=next_correction,
                    gate_budget_limit_improvement_only=(
                        gate_budget_limit_improvement_only
                    ),

                )
                decision = str(metrics["closed_loop_gate"]["decision"])
                write_round_contract(
                    round_index,
                    round_dir,
                    decision,
                    staging_parquet,
                    gate_budget_limit_improvement_only,
                    endpoint_parts,
                    canonical_head_height_m,
                    hand_endpoint_side,
                )

        transition = validate_decision_transition(round_index, decision)
        rounds.append(
            {
                "round": round_index,
                "decision": decision,
                "round_contract": str(round_contract_path),
                "metrics": str(round_dir / "replay_metrics.json"),
            }
        )
        # Every correction archive is hash-bound to the candidate whose
        # measured replay produced it.  Advance the parent after every closed
        # round so round 2 consumes round 1 (rather than round 0) as its
        # initialization and lineage authority.
        initial_candidate = round_dir / "candidate_smpl.npz"
        previous_evaluation = round_dir / "evaluation.npz"

        if transition == "continue":
            previous_correction = episode_root / f"round_{round_index + 1:02d}_correction.npz"
            if not previous_correction.is_file():
                raise ContractError(
                    f"continue decision did not produce correction: {previous_correction}"
                )
            continue
        if transition == "accepted":
            accepted_round = round_index
            terminal_status = "optimized_pass"
        else:
            terminal_status = "optimization_fail"
        break
    else:
        raise ContractError("episode loop ended without a terminal decision")

    final = {
        "format_version": 1,
        **endpoint_contract_identity(
            endpoint_parts, canonical_head_height_m, hand_endpoint_side
        ),
        "dataset": baseline_dataset.name,
        "compact_episode": compact_episode,
        "source_episode": paths.source_episode,
        "status": terminal_status,
        "accepted_round": accepted_round,
        "maximum_replays": MAX_ROUNDS,
        "used_replays": len(rounds),
        "fourth_round_generated": (episode_root / "round_03").exists(),
        "gate_budget_limit_improvement_only": bool(
            gate_budget_limit_improvement_only
        ),
        "force_all_rounds": False,
        "rounds": rounds,
        "production_gate": {
            "status": "pending_dataset_smoke_calibration",
            "accepted": False,
            "note": "closed-loop optimized_pass is not train-ready production acceptance"
        },
        "updated_at": utc_now(),
    }
    atomic_write_json(episode_root / "final_decision.json", final)
    return final


def _final_decision_from_rounds(
    *,
    baseline_dataset: Path,
    paths: EpisodePaths,
    episode_root: Path,
    rounds: list[dict[str, Any]],
    transition: str,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    if not rounds or transition not in {"accepted", "rejected"}:
        raise ContractError("terminal final decision requires terminal round history")
    accepted = transition == "accepted"
    final = {
        "format_version": 1,
        **endpoint_contract_identity(
            endpoint_parts, canonical_head_height_m, hand_endpoint_side
        ),
        "dataset": baseline_dataset.name,
        "compact_episode": paths.compact_episode,
        "source_episode": paths.source_episode,
        "status": "optimized_pass" if accepted else "optimization_fail",
        "accepted_round": rounds[-1]["round"] if accepted else None,
        "maximum_replays": MAX_ROUNDS,
        "used_replays": len(rounds),
        "fourth_round_generated": (episode_root / "round_03").exists(),
        "gate_budget_limit_improvement_only": bool(
            gate_budget_limit_improvement_only
        ),
        "rounds": rounds,
        "production_gate": {
            "status": "pending_dataset_smoke_calibration",
            "accepted": False,
            "note": "closed-loop optimized_pass is not train-ready production acceptance",
        },
        "updated_at": utc_now(),
    }
    if final["fourth_round_generated"]:
        raise ContractError(f"forbidden fourth round exists: {episode_root / 'round_03'}")
    atomic_write_json(episode_root / "final_decision.json", final)
    return final


def inspect_episode_progress(
    *,
    baseline_dataset: Path,
    paths: EpisodePaths,
    episode_root: Path,
    resume: bool,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> tuple[dict[str, Any] | None, int | None]:
    """Return a verified terminal decision or the next required round."""

    if episode_root.exists() and not resume:
        raise ContractError(
            f"episode output exists; pass --resume to inspect it: {episode_root}"
        )
    episode_root.mkdir(parents=True, exist_ok=True)
    if (episode_root / "round_03").exists():
        raise ContractError(f"forbidden fourth round exists: {episode_root / 'round_03'}")

    rounds: list[dict[str, Any]] = []
    terminal_transition: str | None = None
    next_round: int | None = 0
    for round_index in range(MAX_ROUNDS):
        round_dir = episode_root / f"round_{round_index:02d}"
        contract_path = round_dir / "round_contract.json"
        if not contract_path.is_file():
            next_round = round_index
            break
        cached = validate_round_contract(
            contract_path,
            expected_gate_budget_limit_improvement_only=(
                gate_budget_limit_improvement_only
            ),
            expected_endpoint_parts=endpoint_parts,
            expected_canonical_head_height_m=canonical_head_height_m,
            expected_hand_endpoint_side=hand_endpoint_side,
        )
        decision = str(cached["decision"])
        transition = validate_decision_transition(round_index, decision)
        rounds.append(
            {
                "round": round_index,
                "decision": decision,
                "round_contract": str(contract_path),
                "metrics": str(round_dir / "replay_metrics.json"),
            }
        )
        if transition != "continue":
            terminal_transition = transition
            next_round = None
            break
    else:
        raise ContractError("three cached rounds ended without a terminal decision")

    if next_round is not None:
        for later in range(next_round + 1, MAX_ROUNDS):
            if (episode_root / f"round_{later:02d}").exists():
                raise ContractError(
                    "episode has a non-contiguous future round: "
                    f"{episode_root / f'round_{later:02d}'}"
                )
        if (episode_root / "final_decision.json").exists():
            raise ContractError("non-terminal episode unexpectedly has final_decision.json")
        return None, next_round

    assert terminal_transition is not None
    final_path = episode_root / "final_decision.json"
    if final_path.is_file():
        final = load_json(final_path)
        expected_status = (
            "optimized_pass" if terminal_transition == "accepted" else "optimization_fail"
        )
        observed_gate = bool(
            final.get("gate_budget_limit_improvement_only", False)
        )
        gate_compatible = gate_profile_is_safe_resume_upgrade(
            observed=observed_gate,
            requested=gate_budget_limit_improvement_only,
            round_index=int(rounds[-1]["round"]),
            decision=str(rounds[-1]["decision"]),
        )
        identity_mismatch = any(
            final.get(key) != expected
            for key, expected in endpoint_contract_identity(
                endpoint_parts, canonical_head_height_m, hand_endpoint_side
            ).items()
        )
        if (
            identity_mismatch
            or int(final.get("compact_episode", -1)) != paths.compact_episode
            or int(final.get("source_episode", -1)) != paths.source_episode
            or final.get("status") != expected_status
            or int(final.get("used_replays", -1)) != len(rounds)
            or bool(final.get("fourth_round_generated", True))
            or not gate_compatible
        ):
            raise ContractError(f"cached final decision is inconsistent: {final_path}")
        if observed_gate != gate_budget_limit_improvement_only:
            return (
                _final_decision_from_rounds(
                    baseline_dataset=baseline_dataset,
                    paths=paths,
                    episode_root=episode_root,
                    rounds=rounds,
                    transition=terminal_transition,
                    gate_budget_limit_improvement_only=(
                        gate_budget_limit_improvement_only
                    ),
                    endpoint_parts=endpoint_parts,
                    canonical_head_height_m=canonical_head_height_m,
                    hand_endpoint_side=hand_endpoint_side,
                ),
                None,
            )
        return final, None
    return (
        _final_decision_from_rounds(
            baseline_dataset=baseline_dataset,
            paths=paths,
            episode_root=episode_root,
            rounds=rounds,
            transition=terminal_transition,
            gate_budget_limit_improvement_only=(
                gate_budget_limit_improvement_only
            ),
            endpoint_parts=endpoint_parts,
            canonical_head_height_m=canonical_head_height_m,
            hand_endpoint_side=hand_endpoint_side,
        ),
        None,
    )


def prepare_parallel_round(
    *,
    raw_dataset: Path,
    baseline_dataset: Path,
    mapping: EpisodeMapping,
    compact_episode: int,
    round_index: int,
    output_root: Path,
    morphology: Path,
    batch_episode: int,
    resume: bool,
    baseline_control: Path,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> ParallelReplayItem:
    paths = episode_paths(raw_dataset, baseline_dataset, mapping, compact_episode)
    episode_root = output_root / f"episode_{compact_episode:06d}"
    round_dir = episode_root / f"round_{round_index:02d}"
    staging_parquet = round_dir / "staging/data/chunk-000/episode_000000.parquet"
    if round_dir.exists():
        if not resume:
            raise ContractError(f"parallel round already exists: {round_dir}")
        if (round_dir / "round_contract.json").exists():
            raise ContractError(f"scheduler selected an already-closed round: {round_dir}")
        for required in (
            round_dir / "candidate.parquet",
            round_dir / "candidate_smpl.npz",
            round_dir / "ik_metrics.json",
            staging_parquet,
        ):
            if not required.is_file():
                raise ContractError(
                    f"partial parallel round cannot resume before candidate closure: {required}"
                )
        validate_candidate_bundle(
            round_dir, endpoint_parts, canonical_head_height_m, hand_endpoint_side
        )
        replay = round_dir / "replay"
        if (replay / "summary.json").is_file():
            validate_replay(replay, staging_parquet)
        elif replay_has_any_artifact(replay):
            raise ContractError(
                f"partial isolated replay cannot be combined with a new batch: {replay}"
            )
    else:
        round_dir.mkdir(parents=True)
        initial_candidate = (
            episode_root / f"round_{round_index - 1:02d}/candidate_smpl.npz"
            if round_index > 0
            else None
        )
        previous_correction = (
            episode_root / f"round_{round_index:02d}_correction.npz"
            if round_index > 0
            else None
        )
        if round_index > 0:
            for required in (initial_candidate, previous_correction):
                assert required is not None
                if not required.is_file():
                    raise ContractError(f"parallel next-round input missing: {required}")
        build_candidate(
            paths,
            round_index,
            round_dir,
            morphology,
            previous_correction,
            initial_candidate,
            mapping.raw_selection_mode,
            endpoint_parts,
            canonical_head_height_m,
            hand_endpoint_side,
        )
        staging_parquet = build_isolated_staging(
            round_dir / "candidate.parquet",
            baseline_dataset,
            compact_episode,
            round_dir / "staging",
            baseline_control=baseline_control,
            strict_ab=(False),
        )
    return ParallelReplayItem(
        batch_episode=batch_episode,
        compact_episode=compact_episode,
        source_episode=paths.source_episode,
        round_index=round_index,
        round_dir=round_dir,
        staging_parquet=staging_parquet,
    )


def evaluate_prepared_round(
    *,
    item: ParallelReplayItem,
    episode_root: Path,
    baseline_control: Path,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> tuple[dict[str, Any], str]:
    previous_evaluation = (
        episode_root / f"round_{item.round_index - 1:02d}/evaluation.npz"
        if item.round_index > 0
        else None
    )
    next_correction = (
        episode_root / f"round_{item.round_index + 1:02d}_correction.npz"
        if item.round_index < MAX_ROUNDS - 1
        else None
    )
    audit_round_production(item.round_dir)
    metrics = evaluate_round(
        round_index=item.round_index,
        round_dir=item.round_dir,
        baseline_control=baseline_control,
        previous_evaluation=previous_evaluation,
        next_correction=next_correction,
        gate_budget_limit_improvement_only=gate_budget_limit_improvement_only,

    )
    decision = str(metrics["closed_loop_gate"]["decision"])
    transition = validate_decision_transition(item.round_index, decision)
    write_round_contract(
        item.round_index,
        item.round_dir,
        decision,
        item.staging_parquet,
        gate_budget_limit_improvement_only,
        endpoint_parts,
        canonical_head_height_m,
        hand_endpoint_side,
    )
    return metrics, transition


def process_episodes_parallel(
    *,
    raw_dataset: Path,
    baseline_dataset: Path,
    output_root: Path,
    mapping: EpisodeMapping,
    compact_episodes: list[int],
    morphology: Path,
    baseline_replay_root: Path,
    baseline_replay_indices: tuple[int, ...],
    workers: int,
    resume: bool,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> list[dict[str, Any]]:
    """Run round-synchronous endpoint optimization through canonical lanes."""

    decisions: dict[int, dict[str, Any]] = {}
    while len(decisions) < len(compact_episodes):
        pending_by_round: dict[int, list[int]] = {}
        for compact_episode in compact_episodes:
            if compact_episode in decisions:
                continue
            paths = episode_paths(raw_dataset, baseline_dataset, mapping, compact_episode)
            final, next_round = inspect_episode_progress(
                baseline_dataset=baseline_dataset,
                paths=paths,
                episode_root=output_root / f"episode_{compact_episode:06d}",
                resume=resume,
                gate_budget_limit_improvement_only=(
                    gate_budget_limit_improvement_only
                ),
                endpoint_parts=endpoint_parts,
                canonical_head_height_m=canonical_head_height_m,
                hand_endpoint_side=hand_endpoint_side,
            )
            if final is not None:
                decisions[compact_episode] = final
            else:
                assert next_round is not None
                pending_by_round.setdefault(next_round, []).append(compact_episode)
        if len(decisions) == len(compact_episodes):
            break
        if not pending_by_round:
            raise ContractError("parallel scheduler made no progress")

        round_index = min(pending_by_round)
        compact_group = pending_by_round[round_index]
        print(
            f"[endpoint-{endpoint_contract_label(endpoint_parts)}] "
            f"parallel round {round_index:02d}: "
            f"episodes={len(compact_group)}, workers={min(workers, len(compact_group))}",
            flush=True,
        )
        prepared = [
            prepare_parallel_round(
                raw_dataset=raw_dataset,
                baseline_dataset=baseline_dataset,
                mapping=mapping,
                compact_episode=compact_episode,
                round_index=round_index,
                output_root=output_root,
                morphology=morphology,
                batch_episode=ordinal,
                resume=resume,
                baseline_control=baseline_control_path(
                    baseline_replay_root,
                    baseline_replay_indices,
                    compact_episode,
                ),
                endpoint_parts=endpoint_parts,
                canonical_head_height_m=canonical_head_height_m,
                hand_endpoint_side=hand_endpoint_side,
            )
            for ordinal, compact_episode in enumerate(compact_group)
        ]
        missing_replay = [
            item
            for item in prepared
            if not (item.round_dir / "replay/summary.json").is_file()
        ]
        # Batch IDs must be contiguous within the subset actually dispatched.
        missing_replay = [
            ParallelReplayItem(
                batch_episode=ordinal,
                compact_episode=item.compact_episode,
                source_episode=item.source_episode,
                round_index=item.round_index,
                round_dir=item.round_dir,
                staging_parquet=item.staging_parquet,
            )
            for ordinal, item in enumerate(missing_replay)
        ]
        if len(missing_replay) >= 2:
            run_parallel_round(
                output_root=output_root,
                baseline_dataset=baseline_dataset,
                items=missing_replay,
                workers=min(workers, len(missing_replay)),
            )
        elif len(missing_replay) == 1:
            item = missing_replay[0]
            run_replay(item.staging_parquet.parents[2], item.round_dir / "replay")
            validate_replay(item.round_dir / "replay", item.staging_parquet)

        for item in prepared:
            validate_replay(item.round_dir / "replay", item.staging_parquet)
            baseline_replay_episode = baseline_replay_indices[item.compact_episode]
            baseline_control = (
                baseline_replay_root
                / "telemetry"
                / f"episode_{baseline_replay_episode:06d}.control.npz"
            )
            if not baseline_control.is_file():
                raise ContractError(f"baseline control telemetry missing: {baseline_control}")
            _, transition = evaluate_prepared_round(
                item=item,
                episode_root=output_root / f"episode_{item.compact_episode:06d}",
                baseline_control=baseline_control,
                gate_budget_limit_improvement_only=(
                    gate_budget_limit_improvement_only
                ),
                endpoint_parts=endpoint_parts,
                canonical_head_height_m=canonical_head_height_m,
                hand_endpoint_side=hand_endpoint_side,
            )
            if transition != "continue":
                paths = episode_paths(
                    raw_dataset, baseline_dataset, mapping, item.compact_episode
                )
                final, next_round = inspect_episode_progress(
                    baseline_dataset=baseline_dataset,
                    paths=paths,
                    episode_root=output_root / f"episode_{item.compact_episode:06d}",
                    resume=True,
                    gate_budget_limit_improvement_only=(
                        gate_budget_limit_improvement_only
                    ),
                    endpoint_parts=endpoint_parts,
                    canonical_head_height_m=canonical_head_height_m,
                    hand_endpoint_side=hand_endpoint_side,
                )
                if final is None or next_round is not None:
                    raise ContractError("terminal parallel round did not close episode")
                decisions[item.compact_episode] = final
        update_dataset_summary(
            output_root,
            [decisions[index] for index in sorted(decisions)],
            endpoint_parts,
            canonical_head_height_m,
            hand_endpoint_side,
        )
    ordered = [decisions[index] for index in compact_episodes]
    update_dataset_summary(
        output_root,
        ordered,
        endpoint_parts,
        canonical_head_height_m,
        hand_endpoint_side,
    )
    return ordered


def run_four_lane_capacity_probe(
    *,
    raw_dataset: Path,
    baseline_dataset: Path,
    output_root: Path,
    mapping: EpisodeMapping,
    morphology: Path,
    selected_records: list[dict[str, Any]],
    resume: bool,
    baseline_replay_root: Path,
    baseline_replay_indices: tuple[int, ...],
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
) -> dict[str, Any]:
    if len(selected_records) != 4:
        raise ContractError("four-lane capacity probe requires exactly four episodes")
    items = [
        prepare_parallel_round(
            raw_dataset=raw_dataset,
            baseline_dataset=baseline_dataset,
            mapping=mapping,
            compact_episode=int(record["compact_episode"]),
            round_index=0,
            output_root=output_root,
            morphology=morphology,
            batch_episode=ordinal,
            resume=resume,
            baseline_control=baseline_control_path(
                baseline_replay_root,
                baseline_replay_indices,
                int(record["compact_episode"]),
            ),
            endpoint_parts=endpoint_parts,
            canonical_head_height_m=canonical_head_height_m,
        )
        for ordinal, record in enumerate(selected_records)
    ]
    report = run_parallel_round(
        output_root=output_root,
        baseline_dataset=baseline_dataset,
        items=items,
        workers=4,
    )
    payload = {
        "format_version": 1,
        "status": "pass",
        "contract": (
            "four non-smoke round-00 candidates completed concurrently through "
            "four isolated canonical Stage-B lanes; no optimization evaluation run"
        ),
        "validated_at": utc_now(),
        "dataset": baseline_dataset.name,
        **endpoint_contract_identity(endpoint_parts, canonical_head_height_m),
        "selection": selected_records,
        "parallel_validation": report,
        "full_batch_started": False,
    }
    atomic_write_json(output_root / "four_lane_capacity_validation.json", payload)
    return payload


def preflight(
    raw_dataset: Path,
    baseline_dataset: Path,
    output_root: Path,
    morphology: Path,
    baseline_contract: str,
    baseline_replay_root: Path | None,
    gate_budget_limit_improvement_only: bool,
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> tuple[EpisodeMapping, Path, tuple[int, ...], dict[str, Any]]:
    for path in (raw_dataset, baseline_dataset):
        if not path.is_dir():
            raise ContractError(f"dataset directory missing: {path}")
        load_json(path / "meta/info.json")
    for path in (
        DATA_PYTHON,
        SIM_PYTHON,
        REPLAY_LAUNCHER,
        BUILDER,
        EVALUATOR,
        PRODUCTION_AUDITOR,
        ROBOT_XML,
        morphology,
    ):
        if not path.is_file():
            raise ContractError(f"required executable/model/artifact missing: {path}")
    baseline_contract_report = validate_baseline_dataset_contract(
        baseline_dataset, baseline_contract
    )
    if baseline_contract == "minimal_ab":
        mapping, mapping_authority = derive_minimal_ab_mapping(
            raw_dataset, baseline_dataset
        )
    else:
        mapping = derive_mapping(raw_dataset, baseline_dataset)
        mapping_authority = {
            "contract": "unique alignment-cropped row-count mapping",
            "path": None,
            "sha256": None,
        }
    alignment_membership_audit = audit_mapping_against_alignment(raw_dataset, mapping)
    baseline_replay_root = (
        baseline_replay_root
        if baseline_replay_root is not None
        else DEFAULT_BASELINE_REPLAY_PARENT / baseline_dataset.name / "conversion_replay"
    )
    summary = load_json(baseline_replay_root / "summary.json")
    total = len(mapping.compact_to_source)
    successful_episodes = summary.get("successful_episodes")
    if (
        not isinstance(successful_episodes, list)
        or len(successful_episodes) != total
        or len(set(successful_episodes)) != total
        or summary.get("failed_episodes")
    ):
        raise ContractError("baseline replay summary does not cover the converted episode count")
    if summary.get("sim_episode_lifecycle_contract") != ROLLOUT_CONTRACT:
        raise ContractError("baseline replay lifecycle does not match the required runtime protocol")
    baseline_replay_indices, replay_mapping_manifest = derive_baseline_replay_indices(
        baseline_dataset,
        baseline_replay_root,
        [int(index) for index in successful_episodes],
        mapping.baseline_rows,
    )
    report = {
        "format_version": 1,
        "status": "pass",
        **endpoint_contract_identity(
            endpoint_parts, canonical_head_height_m, hand_endpoint_side
        ),
        "checked_at": utc_now(),
        "raw_dataset": str(raw_dataset),
        "baseline_dataset": str(baseline_dataset),
        "baseline_contract": baseline_contract_report,
        "output_root": str(output_root),
        "requested": total,
        "source_total": len(mapping.aligned_rows),
        "mapping_solution_count": mapping.solution_count,
        "mapping_authority": mapping_authority,
        "raw_selection_mode": mapping.raw_selection_mode,
        "compact_to_source": list(mapping.compact_to_source),
        "dropped_sources": list(mapping.dropped_sources),
        "alignment_membership_audit": alignment_membership_audit,
        "baseline_replay_root": str(baseline_replay_root),
        "baseline_compact_to_replay": list(baseline_replay_indices),
        "baseline_replay_mapping_manifest": replay_mapping_manifest,
        "gate_profile": {
            "round_00_01": "strict_endpoint_thresholds",
            "round_02": (
                "budget_limit_improvement_only"
                if gate_budget_limit_improvement_only
                else "strict_endpoint_thresholds"
            ),
            "budget_limit_improvement_only": bool(
                gate_budget_limit_improvement_only
            ),
        },
        "morphology": {"path": str(morphology), "sha256": sha256_file(morphology)},
        "code": {
            "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
            "evaluator": {"path": str(EVALUATOR), "sha256": sha256_file(EVALUATOR)},
            "production_auditor": {
                "path": str(PRODUCTION_AUDITOR),
                "sha256": sha256_file(PRODUCTION_AUDITOR),
            },
            "replay_launcher": {"path": str(REPLAY_LAUNCHER), "sha256": sha256_file(REPLAY_LAUNCHER)},
        },
    }
    return mapping, baseline_replay_root, baseline_replay_indices, report


def update_dataset_summary(
    output_root: Path,
    decisions: list[dict[str, Any]],
    endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    canonical_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> None:
    counts: dict[str, int] = {}
    for decision in decisions:
        status = str(decision["status"])
        counts[status] = counts.get(status, 0) + 1
    source_episode_views = publish_source_episode_views(output_root, decisions)
    atomic_write_json(
        output_root / "summary.json",
        {
            "format_version": 1,
            **endpoint_contract_identity(
                endpoint_parts, canonical_head_height_m, hand_endpoint_side
            ),
            "updated_at": utc_now(),
            "completed": len(decisions),
            "status_counts": counts,
            "episodes": decisions,
            "human_view": {
                "contract": (
                    "MuJoCo replays use immutable source episode IDs; "
                    "compact episode IDs are machine-internal"
                ),
                "root": str(output_root / SOURCE_EPISODE_VIEW_DIR),
                "episodes": source_episode_views,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--baseline-dataset", type=Path, required=True)
    parser.add_argument(
        "--baseline-contract",
        choices=sorted(BASELINE_CONTRACTS),
        default="validated_baseline",
        help=(
            "validated_baseline requires a completed captured-C++ Stage C dataset; "
            "minimal_ab accepts zero-history Stage A plus an explicit matching "
            "Stage-B replay root"
        ),
    )
    parser.add_argument(
        "--baseline-replay-root",
        type=Path,
        default=None,
        help=(
            "Matching unretargeted Stage-B replay. Required explicitly for "
            "minimal_ab; validated_baseline defaults to outputs/replay_velocity_quality."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--endpoint-parts",
        choices=ENDPOINT_PART_CHOICES,
        default=ENDPOINT_PARTS_HANDS,
        help=(
            "Bilateral palm alignment with bounded Pico frame-switch repair "
            "and elbow-local wrist denoising; EgoAlign uses arm-first gated spine assistance."
        ),
    )
    parser.add_argument(
        "--hand-endpoint-side",
        choices=HAND_ENDPOINT_SIDE_CHOICES,
        default=HAND_ENDPOINT_SIDE_BOTH,
        help=(
            "Both palm centers participate in geometric and replay-residual alignment."
        ),
    )
    parser.add_argument(
        "--canonical-head-height-m",
        type=float,
        default=DEFAULT_CANONICAL_HEAD_HEIGHT_M,
        help=(
            "Canonical SMPL head-joint height used to scale endpoint targets. "
            "The value is part of every cache identity."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "capacity", "full", "promote"),
        required=True,
    )
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gate-budget-limit-improvement-only",
        action="store_true",
        help=(
            "Relax only round 02 to require improvement over baseline. This is "
            "off by default and requires dataset-scoped user authorization."
        ),
    )
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--morphology", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write selection/plan artifacts without builder or replay.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_replay_workers(args.workers)
    if args.mode == "capacity" and args.workers != 4:
        raise ContractError("capacity mode requires --workers 4 and a compatible custom replay driver")
    if not np.isfinite(args.canonical_head_height_m) or args.canonical_head_height_m <= 0:
        raise ContractError("--canonical-head-height-m must be finite and positive")
    raw_dataset = args.raw_dataset.expanduser().resolve()
    baseline_dataset = args.baseline_dataset.expanduser().resolve()
    baseline_replay_root_arg = (
        args.baseline_replay_root.expanduser().resolve()
        if args.baseline_replay_root is not None
        else None
    )
    if args.baseline_contract == "minimal_ab" and baseline_replay_root_arg is None:
        raise ContractError(
            "--baseline-contract minimal_ab requires --baseline-replay-root"
        )
    output_root = args.output_root.expanduser().resolve()
    state_json = args.state_json.expanduser().resolve()
    morphology = (
        args.morphology.expanduser().resolve()
        if args.morphology
        else output_root.parent.parent / "g1_endpoint_morphology.json"
    )
    state = load_json(state_json)
    if state.get("status") not in {
        "running",
        "complete",
        "complete_with_rejections",
    }:
        raise ContractError(f"execution state is not resumable: {state_json}")

    mapping, baseline_replay_root, baseline_replay_indices, report = preflight(
        raw_dataset,
        baseline_dataset,
        output_root,
        morphology,
        args.baseline_contract,
        baseline_replay_root_arg,
        args.gate_budget_limit_improvement_only,
        args.endpoint_parts,
        args.canonical_head_height_m,
        args.hand_endpoint_side,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "preflight.json", report)
    if args.mode == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    ranking = rank_smoke_episodes(
        raw_dataset,
        baseline_dataset,
        mapping,
        args.endpoint_parts,
        args.canonical_head_height_m,
        args.hand_endpoint_side,
    )
    eligible = [record for record in ranking if record["eligible"]]
    selection_payload = {
        "format_version": 1,
        "generated_at": utc_now(),
        "dataset": baseline_dataset.name,
        **endpoint_contract_identity(
            args.endpoint_parts,
            args.canonical_head_height_m,
            args.hand_endpoint_side,
        ),
        "ranking_contract": [
            "minimum selected-endpoint active_frames desc",
            "sum selected-endpoint active_frames desc",
            "minimum selected-endpoint max_displacement desc",
            "sum selected-endpoint path_length desc",
            "source_episode asc",
        ],
        "eligible_count": len(eligible),
        "selected": eligible[0] if eligible else None,
        "candidates": ranking,
    }
    atomic_write_json(output_root / "smoke_selection.json", selection_payload)
    capacity_records: list[dict[str, Any]] = []
    if args.mode == "smoke":
        if not eligible:
            raise ContractError("no current-data episode satisfies the smoke eligibility contract")
        selected = [int(eligible[0]["compact_episode"])]
        if args.episodes.strip().lower() not in {"", "all", "*"}:
            requested = parse_episode_selection(args.episodes, len(mapping.compact_to_source))
            if requested != selected:
                raise ContractError(
                    f"smoke episode is deterministic ({selected}); --episodes requested {requested}"
                )
    elif args.mode == "capacity":
        if args.workers != 4:
            raise ContractError("capacity mode requires --workers 4")
        if len(eligible) < 5:
            raise ContractError(
                "capacity mode requires four eligible non-smoke episodes"
            )
        smoke_episode = int(eligible[0]["compact_episode"])
        capacity_records = [
            record
            for record in eligible
            if int(record["compact_episode"]) != smoke_episode
        ][:4]
        selected = [int(record["compact_episode"]) for record in capacity_records]
        if args.episodes.strip().lower() not in {"", "all", "*"}:
            requested = parse_episode_selection(
                args.episodes, len(mapping.compact_to_source)
            )
            if requested != selected:
                raise ContractError(
                    f"capacity selection is deterministic ({selected}); "
                    f"--episodes requested {requested}"
                )
    elif args.mode == "full":
        selected = parse_episode_selection(args.episodes, len(mapping.compact_to_source))
    else:
        raise ContractError(
            "promote is fail-closed until smoke-approved production gates are recorded"
        )

    plan = {
        "format_version": 1,
        **endpoint_contract_identity(
            args.endpoint_parts,
            args.canonical_head_height_m,
            args.hand_endpoint_side,
        ),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "workers": args.workers,
        "episodes": selected,
        "baseline_contract": args.baseline_contract,
        "baseline_replay_root": str(baseline_replay_root),
        "gate_budget_limit_improvement_only": bool(
            args.gate_budget_limit_improvement_only
        ),
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "run_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "capacity":
        result = run_four_lane_capacity_probe(
            raw_dataset=raw_dataset,
            baseline_dataset=baseline_dataset,
            output_root=output_root,
            mapping=mapping,
            morphology=morphology,
            selected_records=capacity_records,
            resume=args.resume,
            baseline_replay_root=baseline_replay_root,
            baseline_replay_indices=baseline_replay_indices,
            endpoint_parts=args.endpoint_parts,
            canonical_head_height_m=args.canonical_head_height_m,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # Same-episode rounds are inherently serial. Dataset-level parallelism is
    # enabled only after the smoke gate; current smoke always contains one row.
    if args.workers != 1 and args.mode == "smoke":
        raise ContractError("smoke mode requires --workers 1")
    if args.mode == "full" and args.workers > 1:
        process_episodes_parallel(
            raw_dataset=raw_dataset,
            baseline_dataset=baseline_dataset,
            output_root=output_root,
            mapping=mapping,
            compact_episodes=selected,
            morphology=morphology,
            baseline_replay_root=baseline_replay_root,
            baseline_replay_indices=baseline_replay_indices,
            workers=args.workers,
            resume=args.resume,
            gate_budget_limit_improvement_only=(
                args.gate_budget_limit_improvement_only
            ),
            endpoint_parts=args.endpoint_parts,
            canonical_head_height_m=args.canonical_head_height_m,
            hand_endpoint_side=args.hand_endpoint_side,
        )
        return 0

    decisions: list[dict[str, Any]] = []
    for ordinal, compact_episode in enumerate(selected, start=1):
        print(
            f"[endpoint-{endpoint_contract_label(args.endpoint_parts)}] "
            f"episode {ordinal}/{len(selected)} compact={compact_episode} "
            f"source={mapping.compact_to_source[compact_episode]}",
            flush=True,
        )
        decisions.append(
            process_episode(
                raw_dataset=raw_dataset,
                baseline_dataset=baseline_dataset,
                output_root=output_root,
                mapping=mapping,
                compact_episode=compact_episode,
                morphology=morphology,
                baseline_replay_root=baseline_replay_root,
                baseline_replay_indices=baseline_replay_indices,
                resume=args.resume,
                gate_budget_limit_improvement_only=(
                    args.gate_budget_limit_improvement_only
                ),

                endpoint_parts=args.endpoint_parts,
                canonical_head_height_m=args.canonical_head_height_m,
                hand_endpoint_side=args.hand_endpoint_side,
            )
        )
        update_dataset_summary(
            output_root,
            decisions,
            args.endpoint_parts,
            args.canonical_head_height_m,
            args.hand_endpoint_side,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

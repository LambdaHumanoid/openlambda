"""Export raw episodes to timestamped Parquet + RGB MP4 without training dependencies."""

import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .alignment import (
    load_and_verify_manifest,
    _load_hands,
    _load_hand_actions,
    _camera_frames,
    NearestCamera,
    _nearest_hand,
    _causal_hand_action,
    _robot_records,
)
from .schema import (
    validate_robot_sample,
    normalize_hand_positions,
    prepare_ego_image,
    G1_MUJOCO_JOINT_NAMES,
    JOINT_NAMES,
)
from .wire import atomic_write_json


def projected_gravity(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if (
        q.shape != (4,)
        or not np.isfinite(q).all()
        or not 0.9 <= np.linalg.norm(q) <= 1.1
    ):
        raise ValueError("invalid WXYZ base quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1]
    )


def aligned_frames(raw):
    hands = _load_hands(raw / "brainco.records")
    actions = _load_hand_actions(raw / "brainco.records")
    cameras = NearestCamera(_camera_frames(raw))
    previous_index = None
    previous_clock = None
    for robot in _robot_records(raw / "sonic.records"):
        q, dq, token, index, _ = validate_robot_sample(
            robot["body_q"],
            robot["body_dq"],
            robot["token_state"],
            robot_index=robot["index"],
            previous_robot_index=previous_index,
            realtime_ns=robot["source_realtime_ns"],
        )
        previous_index = index
        anchor = int(robot["source_g1_realtime_ns"])
        mono = int(robot["source_g1_monotonic_ns"])
        if previous_clock is not None and (
            anchor <= previous_clock[0] or mono <= previous_clock[1]
        ):
            raise ValueError(
                "source clocks must advance; split episodes at clock resets"
            )
        if (
            previous_clock
            and abs((anchor - previous_clock[0]) - (mono - previous_clock[1]))
            > 5_000_000
        ):
            raise ValueError("source wall clock stepped during recording")
        previous_clock = (anchor, mono)
        camera = cameras.select(anchor)
        left = _nearest_hand(hands["left"], anchor)
        right = _nearest_hand(hands["right"], anchor)
        action = _causal_hand_action(actions, anchor)
        if any(v is None for v in (camera, left, right, action)):
            yield None
            continue
        deltas = [
            abs(camera.timestamp.g1_realtime_ns - anchor),
            abs(left.realtime_ns - anchor),
            abs(right.realtime_ns - anchor),
            anchor - action.realtime_ns,
        ]
        if any(
            delta > limit
            for delta, limit in zip(
                deltas, (75_000_000, 50_000_000, 50_000_000, 50_000_000)
            )
        ):
            yield None
            continue
        left_q, right_q = (
            normalize_hand_positions(left.positions),
            normalize_hand_positions(right.positions),
        )
        gravity = projected_gravity(robot["base_quat"])
        state = np.concatenate((q, left_q, right_q, gravity)).astype(np.float32)
        targets = np.concatenate((token, action.left, action.right)).astype(np.float32)
        yield (
            prepare_ego_image(camera.image),
            {
                "observation.state": state.tolist(),
                "observation.valid_mask": [True] * 44,
                "action": targets.tolist(),
                "action.valid_mask": [True] * 76,
                "observation.body_joint_position": q.tolist(),
                "observation.body_joint_velocity": dq.tolist(),
                "observation.left_hand_position": left_q.tolist(),
                "observation.right_hand_position": right_q.tolist(),
                "observation.projected_gravity": gravity.tolist(),
                "observation.root_orientation": robot["base_quat"],
                "action.motion_token": token.tolist(),
                "action.left_hand_joints": list(action.left),
                "action.right_hand_joints": list(action.right),
                "source.robot_index": index,
                "source.robot_realtime_ns": anchor,
                "source.robot_monotonic_ns": mono,
                "source.robot_received_monotonic_ns": robot["g1_received_monotonic_ns"],
                "source.camera_realtime_ns": camera.timestamp.g1_realtime_ns,
                "source.camera_sequence": camera.timestamp.sequence,
                "source.left_hand_realtime_ns": left.realtime_ns,
                "source.right_hand_realtime_ns": right.realtime_ns,
                "source.hand_action_realtime_ns": action.realtime_ns,
                "source.hand_action_sequence": action.sequence,
                "quality.alignment_delta_ms": [value / 1e6 for value in deltas],
            },
        )


def export_episode(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    manifest = load_and_verify_manifest(raw)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".robot-export-", dir=output.parent))
    writer = None
    try:
        count = rejected = segment = 0
        start_ns = previous_ns = previous_index = None
        batch = []
        segments = []
        with av.open(str(temporary / "ego_view.mp4"), "w") as video:
            stream = video.add_stream("libx264", rate=50)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 90000)
            stream.codec_context.time_base = Fraction(1, 90000)
            stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
            for result in aligned_frames(raw):
                if result is None:
                    rejected += 1
                    continue
                rgb, row = result
                mono, index = (
                    row["source.robot_monotonic_ns"],
                    row["source.robot_index"],
                )
                if start_ns is None:
                    start_ns = mono
                    segments.append({"segment_index": 0, "start_frame": 0})
                elif (
                    index != previous_index + 1
                    or not 10_000_000 <= mono - previous_ns <= 30_000_000
                ):
                    segments[-1]["end_frame_exclusive"] = count
                    segment += 1
                    segments.append({"segment_index": segment, "start_frame": count})
                row.update(
                    timestamp=(mono - start_ns) / 1e9,
                    frame_index=count,
                    segment_index=segment,
                    task=manifest["task"],
                )
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = round((mono - start_ns) * 90000 / 1e9)
                frame.time_base = Fraction(1, 90000)
                for packet in stream.encode(frame):
                    video.mux(packet)
                batch.append(row)
                count += 1
                previous_ns, previous_index = mono, index
                if len(batch) >= 256:
                    table = pa.Table.from_pylist(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary / "data.parquet", table.schema
                        )
                    writer.write_table(table)
                    batch.clear()
            if count == 0:
                raise ValueError(
                    "no aligned samples; inspect source clocks and telemetry"
                )
            if batch:
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(temporary / "data.parquet", table.schema)
                writer.write_table(table)
            for packet in stream.encode():
                video.mux(packet)
        writer.close()
        writer = None
        segments[-1]["end_frame_exclusive"] = count
        metadata = dict(
            format="openlambda_robot_aligned_v1",
            task=manifest["task"],
            frames=count,
            rejected_frames=rejected,
            nominal_fps=50,
            timing="variable; use Parquet timestamp and video PTS",
            segments=segments,
            images={"observation.images.ego_view": "ego_view.mp4"},
            state_dim=44,
            action_dim=76,
            hand_order=list(JOINT_NAMES),
            body_joint_order=list(G1_MUJOCO_JOINT_NAMES),
            state_layout=[
                "body_q[29]",
                "left_feedback[6]",
                "right_feedback[6]",
                "gravity[3]",
            ],
            action_layout=[
                "sonic_encoder_token[64]",
                "left_applied_target[6]",
                "right_applied_target[6]",
            ],
            source_manifest=manifest,
        )
        atomic_write_json(temporary / "episode.json", metadata)
        # Recheck before publication; never merge into an existing episode.
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        return {"frames": count, "rejected_frames": rejected, "segments": len(segments)}
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            shutil.rmtree(temporary)

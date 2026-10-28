# SPDX-License-Identifier: Apache-2.0
"""Read raw episodes and align camera/hand samples to the robot clock."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import av
import numpy as np

from .timing import (
    FRAME_TIMESTAMP_PACKET_SIZE,
    MPEG_CLOCK_HZ,
    MPEG_PTS_WRAP,
    FrameIngressTimestamp,
    unpack_frame_timestamp,
)
from .schema import (
    validate_hand_actions,
)
from .wire import read_records, sha256_file

CAMERA_MAX_DELTA_NS = 75_000_000
HAND_MAX_DELTA_NS = 50_000_000
HAND_ACTION_MAX_DELTA_NS = 50_000_000


class OfflineExportError(RuntimeError):
    """A local raw Episode cannot be verified or exported."""


@dataclass(frozen=True)
class HandSample:
    realtime_ns: int
    source_monotonic_ns: int
    received_monotonic_ns: int
    positions: Tuple[int, ...]


@dataclass(frozen=True)
class HandActionSample:
    sequence: int
    realtime_ns: int
    source_monotonic_ns: int
    received_monotonic_ns: int
    left: Tuple[float, ...]
    right: Tuple[float, ...]


@dataclass(frozen=True)
class CameraFrame:
    timestamp: FrameIngressTimestamp
    image: np.ndarray


def load_and_verify_manifest(raw_directory: Path) -> Dict[str, object]:
    manifest_path = raw_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OfflineExportError("G1 raw Episode has no valid manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "openlambda_robot_raw_v1"
    ):
        raise OfflineExportError("G1 raw Episode format is not supported")
    if manifest.get("status") != "complete":
        raise OfflineExportError(
            "recording is incomplete; retain raw files for diagnosis"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise OfflineExportError("G1 raw Episode manifest has no files map")
    for name in (
        "camera.ts",
        "camera.timestamps.bin",
        "sonic.records",
        "brainco.records",
    ):
        descriptor = files.get(name)
        path = raw_directory / name
        if not isinstance(descriptor, dict) or not path.is_file():
            raise OfflineExportError("G1 raw Episode is missing {}".format(name))
        expected_size = int(descriptor.get("bytes", -1))
        expected_hash = str(descriptor.get("sha256", ""))
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise OfflineExportError(
                "G1 raw Episode integrity check failed for {}".format(name)
            )
    return manifest


def _read_camera_timestamps(path: Path) -> Dict[int, list]:
    by_pts: Dict[int, list] = {}
    with path.open("rb") as stream:
        while True:
            payload = stream.read(FRAME_TIMESTAMP_PACKET_SIZE)
            if not payload:
                break
            if len(payload) != FRAME_TIMESTAMP_PACKET_SIZE:
                raise OfflineExportError("camera timestamp sidecar is truncated")
            try:
                value = unpack_frame_timestamp(payload)
            except ValueError as exc:
                raise OfflineExportError("camera timestamp sidecar is corrupt") from exc
            by_pts.setdefault(value.pts_90khz, []).append(value)
    if not by_pts:
        raise OfflineExportError("camera timestamp sidecar contains no video frames")
    return by_pts


def _camera_frames(raw_directory: Path) -> Iterator[CameraFrame]:
    timestamps = _read_camera_timestamps(raw_directory / "camera.timestamps.bin")
    video_path = raw_directory / "camera.ts"
    matched = 0
    with av.open(str(video_path), mode="r") as container:
        video_streams = tuple(container.streams.video)
        if len(video_streams) != 1:
            raise OfflineExportError("camera.ts must contain exactly one video stream")
        stream = video_streams[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            pts_90khz = (
                int(round(float(frame.pts * frame.time_base) * MPEG_CLOCK_HZ))
                % MPEG_PTS_WRAP
            )
            values = timestamps.get(pts_90khz)
            if not values:
                continue
            timestamp = values.pop(0)
            matched += 1
            yield CameraFrame(
                timestamp=timestamp,
                image=frame.to_ndarray(format="rgb24"),
            )
    if matched == 0:
        raise OfflineExportError(
            "decoded camera frames did not match any G1 PTS timestamps"
        )


class NearestCamera:
    def __init__(self, frames: Iterator[CameraFrame]) -> None:
        self.frames = frames
        self.previous: Optional[CameraFrame] = None
        self.current = next(frames, None)

    def select(self, anchor_realtime_ns: int) -> Optional[CameraFrame]:
        while (
            self.current is not None
            and self.current.timestamp.g1_realtime_ns < anchor_realtime_ns
        ):
            self.previous = self.current
            self.current = next(self.frames, None)
        choices = [
            value for value in (self.previous, self.current) if value is not None
        ]
        return min(
            choices,
            key=lambda value: abs(value.timestamp.g1_realtime_ns - anchor_realtime_ns),
            default=None,
        )


def _load_hands(path: Path) -> Dict[str, Tuple[list, list]]:
    values: Dict[str, list] = {"left": [], "right": []}
    for record in read_records(path):
        if record.get("type") != "brainco":
            continue
        side = str(record.get("side", ""))
        if side not in values:
            raise OfflineExportError("BrainCo record has an invalid side")
        positions_raw = record.get("positions")
        if not isinstance(positions_raw, (list, tuple)) or len(positions_raw) != 6:
            raise OfflineExportError("BrainCo record has invalid positions")
        sample = HandSample(
            realtime_ns=int(record["source_realtime_ns"]),
            source_monotonic_ns=int(record["source_monotonic_ns"]),
            received_monotonic_ns=int(record["g1_received_monotonic_ns"]),
            positions=tuple(int(item) for item in positions_raw),
        )
        if (
            sample.realtime_ns <= 0
            or sample.source_monotonic_ns <= 0
            or sample.received_monotonic_ns <= 0
        ):
            raise OfflineExportError("BrainCo record has invalid G1 clocks")
        values[side].append(sample)
    result = {}
    for side, samples in values.items():
        samples.sort(key=lambda sample: sample.realtime_ns)
        if not samples:
            raise OfflineExportError(
                "G1 raw Episode contains no {} hand data".format(side)
            )
        result[side] = ([sample.realtime_ns for sample in samples], samples)
    return result


def _nearest_hand(
    values: Tuple[list, list], anchor_realtime_ns: int
) -> Optional[HandSample]:
    timestamps, samples = values
    index = bisect.bisect_left(timestamps, anchor_realtime_ns)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index > 0:
        candidates.append(samples[index - 1])
    return min(
        candidates,
        key=lambda sample: abs(sample.realtime_ns - anchor_realtime_ns),
        default=None,
    )


def _load_hand_actions(path: Path) -> Tuple[list, list]:
    samples = []
    for record in read_records(path):
        if record.get("type") != "brainco_action":
            continue
        try:
            left = validate_hand_actions(record["left"])
            right = validate_hand_actions(record["right"])
            sample = HandActionSample(
                sequence=int(record["sequence"]),
                realtime_ns=int(record["source_realtime_ns"]),
                source_monotonic_ns=int(record["source_monotonic_ns"]),
                received_monotonic_ns=int(record["g1_received_monotonic_ns"]),
                left=tuple(float(item) for item in left),
                right=tuple(float(item) for item in right),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OfflineExportError("BrainCo action record is invalid") from exc
        if (
            sample.sequence < 0
            or sample.realtime_ns <= 0
            or sample.source_monotonic_ns <= 0
            or sample.received_monotonic_ns <= 0
        ):
            raise OfflineExportError("BrainCo action record has invalid clocks")
        samples.append(sample)
    samples.sort(key=lambda sample: sample.realtime_ns)
    if not samples:
        raise OfflineExportError("G1 raw Episode contains no applied hand actions")
    return [sample.realtime_ns for sample in samples], samples


def _causal_hand_action(
    values: Tuple[list, list], anchor_realtime_ns: int
) -> Optional[HandActionSample]:
    timestamps, samples = values
    index = bisect.bisect_right(timestamps, anchor_realtime_ns) - 1
    return samples[index] if index >= 0 else None


def _robot_records(path: Path) -> Iterator[Dict[str, object]]:
    for record in read_records(path):
        if record.get("type") == "sonic":
            yield record

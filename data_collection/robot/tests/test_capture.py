"""Offline regression tests: synthetic source data only, no hardware access."""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import msgpack
import numpy as np
import pyarrow.parquet as pq

from lambda_robot_data.sources import (
    decode_sonic,
    decode_brainco,
    ROBOT_TOPIC,
    APPLIED_HAND_TOPIC,
)
from lambda_robot_data.wire import (
    RecordWriter,
    read_records,
    sha256_file,
    EpisodeWireError,
)
from lambda_robot_data.timing import (
    MpegTsPtsExtractor,
    FrameIngressTimestamp,
    pack_frame_timestamp,
)
from lambda_robot_data.export import export_episode, projected_gravity
from lambda_robot_data.hand_telemetry import command_values, JOINT_NAMES

BASE = 1_800_000_000_000_000_000
MONO = 10_000_000_000


def fixture(root, gap=False, future=False):
    root.mkdir()
    with av.open(str(root / "camera.ts"), "w", format="mpegts") as video:
        stream = video.add_stream("libx264", rate=50)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"bf": "0", "preset": "ultrafast", "tune": "zerolatency"}
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 10, np.uint8), format="rgb24"
            )
            frame.pts = i * 1800
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    pts = MpegTsPtsExtractor().feed((root / "camera.ts").read_bytes())
    assert len(pts) == 12, pts
    with (root / "camera.timestamps.bin").open("wb") as f:
        for i, value in enumerate(pts):
            f.write(
                pack_frame_timestamp(
                    FrameIngressTimestamp(
                        1, i + 1, value, MONO + i * 20_000_000, BASE + i * 20_000_000
                    )
                )
            )
    robot = RecordWriter(root / "sonic.records")
    hands = RecordWriter(root / "brainco.records")
    robot.write(dict(type="robot_config", value={"control_frequency": 50}))
    for i in range(12):
        wall, mono = BASE + i * 20_000_000, MONO + i * 20_000_000
        if not (gap and i == 5):
            robot.write(
                dict(
                    type="sonic",
                    index=i,
                    body_q=[0.0] * 29,
                    body_dq=[0.0] * 29,
                    token_state=[0.25] * 64,
                    base_quat=[1, 0, 0, 0],
                    source_realtime_ns=wall,
                    source_g1_realtime_ns=wall,
                    source_g1_monotonic_ns=mono,
                    source_monotonic_ns=mono,
                    g1_received_monotonic_ns=mono + 1_000_000,
                )
            )
        for side in ("left", "right"):
            hands.write(
                dict(
                    type="brainco",
                    side=side,
                    positions=[100] * 6,
                    source_realtime_ns=wall,
                    source_monotonic_ns=mono,
                    g1_received_monotonic_ns=mono,
                )
            )
        hands.write(
            dict(
                type="brainco_action",
                sequence=i,
                left=[0.8] * 6,
                right=[0.7] * 6,
                source_realtime_ns=wall + (1_000_000_000 if future else 0),
                source_monotonic_ns=mono,
                g1_received_monotonic_ns=mono,
            )
        )
    robot.close()
    hands.close()
    files = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p))
        for p in root.iterdir()
    }
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                format="openlambda_robot_raw_v1",
                status="complete",
                task="synthetic test only",
                files=files,
            )
        )
    )


class CaptureTests(unittest.TestCase):
    def test_wire_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "records"
            w = RecordWriter(p)
            w.write({"test": 1})
            w.close()
            self.assertEqual(list(read_records(p)), [{"test": 1}])
            data = bytearray(p.read_bytes())
            data[-1] ^= 1
            p.write_bytes(data)
            with self.assertRaises(EpisodeWireError):
                list(read_records(p))

    def test_truncated_records(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "records"
            w = RecordWriter(p)
            w.write({"test": 1})
            w.close()
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(EpisodeWireError):
                list(read_records(p))

    def test_source_rejects_nan(self):
        payload = dict(
            index=1,
            timestamp_monotonic_ns=1,
            timestamp_realtime_ns=BASE,
            body_q=[float("nan")] * 29,
            body_dq=[0] * 29,
            token_state=[0] * 64,
            base_quat=[1, 0, 0, 0],
        )
        with self.assertRaises(ValueError):
            decode_sonic(ROBOT_TOPIC + msgpack.packb(payload), 0, 0)

    def test_applied_targets_are_required(self):
        payload = dict(
            sequence=1, monotonic_ns=1, wall_time_ns=BASE, positions=[100] * 6
        )
        with self.assertRaises(ValueError):
            decode_brainco([APPLIED_HAND_TOPIC, msgpack.packb(payload)])

    def test_hand_order(self):
        self.assertEqual(command_values(JOINT_NAMES, [0.5] * 6), [0.5] * 6)
        with self.assertRaises(ValueError):
            command_values(JOINT_NAMES[::-1], [0.5] * 6)
        with self.assertRaises(ValueError):
            command_values(JOINT_NAMES, [100] * 6)

    def test_gravity(self):
        np.testing.assert_allclose(projected_gravity([1, 0, 0, 0]), [0, 0, -1])
        np.testing.assert_allclose(
            projected_gravity([2**-0.5, 0, 2**-0.5, 0]), [1, 0, 0], atol=1e-12
        )
        with self.assertRaises(ValueError):
            projected_gravity([0, 0, 0, 0])

    def test_export_roundtrip_and_distinct_actions(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            fixture(raw)
            result = export_episode(raw, out)
            self.assertEqual(result["frames"], 12)
            rows = pq.read_table(out / "data.parquet").to_pylist()
            self.assertEqual(len(rows[0]["observation.state"]), 44)
            self.assertEqual(len(rows[0]["action"]), 76)
            self.assertAlmostEqual(rows[0]["observation.state"][29], 0.1, places=5)
            self.assertAlmostEqual(rows[0]["action"][64], 0.8, places=5)
            with av.open(str(out / "ego_view.mp4")) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), len(rows))
                for frame, row in zip(frames, rows):
                    self.assertAlmostEqual(
                        float(frame.pts * frame.time_base), row["timestamp"], places=4
                    )
            with self.assertRaises(FileExistsError):
                export_episode(raw, out)

    def test_gaps_preserve_time_and_split_segments(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            fixture(raw, gap=True)
            result = export_episode(raw, out)
            self.assertEqual(result["segments"], 2)
            rows = pq.read_table(out / "data.parquet").to_pylist()
            self.assertAlmostEqual(rows[5]["timestamp"], 0.12)
            self.assertEqual(rows[5]["segment_index"], 1)

    def test_future_actions_never_fill_missing_past(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            fixture(raw, future=True)
            with self.assertRaisesRegex(ValueError, "no aligned"):
                export_episode(raw, out)
            self.assertFalse(out.exists())

    def test_hash_mismatch_fails_before_export(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            fixture(raw)
            with (raw / "camera.ts").open("ab") as f:
                f.write(b"bad")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                export_episode(raw, out)
            self.assertFalse(out.exists())

    def test_incomplete_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            fixture(raw)
            p = raw / "manifest.json"
            v = json.loads(p.read_text())
            v["status"] = "incomplete"
            p.write_text(json.dumps(v))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                export_episode(raw, out)


if __name__ == "__main__":
    unittest.main()

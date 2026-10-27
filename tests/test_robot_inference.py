"""Isolated client checks; no checkpoint, controller or hardware is started."""

import ast
import json
import sys
import tempfile
import threading
import time
import unittest
from base64 import b64decode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy.lib.format import descr_to_dtype
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from infer_robot import (
    ActionBatch,
    ActionSpec,
    LambdaPolicy,
    LiveObservations,
    RouterSession,
    action_chunk,
    build,
    decode_action,
    encode_action,
    gravity,
    pose_matches,
    require_compatible,
    unpack,
    wait_initial_pose,
)
from infer_robot import (
    G1Frame as Observation,
)
from infer_robot import (
    Observation as NamedObservation,
)


def deployment():
    return json.loads(
        (
            Path(__file__).parents[1] / "configs/robot_inference/lambda0_g1.json"
        ).read_text()
    )


def config(**overrides):
    d = deployment()
    p = d["policy"]
    o = d["observation"]
    c = d["controller"]
    values = dict(
        p,
        instruction=d["instruction"],
        policy_url=p["url"],
        body_endpoint=o["body_endpoint"],
        hand_endpoint=o["hand_endpoint"],
        camera_endpoint=o["cameras"]["ego"]["endpoint"],
        router_control=c["control_endpoint"],
        router_policy=c["policy_endpoint"],
        execute_steps=50,
        digest="test",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def policy_client(**overrides):
    d = deployment()
    d["policy"].update(overrides)
    return LambdaPolicy(d["policy"], d["instruction"])


def state():
    return np.array([0.0] * 29 + [0.1] * 12 + [0.0, 0.0, -1.0], dtype=np.float32)


class InferenceTests(unittest.TestCase):
    def test_config_rejects_remote_actuation_and_wrong_contract(self):
        for section, key, value in [
            ("controller", "policy_endpoint", "tcp://10.1.2.3:5556"),
            ("policy", "horizon", 16),
            ("policy", "action_space_id", 3),
            ("policy", "image_order", ["ego", "wrist"]),
        ]:
            d = deployment()
            d[section][key] = value
            with self.assertRaises(ValueError):
                build(d)

    def test_hand_clamp_keeps_token_and_reports(self):
        raw = np.full((50, 76), 0.2, dtype=np.float32)
        raw[0, 64] = -1
        raw[1, 65] = 2
        value, info = action_chunk(raw)
        self.assertEqual(info["hand_clipped_count"], 2)
        self.assertEqual(value[0, 64], 0)
        self.assertEqual(value[1, 65], 1)
        np.testing.assert_array_equal(value[:, :64], raw[:, :64])
        self.assertEqual(raw[0, 64], -1)

    def test_bad_actions_fail(self):
        for value in (
            np.zeros((16, 76)),
            np.full((50, 76), np.nan),
            np.full((50, 76), 2),
        ):
            with self.assertRaises(ValueError):
                action_chunk(value)

    def test_measured_state_and_gravity(self):
        np.testing.assert_allclose(gravity([1, 0, 0, 0]), [0, 0, -1])
        value = state()
        value[29] = 1.1
        with self.assertRaises(ValueError):
            Observation(np.zeros((2, 2, 3), np.uint8), value, {})
        with self.assertRaises(ValueError):
            gravity([0, 0, 0, 0])

    def test_pose_requires_hands_and_low_velocity(self):
        expected = state()
        self.assertTrue(pose_matches(expected, expected, np.zeros(29)))
        actual = expected.copy()
        actual[29] = 0.9
        self.assertFalse(pose_matches(actual, expected, np.zeros(29)))
        self.assertFalse(pose_matches(expected, expected, np.ones(29)))

    def test_pose_wait_requires_consecutive_distinct_samples(self):
        class Sensors:
            index = 0

            def read(self, timeout):
                self.index += 1
                value = state()
                if self.index == 25:
                    value[0] = 1
                return Observation(
                    np.zeros((2, 2, 3), np.uint8),
                    value,
                    {"body_index": self.index, "body_dq": [0] * 29},
                )

        sensors = Sensors()
        wait_initial_pose(sensors, state(), timeout=2)
        self.assertEqual(sensors.index, 75)

    def test_signed_packets_reject_tamper_replay_stale(self):
        identity = {"session": "test"}
        secret = "x" * 64
        action = np.zeros(76)
        packet = encode_action(action, 1, identity, secret)
        value, seq = decode_action(packet, identity, secret, 0)
        self.assertEqual(seq, 1)
        self.assertEqual(value.shape, (76,))
        with self.assertRaises(ValueError):
            decode_action(packet, identity, secret, 1)
        with self.assertRaises(ValueError):
            decode_action(packet[:-1] + b"x", identity, secret, 0)
        with patch(
            "infer_robot.time.monotonic_ns",
            return_value=time.monotonic_ns() + 500_000_000,
        ):
            with self.assertRaises(ValueError):
                decode_action(packet, identity, secret, 0)

    def test_router_refuses_planner_or_owned_session(self):
        router = RouterSession(config(), "test")
        try:
            router.request = lambda *a, **k: {
                "phase": "planner",
                "stable": True,
                "policy_enabled": False,
            }
            with self.assertRaises(RuntimeError):
                router.ready()
            router.request = lambda *a, **k: {
                "phase": "hold",
                "stable": True,
                "policy_enabled": True,
            }
            with self.assertRaises(RuntimeError):
                router.ready()
        finally:
            router.close()

    def test_clock_mapping_and_post_chunk_barrier(self):
        sensors = LiveObservations(config())
        now = time.monotonic_ns() - 10_000_000
        offset = 123_000_000_000

        def publish(stamp, index):
            sensors.accept(
                "body",
                stamp,
                (np.zeros(29), np.array([0, 0, -1]), np.zeros(29), index, np.zeros(64)),
            )
            for side in ("left", "right"):
                sensors.accept(side, stamp + offset, np.full(6, 0.1))
            sensors.accept(
                "camera", stamp, (np.zeros((2, 2, 3), np.uint8), stamp + offset)
            )

        publish(now, 0)
        value = sensors.read(timeout=0.1)
        np.testing.assert_allclose(value.state, state())
        self.assertEqual(value.metadata["alignment_max_ms"], 0)
        with self.assertRaises(TimeoutError):
            sensors.read(timeout=0.02, after_ns=now)
        publish(time.monotonic_ns() - 1_000_000, 1)
        newer = sensors.read(timeout=0.1, after_ns=now)
        self.assertTrue(
            all(x > now for x in newer.metadata["alignment_monotonic_ns"].values())
        )

    def test_stale_sensor_fails(self):
        sensors = LiveObservations(config())
        stamp = time.monotonic_ns() - 1_000_000_000
        for key in sensors.samples:
            sensors.samples[key].append((stamp, stamp, None))
        with self.assertRaises(RuntimeError):
            sensors.assert_fresh()

    def test_http_client_matches_public_request_parser(self):
        settings = config()
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                health = {
                    "status": "ok",
                    "model_path": settings.model_path,
                    "normalization_dataset": settings.normalization_dataset,
                    "normalization_action_space_id": 2,
                    "inference_steps": 10,
                    "rtc_enabled": False,
                }
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(health).encode())

            def do_POST(self):
                received.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"action": np.zeros((50, 76)).tolist(), "action_space_id": 2}
                    ).encode()
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = policy_client(url=f"http://127.0.0.1:{server.server_port}")
            value = Observation(np.full((48, 64, 3), 128, np.uint8), state(), {})
            actions, _ = policy.predict(unpack(value), "test-session", 0)
            self.assertEqual(len(actions), 50)
            self.assertEqual(actions.values["motion"].shape, (50, 64))
            source = Path(__file__).parents[1] / "src/openlambda/inference/server.py"
            tree = ast.parse(source.read_text())
            names = {"decode_numpy_envelope", "decode_image", "parse_request"}
            selected = [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ]
            namespace = {
                "Any": Any,
                "np": np,
                "Image": Image,
                "b64decode": b64decode,
                "descr_to_dtype": descr_to_dtype,
                "SIMPLE_IMAGE_KEY": "rgb_head_stereo_left",
                "SIMPLE_STATE_KEY": "states",
            }
            exec(
                compile(
                    ast.Module(body=selected, type_ignores=[]), str(source), "exec"
                ),
                namespace,
            )
            images, task, proprio, domain, space, simple = namespace["parse_request"](
                received[0], default_embodiment_id=None, default_action_space_id=None
            )
            self.assertEqual(images[0].size, (448, 448))
            self.assertEqual(space, 2)
            self.assertFalse(simple)
            self.assertEqual(task, settings.instruction)
            np.testing.assert_allclose(proprio, state())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_generic_preview_multiple_views_and_different_action_dimensions(self):
        from unittest.mock import MagicMock

        from infer_robot import live

        value = NamedObservation(
            {
                "front": np.zeros((4, 4, 3), np.uint8),
                "wrist": np.zeros((2, 2, 3), np.uint8),
            },
            {"joints": np.zeros(7)},
            {},
        )
        spec = ActionSpec("joint_position", {"joints": 7}, {"joints": "rad"}, 20)
        batch = ActionBatch(spec, {"joints": np.zeros((3, 7))})
        source = MagicMock()
        source.assert_fresh = MagicMock()
        source.read.return_value = value
        policy = SimpleNamespace(
            required_images=("front", "wrist"),
            required_state=("joints",),
            output_spec=spec,
            predict=lambda *args: (batch, {}),
        )
        controller = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            live(
                {"schedule": {"execute_steps": 2}},
                source,
                policy,
                controller,
                SimpleNamespace(execute=False, chunks=1, output=output),
                "test",
            )
            controller.prepare.assert_not_called()
            controller.execute.assert_not_called()
            controller.close.assert_not_called()
            self.assertTrue((output / "chunk-0000-wrist.jpg").exists())
            arrays = np.load(output / "chunk-0000.npz")
            self.assertEqual(arrays["action.joints"].shape, (3, 7))

    def test_action_contract_rejects_same_dimensions_with_different_units(self):
        policy = SimpleNamespace(
            output_spec=ActionSpec("joints", {"q": 7}, {"q": "rad"}, 20)
        )
        controller = SimpleNamespace(
            input_spec=ActionSpec("joints", {"q": 7}, {"q": "deg"}, 20)
        )
        with self.assertRaises(ValueError):
            require_compatible(policy, controller)

    def test_named_observation_requires_missing_view(self):
        value = NamedObservation(
            {"ego": np.zeros((2, 2, 3), np.uint8)}, {"q": np.zeros(7)}, {}
        )
        with self.assertRaises(ValueError):
            value.require(("ego", "wrist"), ("q",))

    def test_execution_barrier_and_cleanup_after_partial_failure(self):
        from unittest.mock import MagicMock

        from infer_robot import live

        spec = ActionSpec("joint_position", {"q": 7}, {"q": "rad"}, 20)
        batch = ActionBatch(spec, {"q": np.zeros((3, 7))})
        value = NamedObservation(
            {"front": np.zeros((2, 2, 3), np.uint8)}, {"q": np.zeros(7)}, {}
        )
        source = MagicMock()
        source.assert_fresh = MagicMock()
        source.read.return_value = value
        policy = SimpleNamespace(
            required_images=("front",),
            required_state=("q",),
            output_spec=spec,
            predict=lambda *args: (batch, {}),
        )
        controller = MagicMock()
        controller.published_steps = 0
        controller.last_publication_ns = None

        def execute(*args):
            if controller.published_steps == 0:
                controller.published_steps = 2
                controller.last_publication_ns = 123
                return 123
            controller.published_steps += 1
            controller.last_publication_ns = None
            raise RuntimeError("sender failed")

        controller.execute.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "sender failed"):
                live(
                    {"schedule": {"execute_steps": 2}},
                    source,
                    policy,
                    controller,
                    SimpleNamespace(
                        execute=True,
                        chunks=2,
                        output=output,
                        initial_episode="reference",
                    ),
                    "test",
                )
            self.assertEqual(source.read.call_args_list[1].kwargs["after_ns"], 123)
            controller.close.assert_called_once()
            saved = json.loads((output / "chunk-0001.json").read_text())
            self.assertEqual(saved["executed_steps"], 1)

    def test_sonic_rejects_expired_observation_before_enabling(self):
        from unittest.mock import MagicMock

        from infer_robot import SonicController

        controller = SonicController(deployment()["controller"])
        controller.process = MagicMock()
        controller.router = MagicMock()
        source = MagicMock()
        source.assert_fresh = MagicMock()
        value = unpack(
            Observation(
                np.zeros((2, 2, 3), np.uint8),
                state(),
                {"camera_host_monotonic_ns": time.monotonic_ns() - 2_000_000_000},
            )
        )
        batch = ActionBatch(
            controller.input_spec,
            {
                "motion": np.zeros((2, 64)),
                "left_hand": np.zeros((2, 6)),
                "right_hand": np.zeros((2, 6)),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "expired"):
            controller.execute(batch, source, value, 2)
        controller.router.enable.assert_not_called()
        controller.router.send_chunk.assert_not_called()

    def test_episode_reader_preserves_recorded_frame(self):
        from fractions import Fraction

        import av
        import pyarrow as pa
        import pyarrow.parquet as pq
        from infer_robot import read_episode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode.json").write_text(
                json.dumps({"format": "openlambda_robot_aligned_v1"})
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "observation.state": state().tolist(),
                            "observation.valid_mask": [True] * 44,
                            "timestamp": 0.0,
                            "task": config().instruction,
                            "segment_index": 0,
                        }
                    ]
                ),
                root / "data.parquet",
            )
            with av.open(str(root / "ego_view.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=50)
                stream.width = 64
                stream.height = 48
                stream.pix_fmt = "yuv420p"
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((48, 64, 3), np.uint8), format="rgb24"
                )
                frame.pts = 0
                frame.time_base = Fraction(1, 50)
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            value = read_episode(root)
            self.assertEqual(value.rgb.shape, (48, 64, 3))
            np.testing.assert_allclose(value.state, state())
            with self.assertRaises(ValueError):
                read_episode(root, 1)

    def test_service_identity_mismatch(self):
        policy = policy_client()
        policy.request = lambda *a, **k: {"status": "ok", "model_path": "different"}
        with self.assertRaises(ValueError):
            policy.check()


if __name__ == "__main__":
    unittest.main()

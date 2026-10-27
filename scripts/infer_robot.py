"""G1/Lambda0 inference with recorded or live observations and SONIC execution.

Run from the repository root:
    python scripts/infer_robot.py --config configs/robot_inference/local.json run --output outputs/preview

See docs/robot_inference/technical.zh-CN.md for configuration and runtime interfaces.
"""

import argparse
import base64
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import signal
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import msgpack
import numpy as np
import zmq
from PIL import Image

# Shared data structures
# ----------------------------------------------------------------------


@dataclass
class Observation:
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    timestamps: dict[str, int]
    clock: str = "host_monotonic"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, value in self.images.items():
            image = np.asarray(value)
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or not image.size
            ):
                raise ValueError(f"{name}: expected a nonempty uint8 RGB image")
            if image.shape[0] * image.shape[1] > 16_000_000:
                raise ValueError(f"{name}: image too large")
            self.images[name] = image
        for name, value in self.state.items():
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                raise ValueError(f"{name}: expected a finite state vector")
            self.state[name] = array
        if any(
            type(value) is not int or value < 0 for value in self.timestamps.values()
        ):
            raise ValueError("Timestamps must be non-negative integer nanoseconds")

    def require(self, images, state):
        missing = [f"images.{name}" for name in images if name not in self.images]
        missing += [f"state.{name}" for name in state if name not in self.state]
        if missing:
            raise ValueError("Missing model inputs: " + ", ".join(missing))


@dataclass(frozen=True)
class ActionSpec:
    representation: str
    fields: dict[str, int]
    units: dict[str, str]
    frequency_hz: float
    reference: str = "absolute"

    def __post_init__(self):
        if not self.representation or not self.reference or not self.fields:
            raise ValueError("Action representation, fields and reference are required")
        if set(self.fields) != set(self.units) or any(
            not value for value in self.units.values()
        ):
            raise ValueError("Every action field requires units")
        if any(type(size) is not int or size <= 0 for size in self.fields.values()):
            raise ValueError("Action field dimensions must be positive integers")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Action frequency must be positive")


@dataclass
class ActionBatch:
    spec: ActionSpec
    values: dict[str, np.ndarray]

    def __post_init__(self):
        if set(self.values) != set(self.spec.fields):
            raise ValueError("Action fields do not match declared contract")
        lengths = set()
        for name, width in self.spec.fields.items():
            array = np.asarray(self.values[name], dtype=np.float32)
            if (
                array.ndim != 2
                or array.shape[1] != width
                or not len(array)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid action field: {name}")
            lengths.add(len(array))
            self.values[name] = array
        if len(lengths) != 1:
            raise ValueError("Action fields have different horizons")

    def __len__(self):
        return len(next(iter(self.values.values())))


def require_compatible(policy, controller):
    if policy.output_spec != controller.input_spec:
        raise ValueError(
            "Policy output and controller input differ in representation, dimensions, units, reference or frequency"
        )


# Configuration
# ----------------------------------------------------------------------


def local_endpoint(value, scheme):
    parsed = urlparse(value)
    if (
        parsed.scheme != scheme
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or not parsed.port
    ):
        raise ValueError(f"Expected a local {scheme} endpoint: {value}")
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Endpoint must not contain credentials, query or path")
    return value.rstrip("/")


def load(path):
    data = json.loads(Path(path).read_text())
    for section in ("observation", "policy", "controller"):
        if not isinstance(data.get(section), dict) or not isinstance(
            data[section].get("type"), str
        ):
            raise ValueError(f"{section}.type is required")
    if not isinstance(data.get("instruction"), str) or not data["instruction"].strip():
        raise ValueError("instruction is required")
    schedule = data["schedule"]
    if schedule.get("type") != "serial":
        raise ValueError("Only serial scheduling is implemented")
    if type(schedule.get("execute_steps")) is not int or schedule["execute_steps"] <= 0:
        raise ValueError("execute_steps must be a positive integer")
    return data


# G1 state and action layout
# ----------------------------------------------------------------------


def vector(value, size, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return array


def gravity(quaternion):
    q = vector(quaternion, 4, "WXYZ root quaternion")
    norm = np.linalg.norm(q)
    if not 0.9 <= norm <= 1.1:
        raise ValueError("Invalid root quaternion norm")
    w, x, y, z = q / norm
    return np.array(
        [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1],
        dtype=np.float32,
    )


def state_vector(value):
    result = vector(value, 44, "state")
    if np.max(np.abs(result[:29])) > 16:
        raise ValueError("Body position exceeds validity bound")
    if np.any(result[29:41] < 0) or np.any(result[29:41] > 1):
        raise ValueError("Measured hands must be in [0,1]")
    if not 0.95 <= np.linalg.norm(result[41:]) <= 1.05:
        raise ValueError("Projected gravity must have unit length")
    return result


def action_chunk(value, horizon=50):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (horizon, 76) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite [{horizon},76] action chunk")
    if np.max(np.abs(array[:, :64])) > 1.25:
        raise ValueError("SONIC token exceeds runtime limit 1.25")
    # Preserve production Lambda0 hand clamping, with diagnostics before modification.
    array = array.copy()
    hands = array[:, 64:]
    info = {
        "hand_clipped_count": int(np.count_nonzero((hands < 0) | (hands > 1))),
        "hand_raw_min": float(hands.min()),
        "hand_raw_max": float(hands.max()),
    }
    np.clip(hands, 0, 1, out=hands)
    return array, info


@dataclass
class G1Frame:
    rgb: np.ndarray
    state: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.rgb = np.asarray(self.rgb)
        if (
            self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
            or not self.rgb.size
        ):
            raise ValueError("RGB must be uint8 HWC")
        if self.rgb.shape[0] * self.rgb.shape[1] > 16_000_000:
            raise ValueError("Image too large")
        self.state = state_vector(self.state)


# Recorded episodes
# ----------------------------------------------------------------------


def read_episode(path, frame=0):
    import av
    import pyarrow.parquet as pq

    root = Path(path)
    metadata = json.loads((root / "episode.json").read_text())
    if metadata.get("format") != "openlambda_robot_aligned_v1":
        raise ValueError("Unsupported episode format")
    if frame < 0:
        raise ValueError("Frame index must be non-negative")
    rows = pq.read_table(root / "data.parquet").slice(frame, 1).to_pylist()
    if not rows:
        raise ValueError("Frame is outside episode")
    row = rows[0]
    if len(row["observation.valid_mask"]) != 44 or not all(
        row["observation.valid_mask"]
    ):
        raise ValueError("G1Frame has invalid state fields")
    with av.open(str(root / "ego_view.mp4")) as container:
        for index, image in enumerate(container.decode(video=0)):
            if index == frame:
                if (
                    image.pts is None
                    or abs(float(image.pts * image.time_base) - row["timestamp"])
                    > 0.001
                ):
                    raise ValueError("Video/Parquet timestamps disagree")
                return G1Frame(
                    image.to_ndarray(format="rgb24"),
                    np.asarray(row["observation.state"]),
                    {
                        "source": "episode",
                        "frame_index": frame,
                        "timestamp": row["timestamp"],
                        "task": row["task"],
                        "segment_index": row["segment_index"],
                    },
                )
    raise ValueError("Video does not contain the requested frame")


# Live observation streams
# ----------------------------------------------------------------------


class LiveObservations:
    def __init__(self, config):
        self.endpoints = {
            "body": config.body_endpoint,
            "hands": config.hand_endpoint,
            "camera": config.camera_endpoint,
        }
        self.samples = {
            name: deque(maxlen=2 if name == "camera" else 128)
            for name in ("body", "left", "right", "camera")
        }
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.error = None
        self.last_camera = -1
        self.last_body_index = -1
        self.thread = threading.Thread(target=self._receive, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.closed.set()
        self.thread.join(timeout=2)

    def accept(self, key, stamp, value):
        if type(stamp) is not int or stamp <= 0:
            raise ValueError("Missing source timestamp")
        with self.lock:
            rows = self.samples[key]
            if rows and stamp <= rows[-1][0]:
                return
            rows.append((stamp, time.monotonic_ns(), value))

    def decode(self, key, frames):
        if key == "body":
            if len(frames) != 1 or not frames[0].startswith(b"g1_debug"):
                return
            row = msgpack.unpackb(frames[0][len(b"g1_debug") :], raw=False)
            stamp = int(row["timestamp_monotonic_ns"])
            index = int(row["index"])
            if not 0 <= time.monotonic_ns() - stamp <= 250_000_000:
                raise ValueError("Stale SONIC source clock")
            if index <= self.last_body_index:
                return
            q = vector(row["body_q"], 29, "body_q")
            dq = vector(row["body_dq"], 29, "body_dq")
            token = vector(row["token_state"], 64, "token_state")
            if np.max(np.abs(q)) > 16 or np.max(np.abs(dq)) > 200:
                raise ValueError("Invalid measured body state")
            self.last_body_index = index
            self.accept("body", stamp, (q, gravity(row["base_quat"]), dq, index, token))
        elif key == "hands":
            if len(frames) != 2 or frames[0] not in (
                b"rt/brainco/left/state",
                b"rt/brainco/right/state",
            ):
                return
            row = msgpack.unpackb(frames[1], raw=False)
            positions = vector(row["positions"], 6, "hand positions")
            if np.any(positions < 0) or np.any(positions > 1000):
                raise ValueError("Invalid hand feedback units")
            self.accept(
                frames[0].decode().split("/")[2],
                int(row["monotonic_ns"]),
                positions / 1000,
            )
        else:
            row = msgpack.unpackb(frames[-1], raw=False)
            stamps = row["timestamps"]
            host = float(stamps["ego_view_host_monotonic"])
            robot = float(stamps["ego_view_g1_monotonic"])
            if not all(math.isfinite(x) and x > 0 for x in (host, robot)):
                raise ValueError("Invalid camera clocks")
            host_ns, robot_ns = int(host * 1e9), int(robot * 1e9)
            if not -50_000_000 <= time.monotonic_ns() - host_ns <= 250_000_000:
                raise ValueError("Stale camera source timestamp")
            encoded = row["images"]["ego_view"]
            if len(encoded) > 24 * 1024 * 1024:
                raise ValueError("Camera message too large")
            jpeg = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("Camera image too large")
                rgb = np.asarray(image.convert("RGB")).copy()
            self.accept("camera", host_ns, (rgb, robot_ns))

    def _receive(self):
        context = zmq.Context()
        sockets = {}
        poller = zmq.Poller()
        try:
            for key, endpoint in self.endpoints.items():
                sock = context.socket(zmq.SUB)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVHWM, 2 if key == "camera" else 8)
                sock.setsockopt(zmq.SUBSCRIBE, b"g1_debug" if key == "body" else b"")
                sock.connect(endpoint)
                sockets[sock] = key
                poller.register(sock, zmq.POLLIN)
            while not self.closed.is_set():
                for sock in dict(poller.poll(50)):
                    frames = sock.recv_multipart()
                    key = sockets[sock]
                    if key == "camera":
                        for _ in range(8):
                            try:
                                frames = sock.recv_multipart(zmq.NOBLOCK)
                            except zmq.Again:
                                break
                    self.decode(key, frames)
        except Exception as error:
            self.error = str(error)
        finally:
            for sock in sockets:
                sock.close()
            context.term()

    def assert_fresh(self):
        if self.closed.is_set() or self.error:
            raise RuntimeError(self.error or "Observer stopped")
        now = time.monotonic_ns()
        with self.lock:
            latest = {
                name: (rows[-1] if rows else None)
                for name, rows in self.samples.items()
            }
        if any(row is None or now - row[1] > 250_000_000 for row in latest.values()):
            raise RuntimeError("Measured sensor stream missing or stale")
        for key in ("body", "camera"):
            if not -50_000_000 <= now - latest[key][0] <= 250_000_000:
                raise RuntimeError(key + " source is stale")
        camera = latest["camera"]
        g1_stamp = camera[2][1]
        for key in ("left", "right"):
            mapped = camera[0] + latest[key][0] - g1_stamp
            if not -50_000_000 <= now - mapped <= 250_000_000:
                raise RuntimeError(key + " source is stale")

    def read(self, timeout=3, after_ns=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.closed.is_set() or self.error:
                raise RuntimeError(self.error or "Observer stopped")
            with self.lock:
                data = {key: list(rows) for key, rows in self.samples.items()}
            if all(data.values()):
                camera = data["camera"][-1]
                if camera[0] <= self.last_camera or (
                    after_ns is not None and camera[0] <= after_ns
                ):
                    self.closed.wait(0.005)
                    continue
                mapped = {}
                for key in ("body", "left", "right"):
                    rows = [
                        (
                            row[0]
                            if key == "body"
                            else camera[0] + row[0] - camera[2][1],
                            row[1],
                            row[2],
                        )
                        for row in data[key]
                    ]
                    if after_ns is not None:
                        rows = [row for row in rows if row[0] > after_ns]
                    if rows:
                        mapped[key] = min(rows, key=lambda row: abs(row[0] - camera[0]))
                now = time.monotonic_ns()
                rows = [camera, *mapped.values()]
                if len(mapped) == 3 and all(
                    now - row[1] <= 250_000_000
                    and -50_000_000 <= now - row[0] <= 250_000_000
                    and abs(row[0] - camera[0]) <= 100_000_000
                    for row in rows
                ):
                    self.last_camera = camera[0]
                    body = mapped["body"][2]
                    state = np.concatenate(
                        (body[0], mapped["left"][2], mapped["right"][2], body[1])
                    )
                    return G1Frame(
                        camera[2][0],
                        state,
                        {
                            "camera_host_monotonic_ns": camera[0],
                            "camera_g1_monotonic_ns": camera[2][1],
                            "body_index": body[3],
                            "body_dq": body[2].tolist(),
                            "alignment_monotonic_ns": {
                                key: row[0] for key, row in mapped.items()
                            },
                            "alignment_max_ms": max(
                                abs(row[0] - camera[0]) for row in mapped.values()
                            )
                            / 1e6,
                            "after_publication_ns": after_ns,
                        },
                    )
            self.closed.wait(0.005)
        raise TimeoutError(
            "No fresh aligned RGB/body/hand observation; check source interfaces"
        )


# Named G1 observations
# ----------------------------------------------------------------------

STATE_FIELDS = ("body_joints", "left_hand", "right_hand", "projected_gravity")


def unpack(value, image_name="ego"):
    state = dict(zip(STATE_FIELDS, np.split(value.state, [29, 35, 41])))
    if "body_dq" in value.metadata:
        state["body_velocity"] = np.asarray(value.metadata["body_dq"])
    stamp = value.metadata.get("camera_host_monotonic_ns")
    return Observation(
        {image_name: value.rgb},
        state,
        {}
        if stamp is None
        else {
            "anchor": stamp,
            "images." + image_name: stamp,
            **value.metadata.get("alignment_monotonic_ns", {}),
        },
        clock="episode" if stamp is None else "host_monotonic",
        metadata=value.metadata,
    )


class G1Observations:
    def __init__(self, config):
        self.image_name = config["anchor_view"]
        cameras = config["cameras"]
        if (
            set(cameras) != {self.image_name}
            or cameras[self.image_name]["image_key"] != "ego_view"
        ):
            raise ValueError(
                "The supplied G1 source reads one ego_view stream; add a source adapter for other camera layouts"
            )
        self.images = (self.image_name,)
        self.state_fields = (*STATE_FIELDS, "body_velocity")
        self.config = SimpleNamespace(
            body_endpoint=local_endpoint(config["body_endpoint"], "tcp"),
            hand_endpoint=local_endpoint(config["hand_endpoint"], "tcp"),
            camera_endpoint=local_endpoint(cameras[self.image_name]["endpoint"], "tcp"),
        )
        self.stream = None

    def validate(self, policy):
        if not set(policy.required_images) <= set(self.images) or not set(
            policy.required_state
        ) <= set(self.state_fields):
            raise ValueError("Observation source does not provide the policy inputs")

    def read_episode(self, path, frame=0):
        return unpack(read_episode(path, frame), self.image_name)

    def __enter__(self):
        self.stream = LiveObservations(self.config)
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def read(self, **kwargs):
        return unpack(self.stream.read(**kwargs), self.image_name)

    def assert_fresh(self):
        self.stream.assert_fresh()


# Lambda0 HTTP client
# ----------------------------------------------------------------------


class LambdaPolicy:
    required_state = ("body_joints", "left_hand", "right_hand", "projected_gravity")
    output_spec = ActionSpec(
        "sonic_token_revo2_v1",
        {"motion": 64, "left_hand": 6, "right_hand": 6},
        {
            "motion": "latent",
            "left_hand": "normalized_stark",
            "right_hand": "normalized_stark",
        },
        50,
    )

    def __init__(self, config, instruction):
        if config["action_space_id"] != 2 or config["horizon"] != 50:
            raise ValueError(
                "This Lambda0 adapter requires action space 2 and horizon 50"
            )
        self.required_images = tuple(config["image_order"])
        if len(self.required_images) != 1:
            raise ValueError("This Lambda0 checkpoint supports one image only")
        if tuple(config["state_order"]) != self.required_state:
            raise ValueError("Lambda0 G1 state order must match the training layout")
        if type(config["inference_steps"]) is not int or config["inference_steps"] <= 0:
            raise ValueError("inference_steps must be a positive integer")
        if type(config["embodiment_id"]) is not int or config["embodiment_id"] < 0:
            raise ValueError("Invalid embodiment_id")
        if (
            not config["model_path"].strip()
            or not config["normalization_dataset"].strip()
        ):
            raise ValueError("Model and normalization dataset are required")
        self.config = SimpleNamespace(
            **config,
            instruction=instruction,
            policy_url=local_endpoint(config["url"], "http"),
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None):
        request = urllib.request.Request(
            self.config.policy_url.rstrip("/") + path,
            data=None
            if payload is None
            else json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Policy response exceeds limit")
        result = json.loads(raw)
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("Policy returned an error")
        return result

    def check(self):
        health = self.request("/health")
        expected = {
            "status": "ok",
            "model_path": self.config.model_path,
            "normalization_dataset": self.config.normalization_dataset,
            "normalization_action_space_id": self.config.action_space_id,
            "inference_steps": self.config.inference_steps,
            "rtc_enabled": False,
        }
        for key, value in expected.items():
            if health.get(key) != value:
                raise ValueError(f"Policy {key} does not match deployment config")
        for key, value in [
            ("default_embodiment_id", self.config.embodiment_id),
            ("default_action_space_id", self.config.action_space_id),
        ]:
            if health.get(key) not in (None, value):
                raise ValueError(f"Policy {key} mismatch")
        return health

    def predict(self, observation, session, step):
        observation.require(self.required_images, self.required_state)
        sizes = (29, 6, 6, 3)
        if any(
            observation.state[name].shape != (size,)
            for name, size in zip(self.required_state, sizes)
        ):
            raise ValueError(
                "Lambda0 state field dimensions differ from training layout"
            )
        proprio = state_vector(
            np.concatenate([observation.state[name] for name in self.required_state])
        )
        self.check()
        # Match the Lambda0 training image geometry used by the deployed policy.
        image = np.asarray(
            Image.fromarray(observation.images[self.required_images[0]]).resize(
                (448, 448), Image.Resampling.BICUBIC
            )
        )
        payload = {
            "image0": {
                "__numpy__": base64.b64encode(image.tobytes()).decode(),
                "dtype": "|u1",
                "shape": list(image.shape),
            },
            "proprio": proprio.tolist(),
            "language_instruction": self.config.instruction,
            "embodiment_id": self.config.embodiment_id,
            "action_space_id": self.config.action_space_id,
            "steps": self.config.inference_steps,
            "history": {"session_id": session, "step_index": step},
        }
        start = time.monotonic()
        reply = self.request("/act", payload)
        if reply.get("action_space_id") != self.config.action_space_id:
            raise ValueError("Response action-space mismatch")
        action, info = action_chunk(reply["action"], self.config.horizon)
        return ActionBatch(
            self.output_spec,
            {
                "motion": action[:, :64],
                "left_hand": action[:, 64:70],
                "right_hand": action[:, 70:],
            },
        ), {**info, "inference_ms": (time.monotonic() - start) * 1000}


# Signed action messages
# ----------------------------------------------------------------------

PREFIX = b"g1_revo2_policy_v1\0"


def encode_action(action, sequence, identity, secret):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (76,) or not np.isfinite(value).all():
        raise ValueError("Expected finite 76D action")
    body = json.dumps(
        {
            "identity": identity,
            "sequence": sequence,
            "sent_ns": time.monotonic_ns(),
            "action": value.tolist(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return PREFIX + hmac.new(secret.encode(), body, hashlib.sha256).digest() + body


def decode_action(raw, identity, secret, previous):
    if not raw.startswith(PREFIX) or len(raw) > 16384:
        raise ValueError("Unbound Policy output rejected")
    signature, body = raw[len(PREFIX) : len(PREFIX) + 32], raw[len(PREFIX) + 32 :]
    if not hmac.compare_digest(
        signature, hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ):
        raise ValueError("Policy source authentication failed")
    packet = json.loads(body)
    sequence = int(packet["sequence"])
    age = time.monotonic_ns() - int(packet["sent_ns"])
    if (
        packet["identity"] != identity
        or sequence <= previous
        or not 0 <= age <= 250_000_000
    ):
        raise ValueError("Mismatched, replayed or stale Policy action")
    values = np.asarray(packet["action"], dtype=np.float32)
    if (
        values.shape != (76,)
        or not np.isfinite(values).all()
        or np.max(np.abs(values[:64])) > 1.25
    ):
        raise ValueError("Invalid Sonic/Revo2 action")
    if np.any(values[64:] < 0) or np.any(values[64:] > 1):
        raise ValueError("Revo2 target outside [0,1]")
    return values, sequence


# SONIC runtime and publication
# ----------------------------------------------------------------------


def pose_matches(actual, expected, dq):
    actual, expected = state_vector(actual), state_vector(expected)
    error = np.abs((actual[:29] - expected[:29] + np.pi) % (2 * np.pi) - np.pi)
    return bool(
        np.max(error) <= 0.20
        and np.max(np.abs(actual[29:41] - expected[29:41])) <= 0.15
        and np.max(np.abs(actual[41:] - expected[41:])) <= 0.10
        and np.max(np.abs(dq)) <= 0.50
    )


def wait_initial_pose(sensors, expected, timeout=10):
    count = 0
    previous = -1
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = sensors.read(timeout=min(2, max(0.01, deadline - time.monotonic())))
        index = value.metadata["body_index"]
        if index <= previous:
            continue
        previous = index
        count = (
            count + 1
            if pose_matches(value.state, expected, value.metadata["body_dq"])
            else 0
        )
        if count >= 50:
            return
    raise RuntimeError("Initial measured pose did not converge for 50 samples")


class ControllerProcess:
    def __init__(self, pid, binary, robot_ip):
        if pid <= 0:
            raise ValueError("Invalid controller PID")
        self.root = Path("/proc") / str(pid)
        self.binary = Path(binary).resolve(strict=True)
        self.robot_ip = robot_ip
        self.start_identity = self.snapshot()

    def snapshot(self):
        if (self.root / "exe").resolve(strict=True) != self.binary:
            raise RuntimeError(
                "Controller executable does not match the selected binary"
            )
        args = (self.root / "cmdline").read_bytes().decode().strip("\0").split("\0")
        if "--external-token-override" not in args:
            raise RuntimeError("Controller is not explicitly in external-token mode")

        def flag(name):
            for i, arg in enumerate(args):
                if arg == name and i + 1 < len(args):
                    return args[i + 1]
                if arg.startswith(name + "="):
                    return arg.split("=", 1)[1]
            return None

        if flag("--bridge-host") != self.robot_ip:
            raise RuntimeError("Controller robot address mismatch")
        # /proc/stat's comm field can contain spaces: starttime is field 22.
        return (self.root / "stat").read_text().rsplit(")", 1)[1].split()[19]

    def check(self):
        if self.snapshot() != self.start_identity:
            raise RuntimeError("Controller process restarted")


class RouterSession:
    def __init__(self, config, session):
        self.config = config
        self.context = zmq.Context()
        self.publisher = None
        self.identity = {
            "session": session,
            "model": config.model_path,
            "task": config.instruction,
            "manifestSha256": config.digest,
        }
        self.secret = secrets.token_hex(32)
        self.sequence = 0
        self.bound = False

    def request(self, command, **fields):
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        try:
            sock.connect(self.config.router_control)
            sock.send_json({"command": command, **fields})
            response = sock.recv_json()
            if response.get("ok") is not True:
                raise RuntimeError(
                    response.get("error", "Motion router rejected request")
                )
            return response
        finally:
            sock.close()

    def ready(self):
        value = self.request("status")
        if (
            value.get("phase") != "hold"
            or value.get("stable") is not True
            or value.get("policy_enabled")
        ):
            raise RuntimeError(
                "Initialize the robot in the runtime first; expected unowned stable Hold"
            )

    def enable(self):
        self.ready()
        self.request("policy-bind", identity=self.identity, secret=self.secret)
        self.bound = True
        deadline = time.monotonic() + 3
        while not self.request("status").get("hand_command_ready"):
            if time.monotonic() > deadline:
                raise RuntimeError("Hand command subscribers are not ready")
            time.sleep(0.02)
        self.publisher = self.context.socket(zmq.PUSH)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.setsockopt(zmq.SNDHWM, 2)
        self.publisher.setsockopt(zmq.SNDTIMEO, 100)
        self.publisher.connect(self.config.router_policy)
        response = self.request(
            "policy-enable",
            identity=self.identity,
            secret=self.secret,
            allow_missing_hands=False,
            body_only=False,
        )
        if response.get("hand_control_enabled") is not True:
            raise RuntimeError("Hand output was not enabled")

    def check(self):
        value = self.request("status")
        if not value.get("policy_enabled") or not value.get("hand_control_enabled"):
            raise RuntimeError("Policy or hand output disabled")

    def send_chunk(self, actions, sensors, controller):
        deadline = time.monotonic()
        last_ns = None
        for index, action in enumerate(actions[: self.config.execute_steps]):
            sensors.assert_fresh()
            controller.check()
            if index % 10 == 0:
                self.check()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if time.monotonic() - deadline > 0.05:
                raise RuntimeError(
                    "Action sender missed its deadline; refusing to burst queued actions"
                )
            sensors.assert_fresh()
            self.publisher.send(
                encode_action(action, self.sequence, self.identity, self.secret)
            )
            last_ns = time.monotonic_ns()
            self.sequence += 1
            deadline = time.monotonic() + 0.02
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return last_ns

    def close(self):
        try:
            if self.bound:
                result = self.request(
                    "policy-disable", identity=self.identity, secret=self.secret
                )
                if result.get("policy_enabled") is not False:
                    raise RuntimeError("Policy disable was not confirmed")
                self.request(
                    "policy-unbind", identity=self.identity, secret=self.secret
                )
                self.bound = False
        finally:
            if self.publisher is not None:
                self.publisher.close()
            self.context.term()


# Controller lifecycle
# ----------------------------------------------------------------------


def pose(observation):
    fields = ("body_joints", "left_hand", "right_hand", "projected_gravity")
    observation.require((), fields)
    return np.concatenate([observation.state[name] for name in fields])


class PoseSource:
    def __init__(self, source):
        self.source = source

    def read(self, **kwargs):
        value = self.source.read(**kwargs)
        return SimpleNamespace(state=pose(value), metadata=value.metadata)


class SonicController:
    input_spec = ActionSpec(
        "sonic_token_revo2_v1",
        {"motion": 64, "left_hand": 6, "right_hand": 6},
        {
            "motion": "latent",
            "left_hand": "normalized_stark",
            "right_hand": "normalized_stark",
        },
        50,
    )

    def __init__(self, config):
        self.config = config
        local_endpoint(config["control_endpoint"], "tcp")
        local_endpoint(config["policy_endpoint"], "tcp")
        self.router = None
        self.enabled = False
        self.last_publication_ns = None

    @property
    def published_steps(self):
        return 0 if self.router is None else self.router.sequence

    def prepare(self, source, reference, deployment, session):
        if reference is None:
            raise ValueError("--execute requires --initial-episode")
        if reference.metadata.get("task") != deployment["instruction"]:
            raise ValueError("Initial episode task differs from instruction")
        self.expected = pose(reference)
        cfg = self.config
        if type(cfg.get("pid")) is not int:
            raise ValueError("Set controller.pid before execution")
        self.process = ControllerProcess(cfg["pid"], cfg["binary"], cfg["robot_ip"])
        router_config = SimpleNamespace(
            router_control=cfg["control_endpoint"],
            router_policy=cfg["policy_endpoint"],
            model_path=deployment["policy"].get(
                "model_path", deployment["policy"]["type"]
            ),
            instruction=deployment["instruction"],
            digest=hashlib.sha256(
                json.dumps(deployment, sort_keys=True).encode()
            ).hexdigest(),
            execute_steps=deployment["schedule"]["execute_steps"],
        )
        self.router = RouterSession(router_config, session)
        self.router.ready()
        wait_initial_pose(PoseSource(source), self.expected)

    def execute(self, batch, source, observation, steps):
        if batch.spec != self.input_spec:
            raise ValueError("Incompatible action contract")
        if not 0 < steps <= len(batch):
            raise ValueError("execute_steps exceeds action horizon")
        actions = np.concatenate(
            [batch.values[name] for name in ("motion", "left_hand", "right_hand")],
            axis=1,
        )
        if (
            np.max(np.abs(actions[:, :64])) > 1.25
            or np.any(actions[:, 64:] < 0)
            or np.any(actions[:, 64:] > 1)
        ):
            raise ValueError("Action exceeds SONIC/Revo2 bounds")
        self.process.check()
        source.assert_fresh()

        def check_age():
            if (
                observation.clock != "host_monotonic"
                or "anchor" not in observation.timestamps
            ):
                raise ValueError(
                    "SONIC requires a calibrated host-monotonic observation"
                )
            if (
                not 0
                <= time.monotonic_ns() - observation.timestamps["anchor"]
                <= 1_000_000_000
            ):
                raise RuntimeError(
                    "Observation expired during inference or output preparation"
                )

        check_age()
        if not self.enabled:
            fresh = source.read()
            if not pose_matches(
                pose(fresh), self.expected, fresh.state["body_velocity"]
            ):
                raise RuntimeError("Robot left its initial pose during inference")
            self.router.enable()
            self.enabled = True
        check_age()
        self.last_publication_ns = None
        self.router.config.execute_steps = steps
        self.last_publication_ns = self.router.send_chunk(actions, source, self.process)
        return self.last_publication_ns

    def close(self):
        if self.router is not None:
            self.router.close()


# Command-line interface and inference loop
# ----------------------------------------------------------------------

SOURCES = {"g1": G1Observations}
POLICIES = {"lambda0": LambdaPolicy}
CONTROLLERS = {"sonic": SonicController}


def build(config):
    source = SOURCES[config["observation"]["type"]](config["observation"])
    policy = POLICIES[config["policy"]["type"]](config["policy"], config["instruction"])
    controller = CONTROLLERS[config["controller"]["type"]](config["controller"])
    require_compatible(policy, controller)
    source.validate(policy)
    return source, policy, controller


def save_prediction(output, index, observation, batch, info):
    arrays = {f"state.{key}": value for key, value in observation.state.items()}
    arrays.update({f"action.{key}": value for key, value in batch.values.items()})
    np.savez_compressed(output / f"chunk-{index:04d}.npz", **arrays)
    for name, image in observation.images.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                "Image names must contain only letters, numbers, underscores or hyphens"
            )
        Image.fromarray(image).save(output / f"chunk-{index:04d}-{name}.jpg")
    write_info(output, index, observation, batch, info)


def write_info(output, index, observation, batch, info):
    (output / f"chunk-{index:04d}.json").write_text(
        json.dumps(
            {
                "observation": observation.metadata,
                "timestamps": observation.timestamps,
                "clock": observation.clock,
                "action_spec": asdict(batch.spec),
                **info,
            },
            indent=2,
        )
    )


def live(config, source, policy, controller, args, session):
    reference = (
        source.read_episode(args.initial_episode, 0)
        if args.execute and args.initial_episode
        else None
    )
    try:
        with source:
            if args.execute:
                controller.prepare(source, reference, config, session)
            barrier = None
            for index in range(args.chunks):
                observation = source.read(after_ns=barrier)
                observation.require(policy.required_images, policy.required_state)
                batch, info = policy.predict(observation, session, index)
                if batch.spec != policy.output_spec:
                    raise ValueError("Policy returned an undeclared action contract")
                steps = config["schedule"]["execute_steps"]
                if steps > len(batch):
                    raise ValueError("execute_steps exceeds returned action horizon")
                info = {
                    **info,
                    "executed_steps": 0,
                    "last_publication_monotonic_ns": None,
                }
                save_prediction(args.output, index, observation, batch, info)
                if args.execute:
                    before = controller.published_steps
                    try:
                        barrier = controller.execute(batch, source, observation, steps)
                    finally:
                        info.update(
                            executed_steps=controller.published_steps - before,
                            last_publication_monotonic_ns=controller.last_publication_ns,
                        )
                        write_info(args.output, index, observation, batch, info)
                print(
                    f"Chunk {index}: " + ("sent" if args.execute else "preview"),
                    flush=True,
                )
    finally:
        if args.execute:
            controller.close()


def main():
    parser = argparse.ArgumentParser(description="OpenLambda inference")
    parser.add_argument("--config", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate adapters and model service")
    predict = sub.add_parser("predict", help="predict from a recorded episode")
    predict.add_argument("--episode", required=True, type=Path)
    predict.add_argument("--frame", type=int, default=0)
    predict.add_argument("--output", required=True, type=Path)
    run = sub.add_parser("run", help="preview live predictions; opt in to execution")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--chunks", type=int, default=1)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--initial-episode", type=Path)
    args = parser.parse_args()

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        config = load(args.config)
        source, policy, controller = build(config)
        if args.command == "run":
            if args.chunks <= 0:
                raise ValueError("chunks must be positive")
            if args.execute and args.initial_episode is None:
                raise ValueError("--execute requires --initial-episode")
        policy.check()
        if args.command == "check":
            print("Adapter contracts and model service match the configuration.")
            return
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "deployment.json").write_text(json.dumps(config, indent=2))
        session = uuid.uuid4().hex
        status = {"session": session, "status": "started"}
        (args.output / "session.json").write_text(json.dumps(status))
        try:
            if args.command == "predict":
                observation = source.read_episode(args.episode, args.frame)
                if observation.metadata.get("task") != config["instruction"]:
                    raise ValueError("Episode task differs from instruction")
                batch, info = policy.predict(observation, session, 0)
                if batch.spec != policy.output_spec:
                    raise ValueError("Unexpected action contract")
                save_prediction(args.output, 0, observation, batch, info)
            else:
                live(config, source, policy, controller, args, session)
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="interrupted", error=str(error))
            raise
        finally:
            (args.output / "session.json").write_text(json.dumps(status, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. The runtime may still hold its last target.\n")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()

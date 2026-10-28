# SPDX-License-Identifier: Apache-2.0
"""Decode and validate the real-time sources used by the G1 Episode agent."""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import msgpack

ROBOT_TOPIC = b"g1_debug"
CONFIG_TOPIC = b"robot_config"
LEFT_STATE_TOPIC = b"rt/brainco/left/state"
RIGHT_STATE_TOPIC = b"rt/brainco/right/state"
APPLIED_HAND_TOPIC = b"rt/brainco/glove/applied"


class SourceDecodeError(ValueError):
    """A source packet is malformed or unsafe to record."""


def _finite_vector(value: object, length: int, bound: float, label: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SourceDecodeError("{} must contain {} values".format(label, length))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) and abs(item) <= bound for item in result):
        raise SourceDecodeError("{} contains an invalid value".format(label))
    return result


def _six_ints(value: object, label: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise SourceDecodeError("{} must contain six values".format(label))
    return [int(item) for item in value]


def _six_targets(value: object, label: str) -> list:
    result = _finite_vector(value, 6, 1.0, label)
    if any(item < 0.0 for item in result):
        raise SourceDecodeError("{} contains a negative target".format(label))
    return result


def _unit_quaternion(value: object, label: str) -> list:
    result = _finite_vector(value, 4, 1.1, label)
    norm = math.sqrt(sum(item * item for item in result))
    if not 0.9 <= norm <= 1.1:
        raise SourceDecodeError("{} norm is invalid: {:.6g}".format(label, norm))
    return result


def decode_sonic(
    raw: bytes,
    g1_minus_host_monotonic_ns: int,
    g1_minus_host_realtime_ns: int,
) -> Tuple[str, Dict[str, object]]:
    """Decode one compact Sonic config or robot-state packet."""
    now_monotonic_ns = time.monotonic_ns()
    now_realtime_ns = time.time_ns()
    if raw.startswith(CONFIG_TOPIC):
        value = msgpack.unpackb(raw[len(CONFIG_TOPIC) :], raw=False)
        if not isinstance(value, dict):
            raise SourceDecodeError("Sonic robot_config is not an object")
        return "config", {
            "type": "robot_config",
            "g1_received_monotonic_ns": now_monotonic_ns,
            "g1_received_realtime_ns": now_realtime_ns,
            "value": value,
        }
    if not raw.startswith(ROBOT_TOPIC):
        raise SourceDecodeError("unknown Sonic topic")
    value = msgpack.unpackb(raw[len(ROBOT_TOPIC) :], raw=False)
    if not isinstance(value, dict):
        raise SourceDecodeError("Sonic feedback is not an object")
    index = int(value.get("index", -1))
    source_monotonic_ns = int(value.get("timestamp_monotonic_ns", 0))
    source_realtime_ns = int(value.get("timestamp_realtime_ns", 0))
    if index < 0 or source_monotonic_ns <= 0 or source_realtime_ns <= 0:
        raise SourceDecodeError("Sonic feedback has no valid sequence or source clocks")
    return "robot", {
        "type": "sonic",
        "index": index,
        "g1_received_monotonic_ns": now_monotonic_ns,
        "g1_received_realtime_ns": now_realtime_ns,
        "source_monotonic_ns": source_monotonic_ns,
        "source_realtime_ns": source_realtime_ns,
        "source_g1_monotonic_ns": (source_monotonic_ns + g1_minus_host_monotonic_ns),
        "source_g1_realtime_ns": source_realtime_ns + g1_minus_host_realtime_ns,
        "body_q": _finite_vector(value.get("body_q"), 29, 20.0, "body_q"),
        "body_dq": _finite_vector(value.get("body_dq"), 29, 200.0, "body_dq"),
        "token_state": _finite_vector(
            value.get("token_state"), 64, 32.0, "token_state"
        ),
        "base_quat": _unit_quaternion(value.get("base_quat"), "base_quat"),
    }


def decode_brainco(frames: list) -> Tuple[str, Dict[str, object]]:
    """Decode actual hand feedback or the command applied by the glove controller."""
    if len(frames) != 2:
        raise SourceDecodeError("BrainCo telemetry must contain two ZMQ frames")
    topic, payload = frames
    if topic not in {LEFT_STATE_TOPIC, RIGHT_STATE_TOPIC, APPLIED_HAND_TOPIC}:
        raise SourceDecodeError("unknown BrainCo telemetry topic")
    value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    if not isinstance(value, dict):
        raise SourceDecodeError("BrainCo telemetry is not an object")
    source_monotonic_ns = int(value.get("monotonic_ns", 0))
    source_realtime_ns = int(value.get("wall_time_ns", 0))
    sequence = int(value.get("sequence", -1))
    if sequence < 0 or source_monotonic_ns <= 0 or source_realtime_ns <= 0:
        raise SourceDecodeError(
            "BrainCo telemetry has no valid sequence or source clocks"
        )
    if topic == APPLIED_HAND_TOPIC:
        return "hand_action", {
            "type": "brainco_action",
            "sequence": sequence,
            "g1_received_monotonic_ns": time.monotonic_ns(),
            "g1_received_realtime_ns": time.time_ns(),
            "source_monotonic_ns": source_monotonic_ns,
            "source_realtime_ns": source_realtime_ns,
            "left": _six_targets(value.get("left"), "left BrainCo action"),
            "right": _six_targets(value.get("right"), "right BrainCo action"),
        }
    side = "left" if topic == LEFT_STATE_TOPIC else "right"
    return side, {
        "type": "brainco",
        "side": side,
        "sequence": sequence,
        "g1_received_monotonic_ns": time.monotonic_ns(),
        "g1_received_realtime_ns": time.time_ns(),
        "source_monotonic_ns": source_monotonic_ns,
        "source_realtime_ns": source_realtime_ns,
        "positions": _six_ints(value.get("positions"), "BrainCo positions"),
        "speeds": _six_ints(value.get("speeds"), "BrainCo speeds"),
        "currents": _six_ints(value.get("currents"), "BrainCo currents"),
        "states": _six_ints(value.get("states"), "BrainCo states"),
    }

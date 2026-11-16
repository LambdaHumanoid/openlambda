#!/usr/bin/env python3
"""Loopback, ownership-gated ROS2 adapter for normalized STARK hand targets.

This does not start the STARK driver or enable motors. A live driver and genuine
feedback from both hands are required. Stopping publication is NOT torque-off.
"""
from __future__ import annotations

import argparse
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import signal
import time

import zmq

NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
MAX_AGE_NS = 250_000_000


def values12(value):
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError("Expected twelve normalized STARK targets")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value):
        raise ValueError("Hand targets must be finite normalized STARK values in [0,1]")
    return value


def request(context, endpoint, payload, timeout=100):
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, timeout)
    sock.setsockopt(zmq.RCVTIMEO, timeout)
    try:
        sock.connect(endpoint)
        sock.send_json(payload)
        result = sock.recv_json()
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(result.get("error", "Hand request rejected"))
        return result
    finally:
        sock.close()


class HandLease:
    """Receiver state machine; callbacks are the real ROS readiness/write boundary."""
    def __init__(self, ready, publish, allow=False):
        self.ready, self.publish, self.allow = ready, publish, allow
        self.owner = None
        self.sequence = -1
        self.last_seen = 0
        self.clock_offset = 0

    def expire(self, now):
        if self.owner is not None and now - self.last_seen > MAX_AGE_NS:
            self.owner = None

    def handle(self, data, now=None):
        now = time.monotonic_ns() if now is None else now
        self.expire(now)
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        command = data.get("command")
        ready = self.ready()
        if command == "status":
            return {"ok": True, "protocol": "openlambda_stark_v1", "units": "normalized_stark",
                    "ready": ready, "allow_actuation": self.allow, "owned": self.owner is not None,
                    "monotonic_ns": now}
        owner = data.get("owner")
        if not isinstance(owner, str) or len(owner) < 32:
            raise ValueError("Missing hand owner credential")
        if command == "bind":
            if not self.allow or not ready or self.owner is not None:
                raise ValueError("Hand bind requires enabled, fresh, unowned ROS backend")
            sent = data.get("sent_ns")
            if type(sent) is not int:
                raise ValueError("Missing host clock sample")
            self.owner, self.sequence, self.last_seen = owner, -1, now
            self.clock_offset = sent - now
            return {"ok": True, "monotonic_ns": now}
        if self.owner is None or not hmac.compare_digest(owner, self.owner):
            raise ValueError("Hand owner mismatch or expired lease")
        if command == "unbind":
            self.owner = None
            return {"ok": True}
        sequence, sent = data.get("sequence"), data.get("sent_ns")
        if type(sequence) is not int or sequence <= self.sequence or type(sent) is not int:
            raise ValueError("Hand sequence replay or malformed timestamp")
        age = now + self.clock_offset - sent
        if not -50_000_000 <= age <= MAX_AGE_NS:
            raise ValueError("Stale hand request or clock discontinuity")
        if not self.allow or not ready:
            self.owner = None
            raise ValueError("Hand feedback/command subscriber unavailable")
        if command == "action":
            targets = values12(data.get("action"))
            self.publish(targets)
        elif command != "heartbeat":
            raise ValueError("Unknown hand command")
        self.sequence, self.last_seen = sequence, now
        return {"ok": True, "sequence": sequence, "monotonic_ns": now}


def credential_path(port):
    return Path.home() / ".local/state/openlambda" / ("hands-%d.json" % port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("serve", "status", "stop"))
    parser.add_argument("--port", type=int, default=18561)
    parser.add_argument("--allow-actuation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    endpoint = "tcp://127.0.0.1:%d" % args.port
    context = zmq.Context()
    path = credential_path(args.port)
    if args.operation != "serve":
        try:
            payload = {"command": args.operation}
            if args.operation == "stop":
                payload["admin"] = json.loads(path.read_text())["admin"]
            print(json.dumps(request(context, endpoint, payload, 1000)))
            return 0
        except (OSError, ValueError, RuntimeError, zmq.ZMQError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1
        finally:
            context.term()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from ros2_stark_interfaces.msg import MotorStatus

    rclpy.init()
    node = Node("openlambda_normalized_hands")
    feedback = {}
    publishers = {}
    subscriptions = []
    for side, topic in (("left", "/motor_status"), ("right", "/motor_status_r")):
        if args.allow_actuation:
            publishers[side] = node.create_publisher(JointState, "/joint_commands_" + side, 1)

        def receive(message, side=side):
            positions = list(message.positions)
            if len(positions) == 6 and all(0 <= v <= 1000 for v in positions):
                feedback[side] = time.monotonic_ns()

        subscriptions.append(node.create_subscription(MotorStatus, topic, receive, qos_profile_sensor_data))

    def ready():
        now = time.monotonic_ns()
        return bool(args.allow_actuation and all(
            side in feedback and 0 <= now - feedback[side] <= MAX_AGE_NS
            and publishers[side].get_subscription_count() > 0
            and node.count_publishers("/joint_commands_" + side) == 1
            for side in ("left", "right")))

    def publish(values):
        for side, start in (("left", 0), ("right", 6)):
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            message.name = list(NAMES)
            # STARK's callback multiplies normalized values by 100, exactly once.
            message.position = [float(v) for v in values[start:start + 6]]
            publishers[side].publish(message)

    lease = HandLease(ready, publish, args.allow_actuation)
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.setsockopt(zmq.MAXMSGSIZE, 16384)
    admin = secrets.token_hex(32)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    owned_file = False
    try:
        server.bind(endpoint)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            json.dump({"admin": admin, "pid": os.getpid()}, stream)
        owned_file = True
        while not stopped:
            rclpy.spin_once(node, timeout_sec=0.005)
            lease.expire(time.monotonic_ns())
            if not server.poll(0):
                continue
            try:
                data = server.recv_json()
                if isinstance(data, dict) and data.get("command") == "stop":
                    supplied = data.get("admin")
                    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, admin):
                        raise ValueError("Administrator credential required")
                    lease.owner = None
                    stopped = True
                    result = {"ok": True, "publication_stopped": True, "torque_off": False}
                else:
                    result = lease.handle(data)
            except (ValueError, TypeError, KeyError) as exc:
                result = {"ok": False, "error": str(exc)}
            server.send_json(result)
    finally:
        server.close()
        context.term()
        node.destroy_node()
        rclpy.shutdown()
        if owned_file:
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

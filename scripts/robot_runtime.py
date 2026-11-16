#!/usr/bin/env python3
"""Terminal-only signed Lambda0 receiver with explicit robot initialization.

serve/prepare never start SONIC or enable motors. initialize --execute is an
explicit motion/start request; it requires the patched, operator-started backend.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import struct
import time
from urllib.parse import urlsplit

import msgpack
import numpy as np
import zmq

from infer_robot import decode_action
from robot_hands import request as hand_request

AGE = 250_000_000
ACK = struct.Struct("<8sQQQB64f")
COMMAND = struct.Struct("<8sBQQQ64f")


def loopback(endpoint):
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname != "127.0.0.1" or not parsed.port:
        raise ValueError("Runtime endpoints must be tcp://127.0.0.1:PORT")
    return endpoint


def configuration(path):
    value = json.loads(Path(path).read_text())
    controller = value["controller"]
    return {"control": loopback(controller["control_endpoint"]),
            "policy": loopback(controller["policy_endpoint"]),
            "backend": loopback(controller.get("runtime_endpoint", "tcp://127.0.0.1:5686")),
            "hands": loopback(controller.get("hand_command_endpoint", "tcp://127.0.0.1:18561")),
            "action": loopback(controller.get("action_endpoint", "tcp://127.0.0.1:5656")),
            "body": loopback(value["observation"]["body_endpoint"])}


def credential_path(config):
    name = hashlib.sha256(config["control"].encode()).hexdigest()[:16]
    return Path.home() / ".local/state/openlambda" / ("runtime-" + name + ".json")


def control_request(config, command, **fields):
    context = zmq.Context()
    try:
        return hand_request(context, config["control"], {"command": command, **fields}, 1000)
    finally:
        context.term()


def fresh(stamp, now):
    return type(stamp) is int and 0 <= now - stamp <= AGE

def wire_packet(topic, fields, payload, version):
    header = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields},
                        separators=(",", ":")).encode().ljust(1280, b"\0")
    return topic + header + payload


def seed_packet(token, frame):
    return wire_packet(b"pose", [
        {"name": "token_state", "dtype": "f32", "shape": [1, 64]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]}],
        np.asarray(token, dtype="<f4").tobytes() + struct.pack("<q", frame), 4)


def start_packet():
    return wire_packet(b"command", [
        {"name": key, "dtype": "u8", "shape": [1]}
        for key in ("start", "stop", "planner")], bytes((1, 0, 0)), 1)



class Runtime:
    def __init__(self, config, allow=False):
        self.config, self.allow = config, allow
        self.context = zmq.Context()
        self.sockets = []
        self.control = self.socket(zmq.REP)
        self.policy = self.socket(zmq.PULL)
        self.backend = self.socket(zmq.PAIR)
        self.body = self.socket(zmq.SUB)
        self.action = self.socket(zmq.PUB)
        self.action.setsockopt(zmq.SNDHWM, 2)
        self.body.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        self.body.setsockopt(zmq.CONFLATE, 1)
        self.backend.setsockopt(zmq.RCVHWM, 4)
        self.backend.setsockopt(zmq.SNDHWM, 4)
        self.policy.setsockopt(zmq.RCVHWM, 4)
        self.policy.setsockopt(zmq.MAXMSGSIZE, 16384)
        self.control.setsockopt(zmq.MAXMSGSIZE, 16384)
        self.owner = secrets.randbits(63) + 1
        self.admin = secrets.token_hex(32)
        self.hand_owner = None
        self.hand_sequence = 0
        self.backend_sequence = 0
        self.last_sent_sequence = 0
        self.last_ack_progress_ns = 0
        self.last_ack_sequence = -1
        self.feedback = None
        self.body_feedback = None
        self.hand_feedback = {}
        self.hand_received = 0
        self.last_hand_poll = 0
        self.identity = self.secret = None
        self.sequence = -1
        self.enabled_ns = 0
        self.policy_enabled = False
        self.last_policy_ns = 0
        self.hold_token = None
        self.hold_started = 0
        self.seed_token = None
        self.seed_frame = 0
        self.phase = "monitor"
        self.detail = "No actuation; prepare and explicitly request hold --execute"
        self.stable_samples = deque(maxlen=100)
        self.stable = False
        self.closed = False
        self.rejected = 0
        self.accepted = 0
        self.file_owned = False

    def socket(self, kind):
        sock = self.context.socket(kind)
        sock.setsockopt(zmq.LINGER, 0)
        self.sockets.append(sock)
        return sock

    def open(self):
        self.control.bind(self.config["control"])
        self.policy.bind(self.config["policy"])
        self.action.bind(self.config["action"])
        self.backend.connect(self.config["backend"])
        self.body.connect(self.config["body"])
        path = credential_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            json.dump({"admin": self.admin, "pid": os.getpid()}, stream)
        self.file_owned = True

    def backend_ready(self, now):
        return bool(self.feedback and fresh(self.feedback["stamp"], now)
                    and fresh(self.feedback["received"], now) and self.feedback["flags"] & 1)

    def backend_waiting(self, now):
        return bool(self.feedback and fresh(self.feedback["stamp"], now)
                    and fresh(self.feedback["received"], now) and self.feedback["flags"] & 16)

    def body_ready(self, now):
        return bool(self.body_feedback and fresh(self.body_feedback["stamp"], now)
                    and fresh(self.body_feedback["received"], now))

    def owned(self, now):
        return bool(self.backend_ready(now) and self.feedback["owner"] == self.owner
                    and fresh(self.last_ack_progress_ns, now))

    def hands_ready(self, now):
        data = self.hand_feedback
        return bool(fresh(self.hand_received, now) and data.get("protocol") == "openlambda_stark_v1"
                    and data.get("units") == "normalized_stark" and data.get("ready") is True
                    and data.get("allow_actuation") is True)

    def status(self):
        now = time.monotonic_ns()
        stable = self.stable and self.owned(now) and self.body_ready(now)
        return {"ok": True, "protocol": "openlambda_runtime_v1", "phase": self.phase,
                "stable": bool(stable), "policy_enabled": self.policy_enabled,
                "policy_active": bool(self.policy_enabled and fresh(self.last_policy_ns, now)),
                "hand_command_ready": self.hands_ready(now),
                "hand_control_enabled": bool(self.policy_enabled and self.hand_owner and self.hands_ready(now)),
                "controller_ready": self.backend_ready(now), "body_feedback_ready": self.body_ready(now),
                "controller_waiting": self.backend_waiting(now),
                "initialization_ready": bool(self.allow and self.backend_waiting(now) and self.hands_ready(now)),
                "backend_owned": self.owned(now), "session_bound": self.identity is not None,
                "allow_actuation": self.allow, "detail": self.detail,
                "accepted_packets": self.accepted, "rejected_packets": self.rejected,
                "torque_off": False}

    def send_backend(self, kind, token=None):
        self.backend_sequence += 1
        values = token if token is not None else self.hold_token
        if values is None:
            raise RuntimeError("No measured Hold token")
        raw = COMMAND.pack(b"OLCMD001", kind, self.owner, self.backend_sequence,
                           time.monotonic_ns(), *values)
        self.backend.send(raw, zmq.NOBLOCK)
        self.last_sent_sequence = self.backend_sequence

    def hand(self, command, **fields):
        payload = {"command": command, **fields}
        if self.hand_owner:
            payload["owner"] = self.hand_owner
        if command not in ("status", "unbind"):
            self.hand_sequence += 1
            payload.update(sequence=self.hand_sequence, sent_ns=time.monotonic_ns())
        start = time.monotonic_ns()
        response = hand_request(self.context, self.config["hands"], payload, 50)
        if time.monotonic_ns() - start > 50_000_000:
            raise RuntimeError("Hand transport exceeded 50 ms budget")
        return response

    def authorized(self, data):
        return bool(self.identity is not None and data.get("identity") == self.identity
                    and isinstance(data.get("secret"), str)
                    and hmac.compare_digest(data["secret"], self.secret))

    def require_admin(self, data):
        if not isinstance(data.get("admin"), str) or not hmac.compare_digest(data["admin"], self.admin):
            raise ValueError("Runtime administrator credential required")

    def disable(self, detail):
        self.policy_enabled = False
        self.last_policy_ns = 0
        self.seed_token = None
        self.stable = False
        self.stable_samples.clear()
        self.detail = detail
        if self.hold_token is not None:
            self.phase = "holding"
            try:
                self.send_backend(5)
            except zmq.Again:
                self.phase = "fault"
                self.detail = "Backend unavailable; independent controller watchdog must retain baseline"
        else:
            self.phase = "monitor"

    def release_hands(self):
        if self.hand_owner:
            try:
                self.hand("unbind")
            finally:
                self.hand_owner = None

    def handle(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected control object")
        command = data.get("command")
        if command in ("status", "prepare"):
            return self.status()
        now = time.monotonic_ns()
        if command == "initialize":
            self.require_admin(data)
            if not self.allow or data.get("execute") is not True:
                raise ValueError("Initialization requires serve --allow-actuation and initialize --execute")
            if self.identity is not None or self.hold_token is not None or self.seed_token is not None:
                raise ValueError("Runtime already initialized or owned")
            if not self.backend_waiting(now) or not self.hands_ready(now):
                raise ValueError("Patched external-decoder WAIT_FOR_CONTROL and both hands required")
            token = np.asarray(data.get("token"), dtype=np.float32)
            if token.shape != (64,) or not np.isfinite(token).all() or np.max(np.abs(token)) > 1.25:
                raise ValueError("Expected finite bounded checkpoint-specific seed token[64]")
            self.seed_token = token
            self.hold_started = now
            self.phase = "initializing"
            self.detail = "Explicit operator initialization: publishing seed and controller start"
            return self.status()
        if command == "hold":
            self.require_admin(data)
            if not self.allow or data.get("execute") is not True:
                raise ValueError("Hold requires serve --allow-actuation and hold --execute")
            if self.identity is not None or self.hold_token is not None:
                raise ValueError("Hold is already owned; stop this runtime before recapturing")
            if not self.backend_ready(now) or not self.body_ready(now) or self.feedback["owner"]:
                raise ValueError("Fresh unowned successful CONTROL/external-decoder feedback required")
            self.hold_token = self.feedback["token"].copy()
            self.hold_started = now
            self.phase = "acquiring"
            self.detail = "Awaiting owned applied Hold acknowledgement and measured stability"
            self.send_backend(1)
            return self.status()
        if command == "stop":
            self.require_admin(data)
            self.disable("Stopping receiver; body baseline retained, hand publication stopped; NOT torque-off")
            self.release_hands()
            self.identity = self.secret = None
            self.closed = True
            return {**self.status(), "publication_stopped": True}
        if command == "policy-bind":
            if not self.allow or self.identity is not None or self.policy_enabled or self.phase != "hold" or not self.status()["stable"]:
                raise ValueError("Policy bind requires unowned, acknowledged stable Hold")
            identity, secret = data.get("identity"), data.get("secret")
            if not isinstance(identity, dict) or set(identity) != {"session", "model", "task", "manifestSha256"}:
                raise ValueError("Invalid policy identity")
            if not all(isinstance(v, str) and v for v in identity.values()):
                raise ValueError("Invalid policy identity values")
            digest = identity["manifestSha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid manifest digest")
            if not isinstance(secret, str) or len(secret) < 32:
                raise ValueError("Invalid ephemeral policy secret")
            if not self.hands_ready(now):
                raise ValueError("Both normalized-STARK hands must be ready")
            self.hand_owner = secrets.token_hex(32)
            try:
                self.hand("bind")
            except Exception:
                self.hand_owner = None
                raise
            self.identity, self.secret, self.sequence = identity.copy(), secret, -1
            return {"ok": True, "identity": self.identity}
        if command not in ("policy-enable", "policy-disable", "policy-unbind"):
            raise ValueError("Unknown command; robot start/reset/planner commands are not supported")
        if not self.authorized(data):
            raise ValueError("Policy owner authentication failed")
        if command == "policy-enable":
            if self.policy_enabled:
                raise ValueError("Policy is already enabled")
            if not self.status()["stable"] or not self.hands_ready(now) or not self.owned(now):
                raise ValueError("Fresh acknowledged Hold and both hands required")
            if data.get("body_only") is True or data.get("allow_missing_hands") is True:
                raise ValueError("Body-only and missing-hand bypasses are forbidden")
            # Drain a bounded queue; timestamp barrier rejects any remaining old packets.
            for _ in range(32):
                try:
                    self.policy.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
            self.hand("heartbeat")
            self.enabled_ns = time.monotonic_ns()
            self.policy_enabled = True
            self.phase = "policy-waiting"
            self.detail = "Authenticated policy enabled; baseline Hold until fresh output"
        elif command == "policy-disable":
            self.disable("Policy disabled; returning to acknowledged baseline Hold")
        else:
            if self.policy_enabled:
                raise ValueError("Disable policy before unbinding")
            self.release_hands()
            self.identity = self.secret = None
        return self.status()

    def poll_feedback(self):
        now = time.monotonic_ns()
        for _ in range(8):
            try:
                raw = self.backend.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(raw) != ACK.size:
                continue
            magic, stamp, owner, sequence, flags, *token = ACK.unpack(raw)
            token = np.asarray(token, dtype=np.float32)
            if magic != b"OLACK001" or not fresh(stamp, now) or not np.isfinite(token).all() or np.max(np.abs(token)) > 1.25:
                continue
            if self.feedback and stamp <= self.feedback["stamp"]:
                continue
            self.feedback = dict(stamp=stamp, received=now, owner=owner, sequence=sequence, flags=flags, token=token)
            if owner == self.owner and sequence > self.last_ack_sequence and sequence <= self.last_sent_sequence:
                self.last_ack_sequence, self.last_ack_progress_ns = sequence, now
        try:
            raw = self.body.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            if not raw.startswith(b"g1_debug"):
                return
            data = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
            stamp = data["timestamp_monotonic_ns"]
            dq, quat = np.asarray(data["body_dq"], dtype=float), np.asarray(data["base_quat"], dtype=float)
            if not fresh(stamp, now) or dq.shape != (29,) or quat.shape != (4,) or not np.isfinite(dq).all() or not np.isfinite(quat).all():
                return
            if self.body_feedback and stamp <= self.body_feedback["stamp"]:
                return
            norm = np.linalg.norm(quat)
            if not 0.9 <= norm <= 1.1:
                return
            quat /= norm
            self.body_feedback = dict(stamp=stamp, received=now)
            quiet = np.max(np.abs(dq)) <= 0.5 and 1 - 2 * (quat[1] ** 2 + quat[2] ** 2) >= 0.95
            matched = bool(self.hold_token is not None and self.owned(now) and self.feedback["flags"] & 4
                           and self.feedback["stamp"] > self.hold_started
                           and np.allclose(self.feedback["token"], self.hold_token, atol=1e-5, rtol=0))
            if quiet and matched and not self.policy_enabled:
                if self.stable_samples and stamp - self.stable_samples[-1] > 100_000_000:
                    self.stable_samples.clear()
                self.stable_samples.append(stamp)
                self.stable = len(self.stable_samples) >= 40 and stamp - self.stable_samples[0] >= 1_000_000_000
                if self.stable:
                    self.phase, self.detail = "hold", "Owned applied baseline Hold with one second of fresh measured stability"
            else:
                self.stable_samples.clear()
                self.stable = False
        except (KeyError, TypeError, ValueError, msgpack.UnpackException):
            return

    def poll_policy(self):
        # Process at most one newest packet per tick, never replay a backlog in bursts.
        latest = None
        for _ in range(32):
            try:
                latest = self.policy.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return
        try:
            if not self.policy_enabled or self.identity is None:
                raise ValueError("Policy is not enabled")
            values, sequence = decode_action(latest, self.identity, self.secret, self.sequence)
            # decode_action verified the HMAC before this metadata is trusted.
            packet = json.loads(latest.split(b"\0", 1)[1][32:])
            if type(packet["sequence"]) is not int or type(packet["sent_ns"]) is not int or packet["sent_ns"] < self.enabled_ns:
                raise ValueError("Pre-enable packet or non-integer sequence/time")
            now = time.monotonic_ns()
            if not self.owned(now) or not self.body_ready(now) or not self.hands_ready(now):
                raise ValueError("Controller or hand feedback is not ready")
            self.hand("action", action=values[64:].tolist())
            self.send_backend(2, values[:64])
            self.sequence, self.last_policy_ns = sequence, now
            self.accepted += 1
            self.phase, self.detail = "policy", "Authenticated bounded token/normalized-STARK action delivered"
            self.stable = False
        except (ValueError, TypeError, KeyError, RuntimeError, zmq.ZMQError) as exc:
            self.rejected += 1
            self.disable("Policy rejected: " + str(exc))

    def tick(self):
        self.poll_feedback()
        now = time.monotonic_ns()
        if now - self.last_hand_poll >= 100_000_000:
            self.last_hand_poll = now
            try:
                self.hand_feedback = self.hand("status")
                self.hand_received = time.monotonic_ns()
                if self.hand_owner:
                    self.hand("heartbeat")
            except (RuntimeError, ValueError, zmq.ZMQError):
                self.hand_feedback = {}
                if self.policy_enabled:
                    self.disable("Hand backend unavailable; policy disabled")
        if self.phase == "initializing":
            if not self.hands_ready(now) or not (self.backend_waiting(now) or self.backend_ready(now)):
                self.phase = "fault"
                self.detail = "Initialization readiness lost; publication stopped, controller shutdown NOT confirmed"
                self.seed_token = None
            elif now - self.hold_started > 15_000_000_000:
                self.phase, self.detail = "fault", "Initialization timed out; no policy authorization, controller shutdown NOT confirmed"
                self.seed_token = None
            elif (self.backend_ready(now) and self.body_ready(now)
                  and np.allclose(self.feedback["token"], self.seed_token, atol=1e-5, rtol=0)):
                self.hold_token = self.seed_token.copy()
                self.seed_token = None
                self.hold_started = now
                self.phase = "acquiring"
                self.send_backend(1)
            else:
                self.action.send(seed_packet(self.seed_token, self.seed_frame), zmq.NOBLOCK)
                if self.seed_frame >= 10 and self.seed_frame % 5 == 0:
                    self.action.send(start_packet(), zmq.NOBLOCK)
                self.seed_frame += 1
        if self.hold_token is not None:
            if self.phase == "acquiring":
                if self.owned(now):
                    self.phase = "holding"
                elif now - self.hold_started > 3_000_000_000:
                    self.phase, self.detail = "fault", "Controller did not acknowledge Hold ownership"
                else:
                    self.send_backend(1)
            elif self.owned(now):
                self.send_backend(3)
            elif self.phase != "fault":
                self.disable("Controller ownership/feedback lost; independent watchdog retains baseline")
                self.phase = "fault"
            if not self.body_ready(now) and self.policy_enabled:
                self.disable("Body feedback stale; policy disabled")
        self.poll_policy()
        if self.policy_enabled and self.last_policy_ns and not fresh(self.last_policy_ns, time.monotonic_ns()):
            self.phase = "policy-waiting"
            self.detail = "Policy watchdog: controller retains prepared baseline until fresh action"

    def close(self):
        try:
            if self.hold_token is not None:
                try:
                    self.send_backend(4)
                except zmq.ZMQError:
                    pass
            try:
                self.release_hands()
            except (RuntimeError, ValueError, zmq.ZMQError):
                pass
        finally:
            for sock in self.sockets:
                sock.close()
            self.context.term()
            if self.file_owned:
                credential_path(self.config).unlink()

    def run(self):
        self.open()
        next_tick = time.monotonic()
        while not self.closed:
            if self.control.poll(2):
                try:
                    result = self.handle(self.control.recv_json())
                except (ValueError, TypeError, KeyError, RuntimeError, zmq.ZMQError) as exc:
                    result = {"ok": False, "error": str(exc)}
                self.control.send_json(result)
                if self.closed:
                    break
            if time.monotonic() >= next_tick:
                try:
                    self.tick()
                except zmq.Again:
                    self.disable("Backend send blocked; independent controller watchdog retains baseline")
                next_tick = time.monotonic() + 0.02


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("serve", "prepare", "status", "initialize", "hold", "stop"))
    parser.add_argument("--config", default="configs/robot_inference/lambda0_g1.json")
    parser.add_argument("--allow-actuation", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token", help="JSON array[64] from the matching licensed SONIC checkpoint")
    args = parser.parse_args()
    runtime = None
    try:
        config = configuration(args.config)
        if args.operation == "serve":
            runtime = Runtime(config, args.allow_actuation)
            def stop(*_):
                runtime.closed = True
            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
            runtime.run()
            return 0
        fields = {}
        if args.operation in ("initialize", "hold", "stop"):
            fields["admin"] = json.loads(credential_path(config).read_text())["admin"]
        if args.operation in ("initialize", "hold"):
            fields["execute"] = args.execute
        if args.operation == "initialize":
            if not args.token:
                raise ValueError("initialize requires --token with the checkpoint-specific initial token")
            fields["token"] = json.loads(Path(args.token).read_text())
        result = control_request(config, args.operation, **fields)
        if args.operation in ("initialize", "hold"):
            deadline = time.monotonic() + 20
            while not result["stable"] and time.monotonic() < deadline:
                if result["phase"] == "fault":
                    raise RuntimeError(result["detail"])
                time.sleep(0.1)
                result = control_request(config, "status")
            if not result["stable"]:
                raise RuntimeError("Hold not stable within twenty seconds; no policy authorization")
        print(json.dumps(result, indent=2))
        if args.operation == "prepare":
            prepared = result["initialization_ready"] or (
                result["controller_ready"] and result["body_feedback_ready"] and result["hand_command_ready"])
            return 0 if prepared else 1
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, zmq.ZMQError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

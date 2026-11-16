"""Receiver safety contracts. No controller, ROS node or robot is started."""
import hashlib
import hmac
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from infer_robot import PREFIX, encode_action
from robot_hands import HandLease
from robot_runtime import Runtime, configuration


class HandReceiverTests(unittest.TestCase):
    def setUp(self):
        self.writes = []
        self.ready = True
        self.lease = HandLease(lambda: self.ready, self.writes.append, allow=True)
        self.owner = "a" * 64
        self.base = 10_000_000_000
        self.lease.handle({"command": "bind", "owner": self.owner,
                           "sent_ns": self.base + 2_000_000_000}, self.base)

    def action(self, **overrides):
        data = {"command": "action", "owner": self.owner, "sequence": 0,
                "sent_ns": self.base + 2_010_000_000, "action": [0.25] * 12}
        data.update(overrides)
        return data


    def test_replay_wrong_owner_and_bounds_never_publish(self):
        for data in (self.action(owner="b" * 64), self.action(action=[1.1] * 12),
                     self.action(action=[float("nan")] * 12), self.action(sequence=True)):
            with self.assertRaises(ValueError):
                self.lease.handle(data, self.base + 10_000_000)
        self.assertEqual(self.writes, [])
        self.lease.handle(self.action(), self.base + 10_000_000)
        with self.assertRaises(ValueError):
            self.lease.handle(self.action(), self.base + 20_000_000)
        self.assertEqual(len(self.writes), 1)

    def test_stale_future_and_lease_expiry(self):
        with self.assertRaises(ValueError):
            self.lease.handle(self.action(sent_ns=self.base), self.base + 10_000_000)
        with self.assertRaises(ValueError):
            self.lease.handle(self.action(sent_ns=self.base + 3_000_000_000), self.base + 10_000_000)
        with self.assertRaises(ValueError):
            self.lease.handle(self.action(), self.base + 251_000_000)
        self.assertIsNone(self.lease.owner)
        self.assertEqual(self.writes, [])

    def test_feedback_loss_revokes_owner(self):
        self.ready = False
        with self.assertRaises(ValueError):
            self.lease.handle(self.action(), self.base + 10_000_000)
        self.assertIsNone(self.lease.owner)
        self.assertEqual(self.writes, [])

    def test_bind_takeover_and_monitor_are_rejected(self):
        with self.assertRaises(ValueError):
            self.lease.handle({"command": "bind", "owner": "b" * 64,
                               "sent_ns": self.base}, self.base)
        monitor = HandLease(lambda: True, self.writes.append)
        with self.assertRaises(ValueError):
            monitor.handle({"command": "bind", "owner": self.owner,
                            "sent_ns": self.base}, self.base)
        self.assertEqual(self.writes, [])


class RouterReceiverTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(configuration(Path(__file__).parents[1] / "configs/robot_inference/lambda0_g1.json"))
        self.runtime.policy.bind("inproc://signed-policy")
        self.sender = self.runtime.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.connect("inproc://signed-policy")
        self.identity = {"session": "session", "model": "model", "task": "task", "manifestSha256": "a" * 64}
        self.secret = "b" * 64

    def tearDown(self):
        self.sender.close()
        self.runtime.close()

    def packet(self, action=None, sequence=0, sent=None, secret=None):
        packet = {"identity": self.identity, "sequence": sequence,
                  "sent_ns": time.monotonic_ns() if sent is None else sent,
                  "action": [0.0] * 76 if action is None else action}
        body = json.dumps(packet, separators=(",", ":"), sort_keys=True).encode()
        return PREFIX + hmac.new((secret or self.secret).encode(), body, hashlib.sha256).digest() + body

    def consume(self, packet):
        self.sender.send(packet)
        self.assertTrue(self.runtime.policy.poll(1000))
        self.runtime.poll_policy()

    def test_monitor_never_forwards_signed_action(self):
        self.consume(encode_action(np.zeros(76), 0, self.identity, self.secret))
        self.assertFalse(self.runtime.policy_enabled)
        self.assertEqual(self.runtime.accepted, 0)
        self.assertEqual(self.runtime.rejected, 1)
        self.assertIsNone(self.runtime.hold_token)

    def test_signed_consumer_rejects_auth_age_bounds_and_replay(self):
        cases = [self.packet(secret="x" * 64), self.packet(sent=time.monotonic_ns() - 300_000_000),
                 self.packet(sent=time.monotonic_ns() + 1_000_000_000),
                 self.packet(action=[1.26] * 64 + [0.] * 12),
                 self.packet(action=[0.] * 64 + [-0.1] * 12), self.packet(sequence=-1),
                 self.packet(sequence=True)]
        for raw in cases:
            self.runtime.identity, self.runtime.secret = self.identity, self.secret
            self.runtime.policy_enabled = True
            self.consume(raw)
            self.assertFalse(self.runtime.policy_enabled)
            self.assertEqual(self.runtime.accepted, 0)
        self.assertEqual(self.runtime.rejected, len(cases))

    def test_control_owner_cannot_be_bypassed(self):
        self.runtime.identity, self.runtime.secret = self.identity, self.secret
        for command in ("policy-enable", "policy-disable", "policy-unbind"):
            with self.assertRaises(ValueError):
                self.runtime.handle({"command": command, "identity": self.identity, "secret": "wrong"})
        with self.assertRaises(ValueError):
            self.runtime.handle({"command": "policy-bind", "identity": self.identity, "secret": self.secret})
        status = json.dumps(self.runtime.status())
        self.assertNotIn(self.secret, status)
        self.assertNotIn(self.runtime.admin, status)

    def test_telemetry_or_waiting_ack_does_not_establish_hold(self):
        now = time.monotonic_ns()
        self.runtime.body_feedback = {"stamp": now, "received": now}
        self.runtime.feedback = {"stamp": now, "received": now, "owner": 0,
                                 "sequence": 0, "flags": 16, "token": np.zeros(64)}
        self.assertFalse(self.runtime.status()["stable"])
        self.assertFalse(self.runtime.status()["controller_ready"])
        self.assertTrue(self.runtime.status()["controller_waiting"])
        with self.assertRaises(ValueError):
            self.runtime.handle({"command": "hold", "admin": self.runtime.admin, "execute": True})

    def test_pre_enable_packet_cannot_cross_new_session_barrier(self):
        raw = self.packet()
        self.runtime.identity, self.runtime.secret = self.identity, self.secret
        self.runtime.policy_enabled = True
        self.runtime.enabled_ns = time.monotonic_ns()
        self.consume(raw)
        self.assertFalse(self.runtime.policy_enabled)
        self.assertEqual(self.runtime.accepted, 0)
        self.assertIn("Pre-enable", self.runtime.detail)

    def test_initialization_loses_authorization_when_hands_disappear(self):
        now = time.monotonic_ns()
        self.runtime.allow = True
        self.runtime.phase = "initializing"
        self.runtime.seed_token = np.full(64, .1, dtype=np.float32)
        self.runtime.hold_started = now
        self.runtime.last_hand_poll = now
        self.runtime.feedback = {
            "stamp": now, "received": now, "owner": 0, "sequence": 0,
            "flags": 16, "token": np.zeros(64),
        }
        self.runtime.hand_feedback = {}
        self.runtime.tick()
        self.assertEqual(self.runtime.phase, "fault")
        self.assertIsNone(self.runtime.seed_token)
        self.assertFalse(self.runtime.policy_enabled)


if __name__ == "__main__":
    unittest.main()

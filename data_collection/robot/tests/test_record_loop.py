"""Exercise the real recorder using loopback synthetic publishers, never hardware."""

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import msgpack
import zmq

from lambda_robot_data.wire import read_records, sha256_file
from lambda_robot_data.sources import ROBOT_TOPIC, CONFIG_TOPIC


def pts_packet(pts):
    encoded = bytes(
        [
            0x21 | (((pts >> 30) & 7) << 1),
            (pts >> 22) & 255,
            (((pts >> 15) & 127) << 1) | 1,
            (pts >> 7) & 255,
            ((pts & 127) << 1) | 1,
        ]
    )
    pes = b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" + encoded
    return b"\x47\x41\x00\x10" + pes + bytes(184 - len(pes))


class RecordLoopTests(unittest.TestCase):
    def test_live_loop_and_stale_failure(self):
        context = zmq.Context()
        sonic, hands = context.socket(zmq.PUB), context.socket(zmq.PUB)
        sonic_port = sonic.bind_to_random_port("tcp://127.0.0.1")
        hands_port = hands.bind_to_random_port("tcp://127.0.0.1")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        camera_port = reservation.getsockname()[1]
        reservation.close()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as d:

                def launch(name, duration, timeout):
                    return subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "lambda_robot_data.cli",
                            "record",
                            "--sonic-endpoint",
                            "tcp://127.0.0.1:" + str(sonic_port),
                            "--sonic-local",
                            "--brainco-endpoint",
                            "tcp://127.0.0.1:" + str(hands_port),
                            "--camera-port",
                            str(camera_port),
                            "--duration",
                            str(duration),
                            "--ready-timeout",
                            str(timeout),
                            "--output",
                            str(Path(d) / name),
                            "--task",
                            "synthetic",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                proc = launch("complete", 0.3, 3)
                end = time.monotonic() + 5
                index = 0
                while proc.poll() is None and time.monotonic() < end:
                    wall, mono = time.time_ns(), time.monotonic_ns()
                    sonic.send(CONFIG_TOPIC + msgpack.packb({"control_frequency": 50}))
                    sonic.send(
                        ROBOT_TOPIC
                        + msgpack.packb(
                            dict(
                                index=index,
                                timestamp_realtime_ns=wall,
                                timestamp_monotonic_ns=mono,
                                body_q=[0.0] * 29,
                                body_dq=[0.0] * 29,
                                token_state=[0.2] * 64,
                                base_quat=[1.0, 0.0, 0.0, 0.0],
                            )
                        )
                    )
                    for side in ("left", "right"):
                        hands.send_multipart(
                            [
                                ("rt/brainco/" + side + "/state").encode(),
                                msgpack.packb(
                                    dict(
                                        sequence=index,
                                        wall_time_ns=wall,
                                        monotonic_ns=mono,
                                        positions=[100] * 6,
                                        speeds=[0] * 6,
                                        currents=[0] * 6,
                                        states=[0] * 6,
                                    )
                                ),
                            ]
                        )
                    hands.send_multipart(
                        [
                            b"rt/brainco/glove/applied",
                            msgpack.packb(
                                dict(
                                    sequence=index,
                                    wall_time_ns=wall,
                                    monotonic_ns=mono,
                                    left=[0.8] * 6,
                                    right=[0.7] * 6,
                                )
                            ),
                        ]
                    )
                    udp.sendto(pts_packet(index * 1800), ("127.0.0.1", camera_port))
                    index += 1
                    time.sleep(0.02)
                stdout, stderr = proc.communicate(timeout=3)
                self.assertEqual(proc.returncode, 0, stdout + stderr)
                root = Path(d) / "complete"
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertGreater(manifest["frames"], 2)
                for name, info in manifest["files"].items():
                    self.assertEqual(sha256_file(root / name), info["sha256"])
                self.assertTrue(
                    any(
                        row.get("type") == "sonic"
                        for row in read_records(root / "sonic.records")
                    )
                )
                # No data is sent to the next recorder: it must fail closed.
                proc = launch("missing", 0.2, 0.2)
                stdout, stderr = proc.communicate(timeout=3)
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(
                    json.loads((Path(d) / "missing/manifest.json").read_text())[
                        "status"
                    ],
                    "incomplete",
                )
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            udp.close()
            sonic.close(0)
            hands.close(0)
            context.term()


if __name__ == "__main__":
    unittest.main()

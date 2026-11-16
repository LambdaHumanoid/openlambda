"""Camera timing regressions; synthetic frames/local sockets, never hardware."""

import base64
from contextlib import contextmanager
from fractions import Fraction
import io
import socket
import struct
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import msgpack
from PIL import Image
import zmq

from lambda_robot_data import live_camera as camera


WALL = 1_800_000_000_000_000_000
MONO = 20_000_000_000
OFFSET = 3_000_000_000


def mapping(wall=WALL):
    return dict(
        g1_minus_host_monotonic_ns=OFFSET,
        calibrated_realtime_ns=wall,
        uncertainty_ns=1_000_000,
        source_boot_id="test-g1-boot",
    )


def jpeg_bytes():
    target = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 40, 60)).save(target, format="JPEG")
    return target.getvalue()


def raw_frame(stamp=MONO - OFFSET - 100_000_000, sequence=1):
    return dict(
        format="openlambda_camera_ingress_v1", g1_boot_id="test-g1-boot",
        g1_monotonic_ns=stamp, sequence=sequence, jpeg=jpeg_bytes(),
    )


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.wall = patch.object(camera.time, "time_ns", return_value=WALL).start()
        self.mono = patch.object(camera.time, "monotonic_ns", return_value=MONO).start()
        self.addCleanup(patch.stopall)

    def test_uses_ingress_and_calibrated_offset_not_receipt_time(self):
        value = raw_frame()
        result = camera.HostClock(mapping()).map_frame(value, MONO)
        self.assertEqual(result["timestamps"]["ego_view_g1_monotonic"], value["g1_monotonic_ns"] / 1e9)
        self.assertEqual(result["timestamps"]["ego_view_host_monotonic"], 19.9)
        self.assertEqual(base64.b64decode(result["images"]["ego_view"]), value["jpeg"])

    def test_rejects_stale_future_wrong_boot_and_malformed(self):
        clock = camera.HostClock(mapping())
        for changes in (
            {"g1_monotonic_ns": MONO - OFFSET - 250_000_001},
            {"g1_monotonic_ns": MONO - OFFSET + 50_000_001},
            {"g1_boot_id": "rebooted"}, {"sequence": True},
            {"g1_monotonic_ns": float("nan")}, {"jpeg": b"broken"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clock.map_frame(dict(raw_frame(), **changes), MONO)

    def test_expiry_and_clock_steps_fail_closed(self):
        clock = camera.HostClock(mapping())
        self.wall.return_value = WALL + 300_000_000_001
        self.mono.return_value = MONO + 300_000_000_001
        with self.assertRaisesRegex(ValueError, "expired"):
            clock.check()
        self.wall.return_value = WALL - 1_000_000_000
        self.mono.return_value = MONO
        with self.assertRaisesRegex(ValueError, "stepped"):
            clock.check()

    def test_invalid_initial_mapping(self):
        for changes in (
            {"calibrated_realtime_ns": WALL + 1},
            {"calibrated_realtime_ns": WALL - 300_000_000_001},
            {"uncertainty_ns": 10_000_001}, {"uncertainty_ns": -1},
            {"g1_minus_host_monotonic_ns": 1.5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                camera.HostClock(dict(mapping(), **changes))


class IngressTests(unittest.TestCase):
    def setUp(self):
        patch.object(camera.time, "time_ns", return_value=WALL).start()
        patch.object(camera.time, "monotonic_ns", return_value=MONO).start()
        self.addCleanup(patch.stopall)

    def stream(self, payload=b"not-video", age=100_000_000, ancillary=True, flags=0):
        stamp = WALL - age
        data = struct.pack("@ll", stamp // 1_000_000_000, stamp % 1_000_000_000)
        controls = [(socket.SOL_SOCKET, camera.SO_TIMESTAMPNS, data)] if ancillary else []
        udp = SimpleNamespace(recvmsg=lambda *_: (payload, controls, flags, ("127.0.0.1", 8554)))
        return camera.IngressStream(udp, threading.Event(), 15)

    def test_preserves_kernel_ingress_instead_of_recv_time(self):
        stream = self.stream()
        stream.parser = SimpleNamespace(feed=lambda _: (90000,))
        self.assertEqual(stream.read(3), b"not")
        frame = SimpleNamespace(pts=90000, time_base=Fraction(1, 90000))
        self.assertEqual(stream.frame_stamp(frame), MONO - 100_000_000)
        with self.assertRaisesRegex(ValueError, "no matching"):
            stream.frame_stamp(frame)

    def test_rejects_kernel_backlog_missing_timestamp_and_truncation(self):
        for options in ({"age": 250_000_001}, {"ancillary": False}, {"flags": socket.MSG_CTRUNC}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.stream(**options).read(1316)

    def test_matches_real_decoded_mpeg_pts(self):
        target = io.BytesIO()
        with av.open(target, "w", format="mpegts") as video:
            stream = video.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
            stream.options = {"bf": "0", "preset": "ultrafast", "tune": "zerolatency"}
            for index in range(3):
                frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), (index * 40, 0, 0)))
                frame.pts, frame.time_base = index * 3000, Fraction(1, 90000)
                for packet in stream.encode(frame):
                    video.mux(packet)
            for packet in stream.encode():
                video.mux(packet)
        data = target.getvalue()
        ingress = self.stream(payload=data)
        self.assertEqual(ingress.read(len(data)), data)
        with av.open(io.BytesIO(data), format="mpegts") as video:
            stamps = [ingress.frame_stamp(frame) for frame in video.decode(video=0)]
        self.assertEqual(stamps, [MONO - 100_000_000] * 3)

    def test_loopback_only(self):
        self.assertEqual(camera.loopback_endpoint("tcp://127.0.0.1:5558"), "tcp://127.0.0.1:5558")
        for endpoint in ("tcp://0.0.0.0:5558", "tcp://127.0.0.1:0", "tcp://127.0.0.1:65536"):
            with self.assertRaises(ValueError):
                camera.loopback_endpoint(endpoint)


class PublisherSocketTests(unittest.TestCase):
    def test_real_socket_schema_then_stale_source_closes_publisher(self):
        """Exercise actual adapter loop on localhost, injecting only SSH calibration."""
        stopped = threading.Event()
        errors = []

        @contextmanager
        def signals():
            yield stopped

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            output_port = reservation.getsockname()[1]
        output = "tcp://127.0.0.1:%d" % output_port
        with zmq.Context() as context, context.socket(zmq.PUB) as source, context.socket(zmq.SUB) as receiver:
            for channel in (source, receiver):
                channel.setsockopt(zmq.LINGER, 0)
            source_port = source.bind_to_random_port("tcp://127.0.0.1")
            receiver.setsockopt(zmq.SUBSCRIBE, b"")
            receiver.connect(output)
            args = SimpleNamespace(source="tcp://127.0.0.1:%d" % source_port, bind=output,
                                   clock_map="unused-test-map", ready_timeout=3)

            def worker():
                try:
                    camera.publish(args)
                except Exception as error:
                    errors.append(error)

            with patch.object(camera, "load_mapping", return_value=mapping(time.time_ns())), patch.object(camera, "stop_signals", signals):
                thread = threading.Thread(target=worker)
                thread.start()
                try:
                    deadline, sequence, result = time.monotonic() + 3, 0, None
                    while time.monotonic() < deadline:
                        sequence += 1
                        value = raw_frame(time.monotonic_ns() - OFFSET - 10_000_000, sequence)
                        source.send(msgpack.packb(value, use_bin_type=True))
                        if receiver.poll(30):
                            result = msgpack.unpackb(receiver.recv(), raw=False)
                            break
                    self.assertIsNotNone(result, errors)
                    self.assertEqual(set(result["timestamps"]), {"ego_view_g1_monotonic", "ego_view_host_monotonic"})
                    image = Image.open(io.BytesIO(base64.b64decode(result["images"]["ego_view"])))
                    self.assertEqual(image.size, (16, 12))
                    stale = raw_frame(time.monotonic_ns() - OFFSET - 1_000_000_000, sequence + 1)
                    source.send(msgpack.packb(stale, use_bin_type=True))
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], ValueError)
                    self.assertIn("stale", str(errors[0]))
                finally:
                    stopped.set()
                    thread.join(2)
                    self.assertFalse(thread.is_alive(), "publisher failed to clean up")


if __name__ == "__main__":
    unittest.main()

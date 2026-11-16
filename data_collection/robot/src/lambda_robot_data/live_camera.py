# SPDX-License-Identifier: Apache-2.0
"""Live RGB adapter: G1 ingress/PTS capture and calibrated host publication.

Run capture on G1 after the public GoPro relay, and publish on the SONIC host.
Only loopback sockets are accepted; forward the G1 publisher with SSH. Neither
command controls the camera, robot, hands, or model service. See --help.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from contextlib import contextmanager
import io
import signal
import socket
import struct
import threading
import time

import msgpack
import zmq

from .clock import boot_id, calibrate, load_mapping
from .timing import MPEG_CLOCK_HZ, MPEG_PTS_WRAP, MpegTsPtsExtractor

MAX_AGE_NS = 250_000_000
MAX_FUTURE_NS = 50_000_000
CLOCK_TTL_NS = 300_000_000_000
MAX_UNCERTAINTY_NS = 10_000_000
# Linux asm-generic SO_TIMESTAMPNS_OLD; G1 is 64-bit Linux (two native longs).
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)


def loopback_endpoint(value):
    prefix = "tcp://127.0.0.1:"
    if not value.startswith(prefix):
        raise ValueError("camera endpoints must use tcp://127.0.0.1:PORT (use SSH forwarding)")
    if not value[len(prefix):].isdigit() or not 1 <= int(value[len(prefix):]) <= 65535:
        raise ValueError("invalid camera endpoint port")
    return value


def integer(value, label):
    if type(value) is not int:
        raise ValueError(label + " must be an integer")
    return value


def check_age(stamp, now):
    age = now - stamp
    if not -MAX_FUTURE_NS <= age <= MAX_AGE_NS:
        raise ValueError("camera ingress is stale or in the future; check transport/clock")


class HostClock:
    """Reuse recorder calibration with host as recorder and G1 as source.

    The historical g1_minus_host_* field names mean LOCAL minus REMOTE;
    calibrating in this direction therefore gives HOST minus G1. Do not reuse
    a mapping created by the G1 demonstration recorder in this adapter.
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.started_mono = time.monotonic_ns()
        self.started_wall = time.time_ns()
        self.offset = integer(mapping["g1_minus_host_monotonic_ns"], "clock offset")
        self.calibrated = integer(mapping["calibrated_realtime_ns"], "calibration time")
        self.uncertainty = integer(mapping["uncertainty_ns"], "clock uncertainty")
        if not 0 <= self.uncertainty <= MAX_UNCERTAINTY_NS:
            raise ValueError("clock uncertainty exceeds 10 ms")
        self.check()

    def check(self):
        mono, wall = time.monotonic_ns(), time.time_ns()
        elapsed = mono - self.started_mono
        if abs(wall - self.started_wall - elapsed) > MAX_UNCERTAINTY_NS:
            raise ValueError("host realtime clock stepped; recalibrate and restart")
        age = self.started_wall - self.calibrated + elapsed
        if not 0 <= age <= CLOCK_TTL_NS:
            raise ValueError("camera clock mapping expired; recalibrate and restart (five-minute limit)")

    def map_frame(self, value, now):
        self.check()
        if not isinstance(value, dict) or value.get("format") != "openlambda_camera_ingress_v1":
            raise ValueError("invalid G1 camera ingress message")
        if value.get("g1_boot_id") != self.mapping["source_boot_id"]:
            raise ValueError("G1 camera boot differs from calibration; recalibrate")
        stamp = integer(value.get("g1_monotonic_ns"), "G1 ingress time")
        sequence = integer(value.get("sequence"), "camera sequence")
        if stamp <= 0 or sequence <= 0:
            raise ValueError("invalid G1 camera sequence or ingress time")
        mapped = stamp + self.offset
        check_age(mapped, now)
        jpeg = value.get("jpeg")
        if not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 16 * 1024 * 1024:
            raise ValueError("invalid camera JPEG size/type")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("incomplete camera JPEG")
        return {
            "images": {"ego_view": base64.b64encode(jpeg).decode("ascii")},
            "timestamps": {
                "ego_view_g1_monotonic": stamp / 1e9,
                "ego_view_host_monotonic": mapped / 1e9,
            },
        }


class IngressStream(io.RawIOBase):
    """Feed PyAV the original TS bytes, retaining exact PTS/ingress associations.

    Kernel UDP realtime timestamps, not recv/decode times, anchor each PES. The
    local wall-to-monotonic conversion is guarded against clock steps. Missing
    ancillary timestamps fail closed instead of substituting receive time.
    """

    def __init__(self, udp, stopped, ready_timeout):
        super().__init__()
        self.udp, self.stopped = udp, stopped
        self.parser = MpegTsPtsExtractor()
        self.stamps = OrderedDict()
        self.pending = b""
        self.started_mono = time.monotonic_ns()
        self.started_wall = time.time_ns()
        self.deadline = time.monotonic() + ready_timeout

    def readable(self):
        return True

    def read(self, size=-1):
        if size == 0:
            return b""
        while not self.pending:
            if self.stopped.is_set():
                raise InterruptedError("camera capture stopped")
            if time.monotonic() > self.deadline:
                raise TimeoutError("no fresh decoded camera frame before timeout")
            try:
                payload, ancillary, flags, _ = self.udp.recvmsg(65535, socket.CMSG_SPACE(16))
            except socket.timeout:
                continue
            mono, wall = time.monotonic_ns(), time.time_ns()
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise ValueError("truncated camera datagram or ingress timestamp")
            if abs(wall - self.started_wall - (mono - self.started_mono)) > MAX_UNCERTAINTY_NS:
                raise ValueError("G1 realtime clock stepped; restart capture")
            kernel_wall = None
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
                    seconds, nanoseconds = struct.unpack("@ll", data)
                    if not 0 <= nanoseconds < 1_000_000_000:
                        raise ValueError("invalid kernel UDP ingress time")
                    kernel_wall = seconds * 1_000_000_000 + nanoseconds
            if kernel_wall is None:
                raise ValueError("kernel UDP ingress timestamp unavailable; refusing receive-time fallback")
            ingress = kernel_wall + mono - wall
            check_age(ingress, mono)
            for pts in self.parser.feed(payload):
                if pts in self.stamps:
                    raise ValueError("ambiguous duplicate camera PTS")
                self.stamps[pts] = ingress
                if len(self.stamps) > 512:
                    self.stamps.popitem(last=False)
            self.pending = payload
        length = len(self.pending) if size < 0 else min(size, len(self.pending))
        result, self.pending = self.pending[:length], self.pending[length:]
        return result

    def frame_stamp(self, frame):
        if frame.pts is None or frame.time_base is None:
            raise ValueError("decoder did not expose camera PTS")
        ticks = frame.pts * frame.time_base * MPEG_CLOCK_HZ
        if ticks.denominator != 1:
            raise ValueError("decoded camera PTS is not an exact MPEG clock tick")
        pts = int(ticks) % MPEG_PTS_WRAP
        if pts not in self.stamps:
            raise ValueError("decoded camera frame has no matching G1 ingress PTS")
        return self.stamps.pop(pts)


@contextmanager
def stop_signals():
    stopped = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopped.set())
        yield stopped
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def capture(args):
    import av

    identity = boot_id()  # Linux G1 only; fail before binding on unsupported hosts.
    with stop_signals() as stopped, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        udp.bind(("127.0.0.1", args.camera_port))
        udp.settimeout(0.2)
        stream = IngressStream(udp, stopped, args.ready_timeout)
        with zmq.Context() as context, context.socket(zmq.PUB) as publisher:
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 2)
            publisher.bind(args.bind)
            print("Waiting for G1 loopback video with kernel ingress timestamps...", flush=True)
            sequence = 0
            last_stamp = 0
            try:
                with av.open(stream, mode="r", format="mpegts", buffer_size=1316,
                             options={"probesize": "32768", "analyzeduration": "0"}) as video:
                    if len(video.streams.video) != 1:
                        raise ValueError("camera stream must contain exactly one video track")
                    video.streams.video[0].codec_context.thread_count = 1
                    # The relay may already be streaming when capture joins.
                    # Wait for an independently decodable frame before feeding
                    # inter-frame packets whose references we never received.
                    for packet in video.demux(video=0):
                        if stopped.is_set():
                            break
                        if sequence == 0 and not packet.is_keyframe:
                            continue
                        for frame in packet.decode():
                            stamp = stream.frame_stamp(frame)
                            if stamp <= last_stamp:
                                raise ValueError("camera ingress timestamp did not advance")
                            last_stamp = stamp
                            # Decoder startup/backlog is discarded, never relabeled as fresh.
                            if time.monotonic_ns() - stamp > MAX_AGE_NS:
                                continue
                            jpeg = io.BytesIO()
                            frame.to_image().save(jpeg, format="JPEG", quality=args.jpeg_quality)
                            check_age(stamp, time.monotonic_ns())
                            sequence += 1
                            publisher.send(msgpack.packb({
                                "format": "openlambda_camera_ingress_v1",
                                "g1_boot_id": identity,
                                "g1_monotonic_ns": stamp,
                                "sequence": sequence,
                                "jpeg": jpeg.getvalue(),
                            }, use_bin_type=True))
                            stream.deadline = time.monotonic() + 1.0
                            if sequence == 1:
                                print("Publishing real PTS-matched RGB ingress frames on " + args.bind, flush=True)
            except Exception:
                if not stopped.is_set():
                    raise


def publish(args):
    # This validates local boot, source boot over SSH, age and uncertainty.
    clock = HostClock(load_mapping(args.clock_map))
    with stop_signals() as stopped, zmq.Context() as context:
        with context.socket(zmq.SUB) as source, context.socket(zmq.PUB) as publisher:
            for channel in (source, publisher):
                channel.setsockopt(zmq.LINGER, 0)
            source.setsockopt(zmq.SUBSCRIBE, b"")
            source.setsockopt(zmq.CONFLATE, 1)  # raw ingress uses one ZMQ frame
            source.setsockopt(zmq.MAXMSGSIZE, 17 * 1024 * 1024)
            publisher.setsockopt(zmq.SNDHWM, 2)
            source.connect(args.source)
            publisher.bind(args.bind)
            print("Waiting for calibrated G1 RGB; clock map expires after five minutes...", flush=True)
            deadline = time.monotonic() + args.ready_timeout
            last_sequence = last_stamp = 0
            while not stopped.is_set():
                clock.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("G1 camera publisher stopped or supplied no fresh RGB")
                if not source.poll(100):
                    continue
                value = msgpack.unpackb(source.recv(), raw=False)
                packet = clock.map_frame(value, time.monotonic_ns())
                sequence, stamp = value["sequence"], value["g1_monotonic_ns"]
                if sequence <= last_sequence or stamp <= last_stamp:
                    raise ValueError("G1 camera sequence/ingress did not advance; restart both adapters")
                publisher.send(msgpack.packb(packet, use_bin_type=True))
                if not last_sequence:
                    print("Publishing calibrated ego_view RGB on " + args.bind, flush=True)
                last_sequence, last_stamp = sequence, stamp
                deadline = time.monotonic() + 1.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="run on Linux SONIC host; map G1 into local host clock")
    calibration.add_argument("--g1", required=True, help="G1 SSH destination (BatchMode public-key authentication)")
    calibration.add_argument("--output", required=True, help="new clock JSON file; never overwrites")
    capture_parser = commands.add_parser("capture", help="run on G1; read public relay loopback MPEG-TS")
    capture_parser.add_argument("--camera-port", type=int, default=18554)
    capture_parser.add_argument("--bind", default="tcp://127.0.0.1:5558")
    capture_parser.add_argument("--jpeg-quality", type=int, default=90)
    capture_parser.add_argument("--ready-timeout", type=float, default=15)
    publisher = commands.add_parser("publish", help="run on SONIC host; publish interfaces.md RGB schema")
    publisher.add_argument("--source", default="tcp://127.0.0.1:5558")
    publisher.add_argument("--bind", default="tcp://127.0.0.1:5559")
    publisher.add_argument("--clock-map", required=True)
    publisher.add_argument("--ready-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    try:
        if args.command == "calibrate":
            value = calibrate(args.g1, args.output)
            print("Saved host-minus-G1 mapping; uncertainty %.3f ms; valid for five minutes" % (value["uncertainty_ns"] / 1e6))
            return
        loopback_endpoint(args.bind)
        if not 0 < args.ready_timeout <= 60:
            raise ValueError("ready timeout must be within (0, 60] seconds")
        if args.command == "capture":
            if not 1 <= args.camera_port <= 65535 or not 1 <= args.jpeg_quality <= 100:
                raise ValueError("invalid camera port or JPEG quality")
            capture(args)
        else:
            loopback_endpoint(args.source)
            if args.source == args.bind:
                raise ValueError("camera source and output endpoints must differ")
            publish(args)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, msgpack.UnpackException) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()

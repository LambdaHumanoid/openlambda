"""Subscribe to existing telemetry and record one episode on the robot."""

import shutil
import signal
import socket
import time
from pathlib import Path

import zmq

from .clock import boot_id, load_mapping
from .sources import (
    decode_sonic,
    decode_brainco,
    ROBOT_TOPIC,
    CONFIG_TOPIC,
    LEFT_STATE_TOPIC,
    RIGHT_STATE_TOPIC,
    APPLIED_HAND_TOPIC,
)
from .timing import MpegTsPtsExtractor, FrameIngressTimestamp, pack_frame_timestamp
from .wire import RecordWriter, atomic_write_json, sha256_file

FILES = ("camera.ts", "camera.timestamps.bin", "sonic.records", "brainco.records")


def record(args):
    mapping = (
        dict(
            g1_minus_host_realtime_ns=0,
            g1_minus_host_monotonic_ns=0,
            uncertainty_ns=0,
            recorder_boot_id=boot_id(),
            source="same-host",
        )
        if args.sonic_local
        else load_mapping(args.clock_map)
    )
    if args.sonic_local and not args.sonic_endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError(
            "--sonic-local requires a loopback endpoint (no SSH port forwarding)"
        )
    if args.duration <= 0 or args.duration > 120:
        raise ValueError("episode duration must be within (0, 120] seconds")
    root = Path(args.output).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 2 * 1024**3:
        raise ValueError("at least 2 GiB free space is required")
    root.mkdir(exist_ok=False)
    metadata = dict(
        format="openlambda_robot_raw_v1",
        status="incomplete",
        task=args.task,
        clock_mapping=mapping,
        sonic_endpoint=args.sonic_endpoint,
        brainco_endpoint=args.brainco_endpoint,
        camera_clock="G1 loopback ingress",
        hand_action_semantics="published STARK normalized target; not motor acknowledgement",
        started_realtime_ns=time.time_ns(),
        files={},
    )
    atomic_write_json(root / "manifest.json", metadata)
    context = zmq.Context()
    sonic = context.socket(zmq.SUB)
    hands = context.socket(zmq.SUB)
    for channel in (sonic, hands):
        channel.setsockopt(zmq.LINGER, 0)
        channel.setsockopt(zmq.RCVHWM, 10000)
    for topic in (ROBOT_TOPIC, CONFIG_TOPIC):
        sonic.setsockopt(zmq.SUBSCRIBE, topic)
    for topic in (LEFT_STATE_TOPIC, RIGHT_STATE_TOPIC, APPLIED_HAND_TOPIC):
        hands.setsockopt(zmq.SUBSCRIBE, topic)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    writers = []
    old_handler = signal.getsignal(signal.SIGINT)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    try:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        udp.bind(("127.0.0.1", args.camera_port))
        udp.setblocking(False)
        sonic.connect(args.sonic_endpoint)
        hands.connect(args.brainco_endpoint)
        poller = zmq.Poller()
        poller.register(sonic, zmq.POLLIN)
        poller.register(hands, zmq.POLLIN)
        poller.register(udp.fileno(), zmq.POLLIN)
        robot_writer = RecordWriter(root / "sonic.records")
        hand_writer = RecordWriter(root / "brainco.records")
        writers.extend((robot_writer, hand_writer))
        parser = MpegTsPtsExtractor()
        seen = {}
        last = {}
        camera_sequence = 0
        robot_count = 0
        start = time.monotonic()
        ready_at = None
        disk_check = start
        signal.signal(signal.SIGINT, stop)
        print(
            "Waiting for Sonic, both hands, applied commands and video PTS...",
            flush=True,
        )
        with (
            (root / "camera.ts").open("xb") as video,
            (root / "camera.timestamps.bin").open("xb") as stamps,
        ):
            while not stopped:
                now = time.monotonic()
                if ready_at is not None and now - ready_at >= args.duration:
                    break
                events = dict(poller.poll(50))
                if sonic in events:
                    raw = sonic.recv()
                    kind, value = decode_sonic(
                        raw,
                        mapping["g1_minus_host_monotonic_ns"],
                        mapping["g1_minus_host_realtime_ns"],
                    )
                    if kind == "robot":
                        key = value["index"]
                        if key <= last.get("robot", -1):
                            raise ValueError("Sonic index did not advance")
                        age = time.monotonic_ns() - value["source_g1_monotonic_ns"]
                        if age < -20_000_000 or age > 1_000_000_000:
                            raise ValueError(
                                "Sonic source clock is stale or mapping is invalid"
                            )
                        last["robot"] = key
                        robot_count += 1
                        seen["robot"] = now
                    else:
                        seen["config"] = now
                    robot_writer.write(value)
                if hands in events:
                    kind, value = decode_brainco(hands.recv_multipart())
                    if value["sequence"] <= last.get(kind, -1):
                        raise ValueError("hand sequence did not advance: " + kind)
                    age = time.monotonic_ns() - value["source_monotonic_ns"]
                    if not 0 <= age <= 1_000_000_000:
                        raise ValueError("hand telemetry must use this G1 clock")
                    last[kind] = value["sequence"]
                    seen[kind] = now
                    hand_writer.write(value)
                if udp.fileno() in events:
                    # Bounded draining keeps high-rate video from starving telemetry.
                    for _ in range(64):
                        try:
                            payload, _ = udp.recvfrom(65535)
                        except BlockingIOError:
                            break
                        mono, wall = time.monotonic_ns(), time.time_ns()
                        video.write(payload)
                        for pts in parser.feed(payload):
                            camera_sequence += 1
                            stamps.write(
                                pack_frame_timestamp(
                                    FrameIngressTimestamp(
                                        1, camera_sequence, pts, mono, wall
                                    )
                                )
                            )
                            seen["camera"] = now
                required = ("robot", "left", "right", "hand_action", "camera")
                if ready_at is None:
                    if "config" in seen and all(
                        now - seen.get(k, -1e20) < 1 for k in required
                    ):
                        ready_at = now
                        print(
                            "Recording; Ctrl-C finishes this episode (does not stop the robot).",
                            flush=True,
                        )
                    elif now - start > args.ready_timeout:
                        raise TimeoutError(
                            "missing sources: "
                            + ", ".join(
                                k for k in (*required, "config") if k not in seen
                            )
                        )
                elif any(now - seen.get(k, -1e20) > 1 for k in required):
                    raise TimeoutError("a required telemetry stream stopped")
                if now - disk_check > 1:
                    disk_check = now
                    if shutil.disk_usage(root).free < 512 * 1024**2:
                        raise OSError("low disk space")
            if ready_at is None or robot_count < 2 or camera_sequence < 2:
                raise ValueError("episode ended before all sources became ready")
            import os

            for f in (video, stamps):
                f.flush()
                os.fsync(f.fileno())
        for writer in writers:
            writer.close()
        metadata.update(
            status="complete",
            frames=robot_count,
            camera_frames=camera_sequence,
            stopped_realtime_ns=time.time_ns(),
        )
        metadata["files"] = {
            name: dict(
                bytes=(root / name).stat().st_size, sha256=sha256_file(root / name)
            )
            for name in FILES
        }
        atomic_write_json(root / "manifest.json", metadata)
        print("Saved raw episode: " + str(root))
    except BaseException as error:
        metadata.update(status="incomplete", error=str(error))
        atomic_write_json(root / "manifest.json", metadata)
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)
        for writer in writers:
            writer.close()
        udp.close()
        sonic.close()
        hands.close()
        context.term()

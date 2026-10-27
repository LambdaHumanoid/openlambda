"""Bounded round-trip clock mapping; no cross-machine monotonic subtraction."""

import json
import selectors
import socket
import subprocess
import time
from pathlib import Path


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def calibrate(host, output, max_uncertainty_ms=10):
    if host.startswith("-"):
        raise ValueError("SSH host must not begin with -")
    code = (
        "import sys,time,json,socket; from pathlib import Path; "
        'b=Path("/proc/sys/kernel/random/boot_id").read_text().strip(); '
        'print(json.dumps({"host":socket.gethostname(),"boot_id":b}),flush=True); '
        "\nfor line in sys.stdin: print(json.dumps([time.time_ns(),time.monotonic_ns()]),flush=True)"
    )
    import shlex

    proc = subprocess.Popen(
        [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            host,
            "python3 -u -c " + shlex.quote(code),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    def receive():
        if not selector.select(15):
            raise TimeoutError("clock probe timed out")
        return json.loads(proc.stdout.readline())

    try:
        remote = receive()
        samples = []
        for _ in range(12):
            wall, mono = time.time_ns(), time.monotonic_ns()
            proc.stdin.write("ping\n")
            proc.stdin.flush()
            remote_wall, remote_mono = receive()
            end_wall, end_mono = time.time_ns(), time.monotonic_ns()
            rtt = end_mono - mono
            samples.append(
                (
                    rtt,
                    (wall + end_wall) // 2 - remote_wall,
                    (mono + end_mono) // 2 - remote_mono,
                )
            )
        rtt, wall_offset, mono_offset = min(samples)
        if rtt / 2e6 > max_uncertainty_ms:
            raise ValueError("clock uncertainty exceeds limit; use the robot LAN")
        result = dict(
            recorder_host=socket.gethostname(),
            recorder_boot_id=boot_id(),
            source_host=remote["host"],
            source_boot_id=remote["boot_id"],
            source_ssh=host,
            calibrated_realtime_ns=time.time_ns(),
            g1_minus_host_realtime_ns=wall_offset,
            g1_minus_host_monotonic_ns=mono_offset,
            uncertainty_ns=rtt // 2,
        )
        with Path(output).open("x") as f:
            json.dump(result, f, indent=2)
        return result
    finally:
        selector.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def load_mapping(path):
    value = json.loads(Path(path).read_text())
    if value["recorder_boot_id"] != boot_id():
        raise ValueError("clock mapping was made on another recorder boot")
    age = time.time_ns() - value["calibrated_realtime_ns"]
    if not 0 <= age <= 300_000_000_000:
        raise ValueError("clock mapping is older than five minutes; recalibrate")
    if not 0 <= value["uncertainty_ns"] <= 10_000_000:
        raise ValueError("clock uncertainty exceeds 10 ms")
    # Ensure the source did not reboot after calibration.
    check = subprocess.run(
        [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            value["source_ssh"],
            "cat /proc/sys/kernel/random/boot_id",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    if check.stdout.strip() != value["source_boot_id"]:
        raise ValueError("Sonic host rebooted; recalibrate")
    return value

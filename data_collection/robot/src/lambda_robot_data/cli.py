"""Public command-line entry points. Imports are lazy for robot-side installs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="G1 + SONIC + Revo2 demonstration recorder"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    clock = sub.add_parser(
        "calibrate", help="map the Sonic host clock to the G1 recorder clock"
    )
    clock.add_argument(
        "--host", required=True, help="SSH destination of the Sonic process host"
    )
    clock.add_argument("--output", required=True, type=Path)
    capture = sub.add_parser(
        "record", help="record telemetry on G1; never issue motor commands"
    )
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--task", required=True)
    capture.add_argument("--sonic-endpoint", required=True)
    capture.add_argument("--brainco-endpoint", default="tcp://127.0.0.1:18560")
    capture.add_argument("--camera-port", default=18554, type=int)
    capture.add_argument("--duration", default=60, type=float)
    capture.add_argument("--ready-timeout", default=15, type=float)
    clocks = capture.add_mutually_exclusive_group(required=True)
    clocks.add_argument("--clock-map", type=Path)
    clocks.add_argument(
        "--sonic-local", action="store_true", help="Sonic process runs on this G1 OS"
    )
    verify = sub.add_parser(
        "verify", help="verify raw manifest, hashes and record checksums"
    )
    verify.add_argument("raw", type=Path)
    export = sub.add_parser(
        "export", help="offline RGB MP4 + Parquet export (requires [export])"
    )
    export.add_argument("raw", type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            from .clock import calibrate

            result = calibrate(args.host, args.output)
            print(
                "Saved clock map; uncertainty %.3f ms"
                % (result["uncertainty_ns"] / 1e6)
            )
        elif args.command == "record":
            from .record import record

            if not 1 <= args.camera_port <= 65535 or not 0 < args.ready_timeout <= 60:
                raise ValueError("invalid camera port or ready timeout")
            if not args.brainco_endpoint.startswith("tcp://127.0.0.1:"):
                raise ValueError(
                    "record on G1 and use its local hand telemetry endpoint"
                )
            record(args)
        elif args.command == "verify":
            from .wire import read_records, sha256_file

            manifest = json.loads((args.raw / "manifest.json").read_text())
            if (
                manifest.get("format") != "openlambda_robot_raw_v1"
                or manifest.get("status") != "complete"
            ):
                raise ValueError("episode is not a completed OpenLambda recording")
            from .record import FILES

            for name in FILES:
                descriptor = manifest["files"][name]
                path = args.raw / name
                if (
                    path.stat().st_size != descriptor["bytes"]
                    or sha256_file(path) != descriptor["sha256"]
                ):
                    raise ValueError("integrity check failed: " + name)
                if name.endswith(".records"):
                    if not sum(1 for _ in read_records(path)):
                        raise ValueError("empty record stream: " + name)
            print(
                "Raw integrity verified (not a timing or hardware-quality certification)."
            )
        else:
            from .export import export_episode

            print(json.dumps(export_episode(args.raw, args.output), indent=2))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()

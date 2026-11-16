#!/usr/bin/env python3
"""Prepare the pinned public SONIC source; never start robot control."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

SONIC_REVISION = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
ROOT = Path(__file__).resolve().parents[1]


def run(directory: Path, *args: str, check: bool = True):
    return subprocess.run(args, cwd=directory, check=check, text=True, capture_output=True)


def patch_paths(patch: Path) -> set[str]:
    paths = set()
    for line in patch.read_text().splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        name = line[4:].split("\t", 1)[0]
        if name == "/dev/null":
            continue
        if not name.startswith(("a/", "b/")):
            raise ValueError(f"Unsupported patch path: {name}")
        name = name[2:]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe patch path: {name}")
        paths.add(name)
    if not paths:
        raise ValueError(f"No file changes in {patch}")
    return paths


def plan_patches(source: Path, patches: list[Path]) -> list[Path]:
    """Preflight every patch on a disposable copy of affected working files."""
    source = source.resolve(strict=True)
    paths = set().union(*(patch_paths(p) for p in patches))
    with tempfile.TemporaryDirectory(prefix="openlambda-sonic-preflight-") as directory:
        staging = Path(directory)
        for name in paths:
            original = source / name
            if original.is_symlink() or not original.resolve().is_relative_to(source):
                raise ValueError(f"Patch target escapes checkout: {name}")
            if original.exists():
                if not original.is_file():
                    raise ValueError(f"Patch target is not a regular file: {name}")
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        applied = set()
        # Remove known layers in reverse order inside the disposable copy:
        # the newer patch may change context belonging to the base patch.
        for patch in reversed(patches):
            if run(staging, "git", "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
                run(staging, "git", "apply", "--reverse", str(patch))
                applied.add(patch)
        pending = []
        for patch in patches:
            result = run(staging, "git", "apply", "--check", str(patch), check=False)
            if result.returncode:
                raise ValueError(f"Patch conflicts; checkout left unchanged: {patch.name}\n{result.stderr}")
            run(staging, "git", "apply", str(patch))
            if patch not in applied:
                pending.append(patch)
        return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sonic", type=Path, help="Existing public GR00T-WholeBodyControl checkout")
    parser.add_argument("--apply", action="store_true", help="Apply compatible patches; default only checks")
    parser.add_argument("--build", action="store_true", help="With --apply, configure and build SONIC (no launch)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.build and not args.apply:
        parser.error("--build requires --apply")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = args.sonic.resolve(strict=True)
    if not source.is_dir():
        parser.error("SONIC checkout must be a directory")
    revision = run(source, "git", "rev-parse", "HEAD").stdout.strip()
    if revision != SONIC_REVISION:
        raise ValueError(f"Expected SONIC {SONIC_REVISION}, found {revision}; checkout left unchanged")
    patches = [
        ROOT / "data_collection/robot/setup/sonic.patch",
        ROOT / "scripts/runtime/sonic_runtime.patch",
    ]
    header = ROOT / "scripts/runtime/openlambda_runtime.hpp"
    destination = source / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/openlambda_runtime.hpp"
    header_bytes = header.read_bytes()
    if not destination.resolve().is_relative_to(source) or destination.is_symlink():
        raise ValueError("Runtime header target escapes checkout")
    if destination.exists() and destination.read_bytes() != header_bytes:
        raise ValueError("Existing runtime header differs; checkout left unchanged")
    pending = plan_patches(source, patches)
    print(f"Pinned SONIC source: {revision}")
    print(f"Patches to apply: {', '.join(p.name for p in pending) or 'none (already applied)'}")
    if not args.apply:
        print("Check only; no source changes, downloads, build or controller startup.")
        return
    for patch in pending:
        run(source, "git", "apply", str(patch))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(header_bytes)
    print("Public runtime source prepared. No controller has been started.")
    if args.build:
        subprocess.run(["cmake", "-S", "gear_sonic_deploy", "-B", "gear_sonic_deploy/build", "-DCMAKE_BUILD_TYPE=Release"], cwd=source, check=True)
        subprocess.run(["cmake", "--build", "gear_sonic_deploy/build", "--parallel", str(args.jobs)], cwd=source, check=True)
        print("Build finished. Model assets and GPU/runtime readiness require separate checks.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error

"""Read-only inventory of a trusted, separately installed SONIC runtime.

Does not import the controller, start simulation, install packages, or inspect
environment variables/remote URLs. This is provenance, not a dependency lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

RUNTIME_FILES = (
    "gear_sonic/scripts/run_sim_loop.py",
    "gear_sonic/scripts/run_lerobot_dataset_replay.py",
    "gear_sonic/scripts/run_lerobot_replayer.py",
    "gear_sonic/scripts/send_sim_control_command.py",
    "gear_sonic/rollout_contract.py",
    "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
    "gear_sonic_deploy/policy/release/model_encoder.onnx",
    "gear_sonic_deploy/policy/release/model_decoder.onnx",
    "gear_sonic_deploy/policy/release/observation_config.yaml",
    "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    "gear_sonic/data/human/human_joints_info.pkl",
)
PACKAGES = (
    "numpy", "scipy", "torch", "mujoco", "pin", "pinocchio", "lerobot",
    "pyarrow", "pandas", "av", "pyzmq", "onnxruntime", "onnxruntime-gpu",
    "tyro", "opencv-python", "datasets", "transformers", "torchvision",
)


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return {"returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def file_record(path):
    if not path.is_file():
        return {"present": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"present": True, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def python_record(executable):
    code = ("import importlib.metadata as m,json,sys; "
            "names=json.loads(sys.argv[1]); "
            "installed={d.metadata['Name'].lower().replace('_','-'):d.version "
            "for d in m.distributions()}; "
            "print(json.dumps({'python':sys.version,'packages':"
            "{n:installed.get(n.lower().replace('_','-')) for n in names}}))")
    result = command([str(executable), "-c", code, json.dumps(PACKAGES)])
    if result["returncode"] != 0:
        return result
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"error": "Interpreter produced invalid inventory JSON", **result}


def missing_libraries(output):
    return sorted({line.split("=>", 1)[0].strip()
                   for line in output.splitlines() if "=> not found" in line})


def collect(root, interpreters, assets=(), inspect_linkage=False):
    files = {name: file_record(root / name) for name in RUNTIME_FILES}
    report = {
        "schema": 1, "purpose": "observed provenance, not an install lock",
        "platform": platform.platform(),
        "runtime_revision": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "runtime_tracked_changes": command(["git", "-C", str(root), "diff", "--name-only", "HEAD"]),
        "runtime_submodules": command(["git", "-C", str(root), "submodule", "status", "--recursive"]),
        "runtime_files": files,
        "python_environments": {name: python_record(path) for name, path in interpreters.items()},
        "extra_assets": {str(p): file_record(p) for p in assets},
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "compiler": command(["c++", "--version"]),
        "cuda_compiler": command(["nvcc", "--version"]),
        "native_linkage_checked": inspect_linkage,
        "native_e2e_validated": False,
        "public_runtime_pin_verified": False,
    }
    if inspect_linkage:
        binary = root / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        linkage = command(["ldd", str(binary)])
        report["native_linkage"] = linkage
        report["missing_native_libraries"] = missing_libraries(linkage["stdout"])
    report["missing_runtime_files"] = [name for name, value in files.items() if not value["present"]]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--data-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sim-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--asset", action="append", type=Path, default=[])
    parser.add_argument("--inspect-trusted-native-binary", action="store_true",
                        help="Run ldd only on your trusted local controller executable")
    args = parser.parse_args()
    report = collect(args.runtime_root.resolve(),
                     {"data": args.data_python, "simulation": args.sim_python},
                     args.asset, args.inspect_trusted_native_binary)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(report["missing_runtime_files"] or report.get("missing_native_libraries")
                    or any("python" not in item for item in report["python_environments"].values())
                    or (args.inspect_trusted_native_binary and report["native_linkage"]["returncode"] != 0)))


if __name__ == "__main__":
    raise SystemExit(main())

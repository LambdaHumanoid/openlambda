"""Simulation-only source-runtime launcher; all processes share one environment.

No Docker management, real robot target, global process killing, or joint override.
Fresh-build local replay is validated separately from clean-install qualification.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import tempfile
import time

from egoalign.runtime_common import clean_environment, port_set, verify_runtime
from egoalign.runtime_assets import MANIFEST as ASSET_MANIFEST, materialize

ROOT = Path(__file__).resolve().parents[1]


def controller_command(root, base, binary):
    deploy = root / "gear_sonic_deploy"
    return [str(binary), "lo", str(deploy / "policy/release/model_decoder.onnx"),
            str(deploy / "reference/example"),
            "--obs-config", str(deploy / "policy/release/observation_config.yaml"),
            "--encoder-file", str(deploy / "policy/release/model_encoder.onnx"),
            "--planner-file", str(deploy / "planner/target_vel/V2/planner_sonic.onnx"),
            "--input-type", "zmq_manager", "--output-type", "zmq",
            "--zmq-host", "localhost", "--zmq-port", str(base),
            "--zmq-out-port", str(base+1), "--hand-type", "brainco",
            "--hand-close-threshold", "0.5", "--brainco-clamp-ms", "300",
            "--brainco-feedback-grace-ms", "2000", "--zmq-robot-ip", "localhost",
            "--zmq-robot-state-port", str(base+2), "--zmq-robot-cmd-port", str(base+3),
            "--zmq-robot-lowcmd-udp-port", "0", "--disable-crc-check",
            "--lock-waist-axes", "none", "--waist-lock-kp", "250", "--waist-lock-kd", "5",
            "--joint-measured-speed-limit", "12.5"]


def commands(root, data_python, sim_python, binary, source, output, episodes, base, sentinel):
    script = root / "gear_sonic/scripts"
    prepare = [str(data_python), str(script / "send_sim_control_command.py"), "sim:reset",
               "--host", "localhost", "--port", str(base+4), "--timeout", "30",
               "--retries", "20", "--retry-delay", "0.5", "--expect", "sim reset"]
    deploy = controller_command(root, base, binary)
    sim = [str(sim_python), str(script / "run_sim_loop.py"), "--interface", "lo",
           "--no-enable-onscreen"]
    supervisor = ["bash", str(script / "run_batch_deploy_supervisor.sh"),
                  "--active-file", str(sentinel / "active"), "--request-file", str(sentinel / "request"),
                  "--prepare-command", shlex.join(prepare), "--deploy-command", "exec " + shlex.join(deploy)]
    replay = [str(data_python), str(script / "run_lerobot_dataset_replay.py"),
              "--dataset-root", str(source), "--output-dir", str(output), "--episodes", episodes,
              "--replay-protocol", "pose", "--ramp-seconds", "5", "--startup-timeout", "180",
              "--planner-settle-ticks", "10", "--sim-control-port", str(base+4),
              "--zmq-port", str(base), "--state-zmq-port", str(base+1),
              "--sim-stability-timeout", "15", "--sim-stability-poll-hz", "10",
              "--sim-stability-hold-seconds", "0.5", "--sim-max-root-linear-speed", "0.10",
              "--sim-max-root-angular-speed", "0.35", "--sim-max-body-joint-speed", "1.0",
              "--no-reset-sim", "--sim-prepared-before-deploy", "--deploy-exit-timeout", "15",
              "--deploy-supervisor-active-file", str(sentinel / "active"),
              "--deploy-supervisor-request-file", str(sentinel / "request"), "--retries", "1"]
    return {"sim": sim, "supervisor": supervisor, "replay": replay, "controller": deploy}


def runtime_environment(root, model, output, base, inherited):
    env = clean_environment(root, inherited)
    # Telemetry uses the recorder even when MP4 rendering is disabled.
    env.update(SONIC_REPLAY_VIDEO_DIR=str(output),
               SONIC_BRAINCO_MODEL_DIR=str(model), SONIC_BATCH_REPLAY_LANE="true",
               SONIC_BATCH_REPLAY_LANE_ISOLATED="true", SONIC_SIM_SET_DEFAULT_POSE_ON_RESET="1",
               SONIC_ZMQ_STATE_PORT=str(base+2), SONIC_ZMQ_CMD_PORT=str(base+3),
               SIM_CONTROL_PORT=str(base+4), G1_WBC_DEFAULT_WIRED_ROBOT_IP="127.0.0.1")
    return env


def terminate_group(process, *, replay=False):
    # The group belongs to a Popen(start_new_session=True) made by this launcher.
    # The dataset replayer owns per-episode processes in separate sessions. Its
    # KeyboardInterrupt handler retires those children; SIGTERM skips it.
    try:
        os.killpg(process.pid, signal.SIGINT if replay else signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15 if replay else 5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    # A supervisor may exit before its child; retire only its private group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["sonic_sim"], required=True)
    p.add_argument("--mode", choices=["replay"], required=True)
    p.add_argument("--hand-type", choices=["brainco"], required=True)
    p.add_argument("--replay-dataset", type=Path, required=True)
    p.add_argument("--replay-video-dir", type=Path, required=True)
    p.add_argument("--replay-episodes", default="all")
    p.add_argument("--replay-workers", type=int, choices=[1], default=1)
    p.add_argument("--port-base", type=int, default=int(os.environ.get("EGOALIGN_PORT_BASE", "28700")))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        root = Path(os.environ["EGOALIGN_RUNTIME_ROOT"]).expanduser().resolve()
        data_python = Path(os.environ["EGOALIGN_RUNTIME_PYTHON"]).expanduser().absolute()
        sim_python = Path(os.environ["EGOALIGN_SIM_PYTHON"]).expanduser().absolute()
        model = Path(os.environ["EGOALIGN_BRAINCO_MODEL_DIR"]).expanduser().resolve()
        binary = root / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        manifest = json.loads((ROOT / "runtime/patches/manifest.json").read_text())
        verify_runtime(root, {"files": {r["path"]: r["after_sha256"] for r in manifest["files"]}})
        source, output = a.replay_dataset.resolve(), a.replay_video_dir.resolve()
        if not source.is_dir() or source == output or source in output.parents or output in source.parents:
            raise ValueError("Dataset must exist; source/output must be disjoint")
        ports = port_set(a.port_base, 1)
        cmd = commands(root, data_python, sim_python, binary, source, output,
                       a.replay_episodes, a.port_base, output / "launcher_sentinels")
        if a.dry_run:
            print(json.dumps({"commands": cmd, "cwd": str(root), "ports": ports,
                              "source_hashes_checked": True,
                              "build_and_runtime_dependencies_checked": False}, indent=2))
            return 0
        if os.environ.get("EGOALIGN_ALLOW_SIM_REPLAY") != "1":
            raise ValueError("Set EGOALIGN_ALLOW_SIM_REPLAY=1 after reviewing the command")
        if output.exists():
            raise ValueError("Use a fresh output directory; this launcher never overwrites results")
        for path in (data_python, sim_python, binary, model / "scene_29dof_brainco.xml", model / "g1_29dof_brainco.xml"):
            if not path.is_file():
                raise ValueError(f"Missing dependency: {path}")
        materialize(root, json.loads(ASSET_MANIFEST.read_text()))
        for port in ports:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", port))
        env = runtime_environment(root, model, output, a.port_base, os.environ)
        output.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="egoalign-supervisor-") as tmp:
            sentinel = Path(tmp)
            (sentinel / "active").write_text(sentinel.name + "\n")
            cmd = commands(root, data_python, sim_python, binary, source, output,
                           a.replay_episodes, a.port_base, sentinel)
            (output / "launcher_commands.json").write_text(json.dumps(cmd, indent=2) + "\n")
            processes, logs = [], []
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            try:
                for sig in previous:
                    signal.signal(sig, interrupted)
                for name in ("sim", "supervisor", "replay"):
                    log = (output / f"launcher_{name}.log").open("w")
                    logs.append(log)
                    processes.append(subprocess.Popen(cmd[name], cwd=root, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                while processes[-1].poll() is None:
                    if any(proc.poll() is not None for proc in processes[:-1]):
                        # Successful dataset completion retires the supervisor first.
                        # Allow the replay parent to finish flushing its result logs.
                        try:
                            return processes[-1].wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            pass
                        raise RuntimeError("Simulator/supervisor exited before replay; inspect launcher logs")
                    time.sleep(0.2)
                return processes[-1].returncode
            finally:
                for index in reversed(range(len(processes))):
                    terminate_group(processes[index], replay=index == 2)
                for log in logs:
                    log.close()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    except (KeyError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        p.exit(2, f"Public runtime: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

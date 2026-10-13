"""Export an MP4 from measured MuJoCo replay telemetry, without new dynamics."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_telemetry(sim_path: Path, control_path: Path):
    """Validate recorded arrays and map each sim frame to a preceding command."""
    with np.load(sim_path, allow_pickle=False) as archive:
        sim = {key: archive[key] for key in archive.files}
    with np.load(control_path, allow_pickle=False) as archive:
        control = {key: archive[key] for key in archive.files}
    for source, key in ((sim, "monotonic_ns"), (control, "capture_monotonic_ns")):
        times = source[key]
        if (times.ndim != 1 or not len(times) or times.dtype.kind not in "iu"
                or np.any(times[1:] <= times[:-1])):
            raise ValueError(f"{key} must be a nonempty, strictly increasing integer array")
    times, captured = sim["monotonic_ns"], control["capture_monotonic_ns"]
    if times[-1] < captured[0] or times[0] > captured[-1]:
        raise ValueError("Simulation and control capture do not overlap on the monotonic clock")
    names = sim["joint_names"]
    if names.shape != (29,) or names.dtype.kind not in "US" or len(set(names)) != 29:
        raise ValueError("Expected 29 unique body joint names")
    for key, shape in (("root_position", (len(times), 3)),
                       ("root_quaternion_wxyz", (len(times), 4)),
                       ("body_joint_position", (len(times), 29))):
        value = sim[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or non-finite values: {key}")
    if not np.allclose(np.linalg.norm(sim["root_quaternion_wxyz"], axis=1), 1.0, atol=1e-5):
        raise ValueError("Root quaternions must be unit length")
    fps = sim["fps"]
    if fps.shape != () or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite scalar")
    for side in ("left", "right"):
        key = f"last_{side}_hand_action"
        value = control[key]
        if (value.shape not in {(len(captured), 6), (len(captured), 7)}
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid shape or non-finite values: {key}")
    # The recorder can start just before the first control capture. Preserve all
    # measured body frames and explicitly report boundary command clamping.
    indices = np.clip(np.searchsorted(captured, times, side="right") - 1, 0, len(captured) - 1)
    return sim, control, indices


def hand_mapping(runtime_root: Path):
    root = runtime_root.resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("gear_sonic.utils.mujoco_sim.brainco_hand_mapping")
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"BrainCo mapping was imported outside the specified runtime: {source}")
    return module, source


def set_hand_color(model, color):
    """Apply a visual-only override, copying the pelvis shell's gray material."""
    if color == "original":
        return {"mode": color, "geom_ids": []}
    if color != "body-gray":
        raise ValueError(f"Unknown hand color: {color}")
    pelvis = model.body("pelvis").id
    mesh = model.mesh("pelvis").id
    source = next((g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == pelvis and model.geom_dataid[g] == mesh), None)
    if source is None:
        raise ValueError("No pelvis visual geometry available for body-gray")
    material = int(model.geom_matid[source])
    rgba = (model.mat_rgba[material] if material >= 0 else model.geom_rgba[source]).copy()
    roots = {model.body(f"{side}_wrist_yaw_link").id for side in ("left", "right")}
    changed = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body not in roots:
            body = int(model.body_parentid[body])
        if body in roots:
            model.geom_matid[geom] = -1
            model.geom_rgba[geom] = rgba
            changed.append(geom)
    return {"mode": color, "rgba": rgba.tolist(), "source_geom_id": source,
            "source_body": "pelvis", "geom_ids": changed}


def render(args):
    # Import optional rendering dependencies only after argparse, so --help does
    # not need a simulator, graphics context, robot assets or controller install.
    import av
    import mujoco

    if args.output_dir.exists():
        raise ValueError("Use a fresh --output-dir; existing results are never overwritten")
    sim, control, indices = load_telemetry(args.sim, args.control)
    mapping, mapping_source = hand_mapping(args.runtime_root)
    scene = args.scene.resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    hand_color = set_hand_color(model, args.hand_color)
    data = mujoco.MjData(model)
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(roots) != 1:
        raise ValueError("Expected exactly one floating root in the supplied scene")
    root = int(model.jnt_qposadr[roots[0]])
    names = [name.decode() if isinstance(name, bytes) else str(name) for name in sim["joint_names"]]
    body_ids = [model.joint(name).id for name in names]
    if any(model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE for joint in body_ids):
        raise ValueError("All recorded body joints must be hinge joints")
    body = model.jnt_qposadr[body_ids]
    hands = {}
    for side in ("left", "right"):
        actuators = mapping.brainco_hand_actuator_indices(model, side)
        hands[side] = model.jnt_qposadr[model.actuator_trnid[actuators, 0]]
    av.codec.Codec("libx264", "w")  # fail before creating output if encoder is unavailable
    args.output_dir.mkdir(parents=True)
    video = args.output_dir / "replay_mujoco.mp4"
    fps = Fraction(str(float(sim["fps"]))).limit_denominator(100000)
    model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = args.azimuth, args.elevation, args.distance
    body_delta = root_delta = 0.0
    with av.open(str(video), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = args.width, args.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        stream.codec_context.thread_count = 4
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for frame, ctrl_index in enumerate(indices):
                data.qpos[root:root + 3] = sim["root_position"][frame]
                data.qpos[root + 3:root + 7] = sim["root_quaternion_wxyz"][frame]
                data.qpos[body] = sim["body_joint_position"][frame]
                for side, addresses in hands.items():
                    command = control[f"last_{side}_hand_action"][ctrl_index, :6]
                    data.qpos[addresses] = mapping.brainco_command_to_finger_targets(command)
                mujoco.mj_forward(model, data)  # FK only: never mj_step or a controller call
                body_delta = max(body_delta, float(np.max(np.abs(data.qpos[body] - sim["body_joint_position"][frame]))))
                expected_root = np.r_[sim["root_position"][frame], sim["root_quaternion_wxyz"][frame]]
                root_delta = max(root_delta, float(np.max(np.abs(data.qpos[root:root + 7] - expected_root))))
                camera.lookat[:] = sim["root_position"][frame]
                camera.lookat[2] = max(float(camera.lookat[2]), 0.75)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                encoded_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                encoded_frame.pts = frame
                encoded_frame.time_base = 1 / fps
                for packet in stream.encode(encoded_frame):
                    container.mux(packet)
                if frame % 120 == 0:
                    print(f"Rendered {frame + 1}/{len(indices)} frames", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        decoded_frames = sum(1 for _ in container.decode(video=0))
    if decoded_frames != len(indices):
        raise ValueError(f"Video frame count mismatch: {decoded_frames} != {len(indices)}")
    times, captured = sim["monotonic_ns"], control["capture_monotonic_ns"]
    report = {
        "status": "complete", "video": str(video.resolve()), "video_sha256": digest(video),
        "sim_telemetry": str(args.sim.resolve()), "sim_sha256": digest(args.sim),
        "control_telemetry": str(args.control.resolve()), "control_sha256": digest(args.control),
        "scene": str(scene), "scene_sha256": digest(scene),
        "hand_mapping": str(mapping_source), "hand_mapping_sha256": digest(mapping_source),
        "versions": {"mujoco": mujoco.__version__, "av": av.__version__},
        "hand_color": hand_color,
        "width": args.width, "height": args.height, "fps": float(fps),
        "frames": len(indices), "decoded_frames": decoded_frames, "codec": "h264", "pixel_format": "yuv420p",
        "body_q_max_delta_rad": body_delta, "root_qpos_max_delta": root_delta,
        "camera": {"azimuth": args.azimuth, "elevation": args.elevation, "distance": args.distance,
                   "lookat": "measured root, z clipped below at 0.75m"},
        "body_source": "measured replay root/body qpos; no new physics or controller inference",
        "finger_source": "canonical mapping of preceding captured BrainCo command, not measured finger qpos",
        "command_boundary_frames": {"before_first_capture": int(np.sum(times < captured[0])),
                                    "after_last_capture": int(np.sum(times > captured[-1]))},
        "timebase": "one output frame per sim sample, at recorded fps; monotonic clock pairs hand commands only",
        "argv": sys.argv,
    }
    (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Video: {video}\nReport: {args.output_dir / 'render_report.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True, help="Selected replay's episode_*.sim.npz")
    parser.add_argument("--control", type=Path, required=True, help="Same replay's episode_*.control.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for MP4 and report")
    parser.add_argument("--runtime-root", type=Path, default=os.environ.get("EGOALIGN_RUNTIME_ROOT"))
    default_scene = os.environ.get("EGOALIGN_BRAINCO_MODEL_DIR")
    parser.add_argument("--scene", type=Path, default=Path(default_scene) / "scene_29dof_brainco.xml" if default_scene else None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--azimuth", type=float, default=120.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--hand-color", choices=("original", "body-gray"), default="original",
                        help="Visual-only hand color; body-gray copies the pelvis shell")
    args = parser.parse_args(argv)
    if args.runtime_root is None or args.scene is None:
        parser.error("Set EGOALIGN_RUNTIME_ROOT and EGOALIGN_BRAINCO_MODEL_DIR, or pass --runtime-root and --scene")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("H.264 width/height must be positive even integers")
    if not all(np.isfinite(value) for value in (args.azimuth, args.elevation, args.distance)) or args.distance <= 0:
        parser.error("Camera angles must be finite and distance must be positive")
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check a single-episode final export against captured control telemetry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scripts.human_data.processing.apply_sonic_mujoco_feedback import (
    BODY_STATE_INDICES, LEFT_HAND_STATE_INDICES, RIGHT_HAND_STATE_INDICES,
    physical_qpos_to_isaaclab_offset_state,
)


def _camera_inventory(root, table):
    """Check files against metadata and any explicit per-row video references."""
    info = json.loads((root / "meta/info.json").read_text())
    cameras = {key for key, feature in info["features"].items() if feature.get("dtype") == "video"}
    if not cameras:
        raise ValueError("No camera features in metadata")
    chunks_size = int(info["chunks_size"])
    if chunks_size <= 0:
        raise ValueError("Invalid chunks_size")
    episodes = set(table["episode_index"].to_pylist())
    if len(episodes) != 1:
        raise ValueError("Sample validator requires one episode_index")
    episode = next(iter(episodes))
    expected_by_key = {}
    def relative_video(value):
        path = Path(value)
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Camera path escapes dataset: {value}")
        return path.as_posix()
    for key in cameras:
        expected_by_key[key] = relative_video(info["video_path"].format(
            episode_chunk=episode // chunks_size, episode_index=episode, video_key=key))
    expected = set(expected_by_key.values())
    if len(expected) != len(cameras):
        raise ValueError("Camera metadata maps multiple features to the same file")
    actual = {p.relative_to(root).as_posix() for p in (root / "videos").rglob("*.mp4")}
    if actual != expected:
        raise ValueError(f"Camera file set mismatch: missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")
    for name in table.column_names:
        if not name.startswith("observation.images."):
            continue
        if name not in cameras:
            raise ValueError(f"Camera column absent from metadata: {name}")
        for value in table[name].to_pylist():
            reference = value.get("path") if isinstance(value, dict) else value
            if not isinstance(reference, str) or relative_video(reference) != expected_by_key[name]:
                raise ValueError(f"Invalid camera reference in {name}: {reference}")
    return {"files": actual, "features": {key: info["features"][key] for key in cameras},
            "video_path": info["video_path"], "chunks_size": chunks_size, "episode": episode}


def verify(stage, final, control):
    stage, final, control = Path(stage), Path(final), Path(control)
    def single(root):
        files = list((root / "data").rglob("*.parquet"))
        if len(files) != 1:
            raise ValueError("Sample validator requires exactly one parquet episode")
        return pq.read_table(files[0])
    source, table = single(stage), single(final)
    token = np.asarray(table["action.motion_token"].to_pylist())
    state = np.asarray(table["observation.state"].to_pylist())
    n = len(table)
    if n == 0 or len(source) != n or state.shape != (n,43) or token.shape != (n,64):
        raise ValueError("Invalid frame counts or state/token dimensions")
    if not np.isfinite(state).all() or not np.isfinite(token).all():
        raise ValueError("Non-finite exported state/token")
    with np.load(control, allow_pickle=False) as captured:
        payloads = {}
        for name, width in (("body_q", len(BODY_STATE_INDICES)),
                            ("left_hand_q", len(LEFT_HAND_STATE_INDICES)),
                            ("right_hand_q", len(RIGHT_HAND_STATE_INDICES)),
                            ("token_state", 64)):
            if name not in captured:
                raise ValueError(f"Missing captured payload: {name}")
            values = captured[name]
            if values.shape != (n, width):
                raise ValueError(f"Invalid captured {name} shape: {values.shape}; expected {(n, width)}")
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError(f"Captured {name} must contain finite real numbers")
            payloads[name] = values
        physical = np.zeros((n,43))
        physical[:, BODY_STATE_INDICES] = payloads["body_q"]
        physical[:, LEFT_HAND_STATE_INDICES] = payloads["left_hand_q"]
        physical[:, RIGHT_HAND_STATE_INDICES] = payloads["right_hand_q"]
        expected = physical_qpos_to_isaaclab_offset_state(physical)
        token_delta = float(np.max(np.abs(token - payloads["token_state"])))
        state_delta = float(np.max(np.abs(state - expected)))
    if not np.isfinite([token_delta,state_delta]).all() or token_delta != 0 or state_delta > 1e-7:
        raise ValueError(f"Capture mismatch: token={token_delta}, state={state_delta}")
    unchanged = []
    for name in source.column_names:
        if name in {"timestamp","teleop.left_hand_joints","teleop.right_hand_joints"} or name.startswith("observation.images."):
            if table[name].to_pylist() != source[name].to_pylist():
                raise ValueError(f"Changed aligned field: {name}")
            unchanged.append(name)
    source_cameras = _camera_inventory(stage, source)
    final_cameras = _camera_inventory(final, table)
    if source_cameras != final_cameras:
        raise ValueError("Changed camera metadata or episode identity")
    hashes = {}
    for relative in sorted(final_cameras["files"]):
        digest = hashlib.sha256((final / relative).read_bytes()).hexdigest()
        if digest != hashlib.sha256((stage / relative).read_bytes()).hexdigest():
            raise ValueError(f"Changed camera file: {relative}")
        hashes[relative] = digest
    return {"passed": True, "frames": n, "state_dimension":43, "token_dimension":64,
            "token_max_delta":token_delta, "state_max_delta":state_delta,
            "unchanged_columns":unchanged, "camera_sha256":hashes,
            "scope":"single-episode payload pairing; not a replacement for timing/rollout/production gates"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a",type=Path,required=True)
    p.add_argument("--final",type=Path,required=True)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--output",type=Path,help="Optional JSON verification report")
    a=p.parse_args()
    report = json.dumps(verify(a.stage_a,a.final,a.control),indent=2)
    if a.output:
        a.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()

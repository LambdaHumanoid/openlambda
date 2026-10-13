"""
Encode human SMPL data into motion tokens using SONIC's encoder ONNX model.

Reads smpl_joints, body_quat_w, left/right_wrist_joints from final HDF5,
constructs encoder observation vectors, runs onnxruntime inference, and
writes motion_token (N, 64) back to HDF5.

Usage:
    python encode_motion_tokens.py \
        --data-dir data_collection/body_data_final \
        --encoder-model gear_sonic_deploy/policy/release/model_encoder.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

try:
    from .wrist_contract import interleave_wrist_positions
except ImportError:
    from wrist_contract import interleave_wrist_positions

ENCODER_INPUT_DIM = 1762
ENCODER_OUTPUT_DIM = 64
SMPL_MODE_ID = 2

# Offsets from observation_config.yaml encoder_observations order
OFF_ENCODER_MODE = 0          # dim 4
OFF_SMPL_JOINTS = 922         # dim 720 = 24*3*10
OFF_SMPL_ANCHOR_ORI = 1642   # dim 60  = 6*10
OFF_WRIST_JOINTS = 1702      # dim 60  = 6*10

NUM_FUTURE_FRAMES = 10
SMPL_JOINTS_PER_FRAME = 72   # 24 joints × 3
ANCHOR_ORI_PER_FRAME = 6     # rot6d
WRIST_PER_FRAME = 6          # 3 left + 3 right


def _quat_inv_wxyz(q):
    """Invert quaternion (wxyz format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_wxyz(a, b):
    """Hamilton product of two quaternions (wxyz format)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_to_rot6d_rowmajor(q_wxyz):
    """Convert wxyz quaternion to 6D rotation (row-major flatten of R[:,:2]).

    This matches the C++ encoder format (g1_deploy_onnx_ref.cpp:691):
      [R00, R01, R10, R11, R20, R21]

    NOT the same as body_transforms.py which uses transpose format.
    """
    q_xyzw = q_wxyz[[1, 2, 3, 0]]
    R = sRot.from_quat(q_xyzw).as_matrix()
    return R[:, :2].flatten().astype(np.float32)


def build_encoder_observations(
    smpl_joints: np.ndarray,
    body_quat_w: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    """Build encoder input observations for all frames (vectorized).

    Args:
        smpl_joints: (N, 72) float32
        body_quat_w: (N, 4) float32 wxyz
        left_wrist: (N, 3) float32
        right_wrist: (N, 3) float32

    Returns:
        (N, 1762) float32 encoder input
    """
    N = smpl_joints.shape[0]
    obs = np.zeros((N, ENCODER_INPUT_DIM), dtype=np.float32)

    # encoder_mode_4: [SMPL_MODE_ID, 0, 0, 0]
    obs[:, OFF_ENCODER_MODE] = float(SMPL_MODE_ID)

    wrist_joints = interleave_wrist_positions(left_wrist, right_wrist)
    if wrist_joints.shape[0] != N:
        raise ValueError(
            f"wrist row count {wrist_joints.shape[0]} != SMPL row count {N}"
        )

    for t in range(N):
        base_quat = body_quat_w[t]

        for fi in range(NUM_FUTURE_FRAMES):
            f = min(t + fi, N - 1)

            # smpl_joints
            start = OFF_SMPL_JOINTS + fi * SMPL_JOINTS_PER_FRAME
            obs[t, start:start + SMPL_JOINTS_PER_FRAME] = smpl_joints[f]

            # smpl_anchor_orientation (relative rotation from current to future)
            rel_quat = _quat_mul_wxyz(_quat_inv_wxyz(base_quat), body_quat_w[f])
            start = OFF_SMPL_ANCHOR_ORI + fi * ANCHOR_ORI_PER_FRAME
            obs[t, start:start + ANCHOR_ORI_PER_FRAME] = _quat_to_rot6d_rowmajor(rel_quat)

            # wrist joint positions
            start = OFF_WRIST_JOINTS + fi * WRIST_PER_FRAME
            obs[t, start:start + WRIST_PER_FRAME] = wrist_joints[f]

    return obs


def process_episode(
    hdf5_path: Path,
    session,
) -> tuple[str, int, str | None]:
    """Process a single episode: encode motion tokens and write to HDF5."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            required = ["smpl_joints", "body_quat_w", "left_wrist_joints", "right_wrist_joints"]
            for key in required:
                if key not in f:
                    return str(hdf5_path), 0, f"Missing field: {key}"

            smpl_joints = np.array(f["smpl_joints"][:], dtype=np.float32)
            body_quat_w = np.array(f["body_quat_w"][:], dtype=np.float32)
            left_wrist = np.array(f["left_wrist_joints"][:], dtype=np.float32)
            right_wrist = np.array(f["right_wrist_joints"][:], dtype=np.float32)

        N = smpl_joints.shape[0]
        obs_all = build_encoder_observations(smpl_joints, body_quat_w, left_wrist, right_wrist)

        # Inference (model has fixed batch_dim=1)
        motion_tokens = np.zeros((N, ENCODER_OUTPUT_DIM), dtype=np.float32)
        for i in range(N):
            result = session.run(None, {"obs_dict": obs_all[i:i+1]})
            motion_tokens[i] = result[0][0]

        # Write to HDF5
        with h5py.File(hdf5_path, "a") as f:
            if "motion_token" in f:
                del f["motion_token"]
            f.create_dataset("motion_token", data=motion_tokens, compression="gzip", compression_opts=4)

        return str(hdf5_path), N, None

    except Exception as e:
        import traceback
        return str(hdf5_path), 0, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(description="Encode motion tokens from human SMPL data")
    parser.add_argument("--data-dir", required=True, help="Directory with episode_*.hdf5")
    parser.add_argument("--encoder-model", required=True, help="Path to model_encoder.onnx")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import onnxruntime as ort

    data_dir = Path(args.data_dir)
    episode_files = sorted(data_dir.glob("episode_*.hdf5"), key=lambda x: int(x.stem.split("_")[-1]))

    if not episode_files:
        print(f"No episode_*.hdf5 found in {data_dir}")
        return

    print(f"Loading encoder model: {args.encoder_model}")
    session = ort.InferenceSession(args.encoder_model, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input: {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")

    print(f"\nProcessing {len(episode_files)} episodes (batch_size={args.batch_size})")

    total_frames = 0
    for ep_file in tqdm(episode_files, desc="Encoding motion tokens"):
        path, n_frames, err = process_episode(ep_file, session)
        if err:
            print(f"  Error {Path(path).name}: {err}")
        else:
            total_frames += n_frames

    print(f"\nDone. Encoded {total_frames} frames across {len(episode_files)} episodes.")


if __name__ == "__main__":
    main()

"""Camera observations and independent action clocks; all times are episode-relative seconds."""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def future_action_indices(camera_times, action_times, horizon=50):
    camera_times = np.asarray(camera_times, dtype=np.float64)
    action_times = np.asarray(action_times, dtype=np.float64)
    if horizon <= 0 or not len(action_times):
        raise ValueError("A positive horizon and nonempty action timeline are required.")
    if not np.isfinite(camera_times).all() or not np.isfinite(action_times).all():
        raise ValueError("Nonfinite timestamps.")
    if np.any(np.diff(camera_times) <= 0) or np.any(np.diff(action_times) <= 0):
        raise ValueError("Camera and action timestamps must be strictly increasing.")
    # Timestamp equality is not a future action. The tolerance handles float roundoff only.
    starts = np.searchsorted(action_times, camera_times + 1e-9, side="right")
    return np.minimum(starts[:, None] + np.arange(horizon), len(action_times) - 1)


def episode_path(meta, item, key):
    i = int(item["episode_index"])
    return Path(meta["root_path"]) / meta[key].format(
        episode_index=i, episode_chunk=i // int(meta.get("chunks_size", 1000)),
        video_key="observation.images.front")


def load_camera_episode(meta, item):
    if float(meta.get("action_fps", 0)) != 50:
        raise ValueError("Camera-anchored posttraining requires 50 Hz actions.")
    obs = pq.read_table(episode_path(meta, item, "data_path")).to_pydict()
    act = pq.read_table(episode_path(meta, item, "action_path")).to_pydict()
    state = np.asarray(obs["observation.state"], np.float32)
    sm = np.asarray(obs["observation.valid_mask"], bool)
    action = np.asarray(act["action"], np.float32)
    am = np.asarray(act["action.valid_mask"], bool)
    ct = np.asarray(obs["timestamp"], np.float64)
    at = np.asarray(act["timestamp"], np.float64)
    video_indices = np.asarray(obs["video_frame_index"], np.int64)
    if state.shape != (item["length"], 44) or action.shape != (item["action_length"], 76):
        raise ValueError("Camera/action shape disagrees with episode metadata.")
    if sm.shape != state.shape or am.shape != action.shape:
        raise ValueError("Invalid validity masks.")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Nonfinite state/action values.")
    if np.any(state[~sm] != 0) or np.any(action[~am] != 0):
        raise ValueError("Invalid dimensions must be zero.")
    if len(ct) != len(state) or len(at) != len(action) or video_indices.shape != ct.shape:
        raise ValueError("Invalid timestamp/video index lengths.")
    if np.any(video_indices < 0) or np.any(np.diff(video_indices) <= 0):
        raise ValueError("Observation frames must be unique and ordered.")
    future_action_indices(ct, at)
    return state, action, sm, am, ct, at, video_indices


def camera_valid_observations(state_mask, action_mask, camera_times, action_times, horizon=50):
    indices = future_action_indices(camera_times, action_times, horizon)
    sampled_times = np.asarray(action_times)[indices]
    # Retain measured robot timing, but never train through dropped control ticks.
    # Repeated tail targets have dt=0 and remain supervised.
    continuous = (np.diff(sampled_times, axis=1) <= .030).all(axis=1)
    continuous &= sampled_times[:, 0] - np.asarray(camera_times) <= .030
    return np.flatnonzero(state_mask.any(axis=1) & action_mask[indices].any(axis=(1, 2)) & continuous).tolist()

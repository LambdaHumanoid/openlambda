"""Pure BrainCo mapping extracted without device/network services."""
from __future__ import annotations
import numpy as np
_OPENXR_HAND_CHAINS = {
    "thumb": (2, 3, 4, 5),
    "index": (6, 7, 8, 9, 10),
    "middle": (11, 12, 13, 14, 15),
    "ring": (16, 17, 18, 19, 20),
    "pinky": (21, 22, 23, 24, 25),
}

_LONG_FINGER_MAX_CURL_RAD = 2.8

_THUMB_MAX_CURL_RAD = 1.6

_BRAINCO_THUMB_SIGNAL_GAIN = 1.5

_BRAINCO_THUMB_AUX_SIGNAL_GAIN = 1.15

def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def _coerce_openxr_hand_state(hand_state) -> np.ndarray | None:
    if hand_state is None:
        return None
    try:
        arr = np.asarray(hand_state, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    elif arr.ndim == 1 and arr.size >= 26 * 7:
        arr = arr.reshape(-1, 7)
    if arr.ndim != 2 or arr.shape[0] < 26 or arr.shape[1] < 3:
        return None
    arr = arr[:26]
    if not np.isfinite(arr[:, :3]).all():
        return None
    if float(np.linalg.norm(arr[:, :3])) < 1e-6:
        return None
    return arr

def _segment_angle_rad(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    cosine = float(np.dot(v0, v1) / (n0 * n1))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _chain_curl_from_positions(positions: np.ndarray, joint_indices, max_total_angle: float) -> float:
    pts = positions[np.asarray(joint_indices, dtype=np.int64)]
    total = 0.0
    for i in range(len(pts) - 2):
        total += _segment_angle_rad(pts[i + 1] - pts[i], pts[i + 2] - pts[i + 1])
    return _clip01(total / max_total_angle)

def _chain_length(positions: np.ndarray, joint_indices) -> float:
    pts = positions[np.asarray(joint_indices, dtype=np.int64)]
    total = 0.0
    for i in range(len(pts) - 1):
        total += float(np.linalg.norm(pts[i + 1] - pts[i]))
    return total

def _distance_opposition(distance: float, hand_scale: float) -> float:
    return _clip01((1.05 - (distance / hand_scale)) / 0.70)

def _thumb_opposition_from_positions(positions: np.ndarray, long_finger_curl: float) -> float:
    thumb_tip = positions[5]
    index_len = _chain_length(positions, _OPENXR_HAND_CHAINS["index"])
    middle_len = _chain_length(positions, _OPENXR_HAND_CHAINS["middle"])
    palm_width = float(np.linalg.norm(positions[6] - positions[21]))
    hand_scale = max(index_len, middle_len, palm_width * 1.5, 1e-4)

    fingertip_targets = positions[np.asarray((10, 15), dtype=np.int64)]
    fingertip_distance = float(np.min(np.linalg.norm(fingertip_targets - thumb_tip, axis=1)))
    fingertip_opposition = _distance_opposition(fingertip_distance, hand_scale)

    chain_targets = positions[
        np.asarray((7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20), dtype=np.int64)
    ]
    chain_distance = float(np.min(np.linalg.norm(chain_targets - thumb_tip, axis=1)))
    chain_opposition = _distance_opposition(chain_distance, hand_scale)
    chain_gate = _clip01((long_finger_curl - 0.15) / 0.35)
    return max(fingertip_opposition, chain_gate * chain_opposition)

def extract_openxr_hand_features_for_brainco(hand_state) -> np.ndarray:
    """Extract 6 normalized curl/opposition features from a Pico/OpenXR hand."""
    arr = _coerce_openxr_hand_state(hand_state)
    if arr is None:
        raise ValueError("invalid OpenXR hand tracking state")
    positions = arr[:, :3].astype(np.float32, copy=False)

    index = _chain_curl_from_positions(
        positions, _OPENXR_HAND_CHAINS["index"], _LONG_FINGER_MAX_CURL_RAD
    )
    middle = _chain_curl_from_positions(
        positions, _OPENXR_HAND_CHAINS["middle"], _LONG_FINGER_MAX_CURL_RAD
    )
    ring = _chain_curl_from_positions(
        positions, _OPENXR_HAND_CHAINS["ring"], _LONG_FINGER_MAX_CURL_RAD
    )
    pinky = _chain_curl_from_positions(
        positions, _OPENXR_HAND_CHAINS["pinky"], _LONG_FINGER_MAX_CURL_RAD
    )
    thumb_curl = _chain_curl_from_positions(
        positions, _OPENXR_HAND_CHAINS["thumb"], _THUMB_MAX_CURL_RAD
    )
    long_finger_curl = max(index, middle, ring, pinky)
    thumb_opposition = _thumb_opposition_from_positions(positions, long_finger_curl)

    thumb_signal = max(thumb_curl, 0.75 * thumb_opposition)
    thumb_aux_signal = max(0.4 * thumb_curl, thumb_opposition)
    return np.array([thumb_signal, thumb_aux_signal, index, middle, ring, pinky], dtype=np.float32)

def _brainco_features_to_default_cmd(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32).reshape(6)
    thumb = 0.5 * _clip01(_BRAINCO_THUMB_SIGNAL_GAIN * float(features[0]))
    thumb_aux = 0.5 * _clip01(_BRAINCO_THUMB_AUX_SIGNAL_GAIN * float(features[1]))
    return np.array(
        [
            thumb,
            thumb_aux,
            _clip01(float(features[2])),
            _clip01(float(features[3])),
            _clip01(float(features[4])),
            _clip01(float(features[5])),
            0.0,
        ],
        dtype=np.float32,
    )

def retarget_openxr_hand_to_brainco(hand_state) -> np.ndarray:
    """Map Pico/OpenXR 26-joint hand tracking to BrainCo Revo2 6-DOF cmd + pad.

    Output is the SONIC 7-float hand wire format:
    [thumb, thumb_aux, index, middle, ring, pinky, 0].
    """
    return _brainco_features_to_default_cmd(extract_openxr_hand_features_for_brainco(hand_state))

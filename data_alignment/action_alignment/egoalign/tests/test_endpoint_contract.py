import numpy as np
import pytest
from scripts.human_data.smpl_endpoint_retargeting import (
    endpoint_retarget_contract, optimize_virtual_smpl_trajectory,
    DEFAULT_OPTIMIZED_JOINT_INDICES, ENDPOINT_RETARGET_ALGORITHM,
)
from scripts.human_data.evaluate_endpoint_retarget_replay import (
    next_correction, error_metrics, estimate_lag_frames, evaluate_early_stop_gate,
    EarlyStopThresholds,
)

def test_identity_and_whitelist():
    contract = endpoint_retarget_contract("hands", "both")
    assert "egoalign" in ENDPOINT_RETARGET_ALGORITHM
    assert contract.optimized_joint_indices == (3, 6, 9, 13, 14, 16, 17, 18, 19)
    for part, side in [("feet", "both"), ("hands", "right"), ("both", "both")]:
        with pytest.raises(ValueError):
            endpoint_retarget_contract(part, side)
    with pytest.raises(ValueError):
        endpoint_retarget_contract("hands", "both", experimental_lock_spine=True)


def test_evaluation_rejects_unsupported_hand_side():
    points = np.zeros((30, 3))
    with pytest.raises(ValueError):
        estimate_lag_frames(points, points, points, points,
                            max_lag_frames=2, hand_endpoint_side="right")
    with pytest.raises(ValueError):
        error_metrics(points, points, points, points, np.ones(30, bool),
                      hand_endpoint_side="right")
    with pytest.raises(ValueError):
        evaluate_early_stop_gate(
            baseline_metrics={}, metrics={}, dynamics={}, lag_frames=0,
            max_lag_frames=20, round_index=0, thresholds=EarlyStopThresholds(),
            hand_endpoint_side="right",
        )


@pytest.mark.parametrize("override", [
    {"hand_endpoint_side": "right"},
    {"optimized_joint_indices": (1, 3, 6, 9, 13, 14, 16, 17, 18, 19)},
])
def test_optimizer_rejects_unsupported_joint_or_side_override(override):
    with pytest.raises(ValueError):
        optimize_virtual_smpl_trajectory(
            baseline_pose=np.zeros((2, 63)), virtual_rest_joints=np.zeros((55, 3)),
            parents=[-1] + [0] * 54, target_left=np.zeros((2, 3)),
            target_right=np.zeros((2, 3)), iterations=1, device="cpu", **override,
        )

def test_three_rounds_preserve_all_non_upper_body_joints():
    # Synthetic skeleton: no licensed human model is needed for this invariant.
    rng = np.random.default_rng(7)
    rest = rng.normal(0, .1, (55, 3)).astype(np.float32)
    parents = [-1] + [0] * 54
    for j in (25, 28, 31, 34):
        parents[j] = 18
    for j in (40, 43, 46, 49):
        parents[j] = 19
    baseline = rng.normal(0, .02, (4, 63)).astype(np.float32)
    initial = None
    locked = [j for j in range(21) if j + 1 not in DEFAULT_OPTIMIZED_JOINT_INDICES]
    for round_index in range(3):
        result = optimize_virtual_smpl_trajectory(
            baseline_pose=baseline, virtual_rest_joints=rest, parents=parents,
            target_left=np.full((4, 3), .1 + round_index * .01),
            target_right=np.full((4, 3), .2 + round_index * .01),
            initial_pose=initial, iterations=2, device="cpu",
        )
        np.testing.assert_array_equal(result.pose.reshape(4, 21, 3)[:, locked], baseline.reshape(4, 21, 3)[:, locked])
        assert np.isfinite(result.pose).all()
        initial = result.pose

def test_residual_correction_is_bounded():
    correction, _ = next_correction(
        target=np.ones((30, 3)), actual_aligned=np.zeros((30, 3)),
        valid=np.ones(30, bool), current_correction=np.zeros((30, 3)),
        previous_actual_aligned=None, previous_correction=None,
        alpha=.5, max_increment_m=.03, max_total_m=.08,
    )
    assert np.linalg.norm(correction, axis=1).max() <= .030000001

def test_live_device_module_is_pure():
    from egoalign.robots.g1.hand_mapping import retarget_openxr_hand_to_brainco
    assert callable(retarget_openxr_hand_to_brainco)

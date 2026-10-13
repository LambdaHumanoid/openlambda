import numpy as np
import pytest
import torch
from egoalign.staged_ik import CONFIG, SPINE_JOINTS, ARM_JOINTS, release_gate, validate_proof
from scripts.human_data.smpl_endpoint_retargeting import (
    optimize_virtual_smpl_trajectory, forward_joints_torch,
    output_frame_joints_torch, palm_centers_torch,
)


def skeleton():
    parents = [-1] + [0] * 54
    rest = np.zeros((55, 3), np.float32)
    for j, p, v in [(3,0,[0,0,.15]), (6,3,[0,0,.15]), (9,6,[0,0,.15]),
                    (13,9,[0,.1,0]), (14,9,[0,-.1,0]),
                    (16,13,[0,.1,0]), (17,14,[0,-.1,0]),
                    (18,16,[0,.25,0]), (19,17,[0,-.25,0]),
                    (20,18,[0,.22,0]), (21,19,[0,-.22,0])]:
        parents[j] = p
        rest[j] = rest[p] + v
    for side, js in [(20,[25,28,31,34]), (21,[40,43,46,49])]:
        for j in js:
            parents[j] = side
            rest[j] = rest[side] + [0,.05 if side==20 else -.05,0]
    return rest, parents


def endpoints(pose, rest, parents):
    with torch.no_grad():
        j = output_frame_joints_torch(forward_joints_torch(torch.from_numpy(pose.reshape(-1,63)), torch.from_numpy(rest), parents))
        return torch.stack(palm_centers_torch(j), 1).numpy()


def test_gate_never_releases_reachable_frames_and_tapers():
    errors = np.full((51,2), .08)
    errors[0] = errors[25] = errors[-1] = .02
    gate = release_gate(errors)
    assert np.all(gate[[0,25,50]] == 0)
    assert gate.max() == 1
    assert abs(np.diff(gate)).max() <= CONFIG.gate_max_step + 1e-7
    with pytest.raises(ValueError):
        release_gate([[float('nan'), .1]])


def test_reachable_large_twist_preserved_across_rounds():
    rest, parents = skeleton()
    base = np.zeros((5,21,3),np.float32)
    base[:,np.array(SPINE_JOINTS)-1,2] = np.linspace(.2,.8,5)[:,None]
    target = endpoints(base,rest,parents)
    # A previous feedback round may have assisted the spine. Stage 1 must
    # discard that correction and restore each original frame before solving.
    initial = base.copy()
    initial[:, np.array(SPINE_JOINTS)-1, 0] += .08
    for _ in range(3):
        r = optimize_virtual_smpl_trajectory(baseline_pose=base,virtual_rest_joints=rest,
            parents=parents,target_left=target[:,0],target_right=target[:,1],
            initial_pose=initial,iterations=1,device='cpu')
        np.testing.assert_array_equal(r.pose,base.reshape(-1,63))
        assert np.all(r.staged_proof['staged_spine_release_gate']==0)
        initial = r.pose


def test_unreached_targets_bounded_assistance_and_tamper_rejected():
    rest, parents = skeleton()
    base = np.zeros((7,21,3),np.float32)
    base[:,np.array(SPINE_JOINTS)-1,2] = .3
    target = endpoints(base,rest,parents)
    target[:,:,1] += .18
    r = optimize_virtual_smpl_trajectory(baseline_pose=base,virtual_rest_joints=rest,
        parents=parents,target_left=target[:,0],target_right=target[:,1],iterations=80,device='cpu')
    proof=r.staged_proof
    assert np.any(proof['staged_spine_release_gate']>0)
    delta=r.pose.reshape(base.shape)-base
    locked=[j for j in range(21) if j+1 not in SPINE_JOINTS+ARM_JOINTS]
    assert np.all(delta[:,locked]==0)
    assert np.linalg.norm(delta[:,np.array(SPINE_JOINTS)-1],axis=-1).max()<=.100001
    assert proof['staged_final_errors_m'].mean() < proof['staged_arm_stage_errors_m'].mean()
    assert proof['staged_selected_assistance_iteration']>0
    changed=r.pose.copy().reshape(base.shape)
    changed[0,2,0]+=1
    with pytest.raises(ValueError):
        validate_proof(proof,base,changed,target)
    bad=dict(proof,staged_config_json=np.asarray('{}'))
    with pytest.raises(ValueError):
        validate_proof(bad,base,r.pose,target)


def test_original_clavicles_preserved_with_dirty_warm_start_and_proof_checked():
    rest, parents = skeleton()
    base = np.zeros((5, 21, 3), np.float32)
    base[:, [12, 13], 0] = np.linspace(-.15, .2, 5)[:, None]
    target = endpoints(base, rest, parents)
    target[:, :, 1] += .18
    initial = base.copy()
    initial[:, [12, 13], :] += .2
    r = optimize_virtual_smpl_trajectory(
        baseline_pose=base, virtual_rest_joints=rest, parents=parents,
        target_left=target[:, 0], target_right=target[:, 1],
        initial_pose=initial, iterations=20, device='cpu')
    proof = r.staged_proof
    assert np.any(proof['staged_spine_release_gate'] > 0)
    for pose in (proof['staged_arm_stage_pose'], r.pose):
        np.testing.assert_array_equal(pose.reshape(base.shape)[:, [12, 13]], base[:, [12, 13]])
    changed = r.pose.copy().reshape(base.shape)
    changed[0, 12, 0] += .01
    with pytest.raises(ValueError, match='clavicle'):
        validate_proof(proof, base, changed, target)
    arm = proof['staged_arm_stage_pose'].copy().reshape(base.shape)
    arm[0, 13, 1] += .01
    with pytest.raises(ValueError, match='clavicle'):
        validate_proof(dict(proof, staged_arm_stage_pose=arm), base, r.pose, target)

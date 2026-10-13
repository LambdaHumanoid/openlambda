"""Staged IK: arm-first IK, then bounded spine assistance for residual-error frames.

The reach test is a numerical residual test, not a proof of anatomical
unreachability. All corrections are relative to the original per-frame pose.
"""
from dataclasses import asdict, dataclass
import json

import numpy as np
import torch


@dataclass(frozen=True)
class StagedIKConfig:
    clavicle_policy: str = "preserve_original_per_frame"
    release_start_m: float = 0.03
    release_full_m: float = 0.06
    gate_max_step: float = 0.1
    spine_delta_limit_rad: float = 0.10
    spine_prior: float = 6.0
    regression_weight: float = 2000.0
    per_hand_regression_tolerance_m: float = 0.002
    assistance_iterations: int = 200


CONFIG = StagedIKConfig()
CONFIG_JSON = json.dumps(asdict(CONFIG), sort_keys=True, separators=(",", ":"))
SPINE_JOINTS = (3, 6, 9)
# Keep six parameter slots to preserve the validated loss denominators.
# The first two slots are masked; only shoulders and elbows can change.
CLAVICLE_JOINTS = (13, 14)
ARM_JOINTS = (13, 14, 16, 17, 18, 19)
ACTIVE_JOINTS = SPINE_JOINTS + ARM_JOINTS


def release_gate(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 2 or not np.isfinite(errors).all():
        raise ValueError("arm-stage errors must be finite (frames, 2)")
    if np.any(errors < 0) or not len(errors):
        raise ValueError("arm-stage errors must be nonnegative and nonempty")
    x = np.clip((errors.max(axis=1) - CONFIG.release_start_m) /
                (CONFIG.release_full_m - CONFIG.release_start_m), 0, 1)
    gate = x * x * (3 - 2 * x)
    # Lipschitz minorant: never release any frame passing the arm-only test;
    # taper approaching/leaving those frames rather than dilating the mask.
    for i in range(1, len(gate)):
        gate[i] = min(gate[i], gate[i - 1] + CONFIG.gate_max_step)
    for i in range(len(gate) - 2, -1, -1):
        gate[i] = min(gate[i], gate[i + 1] + CONFIG.gate_max_step)
    return gate.astype(np.float32)


def optimize_staged(*, baseline_pose, virtual_rest_joints, parents,
                    target_left, target_right, arm_solver, initial_pose=None,
                    weights, ramp_frames, iterations, learning_rate, device,
                    log_every, **unused):
    from scripts.human_data.smpl_endpoint_retargeting import (
        OptimizationResult, forward_joints_torch, output_frame_joints_torch,
        palm_centers_torch, smoothstep_ramp,
    )
    base_np = np.asarray(baseline_pose, dtype=np.float32).reshape(-1, 21, 3)
    if not len(base_np) or not np.isfinite(base_np).all():
        raise ValueError("baseline pose must be finite and nonempty")
    target_np = np.stack([target_left, target_right], axis=1).astype(np.float32)
    if target_np.shape != (len(base_np), 2, 3) or not np.isfinite(target_np).all():
        raise ValueError("targets must be finite (frames, 3)")
    # A previous round may warm-start arms, never its spine correction.
    arm = arm_solver(
        baseline_pose=base_np, virtual_rest_joints=virtual_rest_joints,
        parents=parents, target_left=target_left, target_right=target_right,
        initial_pose=initial_pose, optimized_joint_indices=ARM_JOINTS,
        weights=weights, ramp_frames=ramp_frames, iterations=iterations,
        learning_rate=learning_rate, device=device, log_every=log_every,
    )
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    baseline = torch.as_tensor(base_np, device=dev)
    rest = torch.as_tensor(virtual_rest_joints, dtype=torch.float32, device=dev)
    target = torch.as_tensor(target_np, device=dev)
    arm_pose = torch.as_tensor(arm.pose.reshape(-1, 21, 3), device=dev)
    arm_idx = np.array(ARM_JOINTS) - 1
    spine_idx = np.array(SPINE_JOINTS) - 1

    def fk(pose):
        joints = output_frame_joints_torch(forward_joints_torch(pose.reshape(-1, 63), rest, parents))
        return joints, torch.stack(palm_centers_torch(joints), dim=1)

    with torch.no_grad():
        _, arm_endpoints = fk(arm_pose)
        arm_errors = torch.linalg.vector_norm(arm_endpoints - target, dim=-1)
    gate_np = release_gate(arm_errors.cpu().numpy())
    gate = torch.as_tensor(gate_np, device=dev)[:, None, None]
    ramp_np = smoothstep_ramp(len(base_np), ramp_frames).astype(np.float32)
    ramp = torch.as_tensor(ramp_np, device=dev)[:, None, None]
    arm_limit = torch.as_tensor(arm.joint_delta_limits_rad, device=dev)[None, :, None]
    initial_delta = arm_pose[:, arm_idx] - baseline[:, arm_idx]
    ap = torch.nn.Parameter(torch.atanh(torch.clamp(initial_delta / (arm_limit * ramp).clamp_min(1e-7), -.999999, .999999)))
    sp = torch.nn.Parameter(torch.zeros((len(base_np), 3, 3), device=dev))
    optimizer = torch.optim.Adam([ap, sp], lr=learning_rate)
    best_pose = arm_pose.clone()
    best_mean = float(arm_errors.mean())
    selected_iteration = 0
    losses = list(arm.losses)

    def candidate():
        pose = baseline.clone()
        pose[:, arm_idx] = baseline[:, arm_idx] + arm_limit * torch.tanh(ap) * ramp
        pose[:, np.array(CLAVICLE_JOINTS) - 1] = baseline[:, np.array(CLAVICLE_JOINTS) - 1]
        # A spherical correction bound, not three independently capped axes.
        sd = CONFIG.spine_delta_limit_rad * sp / (1 + torch.linalg.vector_norm(sp, dim=-1, keepdim=True))
        sd = sd * gate * ramp
        pose[:, spine_idx] = baseline[:, spine_idx] + sd
        return pose, sd

    for it in range(CONFIG.assistance_iterations if np.any(gate_np > 0) else 0):
        optimizer.zero_grad(set_to_none=True)
        pose, sd = candidate()
        _, endpoints = fk(pose)
        errors = torch.linalg.vector_norm(endpoints - target, dim=-1)
        delta = pose - baseline
        vel = (delta[1:] - delta[:-1]).square().mean() if len(base_np) > 1 else delta.sum() * 0
        acc = (delta[2:] - 2 * delta[1:-1] + delta[:-2]).square().mean() if len(base_np) > 2 else delta.sum() * 0
        loss = (weights.position * (endpoints - target).square().mean()
                + weights.pose_prior * delta[:, arm_idx].square().mean()
                + CONFIG.spine_prior * sd.square().mean()
                + weights.velocity * vel + weights.acceleration * acc
                + weights.first_frame * delta[0].square().mean()
                + CONFIG.regression_weight * torch.relu(errors - arm_errors).square().mean())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite Staged IK assistance loss")
        # Select only whole trajectories; no per-frame splicing that can jump.
        # The unmodified arm result is always an eligible fallback.
        with torch.no_grad():
            mean = float(errors.mean())
            if mean < best_mean and bool(torch.all(errors <= arm_errors + CONFIG.per_hand_regression_tolerance_m)):
                best_mean, best_pose, selected_iteration = mean, pose.detach().clone(), it + 1
        loss.backward()
        optimizer.step()
        if it == 0 or (it + 1) % max(1, log_every) == 0:
            losses.append({"stage": 2., "iteration": float(it + 1), "total": float(loss.detach()),
                           "mean_error_m": float(errors.mean().detach())})
            print(f"staged spine-assist iter={it+1} mean_m={float(errors.mean().detach()):.6f} selected={selected_iteration}", flush=True)

    with torch.no_grad():
        joints, endpoints = fk(best_pose)
        final_errors = torch.linalg.vector_norm(endpoints - target, dim=-1).cpu().numpy()
    pose_np = best_pose.cpu().numpy().reshape(-1, 63).astype(np.float32)
    proof = {
        "staged_config_json": np.asarray(CONFIG_JSON),
        "staged_arm_stage_pose": arm.pose,
        "staged_arm_stage_errors_m": arm_errors.cpu().numpy(),
        "staged_final_errors_m": final_errors,
        "staged_spine_release_gate": gate_np,
        "staged_ramp": ramp_np.astype(np.float32),
        "staged_selected_assistance_iteration": np.asarray(selected_iteration, dtype=np.int32),
        "staged_virtual_rest_joints": np.asarray(virtual_rest_joints, dtype=np.float32),
        "staged_parents": np.asarray(parents, dtype=np.int32),
    }
    validate_proof(proof, base_np.reshape(-1, 63), pose_np, target_np)
    return OptimizationResult(pose_np, joints.cpu().numpy().astype(np.float32), losses,
                              ACTIVE_JOINTS, np.r_[np.full(3, CONFIG.spine_delta_limit_rad), arm.joint_delta_limits_rad], proof)


def validate_proof(archive, baseline_pose, candidate_pose, targets):
    from scripts.human_data.smpl_endpoint_retargeting import forward_joints_torch, output_frame_joints_torch, palm_centers_torch
    if str(np.asarray(archive["staged_config_json"]).item()) != CONFIG_JSON:
        raise ValueError("Staged IK staged IK config mismatch")
    base = np.asarray(baseline_pose, dtype=np.float32).reshape(-1, 21, 3)
    cand = np.asarray(candidate_pose, dtype=np.float32).reshape(base.shape)
    arm = np.asarray(archive["staged_arm_stage_pose"], dtype=np.float32).reshape(base.shape)
    for x in (base, cand, arm, targets):
        if not np.isfinite(x).all():
            raise ValueError("Staged IK proof contains nonfinite poses/targets")
    clavicles = np.array(CLAVICLE_JOINTS) - 1
    if not np.array_equal(arm[:, clavicles], base[:, clavicles]) or not np.array_equal(cand[:, clavicles], base[:, clavicles]):
        raise ValueError("Staged IK changed a locked original clavicle pose")
    locked = [i for i in range(21) if i + 1 not in ARM_JOINTS]
    if not np.array_equal(arm[:, locked], base[:, locked]):
        raise ValueError("Staged IK arm stage changed spine or another locked joint")
    rest = torch.as_tensor(np.asarray(archive["staged_virtual_rest_joints"]), dtype=torch.float32)
    parents = np.asarray(archive["staged_parents"]).astype(int).tolist()
    with torch.no_grad():
        measured = []
        for pose in (arm, cand):
            j = output_frame_joints_torch(forward_joints_torch(torch.from_numpy(pose.reshape(-1, 63)), rest, parents))
            e = torch.stack(palm_centers_torch(j), dim=1).numpy()
            measured.append(np.linalg.norm(e - targets, axis=-1))
    for key, actual in zip(("staged_arm_stage_errors_m", "staged_final_errors_m"), measured):
        if not np.allclose(actual, archive[key], atol=2e-6, rtol=0):
            raise ValueError("Staged IK endpoint proof does not match FK: " + key)
    gate = release_gate(np.asarray(archive["staged_arm_stage_errors_m"]))
    if not np.array_equal(gate, np.asarray(archive["staged_spine_release_gate"])):
        raise ValueError("Staged IK release gate does not match arm-stage errors")
    ramp = np.asarray(archive["staged_ramp"])
    if ramp.shape != gate.shape or not np.isfinite(ramp).all() or np.any((ramp < 0) | (ramp > 1)):
        raise ValueError("Staged IK invalid ramp")
    sd = cand[:, np.array(SPINE_JOINTS)-1] - base[:, np.array(SPINE_JOINTS)-1]
    if np.any(np.linalg.norm(sd, axis=-1) > CONFIG.spine_delta_limit_rad * gate[:, None] * ramp[:, None] + 2e-7):
        raise ValueError("Staged IK spine correction exceeds gated norm bound")
    if np.any(sd[gate == 0] != 0):
        raise ValueError("Staged IK modified spine on arm-reachable frames")
    if np.any(measured[1] > measured[0] + CONFIG.per_hand_regression_tolerance_m + 3e-6):
        raise ValueError("Staged IK assistance worsened a palm beyond tolerance")
    if measured[1].mean() > measured[0].mean() + 2e-6:
        raise ValueError("Staged IK assistance increased mean endpoint error")

"""Human-reference alignment with fixed physical palm targets.

Implements Lambda's human-reference stage: a uniformly scaled virtual body,
bounded collar/shoulder/elbow corrections, and fixed original human targets.
Magnitude and temporal penalties regularize the pose corrections.
"""

from pathlib import Path

import numpy as np
import torch

from .kinematics import gravity_rotation


ACTIVE_JOINTS = np.array([13, 14, 16, 17, 18, 19]) - 1
DELTA_LIMITS = np.array([.35, .35, .90, .90, .75, .75])
OUTPUT_FROM_SMPL = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
WIRE_JOINTS = [*range(22), 39, 54]


def load_skeleton(sonic_root):
    """Read the same released skeleton used by SONIC's human encoder."""
    path = Path(sonic_root)/'gear_sonic/data/human/human_joints_info.pkl'
    data = torch.load(path, map_location='cpu', weights_only=True)
    rest = np.asarray(data['J'], dtype=np.float64)
    parents = tuple(int(i) for i in data['parents_list'])
    if (rest.shape != (55, 3) or len(parents) != 55 or parents[0] != -1
            or not np.isfinite(rest).all()
            or any(not 0 <= parents[i] < i for i in range(1, 55))):
        raise ValueError('expected a finite, parent-first 55-joint SONIC skeleton')
    return rest, parents


def rotation_matrices(vectors):
    """Differentiable Rodrigues formula, including zero axis-angle rotations."""
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*vectors.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vectors, dim=-1)[..., None, None]
    identity = torch.eye(3, device=vectors.device, dtype=vectors.dtype)
    return identity + torch.sinc(angle/torch.pi)*skew + .5*torch.sinc(angle/(2*torch.pi))**2*(skew@skew)


def forward_kinematics(poses, rest, parents):
    """Return root-neutral SMPL positions and rotations, (N,J,3[/3])."""
    rest = torch.as_tensor(rest, dtype=poses.dtype, device=poses.device)
    n, joints = len(poses), len(parents)
    full = torch.cat((poses.new_zeros(n, 1, 3), poses,
                      poses.new_zeros(n, joints-22, 3)), dim=1)
    local = rotation_matrices(full)
    positions, rotations = [rest[0].expand(n, 3)], [local[:, 0]]
    for i in range(1, joints):
        parent = parents[i]
        positions.append(positions[parent] + (rotations[parent] @ (rest[i]-rest[parent])[:, None]).squeeze(-1))
        rotations.append(rotations[parent] @ local[:, i])
    return torch.stack(positions, 1), torch.stack(rotations, 1)


class HumanGeometry:
    """Attach measured OpenXR palms to the original SMPL wrist frames.

    Per-frame offsets retain the original observed Palm point, including the
    existing hand recovery. They are frozen while upstream SMPL joints change.
    Scaling applies only to the virtual optimization body, never the encoder.
    """

    def __init__(self, prepared, sonic_root, *, device):
        self.rest, self.parents = load_skeleton(sonic_root)
        self.pose = torch.as_tensor(prepared['poses'], dtype=torch.float32, device=device)
        n = len(self.pose)
        fields = {'poses': (n, 21, 3), 'joints': (n, 24, 3),
                  'quaternions': (n, 4), 'targets': (n, 4, 7), 'human_height': (n,)}
        if n < 1:
            raise ValueError('human correction requires at least one frame')
        for key, shape in fields.items():
            array = np.asarray(prepared[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f'{key} must be finite with shape {shape}')
        if np.any(np.asarray(prepared['human_height']) <= 0):
            raise ValueError('human root height must be positive')
        self.height = self.pose.new_tensor(prepared['human_height'])
        self.target = self.pose.new_tensor(prepared['targets'][:, :2, :3])
        # Convert encoder-local SMPL to the existing gravity-aligned target frame.
        tilt = np.array([gravity_rotation(q) for q in prepared['quaternions']])
        self.frame = self.pose.new_tensor(tilt @ OUTPUT_FROM_SMPL)
        with torch.no_grad():
            positions, rotations = forward_kinematics(self.pose, self.rest[:22], self.parents[:22])
            wrist = positions[:, 20:22]-positions[:, :1]
            local_target = torch.einsum('nji,nkj->nki', self.frame, self.target)
            self.palm_offset = torch.einsum('nkji,nkj->nki', rotations[:, 20:22], local_target-wrist)

    def palms(self, poses):
        positions, rotations = forward_kinematics(poses, self.rest[:22], self.parents[:22])
        palms = positions[:, 20:22]-positions[:, :1]
        palms = palms + torch.einsum('nkij,nkj->nki', rotations[:, 20:22], self.palm_offset)
        palms = torch.einsum('nij,nkj->nki', self.frame, palms)
        return palms + torch.stack((torch.zeros_like(self.height),
                                    torch.zeros_like(self.height), self.height), -1)[:, None]

    def encoder_joints(self, poses, original_joints):
        """Update original, unscaled wire joints without redefining slots 22/23."""
        with torch.no_grad():
            before, _ = forward_kinematics(self.pose, self.rest, self.parents)
            after, _ = forward_kinematics(poses, self.rest, self.parents)
            delta = (after-before)[:, WIRE_JOINTS]
            delta = delta @ self.pose.new_tensor(OUTPUT_FROM_SMPL.T)
        return np.asarray(original_joints).copy() + delta.cpu().numpy()


def human_scale(robot, sonic_root, *, urdf=None):
    """Scale canonical foot-to-head height to the G1 standing mesh height."""
    import pinocchio as pin

    urdf = Path(urdf) if urdf is not None else Path(sonic_root)/'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf'
    geometry = pin.buildGeomFromUrdf(robot.model, str(urdf), pin.GeometryType.VISUAL,
                                     package_dirs=[str(urdf.parent)])
    head_indices = [i for i, obj in enumerate(geometry.geometryObjects)
                    if robot.model.frames[obj.parentFrame].name == 'head_link']
    if not head_indices:
        raise ValueError('G1 model requires head_link visual geometry for scaling')
    q = pin.neutral(robot.model)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    placements = pin.GeometryData(geometry)
    pin.updateGeometryPlacements(robot.model, robot.data, geometry, placements)
    floor = min((robot.data.oMf[frame].translation
                 + robot.data.oMf[frame].rotation @ offset)[2]
                for frame, offset in zip(robot.frames[2:], robot.offsets[2:]))
    top = -np.inf
    for i in head_indices:
        obj, placement = geometry.geometryObjects[i], placements.oMg[i]
        vertices = np.asarray(obj.geometry.vertices())
        top = max(top, float((vertices @ placement.rotation.T + placement.translation)[:, 2].max()))
    robot_height = top-floor
    rest, _ = load_skeleton(sonic_root)
    human_height = rest[15, 1]-min(rest[10, 1], rest[11, 1])
    scale = float(robot_height/human_height)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('human/robot standing heights must be positive')
    return scale


def correct_human_reference(prepared, *, sonic_root, scale, iterations=350, device=None):
    """Return a new reference; targets, fingers, root and locked poses are intact."""
    if not np.isfinite(scale) or scale <= 0 or iterations < 1:
        raise ValueError('scale and iteration count must be positive')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    geometry = HumanGeometry(prepared, sonic_root, device=device)
    baseline = geometry.pose
    target = geometry.palms(baseline).detach()
    parameter = torch.nn.Parameter(torch.zeros_like(baseline[:, ACTIVE_JOINTS]))
    optimizer = torch.optim.Adam([parameter], lr=.035)
    limits = baseline.new_tensor(DELTA_LIMITS)[None, :, None]
    phase = ((torch.arange(len(baseline), device=baseline.device,
                           dtype=baseline.dtype)-10)/50).clamp(0, 1)
    startup = (phase.square()*(3-2*phase))[:, None, None]
    best_loss, best = float('inf'), baseline.detach().clone()
    for step in range(iterations+1):
        # Radial bound limits each joint's entire correction, not each component.
        delta = startup*limits*parameter/(1+torch.linalg.vector_norm(parameter, dim=-1, keepdim=True))
        candidate = baseline.clone()
        candidate[:, ACTIVE_JOINTS] = baseline[:, ACTIVE_JOINTS]+delta
        position = (scale*geometry.palms(candidate)-target).square().mean()
        velocity = (delta[1:]-delta[:-1]).square().mean() if len(delta) > 1 else delta.sum()*0
        acceleration = (delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean() if len(delta) > 2 else delta.sum()*0
        loss = 500*position + .6*delta.square().mean() + 18*velocity + 8*acceleration
        if not torch.isfinite(loss):
            raise ValueError('nonfinite human-reference optimization loss')
        if float(loss.detach()) < best_loss:
            best_loss, best = float(loss.detach()), candidate.detach().clone()
        if step == iterations:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = dict(prepared)
    result['poses'] = np.asarray(prepared['poses']).copy()
    result['poses'][:, ACTIVE_JOINTS] = best.cpu().numpy()[:, ACTIVE_JOINTS]
    result['poses'][:10] = prepared['poses'][:10]
    result['joints'] = geometry.encoder_joints(best, prepared['joints'])
    return result

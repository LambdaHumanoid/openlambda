# Data and causality contract

## Raw input layout

The adapter consumes a LeRobot-style dataset. See the bundled
[machine-readable feature schema](../examples/data/basket/meta/info.json) and
[sample recipe](../examples/README.md). Paths are resolved from `meta/info.json`;
the sample uses:

```text
meta/info.json                 # FPS, feature shapes, episode count, path templates
meta/episodes.jsonl
meta/tasks.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.gopro/episode_000000.mp4
videos/chunk-000/observation.images.level-cam/episode_000000.mp4
sync/episode_000000.gopro_alignment.json
sync/episode_000000.level-cam_alignment.json
```

Every camera declared in the input metadata must have its referenced video.
One or two cameras can be processed. Source camera keys `gopro` and `level-cam`
map to output `ego_view` and `level_cam`, respectively. Keep source episode IDs
and the conversion's source-to-compact mapping, even when processing a subset.

### Numerical fields

Shapes below are per frame; pose arrays use the adapter's fixed joint order.
Positions are in meters and angles are in radians.

| Raw column | Shape | Meaning |
|---|---|---|
| `observation.pico.body_pose` | 24 × 7 | Pico world-space body positions and quaternion poses: `[x,y,z,qx,qy,qz,qw]` |
| `timestamp` | scalar | Episode-relative Pico time, in seconds; provide finite, increasing timestamps for alignment |
| `observation.pico.left_hand_pose`, `...right_hand_pose` | 26 × 7 each | OpenXR-ordered hand joint poses for BrainCo hand retargeting |
| `observation.pico.left_hand_active`, `...right_hand_active` | scalar bool each | Whether the corresponding optical hand pose is usable |
| `observation.pico.left_controller_pose`, `...right_controller_pose` | 7 each | Optional controller pose observations used by the adapter when available |
| `episode_index`, `frame_index`, `task_index`, `index` | scalar each | LeRobot row/episode/task metadata |

The body pose is required. Supply valid hand poses and active flags when hand
actions matter: the adapter falls back to open/zero hand commands for unavailable
or unusable hand tracking. Stage A can synthesize timestamps at 50 Hz when absent,
but endpoint alignment needs the original `timestamp` column.
An explicit Stage-A `--task` overrides task metadata; otherwise the adapter uses
the first task in the source. Convert different task instructions separately.

### Reference coordinates and skeletons

Use the supplied body transforms for Pico/Unity coordinates and quaternions;
do not apply an extra axis flip. Derived `teleop.smpl_pose` is 21 **local** body
axis-angle rotations (63 floats), excluding the root. `teleop.body_quat_w` is
the separate root quaternion in **wxyz** order. `teleop.smpl_joints` contains
24 × 3 body-local reference coordinates.

The trusted FK skeleton has 55 joints. Human palm targets are the mean
of four non-thumb MCP roots: left indices 25/28/31/34, right 40/43/46/49. These
are **not** indices into raw Pico's 24 body or 26 hand slots. Endpoint-aligned
SONIC references place the two palm centers in their 24-joint output slots 22/23.
Do not substitute bare wrists for MCP geometry. Arbitrary SMPL arrays without
the raw adapter schema require a separate input adapter.

The canonical lower-foot-joint-to-head-joint height is
1.5549421608448029 m, not mesh crown-to-floor stature. Virtual G1-sized morphology
is calibrated from robot geometry, not a second fixed human-height constant.
Wrist repair and denoising precede endpoint optimization.

## Camera and frame correspondence

Each camera alignment JSON supplies finite `offset_s` and a positive finite
`camera_duration_s` or legacy `gopro_duration_s`. If both duration keys exist,
they must agree (absolute tolerance 1e-6 s, relative tolerance 1e-9). Stage A
permits absent duration; candidate alignment cropping requires one.

The camera covers Pico time `[offset_s, offset_s + duration_s]`; conversion
selects the cameras' common interval and samples camera time as
`pico_timestamp - offset_s`. Exported timestamps start at zero for the retained
interval. Stage A samples/resizes/encodes the camera frames; raw inputs are not
modified. Final Stage C preserves the Stage-A/candidate video files and row
timestamps. Residual lag compensation does **not** shift these image/state/action
pairs.

Legacy-duration parsing can change admission relative to older converters.
Reuse baseline mappings only after hash, row-count and timing checks; never pair
a newly cropped candidate with stale telemetry. A validated full-row mapping can
use the builder's `full` selection mode without a second crop.

## Endpoint alignment

Round 0 performs geometric dual-palm alignment; Rounds 1/2 refine targets using
actual SONIC–MuJoCo replay residuals. Both hands use the implementation's temporal
alignment, smoothing, gain estimation and correction clipping. The interaction
frame centers x/y on the pelvis and z on the lower foot at each frame.

Actual robot palms use MuJoCo FK wrist-yaw anchors with explicit offsets
`[0.0415, +0.003, 0]` / `[0.0415, -0.003, 0]` m. Preserve model and offset identity.
Candidate FK error is not measured controller error. Internal distances are in
meters; convert to centimeters only for reports that explicitly request it.

Candidate archives include optimizer configuration, validation evidence and FK
arrays. Preserve all generated fields. Algorithm identity and configuration are
checked before reuse; candidates from earlier implementations are not compatible
even though the archive schema remains version 4.
See [the processing workflow](PROCESSING.md#endpoint-alignment).
SONIC still determines physical G1 waist motion; there is no robot waist lock.

## Causal output: A / B / C

- A — reference conversion: zero-history staging with real wrist references.
- B — closed-loop replay: actual SONIC–MuJoCo motion and per-control-tick capture.
- C — causal export: measured state and the same-tick captured 64D encoder token.

At tick t, state is sampled before action t; decoder history contains t−1 and
earlier. The 43D exported state is not the entire controller history tensor.

| Final field | Shape | Representation |
|---|---|---|
| `observation.state` | 43 | 29 body values plus two padded 7D hand slots |
| `action.motion_token` | 64 | Actual C++ encoder token captured on the matching control tick |
| `teleop.left_hand_joints`, `teleop.right_hand_joints` | 6 each | Retained BrainCo hand labels; together with the token form 76D learned actions |

Generated hand labels are dimensionless commands ordered as
`[thumb, thumb_aux, index, middle, ring, pinky]`, not joint angles in radians.
The adapter maps the two thumb channels to [0, 0.5] and other fingers to [0, 1];
its 7D wire format adds a final zero padding value.

The default state schema is named `isaaclab_offsets` and has 43 dimensions.
Body slots are indices 0–21 and 29–35, encoded in IsaacLab order
as offsets from default angles, not absolute MuJoCo qpos. Hand slots 22–28 and
36–42 retain the physical hand representation. Use the provided state-space
conversion helpers; do not assume contiguous 29+7+7 slicing.

Reject missing capture, mismatched identity/timing and invalid replay. Do not
substitute staging tokens, shift labels by encoder-window length, zero-fill
missing feedback or override G1 upper-body joints. Validate final examples
against the selected replay as described in [the sample guide](../examples/README.md).

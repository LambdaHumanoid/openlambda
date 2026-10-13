# External assets

No robot meshes, SMPL model parameters or SONIC weights are bundled.

## BrainCo-equipped G1

Set `EGOALIGN_BRAINCO_MODEL_DIR` to a directory containing:

```text
scene_29dof_brainco.xml
g1_29dof_brainco.xml
brainco_meshes/             # every mesh referenced by the XML
```

Obtain the compatible complete model from its rights holder. There is no automatic
download or fallback to a different hand. The qualified model's file names and
SHA-256 hashes (not asset contents) are in `configs/brainco_reference_hashes.json`.
Check it with the simulation interpreter, without starting a simulator service:

```bash
export EGOALIGN_BRAINCO_MODEL_DIR=/absolute/path/to/your/g1
"$EGOALIGN_SIM_PYTHON" -m egoalign.check_robot_assets \
  --expected configs/brainco_reference_hashes.json --load-mujoco
```

Missing files, unresolved LFS pointers, references outside that directory and
hash mismatches fail explicitly. Omit `--expected` only to inventory a different
model; a successful inventory does not qualify its dynamics. The reference MJCF
loads with nq=68, nv=67, nu=61. Preserve joint names/order, mass/inertia, collision
geometry, wrist-yaw anchors and palm offsets. Do not tune these to match sample
numbers. The optimizer's morphology XML is a separate explicit input; the
qualified recipe uses the pinned public G1 `g1_29dof_with_hand.xml` listed below.

The official [BrainCo description repository](https://github.com/BrainCoTech/brainco-description)
and [G1 integration repository](https://github.com/BrainCoTech/unitree-g1-brainco-hand)
provide model and integration information. Their models have not been established
here as identically hashed replacements for the qualified local asset; use the
inventory check above to identify your supplied model.

## SONIC, public skeleton and calibration geometry

Follow [the public runtime guide](public_runtime.md). Source commit, model
revision and hashes are pinned in `runtime/`; downloading is explicit.
The public skeleton is retrieved through Git LFS, not generated
from an unspecified body model. Only load trusted pickle files.

The Object Relocation sample's calibration input is:
`$EGOALIGN_RUNTIME_ROOT/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`.
Fetch its referenced LFS meshes too. It is not interchangeable with the BrainCo
scene path. Actual endpoint evaluation still uses the BrainCo replay model.

## SMPL and samples

The included raw samples contain numeric body/hand observations and video, not
SMPL mesh templates, blend-shape parameters or model weights. Any separately
required SMPL/SMPL-X assets are obtained and configured separately from the
numeric sample inputs.

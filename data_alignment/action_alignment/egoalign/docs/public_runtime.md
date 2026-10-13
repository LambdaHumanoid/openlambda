# Public SONIC runtime and compatibility patches

This is the source-distribution route, independent of any private Docker image
or private deployment scripts. **Fresh controller compilation and end-to-end Object
Relocation processing through this launcher passed on 2026-10-08 using existing
dependencies.** Clean-machine installation remains unverified. See
[validation evidence and limitations](VALIDATION.md).

## Immutable sources

- Runtime: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
  commit `7f151314d4d1606544bf249d2a7a1cb754c64582`.
- Models: [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC), revision
  `6733128a3d8a523b1418b06bca3cdf61c8b0987f`. Use the original top-level encoder,
  decoder and planner, **not** the low-latency or later model variants.
- Patches: [controller](../runtime/patches/0001-controller.patch) and
  [replay/simulation](../runtime/patches/0002-replay.patch).
- [Source manifest](../runtime/patches/manifest.json): 147 selected files,
  before/after SHA-256 and patch checksums. Apply the patches to the public
  upstream commit above; no development repository is needed.

The patches preserve the controller and recursive replay dependency closure.
They are a compatibility snapshot, not a claim of a minimal line-level diff.
They include causal history, actual encoder-token capture, same-tick telemetry,
BrainCo simulation support, per-episode controller restart/reset and measured
stability gates. They do not include weights, meshes, SMPL assets, credentials,
or the development workspace's top-level deployment scripts.

The Pico-named Python module is a pure hand-mapping extraction. The supported
replay input is the Stage-A/candidate **pose** contract; raw live Pico/VR3pt device
streaming is not supported by this public launcher. Inherited controller features
outside this workflow are not enabled by its command line.

## Obtain and apply (source-only operations)

Run from the EgoAlign checkout. Choose a new runtime directory, not production.

```bash
export EGOALIGN_RUNTIME_ROOT=/absolute/path/to/new-sonic-runtime
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$EGOALIGN_RUNTIME_ROOT"
GIT_LFS_SKIP_SMUDGE=1 git -C "$EGOALIGN_RUNTIME_ROOT" checkout --detach 7f151314d4d1606544bf249d2a7a1cb754c64582
python3 tools/apply_runtime_patches.py --runtime-root "$EGOALIGN_RUNTIME_ROOT"
python3 tools/apply_runtime_patches.py --runtime-root "$EGOALIGN_RUNTIME_ROOT" --apply
python3 tools/apply_runtime_patches.py --runtime-root "$EGOALIGN_RUNTIME_ROOT" --verify-applied
```

Default is check-only. The installer rejects a wrong HEAD, modified source,
incorrect patch hash or wrong base files; it does not reset your repository.
Apply patches before downloading assets so the clean-checkout preflight passes.
Record `git submodule status --recursive`; initialize any declared dependencies
at their pinned gitlink revisions, not their latest branches.

Before compiling or launching, materialize the pinned Git LFS dependencies:

```bash
git -C "$EGOALIGN_RUNTIME_ROOT" lfs pull --include='gear_sonic_deploy/thirdparty/unitree_sdk2/**,gear_sonic_deploy/reference/example/**,gear_sonic/data/robot_model/model_data/g1/**,gear_sonic/data/human/human_joints_info.pkl'
```

The SDK libraries, startup motions and robot meshes must be real files, not
small LFS pointer text files. Otherwise linking fails with “file format not
recognized”, or simulation fails while loading STL geometry. This step does
not download the separately pinned ONNX models below.

## Build the controller

Follow `docs/source/getting_started/installation_deploy.md` in the pinned
checkout. Its x86_64 profile calls for **TensorRT 10.13**; Jetson uses a different
profile and is not qualified here. Obtain TensorRT through the
[official NVIDIA download](https://developer.nvidia.com/tensorrt/download/10x).
Match CUDA and the GPU driver to that SDK.

C++ requires C++20, CMake, CUDA, TensorRT, ONNX Runtime C++ SDK, Eigen,
yaml-cpp, cppzmq/ZeroMQ, JSON headers, zlib and Unitree SDK2. A Python
`onnxruntime` wheel does not supply the controller SDK. ROS2 is optional; the
simulation launcher uses loopback ZMQ. Inspect the supplied install scripts
before running them: they perform system-wide installs and some dependency
downloads are not immutable. This package is not yet a hermetic dependency lock.

After installing these dependencies, the upstream build entry is:

```bash
cd "$EGOALIGN_RUNTIME_ROOT/gear_sonic_deploy"
source scripts/setup_env.sh
just build
```

Expected binary: `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`.
Check its `ldd` output for missing libraries in the environment where it will
run. Never replace the active production binary to test this checkout.

## Prepare Python and GPU rendering

Install EgoAlign in a Python 3.10 data environment as shown in the
[README](../README.md). Stage A/C additionally require the compatible legacy
LeRobot API and Pinocchio; replay requires pyzmq and tyro. Resolve these using the
pinned runtime instructions and [qualified dependency inventory](DEPENDENCIES.md).
Package extras alone do not install every runtime dependency, and the inventory
is evidence of a tested environment, not a portable lockfile.

For simulation, create a new environment in the runtime checkout:

```bash
cd "$EGOALIGN_RUNTIME_ROOT"
python3.10 -m venv .venv_egoalign_sim
.venv_egoalign_sim/bin/python -m pip install -e './gear_sonic[sim]'
.venv_egoalign_sim/bin/python -m pip install -e ./external_dependencies/unitree_sdk2_python
```

Resolve native SDK prerequisites first. Do not blindly run the upstream legacy
`install_mujoco_sim.sh`: it deletes an existing `.venv_sim`. Configure the
interpreters before using them; replay and simulation may share an environment
if it has all their dependencies:

```bash
export EGOALIGN_DATA_PYTHON=/absolute/path/to/data-env/bin/python
export EGOALIGN_RUNTIME_PYTHON="$EGOALIGN_RUNTIME_ROOT/.venv_egoalign_sim/bin/python"
export EGOALIGN_SIM_PYTHON="$EGOALIGN_RUNTIME_ROOT/.venv_egoalign_sim/bin/python"
```

`python -m egoalign` selects the simulation interpreter for evaluate/audit/calibrate/render-replay
and the data interpreter for other commands; unset interpreter variables default
to the invoking Python. The public driver uses the runtime interpreter for the
replay publisher and the simulation interpreter for MuJoCo. The CLI adds the
EgoAlign root to `PYTHONPATH`; use it instead of launching legacy scripts directly.

All launcher subprocesses must share one filesystem,
network namespace and usable C++ library environment. Native installation or
running the **whole launcher inside your own GPU container** are possible;
this entry does not invoke or depend on a named developer container.

For containerized MuJoCo, CUDA access alone is insufficient. Expose NVIDIA
graphics libraries as well, e.g. `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
Verify the EGL vendor configuration is installed (commonly
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` pointing to `libEGL_nvidia.so.0`).
An explicit `__EGL_VENDOR_LIBRARY_FILENAMES=/path/to/10_nvidia.json` may be
needed if the image lacks that manifest. Check the actual renderer:

```bash
MUJOCO_GL=egl "$EGOALIGN_SIM_PYTHON" -c 'import mujoco; from OpenGL import GL; c=mujoco.GLContext(64,64); c.make_current(); print(GL.glGetString(GL.GL_VENDOR), GL.glGetString(GL.GL_RENDERER)); c.free()'
```

Expect the intended NVIDIA GPU, not Mesa llvmpipe. The current replay runtime
creates an offscreen renderer even with MP4 output disabled; software rendering
can disrupt real-time controller/simulator pacing. Do not lower physical
stability thresholds to work around a renderer problem.

## Models and external assets

The [asset manifest](../runtime/assets.json) pins four public files and SHA-256.
Their hashes match the models/configuration used by the qualified local source.
Return to the EgoAlign root to fetch the pinned models:

```bash
cd /absolute/path/to/EgoAlign
"$EGOALIGN_DATA_PYTHON" -m egoalign.runtime_assets --runtime-root "$EGOALIGN_RUNTIME_ROOT" --download
"$EGOALIGN_DATA_PYTHON" -m egoalign.runtime_assets --runtime-root "$EGOALIGN_RUNTIME_ROOT"
export EGOALIGN_ASSET_ROOT="$EGOALIGN_RUNTIME_ROOT"
```

The downloader only creates missing files; a different existing file or LFS
pointer is rejected. Explicitly back up such a file before retrying. Downloads
are checked before atomic publication; no fallback to a newer model is allowed.
The launcher verifies these hashes before execution too. Setting the asset root
to the runtime checkout also lets Stage A find the downloaded encoder/decoder;
do not point it at an unrelated empty asset directory. The resulting hierarchy is:

```text
$EGOALIGN_ASSET_ROOT/
  gear_sonic/data/human/human_joints_info.pkl
  gear_sonic/data/robot_model/model_data/g1/
  gear_sonic_deploy/policy/release/model_encoder.onnx
  gear_sonic_deploy/policy/release/model_decoder.onnx
```

Materialize the trusted public skeleton through Git LFS:

```bash
git -C "$EGOALIGN_RUNTIME_ROOT" lfs pull --include=gear_sonic/data/human/human_joints_info.pkl
```

Expected skeleton SHA-256:
`4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975`.
Treat pickle files as executable inputs; load only the trusted pinned artifact.
The sample pipeline uses this skeleton, not a bundled SMPL mesh template. Any
additional SMPL assets must be obtained under their own terms. The exact compatible BrainCo XML/mesh
tree remains an explicit external asset: configure `EGOALIGN_BRAINCO_MODEL_DIR`.
It must contain `scene_29dof_brainco.xml`, `g1_29dof_brainco.xml` and referenced
meshes. **Its public redistribution/acquisition path has not been established**;
do not substitute another hand morphology and claim equivalent results.

## Simulation-only entry and sample recipes

The processing pipelines invoke `tools/run_sim_replay.sh`. This bundled adapter
checks that `EGOALIGN_REPLAY_DRIVER` is configured and `EGOALIGN_ALLOW_SIM_REPLAY=1`,
then forwards arguments to the driver. The bundled `tools/public_replay_driver.sh`
starts `egoalign.public_runtime`, which performs runtime and port checks. Use the
Python processing commands after setup; the adapter is not a project-wide launcher.

From the EgoAlign checkout, with dependencies and assets prepared:

```bash
export EGOALIGN_HUMAN_JOINTS_PATH="$EGOALIGN_RUNTIME_ROOT/gear_sonic/data/human/human_joints_info.pkl"
export EGOALIGN_BRAINCO_MODEL_DIR=/absolute/path/to/g1_brainco
export EGOALIGN_REPLAY_DRIVER="$(pwd)/tools/public_replay_driver.sh"
export EGOALIGN_PORT_BASE=28700
"$EGOALIGN_REPLAY_DRIVER" --preset sonic_sim --mode replay --hand-type brainco \
  --replay-dataset /path/to/stage_a --replay-video-dir /path/to/new/replay --dry-run
```

Dry-run checks source hashes and prints commands; it does **not** validate
compiled libraries, models, ports, or dynamics. After reviewing the environment,
set `EGOALIGN_ALLOW_SIM_REPLAY=1` and follow [the Object Relocation recipe](../examples/README.md).
Replay-only usage is also documented there but requires user-supplied inputs.
The single-lane launcher supports `--replay-workers 1` only. Processing commands
reject larger worker counts before doing any work, including in dry-run mode.
A compatible custom driver may explicitly declare its capacity with
`EGOALIGN_REPLAY_MAX_WORKERS`; this does not enable parallelism in the bundled
driver. The endpoint `capacity` mode requires a custom four-lane driver.
Ports are
base through base+4 (default 28700–28704); each concurrent launch needs a separate
free range. Preflight is not an OS port reservation. Cleanup targets only process
groups it created, never other deploy sessions.

The lifecycle is reset-before-controller-init, PLANNER/support release,
measured stability, guarded first-pose ramp and actual replay. It captures real
history/state and actual encoder tokens; it neither substitutes offline labels
nor directly overrides G1 upper-body joints. The runtime does not physically lock the G1 waist (`--lock-waist-axes none`).

The supplied driver accepts the batch interface below; a custom adapter must
preserve the same behavior and artifact contract:

```text
--preset sonic_sim --mode replay --hand-type brainco
--replay-dataset STAGING --replay-video-dir OUTPUT
--replay-episodes all --replay-workers 1
```

Required outputs include per-episode simulator/control telemetry, captured
`control.npz/token_state`, and rollout identity/hash/timing JSON. Stage C checks
same-tick state/token pairing and finite, complete capture. Reject fallen, frozen
or truncated rollouts; a generic qpos log or MuJoCo video is insufficient.
Wire-protocol revision tags identify telemetry schemas, not processing modes;
do not rename them or relabel older artifacts. Never use a real-robot launcher
as this driver.

## Preflight and troubleshooting

From the EgoAlign checkout, these checks do not start simulation:

```bash
"$EGOALIGN_DATA_PYTHON" -m pytest -q
"$EGOALIGN_DATA_PYTHON" tools/check_cli.py
"$EGOALIGN_DATA_PYTHON" tools/audit_source.py
"$EGOALIGN_DATA_PYTHON" tools/dependency_inventory.py
```

Use the [processing guide](PROCESSING.md) for existing-baseline preflight, or
the [sample recipe](../examples/README.md) for a complete run. Keep command logs,
resolved configuration, source/asset hashes, dependency inventories and telemetry.

- Missing controller libraries: inspect `ldd` in the actual execution environment;
  install the C++ SDKs, not just Python wheels.
- LeRobot import errors: use the recorded legacy API, not an arbitrary latest release.
- Missing/invalid final tokens: check capture patches, timing and identity; never
  substitute offline staging labels or bypass validators.
- Occupied ports: select another free range; do not stop another deployment.
- Software EGL rendering: correct NVIDIA graphics library exposure before replay.

## Validation scope

The October 8 runtime qualification covered compilation from the pinned public
source and end-to-end replay using installed dependencies and external assets.
The October 12 processing qualification covered the current alignment method,
final captured 43D state/64D token pairing and camera/timestamp preservation.
GPU EGL was checked in the qualified environment. See the dated results in
[VALIDATION.md](VALIDATION.md); source-only packaging checks are not a new build
or physics replay qualification.
Not verified: isolated installation, complete production quality profiles,
or numerical equivalence across newly installed dependency stacks.

Configure [external assets](EXTERNAL_ASSETS.md). A new dependency
environment requires its own qualification. Do not waive production quality gates.

For final MP4 export, install `av` in `EGOALIGN_SIM_PYTHON` and follow the
[offline video command](../examples/README.md#export-the-final-mujoco-video).
It uses saved replay telemetry and does not start simulator/controller services.

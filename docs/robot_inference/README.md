# Inference

Technical documentation for the observation–policy–controller pipeline. The current implementation connects G1 feedback and an ego camera to Lambda0, then optionally sends actions through a SONIC/Revo2 runtime.

[Beginner tutorial / 中文入门教程](tutorial.zh-CN.md) · [中文技术说明](technical.zh-CN.md) · [Runtime protocols](interfaces.md)

## Scope

The client handles observation alignment, model requests, action publication and logging. Model loading and normalization belong to `lambda0 serve`. Robot startup, homing, pose preparation and shutdown belong to the external control system.

Live execution requires the signed motion router, timestamped RGB publisher and robot-side services described in the [terminal tutorial](tutorial.zh-CN.md). The public runtime integrates with separately installed SONIC and hardware SDKs; stock SONIC and the basic robot bridge are not drop-in substitutes for the required interfaces. Runtime monitoring or a successful localhost protocol check is not hardware validation.

The supplied adapters use one ego image, a 44-dimensional model state and 50 predictions of 76-dimensional actions. Scheduling is serial, with no RTC or action prefetch.

## Implementation

The policy client is `scripts/infer_robot.py`, organized into data contracts, configuration, observation decoding and alignment, the Lambda0 HTTP client, signed action messages, SONIC session management, and the command-line runner. Companion runtime and camera commands use the same configuration and protocols; they do not replace `lambda0 serve`.

`G1Observations`, `LambdaPolicy` and `SonicController` define the current observation, policy and controller implementations. Model serving remains in `src/openlambda/inference/server.py`.

## Data contracts

`Observation` contains named RGB images, named state vectors, nanosecond timestamps, a clock identifier and metadata. Images are nonempty `uint8[H,W,3]` arrays; state fields are finite `float32[D]` vectors.

`ActionSpec` declares representation, field dimensions, units, frequency and reference. `ActionBatch` contains finite `float32[T,D]` arrays with a shared horizon. Policy output and controller input specifications must match exactly.

The current policy concatenates body joints (29), left and right hand feedback (6 each), and projected gravity (3). Images are resized to 448 × 448. Output fields are `motion[50,64]`, `left_hand[50,6]` and `right_hand[50,6]`. Hand targets use normalized STARK units and are clipped to `[0,1]`; tokens outside ±1.25 are rejected.

## Configuration

The configuration is divided into `observation`, `policy`, `controller` and `schedule`, with a top-level task `instruction`. See [`configs/robot_inference/lambda0_g1.json`](../../configs/robot_inference/lambda0_g1.json) for the schema example and the [parameter reference](technical.zh-CN.md#4-配置参数) for client field descriptions.

The checkpoint path, normalization dataset and inference settings must agree with the model service. The sample embodiment ID is a placeholder. Controller PID, binary path and robot address are required for execution. All configured service endpoints use loopback addresses.

## Terminal runtime entry points

The [step-by-step tutorial](tutorial.zh-CN.md#6-实时预览与显式执行) labels which commands run on G1 versus the Linux SONIC host. The model remains the existing `lambda0 serve`; no web console is required.

| Component | Terminal entry point | Boundary |
| --- | --- | --- |
| Pinned external SONIC integration | `python scripts/setup_robot_runtime.py sonic` | Check only; `--apply --build` explicitly patches/builds, never launches the controller |
| Local signed router | `python scripts/robot_runtime.py serve --config CONFIG` | Monitor by default; `--allow-actuation` explicitly enables the execution path |
| Readiness inspection | `python scripts/robot_runtime.py status --config CONFIG` / `prepare --config CONFIG` | No automatic homing or motor enable; incompatible backend refuses readiness |
| Explicit controller initialization | `python scripts/robot_runtime.py initialize --config CONFIG --execute --token TOKEN.json` | Sends a checkpoint-specific external token and start command; causes physical motion, never implicit in setup/monitoring |
| Capture current controller Hold | `python scripts/robot_runtime.py hold --config CONFIG --execute` | Physically meaningful, explicit operation; requires actual CONTROL feedback and measured stability |
| Router stop | `python scripts/robot_runtime.py stop --config CONFIG` | Stops policy control, not robot power or physical motion |
| G1 hand adapter | `python scripts/robot_hands.py serve --port 18561` | ROS/STARK dependencies required; no command publication unless explicitly started with `--allow-actuation` |
| G1 camera ingress | `python -m lambda_robot_data.live_camera capture` | Preserves Linux loopback UDP ingress time through PTS matching; not camera exposure time |
| Host camera clock/publisher | `python -m lambda_robot_data.live_camera calibrate --g1 G1 --output CLOCK.json`, then `publish --clock-map CLOCK.json` | Calibrated host/G1 timestamps; mapping expires after five minutes and must be recreated |

Replace `CONFIG`, `G1`, `CLOCK.json` and `TOKEN.json` with your local config, SSH destination, new clock-map filename and verified SONIC initialization token. Optional controller endpoints default to local ports 5686 (`runtime_endpoint`), 18561 (`hand_command_endpoint`) and 5656 (`action_endpoint`). Forward G1 camera/hand loopback ports over SSH, not remote SONIC timestamps. Keep model and native-controller environments separate.

**Hand-unit gate:** policy targets are already normalized STARK `[0,1]`. Use the direct normalized-STARK adapter, not a project-space bridge that doubles the thumb targets. Do not bypass readiness or run multiple hand command publishers to make a status check pass. SONIC startup/enable, SDK installation and physical emergency-stop procedures remain external prerequisites under their original licenses.

## Execution

| Command | Behavior |
| --- | --- |
| `check` | Validate adapter contracts and model service settings |
| `predict` | Read an episode frame and save a prediction |
| `run` | Read live observations and save predictions |
| `run --execute` | Check initialization and publish predicted actions |

Execution requires stable Hold and agreement with the reference episode's initial measured pose. Each cycle reads an observation, calls the model, saves the prediction and publishes the first `schedule.execute_steps` actions at 50 Hz. The next observation must be newer than the last publication. Observation and inference waits add to the cycle duration.

Sensor faults, invalid actions, controller restarts or missed publication deadlines interrupt execution. Session cleanup requests policy disable and unbind. The underlying controller may hold its last target; session termination does not imply physical stop or power-off.

## Invocation

Use Python 3.11 from the repository root for the model and client environment (the model package supports 3.11/3.12). Keep the G1 Python 3.8 / ROS 2 Foxy environment separate. Install the model with HTTP serving dependencies:

```bash
PYTHON=python3.11 bash scripts/install.sh serve
source .venv/bin/activate
python -m pip install -r scripts/requirements-robot-inference.txt
cp configs/robot_inference/lambda0_g1.json configs/robot_inference/local.json
```

Obtain a real compatible checkpoint, processor and checkpoint-bound normalizer before starting `lambda0 serve`: its dataset must be present in the checkpoint's `data_bundle/manifest.json`, with the correct embodiment ID and action space 2. The public Stage II weights alone are not a validated task-specific deployment package. Keep RTC disabled. The [beginner tutorial](tutorial.zh-CN.md#3-核对真实模型资产) explains these asset requirements and the server command.

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json check
python scripts/infer_robot.py --config configs/robot_inference/local.json predict \
  --episode data/robot/aligned/episode-001 --frame 0 \
  --output outputs/lambda0-offline
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --chunks 5 --output outputs/lambda0-preview
```

`check` validates the model service, not live sensors or robot safety. `predict` requires an exported aligned episode and does not publish robot actions. Before `run`, establish camera/hand readiness and explicitly initialize the real base controller under the hardware safety procedure: pinned SONIC publishes `g1_debug` only in CONTROL, not WAIT_FOR_CONTROL. This physical initialization is separate from executing VLA predictions. Live preview publishes no VLA actions, but its base controller is already active and may hold the robot. Use a new output directory for every command.

Execution is an explicit additional step, not the default:

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --execute --chunks 1 \
  --initial-episode data/robot/aligned/episode-001 \
  --output outputs/lambda0-execute
```

Do not run it until the [external runtime, hand-unit and measured initial-pose gates](tutorial.zh-CN.md#6-实时预览与显式执行) are satisfied. The reference episode is not a homing command. Stopping the client, disabling policy into Hold, and physically stopping/powering down the robot are [three different operations](tutorial.zh-CN.md#7-三种停止不可混为一谈).

## Output

Each run creates `deployment.json` and `session.json`. Each prediction creates per-view JPEGs, an NPZ containing `state.*` and `action.*` arrays, and JSON containing observation metadata, timestamps, action specifications and inference diagnostics.

Live runs also record published step counts. These counts indicate client publication, not actuator completion. Full predicted horizons are retained even when only a prefix is executed. Output directories must be new for each run.

## Extension

Register new adapters in the `SOURCES`, `POLICIES` and `CONTROLLERS` dictionaries in `scripts/infer_robot.py` and select their configuration `type`. Observation sources own acquisition and alignment, policies own model-specific transformations, and controllers own initialization, publication timing and shutdown.

The common observation structure can carry multiple views. The supplied source and policy accept one ego image; other layouts require suitable source and model implementations. Adapter method signatures are listed in the [extension reference](technical.zh-CN.md#8-扩展接口).

## Verification boundary

The following paths have been exercised:

- Client and runtime regression tests cover signed messages, hand leases, camera timing and source preparation.
- The `serve → status → prepare → stop` workflow was checked without hardware. Monitor mode does not authorize actions, and missing backends fail preparation.
- Both patches applied to the pinned public SONIC revision. The full executable compiled and loaded on Linux x86_64 with CUDA runtime libraries, TensorRT and ONNX Runtime. This does not verify GPU controller execution.
- A C++ ZeroMQ harness exercised ownership, replay/stale rejection, watchdog behavior and release.
- Linux camera capture received an FFmpeg-generated H.264/MPEG-TS stream with advancing kernel-ingress timestamps and stopped on video loss. Capture waits for a keyframe when joining an existing stream. This does not verify GoPro hardware.
- A compatible G1 checkpoint and held-out recording passed `infer_robot.py check` and offline `predict`, producing finite `[50,64]` motion and two `[50,6]` hand arrays. The input was adapted from canonical data; recording/export and physical execution were not exercised. No robot commands were sent.

These checks do not establish task success, real-time performance or safe operation on a particular robot. Validate live sensors, hand units, controller behavior and emergency-stop procedures on your own hardware before execution.

This release provides terminal runtime code, integration instructions and a beginner tutorial. Users supply their own authorized G1 checkpoint, normalization bundle and recordings. Full hardware acceptance is outside the code-and-documentation release scope. SONIC, SDKs and weights remain under their original licenses; see [NOTICE](NOTICE).

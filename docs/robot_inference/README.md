# Inference

Technical documentation for the observation–policy–controller pipeline. The current implementation connects G1 feedback and an ego camera to Lambda0, then optionally sends actions through a SONIC/Revo2 runtime.

[中文技术说明](technical.zh-CN.md) · [Runtime protocols](interfaces.md)

## Scope

The client handles observation alignment, model requests, action publication and logging. Model loading and normalization belong to `lambda0 serve`. Robot startup, homing, pose preparation and shutdown belong to the external control system.

Live execution requires the existing signed motion router, timestamped RGB publisher and robot-side services. These components are not included here; stock SONIC and the basic robot bridge do not implement all required interfaces.

The supplied adapters use one ego image, a 44-dimensional model state and 50 predictions of 76-dimensional actions. Scheduling is serial, with no RTC or action prefetch.

## Implementation

All client logic is in `scripts/infer_robot.py`. The file is organized into data contracts, configuration, observation decoding and alignment, the Lambda0 HTTP client, signed action messages, SONIC session management, and the command-line runner.

`G1Observations`, `LambdaPolicy` and `SonicController` define the current observation, policy and controller implementations. Model serving remains in `src/openlambda/inference/server.py`.

## Data contracts

`Observation` contains named RGB images, named state vectors, nanosecond timestamps, a clock identifier and metadata. Images are nonempty `uint8[H,W,3]` arrays; state fields are finite `float32[D]` vectors.

`ActionSpec` declares representation, field dimensions, units, frequency and reference. `ActionBatch` contains finite `float32[T,D]` arrays with a shared horizon. Policy output and controller input specifications must match exactly.

The current policy concatenates body joints (29), left and right hand feedback (6 each), and projected gravity (3). Images are resized to 448 × 448. Output fields are `motion[50,64]`, `left_hand[50,6]` and `right_hand[50,6]`. Hand targets use normalized STARK units and are clipped to `[0,1]`; tokens outside ±1.25 are rejected.

## Configuration

The configuration is divided into `observation`, `policy`, `controller` and `schedule`, with a top-level task `instruction`. See [`configs/lambda0_g1.json`](../../configs/robot_inference/lambda0_g1.json) for the complete schema example and the [parameter reference](technical.zh-CN.md#4-配置参数) for field descriptions.

The checkpoint path, normalization dataset and inference settings must agree with the model service. The sample embodiment ID is a placeholder. Controller PID, binary path and robot address are required for execution. All configured service endpoints use loopback addresses.

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

Use Python 3.10+ from the repository root. Install the model environment according to the [repository instructions](../../README.md#lambda0-install).

```bash
python -m pip install -r scripts/requirements-robot-inference.txt
cp configs/robot_inference/lambda0_g1.json configs/robot_inference/local.json
```

Start `lambda0 serve` with the checkpoint's normalization file and embodiment ID, action space 2, and RTC disabled. The [technical specification](technical.zh-CN.md#6-运行示例) includes the server command and execution examples.

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json check
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --chunks 5 --output outputs/lambda0-preview
```

## Output

Each run creates `deployment.json` and `session.json`. Each prediction creates per-view JPEGs, an NPZ containing `state.*` and `action.*` arrays, and JSON containing observation metadata, timestamps, action specifications and inference diagnostics.

Live runs also record published step counts. These counts indicate client publication, not actuator completion. Full predicted horizons are retained even when only a prefix is executed. Output directories must be new for each run.

## Extension

Register new adapters in the `SOURCES`, `POLICIES` and `CONTROLLERS` dictionaries in `scripts/infer_robot.py` and select their configuration `type`. Observation sources own acquisition and alignment, policies own model-specific transformations, and controllers own initialization, publication timing and shutdown.

The common observation structure can carry multiple views. The supplied source and policy accept one ego image; other layouts require suitable source and model implementations. Adapter method signatures are listed in the [extension reference](technical.zh-CN.md#8-扩展接口).

Validation currently covers synthetic observations and a mock model service. Real-checkpoint inference and robot operation have not been validated for this refactor.

# SIMPLE simulation evaluation

Run the OpenLambda HTTP server in the model environment and the AMO-based SIMPLE evaluator in its own simulation environment. The six-task integration uses the AMO adapter and `simple.cli.eval`; each task must provide a compatible AMO environment. No simulator dependencies are added to the model package, and the integration does not modify the installed SIMPLE checkout.

## Prerequisites

- Install OpenLambda and its `serve` extra in the model environment following the repository installation guide.
- Install the original AMO-based SIMPLE tasks, Isaac/MuJoCo runtime, AMO controller weights, scene assets and LeRobot evaluation datasets. The current launcher requires `simple.evals.api.EvalConfig` and `simple.cli.eval.run_eval`; this API requirement is separate from the controller version.
- Run the evaluation command with SIMPLE's Python interpreter. The launcher itself uses only the Python standard library; `--help` does not load the simulator.
- Keep this checkout's `scripts/`, `configs/simple/` and `examples/simple/` together.
- Configure GPU/rendering and network access in the environment. The launcher inherits CUDA, Vulkan and EGL settings; it does not allocate GPUs or create SSH tunnels.

## Prepare the model

A bare `model.safetensors` plus normalization files is not a loadable model directory for the current server. Use the original full checkpoint, or an inference export containing its original `config.json`, processor/tokenizer assets and `data_bundle/manifest.json`, in addition to the main weights. EMA and optimizer/training state are unnecessary. Do not reconstruct model configuration from defaults.

The server reads a local checkpoint directory, not a Hugging Face repository ID. Download the [complete public inference export](https://huggingface.co/lambda-humanoid/lambda0-simple-6task-100k/tree/8c8fbd81c852192c0f8c1128755ccd077358a638) before starting it. Pin the linked revision to keep the weights, configuration, processor and manifest consistent.

Renaming statistics files does not change their internal dataset identities. The six tasks share model weights, not normalization statistics:

| Task | Backend | Statistics dataset | Suggested statistics path |
| --- | --- | --- | --- |
| `handover` | AMO | `psi_simple_task_00` | `normalization/handover.json` |
| `bend_pick` | AMO | `psi_simple_task_01` | `normalization/bend_pick.json` |
| `tabletop_grasp` | AMO | `psi_simple_task_02` | `normalization/tabletop_grasp.json` |
| `xmove_pick` | AMO | `psi_simple_task_03` | `normalization/xmove_pick.json` |
| `mobile_pick_place` | AMO | `psi_simple_task_10` | `normalization/mobile_pick_place.json` |
| `xmove_bend_pick` | AMO | `psi_simple_task_11` | `normalization/xmove_bend_pick.json` |

The environment IDs, policy names and dataset identities are defined in `configs/simple/tasks.json`.

## Start the policy server

Example for `bend_pick`, from the OpenLambda checkout in the model environment:

```bash
CHECKPOINT=/path/to/complete/ckpt-100000
NORMALIZATION=/path/to/normalization/bend_pick.json
python scripts/serve.py \
  --model "$CHECKPOINT" \
  --normalization "$NORMALIZATION" \
  --default-embodiment-id 7 \
  --default-action-space-id 3 \
  --inference-steps 10 \
  --device cuda \
  --host 127.0.0.1 --port 8000 \
  --output /path/to/new/server-output
```

The original checkpoint statistics path `data_bundle/stats/psi_simple_task_01.json` also works. Keep RTC and inference seeding disabled for this baseline interface. One server binds one task's statistics; restart with the appropriate file when changing tasks, or run separate instances. Do not infer the task from the language instruction.

Use a reachable private interface or a tunnel when simulation runs on another machine. Do not expose an unauthenticated inference server publicly.

## Run one task and level

For a headless EGL setup, configure the simulation process before importing MuJoCo:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Also source your SIMPLE installation's GPU/runtime environment setup. Vulkan ICDs, EGL device selection and library paths are machine-specific; the launcher does not override them.

From this checkout, using SIMPLE's Python/environment:

```bash
python scripts/eval_simple.py \
  --task bend_pick \
  --host 127.0.0.1 --port 8000 \
  --level level-0 \
  --data-root /path/to/SIMPLE/data/evals/simple-eval/G1WholebodyBendPickMP-v0/level-0 \
  --num-episodes 10 \
  --output /path/to/new/results/bend_pick-level-0
```

`--data-root` is the actual task/level LeRobot directory containing `meta/info.json`, not its parent collection directory. `--level` labels the native split; it does not select a subdirectory automatically. The output directory must not already exist.

For an integration smoke, use `--num-episodes 1 --max-episode-steps 120`. This short budget checks transport, controller execution and result persistence, not task success or benchmark performance. AMO has a 60-step stand warmup, so do not choose a budget that only exercises warmup.

Other options include `--episode-start`, `--success-criteria`, `--gui`, `--no-save-video` and `--num-workers`. One worker is the default; additional workers use the installed native evaluator's spawn implementation and resource requirements.

## Protocol and control semantics

Explicit integration settings are LeRobot input, `mujoco_isaac`, headless rendering, videos enabled, one worker, 10 inference steps, embodiment ID 7, action-space ID 3, RTC disabled and success threshold 0.9. With no `--max-episode-steps`, the native task metadata chooses the limit.

Use a SIMPLE revision in which all six selected environments are AMO-based. A newer checkout may reuse environment IDs with a different robot/controller; changing the evaluator alone does not convert those environments to AMO. The launcher records the registered environment entry point and rejects non-AMO environments before simulator startup; the adapter also rejects non-AMO robots. It never overrides task registrations or patches reset methods. These settings are not a claim to reproduce every historical evaluation. The launcher records the installed evaluator revision/source hash and working-tree status and delegates reset and warmup to it. A step override is passed directly to SIMPLE; it is not automatically a policy-only budget.

The AMO adapter consumes one left-head RGB image, the task instruction and a 32-dimensional state, then executes the returned 36-dimensional action chunk. A missing instruction or malformed/nonfinite action chunk is an error.

- AMO state uses commanded torso roll/pitch/yaw and absolute commanded height. Predicted waist joint PD targets remain separate from torso commands.
- Episode reset clears queued actions through the native base class and resets policy history and the stand-warmup counter.
- The adapter preserves the training hand/arm ordering, full model action chunks and native AMO controller execution.
- Velocity prediction describes the flow model's sampling parameterization. Do not integrate all returned action dimensions as physical velocities or normalize them a second time.

## Integration mechanism

The wrapper calls the public `run_eval(EvalConfig(...))` API in a subprocess. The inspected SIMPLE implementation accepts `agent_factory` but does not forward it to workers; workers still import `simple.baselines.<policy>`.

`examples/simple/` provides a process-local package search-path overlay exposing `openlambda_amo`. It retains native SIMPLE initialization and native resource loaders so scene/controller resources still resolve to the installed SIMPLE directory. The overlay is inherited by native spawned workers. Do not copy these files over SIMPLE's existing baselines or install them as a replacement `simple` package.

## Results and failure behavior

- `selection.json`: task, environment, dataset path and requested episode range.
- `protocol.json`: effective native configuration, policy health, installed evaluator provenance and selected runtime environment settings.
- Native logs, `eval_stats.txt` and episode videos: produced by SIMPLE.
- `summary.json`: exact episode accounting, process exit code, video completeness and success rate.

The launcher checks the task's normalization identity and model-serving contract before simulation. A native error, missing/duplicate/unexpected episode, malformed result line, or missing/empty requested video results in a nonzero exit and no reported success rate. A completed episode with `False` success is a valid evaluation outcome, not an infrastructure error.

This interface does not perform retries, checkpoint selection, asset repairs, resume merging or multi-node scheduling. Keep protocol variants and outputs separate rather than combining scores from different checkpoints or thresholds.

## Task compatibility

Some SIMPLE versions register the following tasks with `SonicLocoManipEnv` and `g1_sonic`, rather than the AMO `LocoManipulationEnv` and `g1_wholebody` required by this adapter:

| Task | Environment ID to check |
| --- | --- |
| `handover` | `simple/G1WholebodyHandoverTeleop-v0` |
| `xmove_pick` | `simple/G1WholebodyXMovePickTeleop-v0` |
| `mobile_pick_place` | `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0` |
| `xmove_bend_pick` | `simple/G1WholebodyXMoveBendPickTeleop-v0` |

Use matching AMO task definitions, controller weights and scene assets. Renaming a registration or replacing a reset method does not make a SONIC environment compatible. The launcher rejects incompatible environments before evaluation.

## Validation coverage

The following checks used the linked checkpoint, episodes 0–9 at each difficulty, task-specific normalization, ten inference steps, RTC disabled, success threshold 0.9 and native episode budgets (400 steps for `bend_pick`, 800 for `tabletop_grasp`). Videos were recorded. The simulator installation included local modifications, so these results are integration evidence, not a benchmark reproduced from a clean upstream revision.

| Task | Level 0 | Level 1 | Level 2 |
| --- | --- | --- | --- |
| `bend_pick` | 10/10 | 10/10 | 10/10 |
| `tabletop_grasp` | 10/10 | 9/10 | 8/10 |

Each value is successful/completed episodes. Failed task outcomes were retained. The other four tasks were not evaluated with a compatible AMO environment; no full six-task score is claimed.

For your own evaluation, preserve the generated protocol and result files alongside the simulator revision, local changes, controller assets and dataset versions. Use separate output directories for different configurations. If concurrent cold starts fail while loading renderer assets, start simulator processes sequentially; do not count infrastructure failures as completed episodes.

# SIMPLE simulation evaluation

Run the existing OpenLambda HTTP server in the model environment and the original AMO-based SIMPLE evaluator in its own simulation environment. All six tasks use the AMO adapter and `simple.cli.eval`. No simulator dependencies are added to the model package, and the integration does not modify the installed SIMPLE checkout.

## Prerequisites

- Install OpenLambda and its `serve` extra in the model environment following the repository installation guide.
- Install the original AMO-based SIMPLE tasks, Isaac/MuJoCo runtime, AMO controller weights, scene assets and LeRobot evaluation datasets. The current launcher requires `simple.evals.api.EvalConfig` and `simple.cli.eval.run_eval`; this API requirement is separate from the controller version.
- Run the evaluation command with SIMPLE's Python interpreter. The launcher itself uses only the Python standard library; `--help` does not load the simulator.
- Keep this checkout's `scripts/`, `configs/simple/` and `examples/simple/` together.
- Configure GPU/rendering and network access in the environment. The launcher inherits CUDA, Vulkan and EGL settings; it does not allocate GPUs or create SSH tunnels.

## Prepare the model

A bare `model.safetensors` plus normalization files is not a loadable model directory for the current server. Use the original full checkpoint, or an inference export containing its original `config.json`, processor/tokenizer assets and `data_bundle/manifest.json`, in addition to the main weights. EMA and optimizer/training state are unnecessary. Do not reconstruct model configuration from defaults.

The current server reads a local checkpoint directory, not a Hugging Face repository ID. Download a complete export before starting it. The initial `lambda-humanoid/lambda0-simple-6task-100k` upload contains weights and normalization only; it needs the matching original configuration and processor assets before standalone serving.

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

## HOPE verification (2026-11-11)

The uploaded checkpoint at HF revision `fed5851e57c81105c3eeb536fc84da4c1ff810bf` was copied to `hope-5090-01`, with its original configuration, processor/tokenizer and manifest restored from the training checkpoint. Its main-weight SHA-256 is `8dd9ebc7ec148ee58ff67d76f61fa4ec37bc1544b48596b3192deab1597f0827`, identical to the HF LFS object. This did not change the HF repository.

`bend_pick` and `tabletop_grasp` each completed level-0 episode 0 successfully with both inference and simulation on HOPE RTX 5090 D v2 GPUs, using the native episode budget and recording videos. This is an integration check, not a six-task benchmark score.

The prepared model/runtime and evidence are under the following HOPE directory. Run the server in one terminal:

```bash
BASE=/cfs/mlp/training/docker/user/hadoop-aipnlp/xuchongyang04
RUN=$BASE/lambda0-eval/openlambda-hf-amo-20261111
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$RUN/site" \
  "$BASE/lambda0-eval/policy-venv/bin/python" "$RUN/openlambda/scripts/serve.py" \
  --model "$RUN/model" --normalization "$RUN/model/normalization/bend_pick.json" \
  --default-embodiment-id 7 --default-action-space-id 3 --inference-steps 10 \
  --device cuda --host 127.0.0.1 --port 23582 \
  --output "$RUN/policy-bend-pick-$(date +%s)"
```

Run simulation in another terminal after sourcing the native runtime. These library paths are specific to the prepared HOPE installation:

```bash
BASE=/cfs/mlp/training/docker/user/hadoop-aipnlp/xuchongyang04
RUN=$BASE/lambda0-eval/openlambda-hf-amo-20261111
source "$BASE/SIMPLE/.venv/simple_env.sh"
export LD_LIBRARY_PATH="$BASE/lambda0-eval/ffmpeg7/av.libs:$BASE/lambda0-eval/system-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$BASE/lambda0-eval/system-libs/sbin:$PATH"
export MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SIMPLE_DISABLE_TUI=1
"$BASE/SIMPLE/.venv/bin/python" "$RUN/openlambda/scripts/eval_simple.py" \
  --task bend_pick --host 127.0.0.1 --port 23582 \
  --level level-0 --num-episodes 1 \
  --data-root "$BASE/SIMPLE/data/evals/simple-eval/G1WholebodyBendPickMP-v0/level-0" \
  --output "$RUN/results-bend-pick-$(date +%s)"
```

Check available GPUs before reuse. Evidence from the HOPE-local runs is retained in `results-hope-local/bend_pick/` and `results-hope-local/tabletop_grasp/`. For tabletop, change the server's normalization file to `tabletop_grasp.json`, the evaluation task to `tabletop_grasp` and the data directory's environment name to `G1WholebodyTabletopGraspMP-v0`.

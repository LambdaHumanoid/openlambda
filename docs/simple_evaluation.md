# SIMPLE Simulation Evaluation

Evaluate λ₀ on all six SIMPLE core tasks using OpenLambda's HTTP policy server (`scripts/serve.py`), unified launcher (`scripts/eval_simple.py`), and prebuilt evaluation Docker image (`lambda0-simple-eval:v1`).

## Quickstart

### 1. Prerequisites

- **Model environment (host):** Python 3.11 or 3.12 with OpenLambda (`serve` extra), `huggingface_hub`, and `zstd`:
  ```bash
  PYTHON=python3.11 bash scripts/install.sh serve
  source .venv/bin/activate
  python -m pip install huggingface_hub
  ```
- **Docker with NVIDIA GPU support** (`nvidia-container-toolkit`) for running `lambda0-simple-eval:v1`.
- **SIMPLE evaluation assets (`simple-data/`):** Download the official SIMPLE scene assets, robot models, and LeRobot evaluation datasets (`evals/simple-eval/<EnvName>/level-{0,1,2}`) following the [SIMPLE repository](https://github.com/physical-superintelligence-lab/SIMPLE).

### 2. Download checkpoints and Docker image

*(Optional for faster downloads in China: `export HF_ENDPOINT=https://hf-mirror.com`.)*

```bash
# Download one task checkpoint (e.g. xmove_pick), or omit --include to download all six
hf download lambda-humanoid/lambda0-simple-checkpoints \
  --include "xmove_pick/*" --local-dir ./checkpoints

# Download and load the prebuilt SIMPLE evaluation image (lambda0-simple-eval:v1)
hf download lambda-humanoid/lambda0-simple-eval-docker \
  --repo-type dataset \
  --include "lambda0-simple-eval-v1.tar.zst" \
  --local-dir ./docker

zstd -dc ./docker/lambda0-simple-eval-v1.tar.zst | docker load
```

### 3. Start the policy server (host)

Pick the matching task directory and stats file from the [6-task reference table](#six-task-reference):

```bash
TASK=xmove_pick
STATS=psi_simple_task_03.json

python scripts/serve.py \
  --model "./checkpoints/$TASK" \
  --normalization "./checkpoints/$TASK/data_bundle/stats/$STATS" \
  --default-embodiment-id 7 \
  --default-action-space-id 3 \
  --inference-steps 10 \
  --inference-seed 0 \
  --device cuda --dtype bfloat16 \
  --host 127.0.0.1 --port 8000 \
  --output "./server_out/$TASK"
```

Keep RTC disabled. One server instance binds one task's checkpoint and normalization statistics; restart with the matching files when switching tasks, or run separate ports.

### 4. Run evaluation in Docker

From the OpenLambda repository root (replace `/path/to/simple-data` with your local SIMPLE `data/` directory):

```bash
docker run --rm --gpus all --network host --shm-size=16g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e SIMPLE_DISABLE_TUI=1 \
  -e no_proxy=127.0.0.1,localhost -e NO_PROXY=127.0.0.1,localhost \
  -v "$(pwd):/openlambda:ro" \
  -v "/path/to/simple-data:/workspace/SIMPLE/data:ro" \
  -v "$(pwd)/eval_out:/eval_out" \
  lambda0-simple-eval:v1 \
  /workspace/SIMPLE/.venv/bin/python /openlambda/scripts/eval_simple.py \
    --task xmove_pick \
    --host 127.0.0.1 --port 8000 \
    --level level-0 \
    --data-root /workspace/SIMPLE/data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/level-0 \
    --num-episodes 10 \
    --output /eval_out/xmove_pick-level-0
```

- `--data-root` must point to the specific `level-{0,1,2}` directory containing `meta/info.json`.
- `--output` must be a new directory that does not already exist.
- Passing `no_proxy=127.0.0.1,localhost` prevents host HTTP proxies from intercepting localhost requests between the container and `serve.py`.
- For a fast smoke test, add `--num-episodes 1 --max-episode-steps 120` (for AMO tasks, keep `--max-episode-steps` above the 60-step stand warmup).
- If running in a native host SIMPLE installation instead of Docker, export `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` and run `scripts/eval_simple.py` with SIMPLE's Python interpreter.

## Six-task reference

All task definitions, environment IDs, default thresholds and action horizons live in [`configs/simple/tasks.json`](../configs/simple/tasks.json).

| `--task` | Normalization stats (`data_bundle/stats/`) | `--data-root` relative to `simple-data/evals/simple-eval/` | Backend | Default threshold |
| :--- | :--- | :--- | :--- | :-: |
| `bend_pick` | `psi_simple_task_01.json` | `G1WholebodyBendPickMP-v0/level-{0,1,2}` | `amo` | `0.9` |
| `tabletop_grasp` | `psi_simple_task_02.json` | `G1WholebodyTabletopGraspMP-v0/level-{0,1,2}` | `amo` | `0.9` |
| `handover` | `psi_simple_task_00.json` | `G1WholebodyHandoverTeleop-v0/level-{0,1,2}` | `decoupled_wbc` | `0.7` |
| `xmove_pick` | `psi_simple_task_03.json` | `G1WholebodyXMovePickTeleop-v0/level-{0,1,2}` | `decoupled_wbc` | `0.7` |
| `xmove_bend_pick` | `psi_simple_task_11.json` | `G1WholebodyXMoveBendPickTeleop-v0/level-{0,1,2}` | `decoupled_wbc` | `0.7` |
| `mobile_pick_place` | `psi_simple_task_10.json` | `G1WholebodyLocomotionPickBetweenTablesTeleop-v0/level-{0,1,2}` | `decoupled_wbc` | `0.4` |

Common CLI flags for `scripts/eval_simple.py`: `--episode-start`, `--num-episodes`, `--success-criteria`, `--action-horizon`, `--max-episode-steps`, `--gui`, `--no-save-video`, `--num-workers`.

## Outputs and validation

Each run writes to `--output`:

- `selection.json` — task, environment, backend, dataset path and requested episode range.
- `protocol.json` — effective native config, policy health check, SIMPLE source hash and inherited runtime env.
- `eval_stats.txt`, logs and episode videos — generated by SIMPLE.
- `summary.json` — episode completeness check, video completeness check, process exit code and success rate.

Before launching the simulator, the launcher checks the server's `/health` contract (matching dataset, embodiment ID `7`, action-space ID `3`, `10` inference steps, RTC disabled) and verifies that the registered SIMPLE environment matches the task backend (`LocoManipulationEnv` for `amo`; `SonicLocoManipEnv` for `decoupled_wbc`). Missing episodes, malformed lines or missing requested videos cause a nonzero exit with `success_rate: null`.

## Architecture notes

- `examples/simple/` provides a process-local namespace overlay exposing `openlambda_amo` and `openlambda_decoupled_wbc` without modifying the installed SIMPLE checkout.
- Both adapters send one left-head RGB image, the task instruction and a 32-dim proprioceptive state, and execute the returned 36-dim action chunk:
  - **`openlambda_amo`:** observes commanded torso roll/pitch/yaw and absolute height (`+0.75`); executes the full model chunk by default (overridable via `--action-horizon`).
  - **`openlambda_decoupled_wbc`:** observes measured waist roll/pitch/yaw (`qpos[[13, 14, 12]]`) and the last executed base height (`0.74`, `SIMPLE_HEIGHT_SNAP=1`), executes `SIMPLE_ACTION_HORIZON=20` steps per chunk, and advances the controller clock with `SIMPLE_CLOCK_MODE=sim_step` when supported.
- Checkpoints must be complete directories (`model.safetensors`, `config.json`, processor/tokenizer files, `data_bundle/manifest.json`, and `data_bundle/stats/*.json`). EMA and optimizer states are not required.
- If concurrent cold starts fail while loading renderer assets on the same host, stagger simulator startups by a few seconds.

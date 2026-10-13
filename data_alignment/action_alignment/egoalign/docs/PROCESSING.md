# Processing workflow and commands

This guide covers the data-processing implementation accompanying
[EgoAlign](../README.md), not the VLA training or physical deployment stack.
Install the [Python package](../README.md) and configure the
[compatible runtime](public_runtime.md) before running replay.

CLI paths are relative to your current working directory. Keep input, output and
telemetry directories disjoint (neither identical nor nested); `--overwrite`
does not bypass this protection. Prefer a fresh output directory. The sample's
`state.json` records `running`, then `complete` or `failed`, with the current
step and timestamps. `commands.json` records completed subprocess return codes.

## Two processing modes

- `endpoint-align`: bilateral palm alignment,
  followed by measured replay residual refinement; used by the Object Relocation example.
- `replay-only`: standard preprocessing and actual SONIC–MuJoCo replay without
  endpoint optimization; requires user-supplied inputs (no pedal data bundled).

Both retain causal controller history and export the same 43D state and captured
64D motion token, plus two 6D hand actions. Replay-only never means zero history.

## Endpoint alignment

SMPL/Pico-derived reference → canonical scale and wrist cleanup → bilateral
geometric alignment (Round 0) → measured SONIC–MuJoCo residual refinement (Rounds
1/2) → final causal state and same-control-tick motion-token export.

Wrist cleanup precedes optimization. Each candidate is validated before replay;
geometric candidate error is distinct from measured controller error. SONIC
determines the simulated robot motion. No synthetic arm swing or
outward-abduction prior is applied.

EgoAlign has its own algorithm identity; candidate **schema** version remains 4.
Do not relabel artifacts from other algorithms as this mode. Optimizer parameters
and causal reconstruction semantics are retained from the source implementation.

## Object Relocation sample

Follow [the sample recipe](../examples/README.md) for the complete sequence:
Stage A → morphology calibration → unretargeted baseline replay → endpoint
alignment/refinement → accepted-round Stage C export. The recipe initializes
its execution state and records commands and per-step logs.

## Endpoint alignment with existing Stage A/B artifacts

Read [the data contract](DATA_CONTRACT.md) and [runtime setup](public_runtime.md).
The adapter expects aligned Pico/LeRobot inputs, not an arbitrary `poses.npy`.
Supply matching Stage-A data, its successful baseline replay and a calibrated
morphology JSON. Their source mapping, row counts, timing and asset identity must
agree; directory names alone do not establish compatibility. Manifests also bind
artifact paths. Moving old outputs can invalidate reuse: do not rewrite identity
metadata to bypass validation; create fresh artifacts at their intended paths.

Initialize the execution state **once**, before preflight. Exclusive creation
below refuses to overwrite an existing state file; for a resumed run, skip this
block and keep its original state and artifact paths.

```bash
export EGOALIGN_WORK_ROOT=/absolute/path/to/new_egoalign_work
"$EGOALIGN_DATA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EGOALIGN_WORK_ROOT"])
root.mkdir(parents=True, exist_ok=True)
with (root / "state.json").open("x") as stream:
    json.dump({"status": "running"}, stream)
    stream.write("\n")
PY
```

Preflight validates existing artifacts and writes a report; it does not start
simulation. Use the runtime/asset environment configured in the installation guide.

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign endpoint-align \
  --raw-dataset /path/to/aligned_raw \
  --baseline-dataset /path/to/baseline_stage_a \
  --baseline-contract minimal_ab \
  --baseline-replay-root /path/to/baseline_stage_b \
  --output-root "$EGOALIGN_WORK_ROOT/alignment" \
  --state-json "$EGOALIGN_WORK_ROOT/state.json" \
  --morphology /path/to/g1_morphology.json \
  --endpoint-parts hands --hand-endpoint-side both \
  --workers 1 --mode preflight --dry-run
```

After checking assets, libraries and unused ports, set
`EGOALIGN_ALLOW_SIM_REPLAY=1`. Reuse the arguments above with
`--mode smoke --episodes <one-compact-id>` and remove `--dry-run`. Inspect the
smoke before using `--mode full`; use `--resume` when continuing existing work.
Early stopping may select Round 0 or Round 1: never claim an unexecuted round was
tested. Promote only the accepted round with `python -m egoalign promote --help`.
Stage-A output alone is not train-ready.

## Replay only

The [user-input recipe](../examples/README.md#optional-replay-only-bring-your-own-input)
shows Stage-A conversion with an explicit task instruction. Given the complete
Stage-A output, plan replay and export with:

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign replay-only \
  --source-root /path/to/staging \
  --replay-root /path/to/new_replay \
  --output-root /path/to/new_final --dry-run
```

Remove `--dry-run` after checking the isolated simulation driver. This dry run
only prints the plan; it does not validate the input dataset. Existing
telemetry requires explicit `--reuse-replay`; source/timing/capture checks still
run. All source episodes are exported, preventing a mixture of processed and
untouched rows. This mode does not apply a bilateral endpoint-error gate.

Keep command logs, resolved configuration, source/asset hashes and validation
reports with every output. Follow [validation checks](VALIDATION.md) before using
the exported state–action pairs.

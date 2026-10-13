# Object Relocation reproduction and user-supplied replay-only inputs

The public sample demonstrates **Object Relocation** (物体转运), the paper's task
of picking up a basket, transporting it, and placing it on a table. `basket`
remains the CLI selector, directory name, and historical validation identifier;
it describes the example object, not a separate task. The recorded instruction
`Move the basket to the table` is unchanged.

First configure the [public source runtime and launcher](../docs/public_runtime.md),
including interpreter/asset paths, unused ports and `EGOALIGN_ALLOW_SIM_REPLAY=1`.
Run the commands below from the EgoAlign checkout. The public launcher's fresh
build and Object Relocation recipe passed on 2026-10-08
using existing dependencies (not a clean-machine installation).
Inputs use the [documented raw schema](../docs/DATA_CONTRACT.md), not arbitrary
SMPL arrays. The recipe selects compact raw episode `0`; retain its raw parquet,
sync/alignment metadata, dataset metadata and referenced camera files.
Only the object-relocation sample and its video-only camera files are bundled, in
`examples/data/basket/`.
See [sample contents and integrity](SAMPLE_DATA.md) and the sample's `sample_manifest.json`.

## Object Relocation: endpoint alignment

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign.sample_pipeline basket \
  --raw-root examples/data/basket \
  --robot-xml "$EGOALIGN_RUNTIME_ROOT/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml" \
  --output-root /path/to/new/basket-result --dry-run
```

Remove `--dry-run` to execute. This runs Stage A, morphology calibration, fresh
unretargeted replay, bilateral alignment/refinement and promotion.
Normal acceptance gates determine the final round; no threshold relaxation or
forced extra rounds is enabled. Failure stops the recipe with logs; it must not
be hidden by manually choosing an earlier round.

## Optional replay only: bring your own input

Supply your own schema-compatible dataset and its camera alignment metadata.
Use the general commands below so that the task instruction is yours, not a
sample-specific label. The first command **executes** Stage A (it has no dry-run
option); choose a fresh output directory. `--episodes 0` selects source episode 0;
omit `--episodes` to process all episodes, or use a selection such as `0-4,7`.

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign stage-a \
  --source-root /path/to/your/compatible_raw_dataset \
  --output-root /path/to/new/replay-only-result/stage_a \
  --episodes 0 --require-alignment --decoder-history-mode zero --workers 1 \
  --task "Your task instruction"

"$EGOALIGN_DATA_PYTHON" -m egoalign replay-only \
  --source-root /path/to/new/replay-only-result/stage_a \
  --replay-root /path/to/new/replay-only-result/baseline_replay \
  --output-root /path/to/new/replay-only-result/final --workers 1 --dry-run
```

The second command only prints the replay/export plan; remove its `--dry-run`
after checking the runtime. It runs actual SONIC–MuJoCo replay and causal export
without endpoint optimization. Omitting `--task` from Stage A uses the source's
first task; an explicit task applies to every converted episode. Zero-history
in Stage A is staging only; final labels come from the actual controller replay.

The Object Relocation recipe writes `commands.json` and per-step logs. For manual
commands, retain your command line and logs; replay-only also writes
`final/meta/processing_mode.json`. Outputs include `stage_a/`, `baseline_replay/`
and `final/`; endpoint alignment also has `alignment/`, `morphology.json` and
`state.json`. Raw inputs are untouched. Stage A aligns/samples camera video;
Stage C preserves those converted videos and row timestamps. Intermediate
simulation-video generation is disabled by the public adapter.

## Export the final MuJoCo video

After processing, select the accepted replay from
`alignment/episode_000000/final_decision.json` (`accepted_round`). For example,
an accepted round of `2` corresponds to `round_02/replay`. For `replay-only`,
use `baseline_replay` instead. Always take both NPZ files from the same replay.

Install the video encoder in your simulation environment once:

```bash
"$EGOALIGN_SIM_PYTHON" -m pip install av
```

Then export from the EgoAlign checkout, with the runtime and BrainCo asset
environment variables configured as in the runtime guide:

```bash
export EGOALIGN_SELECTED_REPLAY=/path/to/result/alignment/episode_000000/round_02/replay
MUJOCO_GL=egl "$EGOALIGN_DATA_PYTHON" -m egoalign render-replay \
  --sim "$EGOALIGN_SELECTED_REPLAY/telemetry/episode_000000.sim.npz" \
  --control "$EGOALIGN_SELECTED_REPLAY/telemetry/episode_000000.control.npz" \
  --hand-color body-gray \
  --output-dir /path/to/new/final-visualization
```

Outputs are `replay_mujoco.mp4` (1920×1080 H.264, recorded frame rate) and
`render_report.json` (input/video hashes, camera, frame count and fidelity checks).
Use a fresh output directory; existing results are never overwritten.
`--width`, `--height`, `--azimuth`, `--elevation` and `--distance` customize the view.
`--hand-color body-gray` matches the hands to the pelvis shell; `original`
retains the supplied robot asset's colors. This affects rendering only.
Run `python -m egoalign render-replay --help` for all options.

This is offline rendering of measured root/body poses: it does not rerun
optimization or dynamics, start the controller, or open deployment ports.
BrainCo fingers are reconstructed from captured commands using the runtime's
canonical mapping, not measured finger joint trajectories. Commands use the
preceding same-clock capture, with boundary clamping reported explicitly.
Frames retain the recorder's fixed FPS; this video is not a camera/video
resynchronization or an endpoint evaluation. Use the same scene/model as replay;
the standard scene shows the robot, not simulated object-contact success.

## Validation expectations

The current implementation's qualification yielded 1006 frames, with 43D state and
64D motion tokens matching same-tick control capture exactly. Its three measured
palm means were 7.0301 / 3.5260 / 1.4045 cm (11-frame lag-aligned at 50 Hz,
not zero-lag). Round 2 passed the endpoint gate. See
[validation details](../docs/VALIDATION.md). Treat these measurements as
provenance, not bit-exact physics regression targets.
Quality-profile calibration and task-success certification are separate; a
successful sample run does not make every production audit pass.

Validate the final sample against its actual selected replay (not the Stage-A
zero-history placeholders):

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign.verify_sample \
  --stage-a /path/to/result/stage_a --final /path/to/result/final \
  --control /path/to/selected-replay/telemetry/episode_000000.control.npz \
  --output /path/to/result/payload_verification.json
```

For replay-only, the selected replay is `baseline_replay`. For object
relocation (`basket`), use the accepted
round recorded by alignment/promotion; do not assume Round 2 is always selected.
The validator checks finite 43D state/64D token, same-tick payload equality,
hand labels, timestamps and camera hashes. Replay timing and acceptance gates
remain the responsibility of the processing pipeline.

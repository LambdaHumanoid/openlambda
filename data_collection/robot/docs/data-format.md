# Source protocol and episode format

## Supported sources

Use this reference when connecting a different telemetry producer or writing a training adapter. The recorder, hand observer and GoPro relay run on G1; a remote SONIC host needs the clock calibration described in the collection guide.

### SONIC debug (ZeroMQ SUB)

One ZMQ frame: ASCII topic immediately followed by a MessagePack object (no separator). Subscribe to both topics:

| Topic | Required payload |
| --- | --- |
| `robot_config` | Object describing the running robot/controller; periodically republish for late subscribers |
| `g1_debug` | `index`, `timestamp_realtime_ns`, `timestamp_monotonic_ns`, `body_q[29]`, `body_dq[29]`, `base_quat[4]`, `token_state[64]` |

`index` advances strictly. Body positions/velocities are measured radians/radians per second in the order below. `base_quat` is WXYZ, unit length. `token_state` is the online encoder output for this sample. Both timestamps belong to the SONIC producer OS. Nominal rate: 50 Hz. A restart/index reset ends the episode. Joint-vector lengths, finite values, physical bounds and clock freshness are checked.

Configure the SONIC debug publisher to include these fields. A raw Unitree DDS state stream needs an adapter and the corresponding SONIC encoder output before it can be used here.

### Revo2 telemetry (ZeroMQ SUB)

Two ZMQ frames: topic, MessagePack object. All timestamps are on G1. Every object has `sequence`, `monotonic_ns`, `wall_time_ns`.

| Topic | Additional fields |
| --- | --- |
| `rt/brainco/left/state` | Six-element `positions`, `speeds`, `currents`, `states` |
| `rt/brainco/right/state` | Same as left |
| `rt/brainco/glove/applied` | Six-element `left` and `right` normalized STARK targets in `[0,1]` |

Joint order: `thumb, thumb_aux, index, middle, ring, pinky`. Position feedback is in STARK `[0,1000]` units. Feedback and targets are separate measurements with separate timestamps.

The provided `lambda-robot-hand-telemetry` emits this protocol on loopback port 18560. It timestamps ROS callback receipt. It records targets published on `/joint_commands_*` in their existing normalized STARK units. The timestamps describe callback receipt, rather than a hardware acknowledgement. It only emits a combined hand target when both commands were observed within 50 ms. Targets are emitted when commands arrive. The topic keeps the original `glove/applied` name for compatibility; any controller publishing the documented ROS commands can supply them.

For a custom driver, check its joint order, timestamp origin and command units. Thumb targets in project units need conversion to STARK units before publication.

### Video

The existing robot GoPro relay forwards MPEG-TS UDP datagrams to `127.0.0.1:18554`. The recorder saves the unmodified bytes and extracts H.264/H.265 video PES presentation timestamps. Each distinct PTS is associated with a G1 monotonic and realtime ingress timestamp. These timestamps approximate local video arrival, **not exposure time**. Software/network/camera buffering contributes to alignment error.

## Raw episode

```text
manifest.json          format/status/task/clock mapping, counts, SHA256 + byte sizes
sonic.records          original validated SONIC samples + source/receive clocks
brainco.records        validated hand feedback and published targets
camera.ts              original video stream
camera.timestamps.bin  PTS-to-G1-ingress mapping
```

Record streams use an eight-byte `G1EP\x02\x00\x00\x00` magic. Each record is a big-endian payload length and CRC32 (`!II`) followed by a MessagePack object. The timestamp sidecar uses the fixed structure `!4sB3xQQQQQ`: magic `G1FT`, version 1, epoch, sequence, PTS in 90 kHz units, G1 monotonic ns, G1 realtime ns. The exporter reads these files to produce the aligned episode described below.

Only `status: complete` with matching SHA256/size descriptors is exportable. Interrupted or failed recordings stay `incomplete`. An existing output directory is left unchanged. Keep the full raw directory if you need to change alignment rules later.

## Alignment

For each SONIC anchor, convert both host clocks into the G1 clock domain using the recorded offsets. Select the nearest camera and feedback samples, and the latest hand target whose timestamp is **at or before** the anchor.

- Camera max absolute delta: 75 ms.
- Each hand feedback max absolute delta: 50 ms.
- Applied target max age: 50 ms; future targets are never selected.
- Missing/out-of-window modalities reject that row. Source counters and timestamps are retained.
- Nonfinite/malformed physical samples, clock regressions and clock steps fail export.
- A source-index gap or interval outside 10–30 ms starts a new segment in the nominal 50 Hz output. Do not make training windows across segment boundaries.

The tolerances allow for differences in sampling rate and transport delay. Camera and feedback matching can select a later sample; hand targets are always selected from the past. Matching uses the recorded samples directly, without interpolation.

## Aligned episode

`data.parquet` contains one row per accepted robot anchor. `ego_view.mp4` has one center-cropped, 640×480 RGB-derived image per accepted row. The MP4 uses source-relative variable presentation times (quantized to 1/90000 second); time is not compressed across rejected rows. Inspect using timestamps rather than assuming `row_index / 50`.

| Column | Shape / meaning |
| --- | --- |
| `observation.state` | 44: body q (29), left measured hand (6), right measured hand (6), projected gravity (3) |
| `action` | 76: online SONIC token (64), left published STARK target (6), right target (6) |
| `observation.valid_mask`, `action.valid_mask` | 44/76 booleans; accepted finite dimensions are true |
| `observation.body_joint_position/velocity` | 29 each; raw measured body values |
| `observation.left_hand_position/right_hand_position` | 6 each; measured STARK feedback divided by 1000 |
| `observation.root_orientation` | 4, WXYZ; retained separately, not added to 44D state |
| `observation.projected_gravity` | World `[0,0,-1]` rotated into body coordinates |
| `action.motion_token`, `action.left_hand_joints/right_hand_joints` | Explicit unflattened action fields |
| `timestamp` | Seconds from the first exported source anchor |
| `frame_index`, `segment_index`, `task` | Output row, continuous segment, instruction |
| `source.*` | Source sequence, mapped/source/receive clocks and camera identity |
| `quality.alignment_delta_ms` | Camera, left feedback, right feedback, target age |

Body order is left leg (6), right leg (6), waist yaw/roll/pitch (3), left arm (7), right arm (7). Exact joint names are in `schema.py` and exported in `episode.json`. This ordering differs from the raw Unitree motor-index order.

`episode.json` records layouts, segment ranges (end exclusive), task, nominal rate, image key mapping, rejection counts and source manifest. When writing a training adapter, use these segment ranges to build continuous windows and map the video and task metadata to your loader. LeRobot dataset metadata and any fixed-rate resampling are handled by that adapter.

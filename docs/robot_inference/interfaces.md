# Runtime interfaces

The client runs on the same OS as SONIC, the RGB subscriber and the signed motion router. All configured endpoints are loopback addresses. SSH forwarding is suitable for hand telemetry, whose source clock remains G1 time. A forwarded remote SONIC publisher does not share the workstation monotonic clock and cannot be used without a clock adapter.

## Lambda0 service

`GET /health` must match `model_path`, `normalization_dataset`, `normalization_action_space_id`, `inference_steps`, and `rtc_enabled: false`. Optional default IDs must agree with the client configuration. The server supplied by `src/openlambda/inference/server.py` implements this API.

`POST /act` receives `image0` (NumPy JSON envelope, uint8 RGB), `proprio[44]`, `language_instruction`, `embodiment_id`, `action_space_id: 2`, `steps`, and `history` with a unique session ID and increasing request index. The response contains `action[50][76]` in physical token/hand units and `action_space_id: 2`. Normalization belongs to the model server.

## SONIC debug

A single ZMQ frame contains `b"g1_debug"` followed immediately by MessagePack. Required fields: `index`, `timestamp_monotonic_ns`, `body_q[29]`, `body_dq[29]`, `base_quat[4]` in WXYZ, and `token_state[64]`. Body positions use the MuJoCo/URDF joint order used by the collection module. The source timestamp is the local SONIC host clock.

## Hands

Two ZMQ frames: `rt/brainco/left/state` or `rt/brainco/right/state`, then a MessagePack object with `positions[6]` in `[0,1000]` and `monotonic_ns` from G1. Order: thumb, thumb_aux, index, middle, ring, pinky. Measured positions are divided by 1000 when constructing state.

The read-only hand observer in `data_collection/robot` can provide these messages. Forward its G1 loopback port over SSH if needed. A controller's own hand telemetry publisher with the same schema also works.

## Camera

The last ZMQ frame is a MessagePack object:

```text
images.ego_view                         base64 JPEG
 timestamps.ego_view_g1_monotonic       float seconds, G1 video ingress clock
 timestamps.ego_view_host_monotonic     float seconds, ingress mapped to host clock
```

The two timestamps refer to the same frame. The host value must be derived from a calibrated clock mapping; timestamping an old frame when it arrives at the client would conceal capture latency. Software ingress time is not camera exposure time.

Only two images are buffered. Body and hands must align within 100 ms of the camera anchor; source and receiver freshness are checked at 250 ms, with at most 50 ms future clock tolerance. After a chunk is published, the next camera, body and mapped hand samples must all be newer than its last publication time. This is a client publication barrier, not an actuator acknowledgement.

## Motion router

The existing deployed router uses a local ZMQ REQ/REP control endpoint and PUSH/PULL policy endpoint. The client requires:

| Request | Required behavior |
| --- | --- |
| `status` | Return `ok`, `phase`, `stable`, `policy_enabled`, `hand_command_ready`, `hand_control_enabled` |
| `policy-bind` | Bind an unowned stable Hold to `identity` and an ephemeral secret |
| `policy-enable` | Authorize that identity; enable both hands or reject |
| `policy-disable` | Stop consuming policy output; confirm `policy_enabled: false` |
| `policy-unbind` | Release only the authenticated identity |

Identity contains exactly `session`, `model`, `task`, `manifestSha256`. Here `manifestSha256` is the client configuration digest. It is not a checkpoint-content hash.

Policy packets use the deployed `g1_revo2_policy_v1` envelope in `scripts/infer_robot.py`: prefix + HMAC-SHA256 + canonical JSON containing identity, increasing sequence, host monotonic timestamp and 76 float values. Secrets are generated per session, sent only to the local router, and are not written to output files.

A compatible router verifies source identity, signature, sequence, age and action bounds, and retains its controller watchdog/hold behavior. It must consume the 64 token dimensions in external-token mode and deliver the 12 hand values in **normalized STARK units**. Do not connect these targets to a bridge that applies project-to-STARK thumb scaling a second time.

The stock SONIC pose socket and public robot bridge do not implement this signed endpoint. This extraction supplies the client, not a replacement router. A new receiver must implement these semantics and be validated separately before physical use. Starting, homing, resetting, handover and robot shutdown remain runtime operations.

## Execution timing

The client sends up to `execute_steps` from each H50 prediction at 50 Hz, then reads fresh observations. It does not prefetch the next chunk or enable RTC. A missed sender deadline greater than 50 ms ends the session instead of sending a backlog in a burst. The runtime may hold the previous target during observation/inference waits. Loss of a sensor or policy/hand readiness stops further client publication.

Initialization requires a prepared stable Hold and measured agreement with the reference episode. No initialization targets or motor-enable commands are sent by this client. Shutdown disables/unbinds its policy session; a failed acknowledgement is reported, and controller shutdown is not inferred from it.

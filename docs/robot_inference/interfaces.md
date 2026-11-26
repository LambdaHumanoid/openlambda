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

`scripts/robot_runtime.py` implements the local ZMQ REQ/REP control and PUSH/PULL signed policy endpoints. It is a terminal receiver, not a second model service. `serve` defaults to monitoring without actuation. The existing inference client requires:

| Request | Required behavior |
| --- | --- |
| `status` | Return `ok`, `phase`, `stable`, `policy_enabled`, `hand_command_ready`, `hand_control_enabled` |
| `policy-bind` | Bind an unowned stable Hold to `identity` and an ephemeral secret |
| `policy-enable` | Authorize that identity; enable both hands or reject |
| `policy-disable` | Authenticate the owner, stop consuming policy output, request the prepared body baseline and confirm `policy_enabled: false` |
| `policy-unbind` | Release only the authenticated identity |

Identity contains exactly `session`, `model`, `task`, `manifestSha256`. Here `manifestSha256` is the client configuration digest. It is not a checkpoint-content hash.

Policy packets use the deployed `g1_revo2_policy_v1` envelope in `scripts/infer_robot.py`: prefix + HMAC-SHA256 + canonical JSON containing identity, increasing sequence, host monotonic timestamp and 76 float values. Secrets are generated per session, sent only to the local router, and are not written to output files.

A compatible router verifies source identity, signature, sequence, age and action bounds, and retains its controller watchdog/hold behavior. It must consume the 64 token dimensions in external-token mode and deliver the 12 hand values in **normalized STARK units**. Do not connect these targets to a bridge that applies project-to-STARK thumb scaling a second time.

The stock SONIC pose socket and public robot bridge do not implement this signed endpoint. Install `scripts/runtime/sonic_runtime.patch` and `openlambda_runtime.hpp` **after** the public collection patch, on SONIC revision `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`. `scripts/setup_robot_runtime.py` prepares these sources. Build and launch the external SONIC controller explicitly with `OPENLAMBDA_RUNTIME=1`, `--input-type zmq_manager`, no encoder/planner model, and `--hand-type none`. Plain `zmq` input does not consume the manager's start/stop command topic. Controller model weights, CUDA/TensorRT/ONNX Runtime, ROS2, Unitree and STARK SDKs remain separately licensed external dependencies.

The backend is a loopback PAIR socket at 5686. The C++ adapter accepts bounded token overrides only inside successful external-decoder CONTROL cycles, before observations are assembled. It publishes feedback after policy inference, including the applied token, owner and accepted sequence. This acknowledges **controller computation**, not motor attainment. WAIT_FOR_CONTROL is reported separately and cannot satisfy Hold. The sole local router owns the stock pose publisher at 5656 and the signed endpoints; these loopback services assume trusted local OS users. They are not a security boundary against malicious same-host processes.

`prepare` and `status` are read-only. `initialize --execute --token initial_token.json` is an explicit physical start/motion operation: it publishes the checkpoint-specific bounded initial token through SONIC's public v4 pose protocol, then requests non-planner start. It requires the patched WAIT_FOR_CONTROL acknowledgement and both ready hands. The patch blocks no-encoder startup until a real token has arrived. Supply the initial token from the matching public SONIC checkpoint; zeros are never substituted by this runtime. `hold --execute` instead captures an already-running external decoder's current token without starting it. Both require `serve --allow-actuation`, successful owner/applied-token acknowledgement, fresh measured body feedback, at least 40 samples spanning one second, max absolute joint velocity ≤0.5 rad/s and upright cosine ≥0.95 before reporting stable Hold.

Policy packets must be no older than 250 ms, never future-dated, have strictly increasing integer sequences, and contain finite token values in `[-1.25,1.25]` and hand values in `[0,1]`. Enable creates a new timestamp barrier and drains pending packets; re-enable does not reset sequence history. Bind, enable, disable and unbind are owner checked; takeover is rejected. Invalid policy packets disable policy publication. Status never exposes credentials.

The normalized hand backend is `scripts/robot_hands.py`, run explicitly on G1 and forwarded from its loopback port 18561. It publishes directly to `/joint_commands_left` and `/joint_commands_right` with names thumb, thumb_aux, index, middle, ring, pinky. It does **not** use `brainco_command_to_stark` or the project's x2 thumb mapping. Actuation requires `--allow-actuation`, live MotorStatus callbacks for both hands, a ROS command subscriber on each side, and no competing command publishers. The service exposes `openlambda_stark_v1` / `normalized_stark`, checks owner credentials, target bounds, sequence, calibrated host-time age and a 250 ms owner lease. Bind calibration is bounded by a 50 ms host round trip. Feedback timestamps are callback ingress, and the external STARK driver's position-unwrapping limitations remain applicable.

The C++ watchdog restores the separately captured baseline after 250 ms without a policy target, even while owner heartbeats continue during model inference. Missing owner heartbeats also revoke ownership. It does not perpetually repeat the last arbitrary policy token. The hand watchdog ceases publication and expires ownership; the SDK/driver may retain its last target. `stop` requires the mode-0600 local administrator credential, disables policy, releases hands and ends the receiver; it is **not torque-off or a robot emergency stop**. Physical E-stop and the upstream controller/driver shutdown remain operator responsibilities. Credentials are ephemeral; stale files after an unclean exit require operator inspection rather than automatic takeover.

## Execution timing

The client sends up to `execute_steps` from each H50 prediction at 50 Hz, then reads fresh observations. It does not prefetch the next chunk or enable RTC. A missed sender deadline greater than 50 ms ends the session instead of sending a backlog in a burst. The runtime's 250 ms policy watchdog returns to the prepared body baseline during longer observation/inference waits. Loss of a sensor or policy/hand readiness stops further client publication.

Initialization requires a prepared stable Hold and measured agreement with the reference episode. No initialization targets or motor-enable commands are sent by this client. Shutdown disables/unbinds its policy session; a failed acknowledgement is reported, and controller shutdown is not inferred from it.

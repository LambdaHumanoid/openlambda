# Real-robot data collection

Collect demonstrations with a **29-DOF Unitree G1, SONIC, BrainCo Revo2 hands and a wired GoPro**. Video and robot telemetry are saved on G1 during teleoperation, then copied to a workstation for alignment and export.

[中文教程](docs/tutorial.zh-CN.md) · [Data format and source interfaces](docs/data-format.md)

For G1 teleoperation, follow the SONIC documentation:

- [PICO setup](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md) — install the environment, connect and calibrate the headset.
- [Whole-body teleoperation](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vr_wholebody_teleop.md) — run teleoperation in simulation and on the robot.

Once teleoperation is working, follow the steps below to add recording. This guide uses Revo2 hands; see the repository's robot setup for hand and camera dependencies.

## Installation

Start with the repository's [robot setup](../../README.md#robot-runtime-environment). Run the commands below from the OpenLambda repository root on each machine.

**On G1**, use the Python 3.8 / ROS 2 Foxy environment prepared by the robot installer:

```bash
source /opt/ros/foxy/setup.bash
source .venv-robot/ros_ws/install/setup.bash
source .venv-robot/bin/activate
python -m pip install -e data_collection/robot
```

**On the workstation**, create a separate Python 3.10+ environment for video decoding and export. On Ubuntu, install the matching `python3-venv` package if needed.

```bash
python3 -m venv .venv-robot-data
source .venv-robot-data/bin/activate
python -m pip install -e 'data_collection/robot[export]'
```

## Record a demonstration

Open three terminals **on G1**, with the robot environment activated in each.

### 1. Listen to the hands

```bash
lambda-robot-hand-telemetry --port 18560
```

This listens to the STARK feedback and command topics. Both hands' commands need to update regularly during teleoperation. Use this observer even if you already run the public robot bridge: its older feedback messages lack the timestamps and command records needed here.

### 2. Connect the camera

Replace `CAMERA_SERIAL` with your GoPro serial. Make sure another process isn't already using the camera or UDP port 8554.

```bash
eh2-gopro-relay --serial CAMERA_SERIAL \
  --forward-host 127.0.0.1 --forward-port 18554
```

Keep the destination on G1. Saving video locally avoids sending it over Wi-Fi during teleoperation.

### 3. Calibrate clocks and record

G1 and the SONIC workstation have separate clocks. Before each episode, measure their offset over SSH. Replace `USER@SONIC_HOST` with your workstation's SSH destination; key-based login must already work from G1.

```bash
lambda-collect-robot calibrate --host USER@SONIC_HOST --output clock-001.json
```

Use a new clock-map filename for each episode. Maps expire after five minutes and are invalid after either machine reboots. If the estimated uncertainty exceeds 10 ms, check the LAN connection and try again. Each recording is limited to 120 seconds to reduce drift since calibration; this remains software synchronization.

With teleoperation ready, start a 60-second episode:

```bash
lambda-collect-robot record \
  --sonic-endpoint tcp://SONIC_HOST:5557 \
  --clock-map clock-001.json \
  --task 'Pick up the object and place it in the basket.' \
  --duration 60 --output ~/robot-data/raw/episode-001
```

Use your SONIC **debug** publisher's address and port; `5557` is an example. Its required fields are listed in the [source interface](docs/data-format.md#sonic-debug-zeromq-sub).

Wait for `Recording` before beginning the demonstration. The timer starts once all sources are ready. Press Ctrl-C to finish early, and use a new output directory for the next episode. The short startup lead-in is retained in the raw recording and checked during alignment.

**Stopping the recorder does not stop the robot.** Keep the normal robot controls and physical emergency stop available throughout collection.

If SONIC runs on G1 itself, use `--sonic-local` with its loopback endpoint in place of `--clock-map`. This only applies when both processes share the same OS clock; forwarding a remote port over SSH still requires calibration.

## Copy and export

After teleoperation has ended, check the recording on G1:

```bash
lambda-collect-robot verify ~/robot-data/raw/episode-001
```

Then run the following **on the workstation**, replacing `G1` with your SSH alias:

```bash
mkdir -p data/robot/raw
scp -r G1:robot-data/raw/episode-001 data/robot/raw/
source .venv-robot-data/bin/activate
lambda-collect-robot verify data/robot/raw/episode-001
lambda-collect-robot export data/robot/raw/episode-001 \
  --output data/robot/aligned/episode-001
```

The exported episode contains:

```text
episode-001/
├── data.parquet   # robot state, actions, task and timestamps
├── ego_view.mp4   # aligned RGB video, 640×480
└── episode.json  # field layout, segment ranges and recording metadata
```

State has 44 values: body joints, measured hand positions and projected gravity. Action has 76 values: the SONIC encoder token and both hands' published STARK targets. Hand targets describe commands sent to the driver; feedback describes measured positions.

Before using an episode for training, watch the video and check its timestamps and segment ranges. A 30 Hz camera will repeat some images against 50 Hz robot samples. Gaps remain visible in the timestamps and start a new `segment_index`; training windows should stay within one segment.

The output is a single-episode Parquet/MP4 format. To use it with LeRobot or λ₀ training, add an adapter for that loader's video, task and episode metadata. See the [data format](docs/data-format.md) for the full layout.

## Troubleshooting

If a source goes missing or a packet is invalid, the raw directory is kept with `status: incomplete`. Check the reported error and record a new episode once the source is working. `verify` checks file integrity; export also checks alignment and physical fields. The [Chinese tutorial](docs/tutorial.zh-CN.md#常见问题) has a troubleshooting table.

The standalone collector has not yet been tested in a new real-robot session. Check a short recording when first connecting your setup.

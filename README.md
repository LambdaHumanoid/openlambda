<a href="https://lambdahumanoid.github.io/">
  <img src="assets/lambda-teaser.png" width="1280" alt="Lambda teaser: HumanVerse-500, three-stage training, and humanoid loco-manipulation">
</a>

<h1 align="center">OpenLambda</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-λ₀-b31b1b)](https://lambdahumanoid.github.io/paper.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-6b46c1)](https://lambdahumanoid.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Homepage](https://img.shields.io/badge/Homepage-Jin%20Chen-2563eb)](https://jinchen0056.github.io/)
[![Email](https://img.shields.io/badge/Contact-Email-555555)](mailto:cj65000816081@gmail.com)

</div>

<p align="center">
  <strong>From human experience to humanoid foundation models.</strong>
</p>

**OpenLambda** brings together data collection, human-to-robot alignment, foundation-model training, and deployment for humanoid loco-manipulation. It is the code repository for [λ₀](https://lambdahumanoid.github.io/paper.pdf), which learns coordinated locomotion, posture, and manipulation from egocentric human experience.

The project connects **HumanVerse-500**, a 500-hour whole-body human dataset, with a three-stage learning pipeline: egocentric hand-motion pretraining, robot-aligned whole-body human mid-training, and robot post-training. λ₀ combines a **Qwen3.5-2B** vision–language backbone with a generative action expert; on **Unitree G1**, its outputs drive the **SONIC** whole-body controller and **BrainCo Revo2** hands.

The current release includes human capture, synchronization, action alignment, simulation replay, and hardware/environment setup. Dataset access, λ₀ training, model checkpoints, and deployment will be added as they become available.

---

<p></p>

<a id="todo"></a>

### Workflow & Release Checklist

> **Release in progress:** Checked items are available in this repository. The remaining components will be released progressively—stay tuned!

1. **Hardware Setup**
   - [x] [Robot Hardware](#robot-data-collection-hardware)
   - [x] [Human Hardware](#human-data-collection-hardware)
2. **Environment Setup**
   - [x] [Robot Control & Data Collection](#robot-runtime-environment)
   - [x] [Human Data Collection](#human-collection-environment)
   - [x] [OpenPI Baseline Environment](#policy-environment)
3. **Data Collection**
   - [x] [Human Collection](#human-data-collection)
   - [ ] [Robot Collection](#robot-data-collection)
4. **Data Alignment**
   - [x] **[Action Alignment](#action-alignment)** ⭐
   - [ ] [Vision Alignment](#vision-alignment)
5. **[HumanVerse-500](#humanverse-500)**
   - [ ] Dataset access and preparation
6. **[λ₀ Foundation Model](#model-training)**
   - [ ] Stage I · Egocentric hand-motion pretraining
   - [ ] Stage II · Whole-body human mid-training
   - [ ] Stage III · Robot post-training
   - [ ] Model checkpoints
7. **[Evaluation & Deployment](#deployment)**
   - [ ] Simulation evaluation
   - [ ] Real-robot deployment

---

<p></p>

<a id="hardware-setup"></a>

## 🛠️ Hardware Setup

<table width="100%">
<tr>
<td width="50%" valign="top">
<a id="human-data-collection-hardware"></a>
<h3>Human Hardware</h3>
<img src="assets/hardware-human.gif" width="100%" alt="Animated human hardware tour: PICO headset, five body trackers and neck-mounted GoPro">
<details>
<summary>🔴 <strong>PICO 4 Ultra</strong></summary>

- **Tracking:** [PICO 4 Ultra](https://www.picoxr.com/global/products/pico4-ultra) + trackers<br>for whole-body and hand tracking
- **Software:** XRobo&#8203;Toolkit<br>[Setup&nbsp;guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#step-1-install-xrobotoolkit) · [Python&nbsp;SDK](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)
- **Connec&#8203;tion:** same LAN as the PC<br>enable Head, Hand and Send

</details>
<br>
<details>
<summary>🔵 <strong>5× PICO Trackers</strong></summary>

- **Placement:** five trackers<br>both wrists, waist and both ankles
- **Calibration:** PICO Motion Tracker<br>pair and calibrate in the app ([guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#motion-tracker-calibration))
- **Mode:** Full Body in XRobo&#8203;Toolkit<br>enable Hand tracking ([setup&nbsp;guide](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md#step-4-connect-the-pico-to-your-workstation))

</details>
<br>
<details>
<summary>🟠 <strong>GoPro</strong></summary>

- **Mount:** neck-mounted bracket<br>Example: [JD listing](https://item.jd.com/product/5qVT4d95gaChewMJeYVDKQ.html)
- **Alignment:** match height, pitch and framing to the robot camera
- **Connec&#8203;tion:** Bluetooth pairing and WiFi setup;<br>WiFi / COHN recording and download ([API](https://github.com/gopro/OpenGoPro))

</details>
</td>
<td width="50%" valign="top">
<a id="robot-data-collection-hardware"></a>
<h3>Robot Hardware</h3>
<img src="assets/hardware-robot.gif" width="100%" alt="Animated robot hardware tour: Unitree G1, BrainCo Revo2 hands and head-mounted GoPro">
<details>
<summary>🟢 <strong>Unitree G1</strong></summary>

- **Platform:** [Unitree G1](https://www.unitree.com/g1/), 29 body DOF<br>excluding hands ([joint&nbsp;layout](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/resources/robots/g1_description/README.md))
- **Control:** [SONIC whole-body controller](https://github.com/NVlabs/GR00T-WholeBodyControl)<br>runs on an external Linux PC
- **Connec&#8203;tion:** wired or WiFi to the PC<br>via the ZMQ bridge

</details>
<br>
<details>
<summary>🔵 <strong>BrainCo Revo2</strong></summary>

- **Hand:** 6 active DOF per hand<br>11 including passive joints ([specs](https://brainco.tech/docs/revolimb-hand/en/revo2/parameters.html))
- **Software:** [official&nbsp;SDK](https://github.com/BrainCoTech/brainco-hand-sdk)<br>hand control and state feedback
- **Mount:** custom G1–Revo2 adapters<br>STL: [part 1](hardware/Revo2-G1-1.stl) · [part 2](hardware/Revo2-G1-2.stl)

</details>
<br>
<details>
<summary>🟠 <strong>GoPro</strong></summary>

- **Mount:** head-mounted bracket<br>STL: [part 1](hardware/G1_GoPro_1.stl) · [part 2](hardware/G1_GoPro_2.stl)
- **Alignment:** match height, pitch and framing to the human camera
- **Connec&#8203;tion:** USB control and video to robot;<br>wired / WiFi forwarding to host ([API](https://github.com/gopro/OpenGoPro))

</details>

</td>
</tr>
</table>

<p></p>

<a id="environment-setup"></a>

## 💻 Environment Setup

The available tools use separate environments for robot control and action alignment, human data collection, and the OpenPI policy baseline. The λ₀ training environment will accompany the training release.

<a id="prerequisites"></a>

<a id="installation"></a>

<a id="robot-runtime-environment"></a>

<details>
<summary>🤖 <strong>Robot Control &amp; Data Collection</strong></summary>

### Host

Ubuntu 22.04 · NVIDIA driver **580+**.

**1. Install dependencies.**

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl build-essential cmake git git-lfs libgl1 libegl1 libegl-mesa0 libglib2.0-0 ffmpeg libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev nlohmann-json3-dev libssl-dev zlib1g-dev
curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh
source "$HOME/.local/bin/env"
curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends cuda-minimal-build-12-4
```

**2. Install TensorRT and ONNX Runtime.** Download `TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz` from [NVIDIA](https://developer.nvidia.com/tensorrt/download) to the repository root, then run:

```bash
sudo tar -xzf TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz -C /opt
curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz -o /tmp/onnxruntime.tgz
sudo tar -xzf /tmp/onnxruntime.tgz -C /opt
```

**3. Install environment.** Repeat the exports when opening a new terminal.

```bash
export TensorRT_ROOT=/opt/TensorRT-10.13.3.9
export onnxruntime_ROOT=/opt/onnxruntime-linux-x64-1.16.3
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:$CUDAToolkit_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash setup/robot/install_host.sh
```

**4. Activate environment.**

```bash
source sonic/.venv/bin/activate
```

### Robot

ARM64 · Ubuntu 20.04 · ROS 2 Foxy · Python 3.8.

**1. Install environment.**

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv build-essential cmake git python3-colcon-common-extensions ros-foxy-rmw-cyclonedds-cpp ros-foxy-rosidl-generator-dds-idl libyaml-cpp-dev libssl-dev libgl1 libglib2.0-0
bash setup/robot/install_robot.sh
```

**2. Activate environment.**

```bash
source /opt/ros/foxy/setup.bash
source .venv-robot/ros_ws/install/setup.bash
source .venv-robot/bin/activate
```

</details>
<br>
<a id="human-collection-environment"></a>

<details>
<summary>👤 <strong>Human Data Collection</strong></summary>

<p></p>
<p>Ubuntu 22.04 · Python 3.10.</p>

**1. Install environment.**

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates python3.10-dev python3.10-venv build-essential cmake git ffmpeg bluez network-manager
bash setup/human/install.sh
```

**2. Activate environment.**

```bash
source .venv-human/bin/activate
```

</details>
<br>
<a id="policy-environment"></a>

<details>
<summary>🧠 <strong>OpenPI Baseline</strong></summary>

<p></p>
<p>Ubuntu 22.04 · NVIDIA GPU and driver.</p>

**1. Install dependencies.**

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl git build-essential libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh
source "$HOME/.local/bin/env"
```

**2. Install [OpenPI](https://github.com/Physical-Intelligence/openpi#installation).**

```bash
bash setup/policy/install.sh
```

**3. Activate environment.**

```bash
source openpi/.venv/bin/activate
```

</details>

<p></p>

<a id="data-collection"></a>

## 🎥 Data Collection

<a id="human-data-collection"></a>

### 👤 Human Data Collection

**1. Prepare devices.**

Connect the host and PICO headset to the same network. Start the XRoboToolkit PC service on the host and the XRoboToolkit app on the headset, then enable body and hand tracking. Turn on the GoPro, enable Bluetooth pairing and select Video mode; the collector connects it to the host's WiFi network in the next step.

**2. Start recording.**

```bash
eh2-collect-human --task-name "{task_name}" --task "{task_instruction}"
```

Open `http://127.0.0.1:8080`. Enter the host's WiFi credentials if needed, then click **Prepare collection → Start recording → Save episode / Discard episode**. Wait for saving before recording again. Episodes are saved in LeRobot format to `data/{task_name}/`; new recordings append to the dataset.

**3. Check and synchronize data.**

```bash
eh2-sync-human --root "data/{task_name}"
```

Checks the recordings, synchronizes motion and video at 50 Hz, then checks the output. Results are saved to `data/{task_name}_synced/`, with synchronization and data quality reports in `meta/`.

<a id="robot-data-collection"></a>

### 🤖 Robot Data Collection

Coming soon.

<p></p>

<a id="data-processing"></a>
<a id="data-alignment"></a>

## ⚡ Data Alignment

<a id="action-alignment"></a>

### 🦾 Action Alignment

The action-alignment pipeline converts human motion into executable whole-body motion tokens through reference correction and dynamics-aware refinement. We provide four [example trajectories](data_alignment/action_alignment/examples) from [EgoHumanoid-V2](https://opendrivelab.com/EgoHumanoid-V2/). The simulation replays below compare these trajectories before (left) and after (right) alignment.

<table>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/ea461aca-9635-46b3-b522-6a3471bf7dc6

</td>
<td width="50%">

https://github.com/user-attachments/assets/68c5a826-4416-4180-a525-5b584e38e250

</td>
</tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/906522e9-1088-4619-9d0f-6d2844623d29

</td>
<td width="50%">

https://github.com/user-attachments/assets/71c1de99-8f2e-4ee1-875c-fe4be36db618

</td>
</tr>
</table>

**1. Align motion.**

Place the input recording at `{task_name}/input.parquet`. Set `{endpoints}` to `hands` for hand alignment or `feet` for foot alignment.

```bash
eh2-align \
  --input "{task_name}/input.parquet" \
  --endpoints {endpoints} \
  --output "results/{task_name}"
```

Results are saved under `results/{task_name}/`: `data.parquet` contains the human SMPL motion and aligned tokens; `run.json` records per-round metrics and the selected round.

**2. Replay motion.**

**SMPL replay:** encode the human motion online to simulate before alignment.

```bash
eh2-replay \
  --input "{task_name}/data.parquet" \
  --mode smpl \
  --output "results/{task_name}/before"
```

**Token replay:** replay the aligned motion tokens to simulate after alignment.

```bash
eh2-replay \
  --input "{task_name}/data.parquet" \
  --mode token \
  --output "results/{task_name}/after"
```

Each command saves `replay.mp4` in its output directory. Results may vary slightly between runs.

<a id="vision-alignment"></a>

### 🎨 Vision Alignment

Coming soon.

<p></p>

<a id="model-training"></a>

## 🤖 λ₀ Foundation Model

λ₀ learns from human and robot experience in three stages:

| Stage | Supervision | Training objective |
| :--- | :--- | :--- |
| I · Pretraining | Egocentric hand–object datasets | Predict future hand-state tokens with the vision–language backbone. |
| II · Mid-training | Robot-aligned HumanVerse whole-body demonstrations | Introduce the action expert and learn whole-body actions from human experience. |
| III · Post-training | Robot demonstrations | Adapt the policy to downstream tasks and robot execution. |

Training code, data preparation recipes, configurations, and checkpoints are coming soon. The currently provided OpenPI setup is a baseline environment, not the λ₀ training implementation. See the [paper](https://lambdahumanoid.github.io/paper.pdf) for the architecture and training recipe.

<a id="humanverse-500"></a>

### HumanVerse-500

HumanVerse-500 pairs egocentric video with synchronized body and hand motion across open-world loco-manipulation tasks. Observation and action alignment turn these recordings into robot-compatible supervision for Stage II.

Dataset access and preparation instructions are coming soon. The four action-alignment examples above are available now; they are not the full dataset release.

<p></p>

<a id="deployment"></a>

## 🚀 Evaluation & Deployment

At deployment, λ₀ takes robot camera observations, language instructions, and proprioceptive state. It predicts SONIC motion tokens and Revo2 hand commands without a human motion stream.

Simulation evaluation and real-robot deployment instructions are coming soon. For the currently available motion-level simulation tools, see [Action Alignment](#action-alignment).

<p></p>

<a id="citation"></a>

## 📝 Citation

For λ₀, please use the citation provided on the [project page](https://lambdahumanoid.github.io/):

```bibtex
@misc{xu2026generalhumanoidlocomanipulationmodel,
  title={Towards a General Humanoid Loco-Manipulation Model via Egocentric Whole-Body Human Data Pretraining},
  author={Chongyang Xu and Zhao Wu and Jin Chen and Yiming Jiang and Jinhui Ye and Yuming Jiang and Shifeng Zhang and Ziliang Feng and Mu Xu and Yilun Chen and Li Lu and Steven C. H. Hoi},
  year={2026},
  eprint={2610.00438},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.00438}
}
```

If you use the EgoHumanoid-V2 collection and alignment components, please also cite:

```bibtex
@article{chen2026egohumanoidv2,
  title={{EgoHumanoid-V2}: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation},
  author={Chen, Jin and Jiang, Yiming and Xu, Chongyang and Shi, Modi and Peng, Shijia and Chen, Li and Li, Tianyu and Xu, Mu and Chen, Yilun and Hoi, Steven and Li, Hongyang},
  journal={arXiv preprint arXiv:2609.37181},
  year={2026}
}
```

For long-range human-to-humanoid alignment, please also cite EgoAlign:

```bibtex
@misc{jiang2026egoalignbridginghumanhumanoidgap,
  title={EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation},
  author={Yiming Jiang and Jin Chen and Chongyang Xu and Yilun Chen and Aimin Hao and Yisheng He},
  year={2026},
  eprint={2609.38046},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.38046}
}
```

---

<p></p>

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We sincerely thank the following projects and teams:

The initial collection and alignment components, hardware illustrations, and example videos are adapted from [EgoHumanoid-V2](https://opendrivelab.com/EgoHumanoid-V2/). Assets retain their owners' rights. Existing command-line tools retain their `eh2-` names.

| [![StarVLA](https://img.shields.io/badge/StarVLA-VLA%20Framework-d58b45)](https://github.com/starVLA/starVLA) | [![SONIC](https://img.shields.io/badge/SONIC-NVIDIA-76b900)](https://github.com/NVlabs/GR00T-WholeBodyControl) | [![XRoboToolkit](https://img.shields.io/badge/XRoboToolkit-PICO%20VR-397bbf)](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | [![Open GoPro](https://img.shields.io/badge/Open%20GoPro-GoPro-009fdf)](https://github.com/gopro/OpenGoPro) |
| :---: | :---: | :---: | :---: |
| VLA&nbsp;models | Whole-body&nbsp;control | PICO&nbsp;tracking | GoPro&nbsp;interface |

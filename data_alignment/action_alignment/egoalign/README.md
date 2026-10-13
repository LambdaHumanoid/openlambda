<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/figures/title_dark.svg">
    <img src="assets/figures/title_light.svg" width="700" height="128" alt="EgoAlign — Bridging the Human–Humanoid Gap for Long-Range Loco-Manipulation">
  </picture>
  <p>
    <a href="https://iridescentjiang.github.io/">Yiming Jiang</a><sup>1,4,*</sup>  · 
    <a href="https://scholar.google.com/citations?user=4FqHXOsAAAAJ">Jin Chen</a><sup>2,4</sup>  · 
    <a href="https://chongyang-99.github.io/">Chongyang Xu</a><sup>3,4</sup>  · 
    <a href="https://yilunchen.com/about/">Yilun Chen</a><sup>4,†</sup>  · 
    <a href="https://research.buaa.edu.cn/en/persons/aimin-hao/">Aimin Hao</a><sup>1,‡</sup>  · 
    <a href="https://hyshkust.github.io/">Yisheng He</a><sup>4,†,‡</sup>
  </p>
  <p>
    <sup>1</sup> Beihang University  · 
    <sup>2</sup> Shanghai Innovation Institute<br>
    <sup>3</sup> Sichuan University  · 
    <sup>4</sup> Alibaba Group
  </p>
  <p>
    <sup>*</sup> Work done during an internship at Alibaba Token Hub (ATH), Alibaba Group.<br>
    <sup>†</sup> Co-project leaders  ·  <sup>‡</sup> Co-corresponding authors.
  </p>
  <p>
    <a href="https://lambdahumanoid.github.io/EgoAlign/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=github" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2609.38046"><img src="https://img.shields.io/badge/PDF-arXiv-red?logo=arxiv" alt="Paper on arXiv"></a>
  </p>
  <p><a href="README.md">English</a> · <a href="README_zh.md">简体中文</a></p>
</div>

<p align="center">
  <strong>Aligning human demonstrations for zero-shot, long-range humanoid loco-manipulation.</strong>
</p>

**EgoAlign** enables **long-range humanoid loco-manipulation** from egocentric human demonstrations. Scale alignment and controller-in-the-loop refinement produce robot-compatible state–action supervision for VLA policies that deploy zero-shot on Unitree G1, without physical-robot task demonstrations.



## 🔄 Pipeline

![EgoAlign pipeline from the paper](assets/figures/pipeline.png)

Human demonstrations are adapted to the robot, refined through simulation, and
converted into paired state–action supervision for VLA fine-tuning.



## ⭐ Key Highlights

- **Human-only demonstrations** for long-range loco-manipulation.
- **Scale alignment and simulation refinement** across embodiments.
- **Causal replay** for paired robot states and motion tokens.
- **Zero-shot G1 deployment** through SONIC whole-body control.



## 🚀 Features & Roadmap

Turn compatible human demonstrations into **G1 robot-learning data**:

- [x] Bilateral palm alignment and SONIC–MuJoCo residual refinement.
- [x] Causal replay with controller history for paired states and motion tokens.
- [x] Object Relocation example, visualization, and endpoint evaluation.
- [x] Additional task comparison videos.

Choose `endpoint-align` for alignment and refinement, or `replay-only` for direct
causal replay. Both export **43D states** paired with **64D motion tokens** and
two 6D hand commands (**76D actions**) for downstream VLA training.
See the [processing guide](docs/PROCESSING.md) for commands.

We may release our VLA training recipes in the future. For now, refer to
[OpenPI](https://github.com/Physical-Intelligence/openpi) for VLA training.



## 🛠️ Installation

Use Python 3.10 and start with the [dependency guide](docs/DEPENDENCIES.md).
From the code checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,simulation,data]'

python -m egoalign endpoint-align --help
python -m egoalign replay-only --help
```

Next, configure the replay environment. Stage A/C require compatible LeRobot and Pinocchio installations;
CUDA PyTorch must match your environment. Configure the pinned public SONIC
runtime using the [runtime setup and patch guide](docs/public_runtime.md).
The same guide covers native execution and running the launcher in your own GPU container.

Acquire and configure SMPL, BrainCo robot assets, and controller weights using
the [external asset guide](docs/EXTERNAL_ASSETS.md). Fresh controller compilation
and end-to-end processing were verified with existing dependencies; see
[validation details](docs/VALIDATION.md).



## 📦 Long-Range Loco-Manipulation Example

[Object Relocation](examples/data/basket/) combines long-range locomotion with object manipulation. The video
compares the SMPL reference at physical scale (left), G1 before alignment (middle), and
G1 after alignment (right).

<table align="center" width="67%">
  <tr><td colspan="3" align="center">

https://github.com/user-attachments/assets/3129a65a-8715-4151-a03d-8384ddb4d9a1

</td></tr>
  <tr>
    <td align="center" width="33%" nowrap>Mean palm error ↓</td>
    <td align="center" width="33%" nowrap>Before: 15.13 cm</td>
    <td align="center" width="33%" nowrap>After: <strong>1.41 cm</strong></td>
  </tr>
</table>

After [runtime setup](docs/public_runtime.md), preview the processing commands:

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign.sample_pipeline basket \
  --raw-root examples/data/basket \
  --robot-xml "$EGOALIGN_RUNTIME_ROOT/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml" \
  --output-root /path/to/new/basket-result --dry-run
```

Review the commands, then follow the [example guide](examples/README.md) to run
the pipeline and export the replay video.



<!-- additional-examples:start -->

## 🎬 More Long-Range Loco-Manipulation Tasks

Left to right: SMPL at physical scale, G1 before alignment, and G1 after alignment (Round 2).

### Storage Box Placement

---

<table align="center" width="67%">
  <tr><td colspan="3" align="center">

https://github.com/user-attachments/assets/50d807a1-633c-4e5e-ac26-31af12d567a2

</td></tr>
  <tr>
    <td align="center" width="33%" nowrap>Mean palm error ↓</td>
    <td align="center" width="33%" nowrap>Before: 9.22 cm</td>
    <td align="center" width="33%" nowrap>After: <strong>1.47 cm</strong></td>
  </tr>
</table>

### Shelf Interaction

---

<table align="center" width="67%">
  <tr><td colspan="3" align="center">

https://github.com/user-attachments/assets/71dfa4c2-544c-4625-b2d2-8b13f361d30a

</td></tr>
  <tr>
    <td align="center" width="33%" nowrap>Mean palm error ↓</td>
    <td align="center" width="33%" nowrap>Before: 12.00 cm</td>
    <td align="center" width="33%" nowrap>After: <strong>2.15 cm</strong></td>
  </tr>
</table>

### Parcel Delivery

---

<table align="center" width="67%">
  <tr><td colspan="3" align="center">

https://github.com/user-attachments/assets/edc0d6e7-979d-4917-bff6-0a12ba669ecc

</td></tr>
  <tr>
    <td align="center" width="33%" nowrap>Mean palm error ↓</td>
    <td align="center" width="33%" nowrap>Before: 14.44 cm</td>
    <td align="center" width="33%" nowrap>After: <strong>3.42 cm</strong></td>
  </tr>
</table>

### Indoor Cleaning

---

<table align="center" width="67%">
  <tr><td colspan="3" align="center">

https://github.com/user-attachments/assets/198caafd-7a0f-48e2-b434-868fee009aa2

</td></tr>
  <tr>
    <td align="center" width="33%" nowrap>Mean palm error ↓</td>
    <td align="center" width="33%" nowrap>Before: 13.18 cm</td>
    <td align="center" width="33%" nowrap>After: <strong>3.63 cm</strong></td>
  </tr>
</table>

### Measurement Protocol

---

Videos preserve metric world motion. Human fingers visualize the same grasp commands as the after-alignment robot.

All results use a fixed **0.24 s delay compensation**.

| Item         | Definition                                                                    |
| ------------ | ----------------------------------------------------------------------------- |
| Error        | Mean Euclidean distance over both palms and all valid frames; lower is better |
| Time pairing | Target frame i vs. robot frame i+12 at 50 Hz; final 12 target frames omitted  |
| Coordinates  | Pelvis-centered XY; lower-foot-grounded Z                                     |
| Video timing | Original source clock, without delay compensation                             |

See the [measurement record](assets/metrics/media_manifest.json) for full precision and quality checks.

<!-- additional-examples:end -->

## 📁 Code Structure

```text
egoalign/             Command entry points and pipeline orchestration
  geometry/          Coordinate and rotation math
  robots/g1/         G1 model, joint metadata and BrainCo hand mapping
  datasets/          Data schemas, export and video writing
  adapters/sonic/    SONIC state, rollout and telemetry contracts
scripts/human_data/   Alignment, residual refinement, and data export
runtime/             Public-runtime patches and source manifests
configs/             Dependency versions and asset hashes
examples/            Object-relocation input and reproduction instructions
tools/               Runtime setup, replay adapter and validation tools
  run_sim_replay.sh  Simulation-replay adapter used by the pipelines
tests/               Automated tests
docs/                Processing, runtime, data-contract, and validation guides
assets/              Paper figure, task videos, and measurement records
```

The SONIC controller and MuJoCo replay runner are installed separately via the
[runtime guide](docs/public_runtime.md). The local adapter contains their data
interfaces; geometry, robot and dataset utilities live in their own modules.
The pipelines call `tools/run_sim_replay.sh` to forward replay requests to
`EGOALIGN_REPLAY_DRIVER`. Configure it via the runtime guide, then use the Python
commands above; you do not need to invoke this adapter directly.

## 📘 Documentation

The `main` branch contains the source code and technical documentation. The
[project website](https://lambdahumanoid.github.io/EgoAlign/) is maintained in
`docs/` on the separate `gh-pages` branch.

- [Processing workflow and commands](docs/PROCESSING.md)
- [Input/output data contract](docs/DATA_CONTRACT.md)
- [Public SONIC–MuJoCo runtime setup](docs/public_runtime.md)
- [External assets](docs/EXTERNAL_ASSETS.md) and [dependencies](docs/DEPENDENCIES.md)
- [Sample reproduction and verification](examples/README.md)
- [Validation results](docs/VALIDATION.md)
- [Third-party components and terms](docs/THIRD_PARTY.md)

For development checks:

```bash
python -m pytest -q
python tools/check_cli.py
python tools/audit_source.py
```

## 📚 Citation

If you use EgoAlign in your research, please cite our paper:

```bibtex
@misc{jiang2026egoalign,
  title = {{EgoAlign}: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation},
  author = {Yiming Jiang and Jin Chen and Chongyang Xu and
            Yilun Chen and Aimin Hao and Yisheng He},
  year = {2026},
  eprint = {2609.38046},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.38046}
}
```

## 🙏 Acknowledgments

We thank the teams behind these open-source projects:

| [![OpenPI](https://img.shields.io/badge/OpenPI-Physical%20Intelligence-d58b45)](https://github.com/Physical-Intelligence/openpi) | [![SONIC](https://img.shields.io/badge/SONIC-NVIDIA-76b900)](https://github.com/NVlabs/GR00T-WholeBodyControl) | [![XRoboToolkit](https://img.shields.io/badge/XRoboToolkit-PICO%20VR-397bbf)](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | [![Open GoPro](https://img.shields.io/badge/Open%20GoPro-GoPro-009fdf)](https://github.com/gopro/OpenGoPro) |
| :---: | :---: | :---: | :---: |
| VLA&nbsp;models | Whole-body&nbsp;control | PICO&nbsp;tracking | GoPro&nbsp;interface |

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).

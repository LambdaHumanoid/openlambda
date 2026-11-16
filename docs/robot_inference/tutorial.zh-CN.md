# 从一段示范到 G1 终端推理

这篇教程介绍如何在 G1 上运行 Lambda0：先用一段录制数据测试模型，再接入相机和机器人，最后运行策略。全程使用终端，不需要网页控制台。

目前已用真实 G1 权重和一帧录制数据跑通离线预测，Linux SONIC 构建和相机输入检查也已通过；[验收记录](README.md#verification-boundary)列出了所用资产和结果。真机闭环仍未验收，所用 G1 模型与示范也还不是公开下载包。第一次上机请安排熟悉 G1 的人员在场，确认机器人支撑、活动范围和物理急停。后文会标出开始控制机器人的步骤。

## 1. 先分清机器、进程和环境

需要一台带 NVIDIA GPU 的 Linux 工作站，以及 G1 的机载电脑：

| 机器 | 需要运行的程序 |
| --- | --- |
| G1 机载电脑 | ROS 2 Foxy、Revo2/STARK 驱动、手部反馈、GoPro 视频接入 |
| Linux 工作站 | SONIC、动作路由器、相机接收、Lambda0 模型服务和推理客户端 |

本文默认所有工作站程序运行在同一台机器上，G1 通过局域网连接。SONIC、动作路由器和客户端需要共享本机时钟，不能简单用 SSH 转发替代。模型服务可以单独放在另一台 GPU 主机上，通过 SSH 转发 HTTP 端口；配置中的模型路径需与服务报告的路径一致。模型和控制端口只用于本机或机器人局域网，不要暴露到公网。

正文命令默认从仓库根目录执行。每个新终端都要激活对应环境；示例中的 `G1`、相机序列号、真实文件路径和 embodiment ID 需要替换为自己设备的值。

### 工作站模型与客户端环境

```bash
PYTHON=python3.11 bash scripts/install.sh serve
source .venv/bin/activate
python -m pip install -r scripts/requirements-robot-inference.txt
```

这里使用 Python 3.11。`serve` 会安装模型服务所需的 HTTP 依赖；GPU 驱动和 CUDA 请先按[环境说明](../../README.md#robot-runtime-environment)配置好。

SONIC 使用自己的运行环境。G1 则保留 Python 3.8 / ROS 2 Foxy 环境，不要把它们合并到模型的 `.venv`。

## 2. 录一段样例并准备离线输入

准备一段采集器导出的 G1 示范。如果还没有，先按[真机采集教程](../../data_collection/robot/docs/tutorial.zh-CN.md)录制：启动手部反馈和 GoPro，标定时钟，再用已有遥操系统完成一段短示范。录制器只负责保存数据，遥操和机器人停机仍按采集流程操作。

没有机器人时，也可以从已有的兼容 episode 开始，完成离线预测部分。

在工作站按采集教程安装导出环境后：

```bash
source .venv-robot-data/bin/activate
lambda-collect-robot verify data/robot/raw/episode-001
lambda-collect-robot export data/robot/raw/episode-001 \
  --output data/robot/aligned/episode-001
```

导出后应得到：

```text
data/robot/aligned/episode-001/
  data.parquet
  ego_view.mp4
  episode.json
```

打开 `ego_view.mp4`，检查视角、任务和起始姿态。`verify` 检查原始记录，`export` 负责对齐数据。导出结果使用 `openlambda_robot_aligned_v1` 格式，可直接用于后面的离线预测。

### “导出”与 `lambda0 prepare` 是不同步骤

**只做推理可以跳过这一小节。** `lambda0 prepare` 用于训练前准备数据和统计，不接收刚导出的 episode 目录。训练数据需要先整理成 canonical 格式；见[训练格式说明](../../README.md#model-training)、[metadata 示例](../../configs/lambda0/posttrain/metadata.example.json)和[数据选择配置](../../configs/lambda0/posttrain/data.example.json)。

已有 canonical 数据集时，设置 metadata、数据和输出目录，并把数据选择配置改成自己的 dataset 名称：

```bash
source .venv/bin/activate
export METADATA_DIR=/absolute/path/to/canonical-metadata
export DATA_ROOT=/absolute/path/to/canonical-datasets
export BUNDLE_DIR=/absolute/path/to/new-data-bundle
lambda0 prepare --source-config configs/lambda0/posttrain/data.example.json \
  --data-root "$DATA_ROOT" --output "$BUNDLE_DIR"
```

成功后会打印 `Prepared ...; no training started.`，并生成 `manifest.json`、`train/`、`validation/` 和 `stats/`。输出目录必须是新目录；配置文件中的相对路径相对于配置文件解析。

推理已有 checkpoint 时，继续使用它训练时的统计，不要用这段新示范重新计算的统计替换。

<a id="3-核对真实模型资产"></a>

## 3. 准备模型文件

启动模型前，准备以下文件和参数：

- 能由 `Lambda0.from_pretrained` 加载的完整 checkpoint，及匹配的 processor/tokenizer 文件；分开的 processor 可用服务端 `--processor` 指定。
- checkpoint 配套的 `data_bundle/manifest.json`。当加载 `ema` 子目录时，服务从它的父目录找 `data_bundle`。
- 训练时使用的 normalization JSON，其 `dataset_name` 必须被 checkpoint 的 manifest 收录，且 action space 与本次 G1 一致。
- 数据集对应的 embodiment ID，以及 44D state / 76D action 的定义和手部单位。配置示例中的 `embodiment_id: 0` 需要按模型修改。

[公开 Stage II checkpoint](https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k)尚不是经过真机验证的任务部署包。本文需要与你的任务、机器人配置及归一化统计匹配的权重；SIMPLE/AMO 使用另一套动作空间，不能混用。

<a id="4-用一份-json-连接同一个模型服务"></a>

## 4. 配置并启动模型

```bash
source .venv/bin/activate
cp configs/robot_inference/lambda0_g1.json configs/robot_inference/local.json
```

用文本编辑器修改 `local.json`：

| 字段 | 填写内容 |
| --- | --- |
| `instruction` | 示范对应的任务文字；执行时与参考 episode 的 task 一致 |
| `policy.model_path` | 模型服务报告的 checkpoint 路径 |
| `policy.normalization_dataset` | normalization JSON 中的 `dataset_name` |
| `policy.embodiment_id` | 模型对应的 embodiment ID |
| `policy.action_space_id` / `horizon` / `inference_steps` | 分别为 `2`、`50`、与模型服务一致的推理步数（示例为 `10`） |
| `observation.*` | 身体、双手和相机端点，按后文的启动命令配置 |
| `controller.*` | SONIC 启动后填写 PID、二进制绝对路径和 G1 地址 |
| `schedule.execute_steps` | 每次预测发送多少步，取值 1–50 |

这套适配器使用单路 RGB、44 维机器人状态和 76 维动作。字段顺序、单位及图像处理见[接口说明](interfaces.md)，接入其他数据源时需要保持一致。

在一个独立的模型终端启动服务；替换真实路径与 `YOUR_EMBODIMENT_ID`：

```bash
source .venv/bin/activate
lambda0 serve \
  --model /absolute/path/to/checkpoint \
  --normalization /absolute/path/to/checkpoint-normalization.json \
  --default-embodiment-id YOUR_EMBODIMENT_ID \
  --default-action-space-id 2 \
  --inference-steps 10 \
  --host 127.0.0.1 --port 8010 \
  --output outputs/lambda0-server-001
```

保留默认的非 RTC 模式，使用 CUDA 运行模型。每次启动换一个新的输出目录，等待模型加载完成后再执行下一条检查命令。

在另一个工作站终端执行：

```bash
source .venv/bin/activate
python scripts/infer_robot.py --config configs/robot_inference/local.json check
```

看到 `Adapter contracts and model service match the configuration.` 就说明模型服务和 JSON 配置一致，可以开始离线预测。传感器和机器人稍后再检查。

## 5. 先离线预测：不连接机器人动作通道

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json predict \
  --episode data/robot/aligned/episode-001 --frame 0 \
  --output outputs/lambda0-offline-001
```

这里从文件读取观测，仍然需要运行模型服务，但不连接机器人动作通道。

完成后，输出目录中会有 `deployment.json`、`session.json`、`chunk-0000-ego.jpg`、`chunk-0000.npz` 和 `chunk-0000.json`。先看保存的图像是否正确，再检查 NPZ 中的预测：`action.motion` 为 `[50,64]`，左右手各为 `[50,6]`。文件内容和字段解释见[技术说明](technical.zh-CN.md)。

## 6. 实时预览与显式执行

下面接入相机、双手和 SONIC。全部安装完成后再标定相机时钟，避免构建期间标定文件过期。首次启动需要先初始化底层控制器，再进行实时预览。

### 6.1 接入真实相机和手部反馈

先在两台机器上安装相机依赖，分别执行：

```bash
# G1
source /opt/ros/foxy/setup.bash
source .venv-robot/ros_ws/install/setup.bash
source .venv-robot/bin/activate
python -m pip install -e 'data_collection/robot[camera]'
```

```bash
# 工作站
source .venv/bin/activate
python -m pip install -e 'data_collection/robot[camera]'
```

相机依赖包含 PyAV 和 Pillow。采集器和下面的 capture 不能同时占用 UDP 视频端口；开始前先停止录制。

在 G1 的不同终端保持以下三个前台进程运行，每个终端先激活机器人环境：

```bash
# G1 终端 A：只读反馈，不驱动手
lambda-robot-hand-telemetry --port 18560
```

```bash
# G1 终端 B：GoPro 视频转发
eh2-gopro-relay --serial CAMERA_SERIAL \
  --forward-host 127.0.0.1 --forward-port 18554
```

```bash
# G1 终端 C：解码视频并保留帧时间戳
python -m lambda_robot_data.live_camera capture \
  --camera-port 18554 --bind tcp://127.0.0.1:5558
```

收到可解码的关键帧后，capture 会打印 `Publishing real PTS-matched RGB ingress frames on ...`。它记录的是 G1 的 UDP 收包时间，不是相机曝光时间；不要换成旧的 `run_gopro_receiver.py`，两者的时间戳接口不同。

工作站单独终端建立 SSH 转发（`G1` 是已配置公钥登录的设备别名）：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 5558:127.0.0.1:5558 \
  -L 18560:127.0.0.1:18560 G1
```

工作站另一个终端标定并发布相机：

```bash
source .venv/bin/activate
python -m lambda_robot_data.live_camera calibrate \
  --g1 G1 --output camera-clock-001.json
python -m lambda_robot_data.live_camera publish \
  --clock-map camera-clock-001.json \
  --source tcp://127.0.0.1:5558 --bind tcp://127.0.0.1:5559
```

成功后会依次看到 `Saved host-minus-G1 mapping...` 和 `Publishing calibrated ego_view RGB on ...`。

时钟文件有效期为五分钟，允许的标定误差不超过 10 ms。过期或任一机器重启后，换一个文件名重新标定，再启动 publish。这里的标定方向与录制教程相反，不能复用录制时的 clock JSON。

此时，工作站的 `5559` 提供相机数据，`18560` 提供手部反馈，与默认 JSON 一致。按 Ctrl-C 可以关闭相机进程，GoPro relay 则需要在它自己的终端停止。

### 6.2 安装外部 SONIC 并核对手部单位

SONIC 使用固定版本 [b042411f](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0)。依赖和模型的安装见[主机环境说明](../../README.md#robot-runtime-environment)及[上游部署目录](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy)，各组件保留原有许可证。

已有这个版本的 `sonic/` 目录时，先检查，再应用补丁并构建：

```bash
python scripts/setup_robot_runtime.py sonic
python scripts/setup_robot_runtime.py sonic --apply --build --jobs 4
```

如果还没有源码，先运行主机安装说明中的 `data_collection/robot/setup/install_host.sh`，或手动下载：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git sonic
GIT_LFS_SKIP_SMUDGE=1 git -C sonic checkout \
  b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
```

手动 clone 后，还需安装 Git LFS 资产、CUDA/TensorRT/ONNX Runtime、Unitree SDK 和模型。上面的 `setup_robot_runtime.py` 只负责检查版本、应用补丁和构建，不下载这些依赖，也不启动控制器。源码准备成功会打印 `Public runtime source prepared. No controller has been started.`。

**接双手前先核对单位。** 这里的 12 维手部目标已经是 `[0,1]` 的 normalized STARK 值，不能再经过旧 bridge 的拇指乘 2 转换。下面使用独立的 `robot_hands.py`；同时关闭其他手部命令发布程序，避免两套程序抢控制权。

#### 启动只读监测，再明确允许接管

运行时继续使用 `local.json`。默认端口分别是 SONIC 回执 `5686`、手部命令 `18561`、客户端控制 `5683` 和策略动作 `5684`。

在 G1 上激活 ROS 环境，先以监测模式启动手部适配器：

```bash
# G1：在前台运行
python -m pip install 'numpy>=1.24,<3' 'msgpack>=1,<2' 'pyzmq>=25,<28'
python scripts/robot_hands.py serve --port 18561
```

G1 另一个已激活的终端查询或停止：

```bash
python scripts/robot_hands.py status --port 18561
python scripts/robot_hands.py stop --port 18561
```

检查正常后，停止监测实例。现场人员确认手部单位和周围环境后，重新启动并允许发布命令：

```bash
# G1：只有确认 ROS/STARK 单位和现场安全条件后才使用
python scripts/robot_hands.py serve --port 18561 --allow-actuation
```

工作站另外建立手部命令端口转发，保留前面的相机/反馈转发：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 18561:127.0.0.1:18561 G1
```

手部适配器需要运行中的 STARK 驱动，以及 `rclpy`、`sensor_msgs`、`ros2_stark_interfaces`。它会检查双手反馈和命令订阅者。

工作站也先用监测模式启动动作路由器，保持这个终端运行：

```bash
source .venv/bin/activate
python scripts/robot_runtime.py serve --config configs/robot_inference/local.json
```

另开工作站终端检查状态：

```bash
python scripts/robot_runtime.py status --config configs/robot_inference/local.json
python scripts/robot_runtime.py prepare --config configs/robot_inference/local.json
```

`prepare` 只检查状态，不会移动机器人。`initialization_ready: true` 表示可以请求初始化；执行策略还需要后面的稳定 Hold。

接下来检查 G1 的身体控制桥：

```bash
eh2-robot-bridge --check
```

成功后会打印 `Bridge dependencies and native CRC verified; no ROS node, network or hardware initialized.`。

**从下一条命令开始允许身体控制。** 先确认机器人已可靠支撑、活动范围内无人，现场人员可以操作物理急停，再在 G1 上启动控制桥：

```bash
# G1：仅身体桥；明确禁用两种手部桥，避免重复发布和拇指单位转换
eh2-robot-bridge --enable-control \
  --state-port 5560 --cmd-port 5561 \
  --no-dex3 --no-brainco --no-udp
```

已有控制桥时不要重复启动。5560/5561 是未鉴权的底层端口，只应在隔离的机器人局域网内使用。

在工作站另开终端启动 SONIC，将 `G1_IP` 替换为机器人地址：

```bash
(
  cd sonic/gear_sonic_deploy
  OPENLAMBDA_RUNTIME=1 ./target/release/g1_deploy_onnx_ref \
    lo policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --input-type zmq_manager --zmq-host 127.0.0.1 --zmq-port 5656 --zmq-topic pose \
    --output-type zmq --zmq-out-port 5657 --zmq-out-topic g1_debug \
    --zmq-robot-ip G1_IP --zmq-robot-state-port 5560 --zmq-robot-cmd-port 5561 \
    --hand-type none
)
```

启动后，把二进制绝对路径、SONIC 的 PID 和 `G1_IP` 填入 `local.json` 的 `controller`。命令中的 `lo` 用于本地 DDS 初始化，不用替换成机器人地址。

保留 `OPENLAMBDA_RUNTIME=1` 和 `--hand-type none`，不要添加 encoder、planner 或 `--disable-crc-check`。这里用 `zmq_manager` 同时接收初始化指令和 token；换成 `zmq` 会收不到初始化指令。上游 `deploy.sh` 会添加其他参数，本流程直接使用上面的命令。

SONIC 此时在等待初始化，还不会发布完整的 `g1_debug` 身体数据，所以暂时不能跑实时预览。确认前面的离线预测、相机和双手检查都通过后，停止监测路由器，再允许它接管：

```bash
python scripts/robot_runtime.py stop --config configs/robot_inference/local.json
python scripts/robot_runtime.py serve --config configs/robot_inference/local.json \
  --allow-actuation
```

初始化需要与 SONIC decoder 匹配的起始 token。固定版本自带 `LATENT_INITIAL_MOTION_TOKEN`，先将它导出为 JSON；这一步只写文件：

```bash
PYTHONPATH=sonic python -c \
  'import json; from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN; print(json.dumps(LATENT_INITIAL_MOTION_TOKEN.tolist()))' \
  > initial_token.json
```

这个 token 对应的 release decoder SHA-256 应为 `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`。可用下面的命令核对；其他模型文件的哈希见[主机安装脚本](../../data_collection/robot/setup/install_host.sh)。

```bash
sha256sum sonic/gear_sonic_deploy/policy/release/model_decoder.onnx
```

哈希不匹配时不要继续。确认身体桥、SONIC、双手适配器和路由器都已运行后，执行初始化。**这一步会让机器人开始动作。**

```bash
python scripts/robot_runtime.py initialize \
  --config configs/robot_inference/local.json --execute --token initial_token.json
python scripts/robot_runtime.py prepare --config configs/robot_inference/local.json
```

等待状态变为 `phase: hold`、`stable: true`。路由器会检查 CONTROL 回执、起始 token 和至少一秒的实测稳定性；超时或失败时先查看报错，不要重复发送初始化命令。

如果机器人已经通过其他流程进入外部 token CONTROL，跳过 initialize，改用下面的命令接管当前 token：

```bash
python scripts/robot_runtime.py hold --config configs/robot_inference/local.json --execute
python scripts/robot_runtime.py prepare --config configs/robot_inference/local.json
```

`hold --execute` 同样是控制操作，它把当前 token 设为保持基准。固定 token 不保证机器人完全静止，仍需现场观察。watchdog 触发或停止路由器时，身体回到这个基准，手部停止发布新命令，但不会关闭电机。

### 6.3 底层控制就绪后预览，再显式执行策略

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --chunks 5 --output outputs/lambda0-preview-001
```

成功后会打印 `Chunk 0: preview` 等信息，并保存五段预测，`executed_steps` 为 0。此时没有发送 VLA 动作，但底层 SONIC 已在控制机器人。

检查预览结果，确认机器人姿态与参考示范一致、双手正常、现场急停就绪后，先只执行一段：

```bash
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --execute --chunks 1 \
  --initial-episode data/robot/aligned/episode-001 \
  --output outputs/lambda0-execute-001
```

`--initial-episode` 用于比较示范第 0 帧和当前实测姿态，不会自动把机器人摆到该姿态。身体、双手、重力方向和任务文字都要匹配；不匹配时，先准备好姿态或选对示范，不要修改容差绕过检查。

成功发送会打印 `Chunk 0: sent`。`executed_steps` 记录发送步数，不是到位反馈。动作前缀按 50 Hz 发送，之后重新取观测、调用模型；整个闭环的频率还取决于推理和等待数据的时间。

<a id="7-三种停止不可混为一谈"></a>

## 7. 停止运行

正常结束时，先在推理客户端按 Ctrl-C。客户端会请求禁用策略并释放会话；若返回失败，先检查控制状态。

随后分别在工作站和 G1 关闭路由器与手部发布服务，使用启动它们的同一用户：

```bash
# 工作站
python scripts/robot_runtime.py stop --config configs/robot_inference/local.json
```

```bash
# G1
python scripts/robot_hands.py stop --port 18561
```

这些命令只停止策略发布，不会关闭 SONIC、手驱动或电机。保持机器人支撑，按现场的底层控制退出与关机流程处理，再关闭相机、SSH 和模型服务。遇到危险先使用物理急停，不要等待网络命令回复。

<a id="8-常见问题与预期边界"></a>

## 8. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 找不到 `lambda0` / 缺 FastAPI | 激活工作站的 `.venv`，确认已安装 `serve` extra |
| 缺少 manifest 或 normalization 不匹配 | 检查权重是否带有对应的训练 bundle 和统计文件 |
| `/health` 连接失败 | 查看模型加载日志，检查 8010 端口和 SSH 隧道 |
| `check` 配置不匹配 | 对照模型路径、dataset、embodiment、action space、推理步数和 RTC 设置 |
| 离线 episode 无法读取 | 使用导出的 aligned episode，不是 raw 目录或单独的 MP4 |
| 一直没有实时观测 | 检查相机、手部转发和时钟文件；身体反馈需要 SONIC 已进入 CONTROL |
| 相机时钟过期 | 换新文件名重新标定并启动 publish；尽量在完成安装后再标定 |
| `controller_ready: false` / `stable: false` | 初始化前检查 `controller_waiting` 和 `initialization_ready`；初始化后检查 SONIC 日志、补丁和启动参数 |
| `hand_command_ready: false` | 检查 `--allow-actuation`、双手反馈、驱动订阅者，以及是否有其他手部命令发布程序 |
| 起始姿态或 task 不匹配 | 准备好机器人姿态，确认选择了正确任务的参考示范 |
| 手部单位检查失败 | 核对模型与驱动的单位，尤其是拇指是否被重复缩放；确认前不要执行 |
| 发送超时、传感器失联或控制器重启 | 先按现场安全流程处理机器人，再排查；不要自动重试执行 |
| 输出目录已存在 | 换一个新的输出目录，保留上次记录 |

更详细的字段、时间戳和时效阈值见[接口说明](interfaces.md)与[技术说明](technical.zh-CN.md)。

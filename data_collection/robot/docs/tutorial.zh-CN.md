# G1 真机数采

这份教程带你录一段 G1 遥操示范，再把数据取回工作站，导出为 Parquet 和视频。使用的设备是 **29 自由度 G1、Revo2 双手和一台 USB GoPro**，身体控制由 SONIC 完成。

采集时，视频和遥测都先存到 G1 上，结束后再传回工作站处理。这样不用在遥操过程中持续回传视频。

G1 遥操作参考 SONIC 官方文档：

- [PICO 环境与设备配置](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/vr_teleop_setup.md)：安装遥操环境，连接并标定 PICO。
- [全身遥操作教程](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vr_wholebody_teleop.md)：运行仿真和真机遥操。

完成遥操配置后，再按下面的步骤接入数采。本文使用 Revo2 双手，手部和相机依赖见本仓库的机器人环境说明。

## 环境配置

先按首页的[机器人环境说明](../../../README.md#robot-runtime-environment)准备硬件依赖。下面的命令都从仓库根目录执行，注意区分 G1 和工作站。

**在 G1 上**，使用机器人安装脚本准备的 Python 3.8 / ROS 2 Foxy 环境：

```bash
source /opt/ros/foxy/setup.bash
source .venv-robot/ros_ws/install/setup.bash
source .venv-robot/bin/activate
python -m pip install -e data_collection/robot
lambda-collect-robot --help
```

如果你已经有自己的 ROS 工作空间，换成对应路径即可。确认这个 Python 环境能导入 `rclpy`、`sensor_msgs` 和 `ros2_stark_interfaces`。

**在工作站上**，为视频处理单独建一个 Python 3.10+ 环境。可使用 Python 3.12：

```bash
python3 -m venv .venv-robot-data
source .venv-robot-data/bin/activate
python -m pip install -e 'data_collection/robot[export]'
```

Ubuntu 如果提示 `ensurepip is not available`，需要先安装对应的 `python3-venv` 包，也可以用已有的 uv 创建环境。导出依赖装在工作站即可，不需要塞进 G1 的 Foxy 环境。

## 数据采集

在 **G1 上打开三个终端**，每个终端都先激活刚才的机器人环境。

### 手部数据

```bash
lambda-robot-hand-telemetry --port 18560
```

这个程序监听两手的反馈和目标命令：`/motor_status`、`/motor_status_r`、`/joint_commands_left`、`/joint_commands_right`。它只订阅消息，不控制双手。

即使你已经运行仓库里的 robot bridge，这一步也需要做：旧版 bridge 的反馈包没有数采需要的完整时间戳和目标记录。遥操时，两手的目标话题都要持续更新。

这里要区分两种数据：**反馈是手实际测到的位置，action 是发布给 STARK 的目标位置**。记录到目标不代表电机已经到位，后续训练时也不能把两者混用。

### 相机

把 `CAMERA_SERIAL` 换成自己的相机序列号。先确认没有其他程序占用这台相机或 UDP 8554：

```bash
eh2-gopro-relay --serial CAMERA_SERIAL \
  --forward-host 127.0.0.1 --forward-port 18554
```

保留这里的 `127.0.0.1`：视频要交给 G1 本机的录制器，才能和双手数据共用 G1 的时钟。

### 时钟同步与录制

SONIC 通常运行在工作站上，两台机器的时钟需要先对齐。确认 G1 能通过 SSH 密钥登录工作站，然后执行：

```bash
lambda-collect-robot calibrate \
  --host USER@SONIC_HOST --output clock-001.json
```

`USER@SONIC_HOST` 填工作站的 SSH 用户名和地址。每段录制前重新标定，文件名也换一个。标定超过 5 分钟、任一机器重启，或者估计误差超过 10 ms 时，程序会要求重新标定。单段最长 120 秒；这是软件时钟对齐，仍会受到网络延迟和时钟漂移影响。

准备好遥操后，录制一段 60 秒的示范：

```bash
lambda-collect-robot record \
  --sonic-endpoint tcp://SONIC_HOST:5557 \
  --clock-map clock-001.json \
  --task 'Pick up the object and place it in the basket.' \
  --duration 60 \
  --output ~/robot-data/raw/episode-001
```

`SONIC_HOST:5557` 要换成实际的 SONIC **debug 发布地址**。录制器需要其中的实测关节、根姿态、encoder token 和时间戳；字段列表见[数据协议](data-format.md#sonic-debug-zeromq-sub)。

看到 `Recording` 后再开始做动作，计时也是从所有数据源就绪后开始。任务文字写这一段示范要完成的事情，每段保持一个任务。按 Ctrl-C 可以提前保存结束；下一段换成 `episode-002`，已有目录不会被覆盖。

**停止录制不会停止机器人。** 采集全程都要保留正常的控制、支撑和物理急停措施。

如果 SONIC 就运行在 G1 本机，可以用 `--sonic-local` 替代 `--clock-map`，并使用本机 loopback 地址。通过 SSH 转发的远程 SONIC 仍需要时钟标定。

## 数据导出

录制结束后，先在 **G1** 检查文件：

```bash
lambda-collect-robot verify ~/robot-data/raw/episode-001
```

通过后，等遥操结束再从**工作站**取回整段数据。下面的 `G1` 是你配置的 SSH 别名：

```bash
mkdir -p data/robot/raw
scp -r G1:robot-data/raw/episode-001 data/robot/raw/
source .venv-robot-data/bin/activate
lambda-collect-robot verify data/robot/raw/episode-001
lambda-collect-robot export data/robot/raw/episode-001 \
  --output data/robot/aligned/episode-001
```

`verify` 检查文件有没有丢失或损坏，`export` 再做时间对齐和数据校验。如果录制失败，原始文件仍会保留，状态是 `incomplete`。根据报错处理好数据源后，换一个目录重新录制。

## 数据格式

导出目录里有三个文件：

```text
episode-001/
├── data.parquet   # 状态、动作、任务文字和时间戳
├── ego_view.mp4   # 对齐后的 RGB 视频，640×480
└── episode.json  # 字段布局、连续片段范围和录制信息
```

先看视频，确认视角、任务过程和录制范围符合预期。GoPro 通常是 30 Hz，机器人数据目标是 50 Hz，所以相邻几行用到同一张图像是正常的。

训练主要会用到这两组字段：

| 字段 | 内容 |
| --- | --- |
| `observation.state`，44 维 | 身体实测关节 29 + 左右手反馈各 6 + 投影重力 3 |
| `action`，76 维 | SONIC encoder token 64 + 左右手目标各 6 |

手部反馈从 STARK 的 `0..1000` 换算为 `0..1`，目标保留发送给 STARK 时的 `0..1` 单位。原始关节、根姿态和各数据源的时间戳也会单独保留，方便检查。

有两点在接训练前值得看一下：

- **时间是否连续。** `timestamp` 保留实际采样间隔；丢帧或间隔异常处会开始一个新的 `segment_index`。训练窗口应放在同一片段内。
- **输入格式是否匹配。** 这里导出的是单段 Parquet/MP4。接入 LeRobot 或 λ₀ 的训练 loader 时，还需要适配视频、任务和 episode 信息，并核对模型要求的 state 维度。

更详细的字段顺序、对齐阈值和时钟含义放在[数据格式说明](data-format.md)中。

## 常见问题

| 现象 | 先检查什么 |
| --- | --- |
| 一直等 `robot` 或 `config` | SONIC debug 地址、字段配置，以及是否定期发布 `robot_config` |
| 缺 `left/right` | ROS 环境是否 source 完整，STARK 反馈话题是否有消息 |
| 缺 `hand_action` | 两手目标话题是否持续更新，关节名称和顺序是否正确 |
| 缺 `camera` | GoPro 是否出流，relay 是否指向 G1 本机 18554；收到的视频还需要有有效 PTS |
| 时钟误差过大 | G1 到工作站的局域网连接是否稳定，是否有传输任务占带宽 |
| 录制中断 | 查看报错的数据源，原始文件会保留为 incomplete |
| 导出丢弃了很多帧 | 检查时钟标定、相机延迟和双手命令更新频率 |
| 导出有很多 segments | 检查 SONIC 是否按 50 Hz 采样，以及是否有丢帧 |

独立版尚未做新的真机验证，首次使用时请先检查一段短录制的结果。

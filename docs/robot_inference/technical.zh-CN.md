# 真机推理技术说明

## 1. 模块范围

`scripts/infer_robot.py` 负责连接观测数据源、模型服务和底层控制器，完成观测读取、模型请求、动作发送及运行记录。当前实现对应 G1、单路 ego 相机、Lambda0 和 SONIC/Revo2。

模型加载与归一化由仓库中的 `lambda0 serve` 提供。机器人启动、回零、姿态调整和下电由外部控制系统负责。真机执行依赖已有的签名 motion router、带时间戳的 RGB 发布服务及机器人端服务，这些运行时组件尚未包含在本目录中。官方 SONIC 和基础 robot bridge 不能直接替代上述接口。

当前支持串行推理，不包含 RTC 或动作预取。通用数据结构允许多个视角和不同状态字段；已提供的 G1 数据源和 Lambda0 适配器限定为单路 ego 输入。

## 2. 代码结构

真机推理客户端集中在 `scripts/infer_robot.py`，按以下顺序组织：

| 类或函数 | 职责 |
| --- | --- |
| `Observation`、`ActionSpec`、`ActionBatch` | 观测和动作数据结构及规格检查 |
| `load()`、`local_endpoint()` | 配置读取和公共参数检查 |
| `G1Frame`、`state_vector()`、`action_chunk()` | G1/Revo2 原始数据布局和数值检查 |
| `read_episode()` | 数采 episode 读取 |
| `LiveObservations` | 实时消息接收、缓存和时间对齐 |
| `G1Observations` | 将 G1 实测数据转换为命名观测字段 |
| `LambdaPolicy` | Lambda0 输入预处理、HTTP 请求和动作转换 |
| `encode_action()`、`decode_action()` | 动作消息签名与校验 |
| `ControllerProcess`、`RouterSession` | 控制进程检查、motion router 会话和发送调度 |
| `SonicController` | 起始姿态检查与动作执行 |
| `build()`、`live()`、`main()` | 组件创建、推理循环和命令行入口 |

配置位于 `configs/robot_inference/`，本文档及协议说明位于 `docs/robot_inference/`。模型服务复用 `src/openlambda/inference/server.py`。客户端直接作为脚本运行，不需要安装为独立软件包。

## 3. 数据接口

### 3.1 Observation

观测由 `Observation` 表示，图像和状态均按名称组织。

| 字段 | 类型与约束 | 含义 |
| --- | --- | --- |
| `images` | `dict[str, ndarray]`；每张图像为非空 `uint8[H,W,3]`，不超过 1600 万像素 | RGB 图像 |
| `state` | `dict[str, ndarray]`；每个字段为非空、有限值的 `float32[D]` | 实测状态 |
| `timestamps` | `dict[str, int]`；非负整数，单位 ns | 各数据源及对齐基准的时间 |
| `clock` | 字符串 | 时间戳所属时钟域 |
| `metadata` | 字典 | 来源、帧序号、对齐误差等附加信息 |

G1 实时观测的 `clock` 为 `host_monotonic`，`timestamps.anchor` 为相机映射到工作站时钟后的时间。身体和双手的对齐时间保存在同一时钟域中；相机原始 G1 时间保留在 metadata 中。录制数据使用 `episode` 时钟标识，时间位于 metadata，不生成实时主机时间戳。

当前 Lambda0 使用以下输入：

| 字段 | 维度 | 语义 |
| --- | --- | --- |
| `images.ego` | `H × W × 3` | ego RGB 图像 |
| `state.body_joints` | 29 | 身体实测关节位置，单位 rad |
| `state.left_hand` | 6 | 左手实测反馈，归一化到 `[0,1]` |
| `state.right_hand` | 6 | 右手实测反馈，归一化到 `[0,1]` |
| `state.projected_gravity` | 3 | 投影重力方向，单位向量 |

实时观测另含 `state.body_velocity[29]`，用于起始姿态检查，不进入 Lambda0 的 proprio 输入。

### 3.2 ActionSpec 与 ActionBatch

`ActionSpec` 描述动作表示方式、字段维度、单位、频率和参考方式。`ActionBatch.values` 按字段保存 `float32[T,D]` 数组，各字段必须具有相同且非零的步数，所有数值必须有限。

当前动作规格如下：

| 项目 | 值 |
| --- | --- |
| `representation` | `sonic_token_revo2_v1` |
| `fields` | `motion: 64`、`left_hand: 6`、`right_hand: 6` |
| `units` | `motion: latent`；双手为 `normalized_stark` |
| `frequency_hz` | 50 |
| `reference` | `absolute` |

模型的 `output_spec` 与控制器的 `input_spec` 必须完全一致。初始化阶段进行检查，不执行隐式单位转换或维度补齐。

### 3.3 Lambda0 数据转换

图像通过 bicubic 插值缩放到 `448 × 448`。状态按身体 29、左手 6、右手 6、投影重力 3 的顺序拼接为 44 维 proprio。

模型每次返回 `50 × 76` 动作，按 SONIC token 64、左手目标 6、右手目标 6 拆分。归一化和反归一化在模型服务端完成。客户端将手部目标裁剪到 `[0,1]`，并记录裁剪数量及原始范围；token 绝对值超过 `1.25` 或动作包含非有限值时拒绝该预测。

HTTP、ZMQ 和签名消息的字段定义见[运行时接口](interfaces.md)。

## 4. 配置参数

完整示例见 [`configs/lambda0_g1.json`](../../configs/robot_inference/lambda0_g1.json)。所有配置地址均要求使用本机回环地址；跨机器服务需通过本地转发接入。SONIC 数据使用工作站单调时钟，不能仅通过端口转发将远端 SONIC 数据视为本机时钟。

| 参数 | 含义及约束 |
| --- | --- |
| `instruction` | 非空任务文字；录制数据推理和执行参考 episode 要求任务文字一致 |
| `observation.type` | 当前为 `g1` |
| `observation.body_endpoint` | 本机 SONIC debug 流地址 |
| `observation.hand_endpoint` | 双手实测反馈地址，可转发自 G1 |
| `observation.anchor_view` | 相机对齐基准名称，示例为 `ego` |
| `observation.cameras` | 当前 G1 适配器仅允许一个相机，名称与 anchor 一致，`image_key` 为 `ego_view` |
| `policy.type` | 当前为 `lambda0` |
| `policy.url` | 模型服务 HTTP 地址 |
| `policy.model_path` | 与服务 `/health` 一致的 checkpoint 路径 |
| `policy.normalization_dataset` | 与服务 `/health` 一致的归一化数据集名称 |
| `policy.embodiment_id` | checkpoint 对应的非负整数 ID；示例 `0` 为占位值 |
| `policy.action_space_id` | 当前限定为 `2`，对应 `robot_sonic_revo2` |
| `policy.horizon` | 当前限定为 `50` |
| `policy.inference_steps` | 正整数，需与模型服务一致 |
| `policy.image_order` | 单个图像字段名称，默认 `["ego"]` |
| `policy.state_order` | 固定为身体、左手、右手、投影重力 |
| `controller.type` | 当前为 `sonic` |
| `controller.control_endpoint` | motion router 控制地址 |
| `controller.policy_endpoint` | motion router 动作接收地址 |
| `controller.pid` | 执行模式下使用的 SONIC 进程 PID |
| `controller.binary` | SONIC 可执行文件的绝对路径 |
| `controller.robot_ip` | 与 SONIC `--bridge-host` 一致的机器人地址 |
| `schedule.type` | 当前限定为 `serial` |
| `schedule.execute_steps` | 正整数，不超过预测步数；当前 Lambda0 对应 `1..50` |

服务检查核对模型路径、归一化数据集、动作空间、推理步数和 RTC 状态，不校验权重文件内容哈希。

## 5. 执行流程

### 5.1 运行模式

| 模式 | 输入 | 行为 |
| --- | --- | --- |
| `check` | 配置及模型服务 | 检查适配器兼容性和模型服务配置，不检查实时传感器或控制端连接 |
| `predict` | 数采 episode 的指定帧 | 调用模型并保存预测 |
| `run` | 实时观测 | 循环推理并保存预测，默认不发送动作 |
| `run --execute` | 实时观测及参考 episode | 完成控制器初始化检查后发送动作 |

### 5.2 观测对齐

G1 数据源以最新相机帧为基准，选择时间最近的身体和双手数据。双手的 G1 时间通过相机同帧的 G1/主机时间对转换为工作站时间。

| 检查项 | 限制 |
| --- | --- |
| 身体、双手与相机的时间差 | 不超过 100 ms |
| 数据源及接收时间的新鲜度 | 250 ms |
| 源时间戳超前容差 | 50 ms |
| 默认读取等待时间 | 3 s |

每段动作发送结束后，下一次观测中的相机、身体和双手时间均须晚于最后一次动作发布。该条件约束客户端数据时序，不表示执行器已经到达目标。

### 5.3 控制器初始化

执行模式读取参考 episode 的第 0 帧，检查任务文字、SONIC PID、可执行文件、进程启动标识、`--external-token-override` 和机器人地址。motion router 必须处于未启用 policy 的稳定 Hold。

起始姿态需连续满足 50 个不同身体样本的检查，默认等待上限为 10 s：

| 检查项 | 阈值 |
| --- | --- |
| 身体关节角误差，按周期角计算 | ≤ 0.20 rad |
| 双手反馈误差 | ≤ 0.15 |
| 投影重力各分量误差 | ≤ 0.10 |
| 身体关节速度绝对值 | ≤ 0.50 rad/s |

客户端不发送起始姿态目标。场景布置、机器人全局位置和朝向由现场部署保证。

### 5.4 推理与动作发送

每轮依次执行以下操作：

1. 读取满足时间条件的观测，检查模型所需字段。
2. 调用模型，检查返回动作规格与预测步数。
3. 保存完整输入和预测。
4. 执行模式下检查实时数据、控制进程及观测年龄；首次输出前复查起始姿态并绑定、启用 router 会话。
5. 按 50 Hz 发送前 `execute_steps` 步，记录发送数量及最后发布时刻。
6. 使用发布时刻作为下一轮观测的时间下界。

发送前要求推理所用相机基准时间距当前不超过 1 s；首次启用 router 后再次检查。发送过程中持续检查传感器和控制进程，并定期检查 router 状态。发送延迟超过 50 ms 时结束本次发送，不补发积压动作。

模型每轮预测 50 步，未执行部分直接丢弃。观测和推理等待时间叠加在动作发送时间之外，因此串行循环不保证连续无间隙的模型动作输出。

### 5.5 异常与退出

传感器失效、动作异常、控制器重启或发送超时会中断当前流程。执行模式退出时请求 `policy-disable` 和 `policy-unbind`；关闭未得到确认时报告错误。

底层 SONIC 可能继续保持最后目标，policy 会话退出不等于机器人静止或下电。机器人停止仍由外部控制系统处理。

## 6. 运行示例

客户端要求 Python 3.10+，以下命令在仓库根目录执行。模型环境见[仓库安装说明](../../README.md#lambda0-install)。

```bash
python -m pip install -r scripts/requirements-robot-inference.txt
cp configs/robot_inference/lambda0_g1.json configs/robot_inference/local.json
```

模型服务在模型环境中启动，路径及 embodiment ID 使用 checkpoint 对应值：

```bash
lambda0 serve \
  --model /absolute/path/to/checkpoint \
  --normalization /absolute/path/to/normalization.json \
  --default-embodiment-id YOUR_EMBODIMENT_ID \
  --default-action-space-id 2 \
  --inference-steps 10 \
  --host 127.0.0.1 --port 8010 \
  --output outputs/lambda0-server
```

客户端命令如下；每次运行使用新的输出目录：

```bash
# 配置与模型服务检查
python scripts/infer_robot.py --config configs/robot_inference/local.json check

# 实时推理，不发送动作
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --chunks 5 --output outputs/lambda0-preview

# 接入已完成姿态准备的控制系统
python scripts/infer_robot.py --config configs/robot_inference/local.json run \
  --execute --chunks 5 \
  --initial-episode data/robot/aligned/episode-001 \
  --output outputs/lambda0-robot

# 从录制数据读取一帧并推理
python scripts/infer_robot.py --config configs/robot_inference/local.json predict \
  --episode data/robot/aligned/episode-001 --frame 0 \
  --output outputs/lambda0-prediction
```

参考 episode 使用[真机数采](../../data_collection/robot/README.md)导出的 `openlambda_robot_aligned_v1` 格式。执行前需完成外部控制系统的姿态准备、控制权交接和现场急停准备。

## 7. 输出文件

| 文件 | 内容 |
| --- | --- |
| `deployment.json` | 本次运行配置 |
| `session.json` | 会话 ID、状态及异常信息 |
| `chunk-NNNN-<view>.jpg` | 对应视角的输入图像 |
| `chunk-NNNN.npz` | 命名状态字段和完整预测动作 |
| `chunk-NNNN.json` | 观测 metadata、时间戳、时钟域、动作规格和推理信息 |

NPZ 的状态键以 `state.` 开头，动作键以 `action.` 开头。例如 `state.body_joints` 为 `[29]`，`action.motion` 为 `[50,64]`，左右手动作为 `[50,6]`。

实时运行的 JSON 还包含 `executed_steps` 和 `last_publication_monotonic_ns`。前者是客户端发布数量，不是执行器完成数量。部分发送失败时仍记录已发布步数，最后发布时刻可能为 `null`。录制数据预测不包含这两个执行统计字段。

## 8. 扩展接口

新增实现可在脚本中定义相应的类，并注册到 `SOURCES`、`POLICIES` 或 `CONTROLLERS` 字典，通过配置 `type` 选择。

| 组件 | 主循环使用的接口 |
| --- | --- |
| 观测源 | `validate(policy)`、上下文进入/退出、`read(after_ns=...)`、`read_episode(path, frame)` |
| 模型 | `required_images`、`required_state`、`output_spec`、`check()`、`predict(observation, session, step)` |
| 控制器 | `input_spec`、`prepare(source, reference, deployment, session)`、`execute(batch, source, observation, steps)`、`close()`、`published_steps`、`last_publication_ns` |

`predict()` 返回 `(ActionBatch, info)`；`execute()` 返回下一轮观测所用的发布时间下界。观测源须实现相应时钟域的读取约束。当前 SONIC 实现另依赖观测源的 `assert_fresh()`、G1 命名状态及身体样本序号。

接入其他模型时，在模型适配器中定义输入字段、预处理和动作规格。接入其他低层控制器时，在控制器适配器中实现初始化、动作映射、调度及退出处理。多相机输入需要数据源提供时间对齐后的命名图像，并由支持多视角的模型适配器消费。

当前完成了模拟服务和合成观测检查，尚未完成该重构版本的真实 checkpoint 推理与真机验证。

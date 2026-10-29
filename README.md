# X-Trainer Pi05 · LeRobot 部署手册

本仓库用于在 Dobot X-Trainer 双臂平台上完成 PI0.5（Pi05）的数据转换、微调与真机部署。
模型和训练采用 LeRobot 的 PyTorch 实现；通信、相机与机械臂控制沿用
[X-Trainer-XVLA](https://github.com/GC-SHIRO/X-Trainer-XVLA) 的组织方式。

流程：硬件检查 → 遥操作采集 → LeRobot v3 数据集 → Pi05 微调 → 策略服务 → 真机执行。

## 1. 项目分工与数据约定

| 环节 | 入口 | 说明 |
| --- | --- | --- |
| 采集 | 独立的 `dobot_xtrainer` 仓库 | leader 遥操作与 raw episode 录制 |
| 环境安装 | `tools/install_xtrainer_env.sh` | Conda、PyTorch、训练及硬件依赖 |
| 基座下载 | `tools/download_pi05_weights_*.sh` | Hugging Face / ModelScope |
| 数据转换 | `scripts/xtrainer/convert_raw_to_lerobot.py` | raw 数据写入 LeRobot v3 |
| 训练 | `scripts/xtrainer/train_pi05.py` | 数据元信息检查、微调与续训 |
| 推理服务 | `scripts/xtrainer/serve_policy.py` | GPU 服务器加载微调检查点 |
| 真机客户端 | `scripts/xtrainer/run_real.py` | 控制机读取观测、请求并执行动作 |

状态与动作都是 14 维：

| 索引 | 内容 | 单位 |
| --- | --- | --- |
| 0–5 | 左臂关节 1–6 | 弧度 |
| 6 | 左夹爪 | 0–1 |
| 7–12 | 右臂关节 1–6 | 弧度 |
| 13 | 右夹爪 | 0–1 |

相机字段固定为 `observation.images.top`、`observation.images.left_wrist`、
`observation.images.right_wrist`。训练与部署必须保持同一相机视角、朝向、关节顺序与夹爪方向。
Pi05 内部把状态和动作补到 32 维，推理输出裁回实际的 14 维；配置中的
`max_state_dim`、`max_action_dim` 保持 32。

## 2. 环境安装

安装脚本面向 Ubuntu x86_64，推荐 Ubuntu 24.04，需预先安装 Conda。
GPU 模式使用 CUDA 12.8 的 PyTorch wheel，需可用的 NVIDIA 驱动。
脚本创建或复用 `xtrainer-pi05` 环境，并固定 Python 3.12、
PyTorch 2.8.0、TorchVision 0.23.0、TorchCodec 0.7.0。
TorchCodec 用于视频解码，配套版本参考其
[官方兼容表](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)。

在本仓库根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-pi05
python tools/check_environment.py --require-cuda
```

默认 Python 包使用国内镜像，PyTorch wheel 使用官方源。可切换为全官方源，
或先查看将执行的命令：

```bash
bash tools/install_xtrainer_env.sh --source official
bash tools/install_xtrainer_env.sh --dry-run --skip-system-packages
```

只做数据转换和模拟通信时可以安装 CPU 版本：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
conda activate xtrainer-pi05
```

自定义环境名用 `--env-name NAME`；系统依赖已安装时用 `--skip-system-packages`。
脚本会安装 FFmpeg、训练、Pi05、WebSocket、ModelScope、RealSense 和串口依赖，
结束时检查依赖与关键导入。安装本身不下载模型。

本手册后续命令均在已激活的 Conda 环境、仓库根目录执行。
该安装路线用 [tools/constraints-pi05.txt](tools/constraints-pi05.txt) 固定 Torch 配套版本；
`uv.lock` 属于另一条 uv 安装路线，直接运行 `uv sync` 会按锁文件重新解析环境，
不应与这里的 Conda/pip 路线混用。

## 3. 下载 Pi05 基座与 tokenizer

本项目使用 [LeRobot 官方 Pi05 基座](https://huggingface.co/lerobot/pi05_base)。
下载器获取模型仓库完整快照，另下载 PaliGemma tokenizer 文件。
tokenizer 负责将任务描述转换为模型的文字输入，无需再次下载 PaliGemma 的整套模型权重。

Hugging Face：

```bash
bash tools/download_pi05_weights_hf.sh
# 网络需要镜像时：
bash tools/download_pi05_weights_hf.sh --endpoint https://hf-mirror.com
```

ModelScope：

```bash
bash tools/download_pi05_weights_modelscope.sh
```

两个入口都写入 `models/pi05-base`。中断后可重跑，下载库会复用已有文件。
ModelScope 默认模型 ID 为 `lerobot/pi05_base`，tokenizer ID 为
`AI-ModelScope/paligemma-3b-pt-224`；可用 `--model-id`、
`--tokenizer-model-id` 指定其他兼容仓库。

需要授权的 Hugging Face 资源先完成模型页面授权并运行 `hf auth login`，
也可设置 `HF_TOKEN`。ModelScope 可使用 `MODELSCOPE_API_TOKEN`。
自定义目录或固定版本：

```bash
bash tools/download_pi05_weights_hf.sh \
  --output-dir /data/models/pi05-base \
  --revision main \
  --tokenizer-revision main
```

模型与 tokenizer 属于两个仓库，revision 分别指定。完成后下载器离线加载 tokenizer
校验，并修改本地 `config.json` 和 `policy_preprocessor.json` 中的 tokenizer 路径。
路径为绝对路径，移动整个模型目录后应重新运行下载器。
来源记录在 `xtrainer_download.json`，模型文件默认不纳入 Git。

默认模型目录存在时，训练启动脚本自动使用它；自定义目录需要
`--policy-path /data/models/pi05-base`。这是训练用基座，真机服务加载后续微调产物。

## 4. 硬件检查与数据采集

采集仍使用独立的 `dobot_xtrainer` 工程，其安装、主手标定与采集配置以该仓库 README 为准。
常用入口如下，均在该采集仓库执行：

```bash
python scripts/1_find_port.py
python scripts/2_get_offset.py
python scripts/5_camera_read.py
```

分别在两个终端启动 follower 服务与遥操作：

```bash
python experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
python experiments/run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img True
```

采集输出应为：

```text
collect_data/
└── <episode_id>/
    ├── observation/<frame_id>.pkl
    ├── topImg/<frame_id>.jpg
    ├── leftImg/<frame_id>.jpg
    └── rightImg/<frame_id>.jpg
```

每个 pkl 的 `joint_positions` 和 `control` 分别保存状态与目标动作。
图像与观测按同一帧号对应。转换器按固定 FPS 生成时间轴，因此
`--fps` 要与采集实际频率一致；仅处理自己采集或可信来源的 pickle 数据。

本仓库也提供硬件检查入口：

```bash
python scripts/xtrainer/check_real_hardware.py --help
```

添加 `--execute` 会连接并实际移动关节和夹爪；IP、串口、相机序列号按现场填写。

## 5. 转换为 LeRobot v3

```bash
python scripts/xtrainer/convert_raw_to_lerobot.py \
  --raw-root /data/xtrainer/collect_data \
  --output /data/xtrainer/dataset_v3 \
  --repo-id xtrainer/pi05 \
  --task "将桌面上的方块放入收纳盒" \
  --fps 30
```

输出目录必须是新目录。默认生成视频，追加 `--images` 可存为图片。
缺少同步图像、观测帧号不连续、状态或动作维度错误时会报错。
当前一次转换对应一个任务描述；不同任务分别转换。

| raw 数据 | LeRobot 字段 |
| --- | --- |
| `joint_positions` | `observation.state` |
| `control` | `action` |
| `topImg` / `leftImg` / `rightImg` | 三路 `observation.images.*` |
| `--task` | `task` |

该流程使用 LeRobot v3，不直接接收 XVLA 工程的 v2.1 数据。
XVLA 的 v2.1 相机翻转脚本不能直接用于本数据格式；当前转换器保留采集图像方向。

## 6. Pi05 微调

配置见 [configs/xtrainer/train_pi05.yaml](configs/xtrainer/train_pi05.yaml)。
默认全参数微调，开启 gradient checkpointing（通过重算部分中间结果节省显存）；
batch size 为 4，预测动作段长度为 50。显存需求应在目标 GPU 上实测，
可先从 batch size 1 开始。

启动脚本会检查 v3 元信息、14 维状态与动作、三路相机及分位数统计量。
这些统计量用于将不同关节数值转换到模型所需尺度，转换时由 LeRobot 计算。

先打印并检查命令：

```bash
python scripts/xtrainer/train_pi05.py --dataset-root /data/xtrainer/dataset_v3 --dry-run
```

先做少量步骤，确认数据、权重与训练链路可以工作：

```bash
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --batch-size 1 --steps 50 \
  --output-dir outputs/train/pi05_smoke
```

正式训练：

```bash
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --batch-size 4 --steps 30000 \
  --output-dir outputs/train/xtrainer_pi05
```

默认优先使用 `models/pi05-base`，否则从配置中的 Hub 基座加载。
使用本地模型目录时，脚本还会读取其 `tokenizer/`，并重新从数据集推导输入相机。
需要离线运行时，在下载完成后设置 `HF_HUB_OFFLINE=1`；若仍缺少文件，加载将报错。

当前未提供 Pi05 专用 LoRA 训练与合并脚本。LoRA 只训练少量新增参数，
其独立适配器不能直接交给本项目真机服务。

## 7. 冻结主干训练：只训动作专家

第 6 节默认是全参数微调。显存紧张或数据量还小时，可以先冻结视觉语言主干、
只训练动作专家，等数据补足再解冻。

Pi05 由两部分组成：视觉语言主干（看画面、读任务描述）和动作专家
（默认 `gemma_300m`，可切 `gemma_2b`）。冻结主干后，视觉编码器和语言部分
一起冻住，梯度只回传到动作专家和两端的投影层。

### 7.1 新建一份训练配置

复制默认配置，保留原文件不动：

```bash
cp configs/xtrainer/train_pi05.yaml configs/xtrainer/train_pi05_expert_only.yaml
```

在 `configs/xtrainer/train_pi05_expert_only.yaml` 的 `policy` 段里改两行：

```yaml
policy:
  type: pi05
  # ... 其余保持不变 ...
  train_expert_only: true      # 冻结整个视觉语言主干，只训动作专家与投影层
  optimizer_lr: 1.0e-4         # 可训练参数变少，学习率一般可以抬高
```

### 7.2 给训练脚本加一个 --config 参数（两处改动）

`scripts/xtrainer/train_pi05.py` 目前把配置路径写死在代码里，需要改两处才能
切换到新配置。其余逻辑不动。

第一处，在 `main()` 的参数列表里，`--repo-id` 那行之后加两行：

```python
    parser.add_argument("--repo-id", default="xtrainer/pi05")
+   parser.add_argument("--config",
+                       default=str(ROOT / "configs/xtrainer/train_pi05.yaml"))
```

第二处，在 `build_command()` 里把写死的路径换成参数（约第 37 行）：

```python
-                   f"--config_path={ROOT / 'configs/xtrainer/train_pi05.yaml'}",
+                   f"--config_path={args.config}",
```

改完后：不传 `--config` 时仍读默认配置，行为与改动前完全一致；传了就按给定
路径读取。

### 7.3 确认改动生效

```bash
python scripts/xtrainer/train_pi05.py --help          # 应能看到 --config
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --config configs/xtrainer/train_pi05_expert_only.yaml --dry-run
```

`--dry-run` 打印出来的命令里，`--config_path` 应指向
`configs/xtrainer/train_pi05_expert_only.yaml`。

### 7.4 训练

先跑少量步数确认数据、权重与链路可用：

```bash
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --config configs/xtrainer/train_pi05_expert_only.yaml \
  --batch-size 8 --steps 50 --output-dir outputs/train/pi05_expert_smoke
```

`loss` 稳定下降后再延长步数：

```bash
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --config configs/xtrainer/train_pi05_expert_only.yaml \
  --batch-size 8 --steps 3000 --output-dir outputs/train/pi05_expert
```

batch size 从 8 起试，显存不够就降到 4 或 2；冻结后通常比全参能吃更大的 batch。

### 7.5 三个注意事项

1. `use_policy_training_preset` 为 `true` 时会套用官方训练预设，配置里自定义的
   `optimizer_lr` 可能被覆盖。改学习率时把它设为 `false`，并显式给出
   `optimizer_lr`、`optimizer_weight_decay` 等参数。

2. 冻结训练产出的检查点里，主干与基座同源。部署时 `--checkpoint` 指向微调
   产物，但要保证其主干与 `models/pi05-base` 是同一份，否则会报加载不匹配。

3. 冻结不等于跳过数据校验。第 6 节开头的元信息检查（v3.0、14 维状态与动作、
   三路相机、q01/q99 统计量）仍会先执行，数据不合格训练起不来。

### 7.6 怎么确认主干真的冻住了

训练启动后日志会打印可训练参数数量。冻结主干时应比全参微调小一个量级左右；
若两者接近，说明开关没生效（最常见原因是改了配置却仍在读默认 yaml，用
7.3 的 `--dry-run` 核对即可）。


## 8. 断点续训与检查点

```bash
python scripts/xtrainer/train_pi05.py \
  --dataset-root /data/xtrainer/dataset_v3 \
  --resume outputs/train/xtrainer_pi05/checkpoints/last/pretrained_model
```

`--resume` 可指向保存的 `train_config.json` 或其所在 `pretrained_model` 目录。
续训还需要相邻检查点中的优化器等训练状态，保留整个检查点目录。

部署使用的 `pretrained_model/` 至少包含模型、配置、预处理与后处理流水线，
以及它们引用的归一化参数文件。迁移检查点时还需迁移 tokenizer，
并保证流水线中记录的本地 tokenizer 路径可访问。

## 9. 策略服务

在 GPU 机器运行：

```bash
python scripts/xtrainer/serve_policy.py \
  --checkpoint outputs/train/xtrainer_pi05/checkpoints/last/pretrained_model \
  --host 0.0.0.0 --port 8000 \
  --actions-per-chunk 50
```

默认值见 [configs/xtrainer/deploy.yaml](configs/xtrainer/deploy.yaml)；
不传 `--host` 时只监听本机。服务器使用训练时保存的归一化参数，
要求检查点具有三路约定相机和 14 维输出。
当前服务支持单帧、同步动作分块；启用历史记忆、RTC（实时衔接动作段）
或尚未合并的适配器会被拒绝。

健康检查与元信息：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/metadata
```

只检查网络通信时可启动模拟服务：

```bash
python scripts/xtrainer/serve_mock_policy.py --host 0.0.0.0 --port 8000
```

模拟策略重复输入的关节状态，不加载 Pi05 权重。
通信协议为 X-Trainer 的 WebSocket + MessagePack，只适用于可信局域网，无认证与 TLS。

## 10. 真机客户端

在连接机器人的控制机上使用已安装依赖的环境运行：

```bash
python scripts/xtrainer/run_real.py \
  --host <GPU服务器IP> --port 8000 \
  --task "将桌面上的方块放入收纳盒" \
  --left-robot-ip 192.168.5.1 \
  --right-robot-ip 192.168.5.2 \
  --left-gripper-port /dev/ttyUSB1 \
  --right-gripper-port /dev/ttyUSB0 \
  --camera-top-serial <顶部相机序列号> \
  --camera-left-wrist-serial <左腕相机序列号> \
  --camera-right-wrist-serial <右腕相机序列号> \
  --control-hz 30 --action-horizon 32 --max-steps 100 \
  --execute
```

`--execute` 会实际使能并移动机器人。任务描述、相机朝向和夹爪方向与训练保持一致。
服务器默认预测 50 步，客户端执行前 32 步后重新获取观测。
IP、串口与相机参数由客户端命令行设置，服务端配置不控制这些设备参数。

可选动作处理参数见 `python scripts/xtrainer/run_real.py --help`。
默认关节变化限制与动作平滑关闭，夹爪范围限制为 0–1；
可用 `--max-joint-delta`、`--max-gripper-delta` 等显式设置限制。
实际执行前清空机械臂工作区并确认急停可用。

## 11. 排查与验证范围

| 问题 | 检查 |
| --- | --- |
| CUDA 不可用 | `nvidia-smi`、环境是否激活、是否误安装 CPU wheel |
| TorchCodec 导入失败 | Torch 配套版本、FFmpeg；重新运行环境检查 |
| tokenizer 下载 401/403 | 模型授权与对应平台登录状态 |
| 离线训练仍请求网络 | 下载目录是否完整；训练命令是否选中本地基座 |
| 数据集被拒绝 | v3 元信息、14 维向量、三路相机和 `q01/q99` |
| 模型维度或相机不匹配 | 使用 X-Trainer 微调检查点并核对训练相机字段 |
| 训练显存不足 | 降低 batch size；保留 gradient checkpointing |
| 真机动作异常 | 任务描述、统计量、相机朝向及状态/动作顺序 |

运行无硬件回归测试：

```bash
python -m pytest tests/xtrainer -q
```

现有测试覆盖通信往返、模拟模型处理链、硬件协议及数据转换/训练启动接口。
环境安装脚本与下载工具提供预演模式；Linux 完整安装、整套权重下载、
真实 v3 写入、GPU 训练与真机表现仍需在目标环境验收。

## 12. 来源与许可证

- [LeRobot](https://github.com/huggingface/lerobot)：Pi05、数据集与训练框架；
  初始基线 `3991580a5673f5f0219a4e8835a5f973eeccf9a7`。
- [X-Trainer-XVLA](https://github.com/GC-SHIRO/X-Trainer-XVLA)：项目流程、环境工具结构、通信与真机驱动参考。
- [X-Trainer-Pi0.5-JAX](https://github.com/Dobot-Edu/X-Trainer-Pi0.5-JAX)：采集到训练部署的流程参考。

代码许可证见 [LICENSE](LICENSE)。预训练权重的使用条款以对应模型仓库为准。
更多工具参数见 [tools/README.md](tools/README.md)。

---
pipeline_tag: robotics
tags:
- robotics
- vision-language-action
- video-memory
- jax
- openpi
---

# SLA-MEM：给 SLA 加上 MEM 的短时视频记忆

这是 [MEM](https://arxiv.org/abs/2603.03596) 的短时视频记忆在 SLA 上的实现。SLA（Spatial-Language-Action）是 [OpenHelix-Team/Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) 里用 JAX 写的 pi0.5，这份代码照 LeRobot pi05 的 `use_visual_memory` 移植。加上它之后，每个相机除了当前帧，还能看到过去 5 秒的画面。

它不加任何参数，也不改 SLA 的代码：`install()` 在运行时给 SLA 的几个函数打补丁。

代码拆自 [franzi-sim](https://github.com/heliang-pu/franzi-sim) 项目，那里用它训练出了 `franzi_sla_mem1`。这里只有代码，没有权重。

## 相比原版 SLA 的提升

这里的"原版"指同样在 franzi 数据上微调、但不带记忆的 SLA（`franzi_sla_v1`）。SLA-MEM 是在原版的终点上带着记忆再训 10 000 步得到的（`franzi_sla_mem1`）。

两者用同一套评测：franzi-sim 的十个双臂工业场景，每个场景 20 个留出种子，每次查询执行 25 步，长场景的每一步由同一个调度模型给出。

| 指标 | 原版 SLA | SLA-MEM | 提升 |
|---|---|---|---|
| 十个场景的平均成功率 | 22.0% | **28.5%** | +6.5 个百分点（相对约 +30%） |
| 五个长场景的成功集数（共 100 集） | 13 | **22** | +9（相对约 +69%） |
| 五个短场景的成功集数（共 100 集） | 31 | **35** | +4 |
| 平均完成率（任务做完，不论有没有碰到别的东西） | 29.0% | **32.5%** | +3.5 个百分点 |
| `parcel_sorting` 分好的包裹（共 80 个） | 39 | **53** | +14 |
| `palletizing` 码好的纸箱（共 80 个） | 37 | **49** | +12 |
| `precision_insertion` 拔出插销（共 20 集） | 8 | **11** | +3 |

- **提升主要在长任务。** 长场景多成功的 9 集里，`parcel_sorting`（2 → 6）和 `palletizing`（1 → 5）各占 4 集。
- **长任务里每一步也做得更好。** 按"某一步给出期间做成的比例"算：
  - `parcel_sorting` 从 57%（33/58）升到 73%（46/63）；
  - `palletizing` 从 66%（31/47）升到 71%（41/58）。

  所以记忆帮到的是一步之内的动作。
- **代价：** 推理慢约 75%，训练慢约 80%。
  - 每次查询的中位数 0.23 s，原版 0.13 s。两次评测都是 8 个回合同时跑，GPU 条件相同。
  - 训练每步 7.6 s，原版 4.2 s。两者都是 FSDP 8、batch 256、16 张 A100。

这个提升还不能算定论，原因有两个：
- **统计上还不显著。** 按种子配对做精确 McNemar 检验，p = 0.07。同一个种子两次评测的结果常常不同，单个场景差一两集只是噪声。
- **多训的步数没有对照。** SLA-MEM 比原版多训了 10 000 步，但没有一个同样多训 10 000 步、只是不带记忆的对照。所以提升里有多少来自记忆、有多少来自多训，现在还分不开。

逐场景的数字见[逐场景结果](#逐场景结果)。

## 做法

- 每个相机看 6 帧，间隔 1 s（25 fps 下 25 个控制拍），最后一帧是当前帧。
- SigLIP 共 27 层，其中每 4 层（第 4、8、…、24 层）还在时间上做注意力，分两步合成一次，用的都是该层预训练的 q/k/v（MEM 附录 C 式 3）：
  - 先在同一个图块的各帧之间做因果注意力；
  - 再用混合过的值在图块之间做空间注意力。

  这几层的输入加正弦时间编码，当前帧的编码为 0。
- 最后一个时间层之后只留当前帧，所以语言模型看到的图像 token 和原来一样多（每个相机 256 个）。
- 不加参数：只给一帧时，它就是原来的 SigLIP。所以可以接在不带记忆的权重上继续训练。论文把这种做法叫 MEM-Posttrain-Only，效果比预训练时就带记忆差一些。
- 只有视觉记忆，没有本体感觉记忆，也没有语言记忆。
- 回合开始前的历史用第一帧补齐，不打掩码。训练时 LeRobot 把越界的查询夹到第一帧，推理时 `FrameHistory` 也这样补。
- 和原版 SigLIP 一样，各层用 `lax.scan` 逐层走：每层是同一个函数，用一个标志区分普通层和时间层。
  - 如果把 27 层展开来写，XLA 会把所有层的重算提前做掉，重算就省不了显存。
  - 在 CPU 上量 72 帧的 SigLIP 梯度所需的临时内存：原版 5.8 GiB；扫描写法在 72 个单帧时 6.2 GiB，在 12 段 × 6 帧时 7.4 GiB；展开写法只给单帧也要 58 GiB。

## 怎么接进 SLA

历史帧按通道打包在每个相机的图像里，形状是 `[h, w, 3 × 帧数]`，最旧的在前。这样 SLA 的数据管线不用改。`sla_mem.mem.install(frames, stride, cameras)` 只给需要知道打包的几处打补丁：

| 打补丁的函数 | 补丁的作用 |
|---|---|
| `data_loader._action_delta_timestamps` | 数据加载器向 `cameras` 要历史帧，按数据集自己的帧率换算时间 |
| `LeRobotDataset._get_query_timestamps` | 让 SLA 环境里的 datasets 4 能取视频的历史帧 |
| `image_tools.resize_with_pad` | 打包的图像逐帧缩放 |
| `model.preprocess_observation` | 训练时，一个相机的各帧共用一次增强抽样 |
| `_TorchAlignFeatureShard.__call__` | 3D 教师只看当前帧 |
| `Pi0.embed_prefix` | 打包的图像改走上面的时空 SigLIP |
| `data_loader._worker_init_fn` | SLA 另起的数据加载进程也装上这些补丁（单数据集且 `num_workers > 0` 时才会另起进程） |

你自己只需改一处：在机器人自己的输入变换里，每个相机的图像都用 `sla_mem.mem.as_image()` 处理。它把 LeRobot 给的 `[T, 3, h, w]` 浮点历史帧和服务端给的 `[T, h, w, 3]` 历史帧都打包成 `[h, w, 3T]`，单帧照原样输出。

限制：
- 3D 对齐要用 `use_camera_params=False`。用相机参数时，SLA 走另一个缩放函数，那个函数没有打补丁。
- 在 SLA 的环境里（LeRobot 0.1.0、datasets 4.0.0），LeRobot 自己的单条读取 `__getitem__` 一取动作序列就报错，带不带 MEM 都一样。SLA 的混合数据集用 `_FastLocalLeRobotDataset.__getitems__` 批量读取，不受影响；franzi 用的就是混合数据集。

测试过的环境：Spatial-Forcing 的 4a77812 提交，JAX 0.5.3，Flax 0.10.2，LeRobot 0.1.0，datasets 4.0.0。

## 训练

在 SLA 的环境里运行，工作目录用 SLA 的训练目录：3D 教师从这个目录下的 `checkpoints/dinov2/` 读权重。

```bash
PYTHONPATH=<sla-mem>:<Spatial-Forcing>/SLA python -m sla_mem.train \
    --memory-cameras observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist \
    --memory-frames 6 --memory-stride 25 \
    <SLA 的配置名> --exp-name <名字> [SLA 自己的参数]
```

`--memory-cameras` 是数据集里要取历史帧的相机特征名，其余参数原样交给 SLA 的 `scripts/train_align.py`。如果用自己的启动脚本，在构建数据加载器之前调用 `mem.install(6, 25, cameras)`，再调用 `train_align.main(config)` 即可。

`franzi_sla_mem1` 的训练设置如下：
- 起点：不带记忆的 SLA 微调权重 `franzi_sla_v1`，第 19 999 步。
- 10 000 步，batch 256，预热 500 步，学习率 2.5e-5，余弦降到 2.5e-6。其余设置和那次微调相同。
- 16 张 A100：每卡 16 个样本，FSDP 8，不用梯度累积。
- 每步 7.55 s（训练日志的中位数），分项如下：

  | 分项 | 不带记忆（同样 FSDP 8） | 带记忆 |
  |---|---|---|
  | 学生模型的训练步 | 2.20 s | 4.03 s |
  | 3D 教师 | 1.82 s | 3.26 s |
  | 读数据 | 0.14 s | 0.24 s |
  | 每步合计 | 4.17 s | 7.55 s |

  - 教师只看当前帧，却也多花了 1.4 s。教师的计时从提交算到取回结果，而 JAX 是异步执行的，学生那一步没算完的部分可能被记到了教师头上。具体原因还没查。
  - 每步合计的时间是可靠的。

## 推理

```python
from sla_mem import mem

mem.install(6, 25)  # 推理不需要 cameras；要在加载策略之前调用
policy = ...  # openpi 的 policy_config.create_trained_policy(...)

history = mem.FrameHistory(6, 25)  # 每个回合新建一个；tick 从 0 起，每拍对应一帧
# 从第 0 拍起至少每 25 拍给它看一次画面，查询的那一拍也要给：
history.see(tick, {"cam_high": image, "cam_left_wrist": left, "cam_right_wrist": right})
# 查询时，每个相机得到 [6, h, w, 3]，最旧的在前
images = history.stack(tick)
actions = policy.infer({"images": images, "state": state, "prompt": prompt})["actions"]
```

如果每次查询都落在 25 的倍数上（每次执行 25 或 50 步），每 25 拍看一次就够；否则要每拍都给它看。缺了要用的帧时，`stack()` 会报错。

在 franzi 上，每次查询的耗时包括输入变换（每个相机 6 帧逐帧缩放）和整次推理：
- 条件：RTX PRO 6000，旁边同时跑着 8 个 Isaac Sim 回合；
- 中位数：0.23 s，不带记忆时 0.13 s；
- 平均：0.24 s，不带记忆时 0.14 s。

每次执行 25 步时每秒查询一次，推理占去这一秒的 23%，原版占 13%。

## 逐场景结果

评测设置：
- 仿真器是 Isaac Sim，机器人是汇川 Franzi 轮式双臂机器人。它有四路相机，SLA 用其中三路：胸前的 D435 和两个手腕的 D405。
- 每个场景用 20 个留出种子（10000–10019），每次查询后执行 25 步。
- 用宽松口径计分：穿透超过 1 mm 或关节越限时只记下、不中止回合，但这样的回合不算成功。
- 长场景的每一步由同一个调度模型给出粗提示。调度模型是另训的，不带记忆。
- 原版是 `franzi_sla_v1` 的第 19 999 步，SLA-MEM 是 `franzi_sla_mem1` 的第 9 999 步。

前五个是短场景，后五个是长场景：

| 场景 | 成功集数：原版 → SLA-MEM | 只有原版成功的种子 | 只有 SLA-MEM 成功的种子 |
|---|---|---|---|
| `tote_transfer` | 13 → 13 | 4 | 4 |
| `tote_carry` | 8 → 10 | 1 | 3 |
| `tray_push` | 5 → 6 | 3 | 4 |
| `panel_operation` | 5 → 6 | 3 | 4 |
| `precision_insertion` | 0 → 0 | 0 | 0 |
| `lidded_tote` | 10 → 11 | 3 | 4 |
| `parcel_sorting` | 2 → 6 | 1 | 5 |
| `palletizing` | 1 → 5 | 1 | 5 |
| `shaped_kitting` | 0 → 0 | 0 | 0 |
| `cnc_tending` | 0 → 0 | 0 | 0 |
| **合计（200 集）** | **44 → 57（22.0% → 28.5%）** | **16** | **29** |

同一个种子两次评测的结果常常不同：RTX 每次渲染的画面有细微差别，闭环会把它放大。`tote_transfer` 两边都成功 13 集，其中只有 9 个种子相同。

各场景做到了哪一步，格子里是 20 个种子里达到该里程碑的个数。`palletizing`、`parcel_sorting` 和 `shaped_kitting` 每集有 4 个零件，数的是 80 个零件里做完的：

| 场景 | 里程碑 | 原版 SLA | SLA-MEM |
|---|---|---|---|
| `tote_transfer` | 零件夹出蓝箱 | 18 | 20 |
| `panel_operation` | 拨到 AUTO / 按下启动 | 20 / 6 | 20 / 6 |
| `tray_push` | 推过一半 / 推到挡块 | 19 / 5 | 19 / 6 |
| `tote_carry` | 抬起 / 到传送带上方 / 放上传送带 | 20 / 11 / 8 | 20 / 10 / 10 |
| `precision_insertion` | 夹起轴承 / 轴承压到位 / 拔出插销 | 20 / 1 / 8 | 20 / 2 / 11 |
| `cnc_tending` | 开门 / 取出成品 / 放上中转台 | 20 / 19 / 18 | 20 / 18 / 17 |
| `lidded_tote` | 开盖 / 取出六角坯 / 放上工位板 / 合盖 | 20 / 19 / 18 / 15 | 20 / 20 / 18 / 15 |
| `palletizing` | 码好的纸箱 | 37/80 | 49/80 |
| `parcel_sorting` | 分好的包裹 | 39/80 | 53/80 |
| `shaped_kitting` | 夹起 / 放进托盘的零件 | 57/80 / 16/80 | 61/80 / 19/80 |

长场景按步计算，即某一步给出期间做成的次数 / 给出的次数：

| 场景 | 原版 SLA | SLA-MEM |
|---|---|---|
| `lidded_tote` | 53/60 | 53/57 |
| `parcel_sorting` | 33/58 | 46/63 |
| `palletizing` | 31/47 | 41/58 |
| `shaped_kitting` | 16/59 | 19/63 |
| `cnc_tending` | 38/55 | 38/61 |

- `parcel_sorting` 进步最大：右手两步从 9/26 升到 18/27，左手第一步从 10/15 升到 15/17。
- `shaped_kitting` 和 `cnc_tending` 没有变化。
- 长任务剩下的失败多出在调度模型上，它不带记忆，有两个问题：
  - 一步已经做完，却迟迟不换下一步；
  - 一步还没做成，就提前换走，之后不再回头重试。

  例如 `parcel_sorting` 没成功的 14 集里，有 7 集停在已做完的一步。

## 验证

- 和训练 `franzi_sla_mem1` 时用的代码相比：
  - 逐字没改的：时空 SigLIP、`embed_prefix`、训练增强、逐帧缩放、3D 教师的补丁；
  - 改成参数的：相机名和帧率，原来写死成 franzi 的值，现在由 `install()` 传入；
  - 搬过来的：`FrameHistory` 和 `as_image()`，分别取自 franzi 的服务端和输入变换；
  - 新加的：数据加载进程里的补丁。
- `python -m sla_mem.mem check`（SLA 环境，CPU）：
  - 只给一帧时，输出和原版 SigLIP 完全相同，最大差 0.0；
  - 历史帧会改变当前帧的 token；
  - 梯度能传到每一帧。
- 数据管线（SLA 环境，用 franzi 的一个 25 fps 数据集）：
  - 取到的 6 帧和数据集里对应的帧逐像素相同，回合开头用第一帧补齐；
  - SLA 另起的加载进程里补丁生效，读出的历史帧和主进程相同；
  - 作为对照，不装补丁的加载进程一读历史帧就报错。
- `python -m pytest tests -q` 测打包、取历史帧、`FrameHistory` 和 `as_image()`，只需要 NumPy。

## 文件

- `sla_mem/mem.py`：时空 SigLIP、各个补丁、`as_image()` 和 `FrameHistory`。打包和取历史帧的函数只需要 NumPy，其余要在 SLA 的环境里运行。
- `sla_mem/train.py`：装上 MEM，再运行 SLA 自己的训练。
- `tests/test_mem.py`：不需要 SLA 环境的测试。

## 来源和许可

- 方法来自 MEM（arXiv 2603.03596），参照的实现是 LeRobot 的 `lerobot/policies/pi05/memory.py`。
- 改自 SLA 附带的 openpi（Apache-2.0）的部分：`embed_prefix` 改自 `Pi0.embed_prefix`；`preprocess` 沿用 `preprocess_observation` 的增强链。
- 改自 LeRobot（Apache-2.0）的部分：`query_timestamps` 改自 `LeRobotDataset._get_query_timestamps`。
- 这几个函数改了哪里，写在各自的文档字符串里。Apache-2.0 的全文见 `LICENSE-APACHE-2.0.txt`。
- 其余代码没有附许可证。

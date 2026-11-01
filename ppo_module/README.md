# ppo_module —— 栅格导航 PPO 强化学习模块

本模块是仓库 `NavigationwithOn-linePolicyUpdate` 的算法核心：在 **14 × 18 的内部栅格地图**上，用
**PPO（Stable-Baselines3）** 训练一个"任意起点 → 指定终点"的导航策略，并提供
**训练 → 推理接口 → 可视化评估 → 真机 UDP 部署** 的完整链路。

- 观测：8 维（自身坐标、终点坐标、上下左右可走性）
- 动作：4 个离散动作（0 上 / 1 下 / 2 左 / 3 右）
- 模型：`MlpPolicy`（多层感知机），训练后保存为 `models/ppo_grid_model.zip`
- 输出：`models/`（模型）、`tensorboard_logs/`（训练曲线）、`[x,y].mp4` 与两张策略图（评估）

---

## 1. 目录结构

```
ppo_module/
├── README.md                 # 本文件
├── ppo_train.py              # ① 训练入口：PPO 训练并保存模型
├── model_interface.py        # ② 推理接口：坐标 → 动作 ID（供上层框架/真机调用）
├── eval_with_popup.py        # ③ 评估入口：弹窗指定起终点，录制视频 + 画策略图
├── fixed-route.py            # ④ 部署入口：真机 UDP 控制 + 网格导航主循环
├── env_module/
│   └── GridEnv.py           # 环境定义（gymnasium 接口）+ 表格 Q-learning 基线
└── __pycache__/              # Python 自动生成的字节码缓存，可忽略/删除
```

模块外、但与本模块强相关的目录（均位于仓库根目录）：

| 路径 | 说明 |
| --- | --- |
| `models/ppo_grid_model.zip` | 训练好的 PPO 模型，推理/评估/部署脚本默认从这里加载 |
| `tensorboard_logs/PPO_1/` | 训练时由 `TensorBoard` 写入的曲线数据 |
| `zzz/eval_policy.py` | 早期单局录制脚本（随机起点，直接录一局 mp4），已基本被 `eval_with_popup.py` 取代 |
| `README.md`（根目录） | 仓库整体介绍 |

> 目录名 `zzz` 表示历史/临时脚本，不属于本模块运行主链路。

---

## 2. 各文件作用

### 2.1 `ppo_train.py` —— 训练入口

| 项目 | 内容 |
| --- | --- |
| 作用 | 创建环境 → 构建 PPO 模型 → 训练 → 保存模型 |
| 关键参数 | `MlpPolicy`、`learning_rate=5e-4`、`ent_coef=0.2`、`total_timesteps=100000` |
| 其余超参 | 全部使用 SB3 默认值（`n_steps=2048`、`batch_size=64`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`clip_range=0.2`） |
| 输出 | `models/ppo_grid_model.zip`；日志写入 `./tensorboard_logs/` |
| 说明 | 文件里保留了一段被注释掉的 `ppo_config.yaml` 读取逻辑，当前未启用，超参直接写在代码中 |

### 2.2 `model_interface.py` —— 推理接口（对外 API）

| 项目 | 内容 |
| --- | --- |
| 作用 | 把"当前所在的格子坐标"翻译成"下一步动作 ID"，供真机/上层调度直接调用 |
| 对外函数 | `get_action_from_position(x: int, y: int) -> int`，返回 `0/1/2/3` |
| 懒加载 | `_ensure_loaded()` 用模块级单例 `_MODEL / _ENV / _GOAL_POS` 保证模型只加载一次 |
| 内部逻辑 | 由传入坐标 + 终点 + 障碍物重新拼出 8 维观测 → `model.predict(obs, deterministic=True)` |
| 目标点 | 进程启动时执行一次 `env.reset()`，把**当时随机抽到的终点**固定为本次运行的终点（加载时会打印） |
| 自测 | `python ppo_module/model_interface.py` 会对若干测试点打印"坐标 → 动作" |

调用示例：

```python
from model_interface import get_action_from_position

action = get_action_from_position(6, 4)   # 0=上 1=下 2=左 3=右
```

### 2.3 `eval_with_popup.py` —— 可视化评估入口

| 项目 | 内容 |
| --- | --- |
| 作用 | 交互式指定起点/终点，让模型实际跑一局，并把"过程 + 策略"落盘 |
| 输入方式 | 优先 `tkinter` 弹窗（`simpledialog`），无 `tkinter` 时自动降级为命令行输入 |
| 输入校验 | 越界、落在障碍物、终点与起点重合都会被拒绝；坐标解析很宽松（`3,5` / `3 5` / `(3,5)` 均可） |
| 过程记录 | 直接调用 `env.render()`（Pygame 窗口），用 `pygame.surfarray` 抓帧，最后用 `imageio` 合成 mp4 |
| 策略可视化 | 遍历全图每个空格，用 `model.policy.get_distribution` 取 4 个动作的**概率**，画两类图 |
| 输出 | `[x,y].mp4`（以起点命名）、`[x,y]_decision_map.png`（最优动作 + 实际路径）、`[x,y]_prob_maps.png`（四动作概率热力图） |
| 注意 | `OPEN_RESULTS=True` 时会用 `os.startfile` 自动打开结果，该函数仅在 Windows 可用，其他平台只会打印"自动打开失败" |

### 2.4 `fixed-route.py` —— 真机部署入口

| 项目 | 内容 |
| --- | --- |
| 作用 | 真机执行层：交互式输入起点 → 每步向 `model_interface` 要动作 → 翻译成 UDP 速度指令 → 跟踪所在格子 |
| 通信 | UDP 发 JSON `{"vx":…, "vy":…, "yaw":…}` 到中继 `192.168.123.220:5091` |
| 坐标约定 | 机器狗初始朝向 = 网格 `+y`（"上"）；1 格 = `CELL_SIZE = 0.6 m` |
| 动作映射 | 0 上→前进、1 下→后退、2 左→左横移、3 右→右横移（每个动作走满 0.6 m） |
| 步长修复 | 原实现 `int(total/step)` 向下截断导致每格少走 0.114 m、17 格累计约 1.94 m 漂移；现改为 `ceil` 并把总距离均摊到每一步，保证 `步数 × 每步 == 总距离` |
| 安全兜底 | 撞墙/越界时取消本次移动并发送零速；结束/中断时 `finally` 中发送零速并关闭 socket |
| 终端可视化 | `draw_map()` 用 ASCII 打印地图（`#` 障碍、`G` 终点、`S/*` 路径、`@` 当前位置） |
| 注意 | 文件名带连字符，**只能直接运行，不能 `import`**；真机执行前会二次确认（输入 `y`） |

### 2.5 `env_module/GridEnv.py` —— 环境定义

核心类 `GridNavEnv(gym.Env)`，是训练/推理/评估共用的唯一环境来源。

**地图**

- 原始地图 16 × 20，先去掉最外一圈，得到可用区域 **14 × 18**（坐标 `x: 0~13`，`y: 0~17`）
- 障碍物写死在 `raw_obstacles` 中，构造时自动裁剪 + 平移 `(-1,-1)`，得到 98 个障碍格、154 个空格
- 起点/终点从空格池中随机抽取，**互不重合**，且终点保证从起点 **BFS 可达**（`reset()` 前已预计算可达集与候选池，因此 reset 很快）

**MDP 定义**

| 项目 | 内容 |
| --- | --- |
| 观测 `Box(8,)` | `[x, y, goal_x, goal_y, up, down, left, right]`，后 4 维是该方向是否可走（1 可走 / 0 不可走） |
| 动作 `Discrete(4)` | 0 上(+y)、1 下(-y)、2 左(-x)、3 右(+x) |
| 奖励 | 撞墙/越界 `-5`（位置不变，不终止）；正常移动 `(旧曼哈顿距离 − 新曼哈顿距离) × 2 − 0.05`；到达终点 `+200` 并终止 |
| 终止 | 仅"到达终点"；无截断，靠外部 `max_steps` 兜底 |
| 渲染 | `render()` 用 Pygame 画 40 px/格（560 × 720），左下角为原点，绿色=智能体、红色=终点、黑色=障碍 |

同一文件内还带了**表格 Q-learning 基线**（`train_q_learning` / `evaluate` / `demo` / `obs_to_state`），
直接运行该文件即可对比"传统方法 vs PPO"：

```bash
python ppo_module/env_module/GridEnv.py
```

---

## 3. 环境准备

代码在 **Python 3.9 + Conda 环境 `ppo_env`** 下验证通过：

| 依赖 | 已验证版本 | 用途 |
| --- | --- | --- |
| `stable-baselines3` | 2.7.1 | PPO 算法与模型加载 |
| `torch` | 2.8.0（CPU 亦可） | 神经网络后端 |
| `gymnasium` | 1.1.1 | 环境接口 |
| `pygame` | 2.6.1 | 渲染与录屏取帧 |
| `numpy` | 2.0.2 | 数值计算 |
| `matplotlib` | 3.9.4 | 决策图 / 概率热力图 |
| `imageio` + `imageio-ffmpeg` | 2.37.2 / 0.6.0 | 合成 mp4 |
| `tensorboard` | 2.21.0 | 查看训练曲线 |
| `tkinter` | 系统自带 | `eval_with_popup.py` 的弹窗输入（缺失则自动改用命令行） |

```bash
conda activate ppo_env
# 若需重建环境：
# conda create -n ppo_env python=3.9 -y && conda activate ppo_env
pip install "stable-baselines3[extra]" gymnasium pygame matplotlib imageio imageio-ffmpeg tensorboard
```

> 以下命令**全部在仓库根目录执行**（即包含 `models/`、`ppo_module/` 的那一层），
> 否则 `models/` 与 `tensorboard_logs/` 会落到当前工作目录，推理脚本将找不到模型。

---

## 4. 如何启动

### 4.1 训练模型

```bash
python ppo_module/ppo_train.py
```

- 控制台会打印 SB3 的迭代日志（平均回报、episode 长度等）
- 默认训练 10 万步；想要更好效果就调大 `model.learn(total_timesteps=...)`
- 训练完自动保存到 `models/ppo_grid_model.zip`（覆盖旧模型，注意备份）

查看训练曲线：

```bash
tensorboard --logdir=./tensorboard_logs/
# 浏览器打开 http://localhost:6006
```

### 4.2 快速验证推理接口

```bash
python ppo_module/model_interface.py
```

输出形如：

```
[model_interface] ✅ 模型加载成功: .../models/ppo_grid_model.zip
[model_interface] 目标点固定为: [4, 2]
  位置 ( 1,  1) → 动作 0 (↑上)
  ...
```

### 4.3 指定起点/终点跑一局并出图

```bash
python ppo_module/eval_with_popup.py
```

1. 弹窗输入起点（如 `3,5`）
2. 弹窗输入终点（不能与起点相同）
3. 自动播放 Pygame 窗口、逐帧录制
4. 结束后在 `ppo_module/` 下生成 `[x,y].mp4`、`[x,y]_decision_map.png`、`[x,y]_prob_maps.png`，并尝试自动打开

### 4.4 真机部署（UDP）

```bash
python ppo_module/fixed-route.py
```

1. 先确认中继地址：文件顶部 `RELAY_IP = "192.168.123.220"`、`RELAY_PORT = 5091`
2. 加载模型 → 打印本次固定终点 → 提示输入起点
3. 输入起点后需再确认一次（`y`），随后机器狗开始按网格动作移动，终端实时打印 ASCII 地图
4. 到达终点或按 `Ctrl+C` 结束，程序会发送零速并关闭 UDP

> 真机运行前请把机器狗放稳、留出 0.6 m × 0.6 m 的安全余量；
> 每个动作都是"开环"位移（按时间积分的速度指令），地面打滑会造成累积误差。

---

## 5. 输出文件说明

| 文件 | 生成者 | 内容 |
| --- | --- | --- |
| `models/ppo_grid_model.zip` | `ppo_train.py` | PPO 策略网络 + 优化器状态，可直接 `PPO.load()` |
| `tensorboard_logs/PPO_1/events.out.tfevents.*` | `ppo_train.py` | TensorBoard 事件文件 |
| `ppo_module/[x,y].mp4` | `eval_with_popup.py` | 该起点出发的一局导航过程（10 fps） |
| `ppo_module/[x,y]_decision_map.png` | `eval_with_popup.py` | 全图最优动作箭头 + 障碍 + 起终点 + 实际路径 |
| `ppo_module/[x,y]_prob_maps.png` | `eval_with_popup.py` | 四个动作各自的概率热力图（2 × 2） |

---

## 6. 设计与实现要点

1. **一步一交互的在线训练**：环境每一步都返回新的 8 维观测，PPO 用 `n_steps=2048` 的 rollout 立即更新，
   不依赖离线数据集。
2. **观测只给局部可走性**：8 维观测里没有全局地图，智能体靠"距离变化"的奖励信号隐式学习绕障，
   因此策略在障碍密集区可能绕路。
3. **可达性预计算**：`_reachable_cache` / `_goal_cache` 在 `__init__` 时用 BFS 建好，
   避免 reset 时反复搜索，也保证"终点一定可达"，减少无效 episode。
4. **撞墙不终止**：撞墙只扣分、位置不变，避免智能体把"撞墙"当成快速结束 episode 的捷径。
5. **训练/推理共用同一环境类**：`model_interface.py` 复用 `GridNavEnv` 的障碍表与地图尺寸，
   手工拼观测时上下左右的方向定义与 `_get_obs()` 严格一致，避免"训练-部署观测错位"。

---

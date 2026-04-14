# Handoff: KAT Vision Pipeline — 在 nyu-186 上运行

## 你是谁

你在继续一个已经写好代码但还没跑过的任务。代码在本地 Mac 上写好并同步到了这台机器，现在需要在 nyu-186 服务器上实际运行和调试。

## 项目背景

KAT (Keypoint Action Tokens) 是一个 in-context imitation learning baseline，用 GPT-4o 作为 "policy"，给它几组 (scene, action) 的 demo，让它预测新 scene 对应的 action。在 RLBench 仿真中执行。

**已完成的 baseline（`kat_eval.py`）**：用 privileged `task_low_dim_state`（仿真器直接给的物体真实坐标）作为场景表示，跳过了视觉。已跑完 625 trials，结果在 `results/sweep.csv`。

**新写的 vision pipeline（还没跑过）**：按原论文实现 DINO-ViT 视觉管线，用 RGB-D 图像提取 3D keypoints 替换 privileged state。

## 新增代码（已写好，未测试）

| 文件 | 作用 |
|------|------|
| `kat_baseline/camera_utils.py` | 深度反投影：像素(u,v)+深度 → 3D 世界坐标 |
| `kat_baseline/action_tokens.py` | 论文 triplet pose 表示（SE(3) ↔ 三角形三点） |
| `kat_baseline/dino_keypoints.py` | DINO-ViT 特征提取 + Best Buddies 匹配 + 3D keypoint |
| `kat_baseline/kat_eval_vision.py` | 主评估脚本，用视觉 keypoints 替换 task_low_dim_state |
| `kat_baseline/run_sweep_vision.py` | 全量 sweep 驱动 |

现有 baseline 代码（`kat_eval.py` 等）完全不动。

## 需要做的事

### Step 1: 环境准备（在 nyu-186 上）

```bash
ssh nyu-186
source /home/nyuair/anaconda3/etc/profile.d/conda.sh && conda activate kat
export COPPELIASIM_ROOT=~/zhewen/robo/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export OPENAI_API_KEY=<key>
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
cd ~/zhewen/robo/KAT
```

先确认依赖：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install timm  # DINO 可能需要
```

**注意**：代码在 `/media/zhewen/d/program/robo/KAT/`（这台 huawei 机器），但需要在 nyu-186 上跑。你需要先把代码同步到 nyu-186 的 `~/zhewen/robo/KAT/`，或者直接在 nyu-186 上编辑。nyu-186 上原来就有 baseline 代码，只需要把新增的 5 个文件拷过去。

### Step 2: Smoke Test

```bash
# 在 nyu-186 上
xvfb-run -a python kat_baseline/kat_eval_vision.py --task reach_target --n_demos 5 --n_trials 1
```

这会打印：
- `front_rgb shape` — 应该是 (512, 512, 3)
- `front_depth shape` — 应该是 (512, 512)
- `misc keys` — 需要确认包含 `front_camera_intrinsics` 和 `front_camera_extrinsics`

### Step 3: 可能需要修的问题

1. **`camera_utils.py` 里的 key 名**：如果 `obs.misc` 里相机矩阵的 key 不叫 `front_camera_intrinsics`，需要改 `get_camera_matrices()` 函数。打印 `obs.misc.keys()` 看实际字段名。

2. **DINO 模型下载**：第一次跑会通过 `torch.hub` 下载 DINOv1 ViT-B/8 (~350MB)。如果服务器没网，需要手动下载放到 `~/.cache/torch/hub/`。

3. **深度图格式**：RLBench 的 `front_depth` 可能不是米制。如果反投影出来的 3D 坐标跟 `task_low_dim_state` 差很多，检查深度值的量级。可能需要在 `ObservationConfig` 里加 `obs_config.front_camera.depth_in_meters = True`。

4. **action_tokens.py 的 import**：`from kat_eval import ...` 需要在同一目录下运行，或者把 `kat_baseline/` 加到 `PYTHONPATH`。

### Step 4: 全量 Sweep

```bash
# Smoke pass（1 trial per cell，25 cells）
xvfb-run -a python kat_baseline/run_sweep_vision.py --smoke

# Full sweep（25 trials per cell，625 trials）
xvfb-run -a python kat_baseline/run_sweep_vision.py
```

结果保存到 `results/sweep_vision.csv`。

## 架构要点

### Vision Pipeline 流程

```
Demo RGB image → DINO-ViT → patch features (64×64×768)
                                    ↓
多张 demo 图像 → Best Buddies Nearest Neighbours → K=10 salient descriptors
                                    ↓
新图像 → DINO features → 匹配 K descriptors → 2D pixel coords
                                    ↓
depth map + camera calibration → 3D world coordinates (K×3)
                                    ↓
格式化为 "kp_0: [x, y, z]" → 喂给 GPT-4o
```

### Action Token 格式（论文 Fig. 4）

每个 SE(3) 位姿 → 3 个 3D 点组成的三角形：
- p1 = position
- p2 = position + 0.05 * R[:, 0]（x 轴方向偏移 5cm）
- p3 = position + 0.05 * R[:, 1]（y 轴方向偏移 5cm）

Waypoint 格式：`[p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, gripper]`（10D）

### 跟 baseline 的对比

| | Baseline (`kat_eval.py`) | Vision (`kat_eval_vision.py`) |
|---|---|---|
| 场景表示 | `task_low_dim_state`（物体真实坐标） | DINO-ViT 3D keypoints |
| Action 格式 | `[x,y,z, r1..r6, gripper]`（6D rotation） | `[p1, p2, p3, gripper]`（triplet） |
| 输入 | 纯坐标数字 | RGB + depth 图像 → keypoints |
| 相机 | 全部关闭 | front_camera 开启（512×512） |

## 关键文件路径

- 代码：`kat_baseline/` 下所有 `.py`
- 原论文：`docs/KAT_paper.pdf`
- Baseline 结果：`results/sweep.csv`
- Vision 结果（待生成）：`results/sweep_vision.csv`
- LLM 缓存：`kat_baseline/cache/`（共享）
- DINO 特征缓存：`kat_baseline/dino_cache/`（新建）

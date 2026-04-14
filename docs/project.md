# KAT-style ICIL Baseline — Project Documentation

## Overview

Implementation of a Keypoint Action Tokens (KAT)-style in-context imitation learning baseline on RLBench. An LLM (GPT-4o) acts as the "policy": given N demo (scene, action) pairs as in-context examples, it predicts an action waypoint sequence for a new scene in a single API call. The robot executes the predicted waypoints open-loop in simulation.

This is a CoRL paper baseline for multi-demo ICIL.

## Architecture

```
Demo scenes → extract (scene state, gripper trajectory) → build text prompt
                                                              ↓
New scene → append to prompt → GPT-4o API call → parse JSON waypoints
                                                              ↓
                         6D rotation → quaternion → RLBench action → execute
```

- **State representation**: `task_low_dim_state` (privileged object poses from RLBench, no vision)
- **Action representation**: 10D waypoints `[x, y, z, r1..r6, gripper]` using 6D rotation (no quaternions in prompts)
- **Execution**: `EndEffectorPoseViaPlanning` (RRT-based motion planning), open-loop

## Environment

- **Server**: `nyu-186:~/zhewen/robo/KAT/kat_baseline/`
- **Conda env**: `kat` (cloned from `rlbench`)
- **Stack**: RLBench 1.2.0 + PyRep 4.1.0 + CoppeliaSim V4.1.0 + OpenAI SDK
- **LLM**: GPT-4o via proxy (`yansd666.com/v1`)
- **Headless**: `xvfb-run` (no GPU rendering)

## Results — Full Sweep (25 trials per cell)

| Task                    | n=1  | n=2  | n=5  | n=10 | n=20 |
|-------------------------|------|------|------|------|------|
| **Reach Target**        | 76%  | 88%  | 100% | 100% | 96%  |
| **Push Button**         | 20%  | 72%  | 84%  | 88%  | 92%  |
| **Pick Up Cup**         | 16%  | 20%  | 56%  | 76%  | 60%  |
| **Take Lid Off Saucepan** | 4% | 24%  | 28%  | 36%  | 12%  |
| **Stack Blocks**        | 0%   | 0%   | 0%   | 0%   | —*   |

*stack_blocks n=20 skipped (demo collection timeout >10min, n=1~10 all 0%)

### Key Findings

1. **More demos help** — clear upward trend on simple/medium tasks (reach_target: 76% → 100%, push_button: 20% → 92%)
2. **Diminishing returns at n=20** — pick_up_cup drops from 76% (n=10) to 60% (n=20), take_lid drops from 36% to 12%. Likely due to prompt length hurting LLM performance
3. **Complexity ceiling** — stack_blocks (multi-step manipulation) never succeeds. The open-loop, single-call approach can't handle long-horizon tasks
4. **No parse errors** — GPT-4o reliably outputs valid JSON waypoint arrays
5. **Execution errors** — most failures are `InvalidActionError` (unreachable poses) or task not completed despite successful execution

### Average Success Rate

| n_demos | 5-task avg | 4-task avg (excl. stack_blocks) |
|---------|-----------|-------------------------------|
| 1       | 23.2%     | 29.0%                         |
| 2       | 40.8%     | 51.0%                         |
| 5       | 53.6%     | 67.0%                         |
| **10**  | **60.0%** | **75.0%**                     |
| 20      | 52.0%     | 65.0%                         |

Best average at **n=10** (60.0% all tasks, 75.0% excluding stack_blocks).

### Comparison with Original KAT Paper

| | KAT Paper | Our Reproduction |
|---|-----------|-----------------|
| **Vision** | DINO-ViT 3D keypoints | Privileged `task_low_dim_state` (no vision noise) |
| **LLM** | GPT-4 Turbo | GPT-4o |
| **Platform** | Real Sawyer robot, 9 tasks | RLBench simulation, 5 tasks |
| **Action repr.** | 3D keypoint triplets | 6D rotation + position |
| **Avg success (10 demos)** | **68%** | **60%** (all 5), **75%** (excl. stack_blocks) |

**Consistent trends:**
- Both show clear improvement with more demos (few-shot scaling)
- Both observe performance saturation/decline at high demo counts (paper: ~40-50 tokens, ours: n=20)
- Simple tasks achieve high success, complex multi-step tasks remain challenging
- LLM reliably generates valid action token sequences without parse errors

**Key differences:**
- Our simple tasks score higher than the paper average (reach 100% vs paper avg 68%) — privileged state removes vision noise
- Our complex tasks score lower (stack_blocks 0%) — pure open-loop single-call can't handle long-horizon multi-step manipulation
- Our n=20 performance drop is sharper than the paper's — possibly due to prompt format (raw coordinates vs tokenized keypoints) or GPT-4o vs GPT-4 Turbo context handling

**Conclusion:** As a baseline reproduction, the trends are faithful to the original paper. The implementation is suitable for CoRL paper comparison.

### Cost

- ~$10 total for 625 API calls (baseline sweep)
- LLM response caching via SHA256 hash prevents duplicate charges on reruns

## 视觉管线实验

详见 [docs/experiment.md](experiment.md)。

四种方法对比（n_demos=5）：

| 任务 | Baseline (GT) | DINO-ViT | Depth 原始 | **Depth 改进** |
|------|:---:|:---:|:---:|:---:|
| reach_target | 100% | 0% | 20% | **64%** |
| push_button | 84% | — | — | **72%** |
| pick_up_cup | 56% | — | — | **36%** |
| take_lid_off_saucepan | 28% | — | — | 0% |
| stack_blocks | 0% | — | — | 0% |

Depth 改进版核心技术：红色 hue 过滤 + 跨相机簇匹配 + 机械臂遮罩 + 偏移校正。

## File Structure

```
kat_baseline/
  kat_eval.py               # 基线评估 (GT state)
  kat_eval_depth.py          # 深度管线评估
  kat_eval_vision.py         # DINO 管线评估 (已放弃)
  depth_object_detector.py   # 检测核心
  camera_utils.py            # 深度投影工具
  dino_keypoints.py          # DINO 特征提取
  action_tokens.py           # SE(3) triplet 表示
  run_sweep.py               # 基线全面扫描
  run_sweep_depth.py         # 深度管线全面扫描
  scripts/
    kat_smoke.py             # 单次冒烟测试
    plot_results.py          # 成功率绘图
    record_episode.py        # 录制视频
    save_visualizations.py   # 生成可视化
  diagnostics/
    diagnose_depth.py        # 深度检测诊断
    diagnose_vision.py       # DINO 诊断
    diagnose_overhead.py     # overhead 相机分析
  cache/                     # LLM 响应缓存
  results/
    sweep.csv                # 基线完整结果 (625 rows)
    depth_*.csv              # 深度管线结果
```

## How to Run

```bash
# SSH to server
ssh nyu-186

# Activate environment
source /home/nyuair/anaconda3/etc/profile.d/conda.sh
conda activate kat
export COPPELIASIM_ROOT=~/zhewen/robo/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export OPENAI_API_KEY=<key>
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
cd ~/zhewen/robo/KAT

# Stage 1: Smoke test
xvfb-run -a python kat_baseline/kat_smoke.py

# Stage 2: Single task eval
xvfb-run -a python kat_baseline/kat_eval.py --task reach_target --n_demos 5 --n_trials 25

# Stage 3: Full sweep (smoke first, then full)
xvfb-run -a python kat_baseline/run_sweep.py --smoke
xvfb-run -a python kat_baseline/run_sweep.py

# Stage 4: Plot
python kat_baseline/plot_results.py

# Record episode visualization
xvfb-run -a python kat_baseline/record_episode.py --task reach_target --n_demos 5 --seed 1000
```

## 视觉管线 (Vision Pipeline)

### 三种场景表征方案

| 方案 | 场景输入 | reach_target n=5 | 说明 |
|------|---------|-----------------|------|
| **基线（特权数据）** | `task_low_dim_state` | 100% | 模拟器内部数据，现实不可用 |
| **DINO-ViT（论文方法）** | DINO keypoints 3D | 0% | BBNN 选择静态特征，无法定位小物体 |
| **深度方案 v2** | RGB-D 深度差分 | 20% → 改进中 | 可行但需提高检测可靠性 |

### DINO-ViT 方案（已放弃）

按论文复现：DINO-ViT/8, stride=4, layer 9 KEY features, wrist camera, Best Buddies NN。
- 结果：0/5（0%）— 所有 keypoints 落在桌面 (z≈0.75)
- 根因：BBNN 选择跨 demo **一致**的特征（桌面、机械臂），而非 **变化**的任务物体
- 论文在真实机器人上成功是因为物体更大、纹理更丰富

### 深度方案架构

```
Demo pool (N+5 episodes)
  ├─ 每个 demo 第一帧的 depth map → 中位数 → 背景模型 (left_shoulder + overhead)
  ├─ Demo GT positions → 校准检测阈值
  └─ 每个 demo 第一帧 → 深度差分检测 → demo 场景描述

新场景 observation
  ├─ left_shoulder depth: |当前 - 背景| > 阈值 → 前景
  ├─ overhead depth: 同上（独立检测）
  ├─ 机械臂遮罩: gripper_pose → 线段模型 → 排除臂部像素
  ├─ 颜色过滤: HSV 饱和度 > 阈值 → 排除灰色像素
  ├─ 像素聚类 + 智能评分 → 目标物体 3D 位置
  └─ 多相机交叉验证 → 最终位置估计
        ↓
  build_scene_str() → GPT-4o → 执行 (与基线相同)
```

### 深度管线评测结果

| 任务 | 深度管线 | 基线 (GT) | 深度/基线 |
|------|---------|----------|-----------|
| reach_target n=5 | **64%** (16/25) | 100% | 64% |
| push_button n=5 | **72%** (18/25) | 84% | 86% |
| pick_up_cup n=5 | 待测 | 56% | ? |

### 关键技术要点

1. **深度格式**: RLBench depth 是 Z-buffer (0-1)，非米。转换: `linear = near + buf * (far - near)`
2. **投影方法**: 必须用 PyRep 官方 `pointcloud_from_depth` 方法（负焦距 OpenGL 惯例）
3. **机械臂遮罩**: 用 `obs.gripper_pose` 构建"底座→夹爪"线段，遮罩 0.15m 半径内的像素（含 demo 臂位置）
4. **多相机融合**: left_shoulder + overhead 各自检测所有候选簇，跨相机匹配找 3D 一致的簇对
5. **红色色调过滤**: H<25 或 H>230 + 高饱和度，RLBench 任务物体都是红色
6. **自适应阈值**: 从 demo GT 位置校准（用 pointcloud 反查避免负焦距投影 bug）
7. **20 demo pool**: 比 10 更稳健的背景模型

### 相关文件

```
kat_baseline/
  kat_eval_depth.py         # 深度管线评估入口
  depth_object_detector.py  # 检测核心：背景模型、机械臂遮罩、多相机融合、簇评分
  camera_utils.py           # 深度投影工具（PyRep 方法）
  run_sweep_depth.py        # 深度管线全面扫描
  kat_eval_vision.py        # DINO 管线评估（已放弃）
  dino_keypoints.py         # DINO 特征提取
  action_tokens.py          # SE(3) triplet 表示
  diagnostics/
    diagnose_depth.py       # 深度检测诊断
    diagnose_vision.py      # DINO keypoint 诊断
    diagnose_overhead.py    # overhead 相机分析
  scripts/
    save_visualizations.py  # 生成 RGB/depth/DINO 可视化
```

## Known Limitations

- **No GPU rendering** — CoppeliaSim camera requires OpenGL context, but episode recording works via `record_episode.py`
- **stack_blocks n_demos≥10** — demo collection takes >5 min, may timeout
- **Open-loop execution** — no replanning. Single LLM call per episode
- **World frame only** — no relative-to-keypoint transformation (planned ablation)
- **深度检测局限** — 对极小物体（<1cm）检测不稳定；依赖物体与背景的深度差异

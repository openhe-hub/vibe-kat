# KAT 视觉管线实验报告

## 概述

本实验对比了四种场景表征方法在 RLBench 上的 KAT (Keypoint Action Tokens) in-context imitation learning 性能。目标是用视觉输入替代模拟器特权数据 `task_low_dim_state`，使方案可迁移到真实场景。

## 四种方法

| 方法 | 场景输入 | 可真实部署？ |
|------|---------|:---:|
| **Baseline (GT state)** | 模拟器 `task_low_dim_state` — 物体精确坐标+姿态 | 否 |
| **DINO-ViT** | DINO-ViT/8 特征 → Best Buddies NN → 3D keypoints | 是 |
| **Depth v2 原始** | 单相机背景减除 → 深度差分 → 3D 位置 | 是 |
| **Depth v2 改进** | 多相机深度+颜色+跨相机匹配 → 3D 位置 | 是 |

## 实验结果（n_demos=5, 25 trials）

| 任务 | Baseline (GT) | DINO-ViT | Depth v2 原始 | Depth v2 改进 |
|------|:---:|:---:|:---:|:---:|
| reach_target | **100%** | 0% | 20% | **64%** |
| push_button | **84%** | — | — | **72%** |
| pick_up_cup | **56%** | — | — | **36%** |
| take_lid_off_saucepan | **28%** | — | — | 0% |
| stack_blocks | 0% | — | — | 0% |
| **平均（5 任务）** | **53.6%** | — | — | **34.4%** |
| **平均（排除 stack）** | **67.0%** | — | — | **43.0%** |
| **红色物体任务平均** | **80.0%** | — | — | **57.3%** |

### 检测准确率 vs 任务成功率

| 任务 | 检测准确率 (<3cm) | 准确时成功率 | 物体特征 |
|------|:---:|:---:|------|
| reach_target | 68% | 94% | 红色小球 ~1cm |
| push_button | ~92% | ~78% | 红色按钮 ~3cm |
| pick_up_cup | 76% | 37% | 红色杯子 ~8cm（偏移校正后） |
| take_lid_off_saucepan | ~30% | 0% | 灰色金属锅，颜色过滤失效，检测锁定错误位置 |

## 各方法详解

### 1. Baseline (GT state)

直接使用 RLBench 提供的 `task_low_dim_state`，包含物体精确位置和姿态。无视觉噪声，是性能上限。

- **优点**：精确、无噪声
- **缺点**：仅限模拟器，无法部署到真实机器人

### 2. DINO-ViT（论文方法）

按 KAT 论文 (Di Palo & Johns, RSS 2024) 复现：
- DINO-ViT/8, stride=4, layer 9 KEY features
- Best Buddies Nearest Neighbours 选择 K=10 salient descriptors
- Wrist camera (论文使用末端执行器相机)
- "Pattern generator" 系统提示 + INPUTS/OUTPUTS 格式

**结果**: 0% on reach_target

**失败原因**: BBNN 结构性缺陷 — 它选择跨 demo **一致**的特征（桌面、机械臂），而非**变化**的任务物体。在论文的真实机器人场景中有效是因为物体更大、纹理更丰富。RLBench 中 ~1cm 的目标球在 DINO 特征空间中不够显著。

### 3. Depth v2 原始

单相机（left_shoulder）背景减除方案：
- 10 demo 中位数深度作为背景模型
- |当前深度 - 背景| > 阈值 → 前景像素
- 饱和度过滤 + 工作空间边界过滤
- 选最大簇的质心作为物体位置

**结果**: 20% on reach_target (2/10)

**问题**: ~1000 噪声前景像素（机械臂移动、幻影目标位置）远多于真正目标 (~30 像素)

### 4. Depth v2 改进

在 v2 基础上的系统性改进：

#### 改进链（按影响从大到小）

| 改进 | reach_target 效果 | 说明 |
|------|:---:|------|
| 红色 hue 过滤 | 32% → 64% | H<25 or H>230 + 高饱和度，RLBench 红色物体通用 |
| 跨相机簇匹配 | 20% → 32% | left_shoulder + overhead 全候选簇 3D 配对 |
| 机械臂遮罩 | 部分贡献 | 0.15m 圆柱体从底座到夹爪，含 demo 臂位置 |
| 偏移校正 | pick_up_cup 24%→36% | 自动校准表面检测 vs GT 原点的系统偏差 |
| 校准修复 | 基础设施 | PyRep 负焦距投影 bug，用 pointcloud 反查替代 |

#### 尝试过但无效的方法

| 方法 | 结果 | 原因 |
|------|------|------|
| 3 相机 (left+overhead+right) | 20%，更差 | left/right 对称视角导致噪声假匹配 |
| 深度方差过滤 (std>0.01) | 32%，无改善 | 过于激进，连真目标也被过滤 |
| 工作空间中心性评分 | 无改善 | 噪声恰好集中在中心区域 |

## 关键发现

### 1. 颜色是 RLBench 中最强的检测信号
RLBench 的任务物体（目标球、按钮、杯子、积木）都是纯红色 (H≈0, S=255)。仅凭红色 hue 过滤就能将噪声像素从 ~1500 降到 ~30。唯一例外是 take_lid_off_saucepan（灰色金属锅）。

### 2. 跨相机 3D 一致性有效但依赖视角多样性
left_shoulder（侧上方）+ overhead（正上方）视角差异大，噪声在不同视角映射到不同 3D 位置，真实物体位置一致。但 left_shoulder + right_shoulder 视角太对称，反而增加假匹配。

### 3. 检测准确时执行基本可靠
检测误差 <3cm 时，reach_target 94%、push_button ~78% 的任务成功率，接近基线。性能差距主要来自检测失败，而非动作预测。

### 4. 物体大小直接决定检测难度
push_button (按钮 ~3cm) > reach_target (球 ~1cm)。按钮更大、在桌面固定高度、纯红色 → 92% 检测准确率。小球 → 68%。

### 5. 系统偏移可自动校正
深度检测到的是物体可见表面（如杯口），而 GT 报告的是物体原点（如杯底）。通过对比 demo 中检测位置与 GT 的中位数差异，自动计算偏移校正向量。pick_up_cup 的 z 偏移 ~8.4cm 被完美校正。

### 6. 灰色物体需要不同策略
take_lid_off_saucepan 的金属锅 (sat=0) 无法用颜色过滤。需要纯深度推理 + 大簇优先策略。

## 失败模式分析

### reach_target 失败 (36% 不准确)
- **7/9 失败案例**: 目标在机械臂底座附近 (x < 0.2)
- **原因**: 相机视角被臂遮挡，无法看到目标
- **改善方向**: 加 wrist camera（不参与背景建模，仅用于 query 辅助）

### pick_up_cup 失败 (64% 不成功)
- 检测准确率 76%（偏移校正后），但成功率仅 36%
- **原因**: 抓取动作本身难度高（基线也只有 56%）
- 检测不是瓶颈，动作预测质量才是

### take_lid_off_saucepan 失败 (0%)
- 灰色金属物体 (sat=0) 无颜色信号，纯深度检测噪声太多
- 检测反复锁定同一错误位置 [0.306, 0.063, 0.733]（场景中某个固定特征）
- 多物体场景（锅体 + 锅盖），单物体检测无法提供足够信息
- 灰色物体特化（大簇优先策略）未能解决，根因是噪声簇恰好最大
- 基线本身也仅 28%（任务操作难度高）

## 技术架构

```
Demo Pool (20 episodes)
  ├─ 每个 demo 第一帧 depth → 中位数背景模型 (left_shoulder + overhead)
  ├─ Demo GT positions → 校准检测阈值 (pointcloud 反查)
  ├─ Demo GT positions → 校准偏移向量 (表面 vs 原点)
  └─ Demo gripper poses → 扩展机械臂遮罩

查询场景 (Query)
  ├─ 两相机各自:
  │   ├─ |depth - background| > threshold → 前景
  │   ├─ 机械臂遮罩 (当前 + demo 位置)
  │   ├─ 红色 hue 过滤 (有颜色信号时) / 纯深度 (灰色物体)
  │   ├─ 像素聚类 + 评分
  │   └─ 返回所有候选簇
  └─ 跨相机匹配: 找 3D 位置最一致的簇对 → 物体位置
      └─ + 偏移校正 → 最终 3D 位置
          └─ build_scene_str() → GPT-4o → 执行
```

## 环境配置

- **服务器**: nyu-186, NVIDIA RTX A6000 48GB
- **RLBench**: 1.2.0 + PyRep 4.1.0 + CoppeliaSim V4.1.0
- **LLM**: GPT-4o via proxy (yansd666.com/v1)
- **渲染**: DISPLAY=:0, QT_PLUGIN_PATH=$COPPELIASIM_ROOT, QT_XCB_GL_INTEGRATION=xcb_glx
- **关键**: 必须用 PyRep 的 `pointcloud_from_depth` 方法（负焦距 OpenGL 惯例）

## 文件结构

```
kat_baseline/
  kat_eval.py               # 基线评估 (GT state)
  kat_eval_depth.py          # 深度管线评估
  depth_object_detector.py   # 检测核心: 背景模型、臂遮罩、颜色过滤、簇评分、多相机融合、偏移校正
  camera_utils.py            # 深度投影 (PyRep 方法)
  run_sweep.py               # 基线全面扫描
  run_sweep_depth.py         # 深度管线全面扫描
  scripts/                   # 工具脚本 (冒烟测试、录制、绘图、可视化)
  diagnostics/               # 诊断脚本 (深度、视觉、overhead 分析)
  archive/                   # 已弃用: DINO 管线 (kat_eval_vision, dino_keypoints, action_tokens)
  cache/                     # LLM 响应缓存
  results/                   # 评估结果 CSV
```

## 下一步

1. 考虑 n_demos=10 测试（基线在 n=10 时平均最优 60%）
2. reach_target 遮挡问题：尝试 wrist camera 辅助检测臂附近目标
3. take_lid_off_saucepan：需要完全不同的检测策略（如学习型分割 SAM/DINO-seg）
4. 整合到 CoRL 论文对比表

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

- ~$10 total for 625 API calls
- LLM response caching via SHA256 hash prevents duplicate charges on reruns

## File Structure

```
kat_baseline/
  kat_smoke.py          # Stage 1: single-trial smoke test
  kat_eval.py           # Stage 2: multi-trial evaluation
  run_sweep.py          # Stage 3: full task × n_demos sweep
  plot_results.py       # Stage 4: success rate plot with Wilson CIs
  record_episode.py     # 3D trajectory visualization (matplotlib)
  cache/                # LLM response cache (SHA256 → JSON)
  results/
    sweep.csv           # Full results (625 rows)
    sweep_smoke.csv     # 1-trial smoke pass (25 rows)
    results.png         # Success rate plot
    results.pdf
    videos/             # Per-episode trajectory visualizations (PNG + GIF)
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

## Known Limitations

- **No GPU rendering** — CoppeliaSim camera requires OpenGL context. Trajectory visualizations use matplotlib instead of simulation screenshots
- **stack_blocks n_demos≥10** — demo collection takes >5 min, may timeout
- **Open-loop execution** — no replanning. Single LLM call per episode
- **World frame only** — no relative-to-keypoint transformation (planned ablation)

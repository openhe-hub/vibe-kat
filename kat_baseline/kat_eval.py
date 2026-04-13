#!/usr/bin/env python3
"""KAT-style ICIL evaluation on RLBench (Stage 2).

Usage:
    python kat_eval.py --task reach_target --n_demos 5 --n_trials 25
"""

import os
import sys
import json
import re
import csv
import signal
import hashlib
import argparse
import numpy as np
from openai import OpenAI

# ── Rotation utilities ──────────────────────────────────────────────

def quat_xyzw_to_rot_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

def rot_matrix_to_6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])

def sixd_to_rot_matrix(r6):
    a1, a2 = r6[:3], r6[3:6]
    e1 = a1 / np.linalg.norm(a1)
    u2 = a2 - np.dot(e1, a2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)

def rot_matrix_to_quat_xyzw(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)

def sixd_to_quat_xyzw(r6):
    return rot_matrix_to_quat_xyzw(sixd_to_rot_matrix(r6))

# ── Waypoint extraction ─────────────────────────────────────────────

def extract_waypoints(demo, n_keyframes=10):
    n = len(demo)
    indices = list(range(n)) if n <= n_keyframes else np.linspace(0, n - 1, n_keyframes, dtype=int).tolist()
    waypoints = []
    for i in indices:
        obs = demo[i]
        pos = obs.gripper_pose[:3]
        quat = obs.gripper_pose[3:]
        R = quat_xyzw_to_rot_matrix(quat)
        r6 = rot_matrix_to_6d(R)
        g = 1.0 if obs.gripper_open > 0.5 else 0.0
        waypoints.append(np.concatenate([pos, r6, [g]]))
    return waypoints

# ── Prompt building ──────────────────────────────────────────────────

def fmt(arr, decimals=3):
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr) + "]"

def build_scene_str(state):
    n = len(state)
    if n % 7 == 0:
        chunk = 7
    elif n % 3 == 0:
        chunk = 3
    else:
        chunk = n
    n_objects = max(1, n // chunk)
    lines = []
    for i in range(n_objects):
        obj_state = state[i*chunk:(i+1)*chunk]
        if len(obj_state) == 7:
            pos = obj_state[:3]
            quat = obj_state[3:]
            R = quat_xyzw_to_rot_matrix(quat)
            r6 = rot_matrix_to_6d(R)
            lines.append(f"  object_{i}: {fmt(np.concatenate([pos, r6]))}")
        elif len(obj_state) == 3:
            r6_id = np.array([1, 0, 0, 0, 1, 0], dtype=float)
            lines.append(f"  object_{i}: {fmt(np.concatenate([obj_state, r6_id]))}")
        else:
            lines.append(f"  object_{i}: {fmt(obj_state)}")
    return "\n".join(lines)

def build_prompt(demos_data, query_scene_str):
    blocks = []
    for scene_str, waypoints in demos_data:
        wp_strs = [fmt(wp) for wp in waypoints]
        block = f"Scene:\n{scene_str}\nActions:\n  [{', '.join(wp_strs)}]"
        blocks.append(block)
    blocks.append(f"Scene:\n{query_scene_str}\nActions:")
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """You are a robot action predictor. Given demonstration pairs of (scene, actions), predict actions for a new scene.

Coordinate frame: world frame, meters. Each waypoint is [x, y, z, r1, r2, r3, r4, r5, r6, g] where:
- x, y, z: end-effector position in meters
- r1..r6: 6D rotation representation (first two columns of the rotation matrix, flattened)
- g: gripper state (0 = closed, 1 = open)

Respond with ONLY a JSON array of waypoint arrays. No prose, no explanation."""

# ── LLM call with caching ───────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def get_cache_path(prompt_str):
    h = hashlib.sha256(prompt_str.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def call_llm_cached(system_prompt, user_prompt):
    """Call LLM with SHA256-based caching. Returns (response_text, usage_dict, cached)."""
    full_prompt = system_prompt + "\n---\n" + user_prompt
    cache_path = get_cache_path(full_prompt)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return cached["response"], cached["usage"], True

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://yansd666.com/v1",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"response": text, "usage": usage}, f)

    return text, usage, False

# ── Response parsing ─────────────────────────────────────────────────

def parse_response(text):
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    if not isinstance(data, list):
        return None, "Response is not a list"
    waypoints = []
    for i, wp in enumerate(data):
        if not isinstance(wp, list):
            return None, f"Waypoint {i} is not a list"
        if len(wp) != 10:
            return None, f"Waypoint {i} has length {len(wp)}, expected 10"
        try:
            wp = [float(v) for v in wp]
        except (ValueError, TypeError) as e:
            return None, f"Waypoint {i} has non-float values: {e}"
        waypoints.append(np.array(wp))
    return waypoints, None

def waypoint_to_action(wp_10d):
    pos = wp_10d[:3]
    r6 = wp_10d[3:9]
    gripper = wp_10d[9]
    quat = sixd_to_quat_xyzw(r6)
    return np.concatenate([pos, quat, [gripper]])


# ── Step timeout ─────────────────────────────────────────────────────

STEP_TIMEOUT = 60  # seconds per task.step() call

class StepTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise StepTimeoutError(f"task.step() timed out after {STEP_TIMEOUT}s")

# ── Task name → class mapping ───────────────────────────────────────

TASK_MAP = {
    "reach_target": "ReachTarget",
    "push_button": "PushButton",
    "pick_up_cup": "PickUpCup",
    "take_lid_off_saucepan": "TakeLidOffSaucepan",
    "stack_blocks": "StackBlocks",
}

def get_task_class(task_name):
    import rlbench.tasks as tasks_module
    class_name = TASK_MAP.get(task_name)
    if class_name is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}")
    return getattr(tasks_module, class_name)

# ── Main evaluation ──────────────────────────────────────────────────

def run_eval(task_name, n_demos, n_trials, demo_pool_size=None):
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.backend.exceptions import InvalidActionError

    if demo_pool_size is None:
        demo_pool_size = max(n_demos + 5, n_demos * 2)

    # Setup
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.front_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task_class = get_task_class(task_name)
    task = env.get_task(task_class)

    # Collect demo pool
    print(f"Collecting {demo_pool_size} demos for pool...")
    demo_pool = task.get_demos(demo_pool_size, live_demos=True)
    print(f"Demo pool collected. Lengths: {[len(d) for d in demo_pool]}")

    # Inspect task_low_dim_state once
    sample_state = demo_pool[0][0].task_low_dim_state
    print(f"task_low_dim_state shape: {sample_state.shape}, contents: {sample_state}")

    # Fixed test seeds for reproducibility
    test_seeds = list(range(1000, 1000 + n_trials))

    # Results
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    cache_hits = 0

    for trial_idx, seed in enumerate(test_seeds):
        print(f"\n--- Trial {trial_idx+1}/{n_trials} (seed={seed}) ---")

        # Select n_demos from pool (deterministic per seed)
        rng = np.random.RandomState(seed)
        demo_indices = rng.choice(len(demo_pool), size=n_demos, replace=False)
        selected_demos = [demo_pool[i] for i in demo_indices]

        # Extract demo data
        demos_data = []
        for demo in selected_demos:
            scene_str = build_scene_str(demo[0].task_low_dim_state)
            waypoints = extract_waypoints(demo, n_keyframes=10)
            demos_data.append((scene_str, waypoints))

        # Reset env for test episode
        descriptions, obs = task.reset()
        query_scene_str = build_scene_str(obs.task_low_dim_state)

        # Build prompt
        user_prompt = build_prompt(demos_data, query_scene_str)

        # Call LLM
        parse_error = ""
        exec_error = ""
        success = False
        n_executed = 0

        try:
            response_text, usage, was_cached = call_llm_cached(SYSTEM_PROMPT, user_prompt)
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            if was_cached:
                cache_hits += 1
            print(f"  LLM: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion tokens {'(cached)' if was_cached else ''}")
        except Exception as e:
            parse_error = f"API error: {e}"
            print(f"  LLM API error: {e}")
            results.append({
                "task": task_name, "n_demos": n_demos, "trial_id": trial_idx,
                "seed": seed, "success": False, "n_waypoints_executed": 0,
                "parse_error": parse_error, "execution_error": "",
            })
            continue

        # Parse
        waypoints_10d, perr = parse_response(response_text)
        if perr:
            parse_error = perr
            print(f"  Parse error: {perr}")
            results.append({
                "task": task_name, "n_demos": n_demos, "trial_id": trial_idx,
                "seed": seed, "success": False, "n_waypoints_executed": 0,
                "parse_error": parse_error, "execution_error": "",
            })
            continue

        # Convert to actions
        actions = [waypoint_to_action(wp) for wp in waypoints_10d]

        # Execute with per-step timeout
        for i, action in enumerate(actions):
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(STEP_TIMEOUT)
                obs, reward, terminate = task.step(action)
                signal.alarm(0)  # cancel alarm
                n_executed += 1
                if terminate:
                    break
            except InvalidActionError as e:
                signal.alarm(0)
                exec_error = f"Step {i}: InvalidActionError: {e}"
                print(f"  InvalidActionError at step {i}: {e}")
                break
            except StepTimeoutError as e:
                signal.alarm(0)
                exec_error = f"Step {i}: {e}"
                print(f"  Timeout at step {i}")
                break
            except Exception as e:
                signal.alarm(0)
                exec_error = f"Step {i}: {e}"
                print(f"  Error at step {i}: {e}")
                break

        # Check success
        success, _ = task._task.success()
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'} ({n_executed}/{len(actions)} waypoints)")

        results.append({
            "task": task_name, "n_demos": n_demos, "trial_id": trial_idx,
            "seed": seed, "success": success, "n_waypoints_executed": n_executed,
            "parse_error": parse_error, "execution_error": exec_error,
        })

    env.shutdown()

    # Summary
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {task_name}, n_demos={n_demos}")
    print(f"  Success: {n_success}/{n_trials} ({100*n_success/n_trials:.1f}%)")
    print(f"  Total tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"  Cache hits: {cache_hits}/{n_trials}")
    print(f"{'='*60}")

    return results

def save_results_csv(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["task", "n_demos", "trial_id", "seed", "success",
                  "n_waypoints_executed", "parse_error", "execution_error"]
    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="KAT ICIL evaluation")
    parser.add_argument("--task", type=str, default="reach_target",
                        choices=list(TASK_MAP.keys()))
    parser.add_argument("--n_demos", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/<task>_<n_demos>.csv)")
    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        args.output = os.path.join(results_dir, f"{args.task}_{args.n_demos}.csv")

    print(f"Running evaluation: task={args.task}, n_demos={args.n_demos}, n_trials={args.n_trials}")
    results = run_eval(args.task, args.n_demos, args.n_trials)
    save_results_csv(results, args.output)

if __name__ == "__main__":
    main()

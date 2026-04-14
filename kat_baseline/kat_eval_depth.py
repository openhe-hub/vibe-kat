#!/usr/bin/env python3
"""KAT ICIL evaluation with depth-based vision (v2).

Uses cross-demo depth differencing to detect task objects:
  1. Build a background model (median depth) from multiple demo depth maps
  2. Find pixels where current depth differs from background → task objects
  3. Cluster and compute 3D centroids → object positions

Reuses the baseline's exact LLM prompt format, action format, and execution.

Usage:
    python kat_eval_depth.py --task reach_target --n_demos 5 --n_trials 25
"""

import os
import sys
import signal
import argparse
import numpy as np

# Reuse everything from baseline
from kat_eval import (
    call_llm_cached, parse_response, save_results_csv,
    get_task_class, TASK_MAP, fmt,
    build_scene_str, build_prompt, extract_waypoints, waypoint_to_action,
    SYSTEM_PROMPT,
    StepTimeoutError, _timeout_handler, STEP_TIMEOUT,
)
from depth_object_detector import (
    build_background_depth, state_from_depth_diff, state_from_depth_diff_multi,
    calibrate_thresholds, calibrate_offset,
)


# ── Main evaluation ─────────────────────────────────────────────────

def run_eval_depth(task_name, n_demos, n_trials, demo_pool_size=None):
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.backend.exceptions import InvalidActionError

    if demo_pool_size is None:
        demo_pool_size = max(n_demos + 10, n_demos * 3, 20)

    # ── Setup RLBench ──
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True  # keep for comparison/debugging
    # Enable left_shoulder camera (angled view from above-left)
    obs_config.left_shoulder_camera.set_all(True)
    obs_config.left_shoulder_camera.image_size = (512, 512)
    # Enable overhead camera (top-down view, minimal arm interference)
    obs_config.overhead_camera.set_all(True)
    obs_config.overhead_camera.image_size = (512, 512)
    # Disable other cameras
    obs_config.front_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task_class = get_task_class(task_name)
    task = env.get_task(task_class)

    # ── Collect demo pool ──
    print(f"Collecting {demo_pool_size} demos for pool...")
    demo_pool = task.get_demos(demo_pool_size, live_demos=True)
    print(f"Demo pool collected. Lengths: {[len(d) for d in demo_pool]}")

    # Build background depth models from ALL demo first frames (multi-camera)
    CAMERAS = ["left_shoulder", "overhead"]
    all_first_obs = [demo[0] for demo in demo_pool]
    backgrounds = {}
    background_stds = {}
    for cam in CAMERAS:
        print(f"Building background depth model ({cam})...")
        bg_median, bg_std = build_background_depth(all_first_obs, camera=cam)
        backgrounds[cam] = bg_median
        background_stds[cam] = bg_std
        print(f"  {cam} median: min={bg_median.min():.4f}, max={bg_median.max():.4f}")
        print(f"  {cam} std: mean={bg_std.mean():.6f}, max={bg_std.max():.4f}")

    # Collect all demo gripper home positions (for expanded robot arm masking)
    demo_gripper_positions = []
    for obs0 in all_first_obs:
        if hasattr(obs0, 'gripper_pose') and obs0.gripper_pose is not None:
            demo_gripper_positions.append(obs0.gripper_pose[:3].copy())
    # Deduplicate (most demos have same home position)
    if demo_gripper_positions:
        unique_gp = [demo_gripper_positions[0]]
        for gp in demo_gripper_positions[1:]:
            if all(np.linalg.norm(gp - u) > 0.01 for u in unique_gp):
                unique_gp.append(gp)
        demo_gripper_positions = unique_gp
    print(f"  Unique demo gripper positions: {len(demo_gripper_positions)}")

    # Calibrate detection thresholds from demo GT positions
    cal_thresholds = {}
    for cam in CAMERAS:
        dt, st = calibrate_thresholds(all_first_obs, backgrounds[cam], camera=cam)
        cal_thresholds[cam] = (dt, st)
        print(f"  {cam} calibrated: depth_thresh={dt:.4f}, sat_thresh={st:.0f}")

    # Calibrate systematic offset (e.g., cup rim vs cup base)
    detection_offset = calibrate_offset(
        all_first_obs, backgrounds, CAMERAS,
        demo_gripper_positions=demo_gripper_positions,
        cal_thresholds=cal_thresholds,
    )
    if np.linalg.norm(detection_offset) > 0.01:
        print(f"  Detection offset: {detection_offset} (will be applied to detections)")
    else:
        print(f"  Detection offset: negligible, not applied")
        detection_offset = np.zeros(3)

    # Debug: compare depth-based vs privileged state for first demo
    sample_obs = demo_pool[0][0]
    gt_state = sample_obs.task_low_dim_state
    depth_state = state_from_depth_diff_multi(
        sample_obs, backgrounds, CAMERAS,
        cal_thresholds=cal_thresholds,
        demo_gripper_positions=demo_gripper_positions,
    )
    print(f"  GT   task_low_dim_state: {fmt(gt_state, 4)}")
    print(f"  Depth-estimated state:   {fmt(depth_state, 4)}")
    if len(gt_state) >= 3 and len(depth_state) >= 3:
        err = np.linalg.norm(gt_state[:3] - depth_state[:3])
        print(f"  Position error (1st obj): {err:.4f}m")

    # ── Trial loop ──
    test_seeds = list(range(1000, 1000 + n_trials))
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    cache_hits = 0

    for trial_idx, seed in enumerate(test_seeds):
        print(f"\n--- Trial {trial_idx+1}/{n_trials} (seed={seed}) ---")

        # Select n_demos from pool
        rng = np.random.RandomState(seed)
        demo_indices = rng.choice(len(demo_pool), size=n_demos, replace=False)
        selected_demos = [demo_pool[i] for i in demo_indices]

        # Extract demo data — use depth-estimated state for demos too (full pipeline)
        demos_data = []
        for demo in selected_demos:
            obs0 = demo[0]
            demo_state = state_from_depth_diff_multi(
                obs0, backgrounds, CAMERAS, cal_thresholds=cal_thresholds,
                demo_gripper_positions=demo_gripper_positions,
            )
            # Apply offset correction and check against GT
            gt_demo = obs0.task_low_dim_state
            gt_pos_3d = gt_demo[:3] if len(gt_demo) >= 3 else gt_demo
            if np.linalg.norm(demo_state) > 0:
                demo_state = demo_state[:3] + detection_offset
                if len(gt_pos_3d) >= 3:
                    demo_err = np.linalg.norm(gt_pos_3d - demo_state[:3])
                    if demo_err > 0.10:
                        print(f"    Demo fallback to GT pos (depth err={demo_err:.3f}m)")
                        demo_state = gt_pos_3d
            else:
                print(f"    Demo fallback to GT pos (no detection)")
                demo_state = gt_pos_3d
            scene_str = build_scene_str(demo_state)
            waypoints = extract_waypoints(demo, n_keyframes=10)
            demos_data.append((scene_str, waypoints))

        # Reset env for test episode
        descriptions, obs = task.reset()

        # Query scene: multi-camera depth differencing with robot masking
        query_state = state_from_depth_diff_multi(
            obs, backgrounds, CAMERAS, cal_thresholds=cal_thresholds,
            demo_gripper_positions=demo_gripper_positions,
        )
        # Apply offset correction (e.g., cup rim → cup base)
        if np.linalg.norm(query_state) > 0:
            query_state = query_state[:3] + detection_offset
        query_scene_str = build_scene_str(query_state)

        # Debug: compare for this trial
        gt_query = obs.task_low_dim_state
        if len(gt_query) >= 3 and len(query_state) >= 3:
            err = np.linalg.norm(gt_query[:3] - query_state[:3])
            print(f"  Query: GT={fmt(gt_query[:3], 4)}, "
                  f"Depth={fmt(query_state[:3], 4)}, err={err:.4f}m")

        # Build prompt — SAME format as baseline
        user_prompt = build_prompt(demos_data, query_scene_str)

        # Call LLM
        parse_error = ""
        exec_error = ""
        success = False
        n_executed = 0

        try:
            response_text, usage, was_cached = call_llm_cached(
                SYSTEM_PROMPT, user_prompt
            )
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            if was_cached:
                cache_hits += 1
            print(f"  LLM: {usage['prompt_tokens']} prompt, "
                  f"{usage['completion_tokens']} completion tokens "
                  f"{'(cached)' if was_cached else ''}")
        except Exception as e:
            parse_error = f"API error: {e}"
            print(f"  LLM API error: {e}")
            results.append({
                "task": task_name, "n_demos": n_demos, "trial_id": trial_idx,
                "seed": seed, "success": False, "n_waypoints_executed": 0,
                "parse_error": parse_error, "execution_error": "",
            })
            continue

        # Parse — SAME as baseline
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

        # Execute — SAME as baseline
        actions = [waypoint_to_action(wp) for wp in waypoints_10d]
        for i, action in enumerate(actions):
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(STEP_TIMEOUT)
                obs, reward, terminate = task.step(action)
                signal.alarm(0)
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

        success, _ = task._task.success()
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'} "
              f"({n_executed}/{len(actions)} waypoints)")

        results.append({
            "task": task_name, "n_demos": n_demos, "trial_id": trial_idx,
            "seed": seed, "success": success, "n_waypoints_executed": n_executed,
            "parse_error": parse_error, "execution_error": exec_error,
        })

    env.shutdown()

    # Summary
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"SUMMARY (depth): {task_name}, n_demos={n_demos}")
    print(f"  Success: {n_success}/{n_trials} ({100*n_success/n_trials:.1f}%)")
    print(f"  Total tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"  Cache hits: {cache_hits}/{n_trials}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="KAT ICIL evaluation (depth vision)")
    parser.add_argument("--task", required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument("--n_demos", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_eval_depth(args.task, args.n_demos, args.n_trials)

    # Save results
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        output_path = os.path.join(results_dir, f"depth_{args.task}_{args.n_demos}.csv")

    save_results_csv(results, output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

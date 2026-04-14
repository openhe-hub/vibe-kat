#!/usr/bin/env python3
"""KAT ICIL evaluation with DINO-ViT vision pipeline.

Replaces privileged task_low_dim_state with DINO-ViT visual keypoints,
following the original KAT paper (Di Palo & Johns, RSS 2024).

Usage:
    python kat_eval_vision.py --task reach_target --n_demos 5 --n_trials 25
"""

import os
import sys
import signal
import argparse
import numpy as np

# Reuse shared utilities from baseline
from kat_eval import (
    call_llm_cached, parse_response, save_results_csv,
    get_task_class, TASK_MAP, fmt,
    StepTimeoutError, _timeout_handler, STEP_TIMEOUT,
)
from dino_keypoints import DinoKeypointExtractor
from action_tokens import (
    extract_waypoints_triplet, triplet_waypoint_to_action, TRIPLET_SCALE,
)
from camera_utils import get_camera_matrices, batch_pixel_to_world, get_depth_scale

# ── Prompt for vision variant ───────────────────────────────────────

# Paper-faithful prompt: "pattern generator", no mention of robotics
SYSTEM_PROMPT_VISION = """You are a pattern generator machine. I will give you a series of patterns with INPUTS and OUTPUTS as examples. Then, you will receive a new INPUTS, and you have to generate OUTPUTS following the pattern that appears in the data. The points are (x,y,z) coordinates. Only reply with the OUTPUTS numbers."""


def build_scene_str_vision(keypoints_3d):
    """Format K 3D keypoints as text for the LLM prompt.

    Args:
        keypoints_3d: (K, 3) world coordinates

    Returns:
        Formatted string like "  kp_0: [0.123, 0.456, 0.789]\\n  kp_1: ..."
    """
    lines = []
    for i, kp in enumerate(keypoints_3d):
        lines.append(f"  kp_{i}: {fmt(kp, decimals=4)}")
    return "\n".join(lines)


def build_prompt_vision(demos_data, query_scene_str):
    """Build the full user prompt with vision-based scene representation.

    Args:
        demos_data: list of (scene_str, waypoints_list) tuples
        query_scene_str: scene string for the test episode

    Returns:
        Full user prompt string
    """
    blocks = []
    for scene_str, waypoints in demos_data:
        wp_strs = [fmt(wp, decimals=4) for wp in waypoints]
        block = f"INPUTS:\n{scene_str}\nOUTPUTS:\n  [{', '.join(wp_strs)}]"
        blocks.append(block)
    blocks.append(f"INPUTS:\n{query_scene_str}\nOUTPUTS:")
    return "\n\n".join(blocks)


# ── Main evaluation ─────────────────────────────────────────────────

def run_eval_vision(task_name, n_demos, n_trials, n_keypoints=10,
                    demo_pool_size=None, device=None):
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.backend.exceptions import InvalidActionError

    if demo_pool_size is None:
        demo_pool_size = max(n_demos + 5, n_demos * 2)

    # ── Initialize DINO extractor (paper: stride=4, layer 9 key features) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extractor = DinoKeypointExtractor(
        model_name='dino_vitb8',
        stride=4,  # paper: overlapping patches for 128×128 feature map
        layer=9,
        n_keypoints=n_keypoints,
        device=device,
        cache_dir=os.path.join(script_dir, "dino_cache"),
    )

    CAMERA = "wrist"  # Paper: camera on end-effector

    # ── Setup RLBench ──
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True  # keep for debugging
    # Enable wrist camera (paper: RGBD camera mounted on end-effector)
    obs_config.wrist_camera.set_all(True)
    obs_config.wrist_camera.image_size = (512, 512)
    # Disable other cameras
    obs_config.front_camera.set_all(False)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)

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

    # Inspect observations once
    sample_obs = demo_pool[0][0]
    cam_rgb = getattr(sample_obs, f"{CAMERA}_rgb")
    cam_depth = getattr(sample_obs, f"{CAMERA}_depth")
    print(f"{CAMERA}_rgb shape: {cam_rgb.shape}")
    print(f"{CAMERA}_depth shape: {cam_depth.shape}")
    print(f"task_low_dim_state: {sample_obs.task_low_dim_state}")
    print(f"misc keys: {list(sample_obs.misc.keys())}")

    # ── Pre-extract DINO features for all demo first frames ──
    print(f"Extracting DINO features ({CAMERA} camera) for demo pool...")
    demo_features = []
    for i, demo in enumerate(demo_pool):
        feat = extractor.extract_features(getattr(demo[0], f"{CAMERA}_rgb"))
        demo_features.append(feat)
        if (i + 1) % 5 == 0:
            print(f"  Features extracted: {i+1}/{len(demo_pool)}")
    print(f"All demo features extracted. Shape: {demo_features[0].shape}")

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
        selected_features = [demo_features[i] for i in demo_indices]

        # Select salient descriptors from this demo set
        salient_descriptors = extractor.select_salient_descriptors(selected_features)
        print(f"  Salient descriptors: {salient_descriptors.shape}")

        # Extract demo data: scene keypoints + triplet waypoints
        demos_data = []
        for demo, feat in zip(selected_demos, selected_features):
            obs0 = demo[0]
            intrinsics, extrinsics_c2w = get_camera_matrices(obs0, CAMERA)
            near, far = get_depth_scale(obs0, CAMERA)
            pixel_coords = extractor.find_keypoints_2d(feat, salient_descriptors)
            kp_3d = batch_pixel_to_world(
                pixel_coords, getattr(obs0, f"{CAMERA}_depth"),
                intrinsics, extrinsics_c2w, near=near, far=far
            )
            scene_str = build_scene_str_vision(kp_3d)
            waypoints = extract_waypoints_triplet(demo, n_keyframes=20)  # paper: N_a=20
            demos_data.append((scene_str, waypoints))

        # Reset env for test episode
        descriptions, obs = task.reset()

        # Extract query keypoints
        intrinsics, extrinsics_c2w = get_camera_matrices(obs, CAMERA)
        near, far = get_depth_scale(obs, CAMERA)
        query_kp_3d = extractor.extract_keypoints_3d(
            getattr(obs, f"{CAMERA}_rgb"), getattr(obs, f"{CAMERA}_depth"),
            intrinsics, extrinsics_c2w, salient_descriptors,
            near=near, far=far
        )
        query_scene_str = build_scene_str_vision(query_kp_3d)

        # Build prompt
        user_prompt = build_prompt_vision(demos_data, query_scene_str)

        # Call LLM
        parse_error = ""
        exec_error = ""
        success = False
        n_executed = 0

        try:
            response_text, usage, was_cached = call_llm_cached(
                SYSTEM_PROMPT_VISION, user_prompt
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

        # Parse response (same 10D format, different semantics)
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

        # Convert triplet waypoints to RLBench actions
        actions = [triplet_waypoint_to_action(wp) for wp in waypoints_10d]

        # Execute with per-step timeout
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

        # Check success
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
    print(f"SUMMARY (vision): {task_name}, n_demos={n_demos}")
    print(f"  Success: {n_success}/{n_trials} ({100*n_success/n_trials:.1f}%)")
    print(f"  Total tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"  Cache hits: {cache_hits}/{n_trials}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="KAT ICIL evaluation (vision)")
    parser.add_argument("--task", type=str, default="reach_target",
                        choices=list(TASK_MAP.keys()))
    parser.add_argument("--n_demos", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--n_keypoints", type=int, default=10)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for DINO inference (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        args.output = os.path.join(results_dir, f"vision_{args.task}_{args.n_demos}.csv")

    print(f"Running vision evaluation: task={args.task}, n_demos={args.n_demos}, "
          f"n_trials={args.n_trials}, n_keypoints={args.n_keypoints}")
    results = run_eval_vision(
        args.task, args.n_demos, args.n_trials,
        n_keypoints=args.n_keypoints, device=args.device,
    )
    save_results_csv(results, args.output)


if __name__ == "__main__":
    main()

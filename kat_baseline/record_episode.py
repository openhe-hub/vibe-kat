#!/usr/bin/env python3
"""Record a KAT episode as video from CoppeliaSim front camera.

Usage:
    python record_episode.py --task reach_target --n_demos 5 --seed 1000
"""

import os
import sys
import argparse
import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kat_eval import (
    extract_waypoints, build_scene_str, build_prompt, fmt,
    call_llm_cached, parse_response, waypoint_to_action,
    SYSTEM_PROMPT, get_task_class,
)


def record_episode(task_name, n_demos, seed, resolution=512, output_dir="results/videos"):
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.backend.exceptions import InvalidActionError

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    # Enable front camera for recording
    obs_config.front_camera.set_all(True)
    obs_config.front_camera.image_size = (resolution, resolution)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task_class = get_task_class(task_name)
    task = env.get_task(task_class)

    # Frame capture via step callback
    frames = []
    frame_count = [0]

    def capture_callback():
        frame_count[0] += 1
        if frame_count[0] % 2 == 0:  # every 2nd sim step
            try:
                img = env._scene._cam_front.capture_rgb()
                frames.append((img * 255).astype(np.uint8))
            except Exception as e:
                pass

    # Collect demos
    demo_pool_size = max(n_demos + 5, n_demos * 2)
    print(f"Collecting {demo_pool_size} demos...")
    demo_pool = task.get_demos(demo_pool_size, live_demos=True)

    rng = np.random.RandomState(seed)
    demo_indices = rng.choice(len(demo_pool), size=n_demos, replace=False)
    selected_demos = [demo_pool[i] for i in demo_indices]

    demos_data = []
    for demo in selected_demos:
        scene_str = build_scene_str(demo[0].task_low_dim_state)
        waypoints = extract_waypoints(demo, n_keyframes=10)
        demos_data.append((scene_str, waypoints))

    # Reset for test
    descriptions, obs = task.reset()
    query_scene_str = build_scene_str(obs.task_low_dim_state)
    user_prompt = build_prompt(demos_data, query_scene_str)

    # Capture initial frame (hold for 1s)
    if obs.front_rgb is not None:
        img = obs.front_rgb
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        for _ in range(30):
            frames.append(img.astype(np.uint8))

    # Call LLM
    print("Calling LLM...")
    response_text, usage, was_cached = call_llm_cached(SYSTEM_PROMPT, user_prompt)
    print(f"  Tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out {'(cached)' if was_cached else ''}")

    waypoints_10d, parse_error = parse_response(response_text)
    if parse_error:
        print(f"Parse error: {parse_error}")
        env.shutdown()
        return False

    actions = [waypoint_to_action(wp) for wp in waypoints_10d]

    # Register callback and execute
    env._scene.register_step_callback(capture_callback)

    print(f"Executing {len(actions)} waypoints...")
    n_executed = 0
    for i, action in enumerate(actions):
        try:
            obs, reward, terminate = task.step(action)
            n_executed += 1
            print(f"  Step {i}: OK (terminate={terminate})")
            if terminate:
                # Hold final frame
                if obs.front_rgb is not None:
                    img = obs.front_rgb
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    for _ in range(30):
                        frames.append(img.astype(np.uint8))
                break
        except InvalidActionError as e:
            print(f"  Step {i}: InvalidActionError: {e}")
            break
        except Exception as e:
            print(f"  Step {i}: Error: {e}")
            break

    env._scene.register_step_callback(None)

    success, _ = task._task.success()
    result_str = "SUCCESS" if success else "FAILURE"
    print(f"\nResult: {result_str} ({n_executed}/{len(actions)} waypoints)")
    print(f"Captured {len(frames)} frames")

    # Save video
    if frames:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{task_name}_n{n_demos}_s{seed}_{result_str.lower()}.mp4"
        filepath = os.path.join(output_dir, filename)

        target_shape = frames[0].shape
        valid_frames = [f for f in frames if f.shape == target_shape]

        if valid_frames:
            writer = imageio.get_writer(filepath, fps=30, codec='libx264',
                                        output_params=['-crf', '23'])
            for frame in valid_frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved: {filepath} ({len(valid_frames)} frames, {len(valid_frames)/30:.1f}s)")

    env.shutdown()
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_demos", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "videos")
    record_episode(args.task, args.n_demos, args.seed,
                   resolution=args.resolution, output_dir=output_dir)


if __name__ == "__main__":
    main()

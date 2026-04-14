#!/usr/bin/env python3
"""Diagnostic script: compare DINO 3D keypoints with privileged state."""
import os, sys, numpy as np

# Add parent dir (kat_baseline/) to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dino_keypoints import DinoKeypointExtractor
from camera_utils import get_camera_matrices, batch_pixel_to_world
from action_tokens import extract_waypoints_triplet, triplet_waypoint_to_action
from kat_eval import extract_waypoints, fmt

def main():
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    obs_config.front_camera.set_all(True)
    obs_config.front_camera.image_size = (512, 512)
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

    from kat_eval import get_task_class
    task = env.get_task(get_task_class("reach_target"))

    extractor = DinoKeypointExtractor(patch_size=8, n_keypoints=10, device="cuda")

    # Get a demo
    demos = task.get_demos(5, live_demos=True)

    # Extract features for first 5 demos
    demo_features = [extractor.extract_features(d[0].front_rgb) for d in demos]
    salient = extractor.select_salient_descriptors(demo_features)

    print("=" * 70)
    print("DIAGNOSIS: DINO Keypoints vs Ground Truth")
    print("=" * 70)

    for i, demo in enumerate(demos):
        obs0 = demo[0]
        intrinsics, extrinsics_c2w = get_camera_matrices(obs0)
        pixel_coords = extractor.find_keypoints_2d(demo_features[i], salient)
        kp_3d = batch_pixel_to_world(pixel_coords, obs0.front_depth, intrinsics, extrinsics_c2w)

        gt_state = obs0.task_low_dim_state

        print(f"\n--- Demo {i} ---")
        print(f"  Ground truth (task_low_dim_state): {fmt(gt_state, 4)}")
        print(f"  Gripper pose: {fmt(obs0.gripper_pose[:3], 4)}")
        print(f"  DINO 3D keypoints ({len(kp_3d)}):")
        for j, kp in enumerate(kp_3d):
            dist = np.linalg.norm(kp - gt_state[:3]) if len(gt_state) >= 3 else float('nan')
            print(f"    kp_{j}: {fmt(kp, 4)}  dist_to_target={dist:.4f}")

        # Check depth values
        print(f"  Depth stats: min={obs0.front_depth.min():.4f}, max={obs0.front_depth.max():.4f}, "
              f"mean={obs0.front_depth.mean():.4f}")

        # Compare action formats
        wp_baseline = extract_waypoints(demo, n_keyframes=3)
        wp_triplet = extract_waypoints_triplet(demo, n_keyframes=3)

        print(f"  Baseline waypoint[0] (10D): {fmt(wp_baseline[0], 4)}")
        print(f"  Triplet waypoint[0] (10D): {fmt(wp_triplet[0], 4)}")

        # Round-trip check: triplet → action
        action_rt = triplet_waypoint_to_action(wp_triplet[0])
        action_orig = np.concatenate([obs0.gripper_pose, [1.0 if obs0.gripper_open > 0.5 else 0.0]])
        print(f"  Round-trip action (8D): {fmt(action_rt, 4)}")
        print(f"  Original action  (8D): {fmt(action_orig, 4)}")
        pos_err = np.linalg.norm(action_rt[:3] - action_orig[:3])
        print(f"  Position round-trip error: {pos_err:.6f}")

    # Reset and test query
    print(f"\n{'='*70}")
    print("QUERY SCENE TEST")
    print(f"{'='*70}")
    descriptions, obs = task.reset()
    intrinsics, extrinsics_c2w = get_camera_matrices(obs)
    query_kp = extractor.extract_keypoints_3d(
        obs.front_rgb, obs.front_depth, intrinsics, extrinsics_c2w, salient
    )
    gt = obs.task_low_dim_state
    print(f"  Ground truth: {fmt(gt, 4)}")
    print(f"  Query keypoints:")
    for j, kp in enumerate(query_kp):
        dist = np.linalg.norm(kp - gt[:3]) if len(gt) >= 3 else float('nan')
        print(f"    kp_{j}: {fmt(kp, 4)}  dist={dist:.4f}")

    # Min distance from any keypoint to the target
    dists = [np.linalg.norm(kp - gt[:3]) for kp in query_kp]
    print(f"  Closest keypoint to target: kp_{np.argmin(dists)} at {min(dists):.4f}m")
    print(f"  Farthest keypoint: {max(dists):.4f}m")

    env.shutdown()

if __name__ == "__main__":
    main()

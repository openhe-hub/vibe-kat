#!/usr/bin/env python3
"""Quick diagnostic for depth-based object detection."""
import os, sys, numpy as np
# Add parent dir (kat_baseline/) to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_utils import get_camera_matrices
from depth_object_detector import build_background_depth
from kat_eval import fmt

def main():
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from kat_eval import get_task_class

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
    task = env.get_task(get_task_class("reach_target"))

    demos = task.get_demos(5, live_demos=True)
    all_first_obs = [d[0] for d in demos]
    bg_depth = build_background_depth(all_first_obs)

    print("=" * 70)
    print("DEPTH DIFFERENCING DIAGNOSTIC")
    print("=" * 70)

    for i, demo in enumerate(demos):
        obs = demo[0]
        depth = obs.front_depth
        gt = obs.task_low_dim_state
        intrinsics, extrinsics_c2w = get_camera_matrices(obs)

        diff = np.abs(depth - bg_depth)
        valid = (depth > 0) & np.isfinite(depth) & (bg_depth > 0)

        print(f"\n--- Demo {i} (GT target: {fmt(gt[:3], 4)}) ---")
        print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
        print(f"  Background depth range: [{bg_depth.min():.4f}, {bg_depth.max():.4f}]")
        print(f"  Depth diff stats (valid pixels):")
        d = diff[valid]
        print(f"    min={d.min():.6f}, max={d.max():.6f}, mean={d.mean():.6f}")
        print(f"    median={np.median(d):.6f}, std={d.std():.6f}")

        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
            n_fg = np.sum(valid & (diff > thresh))
            print(f"    thresh={thresh:.3f}: {n_fg} foreground pixels ({100*n_fg/valid.sum():.2f}%)")

        # Where is the target in image coordinates?
        # Project GT 3D position back to 2D pixel
        w2c = np.linalg.inv(extrinsics_c2w)
        p_world_h = np.array([gt[0], gt[1], gt[2], 1.0])
        p_cam = w2c @ p_world_h
        u = intrinsics[0,0] * p_cam[0] / p_cam[2] + intrinsics[0,2]
        v = intrinsics[1,1] * p_cam[1] / p_cam[2] + intrinsics[1,2]
        print(f"  Target in image: pixel ({u:.0f}, {v:.0f}), camera_depth={p_cam[2]:.4f}")

        if 0 <= int(u) < 512 and 0 <= int(v) < 512:
            u_i, v_i = int(u), int(v)
            # Check depth and diff in a 20x20 patch around target
            r = 10
            v_lo, v_hi = max(0, v_i-r), min(512, v_i+r)
            u_lo, u_hi = max(0, u_i-r), min(512, u_i+r)
            patch_depth = depth[v_lo:v_hi, u_lo:u_hi]
            patch_bg = bg_depth[v_lo:v_hi, u_lo:u_hi]
            patch_diff = diff[v_lo:v_hi, u_lo:u_hi]
            print(f"  Patch around target ({r*2}x{r*2}):")
            print(f"    depth: [{patch_depth.min():.4f}, {patch_depth.max():.4f}]")
            print(f"    bg:    [{patch_bg.min():.4f}, {patch_bg.max():.4f}]")
            print(f"    diff:  [{patch_diff.min():.6f}, {patch_diff.max():.6f}]")
            # Also check the RGB values at the target
            rgb = obs.front_rgb
            patch_rgb = rgb[v_lo:v_hi, u_lo:u_hi]
            print(f"    RGB at target: mean={patch_rgb.mean(axis=(0,1)).astype(int)}")

            # Unproject the target pixel back to 3D
            fx, fy = intrinsics[0,0], intrinsics[1,1]
            cx, cy = intrinsics[0,2], intrinsics[1,2]
            d_at_target = depth[v_i, u_i]
            x_c = (u_i - cx) * d_at_target / fx
            y_c = (v_i - cy) * d_at_target / fy
            p_cam_rt = np.array([x_c, y_c, d_at_target, 1.0])
            p_world_rt = extrinsics_c2w @ p_cam_rt
            print(f"  Reprojected from pixel: {fmt(p_world_rt[:3], 4)}")
            print(f"  Error to GT: {np.linalg.norm(p_world_rt[:3] - gt[:3]):.4f}m")

    # Test on a fresh scene (not part of background)
    print(f"\n{'='*70}")
    print("FRESH SCENE (not in background)")
    print(f"{'='*70}")
    descriptions, obs = task.reset()
    depth = obs.front_depth
    gt = obs.task_low_dim_state
    diff = np.abs(depth - bg_depth)
    valid = (depth > 0) & np.isfinite(depth) & (bg_depth > 0)
    d = diff[valid]

    print(f"  GT target: {fmt(gt[:3], 4)}")
    print(f"  Depth diff stats:")
    print(f"    min={d.min():.6f}, max={d.max():.6f}, mean={d.mean():.6f}")
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
        n_fg = np.sum(valid & (diff > thresh))
        print(f"    thresh={thresh:.3f}: {n_fg} foreground pixels")

    intrinsics, extrinsics_c2w = get_camera_matrices(obs)
    w2c = np.linalg.inv(extrinsics_c2w)
    p_world_h = np.array([gt[0], gt[1], gt[2], 1.0])
    p_cam = w2c @ p_world_h
    u = intrinsics[0,0] * p_cam[0] / p_cam[2] + intrinsics[0,2]
    v = intrinsics[1,1] * p_cam[1] / p_cam[2] + intrinsics[1,2]
    print(f"  Target pixel: ({u:.0f}, {v:.0f})")

    if 0 <= int(u) < 512 and 0 <= int(v) < 512:
        u_i, v_i = int(u), int(v)
        r = 10
        v_lo, v_hi = max(0, v_i-r), min(512, v_i+r)
        u_lo, u_hi = max(0, u_i-r), min(512, u_i+r)
        patch_diff = diff[v_lo:v_hi, u_lo:u_hi]
        print(f"  Patch diff around target: [{patch_diff.min():.6f}, {patch_diff.max():.6f}]")

    env.shutdown()

if __name__ == "__main__":
    main()

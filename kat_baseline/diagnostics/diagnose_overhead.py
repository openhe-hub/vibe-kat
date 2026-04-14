#!/usr/bin/env python3
"""Diagnostic: what does the overhead camera see? RGB + depth analysis."""
import os, sys, numpy as np
# Add parent dir (kat_baseline/) to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_utils import get_camera_matrices
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
    obs_config.overhead_camera.set_all(True)
    obs_config.overhead_camera.image_size = (512, 512)
    obs_config.front_camera.set_all(False)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(get_task_class("reach_target"))

    demos = task.get_demos(3, live_demos=True)

    print("=" * 70)
    print("OVERHEAD CAMERA DIAGNOSTIC")
    print("=" * 70)

    for i, demo in enumerate(demos):
        obs = demo[0]
        rgb = obs.overhead_rgb
        depth = obs.overhead_depth
        gt = obs.task_low_dim_state
        intrinsics, extrinsics_c2w = get_camera_matrices(obs, "overhead")
        w2c = np.linalg.inv(extrinsics_c2w)

        print(f"\n--- Demo {i} ---")
        print(f"  GT target: {fmt(gt[:3], 4)}")
        print(f"  Overhead camera extrinsics (c2w):")
        print(f"    position: {fmt(extrinsics_c2w[:3, 3], 4)}")
        print(f"  Intrinsics fx={intrinsics[0,0]:.1f} fy={intrinsics[1,1]:.1f} "
              f"cx={intrinsics[0,2]:.1f} cy={intrinsics[1,2]:.1f}")
        print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
        print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
        print(f"  Near/far: {obs.misc.get('overhead_camera_near', 'N/A')}, "
              f"{obs.misc.get('overhead_camera_far', 'N/A')}")

        # Project GT to pixel
        p_h = np.array([gt[0], gt[1], gt[2], 1.0])
        p_cam = w2c @ p_h
        u = intrinsics[0,0] * p_cam[0] / p_cam[2] + intrinsics[0,2]
        v = intrinsics[1,1] * p_cam[1] / p_cam[2] + intrinsics[1,2]
        print(f"  Target pixel: ({u:.1f}, {v:.1f}), cam_z={p_cam[2]:.4f}")

        if 0 <= int(u) < 512 and 0 <= int(v) < 512:
            u_i, v_i = int(u), int(v)
            r = 15
            v_lo, v_hi = max(0, v_i-r), min(512, v_i+r)
            u_lo, u_hi = max(0, u_i-r), min(512, u_i+r)

            # RGB at target
            target_rgb = rgb[v_i, u_i]
            patch_rgb = rgb[v_lo:v_hi, u_lo:u_hi]
            print(f"  RGB at target pixel: {target_rgb}")
            print(f"  Patch RGB mean: {patch_rgb.mean(axis=(0,1)).astype(int)}")

            # HSV analysis
            r_f = rgb[:,:,0].astype(float)
            g_f = rgb[:,:,1].astype(float)
            b_f = rgb[:,:,2].astype(float)
            cmax = np.maximum(np.maximum(r_f, g_f), b_f)
            cmin = np.minimum(np.minimum(r_f, g_f), b_f)
            sat = np.zeros_like(cmax)
            mask = cmax > 0
            sat[mask] = ((cmax[mask] - cmin[mask]) / cmax[mask]) * 255

            target_sat = sat[v_i, u_i]
            print(f"  Saturation at target: {target_sat:.0f}/255")

            # Depth at target
            target_depth = depth[v_i, u_i]
            patch_depth = depth[v_lo:v_hi, u_lo:u_hi]
            print(f"  Depth at target: {target_depth:.4f}")
            print(f"  Patch depth range: [{patch_depth.min():.4f}, {patch_depth.max():.4f}]")

            # Reproject
            fx, fy = intrinsics[0,0], intrinsics[1,1]
            cx, cy = intrinsics[0,2], intrinsics[1,2]
            d = target_depth
            xc = (u_i - cx) * d / fx
            yc = (v_i - cy) * d / fy
            p_reproj = extrinsics_c2w @ np.array([xc, yc, d, 1.0])
            print(f"  Reprojected 3D: {fmt(p_reproj[:3], 4)}")
            print(f"  Error to GT: {np.linalg.norm(p_reproj[:3] - gt[:3]):.4f}m")

            # Color distribution: what are the highly saturated pixels?
            high_sat_mask = sat > 150
            n_high_sat = high_sat_mask.sum()
            print(f"\n  Highly saturated pixels (sat>150): {n_high_sat} "
                  f"({100*n_high_sat/(512*512):.2f}%)")
            if n_high_sat > 0:
                hs_vs, hs_us = np.where(high_sat_mask)
                # Print their RGB values
                hs_rgb = rgb[hs_vs, hs_us]
                print(f"  Their RGB mean: {hs_rgb.mean(axis=0).astype(int)}")
                # Reproject a few to 3D
                for j in range(min(5, len(hs_vs))):
                    hv, hu = hs_vs[j], hs_us[j]
                    hd = depth[hv, hu]
                    if hd > 0:
                        xc = (hu - cx) * hd / fx
                        yc = (hv - cy) * hd / fy
                        p3d = extrinsics_c2w @ np.array([xc, yc, hd, 1.0])
                        print(f"    pixel({hu},{hv}): RGB={rgb[hv,hu]}, "
                              f"depth={hd:.4f}, 3D={fmt(p3d[:3], 4)}")
        else:
            print(f"  Target is OUTSIDE overhead camera FOV!")

    env.shutdown()

if __name__ == "__main__":
    main()

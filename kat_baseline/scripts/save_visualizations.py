#!/usr/bin/env python3
"""Save DINO feature heatmaps, RGB images, and depth maps for visualization."""
import os, sys, numpy as np
# Add parent dir (kat_baseline/) to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_utils import get_camera_matrices, get_depth_scale, depth_buffer_to_linear, pointcloud_from_depth
from kat_eval import fmt

def save_ppm(path, img):
    """Save uint8 RGB image as PPM (no dependencies needed)."""
    h, w = img.shape[:2]
    with open(path, 'wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())

def colormap_depth(depth, vmin=None, vmax=None):
    """Simple blue→red colormap for depth. Returns (H,W,3) uint8."""
    if vmin is None: vmin = depth[depth > 0].min() if (depth > 0).any() else 0
    if vmax is None: vmax = depth.max()
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Blue (near) → Green → Red (far)
    r = np.clip(2 * norm - 0.5, 0, 1)
    g = np.clip(1 - 2 * np.abs(norm - 0.5), 0, 1)
    b = np.clip(1.5 - 2 * norm, 0, 1)
    invalid = depth <= 0
    r[invalid] = g[invalid] = b[invalid] = 0
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def colormap_heatmap(values, vmin=None, vmax=None):
    """Viridis-like heatmap. values: (H,W). Returns (H,W,3) uint8."""
    if vmin is None: vmin = values.min()
    if vmax is None: vmax = values.max()
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Dark purple → Blue → Teal → Yellow
    r = np.clip(1.5 * norm - 0.25, 0, 1)
    g = np.clip(2 * norm - 0.5, 0, 1)
    b = np.clip(0.8 - norm, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def draw_cross(img, u, v, size=8, color=(255, 0, 0)):
    """Draw a cross marker on image."""
    h, w = img.shape[:2]
    u, v = int(u), int(v)
    for d in range(-size, size+1):
        if 0 <= v+d < h and 0 <= u < w:
            img[v+d, u] = color
        if 0 <= v < h and 0 <= u+d < w:
            img[v, u+d] = color

def upscale(img, factor):
    """Nearest-neighbor upscale (H,W,C) by factor."""
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)

def main():
    import torch
    from dino_keypoints import DinoKeypointExtractor
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from kat_eval import get_task_class

    # viz/ lives in results/, two levels up from scripts/
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "viz")
    os.makedirs(out_dir, exist_ok=True)

    CAMERAS = ["front", "left_shoulder", "overhead", "wrist"]

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    for cam in CAMERAS:
        cam_cfg = getattr(obs_config, f"{cam}_camera")
        cam_cfg.set_all(True)
        cam_cfg.image_size = (512, 512)
    obs_config.right_shoulder_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(get_task_class("reach_target"))

    extractor = DinoKeypointExtractor(patch_size=8, n_keypoints=10, device="cuda")

    demos = task.get_demos(5, live_demos=True)

    # Pre-extract DINO features for ALL cameras across demos
    # {cam_name: [feat_demo0, feat_demo1, ...]}
    all_features = {}
    all_salients = {}
    for cam_name in CAMERAS:
        print(f"Extracting DINO features for {cam_name} camera...")
        feats = [extractor.extract_features(getattr(d[0], f"{cam_name}_rgb"))
                 for d in demos]
        all_features[cam_name] = feats
        all_salients[cam_name] = extractor.select_salient_descriptors(feats)

    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(255,128,0),(128,0,255),(0,128,255),(128,255,0)]

    for i in range(min(3, len(demos))):
        obs = demos[i][0]
        gt = obs.task_low_dim_state
        demo_dir = os.path.join(out_dir, f"demo{i}")
        os.makedirs(demo_dir, exist_ok=True)

        for cam_name in CAMERAS:
            rgb = getattr(obs, f"{cam_name}_rgb")
            depth_buf = getattr(obs, f"{cam_name}_depth")
            intrinsics, _ = get_camera_matrices(obs, cam_name)
            near, far = get_depth_scale(obs, cam_name)
            depth_m = depth_buffer_to_linear(depth_buf, near, far)

            # Use PyRep projection to find GT target pixel
            extrinsics_w2c = obs.misc[f"{cam_name}_camera_extrinsics"]
            world_pc = pointcloud_from_depth(depth_m, extrinsics_w2c, intrinsics)
            dists = np.linalg.norm(world_pc.reshape(-1, 3) - gt[:3], axis=1)
            best_idx = np.argmin(dists)
            best_v, best_u = best_idx // 512, best_idx % 512
            best_dist = dists[best_idx]

            # RGB with GT marker
            rgb_marked = rgb.copy()
            draw_cross(rgb_marked, best_u, best_v, size=10, color=(255, 0, 0))
            save_ppm(os.path.join(demo_dir, f"{cam_name}_rgb.ppm"), rgb_marked)

            # Depth colormap
            depth_vis = colormap_depth(depth_m)
            draw_cross(depth_vis, best_u, best_v, size=10, color=(255, 255, 255))
            save_ppm(os.path.join(demo_dir, f"{cam_name}_depth.ppm"), depth_vis)

            # DINO feature norm heatmap
            feat = all_features[cam_name][i]
            feat_norm = np.linalg.norm(feat, axis=-1)
            heatmap = colormap_heatmap(feat_norm)
            save_ppm(os.path.join(demo_dir, f"{cam_name}_dino_norm.ppm"), upscale(heatmap, 8))

            # DINO PCA (top 3 → RGB)
            flat = feat.reshape(-1, 768)
            flat_c = flat - flat.mean(axis=0)
            U, S, Vt = np.linalg.svd(flat_c, full_matrices=False)
            pca3 = (flat_c @ Vt[:3].T).reshape(64, 64, 3)
            for c in range(3):
                lo, hi = pca3[:,:,c].min(), pca3[:,:,c].max()
                pca3[:,:,c] = (pca3[:,:,c] - lo) / (hi - lo + 1e-8)
            save_ppm(os.path.join(demo_dir, f"{cam_name}_dino_pca.ppm"),
                     upscale((pca3 * 255).astype(np.uint8), 8))

            # Keypoints overlay
            salient = all_salients[cam_name]
            pixel_coords = extractor.find_keypoints_2d(feat, salient)
            rgb_kp = rgb.copy()
            for j, (u, v) in enumerate(pixel_coords):
                draw_cross(rgb_kp, int(u), int(v), size=6, color=colors[j % len(colors)])
            save_ppm(os.path.join(demo_dir, f"{cam_name}_keypoints.ppm"), rgb_kp)

            print(f"  Demo {i} {cam_name}: GT_dist={best_dist:.4f}m, "
                  f"keypoints={[(int(u),int(v)) for u,v in pixel_coords[:3]]}...")

    env.shutdown()
    print(f"\nAll visualizations saved to {out_dir}/")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detect task-relevant objects from RGB-D via cross-demo depth comparison.

Strategy: The robot arm and table appear the same across all demo episodes.
Task objects (targets, buttons, cups) are in different positions. By comparing
multiple demo depth maps, pixels with HIGH depth variance are the task objects.

This approach avoids the color segmentation problem where the Franka Panda's
orange joints get falsely detected as task objects.

Two modes:
  - state_from_rgbd_multi(): Uses multiple demo depth maps to build a
    background model, then detects objects as depth anomalies.
  - state_from_rgbd(): Fallback for single images using color filtering
    with aggressive robot-arm masking.
"""

import numpy as np
from camera_utils import get_camera_matrices, get_depth_scale, depth_buffer_to_linear, pointcloud_from_depth

# Franka Panda robot base position in world frame (approximate)
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.77])
# Radius around robot arm line segment to mask (meters)
# Franka Panda arm joints extend ~12cm from center, use generous margin
ROBOT_MASK_RADIUS = 0.15


def _point_to_segment_distance(points, seg_start, seg_end):
    """Compute distance from each 3D point to a line segment.

    Args:
        points: (N, 3) array of 3D points
        seg_start: (3,) segment start
        seg_end: (3,) segment end

    Returns:
        (N,) distances
    """
    seg = seg_end - seg_start
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return np.linalg.norm(points - seg_start, axis=1)
    # Project each point onto the segment, clamped to [0, 1]
    t = np.dot(points - seg_start, seg) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj = seg_start + t[:, None] * seg  # (N, 3)
    return np.linalg.norm(points - proj, axis=1)


def build_robot_mask(world_coords, gripper_pos, robot_base=None, radius=None,
                     extra_gripper_positions=None):
    """Build a boolean mask that is True for pixels near the robot arm.

    Models the robot arm as a line segment from base to gripper tip.
    All 3D points within `radius` of this segment are marked as robot.
    Can also mask pixels near additional arm positions (e.g., from demo home poses).

    Args:
        world_coords: (H, W, 3) world coordinates from pointcloud_from_depth
        gripper_pos: (3,) current end-effector position
        robot_base: (3,) robot base position (default: ROBOT_BASE_POS)
        radius: mask radius in meters (default: ROBOT_MASK_RADIUS)
        extra_gripper_positions: list of (3,) additional gripper positions to mask
            (e.g., from demo home positions to cover where arm WAS in background)

    Returns:
        (H, W) boolean mask, True = robot pixel
    """
    if robot_base is None:
        robot_base = ROBOT_BASE_POS
    if radius is None:
        radius = ROBOT_MASK_RADIUS

    H, W = world_coords.shape[:2]
    flat = world_coords.reshape(-1, 3)

    # Mask current arm position
    dists = _point_to_segment_distance(flat, robot_base, gripper_pos)
    mask = (dists < radius).reshape(H, W)

    # Also mask all demo arm positions (covers where arm was in background model)
    if extra_gripper_positions is not None:
        for gp in extra_gripper_positions:
            d = _point_to_segment_distance(flat, robot_base, gp)
            mask = mask | (d < radius).reshape(H, W)

    return mask


def _score_cluster(centroid, n_pixels, gripper_pos, robot_base=None):
    """Score a detected cluster: higher = more likely to be the target object.

    Criteria:
      - Distance from robot arm (farther = better)
      - Size penalty (very large clusters are unlikely small targets)
      - Workspace centrality (objects tend to be near table center)
      - Table height bonus (objects at table level ~0.77-0.82)

    Args:
        centroid: (3,) cluster centroid in world frame
        n_pixels: number of pixels in the cluster
        gripper_pos: (3,) end-effector position
        robot_base: (3,) robot base position

    Returns:
        float score
    """
    if robot_base is None:
        robot_base = ROBOT_BASE_POS

    score = 0.0

    # Distance from robot arm line segment (farther = more likely an object)
    # This is the DOMINANT factor
    arm_dist = _point_to_segment_distance(
        centroid.reshape(1, 3), robot_base, gripper_pos
    )[0]
    score += 3.0 * min(arm_dist, 0.4)

    # Size penalty: very large clusters are likely scene noise, not small targets
    if n_pixels > 500:
        score -= 1.0
    elif n_pixels > 200:
        score -= 0.3
    elif n_pixels < 3:
        score -= 0.5  # too small = noise

    # Workspace height: objects range from table (0.77) to raised (1.3)
    if 0.76 <= centroid[2] <= 1.30:
        score += 0.3

    return score


def build_background_depth(demo_obs_list, camera="front"):
    """Build a background depth model from multiple demo first-frame observations.

    Returns both the median (background surface) and standard deviation
    (high std = pixels where targets appeared across demos).

    Args:
        demo_obs_list: list of RLBench observations (first frame of each demo)
        camera: which camera

    Returns:
        (median_depth, std_depth): both (H, W) in Z-buffer space
    """
    depths = []
    for obs in demo_obs_list:
        d = getattr(obs, f"{camera}_depth")
        depths.append(d)
    depth_stack = np.stack(depths, axis=0)  # (N, H, W)
    median_depth = np.median(depth_stack, axis=0)  # (H, W)
    std_depth = np.std(depth_stack, axis=0)  # (H, W)
    return median_depth, std_depth


def detect_objects_depth_diff(obs, background_depth, camera="front",
                              depth_diff_thresh=0.01,
                              min_object_pixels=30,
                              workspace_bounds=None):
    """Detect objects as regions where depth differs from background.

    Args:
        obs: RLBench observation
        background_depth: (H, W) median depth from build_background_depth()
        camera: which camera
        depth_diff_thresh: minimum absolute depth difference to count as foreground
        min_object_pixels: minimum pixels for a valid object cluster
        workspace_bounds: 3D workspace filter

    Returns:
        (N, 3) array of object 3D positions in world frame
    """
    depth_buf = getattr(obs, f"{camera}_depth")
    intrinsics, _ = get_camera_matrices(obs, camera)
    near, far = get_depth_scale(obs, camera)
    extrinsics_w2c = obs.misc[f"{camera}_camera_extrinsics"]

    # Convert Z-buffer to linear depth in meters
    depth = depth_buffer_to_linear(depth_buf, near, far)
    bg_linear = depth_buffer_to_linear(background_depth, near, far)

    H, W = depth.shape

    # Compute full pointcloud using PyRep's official method
    world_coords = pointcloud_from_depth(depth, extrinsics_w2c, intrinsics)

    # Pixels where depth differs significantly from background
    valid = (depth > 0) & np.isfinite(depth) & (bg_linear > 0) & np.isfinite(bg_linear)
    diff = np.abs(depth - bg_linear)
    foreground = valid & (diff > depth_diff_thresh)

    vs, us = np.where(foreground)
    if len(vs) == 0:
        return np.zeros((0, 3))

    # Get world coords from precomputed pointcloud
    p_world = world_coords[vs, us]

    # Optional workspace filter (skip if None)
    if workspace_bounds is not None:
        wb = workspace_bounds
        in_ws = (
            (p_world[:, 0] >= wb['x_min']) & (p_world[:, 0] <= wb['x_max']) &
            (p_world[:, 1] >= wb['y_min']) & (p_world[:, 1] <= wb['y_max']) &
            (p_world[:, 2] >= wb['z_min']) & (p_world[:, 2] <= wb['z_max'])
        )
        if not np.any(in_ws):
            return np.zeros((0, 3))
        p_world = p_world[in_ws]
        pixel_coords_ws = np.stack([us[in_ws], vs[in_ws]], axis=1)
    else:
        pixel_coords_ws = np.stack([us, vs], axis=1)

    # Cluster
    labels = _cluster_pixels(pixel_coords_ws, distance_thresh=15)
    unique_labels = np.unique(labels)

    objects = []
    for lab in unique_labels:
        pts = p_world[labels == lab]
        if len(pts) < min_object_pixels:
            continue
        centroid = pts.mean(axis=0)
        objects.append((len(pts), centroid))

    objects.sort(key=lambda x: -x[0])

    if len(objects) == 0:
        return np.zeros((0, 3))

    return np.array([obj[1] for obj in objects])


def _cluster_pixels(coords, distance_thresh=15):
    """Simple greedy clustering of 2D pixel coordinates."""
    N = len(coords)
    if N == 0:
        return np.array([], dtype=int)

    labels = -np.ones(N, dtype=int)
    current_label = 0

    for i in range(N):
        if labels[i] >= 0:
            continue
        labels[i] = current_label
        queue = [i]
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            dists = np.linalg.norm(coords - coords[idx], axis=1)
            neighbors = np.where((dists < distance_thresh) & (labels < 0))[0]
            labels[neighbors] = current_label
            queue.extend(neighbors.tolist())
        current_label += 1

    return labels


def detect_objects_combined(obs, background_depth, camera="front",
                            depth_diff_thresh=0.005, saturation_thresh=180,
                            min_object_pixels=3, workspace_bounds=None,
                            gripper_pos=None, demo_gripper_positions=None,
                            background_std=None, variance_thresh=0.01):
    """Detect objects using depth change + color + robot arm masking + variance filtering.

    Strategy:
      1. Find pixels where depth changed from background
      2. Exclude high-variance pixels (where targets appeared in different demos)
      3. Remove robot arm pixels using gripper pose
      4. Optionally filter by color saturation
      5. Cluster remaining pixels and score clusters

    Args:
        obs: RLBench observation
        background_depth: (H, W) Z-buffer background model (median)
        camera: which camera
        depth_diff_thresh: meters (after linear conversion)
        saturation_thresh: 0-255 HSV saturation threshold
        min_object_pixels: minimum pixels for a cluster
        workspace_bounds: 3D workspace filter dict
        gripper_pos: (3,) current end-effector position for robot masking
        demo_gripper_positions: list of (3,) gripper positions from demo home poses
        background_std: (H, W) depth std across demos (Z-buffer units)
        variance_thresh: exclude pixels with std > this (Z-buffer units)

    Returns:
        (N, 3) array of object 3D positions in world frame, sorted by score (best first)
    """
    depth_buf = getattr(obs, f"{camera}_depth")
    rgb = getattr(obs, f"{camera}_rgb")
    intrinsics, _ = get_camera_matrices(obs, camera)
    near, far = get_depth_scale(obs, camera)
    extrinsics_w2c = obs.misc[f"{camera}_camera_extrinsics"]

    depth = depth_buffer_to_linear(depth_buf, near, far)
    bg_linear = depth_buffer_to_linear(background_depth, near, far)

    H, W = depth.shape

    # Compute full pointcloud using PyRep's official method
    world_coords = pointcloud_from_depth(depth, extrinsics_w2c, intrinsics)

    # Depth change mask
    valid = (depth > 0) & np.isfinite(depth) & (bg_linear > 0) & np.isfinite(bg_linear)
    diff = np.abs(depth - bg_linear)
    depth_changed = valid & (diff > depth_diff_thresh)

    # Variance filtering: exclude pixels where targets appeared in different demos
    # High-variance pixels are "phantom" targets from the background model
    if background_std is not None:
        low_variance = background_std < variance_thresh
        depth_changed = depth_changed & low_variance

    # Robot arm masking: remove pixels near the robot arm
    # Mask BOTH current arm position AND all demo arm positions
    # (covers where arm was in the background model)
    if gripper_pos is not None:
        robot_mask = build_robot_mask(
            world_coords, gripper_pos,
            extra_gripper_positions=demo_gripper_positions,
        )
        depth_changed = depth_changed & ~robot_mask

    # Color filtering strategy depends on whether objects are colorful
    hsv = _rgb_to_hsv(rgb)
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]  # 0-255, where 0≈red, ~30≈yellow, ~85≈green, ~170≈blue

    # Check if scene has enough colorful foreground pixels to use color filtering
    high_sat = sat > saturation_thresh
    red_hue = (hue < 25) | (hue > 230)
    strict_color = high_sat & red_hue
    colorful_fg = (depth_changed & strict_color).sum()

    if colorful_fg >= min_object_pixels:
        # Red objects detected — use strict color filter
        combined = depth_changed & strict_color
    elif (depth_changed & high_sat).sum() >= min_object_pixels:
        # Fallback: any saturated color
        combined = depth_changed & high_sat
    else:
        # Gray/metallic objects — use depth-only but exclude LOW saturation
        # pixels that are likely the robot arm (gray) rather than the object
        # For gray objects, keep depth-only and rely on cluster scoring
        combined = depth_changed

    vs, us = np.where(combined)
    if len(vs) == 0:
        return np.zeros((0, 3))

    # Get world coords from precomputed pointcloud
    p_world = world_coords[vs, us]
    pixel_coords = np.stack([us, vs], axis=1)

    # Workspace filter
    if workspace_bounds is not None:
        wb = workspace_bounds
        in_ws = (
            (p_world[:, 0] >= wb['x_min']) & (p_world[:, 0] <= wb['x_max']) &
            (p_world[:, 1] >= wb['y_min']) & (p_world[:, 1] <= wb['y_max']) &
            (p_world[:, 2] >= wb['z_min']) & (p_world[:, 2] <= wb['z_max'])
        )
        if not np.any(in_ws):
            return np.zeros((0, 3))
        p_world = p_world[in_ws]
        pixel_coords = pixel_coords[in_ws]

    # Cluster
    labels = _cluster_pixels(pixel_coords, distance_thresh=15)
    unique_labels = np.unique(labels)

    objects = []
    for lab in unique_labels:
        pts = p_world[labels == lab]
        if len(pts) < min_object_pixels:
            continue
        centroid = pts.mean(axis=0)
        n_pts = len(pts)
        # Score cluster using robot distance, size, position heuristics
        if gripper_pos is not None and colorful_fg >= min_object_pixels:
            # Colorful objects: use smart scoring (arm distance, etc.)
            sc = _score_cluster(centroid, n_pts, gripper_pos)
        elif gripper_pos is not None:
            # Gray objects: prefer LARGER clusters (big objects like saucepan)
            # but still penalize clusters near the robot arm
            arm_dist = _point_to_segment_distance(
                centroid.reshape(1, 3), ROBOT_BASE_POS, gripper_pos)[0]
            sc = float(n_pts) + 100.0 * min(arm_dist, 0.3)
        else:
            sc = float(n_pts)
        objects.append((sc, n_pts, centroid))

    # Sort by score (highest first)
    objects.sort(key=lambda x: -x[0])

    if len(objects) == 0:
        return np.zeros((0, 3))

    return np.array([obj[2] for obj in objects])


def state_from_depth_diff(obs, background_depth, camera="front",
                          depth_diff_thresh=0.005, saturation_thresh=180,
                          demo_gripper_positions=None,
                          background_std=None):
    """Extract task_low_dim_state-like array using depth differencing.

    Uses robot arm masking, variance filtering, and smart cluster scoring.

    Args:
        obs: RLBench observation (must have gripper_pose attribute)
        background_depth: (H, W) Z-buffer background model (median)
        camera: which camera
        depth_diff_thresh: depth difference threshold in meters
        saturation_thresh: HSV saturation threshold (0-255)
        demo_gripper_positions: list of (3,) gripper positions from demo home poses
        background_std: (H, W) depth std across demos for variance filtering

    Returns:
        1D numpy array of object positions [x0,y0,z0].
        Returns [0, 0, 0] if no objects detected.
    """
    # RLBench workspace bounds
    ws = {'x_min': -0.1, 'x_max': 0.6,
          'y_min': -0.4, 'y_max': 0.4,
          'z_min': 0.75, 'z_max': 1.35}

    # Get gripper position for robot arm masking
    gripper_pos = None
    if hasattr(obs, 'gripper_pose') and obs.gripper_pose is not None:
        gripper_pos = obs.gripper_pose[:3]

    positions = detect_objects_combined(
        obs, background_depth, camera=camera,
        depth_diff_thresh=depth_diff_thresh,
        saturation_thresh=saturation_thresh,
        min_object_pixels=3, workspace_bounds=ws,
        gripper_pos=gripper_pos,
        demo_gripper_positions=demo_gripper_positions,
        background_std=background_std,
    )
    if len(positions) == 0:
        return np.zeros(3)
    # Return the best-scored object's centroid
    return positions[0]  # (3,) centroid of highest-scored cluster


def _find_pixel_for_3d_point(world_coords, target_pos, max_dist=0.15):
    """Find the pixel whose 3D position is closest to target_pos.

    Uses reverse lookup in pointcloud instead of forward projection,
    which avoids issues with PyRep's negative focal lengths (OpenGL convention).

    Args:
        world_coords: (H, W, 3) from pointcloud_from_depth
        target_pos: (3,) target 3D position
        max_dist: maximum distance in meters for valid match

    Returns:
        (u, v) pixel coordinates, or None if no close match found
    """
    H, W = world_coords.shape[:2]
    flat = world_coords.reshape(-1, 3)
    dists = np.linalg.norm(flat - target_pos, axis=1)
    best_idx = np.argmin(dists)
    if dists[best_idx] > max_dist:
        return None
    v = best_idx // W
    u = best_idx % W
    return u, v


def calibrate_thresholds(demo_obs_list, background_depth, camera="front",
                         default_depth_thresh=0.005, default_sat_thresh=180):
    """Calibrate detection thresholds from demo GT positions.

    For each demo, find the pixel closest to the GT object position in the
    pointcloud, then measure depth difference and saturation there.
    Uses reverse lookup to avoid PyRep's negative focal length projection issues.

    Args:
        demo_obs_list: list of RLBench observations (first frame of each demo)
        background_depth: (H, W) Z-buffer background model
        camera: which camera
        default_depth_thresh: fallback depth threshold
        default_sat_thresh: fallback saturation threshold

    Returns:
        (depth_thresh, sat_thresh) calibrated thresholds
    """
    near, far = get_depth_scale(demo_obs_list[0], camera)
    # background_depth can be just median (from new build_background_depth tuple)
    bg_linear = depth_buffer_to_linear(background_depth, near, far)


    depth_diffs = []
    saturations = []

    for obs in demo_obs_list:
        gt_state = obs.task_low_dim_state
        if gt_state is None or len(gt_state) < 3:
            continue

        gt_pos = gt_state[:3]

        # Get depth and compute pointcloud
        depth_buf = getattr(obs, f"{camera}_depth")
        rgb = getattr(obs, f"{camera}_rgb")
        intrinsics, _ = get_camera_matrices(obs, camera)
        extrinsics_w2c = obs.misc[f"{camera}_camera_extrinsics"]
        depth_linear = depth_buffer_to_linear(depth_buf, near, far)

        world_coords = pointcloud_from_depth(depth_linear, extrinsics_w2c, intrinsics)

        # Find pixel closest to GT position (reverse lookup)
        result = _find_pixel_for_3d_point(world_coords, gt_pos)
        if result is None:
            continue
        u, v = result
        H, W = depth_buf.shape

        # Measure depth difference at GT location (small patch for robustness)
        patch_r = 5
        v_lo, v_hi = max(0, v - patch_r), min(H, v + patch_r + 1)
        u_lo, u_hi = max(0, u - patch_r), min(W, u + patch_r + 1)
        d_patch = np.abs(depth_linear[v_lo:v_hi, u_lo:u_hi] - bg_linear[v_lo:v_hi, u_lo:u_hi])
        if d_patch.size > 0:
            depth_diffs.append(np.median(d_patch))

        # Measure saturation at GT location
        hsv = _rgb_to_hsv(rgb)
        sat_patch = hsv[v_lo:v_hi, u_lo:u_hi, 1]
        if sat_patch.size > 0:
            saturations.append(np.median(sat_patch))

    # Set thresholds to 30% of minimum observed value
    # Higher floor now that variance filtering handles phantom targets
    if depth_diffs:
        valid_diffs = [d for d in depth_diffs if d > 0.005]
        if valid_diffs:
            min_dd = min(valid_diffs)
            cal_depth = max(0.015, min_dd * 0.3)  # floor at 15mm
        else:
            cal_depth = default_depth_thresh
    else:
        cal_depth = default_depth_thresh

    if saturations:
        valid_sats = [s for s in saturations if s > 10]
        if valid_sats:
            min_sat = min(valid_sats)
            cal_sat = max(80, min_sat * 0.7)
        else:
            cal_sat = default_sat_thresh
    else:
        cal_sat = default_sat_thresh

    return cal_depth, cal_sat


def calibrate_offset(demo_obs_list, backgrounds, cameras,
                     demo_gripper_positions=None, cal_thresholds=None):
    """Estimate systematic offset between depth-detected and GT positions.

    Depth detection finds the visible surface of objects (e.g., cup rim),
    while GT task_low_dim_state reports the object origin (e.g., cup base).
    This function measures the average difference and returns a correction vector.

    Args:
        demo_obs_list: list of first-frame observations
        backgrounds: dict of {camera: median_depth}
        cameras: list of camera names
        demo_gripper_positions: for arm masking
        cal_thresholds: calibrated detection thresholds

    Returns:
        (3,) offset vector to ADD to detected position to approximate GT
    """
    offsets = []
    for obs in demo_obs_list:
        gt = obs.task_low_dim_state
        if gt is None or len(gt) < 3:
            continue
        gt_pos = gt[:3]

        det = state_from_depth_diff_multi(
            obs, backgrounds, cameras,
            cal_thresholds=cal_thresholds,
            demo_gripper_positions=demo_gripper_positions,
        )
        if np.linalg.norm(det) == 0:
            continue
        # Only include reasonable detections (within 0.30m)
        err = np.linalg.norm(gt_pos - det[:3])
        if err < 0.30:
            offsets.append(gt_pos - det[:3])

    if len(offsets) < 3:
        return np.zeros(3)

    # Use median offset (robust to outliers)
    offset = np.median(offsets, axis=0)
    return offset


def state_from_depth_diff_multi(obs, backgrounds, cameras,
                                depth_diff_thresh=0.02, saturation_thresh=180,
                                cal_thresholds=None,
                                demo_gripper_positions=None,
                                background_stds=None):
    """Multi-camera fusion: detect objects in multiple cameras, cross-validate.

    Strategy:
      1. Run detection independently in each camera (with variance filtering)
      2. Cross-camera matching: find the cluster pair that agrees best in 3D
      3. If cameras disagree, prefer detection farther from robot arm

    Args:
        obs: RLBench observation
        backgrounds: dict of {camera_name: median_depth_array}
        cameras: list of camera names
        depth_diff_thresh: default depth difference threshold (meters)
        saturation_thresh: default color saturation threshold
        cal_thresholds: dict of {camera_name: (depth_thresh, sat_thresh)}
        demo_gripper_positions: list of (3,) from demo home poses
        background_stds: dict of {camera_name: std_depth_array} for variance filtering

    Returns:
        (3,) object position or zeros if nothing detected
    """
    AGREE_THRESH = 0.05  # 5cm consistency threshold

    gripper_pos = None
    if hasattr(obs, 'gripper_pose') and obs.gripper_pose is not None:
        gripper_pos = obs.gripper_pose[:3]

    ws = {'x_min': -0.1, 'x_max': 0.6,
          'y_min': -0.4, 'y_max': 0.4,
          'z_min': 0.75, 'z_max': 1.35}

    # Detect in each camera — get ALL candidate clusters (not just top-1)
    all_camera_detections = {}  # {cam: list of (3,) positions}
    for cam in cameras:
        if cam not in backgrounds:
            continue
        if cal_thresholds and cam in cal_thresholds:
            dt, st = cal_thresholds[cam]
        else:
            dt, st = depth_diff_thresh, saturation_thresh
        bg_std = background_stds.get(cam) if background_stds else None
        positions = detect_objects_combined(
            obs, backgrounds[cam], camera=cam,
            depth_diff_thresh=dt,
            saturation_thresh=st,
            min_object_pixels=3, workspace_bounds=ws,
            gripper_pos=gripper_pos,
            demo_gripper_positions=demo_gripper_positions,
            background_std=bg_std,
        )
        if len(positions) > 0:
            all_camera_detections[cam] = positions

    if not all_camera_detections:
        return np.zeros(3)

    # If only one camera detected anything, return its best
    if len(all_camera_detections) == 1:
        cam = list(all_camera_detections.keys())[0]
        return all_camera_detections[cam][0]

    # Cross-camera matching: find the cluster PAIR (one from each camera)
    # with the smallest 3D distance — this is the real target
    cam_list = list(all_camera_detections.keys())
    best_pair_avg = None
    best_pair_dist = float('inf')

    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            cam_a = cam_list[i]
            cam_b = cam_list[j]
            for pos_a in all_camera_detections[cam_a]:
                for pos_b in all_camera_detections[cam_b]:
                    d = np.linalg.norm(pos_a - pos_b)
                    if d < best_pair_dist:
                        best_pair_dist = d
                        best_pair_avg = (pos_a + pos_b) / 2.0

    if best_pair_dist < AGREE_THRESH:
        return best_pair_avg

    # No consistent pair found — fall back to best-scored from each camera,
    # prefer detection farthest from robot arm
    all_best = [(cam, all_camera_detections[cam][0]) for cam in cam_list]
    if gripper_pos is not None:
        best_det = max(all_best, key=lambda x: _point_to_segment_distance(
            x[1].reshape(1, 3), ROBOT_BASE_POS, gripper_pos)[0])
        return best_det[1]

    return all_best[0][1]


# Keep backwards compatibility
def state_from_rgbd(obs, camera="front"):
    """Fallback: single-image detection. Less accurate than depth-diff."""
    return state_from_depth_diff_single(obs, camera)


def state_from_depth_diff_single(obs, camera="front"):
    """Single-image fallback: use color + aggressive workspace filter."""
    rgb = getattr(obs, f"{camera}_rgb")
    depth = getattr(obs, f"{camera}_depth")
    intrinsics, extrinsics_c2w = get_camera_matrices(obs, camera)

    H, W = rgb.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Very aggressive color filter: only highly saturated, bright pixels
    hsv = _rgb_to_hsv(rgb)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_mask = (sat > 120) & (val > 80)  # much higher threshold

    valid_depth = (depth > 0) & np.isfinite(depth)
    mask = color_mask & valid_depth

    vs, us = np.where(mask)
    if len(vs) == 0:
        return np.zeros(3)

    ds = depth[vs, us]
    x_cam = (us - cx) * ds / fx
    y_cam = (vs - cy) * ds / fy
    z_cam = ds
    ones = np.ones(len(ds))
    p_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)
    p_world = (extrinsics_c2w @ p_cam.T).T[:, :3]

    # Very tight workspace: only table-level objects
    in_ws = (
        (p_world[:, 0] >= -0.3) & (p_world[:, 0] <= 0.6) &
        (p_world[:, 1] >= -0.4) & (p_world[:, 1] <= 0.4) &
        (p_world[:, 2] >= 0.75) & (p_world[:, 2] <= 1.3)
    )

    if not np.any(in_ws):
        return np.zeros(3)

    p_world = p_world[in_ws]
    return p_world.mean(axis=0)


def _rgb_to_hsv(rgb):
    """Convert RGB uint8 (H,W,3) to HSV uint8 (H,W,3). Pure numpy."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:,:,0], rgb_f[:,:,1], rgb_f[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    s = np.zeros_like(cmax)
    nonzero = cmax > 0
    s[nonzero] = (diff[nonzero] / cmax[nonzero]) * 255
    v = cmax * 255
    hsv = np.stack([h * 255 / 360, s, v], axis=2).astype(np.uint8)
    return hsv

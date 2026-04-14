#!/usr/bin/env python3
"""Camera utilities: depth unprojection using PyRep's official method.

Uses the same projection math as PyRep's VisionSensor.pointcloud_from_depth()
to ensure correct 3D reconstruction.
"""

import numpy as np


def get_camera_matrices(obs, camera="front"):
    """Extract intrinsics and extrinsics from RLBench observation."""
    intrinsics = obs.misc[f"{camera}_camera_intrinsics"]  # (3, 3)
    extrinsics_w2c = obs.misc[f"{camera}_camera_extrinsics"]  # (4, 4)
    extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
    return intrinsics, extrinsics_c2w


def get_depth_scale(obs, camera="front"):
    """Get near/far clip planes for Z-buffer → linear depth conversion."""
    near = obs.misc[f"{camera}_camera_near"]
    far = obs.misc[f"{camera}_camera_far"]
    return near, far


def depth_buffer_to_linear(depth_buffer, near, far):
    """Convert Z-buffer depth (0-1) to linear depth in meters."""
    return near + depth_buffer * (far - near)


def pointcloud_from_depth(depth_meters, extrinsics, intrinsics):
    """Convert depth (in meters) to world-frame point cloud.

    This is a direct port of PyRep's VisionSensor.pointcloud_from_depth_and_camera_params.

    Args:
        depth_meters: (H, W) depth in meters
        extrinsics: (4, 4) camera pose matrix (from obs.misc['{cam}_camera_extrinsics'])
        intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        (H, W, 3) world coordinates
    """
    H, W = depth_meters.shape

    # Create uniform pixel coordinate image: (H, W, 3) with [u, v, 1]
    us = np.tile(np.arange(W), [H]).reshape(H, W, 1).astype(np.float32)
    vs = np.tile(np.arange(H), [W]).reshape(W, H, 1).astype(np.float32)
    vs = np.transpose(vs, (1, 0, 2))
    upc = np.concatenate([us, vs, np.ones_like(us)], axis=-1)  # (H, W, 3)

    # pixel_coords = upc * depth  → (H, W, 3)
    pc = upc * np.expand_dims(depth_meters, -1)

    # Build camera projection matrix inverse
    C = np.expand_dims(extrinsics[:3, 3], 0).T  # (3, 1)
    R = extrinsics[:3, :3]
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    ext = np.concatenate((R_inv, -R_inv_C), -1)  # (3, 4)
    cam_proj_mat = np.matmul(intrinsics, ext)  # (3, 4)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])  # (4, 4)
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]  # (3, 4)

    # Transform pixel coords to world coords
    # pc: (H, W, 3) → add homogeneous → (H, W, 4) → transform → (H, W, 3)
    pc_homo = np.concatenate([pc, np.ones((H, W, 1))], -1)  # (H, W, 4)
    pc_flat = pc_homo.reshape(H * W, 4).T  # (4, H*W)
    world_flat = np.matmul(cam_proj_mat_inv, pc_flat)  # (3, H*W)
    world_coords = world_flat.T.reshape(H, W, 3)  # (H, W, 3)

    return world_coords


def batch_pixel_to_world(pixel_coords, depth_map, intrinsics, extrinsics_c2w,
                          near=None, far=None):
    """Unproject K pixel coordinates to 3D world coordinates.

    Uses the same method as PyRep's pointcloud_from_depth for consistency.

    Args:
        pixel_coords: (K, 2) array of (u, v) pixel coordinates
        depth_map: (H, W) depth image (Z-buffer or meters)
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics_c2w: (4, 4) camera-to-world (but we use raw extrinsics for PyRep method)
        near: near clip plane (if given, depth is Z-buffer)
        far: far clip plane

    Returns:
        (K, 3) world coordinates
    """
    H, W = depth_map.shape

    if near is not None and far is not None:
        depth_m = depth_buffer_to_linear(depth_map, near, far)
    else:
        depth_m = depth_map

    # Get raw extrinsics (world-to-camera) from c2w
    extrinsics = np.linalg.inv(extrinsics_c2w)

    # Compute full pointcloud
    world = pointcloud_from_depth(depth_m, extrinsics, intrinsics)

    K = len(pixel_coords)
    points = np.zeros((K, 3))
    for i in range(K):
        u, v = pixel_coords[i]
        u_int = int(np.clip(round(u), 0, W - 1))
        v_int = int(np.clip(round(v), 0, H - 1))
        d = depth_m[v_int, u_int]
        if d <= 0 or np.isnan(d) or np.isinf(d):
            continue
        points[i] = world[v_int, u_int]

    return points

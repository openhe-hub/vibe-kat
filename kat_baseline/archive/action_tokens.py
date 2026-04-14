#!/usr/bin/env python3
"""Action token utilities: triplet pose representation from KAT paper (Fig. 4).

Each SE(3) end-effector pose is represented as 3 points forming a triangle:
  p1 = position
  p2 = position + scale * R[:, 0]  (x-axis of rotation)
  p3 = position + scale * R[:, 1]  (y-axis of rotation)
"""

import numpy as np

# Import rotation utilities from kat_eval
from kat_eval import quat_xyzw_to_rot_matrix, rot_matrix_to_quat_xyzw


TRIPLET_SCALE = 0.05  # 5 cm triangle size


def pose_to_triplet(pos, quat_xyzw, scale=TRIPLET_SCALE):
    """Convert SE(3) pose to triplet of 3D points.

    Args:
        pos: (3,) position
        quat_xyzw: (4,) quaternion [x, y, z, w]
        scale: triangle edge length in meters

    Returns:
        (9,) array [p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z]
    """
    R = quat_xyzw_to_rot_matrix(quat_xyzw)
    p1 = pos
    p2 = pos + scale * R[:, 0]
    p3 = pos + scale * R[:, 1]
    return np.concatenate([p1, p2, p3])


def triplet_to_pose(triplet_9d, scale=TRIPLET_SCALE):
    """Convert triplet of 3D points back to SE(3) pose.

    Args:
        triplet_9d: (9,) array [p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z]
        scale: same scale used in pose_to_triplet

    Returns:
        (pos, quat_xyzw) where pos is (3,) and quat is (4,)
    """
    p1 = triplet_9d[:3]
    p2 = triplet_9d[3:6]
    p3 = triplet_9d[6:9]

    pos = p1
    # Recover rotation axes
    a1 = (p2 - p1) / scale  # x-axis (approximately unit vector)
    a2 = (p3 - p1) / scale  # y-axis (approximately unit vector)

    # Gram-Schmidt orthonormalization
    e1 = a1 / np.linalg.norm(a1)
    u2 = a2 - np.dot(e1, a2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=1)

    quat = rot_matrix_to_quat_xyzw(R)
    return pos, quat


def extract_waypoints_triplet(demo, n_keyframes=10, scale=TRIPLET_SCALE):
    """Extract waypoints from demo in triplet format.

    Returns:
        List of (10,) arrays [p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, gripper]
    """
    n = len(demo)
    if n <= n_keyframes:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, n_keyframes, dtype=int).tolist()

    waypoints = []
    for i in indices:
        obs = demo[i]
        pos = obs.gripper_pose[:3]
        quat = obs.gripper_pose[3:]  # xyzw
        triplet = pose_to_triplet(pos, quat, scale=scale)
        g = 1.0 if obs.gripper_open > 0.5 else 0.0
        waypoints.append(np.concatenate([triplet, [g]]))
    return waypoints


def triplet_waypoint_to_action(wp_10d, scale=TRIPLET_SCALE):
    """Convert 10D triplet waypoint to 8D RLBench action.

    Args:
        wp_10d: (10,) [p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, gripper]

    Returns:
        (8,) action [x, y, z, qx, qy, qz, qw, gripper]
    """
    triplet = wp_10d[:9]
    gripper = wp_10d[9]
    pos, quat = triplet_to_pose(triplet, scale=scale)
    return np.concatenate([pos, quat, [gripper]])

#!/usr/bin/env python3
"""KAT-style ICIL smoke test on RLBench reach_target (Stage 1)."""

import os
import sys
import json
import re
import numpy as np
from openai import OpenAI

# ── Rotation utilities ──────────────────────────────────────────────

def quat_xyzw_to_rot_matrix(q):
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

def rot_matrix_to_6d(R):
    """Rotation matrix → 6D representation (first two columns, flattened row-major)."""
    return np.concatenate([R[:, 0], R[:, 1]])

def sixd_to_rot_matrix(r6):
    """6D representation → rotation matrix via Gram-Schmidt."""
    a1, a2 = r6[:3], r6[3:6]
    e1 = a1 / np.linalg.norm(a1)
    u2 = a2 - np.dot(e1, a2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)

def rot_matrix_to_quat_xyzw(R):
    """Rotation matrix → quaternion (x,y,z,w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)

def sixd_to_quat_xyzw(r6):
    """6D rotation → quaternion (x,y,z,w)."""
    R = sixd_to_rot_matrix(r6)
    return rot_matrix_to_quat_xyzw(R)

# ── Waypoint extraction ─────────────────────────────────────────────

def extract_waypoints(demo, n_keyframes=10):
    """Extract uniformly subsampled keyframes from a demo.
    
    Returns list of 10D arrays: [x, y, z, r1..r6, gripper].
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
        R = quat_xyzw_to_rot_matrix(quat)
        r6 = rot_matrix_to_6d(R)
        g = 1.0 if obs.gripper_open > 0.5 else 0.0
        wp = np.concatenate([pos, r6, [g]])
        waypoints.append(wp)
    return waypoints

def extract_scene(obs):
    """Extract scene description from task_low_dim_state.
    
    Returns the raw array reshaped into per-object chunks.
    """
    state = obs.task_low_dim_state
    return state

# ── Prompt building ──────────────────────────────────────────────────

def fmt(arr, decimals=3):
    """Format array to string with fixed decimal places."""
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr) + "]"

def build_scene_str(state):
    """Format task_low_dim_state as object pose lines.
    
    For reach_target the state is just the target position (3D).
    We'll format whatever we get in chunks — print raw first.
    """
    # Try to split into 3D chunks (position only for simple tasks)
    # or 7D chunks (pos + quat). We'll figure out the right chunk size.
    n = len(state)
    lines = []
    if n % 7 == 0:
        chunk = 7
    elif n % 3 == 0:
        chunk = 3
    else:
        chunk = n  # just one object
    
    n_objects = max(1, n // chunk)
    for i in range(n_objects):
        obj_state = state[i*chunk:(i+1)*chunk]
        if len(obj_state) == 7:
            # pos + quat → pos + 6D rotation
            pos = obj_state[:3]
            quat = obj_state[3:]
            R = quat_xyzw_to_rot_matrix(quat)
            r6 = rot_matrix_to_6d(R)
            lines.append(f"  object_{i}: {fmt(np.concatenate([pos, r6]))}")
        elif len(obj_state) == 3:
            # position only — pad with identity rotation 6D
            # Identity rotation 6D = [1,0,0, 0,1,0]
            r6_id = np.array([1, 0, 0, 0, 1, 0], dtype=float)
            lines.append(f"  object_{i}: {fmt(np.concatenate([obj_state, r6_id]))}")
        else:
            # Unknown — just dump raw
            lines.append(f"  object_{i}: {fmt(obj_state)}")
    return "\n".join(lines)

def build_prompt(demos_data, query_scene_str):
    """Build the user message from demo data and query scene."""
    blocks = []
    for scene_str, waypoints in demos_data:
        wp_strs = [fmt(wp) for wp in waypoints]
        block = f"Scene:\n{scene_str}\nActions:\n  [{', '.join(wp_strs)}]"
        blocks.append(block)
    
    # Query
    blocks.append(f"Scene:\n{query_scene_str}\nActions:")
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """You are a robot action predictor. Given demonstration pairs of (scene, actions), predict actions for a new scene.

Coordinate frame: world frame, meters. Each waypoint is [x, y, z, r1, r2, r3, r4, r5, r6, g] where:
- x, y, z: end-effector position in meters
- r1..r6: 6D rotation representation (first two columns of the rotation matrix, flattened)
- g: gripper state (0 = closed, 1 = open)

Respond with ONLY a JSON array of waypoint arrays. No prose, no explanation."""

# ── LLM call ─────────────────────────────────────────────────────────

def call_llm(system_prompt, user_prompt):
    """Call OpenAI API and return response text and token usage."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://yansd666.com/v1",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content
    usage = response.usage
    return text, usage

# ── Response parsing ─────────────────────────────────────────────────

def parse_response(text):
    """Parse LLM response into list of 10D waypoint arrays.
    
    Returns (waypoints, error_msg). waypoints is None on failure.
    """
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = cleaned.strip()
    
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    
    if not isinstance(data, list):
        return None, "Response is not a list"
    
    waypoints = []
    for i, wp in enumerate(data):
        if not isinstance(wp, list):
            return None, f"Waypoint {i} is not a list"
        if len(wp) != 10:
            return None, f"Waypoint {i} has length {len(wp)}, expected 10"
        try:
            wp = [float(v) for v in wp]
        except (ValueError, TypeError) as e:
            return None, f"Waypoint {i} has non-float values: {e}"
        waypoints.append(np.array(wp))
    
    return waypoints, None

def waypoint_to_action(wp_10d):
    """Convert 10D waypoint [x,y,z,r1..r6,g] to 8D RLBench action [x,y,z,qx,qy,qz,qw,g]."""
    pos = wp_10d[:3]
    r6 = wp_10d[3:9]
    gripper = wp_10d[9]
    quat = sixd_to_quat_xyzw(r6)
    return np.concatenate([pos, quat, [gripper]])

# ── Main ─────────────────────────────────────────────────────────────

def main():
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import ReachTarget
    from rlbench.backend.exceptions import InvalidActionError

    # Setup
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    # Disable cameras to speed things up
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.front_camera.set_all(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task = env.get_task(ReachTarget)

    # ── Collect demos ──
    print("=" * 60)
    print("COLLECTING 3 DEMOS...")
    print("=" * 60)
    demos = task.get_demos(3, live_demos=True)
    
    # Inspect task_low_dim_state
    sample_state = demos[0][0].task_low_dim_state
    print(f"\ntask_low_dim_state shape: {sample_state.shape}")
    print(f"task_low_dim_state contents: {sample_state}")
    print(f"gripper_pose sample: {demos[0][0].gripper_pose}")
    print(f"gripper_open sample: {demos[0][0].gripper_open}")
    print(f"Demo lengths: {[len(d) for d in demos]}")

    # ── Extract demo data ──
    demos_data = []
    for i, demo in enumerate(demos):
        scene_str = build_scene_str(demo[0].task_low_dim_state)
        waypoints = extract_waypoints(demo, n_keyframes=10)
        demos_data.append((scene_str, waypoints))
        print(f"\nDemo {i} scene:\n{scene_str}")
        print(f"Demo {i} waypoints ({len(waypoints)}):")
        for j, wp in enumerate(waypoints):
            print(f"  wp{j}: {fmt(wp)}")

    # ── Reset for test episode ──
    print("\n" + "=" * 60)
    print("RESETTING FOR TEST EPISODE...")
    print("=" * 60)
    descriptions, obs = task.reset()
    query_scene_str = build_scene_str(obs.task_low_dim_state)
    print(f"Test scene:\n{query_scene_str}")

    # ── Build prompt ──
    user_prompt = build_prompt(demos_data, query_scene_str)
    print("\n" + "=" * 60)
    print("FULL PROMPT:")
    print("=" * 60)
    print(f"[System]: {SYSTEM_PROMPT}")
    print(f"\n[User]:\n{user_prompt}")

    # ── Call LLM ──
    print("\n" + "=" * 60)
    print("CALLING LLM (gpt-4o)...")
    print("=" * 60)
    try:
        response_text, usage = call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"LLM API error: {e}")
        env.shutdown()
        sys.exit(1)

    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"\nRaw LLM response:\n{response_text}")

    # ── Parse response ──
    waypoints_10d, parse_error = parse_response(response_text)
    if parse_error:
        print(f"\nPARSE ERROR: {parse_error}")
        print("Trial marked as FAILED (parse error)")
        env.shutdown()
        sys.exit(1)

    print(f"\nParsed {len(waypoints_10d)} waypoints:")
    actions = []
    for i, wp in enumerate(waypoints_10d):
        action = waypoint_to_action(wp)
        actions.append(action)
        print(f"  wp{i}: 10D={fmt(wp)} → 8D action={fmt(action)}")

    # ── Execute waypoints ──
    print("\n" + "=" * 60)
    print("EXECUTING WAYPOINTS...")
    print("=" * 60)
    n_executed = 0
    exec_error = None
    for i, action in enumerate(actions):
        try:
            obs, reward, terminate = task.step(action)
            n_executed += 1
            print(f"  Step {i}: OK (reward={reward}, terminate={terminate})")
            if terminate:
                print(f"  Task terminated at step {i}")
                break
        except InvalidActionError as e:
            exec_error = str(e)
            print(f"  Step {i}: InvalidActionError: {e}")
            break
        except Exception as e:
            exec_error = str(e)
            print(f"  Step {i}: Error: {e}")
            break

    # ── Check success ──
    success, _ = task._task.success()
    print("\n" + "=" * 60)
    print(f"RESULT: {'SUCCESS' if success else 'FAILURE'}")
    print(f"  Waypoints executed: {n_executed}/{len(actions)}")
    if exec_error:
        print(f"  Execution error: {exec_error}")
    print("=" * 60)

    env.shutdown()

if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run this from repo root to compute action statistics from demo data (same pipeline as
# inference, no world model). Use the output to set CEM planning bounds and std.
#
# Usage:
#   cd /home/xilunz/cosmos-predict2.5 && uv run python examples/action_conditioned_check_action_range.py -i assets/action_conditioned/basic/inference_params.json
#
# Standalone: only numpy + stdlib, no torch/megatron so "uv run" stays fast.

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _euler2rotm(euler_angles):
    """Euler (ZYX) to rotation matrix. Inlined from dataset_utils to avoid heavy imports."""
    a, b, c = euler_angles[0], euler_angles[1], euler_angles[2]
    ra = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rb = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rc = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rc @ rb @ ra


def _rotm2euler(R):
    """Rotation matrix to Euler (ZYX). Inlined from dataset_utils."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float64)


def _get_actions(arm_states, gripper_states, sequence_length):
    """Relative actions from consecutive states (Euler, 7D). No scaling."""
    action = np.zeros((sequence_length - 1, 7))
    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = _euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = _euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_rot = _rotm2euler(rel_rotm)
        action[k - 1, 0:3] = rel_xyz
        action[k - 1, 3:6] = rel_rot
        action[k - 1, 6] = curr_gripper
    return action


def get_actions_scaled(data, action_scaler=20.0, gripper_scale=1.0, fps_downsample_ratio=1, state_key="state", gripper_key="continuous_gripper_state"):
    """Same scaling as inference: no random noise."""
    arm_states = np.array(data[state_key])
    cont_gripper = np.array(data[gripper_key])
    arm_states = arm_states[::fps_downsample_ratio]
    cont_gripper = cont_gripper[::fps_downsample_ratio]
    n = len(arm_states)
    actions = _get_actions(arm_states, cont_gripper, n)
    scale = np.array([action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, gripper_scale])
    actions = actions * scale
    return actions


def main():
    parser = argparse.ArgumentParser(description="Compute action range from demo JSONs (for CEM planning bounds)")
    parser.add_argument("-i", "--input_file", type=Path, required=True, help="e.g. assets/action_conditioned/basic/inference_params.json")
    args = parser.parse_args()

    with open(args.input_file) as f:
        cfg = json.load(f)

    # input_root in JSON is relative to cwd (repo root when run as instructed)
    input_root = Path(cfg["input_root"]).resolve()
    json_sub = cfg["input_json_sub_folder"]
    json_dir = input_root / json_sub
    if not json_dir.exists():
        print(f"Directory not found: {json_dir}")
        return

    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        print(f"No JSON files in {json_dir}")
        return

    action_scaler = float(cfg.get("action_scaler", 20))
    gripper_scale = float(cfg.get("gripper_scale", 1))
    fps_downsample_ratio = int(cfg.get("fps_downsample_ratio", 1))
    state_key = cfg.get("state_key", "state")
    gripper_key = cfg.get("gripper_key", "continuous_gripper_state")
    camera_id = int(cfg["camera_id"]) if isinstance(cfg["camera_id"], str) and cfg["camera_id"].isdigit() else cfg["camera_id"]

    all_actions = []
    for jpath in json_paths:
        with open(jpath) as f:
            data = json.load(f)
        vid = data["videos"][camera_id]
        video_path = input_root / (vid["video_path"] if isinstance(vid, dict) else vid)
        if not video_path.exists():
            print(f"Skip {jpath.name}: video not found {video_path}")
            continue
        actions = get_actions_scaled(
            data,
            action_scaler=action_scaler,
            gripper_scale=gripper_scale,
            fps_downsample_ratio=fps_downsample_ratio,
            state_key=state_key,
            gripper_key=gripper_key,
        )
        all_actions.append(actions)

    if not all_actions:
        print("No actions collected.")
        return

    actions = np.concatenate(all_actions, axis=0)
    dim_names = ["rel_x", "rel_y", "rel_z", "roll", "pitch", "yaw", "gripper"]

    print("Action statistics (scaled space, same as world model input):")
    print(f"  Shape: {actions.shape} (total steps across all demos)")
    print()
    print(f"{'dim':<10} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} {'p1':>12} {'p99':>12}")
    print("-" * 82)
    mins, maxs = [], []
    for d, name in enumerate(dim_names):
        v = actions[:, d]
        mn, mx = v.min(), v.max()
        p1 = np.percentile(v, 1)
        p99 = np.percentile(v, 99)
        mins.append(mn)
        maxs.append(mx)
        print(f"{name:<10} {mn:>12.4f} {mx:>12.4f} {v.mean():>12.4f} {v.std():>12.4f} {p1:>12.4f} {p99:>12.4f}")

    overall_min = min(mins)
    overall_max = max(maxs)
    std_over_dims = np.std(actions, axis=0)
    suggested_std = float(np.median(std_over_dims))

    print()
    print("Suggested CEM planning params (from data):")
    print(f"  action_bounds_low:  {overall_min:.4f}  (or a bit lower, e.g. {overall_min - 1:.2f})")
    print(f"  action_bounds_high: {overall_max:.4f}  (or a bit higher, e.g. {overall_max + 1:.2f})")
    print(f"  action_std_init:    {suggested_std:.4f}  (median per-dim std)")
    print()
    print("Add these to your planning_params.json or pass via CLI overrides.")


if __name__ == "__main__":
    main()

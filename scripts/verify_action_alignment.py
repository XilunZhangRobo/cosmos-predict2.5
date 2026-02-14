#!/usr/bin/env python3
"""
Verify Bridge vs Cosmos action format using actual data.

Run: cd cosmos-predict2.5 && python scripts/verify_action_alignment.py
(Requires numpy: pip install numpy)
"""
import json
import math
from pathlib import Path

import numpy as np


def euler2rotm(euler_angles):
    """Euler (ZYX) to rotation matrix."""
    a, b, c = euler_angles[0], euler_angles[1], euler_angles[2]
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotm2euler(R):
    """Rotation matrix to euler (x,y,z)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def get_actions_cosmos(arm_states, gripper_states):
    """Cosmos format: body-frame deltas from consecutive states."""
    actions = np.zeros((len(arm_states) - 1, 7))
    for k in range(1, len(arm_states)):
        prev_xyz = arm_states[k - 1, :3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, :3]
        curr_rpy = arm_states[k, 3:6]
        rel_xyz = prev_rotm.T @ (curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ euler2rotm(curr_rpy)
        rel_rpy = rotm2euler(rel_rotm)
        actions[k - 1, :3] = rel_xyz
        actions[k - 1, 3:6] = rel_rpy
        actions[k - 1, 6] = gripper_states[k]
    return actions


def main():
    base = Path(__file__).resolve().parents[1]
    data_path = base / "assets/action_conditioned/basic/bridge/annotation/test/0.json"
    with open(data_path) as f:
        data = json.load(f)

    arm_states = np.array(data["state"])
    gripper_states = np.array(data["continuous_gripper_state"])
    stored_actions = np.array(data["action"], dtype=np.float64)

    # Cosmos format from state (body-frame)
    cosmos_actions = get_actions_cosmos(arm_states, gripper_states)
    action_scaler = 20.0
    cosmos_scaled = cosmos_actions * np.array([20, 20, 20, 20, 20, 20, 1])

    # World-frame deltas (curr - prev) for first 4 steps
    world_xyz = np.array([arm_states[k, :3] - arm_states[k - 1, :3] for k in range(1, 5)])
    body_xyz = np.array(
        [
            euler2rotm(arm_states[k - 1, 3:6]).T @ (arm_states[k, :3] - arm_states[k - 1, :3])
            for k in range(1, 5)
        ]
    )

    print("=" * 70)
    print("ACTION ALIGNMENT VERIFICATION (sample 0.json)")
    print("=" * 70)

    print("\n1. STORED 'action' in JSON (first 4 steps, first 3 dims = xyz):")
    print(stored_actions[:4, :3])

    print("\n2. COSMOS format from state (body-frame xyz) * 20:")
    print(cosmos_scaled[:4, :3])

    print("\n3. Does stored match Cosmos body-frame?")
    diff_body = np.abs(stored_actions[:4] - cosmos_scaled[:4])
    print(f"   Max |stored - cosmos|: {np.max(diff_body):.6f}")
    if np.max(diff_body) < 1e-3:
        print("   -> YES: Stored action IS Cosmos (body-frame) format")
    else:
        print("   -> NO")

    print("\n4. World-frame delta (curr_xyz - prev_xyz) - unscaled:")
    print(world_xyz)

    print("\n5. Body-frame delta (R_prev.T @ world_delta) - unscaled:")
    print(body_xyz)

    print("\n6. If stored were Bridge (world-frame) * 20:")
    bridge_scaled = np.zeros((4, 7))
    bridge_scaled[:, :3] = world_xyz * 20
    bridge_scaled[:, 3:6] = cosmos_actions[:4, 3:6] * 20  # same rotation
    bridge_scaled[:, 6] = cosmos_actions[:4, 6]
    diff_world = np.abs(stored_actions[:4] - bridge_scaled)
    print(f"   Max |stored - world*20|: {np.max(diff_world):.6f}")
    if np.max(diff_world) < np.max(diff_body):
        print("   -> Stored might be world-frame (Bridge) format")
    else:
        print("   -> Stored is NOT world-frame")

    # Key: stored matches body_unscaled (unscaled body-frame)
    diff_stored_vs_body_unscaled = np.abs(stored_actions[:4, :3] - body_xyz)
    print("\n7. Stored xyz vs body_unscaled (from state) - EXACT MATCH?")
    print(f"   Max |stored - body_unscaled|: {np.max(diff_stored_vs_body_unscaled):.2e}")
    if np.max(diff_stored_vs_body_unscaled) < 1e-9:
        print("   -> Stored = BODY-FRAME (unscaled)")

    diff_stored_vs_world = np.abs(stored_actions[:4, :3] - world_xyz)
    print(f"\n8. Stored xyz vs world_unscaled - Max diff: {np.max(diff_stored_vs_world):.2e}")

    # Final verdict
    print("\n" + "=" * 70)
    if np.max(diff_stored_vs_body_unscaled) < 1e-9:
        print("CONCLUSION: Stored action in JSON = BODY-FRAME (unscaled).")
        print("Cosmos uses body-frame * 20 (computed from state via load_default_action_fn).")
        print("")
        print("VLA (INTACT-pi0-finetune-bridge) trained on Bridge typically uses WORLD-VECTOR.")
        print("So: VLA output (world) -> our convert_bridge_to_cosmos -> body-frame for world model.")
    print("=" * 70)

    # Part 2: Try loading VLA and compare (optional, needs VLA env)
    print("\n[Optional] Loading VLA to compare output format...")
    try:
        import os
        os.environ.setdefault("VLA_CLIP_ROOT", str(base / "vla-clip"))
        from cosmos_predict2.vla_inference import (
            generate_vla_actions,
            load_vla_policy,
            get_task_instruction_from_json,
            get_initial_state_from_json,
        )

        vla_policy, vla_adapter = load_vla_policy(
            "juexzz/INTACT-pi0-finetune-rephrase-bridge",
            n_action_steps=4,
            device="cuda:0",
        )

        video_path = base / "assets/action_conditioned/basic/bridge/videos/test/0/rgb.mp4"
        import mediapy
        video = mediapy.read_video(str(video_path))
        first_frame = video[0]
        if first_frame.shape[:2] != (256, 320):
            first_frame = mediapy.resize_image(first_frame, (256, 320))

        state, gripper = get_initial_state_from_json(data, frame_idx=0)
        task = get_task_instruction_from_json(data)

        # VLA output - call ONCE (deterministic with same seed)
        vla_raw = generate_vla_actions(
            first_frame, task, vla_policy, vla_adapter,
            state=state, gripper=gripper,
            convert_bridge_to_cosmos=False,
        )
        # Apply two-step conversion to SAME output: VLA -> dataset -> world_model
        from cosmos_predict2.vla_inference import vla_to_dataset_actions, dataset_to_world_model_actions
        scale = np.array([20, 20, 20, 20, 20, 20, 1], dtype=np.float32)
        raw_bridge = vla_raw / scale
        raw_bridge[:, 6] = vla_raw[:, 6]
        if state is not None:
            # Step 1: VLA -> dataset format (body unscaled)
            dataset_actions = vla_to_dataset_actions(raw_bridge, state)
            # Step 2: dataset -> world model
            vla_converted = dataset_to_world_model_actions(dataset_actions, 20, 1)
        else:
            vla_converted = vla_raw.copy()

        gt_cosmos = cosmos_scaled[:4]  # body-frame * 20 (world model format)

        # Two-step pipeline check: VLA -> dataset (body unscaled) -> world model (body * 20)
        gt_dataset_format = cosmos_actions[:4]  # body unscaled = stored in JSON
        if state is not None:
            vla_dataset = vla_to_dataset_actions(raw_bridge, state)
            err_dataset = np.abs(vla_dataset - gt_dataset_format)
            print("\n9a. TWO-STEP PIPELINE (VLA -> dataset format -> world model):")
            print("    Step 1: VLA/Bridge -> dataset format (body unscaled, matches JSON stored)")
            print("    Step 2: dataset format * 20 -> world model format")
            print(f"    |VLA_dataset - GT_dataset| mean: {err_dataset.mean():.4f} (unscaled units)")

        print("\n9. VLA vs GT (Cosmos body*20) comparison:")
        l2_raw = np.linalg.norm(vla_raw - gt_cosmos)
        l2_conv = np.linalg.norm(vla_converted - gt_cosmos)
        print(f"   VLA raw L2 to GT: {l2_raw:.4f}")
        print(f"   VLA converted L2 to GT: {l2_conv:.4f}")

        # Min/max and scale context
        print("\n10. ACTION SCALE (what do the numbers mean?):")
        print("   GT (Cosmos body*20) - dim [xyz, rpy, gripper]:")
        print(f"      min: {gt_cosmos.min(axis=0)}")
        print(f"      max: {gt_cosmos.max(axis=0)}")
        print(f"      range: {gt_cosmos.max(axis=0) - gt_cosmos.min(axis=0)}")
        print("   VLA raw:")
        print(f"      min: {vla_raw.min(axis=0)}")
        print(f"      max: {vla_raw.max(axis=0)}")
        print("   VLA converted:")
        print(f"      min: {vla_converted.min(axis=0)}")
        print(f"      max: {vla_converted.max(axis=0)}")

        # L2 interpretation
        n_vals = gt_cosmos.size
        rmse_raw = l2_raw / np.sqrt(n_vals)  # RMS error per element
        rmse_conv = l2_conv / np.sqrt(n_vals)
        gt_std = np.std(gt_cosmos)
        gt_range = gt_cosmos.max() - gt_cosmos.min()
        print(f"\n11. L2 INTERPRETATION:")
        print(f"   GT std: {gt_std:.4f}, GT range: {gt_range:.4f}")
        print(f"   VLA raw: L2={l2_raw:.2f} over {n_vals} values -> RMS error/element={rmse_raw:.4f}")
        print(f"   VLA conv: L2={l2_conv:.2f} -> RMS error/element={rmse_conv:.4f}")
        print(f"   -> Avg |error| per dim: raw~{rmse_raw:.2f}, conv~{rmse_conv:.2f}")
        print(f"   -> Error relative to GT range ({gt_range:.1f}): raw {100*rmse_raw/gt_range:.0f}%, conv {100*rmse_conv/gt_range:.0f}%")

        # Per-dimension error
        err_raw = np.abs(vla_raw - gt_cosmos)
        err_conv = np.abs(vla_converted - gt_cosmos)
        dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        print("\n12. PER-DIMENSION mean |error| (raw vs converted):")
        for i, name in enumerate(dim_names):
            m_raw = err_raw[:, i].mean()
            m_conv = err_conv[:, i].mean()
            print(f"   {name}: raw={m_raw:.4f}, conv={m_conv:.4f}  (GT range this dim: {gt_cosmos[:, i].max() - gt_cosmos[:, i].min():.4f})")

        # L2 excluding gripper (gripper may have convention mismatch: 0/1 vs continuous)
        l2_raw_6d = np.linalg.norm(vla_raw[:, :6] - gt_cosmos[:, :6])
        l2_conv_6d = np.linalg.norm(vla_converted[:, :6] - gt_cosmos[:, :6])
        print("\n13. L2 EXCLUDING GRIPPER (xyz+rpy only - gripper often has 0/1 convention mismatch):")
        print(f"   VLA raw (6d): {l2_raw_6d:.4f}, RMS/el={l2_raw_6d/np.sqrt(24):.4f}")
        print(f"   VLA conv (6d): {l2_conv_6d:.4f}, RMS/el={l2_conv_6d/np.sqrt(24):.4f}")

        print("\n14. WHAT THESE NUMBERS MEAN:")
        print(f"   GT xyz range: ~{gt_cosmos[:, :3].max(axis=0) - gt_cosmos[:, :3].min(axis=0)} (meters*20)")
        print(f"   GT rpy range: ~{gt_cosmos[:, 3:6].max(axis=0) - gt_cosmos[:, 3:6].min(axis=0)} (rad*20)")
        print(f"   Error ~0.5 per dim = ~2.5cm position error (at scale 20) or ~0.025rad rotation")
        print(f"   Gripper: GT={gt_cosmos[:, 6].tolist()} (0=closed), VLA={vla_raw[:, 6].tolist()} (~1=open) -> convention check")

        if l2_conv < l2_raw:
            print("\n   -> Conversion IMPROVES alignment (use convert_bridge_to_cosmos=True)")
        else:
            print("\n   -> Conversion does NOT improve; use convert_bridge_to_cosmos=False")

    except Exception as e:
        print(f"   (Skip VLA: {e})")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# VLA (Vision-Language-Action) inference for PI0 policy.
#
# Generates action chunks from an initial frame and language instruction using
# the PI0 model with batch inference (as in run_simpler_eval_with_openpi.py).
#
# Required when use_vla_actions=True: Add the following to your PYTHONPATH:
#   - Path to lerobot (for PI0Policy)
#   - Path to INT-ACT (for BridgeSimplerAdapter; create_bridge_adapter_wrapper is in cosmos)
#
# Example: export PYTHONPATH="${VLA_CLIP_ROOT}/lerobot_custom:${VLA_CLIP_ROOT}/INT-ACT:$PYTHONPATH"
#

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

from loguru import logger

# Ensure VLA paths are in sys.path (VLA_CLIP_ROOT set by run_action_planning_vla.sh)
_VLA_CLIP_ROOT = os.environ.get("VLA_CLIP_ROOT")
if _VLA_CLIP_ROOT:
    for subpath in ("lerobot_custom", "INT-ACT"):
        path = os.path.join(_VLA_CLIP_ROOT, subpath)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _ensure_vla_deps():
    """Lazily import PI0 and Bridge adapter; raise with helpful message if missing."""
    try:
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    except ImportError as e:
        raise ImportError(
            "VLA inference requires lerobot (PI0Policy). "
            "Add lerobot to PYTHONPATH, e.g.: export PYTHONPATH=\"${VLA_CLIP_ROOT}/lerobot_custom:$PYTHONPATH\""
        ) from e

    from cosmos_predict2.vla_bridge_adapter import create_bridge_adapter_wrapper

    return PI0Policy, create_bridge_adapter_wrapper


def _cosmos_state_to_bridge_eef_pos(
    state: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Convert cosmos state [x,y,z,roll,pitch,yaw] + gripper to Bridge eef_pos [x,y,z,qw,qx,qy,qz,gripper].

    Bridge format expects agent.eef_pos with quaternion (w,x,y,z) for preprocess_proprio.
    """
    from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2quat

    xyz = state[:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    rotm = euler2rotm(rpy)
    quat_wxyz = rotm2quat(rotm)  # (w, x, y, z)
    return np.concatenate([xyz, quat_wxyz, [float(gripper)]]).astype(np.float32)


def load_vla_policy(
    checkpoint: str,
    n_action_steps: int = 4,
    action_ensemble_temp: float = -0.8,
    device: str = "cuda",
):
    """Load PI0 policy and preprocess adapter.

    Returns:
        Tuple of (pi0_policy, preprocess_adapter).
    """
    PI0Policy, create_bridge_adapter_wrapper = _ensure_vla_deps()

    logger.info(f"Loading VLA model from {checkpoint}...")
    pi0_policy = PI0Policy.from_pretrained(checkpoint)
    if torch.cuda.is_available():
        pi0_policy.to(device)
        pi0_policy.config.device = device
    pi0_policy.config.n_action_steps = int(n_action_steps)

    preprocess_adapter = create_bridge_adapter_wrapper(action_ensemble_temp)
    if not hasattr(pi0_policy, "_preprocess_adapter"):
        pi0_policy._preprocess_adapter = preprocess_adapter

    return pi0_policy, preprocess_adapter


def vla_to_dataset_actions(
    bridge_actions: np.ndarray,
    state_at_chunk_start: np.ndarray,
) -> np.ndarray:
    """Step 1: Convert VLA (Bridge format) to original dataset format.

    Bridge/VLA outputs [world_vector (3), rotation_delta (3), gripper (1)] in world frame.
    Dataset stores body-frame [rel_xyz, rel_rpy, gripper] unscaled (same as _get_actions output).

    Returns actions in dataset format: body-frame deltas, physical units, unscaled.
    """
    return _bridge_to_cosmos_actions(bridge_actions, state_at_chunk_start)


def dataset_to_world_model_actions(
    dataset_actions: np.ndarray,
    action_scaler: float = 20.0,
    gripper_scale: float = 1.0,
) -> np.ndarray:
    """Step 2: Convert dataset format to world model format.

    Dataset = body-frame unscaled (from _get_actions or vla_to_dataset_actions).
    World model = same * [action_scaler, ..., gripper_scale] (same as get_action_sequence_from_states).
    """
    scale = np.array(
        [action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, gripper_scale],
        dtype=np.float32,
    )
    return (dataset_actions.astype(np.float32) * scale).astype(np.float32)


def _bridge_to_cosmos_actions(
    bridge_actions: np.ndarray,
    state_at_chunk_start: np.ndarray,
) -> np.ndarray:
    """Convert Bridge (world-frame xyz) to dataset format (body-frame, unscaled).

    Bridge outputs [world_vector (3), rotation_delta (3), gripper (1)] in world frame.
    Dataset expects [rel_xyz (3), rel_rpy (3), gripper (1)] with rel_xyz = R.T @ world_vector.

    Propagates state through the chunk to convert each action correctly.
    """
    from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler

    actions_cosmos = np.zeros_like(bridge_actions, dtype=np.float32)
    xyz = state_at_chunk_start[:3].astype(np.float64)
    rpy = state_at_chunk_start[3:6].astype(np.float64)
    R = euler2rotm(rpy)

    for t in range(bridge_actions.shape[0]):
        world_xyz = bridge_actions[t, :3]
        rot_delta = bridge_actions[t, 3:6]
        gripper = bridge_actions[t, 6]

        # Convert xyz: world -> body
        body_xyz = R.T @ world_xyz
        actions_cosmos[t, :3] = body_xyz
        actions_cosmos[t, 3:6] = rot_delta  # Assume rotation_delta is body-frame (common)
        actions_cosmos[t, 6] = gripper

        # Propagate state for next action
        xyz = xyz + world_xyz
        rel_rotm = euler2rotm(rot_delta)
        R = R @ rel_rotm
        rpy = rotm2euler(R)

    return actions_cosmos


def generate_vla_actions(
    initial_frame: np.ndarray,
    task_instruction: str,
    pi0_policy,
    preprocess_adapter,
    *,
    state: Optional[np.ndarray] = None,
    gripper: Optional[float] = None,
    batch_size: int = 2,
    action_noise_std: float = 1.0,
    action_scaler: float = 20.0,
    gripper_scale: float = 1.0,
    convert_bridge_to_cosmos: bool = True,
) -> np.ndarray:
    """Generate a chunk of actions from the VLA model given initial frame and task.

    Args:
        initial_frame: (H, W, 3) uint8 RGB image.
        task_instruction: Language instruction for the task.
        pi0_policy: Loaded PI0Policy instance.
        preprocess_adapter: Bridge preprocess adapter instance.
        state: Optional robot state [x,y,z,roll,pitch,yaw] for proprio. If None, uses zeros.
        gripper: Optional gripper openness (0-1). If None and state not provided, uses 0.5.
        batch_size: Number of samples for batch inference (as in run_simpler_eval).
        action_noise_std: Action noise for stochastic sampling.

    action_scaler: Scale factor for delta actions (cosmos default 20).
    gripper_scale: Scale for gripper dim (default 1).

    Returns:
        actions: (n_action_steps, 7) in cosmos world model format (scaled deltas).

    Pipeline when convert_bridge_to_cosmos=True:
        1. VLA (Bridge format, world-frame) -> vla_to_dataset_actions -> dataset format (body unscaled)
        2. dataset format -> dataset_to_world_model_actions -> world model format (body * action_scaler)
    """
    policy_device = torch.device(pi0_policy.config.device)
    n_action_steps = pi0_policy.config.n_action_steps

    # Build obs for adapter: needs observation.images.top, observation.state, task
    if state is not None and gripper is not None:
        eef_pos = _cosmos_state_to_bridge_eef_pos(state, gripper)
        obs_state = {"agent": {"eef_pos": eef_pos}}
    else:
        # Dummy state when not available (e.g. image-only setup)
        dummy_eef = np.zeros(8, dtype=np.float32)
        dummy_eef[7] = 0.5  # gripper half-open
        obs_state = {"agent": {"eef_pos": dummy_eef}}

    obs_for_adapter = {
        "observation.images.top": initial_frame,
        "observation.state": obs_state,
        "task": task_instruction,
    }
    processed_obs = preprocess_adapter.preprocess(obs_for_adapter)

    # Move to policy device
    processed_obs = {
        k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
        for k, v in processed_obs.items()
    }

    # Batch: repeat image, state, and task for batch_size
    task_list = [task_instruction] * batch_size
    batch_image = processed_obs["observation.images.top"].repeat(batch_size, 1, 1, 1)
    batch_state = processed_obs["observation.state"].repeat(batch_size, 1)

    image_feature_keys = list(pi0_policy.config.image_features.keys())
    image_key = image_feature_keys[0]

    observation = {
        image_key: batch_image,
        "observation.state": batch_state,
        "task": task_list,
    }

    # Reset action queue and run select_action to get full chunk
    pi0_policy.reset()
    with torch.no_grad():
        action_queue = pi0_policy.select_action(observation, noise_std=action_noise_std)

    # action_queue is deque of (batch_size, 7) tensors; take first sample (or mean over batch)
    # Use first sample for reproducibility; could optionally ensemble across batch
    actions_list = []
    for t in range(n_action_steps):
        act_t = action_queue[t][0:1].cpu().numpy()  # (1, 7)
        actions_list.append(act_t)

    # Stack: (n_action_steps, 7) - raw model output (normalized in [-1,1])
    actions_np = np.concatenate(actions_list, axis=0)

    # Denormalize to Bridge delta space (physical units)
    raw_bridge = _denormalize_bridge_actions(actions_np, preprocess_adapter)

    # Two-step conversion: VLA -> dataset format -> world model format
    if convert_bridge_to_cosmos and state is not None:
        # Step 1: Bridge (world-frame) -> dataset (body-frame, unscaled)
        dataset_actions = vla_to_dataset_actions(raw_bridge, state)
    else:
        dataset_actions = raw_bridge

    # Step 2: dataset -> world model (same scaling as get_action_sequence_from_states)
    return dataset_to_world_model_actions(
        dataset_actions, action_scaler=action_scaler, gripper_scale=gripper_scale
    )


def _denormalize_bridge_actions(actions_normalized: np.ndarray, adapter) -> np.ndarray:
    """Denormalize PI0 output to Bridge delta space and scale for cosmos world model.

    PI0 outputs normalized [-1,1]. Bridge uses bound normalization. Cosmos expects
    actions scaled by action_scaler (default 20) on first 6 dims.
    """
    stats = adapter.dataset_statistics["action"]
    p01 = np.array(stats["p01"], dtype=np.float64)
    p99 = np.array(stats["p99"], dtype=np.float64)
    # Gripper is not normalized in Bridge training
    raw = np.clip(actions_normalized.astype(np.float64), -1, 1)
    raw[:, :-1] = 0.5 * (raw[:, :-1] + 1) * (p99[:-1] - p01[:-1]) + p01[:-1]
    return raw.astype(np.float32)


def get_task_instruction_from_json(json_data: dict, task_instruction_override: Optional[str] = None) -> str:
    """Extract task instruction from JSON or use override.

    Supports Bridge format (texts[0]) and generic 'task' / 'instruction' keys.
    """
    if task_instruction_override:
        return task_instruction_override
    if "texts" in json_data and json_data["texts"]:
        return str(json_data["texts"][0])
    if "task" in json_data and isinstance(json_data["task"], str):
        return json_data["task"]
    if "instruction" in json_data:
        return str(json_data["instruction"])
    raise ValueError(
        "No task instruction found. Provide via task_instruction config param, "
        "or ensure JSON has 'texts', 'task', or 'instruction' key."
    )


def get_initial_state_from_json(
    json_data: dict,
    frame_idx: int = 0,
    state_key: str = "state",
    gripper_key: str = "continuous_gripper_state",
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Extract robot state and gripper from JSON for given frame.

    Returns:
        (state, gripper) where state is [x,y,z,roll,pitch,yaw] or None,
        gripper is float in [0,1] or None.
    """
    if state_key not in json_data:
        return None, None

    states = json_data[state_key]
    if not states or frame_idx >= len(states):
        return None, None

    s = np.array(states[frame_idx], dtype=np.float64)
    if s.size >= 7:
        # Assume [x,y,z,roll,pitch,yaw,gripper]
        return s[:6], float(s[6])
    if s.size >= 6:
        state = s[:6]
        if gripper_key in json_data:
            grippers = json_data[gripper_key]
            if frame_idx < len(grippers):
                return state, float(grippers[frame_idx])
        return state, 0.5
    return None, None

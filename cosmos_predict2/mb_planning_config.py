# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for model-based planning with CEM (Cross-Entropy Method)."""

from pathlib import Path

from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import get_overrides_cls


class MBPlanningArguments(ActionConditionedInferenceArguments):
    """Arguments for model-based planning: same as action-conditioned inference plus CEM params."""

    # CEM planning
    planning_horizon: int = 12
    """Number of action steps to optimize for during CEM. The planned sequence length."""
    cem_iterations: int = 8
    """Number of Cross-Entropy Method iterations."""
    num_samples: int = 16
    """Number of action-sequence samples per CEM iteration."""
    num_elite: int = 8
    """Number of elite samples to refit the distribution."""
    action_std_init: float = 0.33
    """Initial std for action sampling (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    action_bounds_low: float = -5.5
    """Lower bound for clipping actions (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    action_bounds_high: float = 6.0
    """Upper bound for clipping actions (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    cost_type: str = "mse"
    """Cost to goal image: 'mse', 'l1', or 'feature_l1' (L1 in encoder feature space; mse/l1 often more stable)."""
    num_cost_rollouts: int = 2
    """Number of rollouts per action sequence when evaluating cost; costs are averaged to reduce variance."""

    # VLA (PI0) action generation
    use_vla_actions: bool = False
    """If True, use VLA model to generate actions from initial frame instead of CEM planning."""
    vla_checkpoint: str = "juexzz/INTACT-pi0-finetune-rephrase-bridge"
    """PI0/VLA model checkpoint (HuggingFace or path)."""
    vla_batch_size: int = 2
    """Batch size for VLA inference (more samples = more diverse actions)."""
    vla_n_action_steps: int = 4
    """Number of action steps per VLA chunk."""
    vla_action_ensemble_temp: float = -0.8
    """Temperature for action ensembling in Bridge adapter."""
    task_instruction: str | None = None
    """Override task instruction for VLA. If None, uses json_data['texts'][0] or ['task']."""

    # Multi-GPU: put VLA and world model on different GPUs when both don't fit on one
    world_model_device: str = "cuda:0"
    """Device for world model. Set vla_device=cuda:1 to put VLA on a different GPU."""
    vla_device: str = "cuda:0"
    """Device for VLA policy. Set world_model_device=cuda:0, vla_device=cuda:1 to split across 2 GPUs."""

    convert_bridge_to_cosmos: bool = True
    """Convert VLA xyz from Bridge (world-frame) to Cosmos (body-frame). Run scripts/verify_action_alignment.py to check."""

    # Replan conditioning: GT frame vs generated frame
    use_gt_frames_for_replan: bool = True
    """If True, use ground-truth demo frame every N steps for next chunk; if False, use world-model generated frame."""
    save_both_versions: bool = False
    """If True, save two videos: _planned.mp4 (full video including GT at chunk boundaries) and
    _planned_no_gt.mp4 (same rollout, but skip GT frames when saving - output has only generated frames)."""
    save_three_versions: bool = False
    """If True, save three videos: _planned.mp4, _planned_no_gt.mp4, and _planned_generated_vla.mp4.
    The third uses world-model generated frames as VLA input (preprocessed via Bridge adapter)."""


MBPlanningOverrides = get_overrides_cls(MBPlanningArguments, exclude=["name"])

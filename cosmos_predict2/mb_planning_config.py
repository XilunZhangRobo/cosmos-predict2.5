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


MBPlanningOverrides = get_overrides_cls(MBPlanningArguments, exclude=["name"])

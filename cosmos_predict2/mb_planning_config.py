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
    cem_iterations: int = 10
    """Number of Cross-Entropy Method iterations."""
    num_samples: int = 32
    """Number of action-sequence samples per CEM iteration."""
    num_elite: int = 16
    """Number of elite samples to refit the distribution."""
    action_std_init: float = 0.33
    """Initial std for action sampling (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    action_bounds_low: float = -5.5
    """Lower bound for clipping actions (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    action_bounds_high: float = 6.0
    """Upper bound for clipping actions (scaled space). Run action_conditioned_check_action_range.py to get from data."""
    cost_type: str = "mse"
    """Cost to goal image: 'mse' or 'l1'."""


MBPlanningOverrides = get_overrides_cls(MBPlanningArguments, exclude=["name"])

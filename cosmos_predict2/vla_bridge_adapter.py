# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Bridge adapter for VLA inference. Extracted from CoVer_VLA eval_utils
# so cosmos does not depend on simpler_env, tensorflow, imageio, etc.

from __future__ import annotations

import os
import sys


def create_bridge_adapter_wrapper(action_ensemble_temp: float = -0.8):
    """Create a BridgeSimplerAdapter for PI0 action pre/post-processing.

    Args:
        action_ensemble_temp: Temperature for action ensembling
            (negative = more recent actions get more weight).

    Returns:
        BridgeSimplerAdapter instance with preprocess() for obs and
        dataset_statistics for denormalization.
    """
    vla_clip_root = os.environ.get("VLA_CLIP_ROOT")
    if not vla_clip_root or not os.path.isdir(vla_clip_root):
        raise RuntimeError(
            "VLA_CLIP_ROOT must be set and point to vla-clip (see run_action_planning_vla.sh)"
        )

    int_act_path = os.path.join(vla_clip_root, "INT-ACT")
    if int_act_path not in sys.path:
        sys.path.insert(0, int_act_path)

    from src.experiments.env_adapters.simpler import BridgeSimplerAdapter

    class EnvConfig:
        def __init__(self):
            self.dataset_statistics_path = os.path.join(
                vla_clip_root, "INT-ACT", "config", "dataset", "bridge_statistics.json"
            )
            self.image_size = (224, 224)
            self.action_normalization_type = "bound"
            self.state_normalization_type = "bound"

    class ModelConfig:
        def __init__(self):
            self.chunk_size = 4
            self.action_ensemble_temp = action_ensemble_temp

    class Config:
        def __init__(self):
            self.env = EnvConfig()
            self.use_bf16 = False
            self.seed = 42
            self.model_cfg = ModelConfig()

    return BridgeSimplerAdapter(Config())

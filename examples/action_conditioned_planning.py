# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-based planning with CEM or VLA action generation.

Uses the action-conditioned world model: first frame of each demo video is the
initial state, last frame is the goal image.

Modes:
- CEM (default): Cross-Entropy Method plans action sequences that minimize
  distance from predicted final frame to goal, then saves the planned rollout video.
- VLA: Use PI0/VLA model to generate action chunks from the initial frame and
  language instruction. Set use_vla_actions=true and add CoVer_VLA/lerobot to
  PYTHONPATH. Task instruction comes from JSON (texts[0] or task) or override.

Example with VLA:
  overrides.use_vla_actions=true overrides.vla_checkpoint=juexzz/INTACT-pi0-finetune-bridge
"""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.mb_planning import run_planning_inference
from cosmos_predict2.mb_planning_config import (
    MBPlanningArguments,
    MBPlanningOverrides,
)
from cosmos_predict2.action_conditioned_config import ActionConditionedSetupArguments
from cosmos_predict2.config import (
    handle_tyro_exception,
    is_rank0,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_file: Annotated[Path, tyro.conf.arg(aliases=("-i",))]
    """Path to the inference/planning parameter file (same layout as action_conditioned + CEM params)."""
    setup: ActionConditionedSetupArguments
    """Setup arguments (model, checkpoint, etc.)."""
    overrides: MBPlanningOverrides
    """Overrides for planning params (CEM iterations, num_samples, etc.)."""


def main(args: Args) -> None:
    planning_args = MBPlanningArguments.from_files(
        [args.input_file], overrides=args.overrides
    )[0]
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    run_planning_inference(args.setup, planning_args)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    main(args)

    cleanup_environment()

#!/usr/bin/env bash
# Run action-conditioned planning with VLA (PI0) action generation.
#
# Prerequisites:
#   - cosmos-predict2.5 dependencies installed (e.g. uv run)
#   - CoVer_VLA / lerobot / INT-ACT available (set VLA_CLIP_ROOT below)
#
# Usage: ./run_action_planning_vla.sh
#   or:  bash run_action_planning_vla.sh

set -e

# Path to vla-clip (copy under cosmos-predict2.5)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_CLIP_ROOT="${VLA_CLIP_ROOT:-$SCRIPT_DIR/vla-clip}"

if [[ ! -d "$VLA_CLIP_ROOT" ]]; then
  echo "Error: vla-clip not found at $VLA_CLIP_ROOT"
  echo "Copy the vla-clip folder under cosmos-predict2.5:"
  echo "  cp -r /path/to/vla-clip $SCRIPT_DIR/vla-clip"
  exit 1
fi

export VLA_CLIP_ROOT
export PYTHONPATH="${VLA_CLIP_ROOT}/lerobot_custom:${VLA_CLIP_ROOT}/INT-ACT:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"

echo "Running action-conditioned planning with VLA..."
echo "  VLA_CLIP_ROOT=$VLA_CLIP_ROOT"
echo "  Config: assets/action_conditioned/basic/planning_params_vla.json"
echo ""

# Multi-GPU: if VLA + world model don't fit on one GPU, set in JSON or via overrides:
#   world_model_device=cuda:0  vla_device=cuda:1
#
# Use --extra cu128 for compatible torch 2.7; --extra vla for lerobot/PI0 deps
# -o/--output-dir is required (logs, config; planned videos go to save_root in the JSON)
uv run --extra cu128 --extra vla python examples/action_conditioned_planning.py \
  -i assets/action_conditioned/basic/planning_params_vla.json \
  -o outputs/action_conditioned/basic_vla

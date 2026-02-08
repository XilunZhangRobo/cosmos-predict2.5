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

"""
Model-based planning with Cross-Entropy Method (CEM).
Uses the action-conditioned world model to roll out action sequences and
optimizes toward a goal image (e.g. last frame of a demo video).
"""

from __future__ import annotations

import json
import os
from glob import glob

import mediapy
import numpy as np
import torch
import torchvision
from loguru import logger
from tqdm import tqdm

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

# Lazy-loaded feature encoder for cost_type="feature_l1" (avoids heavy import when using mse/l1)
_feature_encoder = None


def _get_feature_encoder():
    """Return a shared SigLIP image encoder for feature-space cost (L1 between encodings)."""
    global _feature_encoder
    if _feature_encoder is None:
        from PIL import Image as PILImage

        from cosmos_predict2._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.vision_encoder import (
            SigLIPEncoder,
        )

        _feature_encoder = SigLIPEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    return _feature_encoder


def rollout_world_model(
    video2world_cli: Video2WorldInference,
    initial_frame: np.ndarray,
    actions_chunk: np.ndarray,
    *,
    prompt: str = "",
    guidance: float = 0,
    num_latent_conditional_frames: int = 1,
    resolution: str = "256,320",
    negative_prompt: str | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out the world model from an initial frame given an action chunk.

    Args:
        video2world_cli: Initialized Video2WorldInference.
        initial_frame: (H, W, 3) uint8 image in [0, 255].
        actions_chunk: (T, action_dim) float array, same scaling as training (e.g. action_scaler applied).
        prompt, guidance, resolution, negative_prompt, seed: passed to generate_vid2world.

    Returns:
        video_frames: (T+1, H, W, 3) uint8 numpy.
        last_frame: (H, W, 3) uint8 numpy (last predicted frame).
    """
    img_tensor = torchvision.transforms.functional.to_tensor(initial_frame).unsqueeze(0)  # (1, 3, H, W)
    num_video_frames = int(actions_chunk.shape[0]) + 1
    vid_input = torch.cat(
        [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
        dim=0,
    )
    vid_input = (vid_input * 255.0).to(torch.uint8)
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

    action_tensor = torch.from_numpy(actions_chunk).float()
    if action_tensor.device.type != "cuda":
        action_tensor = action_tensor.cuda()

    video = video2world_cli.generate_vid2world(
        prompt=prompt,
        input_path=vid_input,
        action=action_tensor,
        guidance=guidance,
        num_video_frames=num_video_frames,
        num_latent_conditional_frames=num_latent_conditional_frames,
        resolution=resolution,
        seed=seed,
        negative_prompt=negative_prompt or "",
    )
    # video: (B, C, T, H, W) in [-1, 1]
    video_normalized = (video - (-1)) / (1 - (-1))
    video_clamped = (
        (torch.clamp(video_normalized[0], 0, 1) * 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    last_frame = video_clamped[-1]
    return video_clamped, last_frame


def goal_image_cost(
    pred_last_frame: np.ndarray,
    goal_image: np.ndarray,
    cost_type: str = "mse",
) -> float:
    """
    Cost from predicted last frame to goal image. Lower is better.

    Supports pixel-space costs ("mse", "l1") and feature-space L1 ("feature_l1")
    as in goal-conditioned energy minimization: L1 distance between encoded
    representations of predicted state and goal image (see e.g. V-JEPA / inferring
    actions by planning).

    Args:
        pred_last_frame: (H, W, 3) uint8 or float.
        goal_image: (H, W, 3) same dtype/shape as pred (will be resized to match if needed).
        cost_type: "mse", "l1", or "feature_l1" (L1 between encoder feature vectors).

    Returns:
        Scalar cost (non-negative).
    """
    if cost_type == "feature_l1":
        from PIL import Image as PILImage

        encoder = _get_feature_encoder()
        if pred_last_frame.shape != goal_image.shape:
            goal_pil = PILImage.fromarray(
                goal_image.astype(np.uint8) if goal_image.dtype != np.uint8 else goal_image
            )
            h, w = pred_last_frame.shape[:2]
            goal_pil = goal_pil.resize((w, h), PILImage.BILINEAR)
            goal_image = np.array(goal_pil)
        pred_pil = PILImage.fromarray(
            pred_last_frame.astype(np.uint8)
            if pred_last_frame.dtype != np.uint8
            else pred_last_frame
        )
        goal_pil = PILImage.fromarray(
            goal_image.astype(np.uint8) if goal_image.dtype != np.uint8 else goal_image
        )
        f_pred = encoder.encode_image(pred_pil).cpu().float().numpy()
        f_goal = encoder.encode_image(goal_pil).cpu().float().numpy()
        return float(np.mean(np.abs(f_pred - f_goal)))
    # Pixel-space
    if pred_last_frame.shape != goal_image.shape:
        from PIL import Image

        goal_pil = Image.fromarray(
            goal_image.astype(np.uint8) if goal_image.dtype != np.uint8 else goal_image
        )
        h, w = pred_last_frame.shape[:2]
        goal_pil = goal_pil.resize((w, h), Image.BILINEAR)
        goal_image = np.array(goal_pil)

    p = pred_last_frame.astype(np.float64) / 255.0
    g = goal_image.astype(np.float64) / 255.0
    if cost_type == "mse":
        return float(np.mean((p - g) ** 2))
    if cost_type == "l1":
        return float(np.mean(np.abs(p - g)))
    raise ValueError(f"Unknown cost_type: {cost_type}")


def cross_entropy_plan(
    initial_frame: np.ndarray,
    goal_image: np.ndarray,
    video2world_cli: Video2WorldInference,
    *,
    chunk_size: int = 12,
    action_dim: int = 7,
    cem_iterations: int = 5,
    num_samples: int = 32,
    num_elite: int = 4,
    action_std_init: float = 2.0,
    action_bounds_low: float = -15.0,
    action_bounds_high: float = 15.0,
    cost_type: str = "mse",
    prompt: str = "",
    guidance: float = 0,
    num_latent_conditional_frames: int = 1,
    resolution: str = "256,320",
    negative_prompt: str | None = None,
    seed_base: int = 0,
    progress: bool = True,
) -> tuple[np.ndarray, list[float], dict]:
    """
    Cross-Entropy Method (CEM) to plan an action sequence that minimizes
    distance from predicted final frame to goal image. Intended use: execute
    only the first action of the returned sequence, observe the new state, then replan.

    Actions are in the same scaled space as the world model (e.g. action_scaler=20
    already applied). Sampling is done in that space and clipped to [action_bounds_low, action_bounds_high].

    Args:
        initial_frame: (H, W, 3) uint8 start image.
        goal_image: (H, W, 3) uint8 goal image.
        video2world_cli: Initialized world model.
        chunk_size: Number of action steps (horizon).
        action_dim: 7 for Euler (xyz, rpy, gripper) or 8 for quat.
        cem_iterations: CEM iterations.
        num_samples: Samples per iteration.
        num_elite: Number of elite samples to refit distribution.
        action_std_init: Initial standard deviation for action sampling.
        action_bounds_low, action_bounds_high: Clip sampled actions to this range.
        cost_type: "mse", "l1", or "feature_l1" (L1 in encoder feature space) for goal_image_cost.
        prompt, guidance, resolution, negative_prompt: World model kwargs.
        seed_base: Base seed for rollout (each sample gets seed_base + sample_idx).
        progress: If True, show tqdm progress bar and loss after each iteration.

    Returns:
        best_actions: (chunk_size, action_dim) best action sequence found.
        elite_costs_per_iter: List of mean elite cost per iteration (for logging).
        metrics: Dict with detailed per-iteration and summary stats for tracking/plotting.
    """
    action_shape = (chunk_size, action_dim)
    mean = np.zeros(action_shape, dtype=np.float64)
    std = np.full(action_shape, action_std_init, dtype=np.float64)
    elite_costs_per_iter: list[float] = []
    best_costs_per_iter: list[float] = []
    action_std_mean_per_iter: list[float] = []  # mean(std) over action dims, to see convergence
    all_costs_last_iter: list[float] = []  # cost of every sample on last iter (for distribution)

    iter_range = range(cem_iterations)
    if progress:
        iter_range = tqdm(iter_range, desc="CEM", unit="iter")

    for it in iter_range:
        # Sample
        samples = np.random.normal(mean, std, size=(num_samples,) + action_shape)
        samples = np.clip(samples, action_bounds_low, action_bounds_high).astype(np.float32)

        costs = np.full(num_samples, np.inf, dtype=np.float64)
        sample_range = range(num_samples)
        if progress:
            sample_range = tqdm(sample_range, desc="  rollouts", leave=False, unit="sample")
        for k in sample_range:
            try:
                _, last_frame = rollout_world_model(
                    video2world_cli,
                    initial_frame,
                    samples[k],
                    prompt=prompt,
                    guidance=guidance,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    resolution=resolution,
                    negative_prompt=negative_prompt,
                    seed=seed_base + it * num_samples + k,
                )
                costs[k] = goal_image_cost(last_frame, goal_image, cost_type=cost_type)
            except Exception as e:
                logger.warning(f"CEM sample {k} rollout failed: {e}")
                costs[k] = np.inf

        # Elite set (lowest cost)
        elite_idx = np.argsort(costs)[:num_elite]
        elites = samples[elite_idx]
        elite_costs = costs[elite_idx]
        elite_mean = float(np.mean(elite_costs))
        best_cost = float(costs[elite_idx[0]])
        elite_costs_per_iter.append(elite_mean)
        best_costs_per_iter.append(best_cost)

        # Refit distribution to elites
        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0)
        std = np.maximum(std, 0.15)
        action_std_mean_per_iter.append(float(np.mean(std)))

        if it == cem_iterations - 1:
            all_costs_last_iter = costs.tolist()

        if progress and hasattr(iter_range, "set_postfix"):
            iter_range.set_postfix(
                elite_loss=f"{elite_mean:.5f}",
                best_loss=f"{best_cost:.5f}",
                refresh=True,
            )
        # Print loss on its own line so it stays visible (not scrolled away by nested bars)
        if progress:
            tqdm.write(
                f"CEM iter {it + 1}/{cem_iterations}  elite_loss={elite_mean:.6f}  best_loss={best_cost:.6f}"
            )
        logger.info(
            f"CEM iter {it + 1}/{cem_iterations} "
            f"elite_mean_cost={elite_mean:.6f} best_cost={best_cost:.6f}"
        )

    best_actions = elites[0]
    metrics = {
        "elite_costs_per_iter": elite_costs_per_iter,
        "best_costs_per_iter": best_costs_per_iter,
        "action_std_mean_per_iter": action_std_mean_per_iter,
        "final_elite_mean": float(elite_costs_per_iter[-1]),
        "final_best_cost": float(best_costs_per_iter[-1]),
        "all_costs_last_iter": all_costs_last_iter,
        "cost_mean_last_iter": float(np.mean(costs)),
        "cost_std_last_iter": float(np.std(costs)),
        "cost_min_last_iter": float(np.min(costs)),
        "cost_max_last_iter": float(np.max(costs)),
    }
    return best_actions, elite_costs_per_iter, metrics


def run_planning_inference(setup_args, planning_args):
    """
    Run model-based planning: for each input video, use first frame as initial state,
    last frame as goal image. At each step we plan an action sequence with CEM, execute
    only the first action, observe the new (predicted) state, and replan (receding-horizon).

    Uses the same setup as action_conditioned (checkpoint, config) and the same
    input layout (input_root, input_json_sub_folder, camera_id, etc.).
    """
    from cosmos_predict2._src.imaginaire.utils import distributed
    from cosmos_predict2.config import MODEL_CHECKPOINTS, load_callable

    torch.enable_grad(False)
    if planning_args.num_latent_conditional_frames not in (0, 1, 2):
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, got {planning_args.num_latent_conditional_frames}"
        )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri
    if experiment is None:
        raise ValueError("Experiment name must be provided")

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )
    logger.info("World model loaded.")

    input_video_path = planning_args.input_root
    input_json_path = planning_args.input_root / planning_args.input_json_sub_folder
    input_json_list = sorted(glob(str(input_json_path / "*.json")))
    rank0 = True
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0
    save_root = planning_args.save_root.resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Planned videos will be saved to: {save_root}")

    action_dim = 8 if planning_args.use_quat else 7
    action_load_fn_factory = load_callable(planning_args.action_load_fn)

    annotation_list = input_json_list[planning_args.start : planning_args.end]
    if rank0:
        annotation_list = tqdm(annotation_list, desc="Videos", unit="video")

    for annotation_path in annotation_list:
        with open(annotation_path, "r") as f:
            json_data = json.load(f)
        camera_id = (
            int(planning_args.camera_id)
            if isinstance(planning_args.camera_id, str) and planning_args.camera_id.isdigit()
            else planning_args.camera_id
        )
        if isinstance(json_data["videos"][camera_id], dict):
            video_path = str(input_video_path / json_data["videos"][camera_id]["video_path"])
        else:
            video_path = str(input_video_path / json_data["videos"][camera_id])

        video_array = mediapy.read_video(video_path)
        initial_frame = video_array[0]
        goal_image = video_array[-1]
        if planning_args.resolution != "none":
            try:
                h, w = map(int, planning_args.resolution.split(","))
                initial_frame = mediapy.resize_image(initial_frame, (h, w))
                goal_image = mediapy.resize_image(goal_image, (h, w))
            except Exception as e:
                logger.warning(f"Resize failed: {e}")

        # Get action length from demo to determine number of steps (execute first action only per plan)
        action_data = action_load_fn_factory()(json_data, video_path, planning_args)
        num_actions = len(action_data["actions"])
        logger.info(
            f"Demo has {num_actions} actions; planning horizon={planning_args.chunk_size}, "
            "execute first action only then replan (receding-horizon)"
        )

        img_name = os.path.basename(annotation_path).replace(".json", "")
        out_path = save_root / f"{img_name}_planned.mp4"
        if out_path.exists():
            logger.info(f"Planned video already exists: {out_path}")
            continue

        chunk_metrics_list = []  # per-step CEM metrics for tracking
        current_initial = initial_frame
        # Build video: first frame, then one new frame per executed action
        full_video_list = [current_initial]
        for step_idx in range(num_actions):
            logger.info(
                f"Planning for {img_name} step {step_idx + 1}/{num_actions} (goal = final frame)"
            )
            best_actions, elite_costs, cem_metrics = cross_entropy_plan(
                current_initial,
                goal_image,
                video2world_cli,
                chunk_size=planning_args.chunk_size,
                action_dim=action_dim,
                cem_iterations=planning_args.cem_iterations,
                num_samples=planning_args.num_samples,
                num_elite=planning_args.num_elite,
                action_std_init=planning_args.action_std_init,
                action_bounds_low=planning_args.action_bounds_low,
                action_bounds_high=planning_args.action_bounds_high,
                cost_type=planning_args.cost_type,
                prompt=planning_args.prompt or "",
                guidance=planning_args.guidance,
                num_latent_conditional_frames=planning_args.num_latent_conditional_frames,
                resolution=planning_args.resolution,
                negative_prompt=planning_args.negative_prompt,
                seed_base=planning_args.seed + step_idx * 10000,
                progress=True,
            )
            chunk_metrics_list.append({
                "step_idx": step_idx,
                **cem_metrics,
            })
            # Execute only the first action, then replan
            first_action = best_actions[0:1]
            video_frames, last_frame = rollout_world_model(
                video2world_cli,
                current_initial,
                first_action,
                prompt=planning_args.prompt or "",
                guidance=planning_args.guidance,
                num_latent_conditional_frames=planning_args.num_latent_conditional_frames,
                resolution=planning_args.resolution,
                negative_prompt=planning_args.negative_prompt,
                seed=planning_args.seed + 9999 + step_idx * 10000,
            )
            full_video_list.append(last_frame)
            current_initial = last_frame

        full_video = np.stack(full_video_list, axis=0)
        if rank0:
            out_path_str = str(out_path.resolve())
            try:
                mediapy.write_video(out_path_str, full_video, fps=planning_args.save_fps)
                tqdm.write(
                    f"Saved planned video to {out_path_str} ({num_actions} steps, {len(full_video)} frames)"
                )
                logger.info(
                    f"Saved planned video to {out_path_str} ({num_actions} steps, final elite cost: {chunk_metrics_list[-1]['final_elite_mean']:.6f})"
                )
            except Exception as e:
                logger.exception(f"Failed to save video to {out_path_str}: {e}")
                tqdm.write(f"ERROR: Failed to save video to {out_path_str}: {e}")

            # Save planning metrics to JSON for analysis/plotting
            stats_path = save_root / f"{img_name}_planning_stats.json"
            try:
                stats = {
                    "img_name": img_name,
                    "num_actions": int(num_actions),
                    "chunk_size": int(planning_args.chunk_size),
                    "num_frames_planned": int(len(full_video)),
                    "execute_first_action_only": True,
                    "steps": chunk_metrics_list,
                }
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"Saved planning stats to {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to save planning stats: {e}")

    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()
    video2world_cli.cleanup()

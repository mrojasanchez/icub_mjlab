"""Training script for DextrAH-RGB Teacher FGP with preset configurations and W&B logging.

Usage:
    # Run with recommended preset (RTX 5090 sweet spot, 2048 envs):
    python -m icub_mjlab.scripts.train --preset recommended

    # Run with maximum preset (RTX 5090 max throughput, 4096 envs):
    python -m icub_mjlab.scripts.train --preset maximum

    # Run with base preset (debugging / low VRAM, 64 envs):
    python -m icub_mjlab.scripts.train --preset base

    # Custom W&B project and run name:
    python -m icub_mjlab.scripts.train --preset recommended --wandb-project my-project --run-name run-test-01

    # Offline / No W&B (fallback to local TensorBoard):
    python -m icub_mjlab.scripts.train --preset base --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

# Crucial to prevent CUDA memory fragmentation with recurrent LSTM models on large rollout buffers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import icub_mjlab  # noqa: F401 - registers Mjlab-Lift-Cube-iCub task
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK_ID = "Mjlab-Lift-Cube-iCub"

PRESETS: dict[str, dict[str, Any]] = {
  "base": {
    "num_envs": 64,
    "num_steps_per_env": 24,
    "num_mini_batches": 4,
    "learning_rate": 3.0e-4,
    "num_learning_epochs": 5,
    "max_iterations": 4_000,
    "save_interval": 1000,
    "estimated_vram": "~3 - 4 GB",
    "description": "Base/debug configuration for testing or low-memory GPUs (64 envs)",
  },
  "recommended": {
    "num_envs": 2048,
    "num_steps_per_env": 24,
    "num_mini_batches": 8,
    "learning_rate": 3.0e-4,
    "num_learning_epochs": 5,
    "max_iterations": 4_000,
    "save_interval": 1000,
    "estimated_vram": "~16 - 18 GB",
    "description": "Recommended sweet-spot for RTX 5090 32GB (2048 envs, optimal stability & speed)",
  },
  "maximum": {
    "num_envs": 4096,
    "num_steps_per_env": 24,
    "num_mini_batches": 8,
    "learning_rate": 3.0e-4,
    "num_learning_epochs": 5,
    "max_iterations": 4_000,
    "save_interval": 1000,
    "estimated_vram": "~26 - 28 GB",
    "description": "Maximum throughput for RTX 5090 32GB (4096 envs, 8 mini-batches for safe BPTT memory)",
  },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train DextrAH-RGB Teacher FGP policy on iCub with RTX 5090 presets and W&B logging.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  # Preset selection
  parser.add_argument(
    "--preset",
    type=str,
    choices=list(PRESETS.keys()),
    default="recommended",
    help="Hyperparameter preset configuration ('base', 'recommended', or 'maximum').",
  )

  # W&B configuration
  parser.add_argument(
    "--wandb",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable or disable logging to Weights & Biases (W&B). If disabled, logs to local TensorBoard.",
  )
  parser.add_argument(
    "--wandb-project",
    type=str,
    default="icub-dextrah",
    help="W&B project name to log metrics and checkpoints.",
  )
  parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="W&B entity (team or username). If None, defaults to current wandb login.",
  )
  parser.add_argument(
    "--wandb-mode",
    type=str,
    choices=["online", "offline", "disabled"],
    default="online",
    help="W&B run mode ('online', 'offline' to log locally and sync later, or 'disabled').",
  )
  parser.add_argument(
    "--upload-model",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to upload model checkpoints (.pt) to W&B on save.",
  )
  parser.add_argument(
    "--wandb-tags",
    nargs="*",
    default=[],
    help="Additional tags for the W&B run.",
  )
  parser.add_argument(
    "--experiment-name",
    type=str,
    default="dextrah_teacher_fgp",
    help="Experiment group name under log root.",
  )
  parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Run name for W&B and directory name. Defaults to '{preset}_YYYYMMDD_HHMMSS'.",
  )

  # Custom hyperparameter overrides
  parser.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="Override number of parallel environments specified by preset.",
  )
  parser.add_argument(
    "--num-steps-per-env",
    type=int,
    default=None,
    help="Override rollout steps per environment update specified by preset.",
  )
  parser.add_argument(
    "--num-mini-batches",
    type=int,
    default=None,
    help="Override number of mini-batches specified by preset.",
  )
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=None,
    help="Override PPO learning rate specified by preset.",
  )
  parser.add_argument(
    "--num-learning-epochs",
    type=int,
    default=None,
    help="Override PPO epochs per iteration specified by preset.",
  )
  parser.add_argument(
    "--max-iterations",
    type=int,
    default=None,
    help="Override maximum learning iterations specified by preset.",
  )
  parser.add_argument(
    "--save-interval",
    type=int,
    default=None,
    help="Override checkpoint saving interval (iterations) specified by preset.",
  )
  parser.add_argument(
    "--schedule",
    type=str,
    choices=["fixed", "adaptive"],
    default="adaptive",
    help="PPO learning rate schedule ('fixed' or 'adaptive'). Defaults to 'adaptive'.",
  )
  parser.add_argument(
    "--desired-kl",
    type=float,
    default=0.015,
    help="Desired KL divergence for adaptive schedule.",
  )
  parser.add_argument(
    "--reward-goal-weight",
    type=float,
    default=None,
    help="Override object_to_goal reward weight (default in env_cfg is 2.0).",
  )
  parser.add_argument(
    "--reward-lift-weight",
    type=float,
    default=None,
    help="Override object_lift reward weight (default in env_cfg is 1.5).",
  )
  parser.add_argument(
    "--reward-curl-weight",
    type=float,
    default=None,
    help="Override finger_curl regularization weight (default in env_cfg is 0.05).",
  )
  parser.add_argument(
    "--reward-action-rate-weight",
    type=float,
    default=None,
    help="Override action_rate_l2 penalty weight (default in env_cfg is -0.025).",
  )

  # Training execution & environment options
  parser.add_argument(
    "--gpu-id",
    type=int,
    default=0,
    help="GPU device ID to run simulation and training on.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for environment and policy.",
  )
  parser.add_argument(
    "--log-root",
    type=str,
    default="logs/rsl_rl",
    help="Root directory for local experiment logs and checkpoints.",
  )
  parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Record evaluation/training videos.",
  )
  parser.add_argument(
    "--video-only-final",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record only the final policy video at the end of training (avoids exhausting W&B cloud storage with intermediate videos).",
  )
  parser.add_argument(
    "--video-interval-iter",
    type=int,
    default=None,
    help="Interval (learning iterations) between video recordings (e.g. 1000). Automatically converts to simulation steps.",
  )
  parser.add_argument(
    "--video-interval",
    type=int,
    default=None,
    help="Interval (simulation steps) between video recordings if --video is set.",
  )
  parser.add_argument(
    "--video-length",
    type=int,
    default=500,
    help="Length of video recording in steps (500 steps = 10s full episode at 50 Hz).",
  )
  parser.add_argument(
    "--video-final",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record a video of the final trained policy right before training completes (enabled by default when --video is set).",
  )
  parser.add_argument(
    "--enable-nan-guard",
    action="store_true",
    default=False,
    help="Enable simulation NaN guard.",
  )

  # Checkpoint resume options
  parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from previous checkpoint.",
  )
  parser.add_argument(
    "--load-run",
    type=str,
    default=".*",
    help="Run directory pattern to load from when resuming locally.",
  )
  parser.add_argument(
    "--load-checkpoint",
    type=str,
    default="model_.*.pt",
    help="Checkpoint filename pattern to load when resuming locally.",
  )
  parser.add_argument(
    "--wandb-run-path",
    type=str,
    default=None,
    help="W&B run path to download checkpoint from when resuming (e.g. 'entity/project/run_id').",
  )
  parser.add_argument(
    "--wandb-checkpoint-name",
    type=str,
    default=None,
    help="Specific checkpoint file in W&B to load (e.g. 'model_2000.pt').",
  )

  return parser.parse_args()


def print_training_banner(
  preset_name: str,
  num_envs: int,
  num_steps: int,
  num_mini_batches: int,
  lr: float,
  epochs: int,
  schedule: str,
  desired_kl: float,
  max_iterations: int,
  save_interval: int,
  vram_est: str,
  use_wandb: bool,
  wandb_project: str,
  run_name: str,
  upload_model: bool,
  video: bool,
  video_final: bool,
  video_only_final: bool,
  video_interval: int,
  video_length: int,
  gpu_id: int,
) -> None:
  total_batch = num_envs * num_steps
  mini_batch_size = total_batch // num_mini_batches
  bptt_chunk_size = num_envs // num_mini_batches

  print("\n" + "=" * 80)
  print("  🤖 iCub DextrAH-RGB Teacher FGP — Training Launcher  ".center(80))
  print("=" * 80)
  print(f"  Configuration Preset      : {preset_name.upper()}")
  print(f"  Target Device             : cuda:{gpu_id}")
  print(f"  Estimated VRAM Footprint  : {vram_est}")
  print("-" * 80)
  print(f"  Parallel Environments     : {num_envs:,}")
  print(f"  Steps per Env (Horizon)   : {num_steps}")
  print(f"  Total Batch per Iteration : {total_batch:,} transitions")
  print(f"  PPO Mini-batches          : {num_mini_batches} (size: {mini_batch_size:,} transitions)")
  print(f"  LSTM BPTT Sequences/Batch : {bptt_chunk_size} seqs of length {num_steps}")
  print(f"  PPO Epochs per Iteration  : {epochs}")
  print(f"  PPO Learning Rate         : {lr:.2e} ({schedule})")
  if schedule == "adaptive":
    print(f"  PPO Desired KL            : {desired_kl}")
  print(f"  Max Learning Iterations   : {max_iterations:,}")
  print(f"  Save Checkpoint Interval  : Every {save_interval} iterations (local disk)")
  if video:
    if video_only_final:
      print(f"  Video Recording           : Enabled (FINAL policy video only, len: {video_length} frames)")
    else:
      video_iters = video_interval // num_steps if num_steps > 0 else 0
      final_note = " + final policy" if video_final else ""
      print(f"  Video Recording           : Enabled (every {video_iters:,} iters / {video_interval:,} steps{final_note}, len: {video_length} frames)")
  else:
    print(f"  Video Recording           : Disabled")
  print("-" * 80)
  if use_wandb:
    print(f"  Logger                    : Weights & Biases (W&B)")
    print(f"  W&B Project               : {wandb_project}")
    print(f"  Run Display Name          : {run_name}")
    print(f"  Model Upload to W&B       : {'Enabled (Best & Final policy only)' if upload_model else 'Disabled'}")
  else:
    print(f"  Logger                    : TensorBoard (local only)")
  print("=" * 80 + "\n", flush=True)


def setup_video_recording(
  args: argparse.Namespace,
  max_iterations: int,
  num_steps: int,
) -> None:
  """Configure VideoRecorder to record videos (final policy only by default to save storage)."""
  if not args.video:
    return

  import mjlab.utils.wrappers.video_recorder as vr_mod

  orig_init = vr_mod.VideoRecorder.__init__

  # Calculate iteration and step for the final video
  # Ensure the recording completes ~3 iterations before max_iterations so rsl_rl has time to upload it to W&B
  steps_needed_iters = (args.video_length // num_steps) + 3
  final_video_iter = max(0, max_iterations - steps_needed_iters)
  final_video_start_step = final_video_iter * num_steps

  def wrapped_init(
    self,
    env,
    video_folder,
    episode_trigger=None,
    step_trigger=None,
    video_length=None,
    **kwargs,
  ):
    if step_trigger is not None:
      base_trigger = step_trigger
      only_final = getattr(args, "video_only_final", True)
      if only_final:
        print(
          f"[INFO] 🎥 Video recording configured for FINAL policy only at iteration {final_video_iter:,} "
          f"(simulation step {final_video_start_step:,}). Periodic videos disabled.",
          flush=True,
        )

        def final_step_trigger(step: int) -> bool:
          return step == final_video_start_step

        step_trigger = final_step_trigger
      elif getattr(args, "video_final", True):
        print(
          f"[INFO] Final policy video recording scheduled at iteration {final_video_iter:,} "
          f"(simulation step {final_video_start_step:,}).",
          flush=True,
        )

        def combined_step_trigger(step: int) -> bool:
          if base_trigger(step):
            return True
          if step == final_video_start_step:
            return True
          return False

        step_trigger = combined_step_trigger

    orig_init(
      self,
      env,
      video_folder,
      episode_trigger=episode_trigger,
      step_trigger=step_trigger,
      video_length=video_length,
      **kwargs,
    )

  vr_mod.VideoRecorder.__init__ = wrapped_init


def main() -> None:
  args = parse_args()

  preset = PRESETS[args.preset]

  # Resolve hyperparameters (preset defaults with CLI override precedence)
  num_envs = args.num_envs if args.num_envs is not None else preset["num_envs"]
  num_steps = (
    args.num_steps_per_env if args.num_steps_per_env is not None else preset["num_steps_per_env"]
  )
  num_mini_batches = (
    args.num_mini_batches if args.num_mini_batches is not None else preset["num_mini_batches"]
  )
  lr = args.learning_rate if args.learning_rate is not None else preset["learning_rate"]
  epochs = (
    args.num_learning_epochs
    if args.num_learning_epochs is not None
    else preset["num_learning_epochs"]
  )
  max_iterations = (
    args.max_iterations if args.max_iterations is not None else preset["max_iterations"]
  )
  save_interval = (
    args.save_interval if args.save_interval is not None else preset["save_interval"]
  )
  vram_est = preset["estimated_vram"]

  # Resolve video interval in simulation steps
  if args.video_interval_iter is not None:
    video_interval = args.video_interval_iter * num_steps
  elif args.video_interval is not None:
    video_interval = args.video_interval
  else:
    video_interval = 1000 * num_steps  # Default: every 1000 iterations

  # Generate run name if not explicitly set
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_name = args.run_name or f"{args.preset}_{timestamp}"

  # Configure W&B environment variables
  if args.wandb:
    if args.wandb_entity:
      os.environ["WANDB_USERNAME"] = args.wandb_entity
    os.environ["WANDB_MODE"] = args.wandb_mode

  # Display configuration summary
  print_training_banner(
    preset_name=args.preset,
    num_envs=num_envs,
    num_steps=num_steps,
    num_mini_batches=num_mini_batches,
    lr=lr,
    epochs=epochs,
    schedule=args.schedule,
    desired_kl=args.desired_kl,
    max_iterations=max_iterations,
    save_interval=save_interval,
    vram_est=vram_est,
    use_wandb=args.wandb,
    wandb_project=args.wandb_project,
    run_name=run_name,
    upload_model=args.upload_model,
    video=args.video,
    video_final=args.video_final,
    video_only_final=args.video_only_final,
    video_interval=video_interval,
    video_length=args.video_length,
    gpu_id=args.gpu_id,
  )

  # Load base configurations
  env_cfg = load_env_cfg(TASK_ID)
  rl_cfg = load_rl_cfg(TASK_ID)

  # Apply environment configuration
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = args.seed

  # Apply optional reward overrides if specified
  if args.reward_goal_weight is not None and "object_to_goal" in env_cfg.rewards:
    env_cfg.rewards["object_to_goal"].weight = args.reward_goal_weight
  if args.reward_lift_weight is not None and "object_lift" in env_cfg.rewards:
    env_cfg.rewards["object_lift"].weight = args.reward_lift_weight
  if args.reward_curl_weight is not None and "finger_curl" in env_cfg.rewards:
    env_cfg.rewards["finger_curl"].weight = args.reward_curl_weight
  if args.reward_action_rate_weight is not None and "action_rate_l2" in env_cfg.rewards:
    env_cfg.rewards["action_rate_l2"].weight = args.reward_action_rate_weight

  # Apply RL algorithm configuration
  rl_cfg.algorithm.num_mini_batches = num_mini_batches
  rl_cfg.algorithm.learning_rate = lr
  rl_cfg.algorithm.num_learning_epochs = epochs
  rl_cfg.algorithm.schedule = args.schedule
  rl_cfg.algorithm.desired_kl = args.desired_kl
  rl_cfg.num_steps_per_env = num_steps
  rl_cfg.max_iterations = max_iterations
  rl_cfg.save_interval = save_interval
  rl_cfg.seed = args.seed
  rl_cfg.experiment_name = args.experiment_name
  rl_cfg.run_name = run_name

  # Setup logging
  if args.wandb:
    rl_cfg.logger = "wandb"
    rl_cfg.wandb_project = args.wandb_project
    rl_cfg.upload_model = args.upload_model
    tags = [f"preset:{args.preset}", "icub", "dextrah_teacher_fgp", "rtx5090"]
    if args.wandb_tags:
      tags.extend(args.wandb_tags)
    rl_cfg.wandb_tags = tuple(tags)
  else:
    rl_cfg.logger = "tensorboard"
    rl_cfg.upload_model = False

  # Setup resume parameters if enabled
  if args.resume:
    rl_cfg.resume = True
    rl_cfg.load_run = args.load_run
    rl_cfg.load_checkpoint = args.load_checkpoint

  # Build train config container
  train_cfg = TrainConfig(
    env=env_cfg,
    agent=rl_cfg,
    video=args.video,
    video_length=args.video_length,
    video_interval=video_interval,
    enable_nan_guard=args.enable_nan_guard,
    log_root=args.log_root,
    gpu_ids=[args.gpu_id],
    wandb_run_path=args.wandb_run_path,
    wandb_checkpoint_name=args.wandb_checkpoint_name,
  )

  # Configure video recording hooks (periodic + final policy video)
  setup_video_recording(args, max_iterations=max_iterations, num_steps=num_steps)

  # Launch training loop
  launch_training(task_id=TASK_ID, args=train_cfg)


if __name__ == "__main__":
  main()


"""Custom RL runner for iCub DextrAH manipulation."""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import torch
from rsl_rl.env import VecEnv
from mjlab.rl.runner import MjlabOnPolicyRunner


class IcubOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that persists environment state and gates W&B model uploads to only the best and final policy."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    self._best_metric: float = -float("inf")
    self._best_model_path: str | None = None
    self._upload_only_final: bool = (
      os.environ.get("ICUB_UPLOAD_ONLY_FINAL", "0") == "1"
      or self.cfg.get("upload_only_final", False)
    )

  def save(self, path: str, infos: dict | None = None) -> None:
    """Save checkpoint locally, but only upload best and final models to W&B."""
    max_it = self.cfg.get("max_iterations", 4_000)
    is_final = ("model_final" in path) or (self.current_learning_iteration >= max_it - 1)

    # When upload_only_final is enabled, suppress all intermediate disk saves and W&B uploads
    if self._upload_only_final and not is_final:
      return

    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos

    # 1. Save checkpoint locally to disk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(saved_dict, path)

    # 2. Check if this is the best model so far (only when intermediate uploads are enabled)
    if not self._upload_only_final:
      current_success = 0.0
      try:
        cmd = self.env.unwrapped.command_manager.get_term("lift_height")
        if hasattr(cmd, "metrics") and "episode_success" in cmd.metrics:
          val = cmd.metrics["episode_success"]
          current_success = float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val)
      except Exception:
        pass

      current_rew = (
        statistics.mean(self.logger.rewbuffer)
        if hasattr(self.logger, "rewbuffer") and len(self.logger.rewbuffer) > 0
        else -float("inf")
      )

      # Composite evaluation score: primary is success rate (0-1), secondary is mean reward
      current_score = current_success * 100.0 + (current_rew if current_rew != -float("inf") else 0.0)

      is_best = False
      if current_score > self._best_metric and (current_success > 0.0 or current_rew > 0.0):
        self._best_metric = current_score
        is_best = True
        best_path = os.path.join(self.logger.log_dir, "model_best.pt")
        torch.save(saved_dict, best_path)
        self._best_model_path = best_path
        print(
          f"\n[INFO] 🏆 New best model saved! (Score: {current_score:.2f} | "
          f"Success: {current_success:.1%} | Reward: {current_rew:.2f}) -> {best_path}\n",
          flush=True,
        )

      if self.cfg.get("upload_model", False) and is_best and self._best_model_path:
        self.logger.save_model(self._best_model_path, self.current_learning_iteration)

    # 3. W&B Upload gating for final model
    if self.cfg.get("upload_model", False) and is_final:
      final_path = os.path.join(self.logger.log_dir, "model_final.pt")
      if path != final_path:
        torch.save(saved_dict, final_path)
      self.logger.save_model(final_path, self.current_learning_iteration)
      print(f"\n[INFO] 📦 Final model uploaded to W&B: {final_path}\n", flush=True)


"""DextrAH-RGB reward functions for iCub dexterous manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def hand_to_object_dextrah_reward(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
  palm_site_name: str = "r_hand_dh_frame_site",
  fingertip_body_names: tuple[str, ...] = (
    "r_hand_thumb_3",
    "r_hand_index_3",
    "r_hand_middle_3",
    "r_hand_little_3",
  ),
  scale: float = 10.0,
) -> torch.Tensor:
  """DextrAH hand-to-object reward: r_hand_obj = exp(-10 * max_i ||x_i - x_obj||).

  Maximizing the maximum distance across all fingertips and palm encourages the
  entire hand (fingers + palm) to enclose the object in a grasp cage.
  """
  robot: Entity = env.scene["robot"]
  obj: Entity = env.scene[object_name]

  # Cube position (B, 3)
  obj_pos = obj.data.root_link_pos_w

  # Palm site position (B, 3)
  site_ids, _ = robot.find_sites(palm_site_name)
  palm_pos = robot.data.site_pose_w[:, site_ids[0], :3]

  # Fingertip body positions (B, K, 3)
  body_ids, _ = robot.find_bodies(fingertip_body_names)
  fingertip_pos = robot.data.body_link_pose_w[:, body_ids, :3]  # (B, K, 3)

  # Collect all hand points: palm + fingertips -> (B, 1 + K, 3)
  all_hand_points = torch.cat([palm_pos.unsqueeze(1), fingertip_pos], dim=1)

  # Distances to object -> (B, 1 + K)
  dists = torch.linalg.norm(all_hand_points - obj_pos.unsqueeze(1), dim=-1)

  # Max distance across hand keypoints -> (B,)
  max_dist = torch.max(dists, dim=-1).values

  return torch.exp(-scale * max_dist)


def object_to_goal_dextrah_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
  object_name: str = "cube",
  beta: float = 15.0,
  min_lift_height: float | None = 0.645,
) -> torch.Tensor:
  """DextrAH goal reaching reward with lifted gating.

  r_obj_goal = 1(z_obj > min_lift_height) * exp(-beta * ||x_obj - x_goal||).
  In DextrAH-G / DextrAH-RGB, goal transport is only rewarded when the object
  has been lifted off the table surface, preventing the policy from hacking
  reward by simply dragging the object across the table.
  """
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))

  obj_pos = obj.data.root_link_pos_w
  goal_pos = command.target_pos

  dist = torch.linalg.norm(obj_pos - goal_pos, dim=-1)
  rew = torch.exp(-beta * dist)
  if min_lift_height is not None:
    is_lifted = (obj_pos[:, 2] > min_lift_height).float()
    rew = rew * is_lifted
  return rew


def object_lift_dextrah_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
  object_name: str = "cube",
  beta: float = 50.0,
) -> torch.Tensor:
  """DextrAH lifting reward: r_lift = exp(-beta * (x_z^obj - x_z^goal)^2)."""
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))

  obj_z = obj.data.root_link_pos_w[:, 2]
  goal_z = command.target_pos[:, 2]

  z_err_sq = torch.square(obj_z - goal_z)
  return torch.exp(-beta * z_err_sq)


def finger_curl_regularization_reward(
  env: ManagerBasedRlEnv,
  joint_pattern: str = "r_hand_.*",
  beta: float = 0.01,
) -> torch.Tensor:
  """DextrAH curl regularization: r_curl = -beta * ||q_hand - q_curl||^2.

  Penalizes over-curling the fingers before grasping. Uses default open-hand
  joint positions as q_curl.
  """
  robot: Entity = env.scene["robot"]
  joint_ids, _ = robot.find_joints(joint_pattern)

  q_curr = robot.data.joint_pos[:, joint_ids]
  q_open = robot.data.default_joint_pos[:, joint_ids]

  err_sq = torch.sum(torch.square(q_curr - q_open), dim=-1)
  return -beta * err_sq


def frozen_arm_posture_reward(
  env: ManagerBasedRlEnv,
  arm_joint_pattern: str = "l_.*",
) -> torch.Tensor:
  """Penalize deviation of the unactuated left arm from its home pose."""
  robot: Entity = env.scene["robot"]
  joint_ids, _ = robot.find_joints(arm_joint_pattern)

  q_curr = robot.data.joint_pos[:, joint_ids]
  q_home = robot.data.default_joint_pos[:, joint_ids]

  return -torch.sum(torch.square(q_curr - q_home), dim=-1)

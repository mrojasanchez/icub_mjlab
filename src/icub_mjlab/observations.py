"""Observation functions for DextrAH-RGB in icub_mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from icub_mjlab.actions.fabric_action import FabricAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def fabric_state(
  env: ManagerBasedRlEnv,
  action_name: str = "fabric",
) -> torch.Tensor:
  """Returns the Geometric Fabric state (q_f, qd_f, qdd_f)."""
  action_term = cast(FabricAction, env.action_manager.get_term(action_name))
  return torch.cat([
    action_term.fabric_q,
    action_term.fabric_qd,
    action_term.fabric_qdd,
  ], dim=-1)


def hand_keypoints_pos(
  env: ManagerBasedRlEnv,
  palm_site_name: str = "r_hand_dh_frame_site",
  fingertip_body_names: tuple[str, ...] = (
    "r_hand_thumb_3",
    "r_hand_index_3",
    "r_hand_middle_3",
    "r_hand_little_3",
  ),
) -> torch.Tensor:
  """Returns 3D positions of the palm and 4 fingertips: (B, 15)."""
  robot: Entity = env.scene["robot"]

  # Palm site position (B, 3)
  site_ids, _ = robot.find_sites(palm_site_name)
  palm_pos = robot.data.site_pose_w[:, site_ids[0], :3]

  # Fingertip body positions (B, 4, 3)
  body_ids, _ = robot.find_bodies(fingertip_body_names)
  fingertip_pos = robot.data.body_link_pose_w[:, body_ids, :3]

  all_points = torch.cat([palm_pos.unsqueeze(1), fingertip_pos], dim=1)
  return all_points.view(env.num_envs, -1)


def object_pose(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
) -> torch.Tensor:
  """Returns 3D position and orientation quaternion of the object: (B, 7)."""
  obj: Entity = env.scene[object_name]
  pos = obj.data.root_link_pos_w
  quat = obj.data.root_link_quat_w
  return torch.cat([pos, quat], dim=-1)


def object_velocity(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
) -> torch.Tensor:
  """Returns 6D linear and angular velocity of the object (privileged): (B, 6)."""
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_vel_w


def target_goal_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Returns 3D target goal position for the object: (B, 3)."""
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  return command.target_pos


def measured_joint_torques(
  env: ManagerBasedRlEnv,
  actuator_names: tuple[str, ...] = (
    # Arm (7)
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "r_wrist_pitch",
    "r_wrist_yaw",
    # Hand (9)
    "r_hand_finger",
    "r_thumb_oppose",
    "r_thumb_proximal",
    "r_thumb_distal",
    "r_index_proximal",
    "r_index_distal",
    "r_middle_proximal",
    "r_middle_distal",
    "r_pinky",
  ),
) -> torch.Tensor:
  """Returns measured actuator torques for the controlled joints (privileged): (B, 16)."""
  robot: Entity = env.scene["robot"]
  actuator_ids, _ = robot.find_actuators(actuator_names)
  return robot.data.actuator_force[:, actuator_ids]

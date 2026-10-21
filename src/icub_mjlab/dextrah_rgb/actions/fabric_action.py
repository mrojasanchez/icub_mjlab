"""Fabric Action Term for dexterous manipulation in mjlab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  matrix_from_quat,
  quat_from_matrix,
)
from icub_mjlab.dextrah_rgb.fabrics.core import GeometricFabricCfg, GeometricFabricSolver

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class FabricActionCfg(ActionTermCfg):
  """Configuration for Geometric Fabric action space."""

  arm_joint_names: tuple[str, ...] = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "r_wrist_pitch",
    "r_wrist_yaw",
  )
  """The 7 arm joints."""

  hand_joint_names: tuple[str, ...] = (
    "r_hand_thumb_0_joint",
    "r_hand_thumb_1_joint",
    "r_hand_index_1_joint",
    "r_hand_middle_1_joint",
  )
  """The 4 actuated hand joints."""

  hand_tendon_names: tuple[str, ...] = (
    "r_hand_finger_tendon",
    "r_thumb_distal_tendon",
    "r_index_distal_tendon",
    "r_middle_distal_tendon",
    "r_pinky_tendon",
  )
  """The 5 actuated hand tendons."""

  site_name: str = "r_hand_dh_frame_site"
  """Palm site name for frame tracking and Jacobian evaluation."""

  delta_pos_scale: float = 0.05
  """Max position delta per step in meters."""

  delta_ori_scale: float = 0.1
  """Max orientation delta per step in radians."""

  fabric_cfg: GeometricFabricCfg = field(default_factory=GeometricFabricCfg)
  """Fabric solver configuration."""

  def build(self, env: ManagerBasedRlEnv) -> FabricAction:
    return FabricAction(self, env)


class FabricAction(ActionTerm):
  """Action term using Geometric Fabrics to compute joint targets."""

  cfg: FabricActionCfg
  _entity: Entity

  def __init__(self, cfg: FabricActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # 1. Resolve controlled joints (11: 7 arm + 4 hand joints) in exact specified order
    all_joint_names = cfg.arm_joint_names + cfg.hand_joint_names
    joint_ids = []
    for name in all_joint_names:
      ids, _ = self._entity.find_joints(name)
      joint_ids.append(ids[0])
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_arm_joints = len(cfg.arm_joint_names)
    self._num_hand_joints = len(cfg.hand_joint_names)
    self._arm_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids[:self._num_arm_joints]]

    # 2. Resolve controlled tendons (5 hand tendons) in exact specified order
    tendon_ids = []
    for name in cfg.hand_tendon_names:
      ids, _ = self._entity.find_tendons(name)
      tendon_ids.append(ids[0])
    self._tendon_ids = torch.tensor(tendon_ids, device=self.device, dtype=torch.long)
    self._num_hand_tendons = len(tendon_ids)

    # Total controlled DOFs in fabric: 7 + 4 + 5 = 16
    self._num_total_dofs = len(self._joint_ids) + len(self._tendon_ids)

    # 2.5 Resolve uncontrolled joints and tendons to hold in default posture
    all_joint_indices = torch.arange(self._entity.num_joints, device=self.device)
    joint_mask = torch.ones(self._entity.num_joints, dtype=torch.bool, device=self.device)
    joint_mask[self._joint_ids] = False
    self._uncontrolled_joint_ids = all_joint_indices[joint_mask]

    all_tendon_indices = torch.arange(self._entity.num_tendons, device=self.device)
    tendon_mask = torch.ones(self._entity.num_tendons, dtype=torch.bool, device=self.device)
    tendon_mask[self._tendon_ids] = False
    self._uncontrolled_tendon_ids = all_tendon_indices[tendon_mask]
    self._uncontrolled_tendon_nom = self._entity.data.tendon_len[0, self._uncontrolled_tendon_ids].clone()

    # 3. Resolve palm site
    site_ids, _ = self._entity.find_sites(cfg.site_name)
    local_site_id = site_ids[0]
    self._site_id = int(self._entity.indexing.site_ids[local_site_id].item())
    self._body_id = int(self._env.sim.mj_model.site_bodyid[self._site_id])

    # 4. Limits and nominal rest posture
    # Tendon order:
    # 0: r_hand_finger_tendon (abduction: range [0.0, 1.0467], nominal = 1.0)
    # 1: r_thumb_distal_tendon (range [0.0, 2.9656], nominal = 0.2)
    # 2: r_index_distal_tendon (range [0.0, 2.9656], nominal = 0.0)
    # 3: r_middle_distal_tendon (range [0.0, 2.9656], nominal = 0.0)
    # 4: r_pinky_tendon (range [0.0, 4.36], nominal = 0.0)
    joint_limits = self._entity.data.soft_joint_pos_limits[0, self._joint_ids]
    tendon_lower = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
    tendon_upper = torch.tensor([1.0467, 2.9656, 2.9656, 2.9656, 4.36], device=self.device)
    tendon_nom = torch.tensor([1.0, 0.2, 0.0, 0.0, 0.0], device=self.device)

    self._joint_lower = torch.cat([joint_limits[:, 0], tendon_lower])
    self._joint_upper = torch.cat([joint_limits[:, 1], tendon_upper])

    joint_nom = self._entity.data.default_joint_pos[0, self._joint_ids]
    self._nominal_q = torch.cat([joint_nom, tendon_nom])

    # 5. Instantiate Geometric Fabric solver
    cfg.fabric_cfg.dt = self._env.step_dt
    self._solver = GeometricFabricSolver(
      cfg=cfg.fabric_cfg,
      joint_lower=self._joint_lower,
      joint_upper=self._joint_upper,
      nominal_q=self._nominal_q,
      num_arm_joints=self._num_arm_joints,
      device=self.device,
    )

    # 6. Action buffers: 6D palm + 5D hand synergies = 11D
    self._action_dim = 11
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    self._nominal_palm_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._nominal_palm_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._nominal_palm_quat[:, 0] = 1.0

    self._desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._desired_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._desired_quat[:, 0] = 1.0
    self._desired_synergy = torch.zeros(self.num_envs, 5, device=self.device)

    self._needs_pose_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    # Fabric state buffers
    self._fabric_q = self._nominal_q.unsqueeze(0).repeat(self.num_envs, 1)
    self._fabric_qd = torch.zeros(self.num_envs, self._num_total_dofs, device=self.device)
    self._fabric_qdd = torch.zeros(self.num_envs, self._num_total_dofs, device=self.device)

    num_joints = len(self._joint_ids)
    self._target_joint_pos = self._fabric_q[:, :num_joints].clone()
    self._target_tendon_len = self._fabric_q[:, num_joints:].clone()

    # Warp buffers for GPU batched Jacobian
    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv

    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)

    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def fabric_q(self) -> torch.Tensor:
    return self._fabric_q

  @property
  def fabric_qd(self) -> torch.Tensor:
    return self._fabric_qd

  @property
  def fabric_qdd(self) -> torch.Tensor:
    return self._fabric_qdd

  def set_fabric_damping(self, damping_gain: float) -> None:
    """Update damping gain during ADR curriculum."""
    self._solver.set_damping_gain(damping_gain)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

    # Lazy initialization of nominal palm pose upon first forward pass after reset
    if self._needs_pose_init.any():
      reset_ids = self._needs_pose_init.nonzero(as_tuple=False).squeeze(-1)
      frame_pos, frame_quat = self._get_palm_pose()
      self._nominal_palm_pos[reset_ids] = frame_pos[reset_ids]
      self._nominal_palm_quat[reset_ids] = frame_quat[reset_ids]
      self._desired_pos[reset_ids] = frame_pos[reset_ids]
      self._desired_quat[reset_ids] = frame_quat[reset_ids]
      self._needs_pose_init[reset_ids] = False

    # 1. Palm target accumulation relative to desired target, bounded within workspace
    palm_delta = actions[:, :6].clone()
    palm_delta[:, :3] *= self.cfg.delta_pos_scale
    palm_delta[:, 3:] *= self.cfg.delta_ori_scale

    target_pos, target_quat = apply_delta_pose(self._desired_pos, self._desired_quat, palm_delta)
    # Bounding box (+/- 0.20 m in XYZ around nominal home pose)
    box = 0.20
    target_pos[:, 0] = torch.clamp(target_pos[:, 0], self._nominal_palm_pos[:, 0] - box, self._nominal_palm_pos[:, 0] + box)
    target_pos[:, 1] = torch.clamp(target_pos[:, 1], self._nominal_palm_pos[:, 1] - box, self._nominal_palm_pos[:, 1] + box)
    target_pos[:, 2] = torch.clamp(target_pos[:, 2], self._nominal_palm_pos[:, 2] - box, self._nominal_palm_pos[:, 2] + box)

    self._desired_pos[:] = target_pos
    self._desired_quat[:] = target_quat

    # 2. Hand synergy target (last 5 dims)
    # Zero actions (actions = 0) command 0 synergy (open hand with r_hand_finger = 1.0)
    self._desired_synergy[:] = torch.clamp(actions[:, 6:], 0.0, 1.0)

    # 3. Solve Fabric ONCE per control step (at step_dt = 0.02s)
    frame_pos, _ = self._get_palm_pose()
    self._point_torch[:] = frame_pos
    self._compute_jacobian()

    # Slice arm DoFs (7)
    jac_p = self._jacp_torch[:, :, self._arm_dof_ids]
    jac_r = self._jacr_torch[:, :, self._arm_dof_ids]

    palm_rot_mat = self._env.sim.data.site_xmat[:, self._site_id].view(-1, 3, 3)
    target_rot_mat = matrix_from_quat(self._desired_quat)

    q_next, qd_next, qdd = self._solver.step(
      q=self._fabric_q,
      qd=self._fabric_qd,
      palm_pos=frame_pos,
      palm_rot_mat=palm_rot_mat,
      target_palm_pos=self._desired_pos,
      target_palm_rot_mat=target_rot_mat,
      target_synergy=self._desired_synergy,
      jac_p=jac_p,
      jac_r=jac_r,
    )

    self._fabric_q[:] = q_next
    self._fabric_qd[:] = qd_next
    self._fabric_qdd[:] = qdd

    num_joints = len(self._joint_ids)
    self._target_joint_pos[:] = q_next[:, :num_joints]
    self._target_tendon_len[:] = q_next[:, num_joints:]

  def apply_actions(self) -> None:
    # Command low-level joint and tendon targets for controlled DOFs
    self._entity.set_joint_position_target(self._target_joint_pos, joint_ids=self._joint_ids)
    self._entity.set_tendon_len_target(self._target_tendon_len, tendon_ids=self._tendon_ids)

    # Hold uncontrolled joints and tendons in default home pose
    if len(self._uncontrolled_joint_ids) > 0:
      default_joint_pos = self._entity.data.default_joint_pos[:, self._uncontrolled_joint_ids]
      self._entity.set_joint_position_target(default_joint_pos, joint_ids=self._uncontrolled_joint_ids)
    if len(self._uncontrolled_tendon_ids) > 0:
      default_tendon_len = self._uncontrolled_tendon_nom.unsqueeze(0).expand(self.num_envs, -1)
      self._entity.set_tendon_len_target(default_tendon_len, tendon_ids=self._uncontrolled_tendon_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    self._raw_actions[env_ids] = 0.0
    self._desired_synergy[env_ids] = 0.0

    self._fabric_q[env_ids] = self._nominal_q.unsqueeze(0)
    self._fabric_qd[env_ids] = 0.0
    self._fabric_qdd[env_ids] = 0.0

    num_joints = len(self._joint_ids)
    self._target_joint_pos[env_ids] = self._nominal_q[:num_joints]
    self._target_tendon_len[env_ids] = self._nominal_q[num_joints:]

    self._needs_pose_init[env_ids] = True

    if len(self._uncontrolled_joint_ids) > 0:
      default_joint_pos = self._entity.data.default_joint_pos[env_ids, :][:, self._uncontrolled_joint_ids]
      self._entity.set_joint_position_target(default_joint_pos, joint_ids=self._uncontrolled_joint_ids, env_ids=env_ids)
    if len(self._uncontrolled_tendon_ids) > 0:
      n = self.num_envs if env_ids == slice(None) else len(env_ids)
      default_tendon_len = self._uncontrolled_tendon_nom.unsqueeze(0).expand(n, -1)
      self._entity.set_tendon_len_target(default_tendon_len, tendon_ids=self._uncontrolled_tendon_ids, env_ids=env_ids)

  def _get_palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._env.sim.data
    pos = data.site_xpos[:, self._site_id]
    xmat = data.site_xmat[:, self._site_id]
    quat = quat_from_matrix(xmat)
    return pos, quat

  def _compute_jacobian(self) -> None:
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )


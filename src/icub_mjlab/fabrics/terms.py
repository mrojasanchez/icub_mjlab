"""Geometric Fabric terms for iCub manipulation.

Implements attractors, joint limit repulsions, and synergy projections
following NVIDIA's Geometric Fabrics framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PalmAttractorCfg:
  """Configuration for palm pose attractor."""
  pos_mass: float = 8.0
  rot_mass: float = 2.0
  pos_gain: float = 40.0
  rot_gain: float = 20.0
  pos_damping: float = 25.0
  rot_damping: float = 15.0
  pos_sharpness: float = 20.0
  rot_sharpness: float = 15.0


@dataclass
class HandSynergyCfg:
  """Configuration for hand finger synergy attractor."""
  mass: float = 4.0
  gain: float = 30.0
  damping: float = 20.0
  sharpness: float = 20.0


@dataclass
class JointLimitCfg:
  """Configuration for joint limit repulsion."""
  metric_scalar: float = 0.05
  max_metric: float = 150.0
  margin_offset: float = 0.05
  soft_relu_gain: float = 12.0
  damping_gain: float = 10.0


@dataclass
class NominalPostureCfg:
  """Configuration for nominal C-space posture attractor (redundancy resolution)."""
  mass: float = 0.5
  gain: float = 5.0
  damping: float = 4.0
  sharpness: float = 10.0


class PalmPoseAttractor:
  """Geometric fabric attractor pulling the robot palm towards a target pose in SE(3)."""

  def __init__(self, cfg: PalmAttractorCfg, device: torch.device):
    self.cfg = cfg
    self.device = device

  def evaluate(
    self,
    palm_pos: torch.Tensor,
    palm_rot_mat: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot_mat: torch.Tensor,
    jac_p: torch.Tensor,
    jac_r: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes pulled-back metric and force in joint space.

    Args:
      palm_pos: Current position, shape (B, 3).
      palm_rot_mat: Current rotation matrix, shape (B, 3, 3).
      target_pos: Target position, shape (B, 3).
      target_rot_mat: Target rotation matrix, shape (B, 3, 3).
      jac_p: Linear Jacobian, shape (B, 3, N).
      jac_r: Rotational Jacobian, shape (B, 3, N).
      joint_vel: Current joint velocities, shape (B, N).

    Returns:
      M_palm: Metric pulled back to joint space, shape (B, N, N).
      f_palm: Force pulled back to joint space, shape (B, N).
    """
    batch_size, num_joints = joint_vel.shape

    # 1. Translation error and acceleration
    pos_err = palm_pos - target_pos
    pos_dist = torch.linalg.norm(pos_err, dim=-1, keepdim=True)
    pos_dir = pos_err / (pos_dist + 1e-6)
    num_arm = jac_p.shape[2]
    v_palm = torch.bmm(jac_p, joint_vel[:, :num_arm].unsqueeze(2)).squeeze(2)

    a_pos = -self.cfg.pos_gain * torch.tanh(self.cfg.pos_sharpness * pos_dist) * pos_dir - self.cfg.pos_damping * v_palm
    M_pos_leaf = self.cfg.pos_mass * torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
    f_pos_leaf = -torch.bmm(M_pos_leaf, a_pos.unsqueeze(2)).squeeze(2)

    # 2. Rotation error (using matrix skew-symmetric logarithm)
    # R_err = R_curr * R_target^T
    r_err_mat = torch.bmm(palm_rot_mat, target_rot_mat.transpose(1, 2))
    rot_err = 0.5 * torch.stack([
      r_err_mat[:, 2, 1] - r_err_mat[:, 1, 2],
      r_err_mat[:, 0, 2] - r_err_mat[:, 2, 0],
      r_err_mat[:, 1, 0] - r_err_mat[:, 0, 1],
    ], dim=-1)
    rot_dist = torch.linalg.norm(rot_err, dim=-1, keepdim=True)
    rot_dir = rot_err / (rot_dist + 1e-6)
    w_palm = torch.bmm(jac_r, joint_vel[:, :num_arm].unsqueeze(2)).squeeze(2)

    a_rot = -self.cfg.rot_gain * torch.tanh(self.cfg.rot_sharpness * rot_dist) * rot_dir - self.cfg.rot_damping * w_palm
    M_rot_leaf = self.cfg.rot_mass * torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
    f_rot_leaf = -torch.bmm(M_rot_leaf, a_rot.unsqueeze(2)).squeeze(2)

    # 3. Pullback to joint space
    # M_arm = J_p^T * M_p * J_p + J_r^T * M_r * J_r
    M_palm_arm = torch.bmm(torch.bmm(jac_p.transpose(1, 2), M_pos_leaf), jac_p) + \
                 torch.bmm(torch.bmm(jac_r.transpose(1, 2), M_rot_leaf), jac_r)

    # f_arm = J_p^T * f_p + J_r^T * f_r
    f_palm_arm = torch.bmm(jac_p.transpose(1, 2), f_pos_leaf.unsqueeze(2)).squeeze(2) + \
                 torch.bmm(jac_r.transpose(1, 2), f_rot_leaf.unsqueeze(2)).squeeze(2)

    # Embed into full joint space
    M_palm = torch.zeros(batch_size, num_joints, num_joints, device=self.device)
    f_palm = torch.zeros(batch_size, num_joints, device=self.device)
    M_palm[:, :num_arm, :num_arm] = M_palm_arm
    f_palm[:, :num_arm] = f_palm_arm

    return M_palm, f_palm


class HandSynergyAttractor:
  """Maps 5D synergy actions to 9 hand actuator coordinates and enforces hand attractor."""

  def __init__(
    self,
    cfg: HandSynergyCfg,
    hand_lower: torch.Tensor,
    hand_upper: torch.Tensor,
    hand_nominal: torch.Tensor,
    num_arm_joints: int,
    device: torch.device,
  ):
    self.cfg = cfg
    self.device = device
    self.num_arm_joints = num_arm_joints
    self.num_hand_actuators = len(hand_lower)
    self.num_total = num_arm_joints + self.num_hand_actuators

    self.hand_lower = hand_lower.to(device)
    self.hand_upper = hand_upper.to(device)
    self.hand_nominal = hand_nominal.to(device)

    # Hand actuators order (9):
    # 0: r_hand_thumb_0_joint (thumb oppose)
    # 1: r_hand_thumb_1_joint (thumb proximal)
    # 2: r_hand_index_1_joint (index proximal)
    # 3: r_hand_middle_1_joint (middle proximal)
    # 4: r_hand_finger_tendon (finger abduction, nominal = 1.0)
    # 5: r_thumb_distal_tendon (thumb distal)
    # 6: r_index_distal_tendon (index distal)
    # 7: r_middle_distal_tendon (middle distal)
    # 8: r_pinky_tendon (pinky / ring)
    #
    # Synergies:
    # s0: Thumb opposition
    # s1: Thumb flexion
    # s2: Index flexion
    # s3: Middle flexion
    # s4: Pinky/Ring flexion
    S = torch.zeros(9, 5, device=device)
    S[0, 0] = 1.0   # Thumb oppose
    S[1, 1] = 1.0   # Thumb proximal
    S[5, 1] = 1.0   # Thumb distal
    S[2, 2] = 1.0   # Index proximal
    S[6, 2] = 1.0   # Index distal
    S[3, 3] = 1.0   # Middle proximal
    S[7, 3] = 1.0   # Middle distal
    S[8, 4] = 1.0   # Pinky/Ring
    # r_hand_finger_tendon (index 4) has S[4, :] = 0 so it stays at nominal 1.0
    self.synergy_matrix = S

  def evaluate(
    self,
    q_hand: torch.Tensor,
    qd_hand: torch.Tensor,
    synergy_target: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes metric and force for hand actuators.

    Args:
      q_hand: Current hand actuator positions, shape (B, 9).
      qd_hand: Current hand actuator velocities, shape (B, 9).
      synergy_target: 5D synergy targets in [0, 1], shape (B, 5).

    Returns:
      M_hand_root: Metric in full joint space, shape (B, N, N).
      f_hand_root: Force in full joint space, shape (B, N).
    """
    batch_size = q_hand.shape[0]

    # Map [0, 1] synergy commands to physical actuator limits starting from hand_nominal
    normalized_hand = torch.bmm(
      self.synergy_matrix.unsqueeze(0).expand(batch_size, -1, -1),
      synergy_target.unsqueeze(2),
    ).squeeze(2).clamp(0.0, 1.0)

    q_target = self.hand_nominal + (self.hand_upper - self.hand_nominal) * normalized_hand

    # Leaf attractor in hand space
    e = q_hand - q_target
    dist = torch.linalg.norm(e, dim=-1, keepdim=True)
    dir_vec = e / (dist + 1e-6)

    a_des = -self.cfg.gain * torch.tanh(self.cfg.sharpness * dist) * dir_vec - self.cfg.damping * qd_hand
    M_leaf = self.cfg.mass * torch.eye(self.num_hand_actuators, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
    f_leaf = -torch.bmm(M_leaf, a_des.unsqueeze(2)).squeeze(2)

    # Embed in full space (zeros on arm joints)
    M_hand_root = torch.zeros(batch_size, self.num_total, self.num_total, device=self.device)
    f_hand_root = torch.zeros(batch_size, self.num_total, device=self.device)

    M_hand_root[:, self.num_arm_joints:, self.num_arm_joints:] = M_leaf
    f_hand_root[:, self.num_arm_joints:] = f_leaf

    return M_hand_root, f_hand_root


class JointLimitRepulsion:
  """Repulsion term preventing joints from crossing physical bounds."""

  def __init__(
    self,
    cfg: JointLimitCfg,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    device: torch.device,
  ):
    self.cfg = cfg
    self.device = device
    self.joint_lower = joint_lower.to(device)
    self.joint_upper = joint_upper.to(device)
    self.min_dist = (cfg.metric_scalar / cfg.max_metric) ** 0.5

  def evaluate(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes joint limit repulsion metric and force.

    Args:
      q: Current joint positions, shape (B, N).
      qd: Current joint velocities, shape (B, N).

    Returns:
      M_lim: Metric contribution, shape (B, N, N).
      f_lim: Force contribution, shape (B, N).
    """
    batch_size, num_joints = q.shape

    # Distances to upper and lower limits
    d_upper = self.joint_upper - q
    d_lower = q - self.joint_lower

    # Effective distance clamped above minimum delta
    delta_up = torch.maximum(
      torch.as_tensor(self.min_dist, device=self.device),
      d_upper - self.cfg.margin_offset,
    )
    delta_low = torch.maximum(
      torch.as_tensor(self.min_dist, device=self.device),
      d_lower - self.cfg.margin_offset,
    )

    # Velocity gates: only engage when moving towards limit
    gate_upper = (qd > 0.0).float()
    gate_lower = (qd < 0.0).float()

    diag_m_up = gate_upper * (self.cfg.metric_scalar / (delta_up ** 2))
    diag_m_low = gate_lower * (self.cfg.metric_scalar / (delta_low ** 2))

    diag_m = (diag_m_up + diag_m_low).clamp(max=self.cfg.max_metric)
    M_lim = torch.diag_embed(diag_m)

    # Accelerations / forces pushing back
    a_up = -self.cfg.soft_relu_gain * torch.ones_like(q) - self.cfg.damping_gain * qd
    a_low = self.cfg.soft_relu_gain * torch.ones_like(q) - self.cfg.damping_gain * qd

    f_up = -diag_m_up * a_up
    f_low = -diag_m_low * a_low
    f_lim = f_up + f_low

    return M_lim, f_lim


class NominalPostureAttractor:
  """Nominal joint space attractor to resolve redundancy and guide nominal posture."""

  def __init__(
    self,
    cfg: NominalPostureCfg,
    nominal_q: torch.Tensor,
    device: torch.device,
  ):
    self.cfg = cfg
    self.device = device
    self.nominal_q = nominal_q.to(device)

  def evaluate(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes nominal posture attractor metric and force.

    Args:
      q: Current joint positions, shape (B, N).
      qd: Current joint velocities, shape (B, N).

    Returns:
      M_nom: Metric contribution, shape (B, N, N).
      f_nom: Force contribution, shape (B, N).
    """
    batch_size, num_joints = q.shape

    e = q - self.nominal_q
    dist = torch.linalg.norm(e, dim=-1, keepdim=True)
    dir_vec = e / (dist + 1e-6)

    a_des = -self.cfg.gain * torch.tanh(self.cfg.sharpness * dist) * dir_vec - self.cfg.damping * qd
    M_nom = self.cfg.mass * torch.eye(num_joints, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
    f_nom = -torch.bmm(M_nom, a_des.unsqueeze(2)).squeeze(2)

    return M_nom, f_nom

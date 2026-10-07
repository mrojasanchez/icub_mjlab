"""Geometric Fabric Core Solver for iCub robot.

Combines geometric and forcing fabric terms, computes energization and damping,
limits accelerations and jerks, and integrates dynamics forward in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch

from icub_mjlab.fabrics.integrator import DisplacementIntegrator
from icub_mjlab.fabrics.terms import (
  HandSynergyAttractor,
  HandSynergyCfg,
  JointLimitCfg,
  JointLimitRepulsion,
  NominalPostureAttractor,
  NominalPostureCfg,
  PalmAttractorCfg,
  PalmPoseAttractor,
)


@dataclass
class GeometricFabricCfg:
  """Configuration for Geometric Fabric Controller."""
  dt: float = 1.0 / 60.0
  damping_gain: float = 10.0   # Damping beta, scaled by ADR from 10 to 20
  regularization_mass: float = 0.1

  # Acceleration and jerk limits (7 arm joints + 9 hand actuators)
  accel_limits: tuple[float, ...] = (
    # Arm (shoulder, elbow, wrist)
    15.0, 15.0, 15.0, 20.0, 25.0, 25.0, 25.0,
    # Hand actuators
    30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
  )
  jerk_limits: tuple[float, ...] = (
    # Arm
    1000.0, 1000.0, 1000.0, 1500.0, 2000.0, 2000.0, 2000.0,
    # Hand
    3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0,
  )

  palm_cfg: PalmAttractorCfg = field(default_factory=PalmAttractorCfg)
  hand_cfg: HandSynergyCfg = field(default_factory=HandSynergyCfg)
  joint_limit_cfg: JointLimitCfg = field(default_factory=JointLimitCfg)
  nominal_cfg: NominalPostureCfg = field(default_factory=NominalPostureCfg)


class GeometricFabricSolver:
  """GPU-batched Geometric Fabric Solver."""

  def __init__(
    self,
    cfg: GeometricFabricCfg,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    nominal_q: torch.Tensor,
    num_arm_joints: int,
    device: torch.device,
  ):
    self.cfg = cfg
    self.device = device
    self.num_arm_joints = num_arm_joints
    self.num_joints = len(joint_lower)
    self.damping_gain = cfg.damping_gain

    self.joint_lower = joint_lower.to(device)
    self.joint_upper = joint_upper.to(device)
    self.nominal_q = nominal_q.to(device)

    # Sub-terms
    self.palm_term = PalmPoseAttractor(cfg.palm_cfg, device)
    self.hand_term = HandSynergyAttractor(
      cfg.hand_cfg,
      joint_lower[num_arm_joints:],
      joint_upper[num_arm_joints:],
      nominal_q[num_arm_joints:],
      num_arm_joints,
      device,
    )
    self.limit_term = JointLimitRepulsion(
      cfg.joint_limit_cfg,
      self.joint_lower,
      self.joint_upper,
      device,
    )
    self.nominal_term = NominalPostureAttractor(
      cfg.nominal_cfg,
      self.nominal_q,
      device,
    )

    # Integrator
    self.integrator = DisplacementIntegrator(cfg.dt)

    # Acceleration and jerk limit tensors
    self.accel_limits = torch.tensor(cfg.accel_limits, device=device).unsqueeze(0)
    self.jerk_limits = torch.tensor(cfg.jerk_limits, device=device).unsqueeze(0)

    # Effective acceleration limit respecting jerk
    # a_max_eff = min(a_max, 0.5 * jerk_max * dt)
    self.effective_accel_limits = torch.minimum(
      self.accel_limits,
      0.5 * self.jerk_limits * cfg.dt,
    )

    # Identity for metric regularization
    self.reg_identity = cfg.regularization_mass * torch.eye(self.num_joints, device=device)

  def set_damping_gain(self, damping_gain: float) -> None:
    """Update damping gain (used by Automatic Domain Randomization curriculum)."""
    self.damping_gain = damping_gain

  def step(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
    palm_pos: torch.Tensor,
    palm_rot_mat: torch.Tensor,
    target_palm_pos: torch.Tensor,
    target_palm_rot_mat: torch.Tensor,
    target_synergy: torch.Tensor,
    jac_p: torch.Tensor,
    jac_r: torch.Tensor,
    dt: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes one integration step of the fabric.

    Args:
      q: Current joint positions, shape (B, N).
      qd: Current joint velocities, shape (B, N).
      palm_pos: Palm frame position, shape (B, 3).
      palm_rot_mat: Palm frame rotation matrix, shape (B, 3, 3).
      target_palm_pos: Desired palm position, shape (B, 3).
      target_palm_rot_mat: Desired palm rotation matrix, shape (B, 3, 3).
      target_synergy: 5D hand synergy targets in [0, 1], shape (B, 5).
      jac_p: Palm translation Jacobian, shape (B, 3, N).
      jac_r: Palm rotation Jacobian, shape (B, 3, N).
      dt: Optional timestep override.

    Returns:
      q_next: Next joint position targets, shape (B, N).
      qd_next: Next joint velocity targets, shape (B, N).
      qdd: Net fabric acceleration, shape (B, N).
    """
    batch_size = q.shape[0]
    timestep = self.cfg.dt if dt is None else dt

    # 1. Evaluate terms
    M_palm, f_palm = self.palm_term.evaluate(
      palm_pos, palm_rot_mat, target_palm_pos, target_palm_rot_mat,
      jac_p, jac_r, qd,
    )

    q_hand = q[:, self.num_arm_joints:]
    qd_hand = qd[:, self.num_arm_joints:]
    M_hand, f_hand = self.hand_term.evaluate(q_hand, qd_hand, target_synergy)

    M_lim, f_lim = self.limit_term.evaluate(q, qd)
    M_nom, f_nom = self.nominal_term.evaluate(q, qd)

    # 2. Combine metrics and forces
    M_total = M_palm + M_hand + M_lim + M_nom + self.reg_identity.unsqueeze(0).expand(batch_size, -1, -1)
    f_total = f_palm + f_hand + f_lim + f_nom

    # 3. Add consistent c-space damping: f_damped = f_total + beta * M * qd
    M_qd = torch.bmm(M_total, qd.unsqueeze(2)).squeeze(2)
    f_damped = f_total + self.damping_gain * M_qd

    # 4. Canonical acceleration: M qdd + f_damped = 0  =>  qdd = - M^{-1} f_damped
    qdd_net = torch.linalg.solve(M_total, -f_damped.unsqueeze(2)).squeeze(2)

    # 5. Acceleration and jerk scaling projection
    qdd_abs = torch.abs(qdd_net)
    scale_ratio = self.effective_accel_limits / (qdd_abs + 1e-6)
    min_scale = torch.clamp(torch.min(scale_ratio, dim=-1, keepdim=True).values, max=1.0)
    qdd_scaled = min_scale * qdd_net

    # 6. Numerical integration
    q_next, qd_next = self.integrator.step(q, qd, qdd_scaled, timestep)

    return q_next, qd_next, qdd_scaled


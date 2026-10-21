"""Numerical integrators for Geometric Fabrics."""

from __future__ import annotations

import torch


class DisplacementIntegrator:
  """Second-order displacement integrator for fabric dynamical systems.
  
  Advances joint position and velocity according to:
      q_{t+dt} = q_t + dt * qd_t + 0.5 * dt^2 * qdd_t
      qd_{t+dt} = qd_t + dt * qdd_t
  """

  def __init__(self, dt: float):
    self.dt = dt

  def step(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    dt: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Step fabric state forward in time.

    Args:
      q: Generalized coordinates, shape (batch_size, num_joints).
      qd: Generalized velocities, shape (batch_size, num_joints).
      qdd: Generalized accelerations, shape (batch_size, num_joints).
      dt: Timestep (optional, defaults to self.dt).

    Returns:
      Tuple of (q_next, qd_next).
    """
    timestep = self.dt if dt is None else dt
    q_next = q + timestep * qd + 0.5 * (timestep**2) * qdd
    qd_next = qd + timestep * qdd
    return q_next, qd_next


class SymplecticEulerIntegrator:
  """First-order symplectic Euler integrator for fabric dynamical systems."""

  def __init__(self, dt: float):
    self.dt = dt

  def step(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    dt: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    timestep = self.dt if dt is None else dt
    qd_next = qd + timestep * qdd
    q_next = q + timestep * qd_next
    return q_next, qd_next


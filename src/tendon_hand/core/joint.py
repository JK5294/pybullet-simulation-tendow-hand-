"""Joint dataclass: revolute joint with limits and passive dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Joint:
    """Represents a revolute joint in the hand.

    Stores state and computes passive torque from stiffness/damping.
    """

    id: str
    parent_link: str = ""
    child_link: str = ""
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    angle: float = 0.0  # rad
    velocity: float = 0.0  # rad/s

    lower_limit: float = -np.inf
    upper_limit: float = np.inf

    passive_stiffness: float = 0.0  # Nm/rad
    passive_damping: float = 0.0  # Nm/(rad/s)
    return_spring_k: float = 0.0  # Nm/rad, spring pulling back to rest
    rest_angle: float = 0.0  # rad, angle where return spring is zero
    coulomb_friction: float = 0.0  # Nm

    # External torque accumulator (set by tendon + contact)
    _external_torque: float = field(default=0.0, repr=False)

    def clamp_angle(self, angle: float) -> float:
        """Clamp angle to joint limits."""
        return float(np.clip(angle, self.lower_limit, self.upper_limit))

    def compute_passive_torque(self) -> float:
        """Compute passive torque from stiffness, damping, return spring, friction."""
        tau = 0.0
        # Stiffness
        if self.passive_stiffness > 0:
            tau -= self.passive_stiffness * self.angle
        # Damping
        if self.passive_damping > 0:
            tau -= self.passive_damping * self.velocity
        # Return spring
        if self.return_spring_k > 0:
            tau -= self.return_spring_k * (self.angle - self.rest_angle)
        # Coulomb friction (opposes velocity)
        if self.coulomb_friction > 0 and abs(self.velocity) > 1e-6:
            tau -= self.coulomb_friction * np.sign(self.velocity)
        return tau

    def get_net_torque(self) -> float:
        """Net torque = external + passive."""
        return self._external_torque + self.compute_passive_torque()

    def apply_torque(self, torque: float) -> None:
        """Add external torque (from tendon, contact, etc.)."""
        self._external_torque += torque

    def reset_torque(self) -> None:
        """Clear accumulated external torque."""
        self._external_torque = 0.0

    def step(self, dt: float, inertia: float = 1e-6) -> None:
        """Quasi-static joint update: velocity ∝ net torque, position integrates.

        Args:
            dt: timestep
            inertia: effective rotational inertia (very small for quasi-static)
        """
        tau_net = self.get_net_torque()
        # Quasi-static: velocity proportional to torque
        self.velocity = tau_net / max(inertia, 1e-12) * dt
        # Clamp to reasonable value
        self.velocity = float(np.clip(self.velocity, -10.0, 10.0))
        self.angle += self.velocity * dt
        self.angle = self.clamp_angle(self.angle)
        self.reset_torque()

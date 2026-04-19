"""Compliance models: joint passive torque computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ComplianceModel(Protocol):
    """Protocol for computing joint passive torque."""

    def compute_torque(self, angle: float, velocity: float, rest_angle: float = 0.0) -> float:
        ...


@dataclass
class PassiveStiffnessModel:
    """Standard passive compliance: stiffness + damping + return spring."""

    stiffness: float = 0.0  # Nm/rad
    damping: float = 0.0  # Nm/(rad/s)
    return_spring_k: float = 0.0  # Nm/rad
    coulomb_friction: float = 0.0  # Nm

    def compute_torque(self, angle: float, velocity: float, rest_angle: float = 0.0) -> float:
        tau = 0.0
        if self.stiffness > 0:
            tau -= self.stiffness * angle
        if self.damping > 0:
            tau -= self.damping * velocity
        if self.return_spring_k > 0:
            tau -= self.return_spring_k * (angle - rest_angle)
        if self.coulomb_friction > 0 and abs(velocity) > 1e-6:
            tau -= self.coulomb_friction * np.sign(velocity)
        return tau


@dataclass
class NoComplianceModel:
    """No passive torque."""

    def compute_torque(self, angle: float, velocity: float, rest_angle: float = 0.0) -> float:
        return 0.0

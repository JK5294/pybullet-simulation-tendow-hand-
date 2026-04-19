"""Tendon dataclass: cable with routing, tension, and elasticity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class RoutingElement:
    """Describes how a tendon passes through a joint."""

    joint_id: str
    moment_arm: float  # m
    direction_sign: int = 1  # +1 or -1
    routing_loss: float = 0.0  # 0.0 ~ 1.0, fraction of tension lost


class RoutingModel(Protocol):
    """Protocol for tension propagation along routing."""

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        """Return {joint_id: local_tension} after propagation."""
        ...


@dataclass
class SimpleRoutingLossModel:
    """Simplified routing loss: T_{j+1} = T_j * (1 - λ_j)."""

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        tensions: dict[str, float] = {}
        T = base_tension
        for elem in routing_elements:
            T = T * (1.0 - elem.routing_loss)
            tensions[elem.joint_id] = max(0.0, T)
        return tensions


@dataclass
class Tendon:
    """Represents a cable/tendon connecting an actuator to joints.

    Stores state and computes length/tension based on actuator displacement
    and joint angles.
    """

    id: str
    actuator_id: str
    routed_joints: list[RoutingElement] = field(default_factory=list)

    rest_length: float = 0.18  # m
    current_length: float = 0.18  # m
    tension: float = 0.0  # N
    min_tension: float = 0.0  # cables can't push
    max_tension: float | None = None  # N

    # Elasticity / hysteresis parameters
    elasticity_k: float = 120.0  # N/m
    damping_c: float = 0.2  # N/(m/s)
    hysteresis_eta: float = 1.0  # 1.0 = no hysteresis, <1.0 = releasing tension drop

    # Internal state
    _prev_cmd_length: float = field(default=0.0, repr=False)
    _last_direction: str = field(default="pull", repr=False)  # "pull" or "release"

    def compute_length_from_joints(self, joint_angles: dict[str, float]) -> float:
        """Compute tendon length from joint angles and routing.

        l = rest_length + Σ (sign_i * moment_arm_i * θ_i)
        """
        delta = 0.0
        for elem in self.routed_joints:
            theta = joint_angles.get(elem.joint_id, 0.0)
            delta += elem.direction_sign * elem.moment_arm * theta
        return self.rest_length + delta

    def compute_tension(
        self,
        commanded_length: float,
        actual_length: float,
        dt: float,
    ) -> float:
        """Quasi-static tendon tension: T = max(0, k*(l_cmd - l_act) - c*dl).

        Args:
            commanded_length: target tendon length from actuator displacement
            actual_length: current length from joint configuration
            dt: timestep
        """
        delta_l = commanded_length - actual_length
        dl = (commanded_length - self._prev_cmd_length) / max(dt, 1e-9)

        # Direction detection for hysteresis
        direction = "pull" if delta_l >= 0 else "release"
        if direction == "release" and self._last_direction == "pull":
            # Just started releasing
            pass
        self._last_direction = direction

        eta = 1.0 if direction == "pull" else self.hysteresis_eta

        T = self.elasticity_k * delta_l * eta - self.damping_c * dl
        T = max(self.min_tension, T)
        if self.max_tension is not None:
            T = min(T, self.max_tension)

        self._prev_cmd_length = commanded_length
        self.tension = T
        return T

    def get_joint_torques(
        self,
        joint_angles: dict[str, float],
        routing_model: RoutingModel | None = None,
    ) -> dict[str, float]:
        """Compute joint torques from current tension and routing.

        Returns {joint_id: torque (Nm)}.
        """
        if routing_model is None:
            routing_model = SimpleRoutingLossModel()

        torques: dict[str, float] = {}
        tensions = routing_model.propagate(self.tension, self.routed_joints)
        for elem in self.routed_joints:
            T_local = tensions.get(elem.joint_id, 0.0)
            tau = elem.moment_arm * T_local * elem.direction_sign
            torques[elem.joint_id] = tau
        return torques

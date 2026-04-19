"""Finger class: aggregates joints, tendons, and routing for one finger."""

from __future__ import annotations

from dataclasses import dataclass, field

from tendon_hand.core.joint import Joint
from tendon_hand.core.tendon import Tendon, RoutingElement


@dataclass
class Finger:
    """A single finger with joints, tendons, and routing table.

    Provides finger-level step/update and observation.
    """

    id: str
    joints: list[Joint] = field(default_factory=list)
    tendons: list[Tendon] = field(default_factory=list)

    def get_joint(self, joint_id: str) -> Joint | None:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def get_tendon(self, tendon_id: str) -> Tendon | None:
        for t in self.tendons:
            if t.id == tendon_id:
                return t
        return None

    def get_joint_angles(self) -> dict[str, float]:
        return {j.id: j.angle for j in self.joints}

    def get_joint_velocities(self) -> dict[str, float]:
        return {j.id: j.velocity for j in self.joints}

    def get_observation(self) -> dict:
        return {
            "joint_angles": self.get_joint_angles(),
            "joint_velocities": self.get_joint_velocities(),
            "tendon_tensions": {t.id: t.tension for t in self.tendons},
        }

    def reset(self) -> None:
        for j in self.joints:
            j.angle = 0.0
            j.velocity = 0.0
            j.reset_torque()
        for t in self.tendons:
            t.tension = 0.0
            t.current_length = t.rest_length

    def step(self, dt: float) -> None:
        """Step all joints (tendon torques should be applied before this)."""
        for j in self.joints:
            j.step(dt)

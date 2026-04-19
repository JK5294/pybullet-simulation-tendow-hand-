"""Hand class: aggregates fingers, palm, and actuators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tendon_hand.core.actuator import Actuator
from tendon_hand.core.finger import Finger
from tendon_hand.core.palm import Palm


@dataclass
class Hand:
    """Complete hand model.

    Aggregates fingers, palm joints, and actuators.
    Provides unified step and observation interface.
    """

    id: str = "hand"
    fingers: list[Finger] = field(default_factory=list)
    palm: Palm = field(default_factory=Palm)
    actuators: list[Actuator] = field(default_factory=list)

    # Global state
    time: float = 0.0

    @classmethod
    def from_config(cls, config_path: str) -> "Hand":
        """Construct a Hand from a YAML config file.

        This is a placeholder; full implementation in io.config_loader.
        """
        from tendon_hand.io.config_loader import load_hand_config

        return load_hand_config(config_path)

    def get_finger(self, finger_id: str) -> Finger | None:
        for f in self.fingers:
            if f.id == finger_id:
                return f
        return None

    def get_actuator(self, actuator_id: str) -> Actuator | None:
        for a in self.actuators:
            if a.id == actuator_id:
                return a
        return None

    def get_motor_dict(self) -> dict[str, float]:
        """Return {actuator_id: commanded_angle} for all actuators."""
        return {a.id: a.commanded_angle for a in self.actuators}

    def apply_motor_commands(self, cmds: dict[str, float], mode: str = "position") -> None:
        """Apply motor commands to actuators.

        Args:
            cmds: {actuator_id: value}
            mode: "position" | "velocity" | "effort"
        """
        for aid, val in cmds.items():
            act = self.get_actuator(aid)
            if act is not None:
                act.apply_command(val, mode=mode)

    def get_observation(self) -> dict[str, Any]:
        """Return full hand observation."""
        obs: dict[str, Any] = {
            "time": self.time,
            "motor_angles": {a.id: a.motor_angle for a in self.actuators},
            "motor_commands": {a.id: a.commanded_angle for a in self.actuators},
        }
        for finger in self.fingers:
            obs[f"finger_{finger.id}"] = finger.get_observation()
        obs["palm"] = {
            "joint_angles": self.palm.get_joint_angles(),
        }
        return obs

    def reset(self) -> None:
        self.time = 0.0
        for a in self.actuators:
            a.motor_angle = 0.0
            a.commanded_angle = 0.0
            a.motor_velocity = 0.0
        for f in self.fingers:
            f.reset()
        self.palm.reset()

    def step(self, dt: float) -> None:
        """Step all actuators, then all fingers and palm.

        Note: tendon torques should be applied BEFORE calling this.
        See HandController for the full step loop including tendon physics.
        """
        self.time += dt
        for a in self.actuators:
            a.step(dt)
        for f in self.fingers:
            f.step(dt)
        self.palm.step(dt)

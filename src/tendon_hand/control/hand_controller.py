"""Hand controller: high-level API for hand poses and motor commands.

This is the primary user-facing API. It wraps the core Hand model
and the transmission model to provide convenient finger-level control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tendon_hand.core.hand import Hand
from tendon_hand.core.actuator import Actuator
from tendon_hand.core.models.transmission import CascadeTransmissionModel


@dataclass
class HandController:
    """High-level controller for the V3 dexterous hand.

    Provides finger-level and whole-hand pose control.
    All values are physical motor values (rad), matching real hardware.
    """

    hand: Hand = field(default_factory=Hand)
    transmission: CascadeTransmissionModel = field(
        default_factory=CascadeTransmissionModel
    )

    # Convenience mapping: finger name → motor indices
    _FINGER_MAP: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "thumb": (0, 1, 2),
            "index": (3, 4, 5),
            "middle": (6, 7, 8),
            "ring": (9, 10, 11),
            "pinky": (12, 13, 14),
        },
        repr=False,
    )

    def __post_init__(self):
        """Ensure hand has actuators matching transmission model."""
        existing_ids = {a.id for a in self.hand.actuators}
        for name in self.transmission.motor_names:
            if name not in existing_ids:
                self.hand.actuators.append(
                    Actuator(id=name, motor_torque_limit=0.5, spool_radius=0.008)
                )

    def set_finger(self, name: str, m1: float, m2: float, m3: float) -> None:
        """Set 3 motor values for a single finger.

        Args:
            name: "thumb" | "index" | "middle" | "ring" | "pinky"
            m1, m2, m3: motor values (physical, rad-equivalent)
        """
        indices = self._FINGER_MAP.get(name)
        if indices is None:
            raise ValueError(f"Unknown finger: {name}")
        i1, i2, i3 = indices
        motor_names = self.transmission.motor_names
        self.hand.apply_motor_commands({
            motor_names[i1]: float(m1),
            motor_names[i2]: float(m2),
            motor_names[i3]: float(m3),
        })

    def set_thumb(self, m1: float, m2: float, m3: float) -> None:
        self.set_finger("thumb", m1, m2, m3)

    def set_index(self, m1: float, m2: float, m3: float) -> None:
        self.set_finger("index", m1, m2, m3)

    def set_middle(self, m1: float, m2: float, m3: float) -> None:
        self.set_finger("middle", m1, m2, m3)

    def set_ring(self, m1: float, m2: float, m3: float) -> None:
        self.set_finger("ring", m1, m2, m3)

    def set_pinky(self, m1: float, m2: float, m3: float) -> None:
        self.set_finger("pinky", m1, m2, m3)

    def set_wrist(self, m4: float, m5: float) -> None:
        """Set wrist motors."""
        motor_names = self.transmission.motor_names
        self.hand.apply_motor_commands({
            motor_names[15]: float(m4),
            motor_names[16]: float(m5),
        })

    def set_motor(self, name: str, value: float) -> None:
        """Set a single motor by name."""
        if name not in self.transmission.motor_names:
            raise ValueError(f"Unknown motor: {name}")
        self.hand.apply_motor_commands({name: float(value)})

    def set_all_motors(self, values: np.ndarray) -> None:
        """Set all 17 motors from an array."""
        assert values.shape == (len(self.transmission.motor_names),)
        cmds = {
            name: float(values[i])
            for i, name in enumerate(self.transmission.motor_names)
        }
        self.hand.apply_motor_commands(cmds)

    def open_pose(self) -> None:
        """Open hand: all motors to zero."""
        self.hand.reset()

    def close_pose(self) -> None:
        """Close hand into a fist."""
        close = {
            "thumb_m1": 2.5, "thumb_m2": 1.0, "thumb_m3": -0.5,
            "index_m1": 5.8, "index_m2": 4.2, "index_m3": -0.4,
            "middle_m1": 5.8, "middle_m2": 4.2, "middle_m3": -0.3,
            "ring_m1": 5.8, "ring_m2": 4.2, "ring_m3": -0.4,
            "pinky_m1": 5.8, "pinky_m2": 4.2, "pinky_m3": -0.4,
            "wrist_m4": 0.0, "wrist_m5": 0.0,
        }
        self.hand.apply_motor_commands(close)

    def get_joint_targets(self) -> dict[str, float]:
        """Map current motor values to joint targets via transmission model."""
        action = np.zeros(len(self.transmission.motor_names))
        for i, name in enumerate(self.transmission.motor_names):
            act = self.hand.get_actuator(name)
            if act is not None:
                action[i] = self.transmission.normalize_motor(
                    act.commanded_angle, i
                )
        return self.transmission.map(action)

    def get_finger_joints(self, name: str) -> dict[str, float]:
        """Return joint targets for a single finger."""
        all_joints = self.get_joint_targets()
        prefix = {
            "thumb": "finger_1",
            "index": "finger_2",
            "middle": "finger_3",
            "ring": "finger_4",
            "pinky": "finger_5",
        }[name]
        knuckle_idx = {
            "thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5,
        }[name]
        return {
            k: v for k, v in all_joints.items()
            if prefix in k or f"knuckle_{knuckle_idx}" in k
        }

    def get_observation(self) -> dict[str, Any]:
        """Return full hand observation."""
        obs = self.hand.get_observation()
        obs["joint_targets"] = self.get_joint_targets()
        return obs

    def step(self, dt: float) -> None:
        """Step the hand (actuators + joints)."""
        self.hand.step(dt)

"""Transmission models: motor → tendon displacement → joint angle mapping.

This module implements the cable-driven cascade logic for underactuated
tendon-driven fingers. It is the core of the V3 hand actuation model.

Key concept: motor1 pulls the distal phalanx first; overflow cascades to
middle → proximal → palm. motor2 pulls the middle phalanx; distal follows
middle; overflow cascades to proximal → palm. Both motors' effects are
combined per joint by max().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class JointLimits:
    """Joint angle limits for a finger or thumb."""

    link3: tuple[float, float] = (-0.087, 1.650)
    link2: tuple[float, float] = (-0.087, 1.650)
    link1: tuple[float, float] = (-0.087, 2.268)
    knuckle: tuple[float, float] = (-0.436, 0.436)
    palm: tuple[float, float] = (0.0, 0.087)


@dataclass(frozen=True)
class ThumbJointLimits:
    """Joint angle limits for the thumb (finger_1)."""

    link2: tuple[float, float] = (-0.217, 1.650)
    link1: tuple[float, float] = (-0.087, 2.268)
    knuckle: tuple[float, float] = (-2.260, 1.750)
    palm: tuple[float, float] = (0.0, 0.087)


@dataclass(frozen=True)
class WristTendonCompensation:
    """Motor-space compensation for wrist-induced tendon pull.

    In the real hand the finger motors sit behind the wrist. When the wrist
    rotates, routed tendons can be pulled as if the finger motors had closed.
    The compensation sends the finger motors in the opposite direction before
    the cascade model computes joint targets.

    Formula:
        motor_delta_rad = -(wrist_moment_arm / motor_spool_radius) * wrist_rad

    The default coefficients are conservative placeholders in motor-rad per
    wrist-rad. Calibrate them from measured routing moment arms and spool radii
    for the final hardware.
    """

    enabled: bool = True
    gains: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "thumb_m1": (0.20, 0.20),
        "thumb_m2": (0.12, 0.12),
        "index_m1": (0.30, 0.00),
        "index_m2": (0.20, 0.00),
        "middle_m1": (0.30, 0.00),
        "middle_m2": (0.20, 0.00),
        "ring_m1": (0.00, 0.30),
        "ring_m2": (0.00, 0.20),
        "pinky_m1": (0.00, 0.30),
        "pinky_m2": (0.00, 0.20),
    })

    @classmethod
    def from_geometry(
        cls,
        routing_moment_arms: dict[str, tuple[float, float]],
        *,
        spool_radius: float,
        enabled: bool = True,
    ) -> "WristTendonCompensation":
        """Build compensation gains from tendon routing geometry.

        Args:
            routing_moment_arms: motor name -> (wrist_m4 arm, wrist_m5 arm), meters.
            spool_radius: finger motor spool radius in meters.
            enabled: whether to apply the compensation.
        """
        if spool_radius <= 0.0:
            raise ValueError("spool_radius must be positive")
        gains = {
            motor_name: (float(m4_arm) / spool_radius, float(m5_arm) / spool_radius)
            for motor_name, (m4_arm, m5_arm) in routing_moment_arms.items()
        }
        return cls(enabled=enabled, gains=gains)

    def delta_for_motor(self, motor_name: str, wrist_m4: float, wrist_m5: float) -> float:
        """Return physical motor-angle compensation for one finger motor."""
        if not self.enabled:
            return 0.0
        gain_m4, gain_m5 = self.gains.get(motor_name, (0.0, 0.0))
        return -(gain_m4 * float(wrist_m4) + gain_m5 * float(wrist_m5))


class TransmissionModel(Protocol):
    """Protocol for motor → joint angle mapping."""

    def map(self, motor_values: dict[str, float]) -> dict[str, float]:
        """Return {joint_id: target_angle (rad)}."""
        ...


def _cascade_motor1(
    m1: float,
    link3_lim: tuple[float, float],
    link2_lim: tuple[float, float],
    link1_lim: tuple[float, float],
    palm_lim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Motor1 cascade: distal active → middle → proximal → palm.

    Returns (link3, link2, link1, palm).
    """
    link3 = float(np.clip(m1, *link3_lim))
    remaining = m1 - link3
    link2 = float(np.clip(remaining, *link2_lim))
    remaining = remaining - link2
    link1 = float(np.clip(remaining, *link1_lim))
    remaining = remaining - link1
    palm = float(np.clip(remaining, *palm_lim))
    return link3, link2, link1, palm


def _cascade_motor2(
    m2: float,
    link3_lim: tuple[float, float],
    link2_lim: tuple[float, float],
    link1_lim: tuple[float, float],
    palm_lim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Motor2 cascade: middle active, distal follows middle → proximal → palm.

    Returns (link3, link2, link1, palm).
    """
    link2 = float(np.clip(m2, *link2_lim))
    link3 = float(np.clip(link2, *link3_lim))  # distal follows middle
    remaining = m2 - link2
    link1 = float(np.clip(remaining, *link1_lim))
    remaining = remaining - link1
    palm = float(np.clip(remaining, *palm_lim))
    return link3, link2, link1, palm


def _cascade_thumb_motor1(
    m1: float,
    link2_lim: tuple[float, float],
    link1_lim: tuple[float, float],
    palm_lim: tuple[float, float],
) -> tuple[float, float, float]:
    """Thumb motor1: distal (link2) active → proximal (link1) → palm."""
    link2 = float(np.clip(m1, *link2_lim))
    remaining = m1 - link2
    link1 = float(np.clip(remaining, *link1_lim))
    remaining = remaining - link1
    palm = float(np.clip(remaining, *palm_lim))
    return link2, link1, palm


def _cascade_thumb_motor2(
    m2: float,
    link2_lim: tuple[float, float],
    link1_lim: tuple[float, float],
    palm_lim: tuple[float, float],
) -> tuple[float, float, float]:
    """Thumb motor2: proximal (link1) active, distal (link2) follows → palm."""
    link1 = float(np.clip(m2, *link1_lim))
    link2 = float(np.clip(link1, *link2_lim))  # distal follows proximal
    remaining = m2 - link1
    palm = float(np.clip(remaining, *palm_lim))
    return link2, link1, palm


@dataclass
class FingerTransmission:
    """Transmission model for a single finger (3-link + knuckle + palm)."""

    finger_id: str
    joint_prefix: str
    limits: JointLimits = field(default_factory=JointLimits)

    # Sign correction for finger_2/4/5 (different mesh frame from finger_3)
    motor_sign: tuple[int, int] = (1, 1)
    # Knuckle sign: multiplier for motor3 → knuckle joint
    knuckle_sign: int = 1

    def map(self, m1: float, m2: float, m3: float) -> dict[str, float]:
        """Map 3 motor values to joint angles for this finger.

        Args:
            m1: motor1 value (physical, rad-equivalent)
            m2: motor2 value
            m3: motor3 value (knuckle)

        Returns:
            {joint_name: target_angle}
        """
        s1, s2 = self.motor_sign
        _m1 = s1 * m1
        _m2 = s2 * m2

        j3_1, j2_1, j1_1, p1 = _cascade_motor1(
            _m1, self.limits.link3, self.limits.link2,
            self.limits.link1, self.limits.palm,
        )
        j3_2, j2_2, j1_2, p2 = _cascade_motor2(
            _m2, self.limits.link3, self.limits.link2,
            self.limits.link1, self.limits.palm,
        )

        targets: dict[str, float] = {
            f"{self.joint_prefix}_link3_joint": max(j3_1, j3_2),
            f"{self.joint_prefix}_link2_joint": max(j2_1, j2_2),
            f"{self.joint_prefix}_link1_joint": max(j1_1, j1_2),
        }

        # Palm overflow
        palm_val = max(p1, p2)
        if palm_val > 0:
            targets[f"palm_{self._palm_index()}_joint"] = palm_val

        # Knuckle
        knuckle = self.knuckle_sign * float(np.clip(m3, *self.limits.knuckle))
        targets[f"knuckle_{self._knuckle_index()}_joint"] = knuckle

        return targets

    def _palm_index(self) -> int:
        """Map finger prefix to palm joint index."""
        mapping = {
            "finger_1": "1",
            "finger_2": "3",
            "finger_3": "3",
            "finger_4": "4",
            "finger_5": "4",
        }
        return mapping.get(self.joint_prefix, "3")

    def _knuckle_index(self) -> int:
        """Map finger prefix to knuckle joint index."""
        mapping = {
            "finger_1": 1,
            "finger_2": 2,
            "finger_3": 3,
            "finger_4": 4,
            "finger_5": 5,
        }
        return mapping.get(self.joint_prefix, 2)


@dataclass
class ThumbTransmission:
    """Transmission model for the thumb (2-link + knuckle + palm)."""

    limits: ThumbJointLimits = field(default_factory=ThumbJointLimits)
    knuckle_sign: int = 1

    def map(self, m1: float, m2: float, m3: float) -> dict[str, float]:
        """Map 3 motor values to thumb joint angles."""
        j2_1, j1_1, p1 = _cascade_thumb_motor1(
            m1, self.limits.link2, self.limits.link1, self.limits.palm,
        )
        j2_2, j1_2, p2 = _cascade_thumb_motor2(
            m2, self.limits.link2, self.limits.link1, self.limits.palm,
        )

        targets: dict[str, float] = {
            "finger_1_link2_joint": max(j2_1, j2_2),
            "finger_1_link1_joint": max(j1_1, j1_2),
            "palm_1_joint": max(p1, p2),
            "knuckle_1_joint": self.knuckle_sign * float(np.clip(m3, *self.limits.knuckle)),
        }
        return targets


@dataclass
class CascadeTransmissionModel:
    """Full hand transmission model using the V3 cable-driven cascade logic.

    Maps 17 motor values to 22 joint targets.
    """

    thumb: ThumbTransmission = field(default_factory=ThumbTransmission)
    index: FingerTransmission = field(default_factory=lambda: FingerTransmission(
        finger_id="index", joint_prefix="finger_2",
        motor_sign=(1, 1), knuckle_sign=1,
    ))
    middle: FingerTransmission = field(default_factory=lambda: FingerTransmission(
        finger_id="middle", joint_prefix="finger_3",
        motor_sign=(1, 1), knuckle_sign=1,
    ))
    ring: FingerTransmission = field(default_factory=lambda: FingerTransmission(
        finger_id="ring", joint_prefix="finger_4",
        motor_sign=(1, 1), knuckle_sign=1,
    ))
    pinky: FingerTransmission = field(default_factory=lambda: FingerTransmission(
        finger_id="pinky", joint_prefix="finger_5",
        motor_sign=(1, 1), knuckle_sign=1,
    ))
    wrist_compensation: WristTendonCompensation = field(default_factory=WristTendonCompensation)

    # Motor names in order
    motor_names: list[str] = field(default_factory=lambda: [
        "thumb_m1", "thumb_m2", "thumb_m3",
        "index_m1", "index_m2", "index_m3",
        "middle_m1", "middle_m2", "middle_m3",
        "ring_m1", "ring_m2", "ring_m3",
        "pinky_m1", "pinky_m2", "pinky_m3",
        "wrist_m4", "wrist_m5",
    ])

    # Normalized ranges: action ∈ [-1,1] → motor ∈ [-hi, +hi]
    motor_ranges: list[tuple[float, float]] = field(default_factory=lambda: [
        (-4.0, 4.0),   # thumb_m1
        (-3.0, 3.0),   # thumb_m2
        (-2.5, 2.5),   # thumb_m3
        (-6.0, 6.0),   # index_m1
        (-5.0, 5.0),   # index_m2
        (-0.5, 0.5),   # index_m3
        (-6.0, 6.0),   # middle_m1
        (-5.0, 5.0),   # middle_m2
        (-0.5, 0.5),   # middle_m3
        (-6.0, 6.0),   # ring_m1
        (-5.0, 5.0),   # ring_m2
        (-0.5, 0.5),   # ring_m3
        (-6.0, 6.0),   # pinky_m1
        (-5.0, 5.0),   # pinky_m2
        (-0.5, 0.5),   # pinky_m3
        (-0.1, 0.1),   # wrist_m4
        (-0.1, 0.1),   # wrist_m5
    ])

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """Map [-1, 1] motor actions to physical motor values."""
        motors = np.zeros(len(self.motor_names), dtype=np.float64)
        for i, (lo, hi) in enumerate(self.motor_ranges):
            motors[i] = float(np.clip(action[i], -1.0, 1.0)) * hi
        return motors

    def normalize_motor(self, motor_val: float, idx: int) -> float:
        """Map a physical motor value back to [-1, 1]."""
        hi = self.motor_ranges[idx][1]
        if hi < 1e-6:
            return 0.0
        return float(np.clip(motor_val / hi, -1.0, 1.0))

    def apply_wrist_compensation(self, motors: np.ndarray) -> np.ndarray:
        """Return physical motor values with wrist tendon compensation applied."""
        compensated = np.asarray(motors, dtype=np.float64).copy()
        wrist_m4 = float(compensated[15])
        wrist_m5 = float(compensated[16])
        for idx, motor_name in enumerate(self.motor_names[:15]):
            compensated[idx] += self.wrist_compensation.delta_for_motor(
                motor_name,
                wrist_m4,
                wrist_m5,
            )
            lo, hi = self.motor_ranges[idx]
            compensated[idx] = float(np.clip(compensated[idx], lo, hi))
        return compensated

    def wrist_compensation_report(self, motor_values: dict[str, float]) -> dict[str, float]:
        """Return per-motor compensation deltas for physical motor values."""
        wrist_m4 = float(motor_values.get("wrist_m4", 0.0))
        wrist_m5 = float(motor_values.get("wrist_m5", 0.0))
        return {
            motor_name: self.wrist_compensation.delta_for_motor(motor_name, wrist_m4, wrist_m5)
            for motor_name in self.motor_names[:15]
        }

    def map(self, motor_action: np.ndarray) -> dict[str, float]:
        """Convert 17-D normalized motor action → dict of joint target positions."""
        motors = self.apply_wrist_compensation(self.denormalize(motor_action))
        targets: dict[str, float] = {}

        # Thumb
        t = self.thumb.map(motors[0], motors[1], motors[2])
        targets.update(t)

        # Index
        t = self.index.map(motors[3], motors[4], motors[5])
        targets.update(t)

        # Middle
        t = self.middle.map(motors[6], motors[7], motors[8])
        targets.update(t)

        # Ring
        t = self.ring.map(motors[9], motors[10], motors[11])
        targets.update(t)

        # Pinky
        t = self.pinky.map(motors[12], motors[13], motors[14])
        targets.update(t)

        # Wrist (motor4/motor5 directly drive palm_3/palm_4)
        palm_3 = max(targets.get("palm_3_joint", 0.0),
                     float(np.clip(motors[15], 0.0, 0.087)))
        palm_4 = max(targets.get("palm_4_joint", 0.0),
                     float(np.clip(motors[16], 0.0, 0.087)))
        targets["palm_3_joint"] = palm_3
        targets["palm_4_joint"] = palm_4

        return targets

    def motor_dict_to_joint_dict(self, motor_dict: dict[str, float]) -> dict[str, float]:
        """Convert a dict of motor_name→value (physical) to joint targets."""
        action = np.zeros(len(self.motor_names))
        for i, name in enumerate(self.motor_names):
            val = motor_dict.get(name, 0.0)
            action[i] = self.normalize_motor(val, i)
        return self.map(action)

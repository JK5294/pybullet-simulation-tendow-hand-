"""Actuator dataclass: motor / spool / pulley abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Actuator:
    """Represents a motor / spool / pulley assembly.

    Responsible for converting motor angle to tendon displacement
    and motor torque limit to max tendon tension.

    Formulas:
        Δl = r_spool * α * gear_ratio
        T_max = τ_motor / r_spool
    """

    id: str
    motor_angle: float = 0.0  # rad
    motor_velocity: float = 0.0  # rad/s
    motor_torque_limit: float = 0.5  # Nm
    spool_radius: float = 0.008  # m
    gear_ratio: float = 1.0
    efficiency: float = 0.9

    # State
    commanded_angle: float = field(default=0.0, repr=False)

    def get_tendon_displacement(self, angle: float | None = None) -> float:
        """Convert motor angle (rad) to tendon displacement (m)."""
        a = angle if angle is not None else self.motor_angle
        return self.spool_radius * self.gear_ratio * a

    def get_max_tension(self) -> float:
        """Convert motor torque limit to max tendon tension (N)."""
        return self.motor_torque_limit / self.spool_radius

    def get_commanded_displacement(self) -> float:
        """Tendon displacement from the last commanded motor angle."""
        return self.get_tendon_displacement(self.commanded_angle)

    def apply_command(self, value: float, mode: str = "position") -> None:
        """Apply a motor command.

        Args:
            value: motor angle (rad) if mode="position",
                   motor velocity (rad/s) if mode="velocity".
            mode: "position" | "velocity" | "effort"
        """
        if mode == "position":
            self.commanded_angle = float(value)
        elif mode == "velocity":
            self.motor_velocity = float(value)
        # effort mode requires external torque computation

    def step(self, dt: float) -> None:
        """Update motor state for one timestep (simple integration)."""
        # For position control: move toward commanded angle
        error = self.commanded_angle - self.motor_angle
        self.motor_angle += error * min(1.0, dt * 100.0)

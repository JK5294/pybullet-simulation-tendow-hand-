"""Motor command abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class MotorCommand:
    """A single motor command."""

    actuator_id: str
    mode: Literal["position", "velocity", "effort"] = "position"
    value: float = 0.0


@dataclass
class MotorCommandSet:
    """A set of motor commands for multiple actuators."""

    commands: list[MotorCommand] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.commands is None:
            self.commands = []

    def to_dict(self) -> dict[str, float]:
        """Return {actuator_id: value} for position commands."""
        return {cmd.actuator_id: cmd.value for cmd in self.commands if cmd.mode == "position"}

    @classmethod
    def from_dict(cls, cmds: dict[str, float], mode: str = "position") -> "MotorCommandSet":
        return cls(commands=[MotorCommand(aid, mode, val) for aid, val in cmds.items()])

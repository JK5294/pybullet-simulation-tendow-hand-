"""Palm class: aggregates palm joints."""

from __future__ import annotations

from dataclasses import dataclass, field

from tendon_hand.core.joint import Joint


@dataclass
class Palm:
    """Palm section with its own joints (e.g. palm_1, palm_3, palm_4)."""

    id: str = "palm"
    joints: list[Joint] = field(default_factory=list)

    def get_joint(self, joint_id: str) -> Joint | None:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def get_joint_angles(self) -> dict[str, float]:
        return {j.id: j.angle for j in self.joints}

    def reset(self) -> None:
        for j in self.joints:
            j.angle = 0.0
            j.velocity = 0.0
            j.reset_torque()

    def step(self, dt: float) -> None:
        for j in self.joints:
            j.step(dt)

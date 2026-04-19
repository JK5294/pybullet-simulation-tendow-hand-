"""URDF loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class URDFLoader:
    """Loads a URDF file and provides joint name → index mapping."""

    urdf_path: str

    def __post_init__(self):
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

    def get_asset_dir(self) -> str:
        """Return the directory containing the URDF (for mesh resolution)."""
        return os.path.dirname(self.urdf_path)

    def load(self, physics_client_id: int = 0):
        """Load URDF into PyBullet and return body ID.

        This is a thin wrapper; PyBulletHandAdapter handles the actual loading.
        """
        import pybullet as p

        body_id = p.loadURDF(
            self.urdf_path,
            physicsClientId=physics_client_id,
        )
        return body_id

    @staticmethod
    def build_joint_map(body_id: int, physics_client_id: int = 0) -> dict[str, int]:
        """Build {joint_name: joint_index} from a loaded URDF body."""
        import pybullet as p

        joint_map: dict[str, int] = {}
        for j in range(p.getNumJoints(body_id, physicsClientId=physics_client_id)):
            info = p.getJointInfo(body_id, j, physicsClientId=physics_client_id)
            joint_map[info[1].decode("utf-8")] = j
        return joint_map

    @staticmethod
    def get_actuated_joints(body_id: int, physics_client_id: int = 0) -> list[tuple[str, int]]:
        """Return list of (joint_name, joint_index) for non-fixed joints."""
        import pybullet as p

        actuated = []
        for j in range(p.getNumJoints(body_id, physicsClientId=physics_client_id)):
            info = p.getJointInfo(body_id, j, physicsClientId=physics_client_id)
            joint_type = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, etc.
            if joint_type != p.JOINT_FIXED:
                actuated.append((info[1].decode("utf-8"), j))
        return actuated

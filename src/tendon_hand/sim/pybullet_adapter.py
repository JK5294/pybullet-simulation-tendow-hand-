"""PyBullet simulation adapter for the tendon-driven hand.

This adapter is intentionally thin:
- It loads the URDF and maintains joint mapping
- It writes joint targets from the core library into PyBullet
- It reads joint states and contacts from PyBullet
- It does NOT compute tendon physics (that's in core)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tendon_hand.core.hand import Hand
from tendon_hand.sim.urdf_loader import URDFLoader
from tendon_hand.sim.contact import ContactObserver
from tendon_hand.utils.asset_resolver import resolve_urdf_path


@dataclass
class PyBulletHandAdapter:
    """Adapter that connects a Hand model to PyBullet simulation.

    Usage:
        hand = Hand.from_config("hand.yaml")
        sim = PyBulletHandAdapter(hand, urdf_path="V3.urdf", gui=True)
        sim.reset()
        sim.hand_controller.close_pose()
        sim.apply_joint_targets()
        sim.step()
    """

    hand: Hand
    urdf_path: str
    gui: bool = False
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.22)
    base_orientation_euler: tuple[float, float, float] = (0.0, np.pi, 0.0)
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Internal state
    client_id: int = field(default=0, repr=False)
    hand_body_id: int | None = field(default=None, repr=False)
    joint_map: dict[str, int] = field(default_factory=dict, repr=False)
    contact_observer: ContactObserver | None = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.urdf_path = resolve_urdf_path(self.urdf_path)

    def connect(self) -> None:
        """Connect to PyBullet (GUI or DIRECT)."""
        import pybullet as p

        if self._connected:
            return

        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from PyBullet."""
        import pybullet as p

        if self._connected:
            p.disconnect(physicsClientId=self.client_id)
            self._connected = False
            self.hand_body_id = None
            self.joint_map = {}

    def reset(self) -> None:
        """Load URDF, reset joints, sync state."""
        import pybullet as p

        if not self._connected:
            self.connect()

        # Remove old body if exists
        if self.hand_body_id is not None:
            p.removeBody(self.hand_body_id, physicsClientId=self.client_id)

        # Load URDF
        self.hand_body_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=p.getQuaternionFromEuler(self.base_orientation_euler),
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        # Build joint map
        self.joint_map = URDFLoader.build_joint_map(
            self.hand_body_id, physics_client_id=self.client_id
        )

        # Reset all joints to zero
        for j in range(p.getNumJoints(self.hand_body_id, physicsClientId=self.client_id)):
            p.resetJointState(self.hand_body_id, j, 0.0, physicsClientId=self.client_id)

        # Reset hand model
        self.hand.reset()

        # Contact observer
        self.contact_observer = ContactObserver(
            hand_body_id=self.hand_body_id,
            physics_client_id=self.client_id,
        )

    def apply_joint_targets(
        self,
        targets: dict[str, float] | None = None,
        force: float = 0.3,
        position_gain: float = 0.1,
        velocity_gain: float = 0.1,
    ) -> None:
        """Write joint target positions from the hand model into PyBullet.

        If targets is None, uses hand_controller.get_joint_targets().
        """
        import pybullet as p

        if targets is None:
            return

        for jname, val in targets.items():
            if jname in self.joint_map:
                p.setJointMotorControl2(
                    self.hand_body_id,
                    self.joint_map[jname],
                    p.POSITION_CONTROL,
                    targetPosition=float(val),
                    force=force,
                    positionGain=position_gain,
                    velocityGain=velocity_gain,
                    physicsClientId=self.client_id,
                )

    def get_joint_states(self) -> dict[str, tuple[float, float]]:
        """Read current joint positions and velocities from PyBullet.

        Returns {joint_name: (position, velocity)}.
        """
        import pybullet as p

        states: dict[str, tuple[float, float]] = {}
        for jname, jidx in self.joint_map.items():
            state = p.getJointState(
                self.hand_body_id, jidx, physicsClientId=self.client_id
            )
            states[jname] = (float(state[0]), float(state[1]))
        return states

    def step(self, n: int = 1) -> None:
        """Run n physics simulation steps."""
        import pybullet as p

        for _ in range(n):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_contacts(self) -> list:
        """Get current contact points."""
        if self.contact_observer is not None:
            return self.contact_observer.get_contacts()
        return []

    def capture_camera(
        self,
        filename: str | None = None,
        yaw: float = 60.0,
        pitch: float = -25.0,
        distance: float = 0.30,
        target: tuple[float, float, float] = (0.0, 0.0, 0.12),
        width: int = 800,
        height: int = 600,
    ) -> np.ndarray | None:
        """Capture a camera image from PyBullet."""
        import pybullet as p
        from PIL import Image

        vm = p.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw, pitch, 0, 2,
            physicsClientId=self.client_id,
        )
        pm = p.computeProjectionMatrixFOV(
            60, width / height, 0.1, 100.0,
            physicsClientId=self.client_id,
        )
        (_, _, px, _, _) = p.getCameraImage(
            width, height,
            viewMatrix=vm, projectionMatrix=pm,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )
        rgb = np.array(px).reshape(height, width, 4)[:, :, :3]
        if filename is not None:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            Image.fromarray(rgb).save(filename)
        return rgb

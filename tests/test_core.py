"""Unit tests for core domain models."""

from __future__ import annotations

import numpy as np
import pytest

from tendon_hand.core.actuator import Actuator
from tendon_hand.core.joint import Joint
from tendon_hand.core.tendon import Tendon, RoutingElement, SimpleRoutingLossModel
from tendon_hand.core.finger import Finger
from tendon_hand.core.hand import Hand
from tendon_hand.core.models.compliance import PassiveStiffnessModel
from tendon_hand.core.models.tension import QuasiStaticTensionModel


class TestActuator:
    def test_displacement(self):
        a = Actuator(id="test", spool_radius=0.01)
        a.motor_angle = np.pi
        assert a.get_tendon_displacement() == pytest.approx(0.01 * np.pi)

    def test_max_tension(self):
        a = Actuator(id="test", motor_torque_limit=0.5, spool_radius=0.01)
        assert a.get_max_tension() == pytest.approx(50.0)

    def test_command(self):
        a = Actuator(id="test")
        a.apply_command(0.5)
        assert a.commanded_angle == pytest.approx(0.5)


class TestJoint:
    def test_clamp(self):
        j = Joint(id="j1", lower_limit=0.0, upper_limit=1.0)
        assert j.clamp_angle(1.5) == pytest.approx(1.0)
        assert j.clamp_angle(-0.5) == pytest.approx(0.0)

    def test_passive_torque(self):
        j = Joint(id="j1", passive_stiffness=1.0, passive_damping=0.1)
        j.angle = 0.5
        j.velocity = 1.0
        tau = j.compute_passive_torque()
        assert tau == pytest.approx(-0.5 - 0.1)

    def test_step(self):
        j = Joint(id="j1", lower_limit=-1.0, upper_limit=1.0)
        j.apply_torque(1.0)
        j.step(dt=0.01)
        assert j.angle > 0.0


class TestTendon:
    def test_length_from_joints(self):
        t = Tendon(
            id="t1", actuator_id="a1", rest_length=0.1,
            routed_joints=[
                RoutingElement(joint_id="j1", moment_arm=0.005, direction_sign=1),
            ],
        )
        length = t.compute_length_from_joints({"j1": 1.0})
        assert length == pytest.approx(0.1 + 0.005)

    def test_tension_clamped(self):
        t = Tendon(id="t1", actuator_id="a1", elasticity_k=100.0)
        T = t.compute_tension(commanded_length=0.12, actual_length=0.10, dt=0.01)
        assert T >= 0.0

    def test_joint_torques(self):
        t = Tendon(
            id="t1", actuator_id="a1", tension=10.0,
            routed_joints=[
                RoutingElement(joint_id="j1", moment_arm=0.005, direction_sign=1),
            ],
        )
        torques = t.get_joint_torques({"j1": 0.0})
        assert torques["j1"] == pytest.approx(0.05)


class TestFinger:
    def test_observation(self):
        f = Finger(id="index", joints=[
            Joint(id="j1"), Joint(id="j2"),
        ])
        obs = f.get_observation()
        assert "joint_angles" in obs
        assert len(obs["joint_angles"]) == 2


class TestHand:
    def test_reset(self):
        h = Hand()
        h.fingers.append(Finger(id="index", joints=[Joint(id="j1")]))
        h.actuators.append(Actuator(id="a1"))
        h.reset()
        assert h.time == 0.0

    def test_apply_motor_commands(self):
        h = Hand()
        h.actuators.append(Actuator(id="a1"))
        h.apply_motor_commands({"a1": 0.5})
        assert h.get_actuator("a1").commanded_angle == pytest.approx(0.5)


class TestComplianceModel:
    def test_stiffness(self):
        m = PassiveStiffnessModel(stiffness=2.0, damping=0.0)
        assert m.compute_torque(0.5, 0.0) == pytest.approx(-1.0)


class TestTensionModel:
    def test_quasi_static(self):
        m = QuasiStaticTensionModel(elasticity_k=100.0, damping_c=0.0)
        T = m.compute_tension(0.12, 0.10, 0.12, 0.01)
        assert T == pytest.approx(2.0)

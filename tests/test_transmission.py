"""Tests for CascadeTransmissionModel.

After URDF axis normalization (all fingers use consistent axis direction),
the transmission model no longer needs per-finger sign flips.
+ve motor1/motor2 → inward flexion (closing) for ALL fingers.
+ve motor3 → outward abduction for ALL knuckles.
"""

from __future__ import annotations

import numpy as np
import pytest

from tendon_hand.core.models.transmission import CascadeTransmissionModel


class TestTransmissionBehavior:
    """Verify CascadeTransmissionModel produces correct joint targets."""

    @pytest.fixture
    def model(self):
        return CascadeTransmissionModel()

    def test_open_pose(self, model):
        """All zeros → all joints at zero (neutral open hand)."""
        action = np.zeros(17)
        joints = model.map(action)

        # All finger joints at zero when motors are zero
        for prefix in ["finger_1", "finger_2", "finger_3", "finger_4", "finger_5"]:
            if prefix == "finger_1":
                assert joints[f"{prefix}_link2_joint"] == pytest.approx(0.0, abs=1e-6)
                assert joints[f"{prefix}_link1_joint"] == pytest.approx(0.0, abs=1e-6)
            else:
                assert joints[f"{prefix}_link3_joint"] == pytest.approx(0.0, abs=1e-6)
                assert joints[f"{prefix}_link2_joint"] == pytest.approx(0.0, abs=1e-6)
                assert joints[f"{prefix}_link1_joint"] == pytest.approx(0.0, abs=1e-6)

        # All knuckles at zero
        for i in range(1, 6):
            assert joints[f"knuckle_{i}_joint"] == pytest.approx(0.0, abs=1e-6)

        # Palm joints at zero
        assert joints["palm_1_joint"] == pytest.approx(0.0, abs=1e-6)
        assert joints["palm_3_joint"] == pytest.approx(0.0, abs=1e-6)
        assert joints["palm_4_joint"] == pytest.approx(0.0, abs=1e-6)

    def test_close_pose(self, model):
        """CLOSE pose: all fingers flex inward consistently."""
        close_motors = {
            "thumb_m1": 2.5, "thumb_m2": 1.0, "thumb_m3": -0.5,
            "index_m1": 5.8, "index_m2": 4.2, "index_m3": -0.4,
            "middle_m1": 5.8, "middle_m2": 4.2, "middle_m3": -0.3,
            "ring_m1": 5.8, "ring_m2": 4.2, "ring_m3": -0.4,
            "pinky_m1": 5.8, "pinky_m2": 4.2, "pinky_m3": -0.4,
            "wrist_m4": 0.0, "wrist_m5": 0.0,
        }
        joints = model.motor_dict_to_joint_dict(close_motors)

        # Thumb: m1=2.5 → link2 saturated, m2=1.0 → link1 gets remainder
        assert joints["finger_1_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_1_link1_joint"] == pytest.approx(1.0, abs=1e-5)
        assert joints["knuckle_1_joint"] == pytest.approx(-0.5, abs=1e-5)

        # Fingers 2-5: m1=5.8 → link3/link2 saturated, m2=4.2 → link1 gets remainder
        for prefix in ["finger_2", "finger_3", "finger_4", "finger_5"]:
            assert joints[f"{prefix}_link3_joint"] == pytest.approx(1.65, abs=1e-5)
            assert joints[f"{prefix}_link2_joint"] == pytest.approx(1.65, abs=1e-5)
            assert joints[f"{prefix}_link1_joint"] == pytest.approx(2.268, abs=1e-5)

        # Knuckles: negative m3 = inward (adduction)
        assert joints["knuckle_2_joint"] == pytest.approx(-0.4, abs=1e-5)
        assert joints["knuckle_3_joint"] == pytest.approx(-0.3, abs=1e-5)
        assert joints["knuckle_4_joint"] == pytest.approx(-0.4, abs=1e-5)
        assert joints["knuckle_5_joint"] == pytest.approx(-0.4, abs=1e-5)

    def test_all_fingers_same_sign_convention(self, model):
        """+ve m1/m2 should produce +ve joint angles for ALL fingers."""
        for finger_idx, m1_idx, m2_idx in [
            ("finger_2", 3, 4),   # index
            ("finger_3", 6, 7),   # middle
            ("finger_4", 9, 10),  # ring
            ("finger_5", 12, 13), # pinky
        ]:
            action = np.zeros(17)
            action[m1_idx] = 0.3  # denorm = 1.8
            action[m2_idx] = 0.6  # denorm = 3.0 (enough to overflow to link1)
            joints = model.map(action)

            link3 = joints[f"{finger_idx}_link3_joint"]
            link2 = joints[f"{finger_idx}_link2_joint"]
            link1 = joints[f"{finger_idx}_link1_joint"]

            assert link3 > 0, f"{finger_idx} link3 should be > 0 for +ve m1"
            assert link2 > 0, f"{finger_idx} link2 should be > 0 for +ve m1"
            assert link1 > 0, f"{finger_idx} link1 should be > 0 for +ve m2"

    def test_all_fingers_negative_sign_convention(self, model):
        """-ve m1/m2 should produce -ve joint angles for ALL fingers."""
        for finger_idx, m1_idx, m2_idx in [
            ("finger_2", 3, 4),
            ("finger_3", 6, 7),
            ("finger_4", 9, 10),
            ("finger_5", 12, 13),
        ]:
            action = np.zeros(17)
            action[m1_idx] = -0.3
            action[m2_idx] = -0.2
            joints = model.map(action)

            link3 = joints[f"{finger_idx}_link3_joint"]
            link2 = joints[f"{finger_idx}_link2_joint"]
            link1 = joints[f"{finger_idx}_link1_joint"]

            assert link3 < 0, f"{finger_idx} link3 should be < 0 for -ve m1"
            assert link2 < 0, f"{finger_idx} link2 should be < 0 for -ve m1"
            assert link1 < 0, f"{finger_idx} link1 should be < 0 for -ve m2"

    def test_knuckle_same_sign_convention(self, model):
        """+ve m3 = outward (abduction) for ALL knuckles."""
        for m3_idx, knuckle in [
            (2, "knuckle_1_joint"),
            (5, "knuckle_2_joint"),
            (8, "knuckle_3_joint"),
            (11, "knuckle_4_joint"),
            (14, "knuckle_5_joint"),
        ]:
            action = np.zeros(17)
            action[m3_idx] = 0.3
            joints = model.map(action)
            assert joints[knuckle] > 0, f"{knuckle} should be > 0 for +ve m3"

            action[m3_idx] = -0.3
            joints = model.map(action)
            assert joints[knuckle] < 0, f"{knuckle} should be < 0 for -ve m3"

    def test_index_finger_cascade(self, model):
        """Index finger cascade: action values denormalize then cascade."""
        # action[3]=1.0 → motor=6.0 (saturates link3/link2 at 1.65)
        # action[4]=0.5 → motor=2.5 (fills link1 at 2.268 limit)
        # action[5]=0.3 → motor=0.15 (knuckle)
        action = np.zeros(17)
        action[3] = 1.0  # index_m1 → denorm = 6.0
        action[4] = 0.5  # index_m2 → denorm = 2.5
        action[5] = 0.3  # index_m3 → denorm = 0.15
        joints = model.map(action)

        assert joints["finger_2_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link1_joint"] == pytest.approx(2.268, abs=1e-5)
        assert joints["knuckle_2_joint"] == pytest.approx(0.15, abs=1e-5)

    def test_middle_finger_cascade(self, model):
        """Middle finger cascade with +ve motor values."""
        action = np.zeros(17)
        action[6] = 1.0   # middle_m1 → denorm = 6.0
        action[7] = 0.5   # middle_m2 → denorm = 2.5
        action[8] = 0.3   # middle_m3 → denorm = 0.15
        joints = model.map(action)

        assert joints["finger_3_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_3_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_3_link1_joint"] == pytest.approx(2.268, abs=1e-5)
        assert joints["knuckle_3_joint"] == pytest.approx(0.15, abs=1e-5)

    def test_ring_finger_cascade(self, model):
        """Ring finger cascade with +ve motor values."""
        action = np.zeros(17)
        action[9] = 1.0   # ring_m1 → denorm = 6.0
        action[10] = 0.5  # ring_m2 → denorm = 2.5
        action[11] = 0.3  # ring_m3 → denorm = 0.15
        joints = model.map(action)

        assert joints["finger_4_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_4_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_4_link1_joint"] == pytest.approx(2.268, abs=1e-5)
        assert joints["knuckle_4_joint"] == pytest.approx(0.15, abs=1e-5)

    def test_pinky_finger_cascade(self, model):
        """Pinky finger cascade with +ve motor values."""
        action = np.zeros(17)
        action[12] = 1.0  # pinky_m1 → denorm = 6.0
        action[13] = 0.5  # pinky_m2 → denorm = 2.5
        action[14] = 0.3  # pinky_m3 → denorm = 0.15
        joints = model.map(action)

        assert joints["finger_5_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_5_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_5_link1_joint"] == pytest.approx(2.268, abs=1e-5)
        assert joints["knuckle_5_joint"] == pytest.approx(0.15, abs=1e-5)

    def test_finger_cascade_saturation(self, model):
        """Large motor values saturate at joint limits."""
        action = np.zeros(17)
        action[3] = 1.0  # index_m1 → denorm = 6.0 (way beyond limit)
        action[4] = 1.0  # index_m2 → denorm = 5.0 (way beyond limit)
        joints = model.map(action)

        # Should saturate at upper limits
        assert joints["finger_2_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link1_joint"] == pytest.approx(2.268, abs=1e-5)

    def test_max_combine_logic(self, model):
        """motor1 and motor2 contributions use max(), not sum."""
        # m1 fills link3/link2, m2 affects link1
        action = np.zeros(17)
        action[3] = 0.5   # index_m1 → denorm = 3.0
        action[4] = 0.3   # index_m2 → denorm = 1.5
        joints = model.map(action)

        # m1=3.0: link3=1.65(sat), link2=1.35, link1=0
        # m2=1.5: link2=1.5, link3=1.5, link1=0
        # max: link3=1.65, link2=1.5, link1=0
        assert joints["finger_2_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link2_joint"] == pytest.approx(1.5, abs=1e-5)
        assert joints["finger_2_link1_joint"] == pytest.approx(0.0, abs=1e-5)

        # If m2 overflow > m1 overflow, link1 gets the difference
        action = np.zeros(17)
        action[3] = 0.2   # index_m1 → denorm = 1.2
        action[4] = 0.5   # index_m2 → denorm = 2.5
        joints = model.map(action)

        # m1=1.2: link3=1.2, link2=0, link1=0
        # m2=2.5: link2=1.65(sat), link3=1.65, link1=0.85
        # max: link3=1.65, link2=1.65, link1=0.85
        assert joints["finger_2_link3_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link2_joint"] == pytest.approx(1.65, abs=1e-5)
        assert joints["finger_2_link1_joint"] == pytest.approx(0.85, abs=1e-5)


class TestFingerTransmissionInternals:
    def test_cascade_motor1(self):
        from tendon_hand.core.models.transmission import _cascade_motor1
        j3, j2, j1, p = _cascade_motor1(
            5.0, (-0.087, 1.65), (-0.087, 1.65), (-0.087, 2.268), (0.0, 0.087)
        )
        assert j3 == pytest.approx(1.65)
        assert j2 == pytest.approx(1.65)
        assert j1 == pytest.approx(1.7)
        assert p == pytest.approx(0.0)

    def test_cascade_motor2(self):
        from tendon_hand.core.models.transmission import _cascade_motor2
        j3, j2, j1, p = _cascade_motor2(
            4.0, (-0.087, 1.65), (-0.087, 1.65), (-0.087, 2.268), (0.0, 0.087)
        )
        assert j2 == pytest.approx(1.65)
        assert j3 == pytest.approx(1.65)  # distal follows middle
        remaining = 4.0 - 1.65
        assert j1 == pytest.approx(min(remaining, 2.268))

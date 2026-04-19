"""Finger-by-finger tendon transmission test — no URDF required.

Tests each finger individually to verify:
  1. Motor sign correctness (+motor = flexion/inward for all fingers)
  2. Cascade behavior (motor1 distal→middle→proximal→palm)
  3. Motor2 behavior (middle active, distal follows)
  4. Knuckle direction (+m3 = outward/abduction)

A "block" ASCII visualization makes the under-actuated cascade intuitive.

Usage:
    python examples/demo_finger_test.py [finger_name]

finger_name: thumb | index | middle | ring | pinky | all
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tendon_hand.control.hand_controller import HandController
from tendon_hand.core.models.transmission import (
    _cascade_motor1,
    _cascade_motor2,
    _cascade_thumb_motor1,
    _cascade_thumb_motor2,
    JointLimits,
    ThumbJointLimits,
)


# ---------------------------------------------------------------------------
# Block visualization (same style as demo_hand.py)
# ---------------------------------------------------------------------------

def _bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    span = hi - lo
    if span <= 0:
        return "█" * width
    ratio = float(np.clip((value - lo) / span, 0.0, 1.0))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _motor_block(value: float, max_val: float, width: int = 24) -> str:
    ratio = float(np.clip(abs(value) / max(max_val, 1e-6), 0.0, 1.0))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    block = "█" * filled + "░" * (width - filled)
    sign = "+" if value >= 0 else "-"
    return f"[{sign}{block}]"


def draw_finger_tanks(
    m1: float,
    m2: float,
    m3: float,
    finger_name: str,
    limits: JointLimits | ThumbJointLimits,
    is_thumb: bool = False,
) -> str:
    """Return ASCII tank visualization for a finger."""
    if is_thumb:
        j2_1, j1_1, p1 = _cascade_thumb_motor1(
            m1, limits.link2, limits.link1, limits.palm
        )
        j2_2, j1_2, p2 = _cascade_thumb_motor2(
            m2, limits.link2, limits.link1, limits.palm
        )
        j2 = max(j2_1, j2_2)
        j1 = max(j1_1, j1_2)
        palm = max(p1, p2)
        knuckle = float(np.clip(m3, *limits.knuckle))
        lines = [
            f"  Motor1 block {_motor_block(m1, 4.0)}",
            f"  Motor2 block {_motor_block(m2, 3.0)}",
            f"",
            f"  Distal  (link2) {_bar(j2, *limits.link2)} {np.degrees(j2):6.1f}°",
            f"  Proximal(link1) {_bar(j1, *limits.link1)} {np.degrees(j1):6.1f}°",
            f"  Palm            {_bar(palm, *limits.palm)} {np.degrees(palm):6.1f}°",
            f"  Knuckle         {_bar(knuckle, *limits.knuckle)} {np.degrees(knuckle):6.1f}°",
        ]
    else:
        j3_1, j2_1, j1_1, p1 = _cascade_motor1(
            m1, limits.link3, limits.link2, limits.link1, limits.palm
        )
        j3_2, j2_2, j1_2, p2 = _cascade_motor2(
            m2, limits.link3, limits.link2, limits.link1, limits.palm
        )
        j3 = max(j3_1, j3_2)
        j2 = max(j2_1, j2_2)
        j1 = max(j1_1, j1_2)
        palm = max(p1, p2)
        knuckle = float(np.clip(m3, *limits.knuckle))
        lines = [
            f"  Motor1 block {_motor_block(m1, 6.0)}",
            f"  Motor2 block {_motor_block(m2, 5.0)}",
            f"",
            f"  Distal  (link3) {_bar(j3, *limits.link3)} {np.degrees(j3):6.1f}°",
            f"  Middle  (link2) {_bar(j2, *limits.link2)} {np.degrees(j2):6.1f}°",
            f"  Proximal(link1) {_bar(j1, *limits.link1)} {np.degrees(j1):6.1f}°",
            f"  Palm            {_bar(palm, *limits.palm)} {np.degrees(palm):6.1f}°",
            f"  Knuckle         {_bar(knuckle, *limits.knuckle)} {np.degrees(knuckle):6.1f}°",
        ]
    return "\n".join(lines)


def _get_limits(finger_name: str) -> tuple[JointLimits | ThumbJointLimits, bool]:
    """Return (limits, is_thumb) for a finger."""
    if finger_name == "thumb":
        return ThumbJointLimits(), True
    return JointLimits(), False


def _motor_values(finger_name: str, m1: float, m2: float, m3: float) -> tuple[float, float, float]:
    """Return signed motor values for a finger (sign convention handled by controller)."""
    # The controller handles sign internally; we just pass raw values here.
    return m1, m2, m3


def test_finger(finger_name: str) -> None:
    """Test a single finger with multiple motor configurations."""
    print("\n" + "=" * 60)
    print(f"TESTING: {finger_name.upper()} FINGER")
    print("=" * 60)

    ctrl = HandController()
    limits, is_thumb = _get_limits(finger_name)
    set_fn = {
        "thumb": ctrl.set_thumb, "index": ctrl.set_index,
        "middle": ctrl.set_middle, "ring": ctrl.set_ring, "pinky": ctrl.set_pinky,
    }[finger_name]
    prefix = {
        "thumb": "finger_1", "index": "finger_2", "middle": "finger_3",
        "ring": "finger_4", "pinky": "finger_5",
    }[finger_name]
    knuckle_num = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}[finger_name]

    # Helper to grab expected joint names
    if is_thumb:
        joint_names = [f"{prefix}_link2_joint", f"{prefix}_link1_joint", f"knuckle_{knuckle_num}_joint"]
    else:
        joint_names = [f"{prefix}_link3_joint", f"{prefix}_link2_joint", f"{prefix}_link1_joint", f"knuckle_{knuckle_num}_joint"]

    def _check(label: str, m1: float, m2: float, m3: float) -> None:
        ctrl.open_pose()
        set_fn(m1=m1, m2=m2, m3=m3)
        targets = ctrl.get_joint_targets()

        # Compute expected from raw cascade
        if is_thumb:
            j2_1, j1_1, p1 = _cascade_thumb_motor1(m1, limits.link2, limits.link1, limits.palm)
            j2_2, j1_2, p2 = _cascade_thumb_motor2(m2, limits.link2, limits.link1, limits.palm)
            expected = {
                joint_names[0]: max(j2_1, j2_2),
                joint_names[1]: max(j1_1, j1_2),
                joint_names[2]: float(np.clip(m3, *limits.knuckle)),
            }
        else:
            j3_1, j2_1, j1_1, p1 = _cascade_motor1(m1, limits.link3, limits.link2, limits.link1, limits.palm)
            j3_2, j2_2, j1_2, p2 = _cascade_motor2(m2, limits.link3, limits.link2, limits.link1, limits.palm)
            expected = {
                joint_names[0]: max(j3_1, j3_2),
                joint_names[1]: max(j2_1, j2_2),
                joint_names[2]: max(j1_1, j1_2),
                joint_names[3]: float(np.clip(m3, *limits.knuckle)),
            }

        print(f"\n👉 {label}")
        print(draw_finger_tanks(m1, m2, m3, finger_name, limits, is_thumb))

        print("   Target vs Expected:")
        max_diff = 0.0
        for jname in joint_names:
            t = targets.get(jname, 0.0)
            e = expected.get(jname, 0.0)
            diff = abs(t - e)
            max_diff = max(max_diff, diff)
            marker = "✅" if diff < 1e-4 else "❌"
            short = jname.split("_")[2] if "link" in jname else "knuckle"
            print(f"   {marker} {short:8s} target={t:7.3f}  expected={e:7.3f}  diff={diff:.4f}")

        if max_diff > 1e-3:
            print(f"   ⚠️  Max error {max_diff:.4f} rad — check sign convention!")

    # ============================================================
    # Test A: OPEN pose baseline
    # ============================================================
    print(f"\n👉 [A] OPEN pose (baseline)")
    ctrl.open_pose()
    targets = ctrl.get_joint_targets()
    print("   All joints at zero:")
    for jname in joint_names:
        t = targets.get(jname, 0.0)
        marker = "✅" if abs(t) < 1e-6 else "❌"
        short = jname.split("_")[2] if "link" in jname else "knuckle"
        print(f"   {marker} {short:8s} = {t:7.3f} rad")

    # ============================================================
    # Test B: Motor1 only (distal active → cascade down)
    # ============================================================
    if finger_name == "thumb":
        _check("[B] Motor1 only — large flexion", m1=2.5, m2=0.0, m3=0.0)
    else:
        _check("[B] Motor1 only — large flexion", m1=5.0, m2=0.0, m3=0.0)

    # ============================================================
    # Test C: Motor2 only (middle active, distal follows)
    # ============================================================
    if finger_name == "thumb":
        _check("[C] Motor2 only — middle/proximal flexion", m1=0.0, m2=2.0, m3=0.0)
    else:
        _check("[C] Motor2 only — middle active, distal follows", m1=0.0, m2=4.0, m3=0.0)

    # ============================================================
    # Test D: Both motors combined (max per joint)
    # ============================================================
    if finger_name == "thumb":
        _check("[D] Both motors combined", m1=2.5, m2=2.0, m3=0.0)
    else:
        _check("[D] Both motors combined", m1=5.5, m2=4.5, m3=0.0)

    # ============================================================
    # Test E: Knuckle outward (+m3)
    # ============================================================
    if finger_name == "thumb":
        _check("[E] Knuckle outward (+m3)", m1=0.0, m2=0.0, m3=1.0)
    else:
        _check("[E] Knuckle outward (+m3)", m1=0.0, m2=0.0, m3=0.4)

    # ============================================================
    # Test F: Knuckle inward (-m3)
    # ============================================================
    if finger_name == "thumb":
        _check("[F] Knuckle inward (-m3)", m1=0.0, m2=0.0, m3=-1.0)
    else:
        _check("[F] Knuckle inward (-m3)", m1=0.0, m2=0.0, m3=-0.4)

    # ============================================================
    # Test G: Full curl + knuckle (max everything)
    # ============================================================
    if finger_name == "thumb":
        _check("[G] Full curl + knuckle", m1=2.5, m2=2.0, m3=-0.5)
    else:
        _check("[G] Full curl + knuckle", m1=5.8, m2=4.2, m3=-0.3)

    print(f"\n✅ {finger_name.upper()} test complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finger-by-finger tendon test (no URDF needed)")
    parser.add_argument(
        "finger",
        nargs="?",
        default="all",
        choices=["thumb", "index", "middle", "ring", "pinky", "all"],
        help="Which finger to test",
    )
    args = parser.parse_args()

    fingers = ["thumb", "index", "middle", "ring", "pinky"] if args.finger == "all" else [args.finger]

    for finger in fingers:
        test_finger(finger)

    print("\n" + "=" * 60)
    print("ALL FINGER TESTS COMPLETE")
    print("=" * 60)
    print("""
  Tests performed for each finger:
    [A] Open pose     — baseline (all zeros)
    [B] Motor1 only   — distal-first cascade
    [C] Motor2 only   — middle active, distal follows
    [D] Both motors   — max() combination
    [E] Knuckle +m3   — outward abduction
    [F] Knuckle -m3   — inward adduction
    [G] Full curl     — max flexion + knuckle
""")


if __name__ == "__main__":
    main()

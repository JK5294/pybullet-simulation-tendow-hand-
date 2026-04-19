"""Demo: cable-driven cascade — motor → tendon → joint angle.  No URDF needed.

This demo demonstrates how a REAL tendon-driven hand works using only
console output and "block" ASCII visualization:

  Motor1 pulls tendon-1 → distal (link3/fingertip) bends FIRST.
                        → When distal hits limit, overflow → middle (link2).
                        → When middle hits limit, overflow → proximal (link1).

  Motor2 pulls tendon-2 → middle (link2) bends FIRST.
                        → Distal (link3) follows middle (mechanically coupled).
                        → When middle hits limit, overflow → proximal (link1).

  Both tendons act on the same joints → final angle = max(motor1_effect, motor2_effect).

This is NOT independent joint control. You CANNOT set link3=15° + link2=15° + link1=15°.
The cascade physically enforces the order: distal → middle → proximal.

Usage:
    python examples/demo_precise_angle.py [finger]

finger: thumb | index | middle | ring | pinky | all (default: index)
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
# Block visualization
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


def draw_step(
    step_num: int,
    label: str,
    m1: float,
    m2: float,
    m3: float,
    limits: JointLimits | ThumbJointLimits,
    is_thumb: bool = False,
) -> str:
    """Return a richly-annotated ASCII block diagram for one cascade step."""
    lines: list[str] = []
    lines.append(f"\n{'─' * 60}")
    lines.append(f"STEP {step_num}: {label}")
    lines.append(f"{'─' * 60}")

    if is_thumb:
        j2_1, j1_1, p1 = _cascade_thumb_motor1(m1, limits.link2, limits.link1, limits.palm)
        j2_2, j1_2, p2 = _cascade_thumb_motor2(m2, limits.link2, limits.link1, limits.palm)
        j2 = max(j2_1, j2_2)
        j1 = max(j1_1, j1_2)
        palm = max(p1, p2)
        knuckle = float(np.clip(m3, *limits.knuckle))

        lines.append(f"  Motor1 = {m1:.3f} rad  {_motor_block(m1, 4.0)}")
        lines.append(f"  Motor2 = {m2:.3f} rad  {_motor_block(m2, 3.0)}")
        lines.append(f"")
        lines.append(f"  Distal  (link2)  {_bar(j2, *limits.link2)}  {np.degrees(j2):6.1f}°")
        lines.append(f"  Proximal(link1)  {_bar(j1, *limits.link1)}  {np.degrees(j1):6.1f}°")
        lines.append(f"  Palm             {_bar(palm, *limits.palm)}  {np.degrees(palm):6.1f}°")
        lines.append(f"  Knuckle          {_bar(knuckle, *limits.knuckle)}  {np.degrees(knuckle):6.1f}°")
    else:
        j3_1, j2_1, j1_1, p1 = _cascade_motor1(m1, limits.link3, limits.link2, limits.link1, limits.palm)
        j3_2, j2_2, j1_2, p2 = _cascade_motor2(m2, limits.link3, limits.link2, limits.link1, limits.palm)
        j3 = max(j3_1, j3_2)
        j2 = max(j2_1, j2_2)
        j1 = max(j1_1, j1_2)
        palm = max(p1, p2)
        knuckle = float(np.clip(m3, *limits.knuckle))

        # Annotate which motor drives what
        m1_drives = []
        if j3_1 > 0:
            m1_drives.append("distal")
        if j2_1 > 0:
            m1_drives.append("middle")
        if j1_1 > 0:
            m1_drives.append("proximal")
        if p1 > 0:
            m1_drives.append("palm")

        m2_drives = []
        if j2_2 > 0:
            m2_drives.append("middle")
        if j3_2 > 0:
            m2_drives.append("distal(follows)")
        if j1_2 > 0:
            m2_drives.append("proximal")
        if p2 > 0:
            m2_drives.append("palm")

        lines.append(f"  Motor1 = {m1:.3f} rad  {_motor_block(m1, 6.0)}")
        if m1_drives:
            lines.append(f"         └── drives: {', '.join(m1_drives)}")
        lines.append(f"  Motor2 = {m2:.3f} rad  {_motor_block(m2, 5.0)}")
        if m2_drives:
            lines.append(f"         └── drives: {', '.join(m2_drives)}")
        lines.append(f"")
        lines.append(f"  Distal  (link3)  {_bar(j3, *limits.link3)}  {np.degrees(j3):6.1f}°  ← max(m1_distal={np.degrees(j3_1):.1f}°, m2_distal={np.degrees(j3_2):.1f}°)")
        lines.append(f"  Middle  (link2)  {_bar(j2, *limits.link2)}  {np.degrees(j2):6.1f}°  ← max(m1_middle={np.degrees(j2_1):.1f}°, m2_middle={np.degrees(j2_2):.1f}°)")
        lines.append(f"  Proximal(link1)  {_bar(j1, *limits.link1)}  {np.degrees(j1):6.1f}°  ← max(m1_prox ={np.degrees(j1_1):.1f}°, m2_prox ={np.degrees(j1_2):.1f}°)")
        lines.append(f"  Palm             {_bar(palm, *limits.palm)}  {np.degrees(palm):6.1f}°")
        lines.append(f"  Knuckle          {_bar(knuckle, *limits.knuckle)}  {np.degrees(knuckle):6.1f}°")

    return "\n".join(lines)


def run_demo(finger_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"  CABLE-DRIVEN CASCADE DEMO: {finger_name.upper()} FINGER")
    print("=" * 70)
    print("""
  Real tendon-driven hand mechanics:
    Tendon-1 (Motor1) :  distal → middle → proximal  (distal pulls first)
    Tendon-2 (Motor2) :  middle → proximal            (distal follows middle)
    Final angle       :  max(tendon1_effect, tendon2_effect) per joint
""")

    ctrl = HandController()
    limits: JointLimits | ThumbJointLimits
    is_thumb = finger_name == "thumb"
    if is_thumb:
        limits = ThumbJointLimits()
    else:
        limits = JointLimits()

    set_fn = {
        "thumb": ctrl.set_thumb, "index": ctrl.set_index,
        "middle": ctrl.set_middle, "ring": ctrl.set_ring, "pinky": ctrl.set_pinky,
    }[finger_name]

    # ================================================================
    # STEP 1: Motor1 only → distal (fingertip) bends first
    # ================================================================
    m1, m2 = np.radians(15), 0.0
    print(draw_step(1, "Motor1 = 15°, Motor2 = 0  →  ONLY fingertip bends!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_1 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 2: Add Motor2 → middle (2nd phalanx) bends, distal follows
    # ================================================================
    m1, m2 = np.radians(15), np.radians(15)
    print(draw_step(2, "Motor1 = 15°, Motor2 = 15°  →  middle bends, fingertip follows!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_2 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 3: Increase Motor1 → distal fills up, overflow to middle
    # ================================================================
    m1, m2 = np.radians(45), 0.0
    print(draw_step(3, "Motor1 = 45°, Motor2 = 0  →  distal fills, then middle!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_3 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 4: Increase Motor2 → middle fills, overflow to proximal
    # ================================================================
    m1, m2 = np.radians(45), np.radians(45)
    print(draw_step(4, "Motor1 = 45°, Motor2 = 45°  →  middle fills, then proximal!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_4 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 5: Motor1 saturates distal limit (94.5°), overflow cascades
    # ================================================================
    m1, m2 = np.radians(105), 0.0
    print(draw_step(5, "Motor1 = 105°, Motor2 = 0  →  distal SATURATES, overflow drives middle!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_5 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 6: Full cascade — both motors max out → all joints bend
    # ================================================================
    m1, m2 = 3.50, 3.00
    print(draw_step(6, "Motor1 = 3.50 rad, Motor2 = 3.00 rad  →  FULL flexion cascade!", m1, m2, 0.0, limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=m1, m2=m2, m3=0.0)
    targets_6 = ctrl.get_joint_targets()

    # ================================================================
    # STEP 7: Knuckle only (abduction / adduction)
    # ================================================================
    print(draw_step(7, "Knuckle = +10° outward (abduction), motors = 0", 0.0, 0.0, np.radians(10), limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=0.0, m2=0.0, m3=np.radians(10))
    targets_7 = ctrl.get_joint_targets()

    print(draw_step(8, "Knuckle = -10° inward (adduction), motors = 0", 0.0, 0.0, -np.radians(10), limits, is_thumb))
    ctrl.open_pose()
    set_fn(m1=0.0, m2=0.0, m3=-np.radians(10))
    targets_8 = ctrl.get_joint_targets()

    # ================================================================
    # Controller verification
    # ================================================================
    print("\n" + "=" * 70)
    print("  CONTROLLER OUTPUT VERIFICATION")
    print("=" * 70)

    poses = [
        ("01_open",       0.0,    0.0,    0.0),
        ("02_m1_15deg",   np.radians(15),  0.0,    0.0),
        ("03_m1m2_15deg", np.radians(15),  np.radians(15),  0.0),
        ("04_m1_45deg",   np.radians(45),  0.0,    0.0),
        ("05_m1m2_45deg", np.radians(45),  np.radians(45),  0.0),
        ("06_m1_105deg",  np.radians(105), 0.0,    0.0),
        ("07_full_cascade", 3.50, 3.00, 0.0),
        ("08_knuckle_out",  0.0,  0.0,  np.radians(10)),
        ("09_knuckle_in",   0.0,  0.0, -np.radians(10)),
    ]

    prefix = {
        "thumb": "finger_1", "index": "finger_2", "middle": "finger_3",
        "ring": "finger_4", "pinky": "finger_5",
    }[finger_name]
    knuckle_num = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}[finger_name]

    if is_thumb:
        jnames = [f"{prefix}_link2_joint", f"{prefix}_link1_joint", f"knuckle_{knuckle_num}_joint"]
    else:
        jnames = [f"{prefix}_link3_joint", f"{prefix}_link2_joint", f"{prefix}_link1_joint", f"knuckle_{knuckle_num}_joint"]

    for pose_name, m1_val, m2_val, m3_val in poses:
        ctrl.open_pose()
        set_fn(m1=m1_val, m2=m2_val, m3=m3_val)
        targets = ctrl.get_joint_targets()
        print(f"\n   📦 {pose_name}")
        for jname in jnames:
            t = targets.get(jname, 0.0)
            short = jname.split("_")[2] if "link" in jname else "knuckle"
            print(f"      {short:8s} = {np.degrees(t):7.2f}°")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: What you CAN and CANNOT control")
    print("=" * 70)
    print("""
  ✅ ACHIEVABLE poses (respect cascade order):
     • distal-only bend                (motor1 small)
     • distal+middle bend              (motor1+2 small)
     • distal saturated + middle bend  (motor1 large + motor2 small)
     • full cascade (all joints)       (motor1+2 large)
     • knuckle only (abduction/adduction)  (motor3 only)

  ❌ IMPOSSIBLE poses (violate underactuation):
     • link3=15° + link2=0° + link1=15°  (proximal cannot move before middle)
     • Independent per-joint control     (cascade couples them)

  🔑 KEY INSIGHT:
     Motor1 and Motor2 are NOT independent controllers for each joint.
     They pull TENDONS that cascade force through the finger structure.
     The joint angles emerge from the PHYSICS of this cascade,
     not from direct joint-angle commands.
""")
    print(f"✅ {finger_name.upper()} cascade demo complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cable-driven tendon cascade demo (no URDF needed)"
    )
    parser.add_argument(
        "finger", nargs="?", default="index",
        choices=["thumb", "index", "middle", "ring", "pinky", "all"],
    )
    args = parser.parse_args()

    fingers = (
        ["thumb", "index", "middle", "ring", "pinky"]
        if args.finger == "all" else [args.finger]
    )

    for finger in fingers:
        run_demo(finger)


if __name__ == "__main__":
    main()

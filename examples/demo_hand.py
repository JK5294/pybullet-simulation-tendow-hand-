"""Comprehensive demo for the tendon_hand library — no URDF required.

This script demonstrates every major function in the library using only
pure-Python / console output.  A "block" (ASCII tank) visualization makes
the under-actuated cascade直观.

Usage:
    python examples/demo_hand.py [mode]

Modes:
    control     — Pure control layer demo (HandController API)
    block       — Block / tank visualization of the cascade
    transmission— CascadeTransmissionModel demo
    inverse     — Inverse tendon mapping demo
    all         — Run everything (default)
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tendon_hand.control.hand_controller import HandController
from tendon_hand.core.models.transmission import (
    CascadeTransmissionModel,
    _cascade_motor1,
    _cascade_motor2,
    _cascade_thumb_motor1,
    _cascade_thumb_motor2,
)
from tendon_hand.core.models.inverse_transmission import (
    FingerInverseTransmission,
    ThumbInverseTransmission,
)


# ---------------------------------------------------------------------------
# Block (tank) visualization helpers
# ---------------------------------------------------------------------------

def _bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    """Return an ASCII bar showing where *value* sits in [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return "█" * width
    ratio = float(np.clip((value - lo) / span, 0.0, 1.0))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _motor_block(value: float, max_val: float, width: int = 24) -> str:
    """Motor input block (bidirectional, centred at 0)."""
    ratio = float(np.clip(abs(value) / max(max_val, 1e-6), 0.0, 1.0))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    block = "█" * filled + "░" * (width - filled)
    sign = "+" if value >= 0 else "-"
    return f"[{sign}{block}]"


def draw_finger_cascade(
    m1: float,
    m2: float,
    m3: float = 0.0,
    finger_name: str = "index",
    link3_lim: tuple[float, float] = (-0.087, 1.650),
    link2_lim: tuple[float, float] = (-0.087, 1.650),
    link1_lim: tuple[float, float] = (-0.087, 2.268),
    knuckle_lim: tuple[float, float] = (-0.436, 0.436),
    palm_lim: tuple[float, float] = (0.0, 0.087),
) -> str:
    """Return a multi-line ASCII "tank" drawing for one finger."""
    j3_1, j2_1, j1_1, p1 = _cascade_motor1(m1, link3_lim, link2_lim, link1_lim, palm_lim)
    j3_2, j2_2, j1_2, p2 = _cascade_motor2(m2, link3_lim, link2_lim, link1_lim, palm_lim)
    j3 = max(j3_1, j3_2)
    j2 = max(j2_1, j2_2)
    j1 = max(j1_1, j1_2)
    palm = max(p1, p2)
    knuckle = float(np.clip(m3, *knuckle_lim))

    lines = [
        f"  ┌─ {finger_name.upper()}  m1={m1:6.2f}  m2={m2:6.2f}  m3={m3:6.2f} ─────────────┐",
        f"  │  Motor1 block {_motor_block(m1, 6.0)}                    │",
        f"  │  Motor2 block {_motor_block(m2, 5.0)}                    │",
        f"  │                                                              │",
        f"  │  Distal  (link3) {_bar(j3, *link3_lim)} {np.degrees(j3):6.1f}° │",
        f"  │  Middle  (link2) {_bar(j2, *link2_lim)} {np.degrees(j2):6.1f}° │",
        f"  │  Proximal(link1) {_bar(j1, *link1_lim)} {np.degrees(j1):6.1f}° │",
        f"  │  Palm            {_bar(palm, *palm_lim)} {np.degrees(palm):6.1f}° │",
        f"  │  Knuckle         {_bar(knuckle, *knuckle_lim)} {np.degrees(knuckle):6.1f}° │",
        f"  └──────────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


def draw_thumb_cascade(
    m1: float, m2: float, m3: float = 0.0
) -> str:
    """Return a multi-line ASCII "tank" drawing for the thumb."""
    limits = type("L", (), {
        "link2": (-0.217, 1.650),
        "link1": (-0.087, 2.268),
        "knuckle": (-2.260, 1.750),
        "palm": (0.0, 0.087),
    })()
    j2_1, j1_1, p1 = _cascade_thumb_motor1(m1, limits.link2, limits.link1, limits.palm)
    j2_2, j1_2, p2 = _cascade_thumb_motor2(m2, limits.link2, limits.link1, limits.palm)
    j2 = max(j2_1, j2_2)
    j1 = max(j1_1, j1_2)
    palm = max(p1, p2)
    knuckle = float(np.clip(m3, *limits.knuckle))

    lines = [
        f"  ┌─ THUMB  m1={m1:6.2f}  m2={m2:6.2f}  m3={m3:6.2f} ───────────────────┐",
        f"  │  Motor1 block {_motor_block(m1, 4.0)}                         │",
        f"  │  Motor2 block {_motor_block(m2, 3.0)}                         │",
        f"  │                                                              │",
        f"  │  Distal  (link2) {_bar(j2, limits.link2[0], limits.link2[1])} {np.degrees(j2):6.1f}° │",
        f"  │  Proximal(link1) {_bar(j1, limits.link1[0], limits.link1[1])} {np.degrees(j1):6.1f}° │",
        f"  │  Palm            {_bar(palm, limits.palm[0], limits.palm[1])} {np.degrees(palm):6.1f}° │",
        f"  │  Knuckle         {_bar(knuckle, limits.knuckle[0], limits.knuckle[1])} {np.degrees(knuckle):6.1f}° │",
        f"  └──────────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo routines
# ---------------------------------------------------------------------------

def demo_pure_control() -> None:
    """Demo 1: Pure control layer — no simulation needed."""
    print("\n" + "=" * 60)
    print("DEMO 1: Pure Control Layer (no simulation)")
    print("=" * 60)

    ctrl = HandController()

    # ---- Open pose ----
    print("\n👉 open_pose() — all motors to zero")
    ctrl.open_pose()
    targets = ctrl.get_joint_targets()
    print(f"   finger_2_link3_joint = {targets.get('finger_2_link3_joint', 0):.3f} rad")

    # ---- Close pose ----
    print("\n👉 close_pose() — fist")
    ctrl.close_pose()
    targets = ctrl.get_joint_targets()
    for name in sorted(targets):
        print(f"   {name:25s} = {targets[name]:7.3f} rad ({np.degrees(targets[name]):6.1f}°)")

    # ---- Finger-level control ----
    print("\n👉 set_index(m1=4.0, m2=3.0, m3=-0.3)")
    ctrl.open_pose()
    ctrl.set_index(m1=4.0, m2=3.0, m3=-0.3)
    joints = ctrl.get_finger_joints("index")
    for name in sorted(joints):
        print(f"   {name:25s} = {joints[name]:7.3f} rad ({np.degrees(joints[name]):6.1f}°)")

    # ---- set_finger() generic API ----
    print("\n👉 set_finger('middle', m1=5.0, m2=4.0, m3=0.2)")
    ctrl.set_finger("middle", m1=5.0, m2=4.0, m3=0.2)
    joints = ctrl.get_finger_joints("middle")
    for name in sorted(joints):
        print(f"   {name:25s} = {joints[name]:7.3f} rad")

    # ---- Individual motor control ----
    print("\n👉 set_motor('ring_m1', 3.0) + set_motor('ring_m2', 2.0)")
    ctrl.open_pose()
    ctrl.set_motor("ring_m1", 3.0)
    ctrl.set_motor("ring_m2", 2.0)
    targets = ctrl.get_joint_targets()
    for name in sorted(targets):
        if "finger_4" in name or "knuckle_4" in name:
            print(f"   {name:25s} = {targets[name]:7.3f} rad")

    # ---- set_all_motors() array API ----
    print("\n👉 set_all_motors(array)")
    ctrl.open_pose()
    motors = np.zeros(17)
    motors[12] = 5.0   # pinky_m1
    motors[13] = 3.5   # pinky_m2
    motors[14] = -0.3  # pinky_m3
    ctrl.set_all_motors(motors)
    targets = ctrl.get_joint_targets()
    for name in sorted(targets):
        if "finger_5" in name or "knuckle_5" in name:
            print(f"   {name:25s} = {targets[name]:7.3f} rad")

    # ---- Motor dict inspection ----
    print("\n👉 get_motor_dict()")
    motor_dict = ctrl.hand.get_motor_dict()
    for name, val in list(motor_dict.items())[:5]:
        print(f"   {name:15s} = {val:7.3f}")

    # ---- Full observation ----
    print("\n👉 get_observation()")
    obs = ctrl.get_observation()
    print(f"   Keys: {list(obs.keys())}")

    print("\n✅ Demo 1 complete.")


def demo_block() -> None:
    """Demo 2: Block (tank) visualization of the cable-driven cascade."""
    print("\n" + "=" * 60)
    print("DEMO 2: Block Cascade Visualization")
    print("=" * 60)
    print("""
  Each motor is shown as a "block" (bidirectional progress bar).
  The cascade acts like filling tanks in order:

    Motor1 → Distal tank → Middle tank → Proximal tank → Palm
    Motor2 → Middle tank → Distal follows → Proximal → Palm

  Final angle per joint = max(Motor1_effect, Motor2_effect).
""")

    # Index finger: small motor1 → only distal bends
    print("\n" + "─" * 60)
    print("INDEX: Motor1 = 0.26 rad (15°), Motor2 = 0")
    print("        → ONLY the distal (fingertip) bends!")
    print(draw_finger_cascade(m1=np.radians(15), m2=0.0, m3=0.0, finger_name="index"))

    # Index finger: both motors small → distal + middle
    print("\n" + "─" * 60)
    print("INDEX: Motor1 = 0.26 rad, Motor2 = 0.26 rad")
    print("        → Distal AND Middle bend!")
    print(draw_finger_cascade(m1=np.radians(15), m2=np.radians(15), m3=0.0, finger_name="index"))

    # Index finger: motor1 large → distal saturates, overflow to middle
    print("\n" + "─" * 60)
    print("INDEX: Motor1 = 1.83 rad (105°), Motor2 = 0")
    print("        → Distal SATURATES, overflow drives Middle!")
    print(draw_finger_cascade(m1=np.radians(105), m2=0.0, m3=0.0, finger_name="index"))

    # Index finger: both motors large → full cascade
    print("\n" + "─" * 60)
    print("INDEX: Motor1 = 3.50 rad, Motor2 = 3.00 rad")
    print("        → FULL flexion cascade!")
    print(draw_finger_cascade(m1=3.50, m2=3.00, m3=0.0, finger_name="index"))

    # Thumb
    print("\n" + "─" * 60)
    print("THUMB: Motor1 = 2.5 rad, Motor2 = 1.0 rad")
    print(draw_thumb_cascade(m1=2.5, m2=1.0, m3=-0.5))

    # Knuckle only
    print("\n" + "─" * 60)
    print("INDEX: Knuckle = +0.4 rad outward (abduction)")
    print(draw_finger_cascade(m1=0.0, m2=0.0, m3=0.4, finger_name="index"))

    print("\n✅ Demo 2 complete.")


def demo_transmission_model() -> None:
    """Demo 3: Direct use of the transmission model."""
    print("\n" + "=" * 60)
    print("DEMO 3: CascadeTransmissionModel (motor → joint)")
    print("=" * 60)

    model = CascadeTransmissionModel()

    # ---- Normalize / denormalize ----
    print("\n👉 normalize_motor / denormalize")
    action = np.array([1.0, 0.5, -0.3] + [0.0] * 14)
    motors = model.denormalize(action)
    print(f"   action[0]=1.0  →  motor = {motors[0]:.2f} (range ±4.0)")
    print(f"   action[1]=0.5  →  motor = {motors[1]:.2f} (range ±3.0)")
    print(f"   action[2]=-0.3 →  motor = {motors[2]:.2f} (range ±2.5)")

    # ---- Map action to joints ----
    print("\n👉 map(action) — 17D motor → 22D joint")
    action = np.zeros(17)
    action[3] = 1.0   # index_m1 full flex
    action[4] = 0.8   # index_m2 partial flex
    action[5] = 0.5   # index_m3 knuckle outward
    targets = model.map(action)
    for name in sorted(targets):
        if "finger_2" in name or "knuckle_2" in name or "palm" in name:
            print(f"   {name:25s} = {targets[name]:7.3f} rad")

    # ---- motor_dict_to_joint_dict ----
    print("\n👉 motor_dict_to_joint_dict()")
    motor_dict = {
        "thumb_m1": 2.0, "thumb_m2": 1.0, "thumb_m3": -1.0,
        "index_m1": 4.0, "index_m2": 3.0, "index_m3": -0.3,
    }
    targets = model.motor_dict_to_joint_dict(motor_dict)
    print(f"   Generated {len(targets)} joint targets")
    for name in sorted(targets)[:6]:
        print(f"   {name:25s} = {targets[name]:7.3f} rad")

    print("\n✅ Demo 3 complete.")


def demo_inverse_transmission() -> None:
    """Demo 4: Inverse mapping (joint angles → motor values)."""
    print("\n" + "=" * 60)
    print("DEMO 4: InverseTransmissionModel (joint → motor)")
    print("=" * 60)

    # ---- Finger inverse ----
    inv = FingerInverseTransmission()
    print("\n👉 Finger: link3=30°, link2=60°, link1=90°, knuckle=10°")
    motors = inv.invert(
        link3=np.radians(30),
        link2=np.radians(60),
        link1=np.radians(90),
        knuckle=np.radians(10),
    )
    print(f"   m1 = {motors['m1']:.3f}")
    print(f"   m2 = {motors['m2']:.3f}")
    print(f"   m3 = {motors['m3']:.3f}")

    # ---- Thumb inverse ----
    inv_thumb = ThumbInverseTransmission()
    print("\n👉 Thumb: link2=45°, link1=60°, knuckle=-20°")
    motors = inv_thumb.invert(
        link2=np.radians(45),
        link1=np.radians(60),
        knuckle=np.radians(-20),
    )
    print(f"   m1 = {motors['m1']:.3f}")
    print(f"   m2 = {motors['m2']:.3f}")
    print(f"   m3 = {motors['m3']:.3f}")

    # ---- Forward verification (finger, cascade-compatible pose) ----
    print("\n👉 Verify: invert → forward → compare")
    inv_finger = FingerInverseTransmission()
    desired = {"link3": np.radians(30), "link2": np.radians(30),
               "link1": 0.0, "knuckle": np.radians(10)}
    motors = inv_finger.invert(**desired)
    print(f"   Requested: link3=30°, link2=30°, link1=0°, knuckle=10°")
    print(f"   Inverted motors: m1={motors['m1']:.3f}, m2={motors['m2']:.3f}, m3={motors['m3']:.3f}")

    fwd = CascadeTransmissionModel()
    action = np.zeros(17)
    action[3] = fwd.normalize_motor(motors['m1'], 3)
    action[4] = fwd.normalize_motor(motors['m2'], 4)
    action[5] = fwd.normalize_motor(motors['m3'], 5)
    targets = fwd.map(action)
    print(f"   Forward output:")
    print(f"     finger_2_link3_joint = {np.degrees(targets['finger_2_link3_joint']):.1f}°")
    print(f"     finger_2_link2_joint = {np.degrees(targets['finger_2_link2_joint']):.1f}°")
    print(f"     finger_2_link1_joint = {np.degrees(targets['finger_2_link1_joint']):.1f}°")
    print(f"     knuckle_2_joint      = {np.degrees(targets['knuckle_2_joint']):.1f}°")
    print(f"   (Note: link3 may differ because motor2 couples distal→middle)")

    print("\n✅ Demo 4 complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tendon Hand Library Demo (no URDF needed)")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["control", "block", "transmission", "inverse", "all"],
        help="Which demo to run",
    )
    args = parser.parse_args()

    if args.mode in ("control", "all"):
        demo_pure_control()
    if args.mode in ("block", "all"):
        demo_block()
    if args.mode in ("transmission", "all"):
        demo_transmission_model()
    if args.mode in ("inverse", "all"):
        demo_inverse_transmission()

    if args.mode == "all":
        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()

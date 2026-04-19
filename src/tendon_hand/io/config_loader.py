"""YAML config loader for hand models.

This is a minimal v1 implementation. Full config-driven instantiation
will be expanded in later versions.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from tendon_hand.core.hand import Hand
from tendon_hand.core.actuator import Actuator
from tendon_hand.core.joint import Joint
from tendon_hand.core.finger import Finger
from tendon_hand.core.palm import Palm
from tendon_hand.core.tendon import Tendon, RoutingElement


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_hand_config(config_path: str) -> Hand:
    """Construct a Hand from a YAML config file.

    This is a placeholder that builds a default V3-like hand.
    Full config parsing will be implemented in v2.
    """
    cfg = load_yaml(config_path)
    hand_cfg = cfg.get("hand", {})

    hand = Hand(id=hand_cfg.get("name", "V3"))

    # Create actuators from motor config
    for name, (lo, hi) in hand_cfg.get("motor_ranges", {}).items():
        # spool radius is a default; real config should specify
        act = Actuator(id=name, motor_torque_limit=0.5, spool_radius=0.008)
        hand.actuators.append(act)

    # Create palm joints
    palm_cfg = hand_cfg.get("joint_limits", {})
    thumb_lim = palm_cfg.get("thumb", {})
    finger_lim = palm_cfg.get("fingers", {})

    for pid in ["palm_1_joint", "palm_3_joint", "palm_4_joint"]:
        lo, hi = 0.0, 0.087
        if pid == "palm_1_joint":
            lo, hi = thumb_lim.get("palm", [0.0, 0.087])
        else:
            lo, hi = finger_lim.get("palm", [0.0, 0.087])
        hand.palm.joints.append(Joint(id=pid, lower_limit=lo, upper_limit=hi))

    # Create fingers (minimal placeholder joints)
    finger_names = [
        ("thumb", "finger_1", thumb_lim),
        ("index", "finger_2", finger_lim),
        ("middle", "finger_3", finger_lim),
        ("ring", "finger_4", finger_lim),
        ("pinky", "finger_5", finger_lim),
    ]

    for fname, prefix, limits in finger_names:
        joints = []
        if fname == "thumb":
            for link, (lo, hi) in [
                ("link2", limits.get("link2", [-0.217, 1.650])),
                ("link1", limits.get("link1", [-0.087, 2.268])),
            ]:
                joints.append(Joint(
                    id=f"{prefix}_{link}_joint",
                    lower_limit=lo, upper_limit=hi,
                ))
        else:
            for link, (lo, hi) in [
                ("link3", limits.get("link3", [-0.087, 1.650])),
                ("link2", limits.get("link2", [-0.087, 1.650])),
                ("link1", limits.get("link1", [-0.087, 2.268])),
            ]:
                joints.append(Joint(
                    id=f"{prefix}_{link}_joint",
                    lower_limit=lo, upper_limit=hi,
                ))

        # Knuckle joint
        klo, khi = limits.get("knuckle", [-0.436, 0.436])
        if fname == "thumb":
            klo, khi = limits.get("knuckle", [-2.260, 1.750])
        knuckle_id = f"knuckle_{['thumb','index','middle','ring','pinky'].index(fname)+1}_joint"
        joints.append(Joint(id=knuckle_id, lower_limit=klo, upper_limit=khi))

        finger = Finger(id=fname, joints=joints, tendons=[])
        hand.fingers.append(finger)

    return hand

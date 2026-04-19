"""Inverse transmission: joint angles → motor values.

This is the reverse of the cascade transmission model.
Given desired joint angles, compute the motor values that would produce them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class InverseTransmissionModel(Protocol):
    """Protocol for joint angles → motor values."""

    def invert(
        self,
        link3: float = 0.0,
        link2: float = 0.0,
        link1: float = 0.0,
        knuckle: float = 0.0,
        palm: float = 0.0,
    ) -> dict[str, float]:
        """Return {m1, m2, m3} motor values."""
        ...


@dataclass
class FingerInverseTransmission:
    """Inverse transmission for a 3-link finger."""

    link3_lim: tuple[float, float] = (-0.087, 1.650)
    link2_lim: tuple[float, float] = (-0.087, 1.650)
    link1_lim: tuple[float, float] = (-0.087, 2.268)
    knuckle_lim: tuple[float, float] = (-0.436, 0.436)
    palm_lim: tuple[float, float] = (0.0, 0.087)
    knuckle_sign: int = 1

    def invert(
        self,
        link3: float = 0.0,
        link2: float = 0.0,
        link1: float = 0.0,
        knuckle: float = 0.0,
        palm: float = 0.0,
    ) -> dict[str, float]:
        """Compute motor values from desired joint angles."""
        link3 = float(np.clip(link3, *self.link3_lim))
        link2 = float(np.clip(link2, *self.link2_lim))
        link1 = float(np.clip(link1, *self.link1_lim))
        palm = float(np.clip(palm, *self.palm_lim))

        # motor1 drives distal→middle→proximal→palm
        m1 = link3 + link2 + link1 + palm
        # motor2 drives middle→proximal→palm (distal follows middle)
        m2 = link2 + link1 + palm
        # motor3 = knuckle (with sign correction)
        m3 = self.knuckle_sign * knuckle

        return {"m1": round(m1, 4), "m2": round(m2, 4), "m3": round(m3, 4)}


@dataclass
class ThumbInverseTransmission:
    """Inverse transmission for the thumb (2-link)."""

    link2_lim: tuple[float, float] = (-0.217, 1.650)
    link1_lim: tuple[float, float] = (-0.087, 2.268)
    knuckle_lim: tuple[float, float] = (-2.260, 1.750)
    palm_lim: tuple[float, float] = (0.0, 0.087)
    knuckle_sign: int = 1

    def invert(
        self,
        link3: float = 0.0,
        link2: float = 0.0,
        link1: float = 0.0,
        knuckle: float = 0.0,
        palm: float = 0.0,
    ) -> dict[str, float]:
        """Compute motor values from desired thumb joint angles."""
        link2 = float(np.clip(link2, *self.link2_lim))
        link1 = float(np.clip(link1, *self.link1_lim))
        palm = float(np.clip(palm, *self.palm_lim))

        m1 = link2 + link1 + palm
        m2 = link1 + palm
        m3 = self.knuckle_sign * knuckle

        return {"m1": round(m1, 4), "m2": round(m2, 4), "m3": round(m3, 4)}

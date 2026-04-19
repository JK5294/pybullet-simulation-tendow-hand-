"""Math utilities: angle conversion, clamping, interpolation."""

from __future__ import annotations

import numpy as np


def rad2deg(rad: float) -> float:
    """Radians to degrees."""
    return float(np.degrees(rad))


def deg2rad(deg: float) -> float:
    """Degrees to radians."""
    return float(np.radians(deg))


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value from [lo, hi] to [-1, 1]."""
    mid = (lo + hi) / 2.0
    span = (hi - lo) / 2.0
    if span < 1e-9:
        return 0.0
    return clamp((value - mid) / span, -1.0, 1.0)


def denormalize(norm_value: float, lo: float, hi: float) -> float:
    """Map [-1, 1] back to [lo, hi]."""
    mid = (lo + hi) / 2.0
    span = (hi - lo) / 2.0
    return mid + clamp(norm_value, -1.0, 1.0) * span


def cosine_interpolate(t: float) -> float:
    """Smooth cosine interpolation: t ∈ [0,1] → [0,1]."""
    return 0.5 * (1.0 - np.cos(t * np.pi))

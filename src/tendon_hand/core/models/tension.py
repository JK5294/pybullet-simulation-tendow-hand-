"""Tension models: tendon displacement → tension computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TensionModel(Protocol):
    """Protocol for computing tendon tension from length mismatch."""

    def compute_tension(
        self,
        commanded_length: float,
        actual_length: float,
        prev_cmd_length: float,
        dt: float,
    ) -> float:
        ...


@dataclass
class KinematicTensionModel:
    """Level 0: no tension, just length."""

    def compute_tension(
        self,
        commanded_length: float,
        actual_length: float,
        prev_cmd_length: float,
        dt: float,
    ) -> float:
        return 0.0


@dataclass
class QuasiStaticTensionModel:
    """Level 1: T = max(0, k*(l_cmd - l_act) - c*dl).

    Supports unidirectional tension (cables can't push).
    """

    elasticity_k: float = 120.0  # N/m
    damping_c: float = 0.2  # N/(m/s)
    min_tension: float = 0.0
    max_tension: float | None = None

    def compute_tension(
        self,
        commanded_length: float,
        actual_length: float,
        prev_cmd_length: float,
        dt: float,
    ) -> float:
        delta_l = commanded_length - actual_length
        dl = (commanded_length - prev_cmd_length) / max(dt, 1e-9)
        T = self.elasticity_k * delta_l - self.damping_c * dl
        T = max(self.min_tension, T)
        if self.max_tension is not None:
            T = min(T, self.max_tension)
        return T


@dataclass
class HysteresisTensionModel:
    """Level 2: elasticity + release hysteresis.

    pulling: h = 1.0
    releasing: h = eta (0 < eta < 1)
    """

    elasticity_k: float = 120.0
    damping_c: float = 0.2
    hysteresis_eta: float = 0.8  # tension drops when releasing
    min_tension: float = 0.0
    max_tension: float | None = None

    def compute_tension(
        self,
        commanded_length: float,
        actual_length: float,
        prev_cmd_length: float,
        dt: float,
    ) -> float:
        delta_l = commanded_length - actual_length
        dl = (commanded_length - prev_cmd_length) / max(dt, 1e-9)

        direction = "pull" if delta_l >= 0 else "release"
        eta = 1.0 if direction == "pull" else self.hysteresis_eta

        T = self.elasticity_k * delta_l * eta - self.damping_c * dl
        T = max(self.min_tension, T)
        if self.max_tension is not None:
            T = min(T, self.max_tension)
        return T

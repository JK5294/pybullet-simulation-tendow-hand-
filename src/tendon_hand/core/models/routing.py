"""Routing models: tension propagation along tendon routing elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tendon_hand.core.tendon import RoutingElement


class RoutingModel(Protocol):
    """Protocol for tension propagation along routing."""

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        """Return {joint_id: local_tension} after propagation."""
        ...


@dataclass
class IdentityRoutingModel:
    """No loss: tension is the same at all joints."""

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        return {elem.joint_id: base_tension for elem in routing_elements}


@dataclass
class SimpleRoutingLossModel:
    """Simplified routing loss: T_{j+1} = T_j * (1 - λ_j)."""

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        tensions: dict[str, float] = {}
        T = base_tension
        for elem in routing_elements:
            T = T * (1.0 - elem.routing_loss)
            tensions[elem.joint_id] = max(0.0, T)
        return tensions


@dataclass
class CapstanRoutingModel:
    """Capstan-like friction: T_{out} = T_{in} * exp(-μ*θ).

    v2 enhancement; v1 uses SimpleRoutingLossModel.
    """

    mu: float = 0.15  # friction coefficient

    def propagate(
        self,
        base_tension: float,
        routing_elements: list[RoutingElement],
    ) -> dict[str, float]:
        import math

        tensions: dict[str, float] = {}
        T = base_tension
        for elem in routing_elements:
            # Approximate wrap angle from routing_loss
            theta = -math.log(max(1e-9, 1.0 - elem.routing_loss))
            T = T * math.exp(-self.mu * theta)
            tensions[elem.joint_id] = max(0.0, T)
        return tensions

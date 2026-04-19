"""Contact observation utilities for PyBullet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContactPoint:
    """A single contact point between two bodies."""

    body_a: int
    body_b: int
    link_index_a: int
    link_index_b: int
    position_on_a: tuple[float, float, float]
    position_on_b: tuple[float, float, float]
    contact_normal: tuple[float, float, float]
    contact_distance: float
    normal_force: float
    lateral_friction_1: float
    lateral_friction_dir_1: tuple[float, float, float]
    lateral_friction_2: float
    lateral_friction_dir_2: tuple[float, float, float]


@dataclass
class ContactObserver:
    """Observes contacts on a hand body in PyBullet."""

    hand_body_id: int
    physics_client_id: int = 0

    def get_contacts(self, other_body_id: int | None = None) -> list[ContactPoint]:
        """Get all contact points involving the hand.

        If other_body_id is None, returns contacts with any body.
        """
        import pybullet as p

        if other_body_id is not None:
            pts = p.getContactPoints(
                self.hand_body_id, other_body_id,
                physicsClientId=self.physics_client_id,
            )
        else:
            # Get all contacts by querying with -1 (world) and filtering
            pts = p.getContactPoints(
                self.hand_body_id, -1,
                physicsClientId=self.physics_client_id,
            )

        contacts = []
        for pt in pts:
            contacts.append(ContactPoint(
                body_a=pt[1],
                body_b=pt[2],
                link_index_a=pt[3],
                link_index_b=pt[4],
                position_on_a=(pt[5], pt[6], pt[7]),
                position_on_b=(pt[8], pt[9], pt[10]),
                contact_normal=(pt[11], pt[12], pt[13]),
                contact_distance=pt[14],
                normal_force=pt[9] if len(pt) > 9 else 0.0,  # force is at index 9 in some versions
                lateral_friction_1=pt[10] if len(pt) > 10 else 0.0,
                lateral_friction_dir_1=(0.0, 0.0, 0.0),
                lateral_friction_2=0.0,
                lateral_friction_dir_2=(0.0, 0.0, 0.0),
            ))
        return contacts

    def get_contact_summary(self) -> dict[str, Any]:
        """Return a summary of current contacts."""
        contacts = self.get_contacts()
        total_force = sum(c.normal_force for c in contacts)
        num_contacts = len(contacts)
        return {
            "num_contacts": num_contacts,
            "total_normal_force": total_force,
            "contacts": contacts,
        }

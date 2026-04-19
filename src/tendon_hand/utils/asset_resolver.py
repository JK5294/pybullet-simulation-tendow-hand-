"""Asset path resolver — for users who provide their own URDF and meshes.

This library does NOT bundle any robot model files (URDF/STL).
Users must supply their own URDF path when creating a simulation.
"""

from __future__ import annotations

import os


def resolve_urdf_path(path: str | None = None) -> str:
    """Resolve a URDF path.

    If *path* is given, return it after verifying the file exists.
    If *path* is None, raise an error prompting the user to supply one.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        RuntimeError: If no path is provided.
    """
    if path is None:
        raise RuntimeError(
            "No URDF path provided. Please supply your own URDF file:\n"
            "  sim = PyBulletHandAdapter(hand=..., urdf_path='/path/to/hand.urdf')\n"
            "The tendon_hand library does not bundle robot models."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"URDF not found: {path}")
    return os.path.abspath(path)

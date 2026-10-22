"""Small SPD calibration helpers used by the OpenCOOD format adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def read_transform(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read an SPD rotation and translation transform, or return ``None``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    node = payload.get("transform", payload)
    if not isinstance(node, dict):
        return None
    try:
        rotation = np.asarray(node["rotation"], dtype=float).reshape(3, 3)
        translation = np.asarray(node["translation"], dtype=float).reshape(-1)[:3]
    except (KeyError, TypeError, ValueError):
        return None
    if translation.size != 3:
        return None
    return rotation, translation


def yaw_from_R(rotation: np.ndarray) -> float:
    """Return yaw in degrees under the SPD ENU convention."""
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))

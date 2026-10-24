"""Small 2-D geometry helpers (line sides, crossings, box overlap)."""
from __future__ import annotations

import math

import numpy as np


def signed_distance(p, a, b) -> float:
    """Signed perpendicular distance from p to the infinite line a->b (pixels)."""
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay) or 1.0
    return ((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)) / length


def projection_t(p, a, b) -> float:
    """Position of p projected on segment a->b: 0 at a, 1 at b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy or 1.0
    return ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom


def side_for_direction(a, b, direction: str) -> int:
    """Return +1/-1: the side of the line a person ends up on when walking `direction`."""
    vec = {"down": (0, 1), "up": (0, -1), "left": (-1, 0), "right": (1, 0)}[direction]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    d = signed_distance((mid[0] + vec[0] * 50, mid[1] + vec[1] * 50), a, b)
    if abs(d) < 1e-6:
        raise ValueError(f"Direction '{direction}' runs parallel to the entry line; pick another one.")
    return 1 if d > 0 else -1


def side_for_point(a, b, point) -> int:
    d = signed_distance(point, a, b)
    if abs(d) < 1e-6:
        raise ValueError("Inside point lies on the entry line.")
    return 1 if d > 0 else -1


def box_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def box_containment(b1, b2) -> float:
    """Intersection divided by the area of the smaller box."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (min(a1, a2) + 1e-9)


def angle_between(v1, v2) -> float:
    """Angle between two vectors in degrees."""
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(c))

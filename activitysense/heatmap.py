"""Movement heatmap + activity map accumulated from people's ground points."""
from __future__ import annotations

import cv2
import numpy as np

from .config import ACTIVITY_COLORS


class Heatmap:
    def __init__(self, shape: tuple[int, int], radius: int):
        self.h, self.w = shape
        self.acc = np.zeros((self.h, self.w), np.float32)
        self.r = radius
        k = cv2.getGaussianKernel(2 * radius + 1, radius / 2.5)
        self.kernel = (k @ k.T).astype(np.float32)
        self.kernel /= self.kernel.max()
        self.points: list[tuple[int, int, str]] = []   # for the activity map

    def add(self, x: float, y: float, activity: str):
        x, y = int(round(x)), int(round(y))
        self.points.append((x, y, activity))
        r = self.r
        x0, x1 = max(0, x - r), min(self.w, x + r + 1)
        y0, y1 = max(0, y - r), min(self.h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kx0, ky0 = x0 - (x - r), y0 - (y - r)
        self.acc[y0:y1, x0:x1] += self.kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]

    def colorize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (BGR heat image, alpha mask 0..1)."""
        if self.acc.max() <= 0:
            return np.zeros((self.h, self.w, 3), np.uint8), np.zeros((self.h, self.w), np.float32)
        norm = np.log1p(self.acc) / np.log1p(self.acc.max())
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        alpha = np.clip(norm * 1.6, 0, 0.75)
        return img, alpha

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        img, alpha = self.colorize()
        a = alpha[..., None]
        return (frame * (1 - a) + img * a).astype(np.uint8)

    def activity_map(self, background: np.ndarray) -> np.ndarray:
        """Background faded to 35%, with a dot per observation coloured by activity."""
        out = (background * 0.35).astype(np.uint8)
        for x, y, act in self.points[::2]:
            cv2.circle(out, (x, y), 2, ACTIVITY_COLORS.get(act, ACTIVITY_COLORS["unknown"]), -1, cv2.LINE_AA)
        y = 24
        for act, col in ACTIVITY_COLORS.items():
            if act == "unknown":
                continue
            cv2.circle(out, (18, y - 5), 6, col, -1, cv2.LINE_AA)
            cv2.putText(out, act, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            y += 22
        return out

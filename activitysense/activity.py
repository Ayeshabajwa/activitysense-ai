"""Pose + motion based human activity recognition.

Every frame each person gets a *raw* label from their 17 COCO keypoints and their
speed (normalised by body height so it works at any distance from the camera).
Raw labels are then smoothed with a majority vote over a short window, and a fall
is flagged when someone goes from upright to lying within `fall_window_s`.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .geometry import angle_between

# COCO-17 keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = (
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

UPRIGHT = {"standing", "walking", "running", "hands_up"}


@dataclass
class PoseFeatures:
    torso_angle: float | None = None   # degrees from vertical (0 = upright)
    knee_angle: float | None = None    # degrees at the knee (180 = straight leg)
    thigh_drop: float | None = None    # vertical hip->knee distance / torso length
    hands_up: bool = False
    aspect: float = 0.5                # box width / height
    n_valid: int = 0


def _mid(kp, ok, i, j):
    if ok[i] and ok[j]:
        return (kp[i] + kp[j]) / 2
    if ok[i]:
        return kp[i]
    if ok[j]:
        return kp[j]
    return None


def pose_features(kp: np.ndarray | None, kconf: np.ndarray | None, box, kpt_thr: float) -> PoseFeatures:
    w, h = box[2] - box[0], box[3] - box[1]
    f = PoseFeatures(aspect=float(w / max(h, 1e-6)))
    if kp is None or kconf is None:
        return f
    ok = kconf >= kpt_thr
    f.n_valid = int(ok.sum())

    sh = _mid(kp, ok, L_SH, R_SH)
    hip = _mid(kp, ok, L_HIP, R_HIP)
    knee = _mid(kp, ok, L_KNEE, R_KNEE)

    torso_len = None
    if sh is not None and hip is not None:
        v = sh - hip
        torso_len = float(np.linalg.norm(v))
        if torso_len > 2:
            f.torso_angle = angle_between(v, np.array([0.0, -1.0]))

    knee_angles = []
    for hp, kn, an in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if ok[hp] and ok[kn] and ok[an]:
            knee_angles.append(angle_between(kp[hp] - kp[kn], kp[an] - kp[kn]))
    if knee_angles:
        f.knee_angle = float(np.mean(knee_angles))

    if hip is not None and knee is not None and torso_len and torso_len > 2:
        f.thigh_drop = float((knee[1] - hip[1]) / torso_len)

    # Hands up: a wrist clearly above the head (or above the shoulders if no head).
    ref_y = kp[NOSE][1] if ok[NOSE] else (sh[1] - 0.35 * (torso_len or h * 0.3) if sh is not None else None)
    if ref_y is not None:
        for wr in (L_WR, R_WR):
            if ok[wr] and kp[wr][1] < ref_y:
                f.hands_up = True
    return f


def classify_frame(f: PoseFeatures, speed: float, walk_speed: float, run_speed: float) -> str:
    """Raw per-frame activity label. Priority: lying > sitting > hands_up > motion."""
    # Lying / fallen: torso far from vertical, or a wide box when the torso is not visible.
    if f.torso_angle is not None:
        if f.torso_angle > 60 and f.aspect > 0.9:
            return "lying"
    elif f.aspect > 1.5:
        return "lying"

    # Sitting: upright torso, bent knees, thighs roughly horizontal, and not moving.
    if (f.torso_angle is not None and f.torso_angle < 40 and speed < walk_speed * 1.5
            and f.thigh_drop is not None):
        bent = f.knee_angle is not None and f.knee_angle < 130
        if (bent and f.thigh_drop < 0.6) or f.thigh_drop < 0.3:
            return "sitting"

    if f.hands_up and speed < run_speed:
        return "hands_up"
    if speed >= run_speed:
        return "running"
    if speed >= walk_speed:
        return "walking"
    return "standing"


class ActivityState:
    """Per-person temporal smoothing + fall detection."""

    def __init__(self, window: int, fall_window_frames: int):
        self.raw = deque(maxlen=window)
        self.label = "unknown"
        self.fall_window_frames = fall_window_frames
        self.last_upright_frame: int | None = None
        self.fallen = False

    def update(self, raw_label: str, frame_idx: int) -> tuple[str, bool]:
        """Feed a raw label, return (smoothed label, fall_just_detected)."""
        self.raw.append(raw_label)
        if raw_label in UPRIGHT:
            self.last_upright_frame = frame_idx

        counts = Counter(self.raw)
        best, n = counts.most_common(1)[0]
        # Lying needs a clear majority, other labels just need to win the vote.
        if counts.get("lying", 0) >= max(2, len(self.raw) // 2):
            best = "lying"
        elif best == "lying":
            best = Counter({k: v for k, v in counts.items() if k != "lying"}).most_common(1)[0][0] \
                if len(counts) > 1 else best

        fall = False
        if best == "lying" and self.label != "lying" and not self.fallen:
            if (self.last_upright_frame is not None
                    and frame_idx - self.last_upright_frame <= self.fall_window_frames + len(self.raw)):
                fall = True
                self.fallen = True
        if best in UPRIGHT:
            self.fallen = False
        self.label = best
        return best, fall

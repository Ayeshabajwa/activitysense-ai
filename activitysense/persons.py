"""Person registry: one stable identity per human for the whole video.

BoT-SORT already gives track IDs, but trackers still drop and re-create IDs when
people overlap, turn, or are occluded. This layer sits on top of the tracker and:

* relinks a new tracker ID to a recently lost person (motion prediction + size +
  clothing colour histogram), so the person keeps the same ID;
* guards against ID switches (a tracker ID that teleports is detached);
* drops duplicate boxes on the same person before they can create a ghost person.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .activity import ActivityState, classify_frame, pose_features
from .config import Config
from .geometry import box_containment, box_iou


@dataclass
class Detection:
    tid: int
    box: np.ndarray            # x1, y1, x2, y2
    conf: float
    kpts: np.ndarray | None    # (17, 2)
    kconf: np.ndarray | None   # (17,)

    @property
    def anchor(self):
        """Ground contact point (bottom centre of the box)."""
        return float((self.box[0] + self.box[2]) / 2), float(self.box[3])

    @property
    def height(self):
        return float(self.box[3] - self.box[1])


def appearance_hist(frame: np.ndarray, box) -> np.ndarray | None:
    """Clothing descriptor: HSV histogram of the torso region of the box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    y1t, y2t = y1 + int(0.15 * h), y1 + int(0.65 * h)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    y1t, y2t = max(0, y1t), min(frame.shape[0], y2t)
    if x2 - x1 < 4 or y2t - y1t < 4:
        return None
    hsv = cv2.cvtColor(frame[y1t:y2t, x1:x2], cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    vec = np.concatenate([hs, v]).astype(np.float32)
    return vec / (vec.sum() + 1e-9)


def hist_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.5
    return float(np.sum(np.sqrt(a * b)))  # Bhattacharyya coefficient, 1 = identical


@dataclass
class Person:
    pid: int
    first_frame: int
    last_frame: int
    tracker_ids: set = field(default_factory=set)
    box: np.ndarray | None = None
    kpts: np.ndarray | None = None
    kconf: np.ndarray | None = None
    anchors: deque = field(default_factory=lambda: deque(maxlen=90))   # (frame, x, y)
    heights: deque = field(default_factory=lambda: deque(maxlen=30))
    hist: np.ndarray | None = None
    conf_sum: float = 0.0
    conf_n: int = 0
    speed: float = 0.0
    activity: ActivityState | None = None
    activity_frames: Counter = field(default_factory=Counter)
    falls: int = 0
    just_fell: bool = False
    # entry-line state
    zone: str | None = None          # "out" | "in" | None
    seen_outside: bool = False
    seen_inside: bool = False
    entered: bool = False
    entry_frame: int | None = None
    entry_method: str | None = None
    exited: bool = False
    exit_frame: int | None = None
    last_signed_dist: float | None = None
    doorway_checked: bool = False

    @property
    def anchor(self):
        """Smoothed ground point (mean of the last 3 observations)."""
        pts = list(self.anchors)[-3:]
        return (sum(p[1] for p in pts) / len(pts), sum(p[2] for p in pts) / len(pts))

    @property
    def body_height(self) -> float:
        return float(np.median(self.heights)) if self.heights else 1.0

    @property
    def avg_conf(self) -> float:
        return self.conf_sum / max(self.conf_n, 1)

    def velocity(self, fps: float):
        """Pixels / second of the ground point over the last ~0.5 s."""
        if len(self.anchors) < 2:
            return 0.0, 0.0
        k = min(len(self.anchors) - 1, max(2, int(fps * 0.5)))
        f0, x0, y0 = self.anchors[-1 - k]
        f1, x1, y1 = self.anchors[-1]
        dt = max(f1 - f0, 1) / fps
        return (x1 - x0) / dt, (y1 - y0) / dt

    def predicted_anchor(self, frame_idx: int, fps: float):
        vx, vy = self.velocity(fps)
        gap = min(frame_idx - self.last_frame, int(fps)) / fps   # don't extrapolate too far
        x, y = self.anchor
        return x + vx * gap, y + vy * gap


class PersonRegistry:
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.persons: dict[int, Person] = {}
        self.tid_to_pid: dict[int, int] = {}
        self.next_pid = 1
        self.guard_events: list[tuple[int, str]] = []   # (frame, message)
        self.relinks = 0
        self.dup_suppressed = 0
        self.id_switch_blocked = 0

    # ------------------------------------------------------------------ guards
    def _dedupe(self, dets: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for d in sorted(dets, key=lambda d: -d.conf):
            dup = False
            for k in kept:
                if box_iou(d.box, k.box) > self.cfg.dup_iou:
                    dup = True
                elif box_containment(d.box, k.box) > 0.92 and 0.8 < d.height / max(k.height, 1) < 1.25:
                    dup = True
                if dup:
                    break
            if dup:
                self.dup_suppressed += 1
            else:
                kept.append(d)
        return kept

    def _is_teleport(self, p: Person, d: Detection, frame_idx: int) -> bool:
        gap = max(frame_idx - p.last_frame, 1)
        if gap > 5:
            return False
        dist = math.dist(p.anchor, d.anchor)
        return dist > self.cfg.max_jump_factor * max(p.body_height, d.height) * gap

    # ---------------------------------------------------------------- relinking
    def _relink_candidates(self, d: Detection, frame_idx: int, taken: set[int], hist):
        best, best_score = None, self.cfg.relink_min_score
        for p in self.persons.values():
            if p.pid in taken or p.last_frame >= frame_idx:
                continue
            gap = frame_idx - p.last_frame
            if gap > self.cfg.relink_max_frames:
                continue
            px, py = p.predicted_anchor(frame_idx, self.fps)
            h = max(p.body_height, d.height)
            dist = math.dist((px, py), d.anchor) / h
            max_dist = self.cfg.relink_dist_factor * (1 + gap / self.fps)
            if dist > max_dist:
                continue
            size_ratio = min(p.body_height, d.height) / max(p.body_height, d.height)
            if size_ratio < 0.6:
                continue
            app = hist_similarity(p.hist, hist)
            score = 0.45 * (1 - dist / max_dist) + 0.35 * app + 0.20 * size_ratio
            score -= 0.1 * gap / self.cfg.relink_max_frames
            if score > best_score:
                best, best_score = p, score
        return best

    # ------------------------------------------------------------------ update
    def update(self, frame_idx: int, frame: np.ndarray, dets: list[Detection]) -> list[Person]:
        dets = self._dedupe(dets)
        visible: list[Person] = []
        taken: set[int] = set()

        # Known tracker IDs first, so they claim their persons before relinking runs.
        dets.sort(key=lambda d: d.tid not in self.tid_to_pid)
        for d in dets:
            hist = appearance_hist(frame, d.box)
            pid = self.tid_to_pid.get(d.tid)
            if pid is not None and (pid in taken or self._is_teleport(self.persons[pid], d, frame_idx)):
                self.id_switch_blocked += 1
                self.guard_events.append((frame_idx, f"ID-switch guard: tracker {d.tid} detached from #{pid}"))
                del self.tid_to_pid[d.tid]
                pid = None

            if pid is None:
                p = self._relink_candidates(d, frame_idx, taken, hist)
                if p is not None:
                    self.relinks += 1
                    self.guard_events.append((frame_idx, f"Relinked tracker {d.tid} -> #{p.pid}"))
                else:
                    p = Person(pid=self.next_pid, first_frame=frame_idx, last_frame=frame_idx)
                    p.activity = ActivityState(self.cfg.activity_window,
                                               int(self.cfg.fall_window_s * self.fps))
                    self.persons[p.pid] = p
                    self.next_pid += 1
                self.tid_to_pid[d.tid] = p.pid
                pid = p.pid

            p = self.persons[pid]
            taken.add(pid)
            self._observe(p, d, hist, frame_idx)
            visible.append(p)
        return visible

    def _observe(self, p: Person, d: Detection, hist, frame_idx: int):
        p.tracker_ids.add(d.tid)
        p.box, p.kpts, p.kconf = d.box, d.kpts, d.kconf
        p.anchors.append((frame_idx, *d.anchor))
        p.heights.append(d.height)
        p.last_frame = frame_idx
        p.conf_sum += d.conf
        p.conf_n += 1
        p.doorway_checked = False
        if hist is not None:
            p.hist = hist if p.hist is None else 0.85 * p.hist + 0.15 * hist

        vx, vy = p.velocity(self.fps)
        speed = math.hypot(vx, vy) / max(p.body_height, 1.0)
        p.speed = 0.6 * p.speed + 0.4 * speed
        feats = pose_features(d.kpts, d.kconf, d.box, self.cfg.kpt_conf)
        raw = classify_frame(feats, p.speed, self.cfg.walk_speed, self.cfg.run_speed)
        label, fall = p.activity.update(raw, frame_idx)
        p.activity_frames[label] += 1
        if fall:
            p.falls += 1
        p.just_fell = fall

    def lost_persons(self, frame_idx: int) -> list[Person]:
        return [p for p in self.persons.values() if p.last_frame < frame_idx]

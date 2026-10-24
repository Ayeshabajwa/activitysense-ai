"""Line-crossing entry/exit counter.

Rules
-----
* +1 entry only when a person moves from the *outside* zone to the *inside* zone,
  across the drawn segment (not its infinite extension), once per person.
* A hysteresis band around the line (max of `line_margin_px` and
  `line_margin_factor` x body height) means jitter on the line never toggles the zone.
* Doorway logic: a person who was outside, was walking toward the line, and
  vanished right at it (dark doorway, occlusion) is counted after
  `doorway_lost_frames` frames. If that person reappears outside, the entry is
  rolled back.
* Dedupe guard: a crossing at the same place within `dedupe_seconds` of an entry
  by a person who is no longer visible is treated as the same person under a new
  ID, and is not counted twice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .geometry import projection_t, signed_distance
from .persons import Person


@dataclass
class CrossEvent:
    frame: int
    pid: int
    kind: str      # "entry" | "exit"
    method: str    # "line" | "doorway"
    point: tuple[float, float]


class LineCounter:
    def __init__(self, cfg: Config, a, b, fps: float):
        self.cfg = cfg
        self.a, self.b = a, b
        self.fps = fps
        self.entries = 0
        self.exits = 0
        self.events: list[CrossEvent] = []
        self.suppressed: list[tuple[int, int]] = []   # (frame, pid)
        self.revoked: list[tuple[int, int]] = []      # doorway entries rolled back

    def _inside_dist(self, pt) -> float:
        """Signed distance, positive = inside."""
        return signed_distance(pt, self.a, self.b) * self.cfg.inside_side

    def _on_segment(self, pt) -> bool:
        t = projection_t(pt, self.a, self.b)
        m = self.cfg.line_extent_margin
        return -m <= t <= 1 + m

    def _is_duplicate_entry(self, p: Person, frame_idx: int, visible_ids: set[int]) -> bool:
        window = self.cfg.dedupe_seconds * self.fps
        for e in reversed(self.events):
            if frame_idx - e.frame > window:
                break
            if e.kind != "entry" or e.pid == p.pid or e.pid in visible_ids:
                continue
            if math.dist(e.point, p.anchor) < 0.5 * p.body_height:
                return True
        return False

    def _count_entry(self, p: Person, frame_idx: int, method: str, visible_ids: set[int]):
        if p.entered:
            return None
        p.entered = True
        if self._is_duplicate_entry(p, frame_idx, visible_ids):
            p.entry_method = "duplicate-suppressed"
            self.suppressed.append((frame_idx, p.pid))
            return None
        p.entry_frame = frame_idx
        p.entry_method = method
        self.entries += 1
        ev = CrossEvent(frame_idx, p.pid, "entry", method, p.anchor)
        self.events.append(ev)
        return ev

    def update(self, visible: list[Person], lost: list[Person], frame_idx: int) -> list[CrossEvent]:
        new_events = []
        visible_ids = {p.pid for p in visible}

        for p in visible:
            pt = p.anchor
            d = self._inside_dist(pt)
            # hysteresis band scales with body size, so box jitter on the line never toggles the zone
            margin = max(self.cfg.line_margin_px, self.cfg.line_margin_factor * p.body_height)
            p.last_signed_dist = d
            if d < -margin:
                zone = "out"
            elif d > margin:
                zone = "in"
            else:
                zone = p.zone            # inside the hysteresis band: keep previous zone
            if zone is None:
                continue

            # Doorway rollback: a "vanished into the doorway" person who shows up outside again
            # never actually went in - undo that entry so their real crossing counts later.
            if zone == "out" and p.entered and p.entry_method == "doorway":
                self.entries -= 1
                self.events = [e for e in self.events if not (e.pid == p.pid and e.kind == "entry")]
                self.revoked.append((frame_idx, p.pid))
                p.entered, p.entry_frame, p.entry_method = False, None, None

            if p.zone == "out" and zone == "in" and self._on_segment(pt):
                ev = self._count_entry(p, frame_idx, "line", visible_ids)
                if ev:
                    new_events.append(ev)
            elif p.zone == "in" and zone == "out" and self._on_segment(pt) and not p.exited:
                p.exited = True
                p.exit_frame = frame_idx
                self.exits += 1
                ev = CrossEvent(frame_idx, p.pid, "exit", "line", pt)
                self.events.append(ev)
                new_events.append(ev)

            p.zone = zone
            p.seen_outside |= zone == "out"
            p.seen_inside |= zone == "in"

        if self.cfg.doorway:
            for p in lost:
                ev = self._doorway_check(p, frame_idx, visible_ids)
                if ev:
                    new_events.append(ev)
        return new_events

    def _doorway_check(self, p: Person, frame_idx: int, visible_ids: set[int]):
        if p.doorway_checked or p.entered or p.zone != "out" or p.last_signed_dist is None:
            return None
        if frame_idx - p.last_frame < self.cfg.doorway_lost_frames:
            return None
        p.doorway_checked = True
        # Vanished within a short band outside the line, on the segment, moving toward it.
        band = self.cfg.doorway_band_factor * p.body_height
        margin = max(self.cfg.line_margin_px, self.cfg.line_margin_factor * p.body_height)
        if not (-band <= p.last_signed_dist <= margin) or not self._on_segment(p.anchor):
            return None
        vx, vy = p.velocity(self.fps)
        nx, ny = self._inside_normal()
        toward = vx * nx + vy * ny
        if toward <= 0.05 * p.body_height:   # must be walking into the doorway
            return None
        return self._count_entry(p, frame_idx, "doorway", visible_ids)

    def _inside_normal(self):
        (ax, ay), (bx, by) = self.a, self.b
        length = math.hypot(bx - ax, by - ay) or 1.0
        # signed_distance > 0 is on the side of the left-hand normal (-dy, dx) in image coords
        nx, ny = -(by - ay) / length, (bx - ax) / length
        return nx * self.cfg.inside_side, ny * self.cfg.inside_side

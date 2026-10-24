"""Frame annotation and the live analytics side panel."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .activity import SKELETON
from .config import ACTIVITIES, ACTIVITY_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 22, 20)
CARD = (42, 38, 34)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
ACCENT = (255, 190, 70)
ALERT = (60, 60, 235)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class DashboardState:
    t: float = 0.0
    frame_idx: int = 0
    proc_fps: float = 0.0
    entered: int = 0
    exited: int = 0
    in_view: int = 0
    outside: int = 0
    falls: int = 0
    fallen_now: int = 0
    unique_people: int = 0
    activity_now: dict = field(default_factory=dict)
    avg_conf: float = 0.0
    timeline_entries: list = field(default_factory=list)   # cumulative entries per sample
    timeline_inview: list = field(default_factory=list)
    recent: list = field(default_factory=list)              # list[(time_str, text, color)]


# ---------------------------------------------------------------- video overlay
def annotate_frame(frame, persons, line, inside_side, cfg, scale: float):
    th = max(1, int(round(2 * scale)))
    fs = 0.5 * scale
    a, b = (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1]))
    cv2.line(frame, a, b, (0, 0, 0), th + 4, cv2.LINE_AA)
    cv2.line(frame, a, b, ACCENT, th + 1, cv2.LINE_AA)
    # inside-direction arrow from the middle of the line
    mid = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    d = np.array([b[0] - a[0], b[1] - a[1]], float)
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * inside_side
    tip = mid + n * 40 * scale
    cv2.arrowedLine(frame, tuple(mid.astype(int)), tuple(tip.astype(int)), ACCENT, th + 1, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(frame, "IN", tuple((tip + n * 14 * scale).astype(int) - [8, -5]), FONT, fs * 1.2, ACCENT, th, cv2.LINE_AA)

    for p in persons:
        act = p.activity.label
        col = ACTIVITY_COLORS.get(act, ACTIVITY_COLORS["unknown"])
        x1, y1, x2, y2 = p.box.astype(int)

        pts = [(int(x), int(y)) for _, x, y in list(p.anchors)[-cfg.trail_len:]]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], col, max(1, th - 1), cv2.LINE_AA)

        if p.kpts is not None:
            ok = p.kconf >= cfg.kpt_conf
            for i, j in SKELETON:
                if ok[i] and ok[j]:
                    cv2.line(frame, tuple(p.kpts[i].astype(int)), tuple(p.kpts[j].astype(int)),
                             col, max(1, th - 1), cv2.LINE_AA)
            for k in np.where(ok)[0]:
                cv2.circle(frame, tuple(p.kpts[k].astype(int)), max(2, th), (255, 255, 255), -1, cv2.LINE_AA)

        box_col = ALERT if act == "lying" else col
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, th, cv2.LINE_AA)
        tag = "IN" if p.entered and p.entry_method != "duplicate-suppressed" else ""
        label = f"#{p.pid} {act.replace('_', ' ')} {p.avg_conf:.2f} {tag}".strip()
        (tw, tht), _ = cv2.getTextSize(label, FONT, fs, th)
        cv2.rectangle(frame, (x1, y1 - tht - 8), (x1 + tw + 6, y1), box_col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), FONT, fs, (20, 20, 20), th, cv2.LINE_AA)
    return frame


def fall_banner(img, n_fallen: int, frame_idx: int):
    if n_fallen <= 0 or (frame_idx // 8) % 2:
        return img
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), ALERT, -1)
    msg = f"FALL ALERT - {n_fallen} person{'s' if n_fallen > 1 else ''} down"
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
    cv2.putText(img, msg, ((w - tw) // 2, 32), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- side panel
def _card(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), CARD, -1)


def _kpi(img, x, y, w, h, title, value, color=TEXT):
    _card(img, x, y, w, h)
    cv2.putText(img, title, (x + 10, y + 20), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, str(value), (x + 10, y + h - 14), FONT, 1.1, color, 2, cv2.LINE_AA)


def render_panel(s: DashboardState, height: int, width: int) -> np.ndarray:
    img = np.full((height, width, 3), BG, np.uint8)
    pad = 14
    y = 34
    cv2.putText(img, "ActivitySense AI", (pad, y), FONT, 0.85, TEXT, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(img, f"People counting & activity recognition", (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(img, f"t {fmt_time(s.t)}   frame {s.frame_idx}   {s.proc_fps:.1f} fps", (pad, y),
                FONT, 0.45, MUTED, 1, cv2.LINE_AA)

    # KPI grid 3 x 2
    y += 14
    cw, ch, gap = (width - 2 * pad - 2 * 8) // 3, 66, 8
    kpis = [("ENTERED", s.entered, ACCENT), ("EXITED", s.exited, TEXT), ("IN VIEW", s.in_view, TEXT),
            ("OUTSIDE", s.outside, TEXT), ("UNIQUE", s.unique_people, TEXT),
            ("FALLS", s.falls, ALERT if s.falls else TEXT)]
    for i, (t, v, c) in enumerate(kpis):
        _kpi(img, pad + (i % 3) * (cw + gap), y + (i // 3) * (ch + gap), cw, ch, t, v, c)
    y += 2 * (ch + gap) + 10

    # Activity bars
    cv2.putText(img, "ACTIVITY NOW", (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    y += 10
    total = max(1, sum(s.activity_now.values()))
    bar_x, bar_w = pad + 92, width - 2 * pad - 92 - 34
    for act in ACTIVITIES:
        n = s.activity_now.get(act, 0)
        y += 24
        cv2.putText(img, act.replace("_", " "), (pad, y), FONT, 0.47, TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, y - 13), (bar_x + bar_w, y + 1), CARD, -1)
        cv2.rectangle(img, (bar_x, y - 13), (bar_x + int(bar_w * n / total), y + 1), ACTIVITY_COLORS[act], -1)
        cv2.putText(img, str(n), (bar_x + bar_w + 8, y), FONT, 0.47, TEXT, 1, cv2.LINE_AA)

    # Confidence
    y += 30
    cv2.putText(img, f"AVG DETECTION CONFIDENCE  {s.avg_conf:.2f}", (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    y += 8
    cv2.rectangle(img, (pad, y), (width - pad, y + 8), CARD, -1)
    cv2.rectangle(img, (pad, y), (pad + int((width - 2 * pad) * s.avg_conf), y + 8), (120, 210, 120), -1)

    # Timeline
    y += 32
    cv2.putText(img, "TIMELINE", (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, "entries", (width - 170, y), FONT, 0.42, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "in view", (width - 90, y), FONT, 0.42, (200, 200, 120), 1, cv2.LINE_AA)
    y += 8
    th = 96
    _card(img, pad, y, width - 2 * pad, th)
    _sparkline(img, s.timeline_inview, pad, y, width - 2 * pad, th, (200, 200, 120))
    _sparkline(img, s.timeline_entries, pad, y, width - 2 * pad, th, ACCENT)
    y += th + 26

    # Recent events
    cv2.putText(img, "RECENT EVENTS", (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    for ts, text, col in s.recent[-max(0, (height - y - 10) // 22):][::-1]:
        y += 22
        if y > height - 8:
            break
        cv2.putText(img, ts, (pad, y), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, text, (pad + 56, y), FONT, 0.47, col, 1, cv2.LINE_AA)
    return img


def _sparkline(img, values, x, y, w, h, color):
    if len(values) < 2:
        return
    vals = np.asarray(values[-w:], float)
    vmax = max(vals.max(), 1.0)
    xs = np.linspace(x + 6, x + w - 6, len(vals))
    ys = y + h - 6 - (vals / vmax) * (h - 14)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

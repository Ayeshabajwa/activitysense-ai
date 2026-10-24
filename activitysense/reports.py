"""CSV / JSON reports written at the end of a run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import ACTIVITIES


def write_person_report(path: Path, persons, fps: float, frame_count: int):
    fields = ["person_id", "first_seen_s", "last_seen_s", "time_in_view_s", "entered", "entry_time_s",
              "entry_method", "exited", "exit_time_s", "dominant_activity",
              *[f"{a}_s" for a in ACTIVITIES], "falls", "avg_confidence", "tracker_ids", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in sorted(persons, key=lambda p: p.pid):
            acts = {a: p.activity_frames.get(a, 0) for a in ACTIVITIES}
            dominant = max(acts, key=acts.get) if any(acts.values()) else "unknown"
            counted = p.entered and p.entry_method != "duplicate-suppressed"
            if counted and not p.exited:
                status = "entered"
            elif p.exited:
                status = "exited"
            elif p.zone == "out":
                status = "outside"
            elif p.zone == "in":
                status = "inside (never crossed)"
            else:
                status = "unknown"
            if p.last_frame < frame_count - 1:
                status += " / left view"
            w.writerow({
                "person_id": p.pid,
                "first_seen_s": round(p.first_frame / fps, 2),
                "last_seen_s": round(p.last_frame / fps, 2),
                "time_in_view_s": round(p.conf_n / fps, 2),
                "entered": counted,
                "entry_time_s": round(p.entry_frame / fps, 2) if p.entry_frame is not None else "",
                "entry_method": p.entry_method or "",
                "exited": p.exited,
                "exit_time_s": round(p.exit_frame / fps, 2) if p.exit_frame is not None else "",
                "dominant_activity": dominant,
                **{f"{a}_s": round(acts[a] / fps, 2) for a in ACTIVITIES},
                "falls": p.falls,
                "avg_confidence": round(p.avg_conf, 3),
                "tracker_ids": " ".join(map(str, sorted(p.tracker_ids))),
                "status": status,
            })


def write_event_log(path: Path, events: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_s", "frame", "person_id", "event", "detail"])
        w.writeheader()
        for e in sorted(events, key=lambda e: e["frame"]):
            w.writerow(e)


def write_timeline(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(path: Path, summary: dict):
    path.write_text(json.dumps(summary, indent=2))

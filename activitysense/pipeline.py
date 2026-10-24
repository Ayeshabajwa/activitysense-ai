"""End-to-end pipeline: raw video in -> annotated dashboard video, heatmaps and reports out."""
from __future__ import annotations

import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import ACTIVITIES, Config
from .counter import LineCounter
from .dashboard import ALERT, ACCENT, TEXT, DashboardState, annotate_frame, fall_banner, fmt_time, render_panel
from .heatmap import Heatmap
from .persons import Detection, PersonRegistry
from .reports import write_event_log, write_person_report, write_summary, write_timeline


def parse_results(res) -> list[Detection]:
    boxes = res.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    kxy = kcf = None
    if res.keypoints is not None and res.keypoints.xy is not None and len(res.keypoints.xy):
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
    return [Detection(int(ids[i]), xyxy[i], float(confs[i]),
                      kxy[i] if kxy is not None else None,
                      kcf[i] if kcf is not None else None) for i in range(len(ids))]


def _to_h264(path: Path):
    """Re-encode OpenCV's mp4v output to H.264 so it plays in browsers / on LinkedIn (needs ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)


def run(source: str, out_dir: str | Path, cfg: Config) -> dict:
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    if cfg.line is None:   # default: horizontal line across the middle
        cfg.line = (0.05 * W, 0.5 * H, 0.95 * W, 0.5 * H)
    a, b = (cfg.line[0], cfg.line[1]), (cfg.line[2], cfg.line[3])

    model = YOLO(cfg.model)
    registry = PersonRegistry(cfg, fps)
    counter = LineCounter(cfg, a, b, fps)
    heat = Heatmap((H, W), cfg.heatmap_radius)

    scale = H / cfg.display_height
    disp_w = int(round(W / scale))
    out_w, out_h = disp_w + cfg.panel_width, cfg.display_height
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    state = DashboardState()
    events: list[dict] = []
    timeline_rows: list[dict] = []
    first_frame = None
    last_frame_img = None
    n_guard_logged = n_dup_logged = n_revoked_logged = 0
    canvas = None
    t0 = time.time()
    frame_idx = -1

    def log(fi, pid, kind, detail, color=TEXT):
        events.append({"time_s": round(fi / fps, 2), "frame": fi, "person_id": pid, "event": kind, "detail": detail})
        state.recent.append((fmt_time(fi / fps), f"#{pid} {kind.replace('_', ' ')}" + (f" ({detail})" if detail and kind != "activity" else ""), color))
        state.recent = state.recent[-12:]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if cfg.max_frames and frame_idx >= cfg.max_frames:
            break
        if first_frame is None:
            first_frame = frame.copy()

        res = model.track(frame, persist=True, tracker=cfg.tracker, conf=cfg.conf, iou=cfg.iou,
                          imgsz=cfg.imgsz, classes=[0], device=cfg.device, verbose=False)[0]
        dets = parse_results(res)
        visible = registry.update(frame_idx, frame, dets)
        lost = registry.lost_persons(frame_idx)

        for ev in counter.update(visible, lost, frame_idx):
            log(frame_idx, ev.pid, ev.kind, ev.method, ACCENT if ev.kind == "entry" else TEXT)
        for fi, pid in counter.suppressed[n_dup_logged:]:
            log(fi, pid, "duplicate_blocked", "same person, new ID", (150, 150, 150))
        n_dup_logged = len(counter.suppressed)
        for fi, pid in counter.revoked[n_revoked_logged:]:
            log(fi, pid, "entry_revoked", "reappeared outside", (150, 150, 150))
        n_revoked_logged = len(counter.revoked)
        for fi, msg in registry.guard_events[n_guard_logged:]:
            events.append({"time_s": round(fi / fps, 2), "frame": fi, "person_id": "", "event": "guard", "detail": msg})
        n_guard_logged = len(registry.guard_events)

        for p in visible:
            heat.add(*p.anchor, p.activity.label)
            if p.just_fell:
                log(frame_idx, p.pid, "FALL", "upright -> lying", ALERT)

        # ---- dashboard state
        act_now = Counter(p.activity.label for p in visible)
        fallen_now = sum(1 for p in visible if p.activity.label == "lying" and p.activity.fallen)
        state.t, state.frame_idx = frame_idx / fps, frame_idx
        state.entered, state.exited = counter.entries, counter.exits
        state.in_view = len(visible)
        state.outside = sum(1 for p in visible if p.zone == "out" and not p.entered)
        state.falls = sum(p.falls for p in registry.persons.values())
        state.fallen_now = fallen_now
        state.unique_people = len(registry.persons)
        state.activity_now = dict(act_now)
        state.avg_conf = float(np.mean([d.conf for d in dets])) if dets else state.avg_conf
        state.timeline_entries.append(counter.entries)
        state.timeline_inview.append(len(visible))
        state.proc_fps = (frame_idx + 1) / max(time.time() - t0, 1e-6)

        if frame_idx % max(1, int(round(fps))) == 0:
            timeline_rows.append({"time_s": round(frame_idx / fps, 2), "in_view": len(visible),
                                  "entered_total": counter.entries, "exited_total": counter.exits,
                                  **{a: act_now.get(a, 0) for a in ACTIVITIES}})

        # ---- render
        vis = heat.overlay(frame) if cfg.heatmap_overlay else frame.copy()
        vis = annotate_frame(vis, visible, (a, b), cfg.inside_side, cfg, max(scale, 0.6))
        vis = cv2.resize(vis, (disp_w, out_h), interpolation=cv2.INTER_AREA)
        vis = fall_banner(vis, fallen_now, frame_idx)
        canvas = np.hstack([vis, render_panel(state, out_h, cfg.panel_width)])
        writer.write(canvas)
        last_frame_img = frame

        if cfg.show:
            cv2.imshow("ActivitySense AI", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        if frame_idx % 50 == 0:
            pct = f"{100 * frame_idx / total:5.1f}%" if total else ""
            print(f"[{pct}] frame {frame_idx}  in view {len(visible)}  entered {counter.entries}  "
                  f"exited {counter.exits}  {state.proc_fps:.1f} fps", flush=True)

    cap.release()
    writer.release()
    _to_h264(out_dir / "annotated.mp4")
    if cfg.show:
        cv2.destroyAllWindows()
    n_frames = frame_idx + 1

    # ---- final artefacts
    bg = last_frame_img if last_frame_img is not None else np.zeros((H, W, 3), np.uint8)
    heat_img = heat.overlay(bg)
    cv2.line(heat_img, tuple(map(int, a)), tuple(map(int, b)), ACCENT, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "heatmap.png"), heat_img)
    cv2.imwrite(str(out_dir / "activity_map.png"), heat.activity_map(first_frame if first_frame is not None else bg))
    if canvas is not None:
        cv2.imwrite(str(out_dir / "dashboard_last_frame.png"), canvas)

    persons = list(registry.persons.values())
    write_person_report(out_dir / "persons.csv", persons, fps, n_frames)
    write_event_log(out_dir / "events.csv", events)
    write_timeline(out_dir / "activity_timeline.csv", timeline_rows)

    total_act = Counter()
    for p in persons:
        total_act.update(p.activity_frames)
    summary = {
        "source": str(source),
        "frames": n_frames,
        "fps": round(fps, 2),
        "duration_s": round(n_frames / fps, 2),
        "processing_fps": round(state.proc_fps, 2),
        "entry_line": {"a": list(map(float, a)), "b": list(map(float, b)), "inside_side": cfg.inside_side},
        "people_entered": counter.entries,
        "people_exited": counter.exits,
        "entries_by_method": dict(Counter(e.method for e in counter.events if e.kind == "entry")),
        "unique_people": len(persons),
        "falls_detected": sum(p.falls for p in persons),
        "activity_person_seconds": {k: round(v / fps, 2) for k, v in total_act.items()},
        "guards": {
            "relinked_ids": registry.relinks,
            "duplicate_boxes_suppressed": registry.dup_suppressed,
            "id_switches_blocked": registry.id_switch_blocked,
            "duplicate_entries_blocked": len(counter.suppressed),
            "doorway_entries_revoked": len(counter.revoked),
        },
        "outputs": {k: str(out_dir / v) for k, v in {
            "video": "annotated.mp4", "heatmap": "heatmap.png", "activity_map": "activity_map.png",
            "persons": "persons.csv", "events": "events.csv", "timeline": "activity_timeline.csv"}.items()},
    }
    write_summary(out_dir / "summary.json", summary)
    return summary

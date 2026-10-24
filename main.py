"""ActivitySense AI - people counting, tracking and activity recognition.

Examples
--------
  # 1) draw the entry line once for a camera (needs a display)
  python main.py draw-line --source videos/lobby.mp4 --out lines/lobby.json

  # 2) run the full pipeline
  python main.py run --source videos/lobby.mp4 --line-file lines/lobby.json

  # or give the line directly (pixels, or 0-1 fractions of the frame) + entry direction
  python main.py run --source videos/lobby.mp4 --line 0.1,0.55,0.9,0.55 --in-direction down
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from activitysense.config import Config
from activitysense.geometry import side_for_direction


def parse_line(text: str, w: int, h: int):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise SystemExit("--line needs 4 comma-separated values: x1,y1,x2,y2")
    if all(0 <= v <= 1 for v in vals):   # fractions of the frame
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    return tuple(vals)


def frame_size(source: str):
    cap = cv2.VideoCapture(source)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w:
        raise SystemExit(f"Cannot open video: {source}")
    return w, h


def main():
    ap = argparse.ArgumentParser(description="ActivitySense AI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw-line", help="click-to-draw the entry line on a frame")
    d.add_argument("--source", required=True)
    d.add_argument("--out", default="line.json")
    d.add_argument("--frame", type=int, default=0)

    r = sub.add_parser("run", help="process a video")
    r.add_argument("--source", required=True)
    r.add_argument("--output", default=None, help="output folder (default runs/<video name>)")
    r.add_argument("--model", default=Config.model, help="YOLO26 pose weights, e.g. yolo26s-pose.pt")
    r.add_argument("--line", default=None, help="x1,y1,x2,y2 in pixels or 0-1 fractions")
    r.add_argument("--line-file", default=None, help="JSON saved by draw-line")
    r.add_argument("--in-direction", choices=["down", "up", "left", "right"], default="down",
                   help="walking direction that counts as entering (ignored with --line-file)")
    r.add_argument("--conf", type=float, default=Config.conf)
    r.add_argument("--imgsz", type=int, default=Config.imgsz)
    r.add_argument("--device", default=None, help="cpu, 0, mps ...")
    r.add_argument("--walk-speed", type=float, default=Config.walk_speed)
    r.add_argument("--run-speed", type=float, default=Config.run_speed)
    r.add_argument("--no-doorway", action="store_true", help="disable doorway (vanish-at-line) counting")
    r.add_argument("--heatmap-overlay", action="store_true", help="draw the live heatmap on the video")
    r.add_argument("--show", action="store_true", help="show a live preview window")
    r.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "draw-line":
        from activitysense.line_tool import draw_line
        draw_line(args.source, args.out, args.frame)
        return

    w, h = frame_size(args.source)
    cfg = Config(model=args.model, conf=args.conf, imgsz=args.imgsz, device=args.device,
                 walk_speed=args.walk_speed, run_speed=args.run_speed, doorway=not args.no_doorway,
                 heatmap_overlay=args.heatmap_overlay, show=args.show, max_frames=args.max_frames)
    if args.line_file:
        data = json.loads(Path(args.line_file).read_text())
        cfg.line = tuple(float(v) for v in data["line"])
        cfg.inside_side = int(data["inside_side"])
    else:
        cfg.line = parse_line(args.line, w, h) if args.line else (0.05 * w, 0.5 * h, 0.95 * w, 0.5 * h)
        cfg.inside_side = side_for_direction(cfg.line[:2], cfg.line[2:], args.in_direction)

    from activitysense.pipeline import run
    out = args.output or str(Path("runs") / Path(args.source).stem)
    summary = run(args.source, out, cfg)

    print("\n=== ActivitySense AI summary ===")
    print(f"People entered : {summary['people_entered']}  {summary['entries_by_method']}")
    print(f"People exited  : {summary['people_exited']}")
    print(f"Unique people  : {summary['unique_people']}")
    print(f"Falls detected : {summary['falls_detected']}")
    print(f"Activity (person-seconds): {summary['activity_person_seconds']}")
    print(f"Guards: {summary['guards']}")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()

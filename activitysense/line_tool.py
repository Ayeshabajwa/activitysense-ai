"""Click-to-draw entry line tool.

Click 2 points for the line, then a 3rd point on the INSIDE side.
Keys: r = reset, s / Enter = save, q / Esc = quit without saving.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .geometry import side_for_point


def draw_line(source: str, out_path: str | Path, frame_no: int = 0) -> dict | None:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read a frame from {source}")

    pts: list[tuple[int, int]] = []
    win = "Draw entry line: 2 clicks = line, 3rd click = INSIDE side | s=save r=reset q=quit"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        img = frame.copy()
        for p in pts[:2]:
            cv2.circle(img, p, 6, (70, 190, 255), -1)
        if len(pts) >= 2:
            cv2.line(img, pts[0], pts[1], (70, 190, 255), 3, cv2.LINE_AA)
        if len(pts) == 3:
            cv2.circle(img, pts[2], 8, (90, 220, 90), -1)
            cv2.putText(img, "INSIDE", (pts[2][0] + 10, pts[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 220, 90), 2)
        hint = ["click line start", "click line end", "click a point on the INSIDE side", "press s to save"][len(pts)]
        cv2.putText(img, hint, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k in (ord("s"), 13) and len(pts) == 3:
            (x1, y1), (x2, y2) = pts[:2]
            result = {"line": [x1, y1, x2, y2], "inside_side": side_for_point(pts[0], pts[1], pts[2]),
                      "frame_size": [frame.shape[1], frame.shape[0]]}
            Path(out_path).write_text(json.dumps(result, indent=2))
            print(f"Saved entry line to {out_path}: {result}")
            break
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return result

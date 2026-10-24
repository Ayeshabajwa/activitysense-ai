"""Central configuration for the ActivitySense AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Activity labels, in display order.
ACTIVITIES = ("standing", "walking", "running", "sitting", "lying", "hands_up")

# BGR colours used everywhere (boxes, bars, activity map).
ACTIVITY_COLORS = {
    "standing": (235, 180, 60),   # blue
    "walking": (90, 200, 90),     # green
    "running": (40, 160, 255),    # orange
    "sitting": (220, 120, 200),   # purple
    "lying": (60, 60, 235),       # red
    "hands_up": (40, 220, 240),   # yellow
    "unknown": (160, 160, 160),
}


@dataclass
class Config:
    # --- detection / tracking -------------------------------------------------
    model: str = "yolo26n-pose.pt"
    tracker: str = str(PACKAGE_ROOT / "trackers" / "botsort_people.yaml")
    conf: float = 0.25
    iou: float = 0.50
    imgsz: int = 1280
    device: str | None = None
    kpt_conf: float = 0.35          # min keypoint confidence to trust a joint

    # --- entry line -----------------------------------------------------------
    line: tuple[float, float, float, float] | None = None   # x1, y1, x2, y2 (pixels)
    inside_side: int = 1            # +1 / -1: which side of the line is "inside"
    line_margin_px: float = 4.0     # minimum hysteresis band so jitter on the line never counts
    line_margin_factor: float = 0.12  # ...and at least this fraction of body height
    line_extent_margin: float = 0.08  # crossings may land this fraction beyond the line ends

    # --- identity / guards ----------------------------------------------------
    relink_max_frames: int = 60     # how long a lost person can be relinked to a new tracker ID
    relink_dist_factor: float = 1.5  # max relink distance, in body heights
    relink_min_score: float = 0.45
    dup_iou: float = 0.75           # duplicate-box guard
    max_jump_factor: float = 1.2    # ID-switch guard: max move per frame, in body heights
    dedupe_seconds: float = 1.0     # entry dedupe window for "same person, new ID" crossings

    # --- doorway logic --------------------------------------------------------
    doorway: bool = True
    doorway_lost_frames: int = 8
    doorway_band_factor: float = 0.6  # how close (in body heights) to the line a vanish must be

    # --- activity recognition -------------------------------------------------
    activity_window: int = 9        # frames of majority-vote smoothing
    walk_speed: float = 0.25        # body heights / second
    run_speed: float = 1.8
    fall_window_s: float = 1.2      # upright -> lying within this time = fall

    # --- output ---------------------------------------------------------------
    display_height: int = 720
    panel_width: int = 440
    heatmap_radius: int = 22
    heatmap_overlay: bool = False
    trail_len: int = 40
    show: bool = False
    max_frames: int | None = None

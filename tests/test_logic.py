"""Unit tests for the counting, guard and activity logic (no model / video needed)."""
import numpy as np
import pytest

from activitysense.activity import ActivityState, classify_frame, pose_features
from activitysense.config import Config
from activitysense.counter import LineCounter
from activitysense.geometry import side_for_direction, signed_distance
from activitysense.persons import Detection, PersonRegistry

FPS = 30.0
BLANK = np.zeros((480, 640, 3), np.uint8)


def make_cfg(**kw):
    cfg = Config(line=(0, 240, 640, 240), **kw)
    cfg.inside_side = side_for_direction((0, 240), (640, 240), "down")
    return cfg


def det(tid, cx, bottom, h=100, w=40, conf=0.9):
    return Detection(tid, np.array([cx - w / 2, bottom - h, cx + w / 2, bottom], float), conf, None, None)


def walk(registry, counter, tid, xs_ys, start=0):
    events = []
    for i, (x, y) in enumerate(xs_ys):
        f = start + i
        vis = registry.update(f, BLANK, [det(tid, x, y)])
        events += counter.update(vis, registry.lost_persons(f), f)
    return events


def setup(**kw):
    cfg = make_cfg(**kw)
    reg = PersonRegistry(cfg, FPS)
    return cfg, reg, LineCounter(cfg, cfg.line[:2], cfg.line[2:], FPS)


# ---------------------------------------------------------------- geometry
def test_side_for_direction():
    a, b = (0, 240), (640, 240)
    s = side_for_direction(a, b, "down")
    assert signed_distance((320, 300), a, b) * s > 0
    with pytest.raises(ValueError):
        side_for_direction(a, b, "left")


# ---------------------------------------------------------------- counting
def test_entry_counted_once():
    _, reg, ctr = setup()
    path = [(320, y) for y in range(150, 330, 4)]
    walk(reg, ctr, 1, path)
    assert ctr.entries == 1
    # walk back out and in again: exit counted once, entry still once per person
    walk(reg, ctr, 1, [(320, y) for y in range(330, 150, -4)] + path, start=len(path))
    assert ctr.entries == 1 and ctr.exits == 1


def test_jitter_on_line_does_not_count():
    _, reg, ctr = setup()
    path = [(320, 240 + (5 if i % 2 else -5)) for i in range(60)]
    walk(reg, ctr, 1, [(320, 200)] * 5 + path)
    assert ctr.entries == 0


def test_crossing_outside_segment_ignored():
    cfg, reg, ctr = setup()
    ctr.a, ctr.b = (0, 240), (200, 240)       # short line on the left only
    walk(reg, ctr, 1, [(500, y) for y in range(150, 330, 4)])
    assert ctr.entries == 0


def test_relink_keeps_identity_and_no_double_count():
    _, reg, ctr = setup()
    walk(reg, ctr, 1, [(320, y) for y in range(150, 260, 4)])     # enters
    n = len(range(150, 260, 4))
    # tracker loses the person and gives them a brand-new ID a few frames later
    walk(reg, ctr, 2, [(320, y) for y in range(272, 330, 4)], start=n + 3)
    assert len(reg.persons) == 1 and reg.relinks == 1
    assert ctr.entries == 1


def test_duplicate_boxes_suppressed():
    cfg, reg, _ = setup()
    vis = reg.update(0, BLANK, [det(1, 320, 200), det(2, 322, 201, conf=0.6)])
    assert len(vis) == 1 and reg.dup_suppressed == 1


def test_doorway_entry_and_rollback():
    _, reg, ctr = setup()
    path = [(320, y) for y in range(150, 232, 3)]                   # stops just before the line
    walk(reg, ctr, 1, path)
    f = len(path)
    for k in range(10):                                             # vanishes into the doorway
        ctr.update([], reg.lost_persons(f + k), f + k)
    assert ctr.entries == 1 and ctr.events[-1].method == "doorway"
    # ...but if it shows up outside again, the doorway entry is rolled back
    walk(reg, ctr, 1, [(320, 220), (320, 215), (320, 210)], start=f + 10)
    assert ctr.entries == 0 and len(ctr.revoked) == 1


def test_id_switch_guard():
    _, reg, _ = setup()
    reg.update(0, BLANK, [det(1, 100, 300)])
    reg.update(1, BLANK, [det(1, 101, 302)])
    reg.update(2, BLANK, [det(1, 560, 300)])      # same tracker ID teleports across the frame
    assert reg.id_switch_blocked == 1 and len(reg.persons) == 2


# ---------------------------------------------------------------- activity
def skeleton(kind):
    """Synthetic COCO-17 keypoints for a few postures (person ~ 200 px tall)."""
    kp = np.zeros((17, 2))
    if kind == "standing":
        pts = {0: (50, 10), 5: (40, 40), 6: (60, 40), 9: (35, 100), 10: (65, 100),
               11: (42, 100), 12: (58, 100), 13: (42, 150), 14: (58, 150), 15: (42, 200), 16: (58, 200)}
    elif kind == "hands_up":
        pts = {0: (50, 30), 5: (40, 60), 6: (60, 60), 9: (35, 0), 10: (65, 0),
               11: (42, 120), 12: (58, 120), 13: (42, 165), 14: (58, 165), 15: (42, 210), 16: (58, 210)}
    elif kind == "sitting":
        pts = {0: (50, 10), 5: (40, 40), 6: (60, 40), 11: (42, 100), 12: (58, 100),
               13: (90, 105), 14: (95, 105), 15: (90, 150), 16: (95, 150)}
    elif kind == "lying":
        pts = {0: (10, 50), 5: (40, 40), 6: (40, 60), 11: (100, 42), 12: (100, 58),
               13: (150, 42), 14: (150, 58), 15: (200, 42), 16: (200, 58)}
    conf = np.zeros(17)
    for i, (x, y) in pts.items():
        kp[i] = (x, y)
        conf[i] = 0.9
    xs, ys = kp[conf > 0][:, 0], kp[conf > 0][:, 1]
    box = (xs.min(), ys.min(), xs.max(), ys.max())
    return kp, conf, box


@pytest.mark.parametrize("kind,speed,expected", [
    ("standing", 0.0, "standing"),
    ("standing", 0.8, "walking"),
    ("standing", 2.5, "running"),
    ("hands_up", 0.0, "hands_up"),
    ("sitting", 0.0, "sitting"),
    ("lying", 0.0, "lying"),
])
def test_classify(kind, speed, expected):
    kp, conf, box = skeleton(kind)
    f = pose_features(kp, conf, box, 0.35)
    assert classify_frame(f, speed, 0.25, 1.8) == expected


def test_fall_detected_once():
    st = ActivityState(window=5, fall_window_frames=36)
    falls = 0
    for i in range(20):
        _, fall = st.update("walking", i)
        falls += fall
    for i in range(20, 60):
        label, fall = st.update("lying", i)
        falls += fall
    assert label == "lying" and falls == 1


def test_slowly_lying_down_is_not_a_fall():
    st = ActivityState(window=5, fall_window_frames=36)
    for i in range(10):
        st.update("standing", i)
    for i in range(10, 100):
        st.update("sitting", i)          # sits for 3 s first
    falls = sum(st.update("lying", i)[1] for i in range(100, 130))
    assert falls == 0

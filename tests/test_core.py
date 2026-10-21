import numpy as np
from screensight.core import Detection, Letterbox, parse_yolo_people, TargetTracker, MovementFilter


def test_yolo_decoder_with_transpose_and_clipping():
    raw = np.zeros((1, 84, 4), dtype=np.float32)
    raw[0, :4, 0] = [80, 80, 100, 160]
    raw[0, 4, 0] = .91
    raw[0, :4, 1] = [80, 80, 100, 160]
    raw[0, 4, 1] = .88
    raw[0, :4, 2] = [300, 300, 20, 30]
    raw[0, 4, 2] = .12
    d = parse_yolo_people(raw, Letterbox(1, 0, 0, 640, 640), .40)
    assert len(d) == 1, d
    assert d[0].x1 == 30
    assert d[0].score == .91 or abs(d[0].score-.91) < 1e-5


def test_nms_exported_six_column():
    data = np.array([[10, 20, 60, 140, .9, 0], [400, 20, 450, 100, .8, 2]], dtype=np.float32)
    d = parse_yolo_people(data, Letterbox(1, 0, 0, 500, 200), .40)
    assert len(d) == 1 and d[0].x1 == 10


def test_letterbox_reverse_projection():
    raw = np.zeros((1, 84, 3), dtype=np.float32)
    raw[0, 0:4, 0] = (220, 220, 100, 120)
    raw[0, 4, 0] = .8
    d = parse_yolo_people(raw, Letterbox(2, 20, 10, 250, 240), .4)
    assert len(d) == 1
    assert abs(d[0].x1 - 75) < 1
    assert abs(d[0].y1 - 75) < 1


def test_nearest_person_and_reset():
    tracker = TargetTracker()
    near = Detection(220, 190, 300, 320, .88)
    far = Detection(20, 40, 90, 200, .94)
    chosen = tracker.choose([far, near], 480, 480, .22, 10.0)
    assert chosen is not None and chosen[0] == near
    tracker.reset()
    assert tracker.last is None


def test_motion_bounded_no_nan():
    motion = MovementFilter()
    for _ in range(10):
        dx, dy = motion.compute(1000, -1000, .5, 19)
        assert abs(dx) <= 19 and abs(dy) <= 19
    assert motion.compute(float('nan'), 0, .4, 20) == (0, 0)

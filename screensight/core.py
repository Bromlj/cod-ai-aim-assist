"""Pure vision geometry and target-selection code. No OS input or networking."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def aim_point(self, relative_height: float) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y1 + self.height * relative_height)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    source_w: int
    source_h: int


def parse_yolo_people(raw: np.ndarray, transform: Letterbox,
                      threshold: float = 0.45, limit: int = 12) -> list[Detection]:
    """Read common YOLOv8/YOLO11 ONNX output [1,84,N] (COCO person class 0).

    Also supports [N,6] rows of xyxy, confidence, class for exported NMS graphs.
    Returns frame-relative detections after lightweight class-agnostic NMS.
    """
    import cv2
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, received {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape {arr.shape}")
    # Most YOLO exports are [84,8400] (channels first).
    if (arr.shape[0] in (84, 85) and arr.shape[1] not in (84, 85)) or (arr.shape[0] in (5, 6) and arr.shape[1] > arr.shape[0]):
        arr = arr.T
    if arr.shape[1] < 5:
        raise ValueError(f"Expected at least 5 channels, received {arr.shape}")

    if arr.shape[1] == 6:  # exported with NMS: x1,y1,x2,y2,score,class
        rows = arr[(arr[:, 5].astype(int) == 0) & (arr[:, 4] >= threshold)]
        boxes = rows[:, :4].astype(np.float32)
        scores = rows[:, 4].astype(np.float32)
    else:
        if arr.shape[1] == 85:  # YOLOv5 layout with objectness
            scores_all = arr[:, 4] * arr[:, 5]
        else:  # YOLOv8/YOLO11 layout: class 0 is at index 4
            scores_all = arr[:, 4]
        good = np.flatnonzero(scores_all >= threshold)
        if good.size == 0:
            return []
        # Pre-limit candidates to protect the UI from huge boxes arrays.
        if good.size > 240:
            good = good[np.argpartition(scores_all[good], -240)[-240:]]
        rows = arr[good, :4].astype(np.float32)
        scores = scores_all[good].astype(np.float32)
        boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                                 rows[:, 1] - rows[:, 3] / 2,
                                 rows[:, 0] + rows[:, 2] / 2,
                                 rows[:, 1] + rows[:, 3] / 2))

    if boxes.size == 0:
        return []
    if transform.scale <= 0:
        raise ValueError("Invalid letterbox scale")
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    boxes, scores = boxes[finite], scores[finite]
    if not len(boxes):
        return []
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - transform.pad_x) / transform.scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - transform.pad_y) / transform.scale
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, transform.source_w)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, transform.source_h)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths >= 6) & (heights >= 10) & np.isfinite(scores)
    boxes, scores, widths, heights = boxes[valid], scores[valid], widths[valid], heights[valid]
    if not scores.size:
        return []
    nms_boxes = [[int(b[0]), int(b[1]), int(w), int(h)]
                 for b, w, h in zip(boxes, widths, heights)]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(),
                            score_threshold=float(threshold), nms_threshold=0.45)
    if keep is None or len(keep) == 0:
        return []
    result = []
    for k in np.asarray(keep).reshape(-1)[:limit]:
        x1, y1, x2, y2 = boxes[int(k)]
        result.append(Detection(float(x1), float(y1), float(x2), float(y2), float(scores[int(k)])))
    return result


def box_iou(a: Detection, b: Detection) -> float:
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


class TargetTracker:
    """Conservative association: maintain a visible lock; never move on a lost track.

    Identity is a geometric estimate, not person recognition or team recognition.
    A short loss grace prevents immediately switching to another visible person.
    """
    def __init__(self, loss_grace: float = 0.14, confirmation_frames: int = 2) -> None:
        self.loss_grace = loss_grace
        self.confirmation_frames = confirmation_frames
        self.acquisitions = 0
        self.reset()

    def reset(self) -> None:
        self.last: tuple[float, float] | None = None
        self.raw: tuple[float, float] | None = None
        self.box: Detection | None = None
        self.velocity = (0.0, 0.0)
        self.at = 0.0
        self.hits = 0
        self.confirmed = False
        self.track_id = 0
        self.state = 'SEARCHING'

    def choose(self, boxes: list[Detection], width: int, height: int,
               aim_height: float, now: float, predict_seconds: float = 0.0
               ) -> tuple[Detection, tuple[float, float]] | None:
        centre = (width / 2.0, height / 2.0)
        radius = min(width, height) * .48
        candidates = [b for b in boxes
                      if all(math.isfinite(v) for v in (b.x1, b.y1, b.x2, b.y2, b.score))
                      and b.width > 0 and b.height > 0
                      and math.dist(b.aim_point(aim_height), centre) <= radius]
        elapsed = now - self.at
        previous = self.box
        chosen = None
        if previous is not None and 0 <= elapsed <= self.loss_grace:
            gate = max(18.0, min(90.0, previous.height * .35))
            old_centre = previous.aim_point(.5)
            eligible = [b for b in candidates
                        if .4 <= b.height / max(1, previous.height) <= 2.5
                        and (box_iou(previous, b) >= .12 or
                             math.dist(old_centre, b.aim_point(.5)) <= gate)]
            if eligible:
                chosen = min(eligible, key=lambda b:
                    math.dist(old_centre, b.aim_point(.5)) / gate - box_iou(previous, b))
            else:
                self.hits = 0
                self.confirmed = False
                self.state = 'LOST - INPUT PAUSED'
                return None
        if chosen is None:
            self.reset()
            if not candidates:
                return None
            chosen = min(candidates, key=lambda b:
                         math.dist(b.aim_point(aim_height), centre) + (1-b.score) * 35)
            self.acquisitions += 1
            self.track_id = self.acquisitions
        point = chosen.aim_point(aim_height)
        if self.raw is not None and self.last is not None and 0.001 < elapsed < .15:
            # Time-based smoothing, with a faster response for larger displacements.
            distance = math.dist(point, self.last)
            tau = .008 if distance > 20 else .022
            alpha = 1.0 - math.exp(-elapsed / tau)
            filtered = tuple(a + alpha * (b-a) for a, b in zip(self.last, point))
            velocity_alpha = 1.0 - math.exp(-elapsed / .065)
            self.velocity = tuple(max(-1800., min(1800., old + velocity_alpha *
                                   ((new-prior)/elapsed - old)))
                                  for old, new, prior in zip(self.velocity, point, self.raw))
        else:
            filtered = point
            self.velocity = (0.0, 0.0)
        self.box, self.raw, self.last, self.at = chosen, point, filtered, now
        self.hits += 1
        self.confirmed = self.hits >= self.confirmation_frames
        self.state = 'LOCKED' if self.confirmed else 'CONFIRMING'
        # Optional short lead. Screen-space motion includes camera motion; default OFF.
        lead = max(0., min(.035, predict_seconds))
        cap = min(24., chosen.width * .2)
        predicted = tuple(p + max(-cap, min(cap, v * lead))
                          for p, v in zip(filtered, self.velocity))
        predicted = (max(0., min(float(width), predicted[0])),
                     max(0., min(float(height), predicted[1])))
        return chosen, predicted


class MovementFilter:
    """Time-normalized proportional movement, bounded rate, no stale momentum.

    max_step retains its v0.1 meaning at 60 Hz; the rate limit scales with dt.
    Screen pixels are not game angles: this is not sensitivity calibration.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._carry = [0.0, 0.0]
        self._sign = [0, 0]

    def compute(self, offset_x: float, offset_y: float, strength: float,
                max_step: int, deadzone: float = 2.0,
                dt: float = 1/60) -> tuple[int, int]:
        if (not all(math.isfinite(v) for v in
                    (offset_x, offset_y, strength, max_step, deadzone, dt))
                or dt <= 0 or dt > .12 or max_step <= 0):
            self.reset()
            return 0, 0
        dt = min(.05, dt)  # Do not catch up a delayed frame with a large jump.
        strength = max(.01, min(.85, strength))
        response = 1.0 - math.exp(math.log1p(-strength) * 60.0 * dt)
        cap = max_step * 60.0 * dt
        result = []
        for axis, error in enumerate((offset_x, offset_y)):
            sign = 1 if error > 0 else -1
            if abs(error) <= max(0, deadzone):
                self._carry[axis] = 0.0
                self._sign[axis] = 0
                result.append(0)
                continue
            if sign != self._sign[axis]:
                self._carry[axis] = 0.0
            self._sign[axis] = sign
            self._carry[axis] += max(-cap, min(cap, error * response))
            step = int(self._carry[axis])
            self._carry[axis] -= step
            result.append(step)
        return result[0], result[1]

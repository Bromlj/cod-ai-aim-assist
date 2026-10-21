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


class TargetTracker:
    """Small stickiness/velocity estimator, independent from mouse control."""
    def __init__(self) -> None:
        self.last: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.at = 0.0

    def reset(self) -> None:
        self.last = None
        self.velocity = (0.0, 0.0)
        self.at = 0.0

    def choose(self, boxes: list[Detection], width: int, height: int,
               aim_height: float, now: float, predict_seconds: float = 0.025
               ) -> tuple[Detection, tuple[float, float]] | None:
        if not boxes:
            if now - self.at > 0.35:
                self.reset()
            return None
        centre = (width / 2, height / 2)
        def rank(d: Detection) -> float:
            pt = d.aim_point(aim_height)
            dist = math.dist(pt, centre)
            sticky = 0.0
            if self.last and now - self.at < 0.35:
                last_dist = math.dist(pt, self.last)
                if last_dist < min(90, width * .25):
                    sticky = 45.0
            return dist + (1.0 - d.score) * 65.0 - sticky
        chosen = min(boxes, key=rank)
        point = chosen.aim_point(aim_height)
        elapsed = now - self.at
        if self.last is not None and 0.005 < elapsed < 0.2 and math.dist(point, self.last) < width * .3:
            vx = (point[0] - self.last[0]) / elapsed
            vy = (point[1] - self.last[1]) / elapsed
            self.velocity = (.55 * self.velocity[0] + .45 * vx,
                             .55 * self.velocity[1] + .45 * vy)
        else:
            self.velocity = (0.0, 0.0)
        self.last, self.at = point, now
        prediction = (max(0.0, min(float(width), point[0] + self.velocity[0] * predict_seconds)),
                      max(0.0, min(float(height), point[1] + self.velocity[1] * predict_seconds)))
        return chosen, prediction


class MovementFilter:
    """Bounded movement with smoothing, dead zone, and fractional carry."""
    def __init__(self) -> None:
        self._dx = self._dy = self._carry_x = self._carry_y = 0.0

    def reset(self) -> None:
        self._dx = self._dy = self._carry_x = self._carry_y = 0.0

    def compute(self, offset_x: float, offset_y: float, strength: float,
                max_step: int, deadzone: float = 2.0) -> tuple[int, int]:
        if not all(math.isfinite(x) for x in (offset_x, offset_y, strength)):
            return 0, 0
        if abs(offset_x) <= deadzone:
            offset_x = 0.0
        if abs(offset_y) <= deadzone:
            offset_y = 0.0
        self._dx = self._dx * .38 + offset_x * strength * .62
        self._dy = self._dy * .38 + offset_y * strength * .62
        self._carry_x += max(-max_step, min(max_step, self._dx))
        self._carry_y += max(-max_step, min(max_step, self._dy))
        dx, dy = int(self._carry_x), int(self._carry_y)
        self._carry_x -= dx
        self._carry_y -= dy
        return max(-max_step, min(max_step, dx)), max(-max_step, min(max_step, dy))

"""
inference.py
YOLO inference wrapper optimized for real-time use (best version).

Supports .pt and .engine (TensorRT).
Handles variable frame sizes from capture card / streams.
"""

import os
import time
from typing import List, Tuple, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is required. pip install ultralytics")

from config import CONFIG, get_model_path


class YOLOInference:
    def __init__(self):
        self.model_path = get_model_path()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Place best.engine or best.pt / cod_yolo11n.pt in models/"
            )

        print(f"[Inference] Loading model: {self.model_path}")
        load_start = time.perf_counter()

        self.model = YOLO(self.model_path)

        # Warmup with current ROI size
        dummy = np.zeros((CONFIG.ROI_HEIGHT, CONFIG.ROI_WIDTH, 3), dtype=np.uint8)
        device = 0 if self._has_cuda() else "cpu"
        _ = self.model(
            dummy,
            imgsz=CONFIG.IMGSZ,
            conf=CONFIG.CONF,
            iou=CONFIG.IOU,
            max_det=CONFIG.MAX_DET,
            verbose=False,
            device=device,
        )

        load_time = time.perf_counter() - load_start
        print(f"[Inference] Model loaded in {load_time*1000:.1f} ms")

        try:
            self.names = self.model.names
        except Exception:
            self.names = {CONFIG.ENEMY_CLASS: "enemy"}

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def infer(self, frame_rgb: np.ndarray) -> List[dict]:
        """
        Run detection.
        Returns list of dicts with local coordinates (to the passed frame).
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return []

        results = self.model(
            frame_rgb,
            imgsz=CONFIG.IMGSZ,
            conf=CONFIG.CONF,
            iou=CONFIG.IOU,
            max_det=CONFIG.MAX_DET,
            verbose=False,
            device=0 if self._has_cuda() else "cpu",
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cls = int(clss[i])
            conf = float(confs[i])

            if CONFIG.ENEMY_CLASS is not None and cls != CONFIG.ENEMY_CLASS:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "conf": conf, "cls": cls,
                "cx": float(cx), "cy": float(cy),
            })

        return detections

    def select_best_target(self, detections: List[dict], roi_w: int, roi_h: int) -> Optional[dict]:
        if not detections:
            return None

        roi_cx = roi_w / 2.0
        roi_cy = roi_h / 2.0
        best = None
        best_score = -1e9

        for d in detections:
            dx = d["cx"] - roi_cx
            dy = d["cy"] - roi_cy
            dist = (dx * dx + dy * dy) ** 0.5
            max_dist = (roi_w * roi_w + roi_h * roi_h) ** 0.5 / 2.0
            dist_norm = min(dist / max_dist, 1.0)
            score = d["conf"] * (1.0 - 0.55 * dist_norm)
            if score > best_score:
                best_score = score
                best = d
        return best

    def get_head_position(self, det: dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = x2 - x1
        h = y2 - y1
        head_x = (x1 + x2) / 2.0
        head_y = y1 + h * CONFIG.HEAD_OFFSET
        return head_x, head_y

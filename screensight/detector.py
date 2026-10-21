"""ONNX-only person detector. Does not load .pt, .pkl or executable model files."""
from __future__ import annotations
import time
from pathlib import Path

import cv2
import numpy as np
from .core import Detection, Letterbox, parse_yolo_people


class PersonDetector:
    def __init__(self, model: Path) -> None:
        import onnxruntime as ort
        if model.suffix.lower() != '.onnx' or not model.is_file():
            raise ValueError('Choose an existing .onnx model file (no .pt/.pkl files).')
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # GPU if supported; CPU fallback if not (including Windows CI runner).
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(model), sess_options=opts,
                                                providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(model), sess_options=opts,
                                                providers=['CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[1] not in (3, '3'):
            raise ValueError(f'Unsupported model input: {shape}. Expected [1,3,H,W].')
        self.height = int(shape[2]) if isinstance(shape[2], int) else 640
        self.width = int(shape[3]) if isinstance(shape[3], int) else 640
        if not 256 <= self.height <= 1280 or not 256 <= self.width <= 1280:
            raise ValueError(f'Unsupported model dimensions: {shape}')
        self.input_shape = (1, 3, self.height, self.width)

    def detect(self, rgb: np.ndarray, threshold: float) -> tuple[list[Detection], float]:
        h, w = rgb.shape[:2]
        if min(h, w) < 32:
            return [], 0.0
        scale = min(self.width / w, self.height / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.height, self.width, 3), 114, np.uint8)
        px, py = (self.width - nw) // 2, (self.height - nh) // 2
        canvas[py:py+nh, px:px+nw] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        start = time.perf_counter()
        raw = self.session.run(None, {self.input.name: tensor})[0]
        duration = (time.perf_counter() - start) * 1000
        return parse_yolo_people(raw, Letterbox(scale, px, py, w, h), threshold), duration

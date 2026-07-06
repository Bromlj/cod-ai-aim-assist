"""
capture.py
High-performance small ROI screen capture around the crosshair (screen center).

Uses mss for fast, low-overhead grabs. No unnecessary copies when possible.
"""

import time
from typing import Tuple, Optional

import mss
import numpy as np

from config import CONFIG


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor. Change index if using multi-monitor.
        self.screen_width = self.monitor["width"]
        self.screen_height = self.monitor["height"]
        self.center_x = self.screen_width // 2 + CONFIG.ROI_OFFSET_X
        self.center_y = self.screen_height // 2 + CONFIG.ROI_OFFSET_Y

        self.roi_w = CONFIG.ROI_WIDTH
        self.roi_h = CONFIG.ROI_HEIGHT

        self._update_region()

        print(f"[Capture] Screen: {self.screen_width}x{self.screen_height}")
        print(f"[Capture] ROI: {self.roi_w}x{self.roi_h} centered at ({self.center_x}, {self.center_y})")

    def _update_region(self):
        left = self.center_x - self.roi_w // 2
        top = self.center_y - self.roi_h // 2

        # Clamp to screen bounds
        left = max(0, min(left, self.screen_width - self.roi_w))
        top = max(0, min(top, self.screen_height - self.roi_h))

        self.region = {
            "left": int(left),
            "top": int(top),
            "width": int(self.roi_w),
            "height": int(self.roi_h),
        }
        # Top-left of ROI in screen coords (for converting local detections -> screen)
        self.roi_left = self.region["left"]
        self.roi_top = self.region["top"]

    def grab(self) -> Tuple[np.ndarray, int, int]:
        """
        Grab the ROI.
        Returns:
            frame: np.ndarray (H, W, 3) uint8 RGB
            roi_left, roi_top: screen coordinates of the top-left of the ROI
        """
        img = self.sct.grab(self.region)
        # mss returns BGRA
        frame = np.asarray(img, dtype=np.uint8)[:, :, :3]  # drop alpha -> BGR
        frame = frame[:, :, ::-1]  # BGR -> RGB
        return frame, self.roi_left, self.roi_top

    def get_crosshair_screen_pos(self) -> Tuple[int, int]:
        """Return absolute screen coordinates of the crosshair (center of full screen)."""
        return self.center_x, self.center_y

    def update_roi_size(self, w: int, h: int):
        """Dynamically change ROI size (useful for testing)."""
        self.roi_w = max(64, min(w, 800))
        self.roi_h = max(64, min(h, 800))
        self._update_region()

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass


# Quick test helper
if __name__ == "__main__":
    cap = ScreenCapture()
    t0 = time.perf_counter()
    for i in range(100):
        frame, rx, ry = cap.grab()
    dt = time.perf_counter() - t0
    print(f"100 grabs in {dt:.3f}s -> {100/dt:.1f} FPS capture")
    print("Frame shape:", frame.shape)
    cap.close()

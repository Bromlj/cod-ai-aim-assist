"""
aiming.py
Smooth, human-like aiming calculations.

Supports both mouse pixel moves and controller stick corrections.
"""

import random
from typing import Tuple

from config import CONFIG


class AimController:
    def __init__(self):
        self.last_target_x = 0.0
        self.last_target_y = 0.0
        self.has_target = False

    def reset(self):
        self.has_target = False
        self.last_target_x = 0.0
        self.last_target_y = 0.0

    def compute_move(self, target_dx: float, target_dy: float) -> Tuple[int, int]:
        """Mouse mode pixel delta output."""
        cfg = CONFIG
        if not self.has_target:
            smoothed_dx = target_dx
            smoothed_dy = target_dy
            self.has_target = True
        else:
            alpha = cfg.SMOOTH_FACTOR
            alpha_y = cfg.SMOOTH_FACTOR_Y
            smoothed_dx = self.last_target_x * (1 - alpha) + target_dx * alpha
            smoothed_dy = self.last_target_y * (1 - alpha_y) + target_dy * alpha_y

        self.last_target_x = smoothed_dx
        self.last_target_y = smoothed_dy

        move_x = smoothed_dx * alpha
        move_y = smoothed_dy * alpha_y

        move_x = max(min(move_x, cfg.MAX_MOVE_X), -cfg.MAX_MOVE_X)
        move_y = max(min(move_y, cfg.MAX_MOVE_Y), -cfg.MAX_MOVE_Y)

        if abs(move_x) < cfg.MIN_MOVE: move_x = 0
        if abs(move_y) < cfg.MIN_MOVE: move_y = 0

        if cfg.JITTER > 0 and (move_x or move_y):
            move_x += random.uniform(-cfg.JITTER, cfg.JITTER)
            move_y += random.uniform(-cfg.JITTER, cfg.JITTER)

        return int(round(move_x)), int(round(move_y))

    def compute_controller_correction(self, target_dx: float, target_dy: float,
                                      frame_width: int) -> Tuple[float, float]:
        """
        Controller mode: returns normalized right stick correction (x, y) in -1..1 range.
        Uses same smoothing as mouse but scaled for stick.
        """
        cfg = CONFIG
        if not self.has_target:
            smoothed_dx = target_dx
            smoothed_dy = target_dy
            self.has_target = True
        else:
            alpha = cfg.SMOOTH_FACTOR
            alpha_y = cfg.SMOOTH_FACTOR_Y
            smoothed_dx = self.last_target_x * (1 - alpha) + target_dx * alpha
            smoothed_dy = self.last_target_y * (1 - alpha_y) + target_dy * alpha_y

        self.last_target_x = smoothed_dx
        self.last_target_y = smoothed_dy

        # Convert to normalized stick using shared logic
        from virtual_controller import pixel_delta_to_stick_delta
        sx, sy = pixel_delta_to_stick_delta(smoothed_dx, smoothed_dy, frame_width)
        return sx, sy

    def calculate_target_delta(self, head_local_x: float, head_local_y: float,
                               roi_left: int, roi_top: int,
                               crosshair_x: int, crosshair_y: int) -> Tuple[float, float]:
        screen_head_x = roi_left + head_local_x
        screen_head_y = roi_top + head_local_y
        return screen_head_x - crosshair_x, screen_head_y - crosshair_y


def get_aim_move(target_dx: float, target_dy: float) -> Tuple[int, int]:
    cfg = CONFIG
    move_x = int(max(min(target_dx * cfg.SMOOTH_FACTOR, cfg.MAX_MOVE_X), -cfg.MAX_MOVE_X))
    move_y = int(max(min(target_dy * cfg.SMOOTH_FACTOR_Y, cfg.MAX_MOVE_Y), -cfg.MAX_MOVE_Y))
    if abs(move_x) < cfg.MIN_MOVE: move_x = 0
    if abs(move_y) < cfg.MIN_MOVE: move_y = 0
    return move_x, move_y

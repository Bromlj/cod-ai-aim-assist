"""
overlay.py
On-screen Controller Visual for the AI Aim Assist.

Shows:
- Simplified Xbox-style controller
- Left & Right analog sticks (physical + AI correction)
- AI aim correction vector
- Status (active / detections / FPS)

Uses pygame for a lightweight, always-on-top transparent-ish window.
Does not interfere with your physical controller.

Run in its own thread.
"""

import pygame
import math
from typing import Tuple, Optional
import threading
import time

from config import CONFIG

# Colors
BG = (20, 22, 26)
STICK_BG = (40, 42, 48)
STICK_OUTLINE = (80, 82, 90)
AI_COLOR = (0, 200, 255)       # Cyan for AI correction
PHYSICAL_COLOR = (255, 180, 60) # Orange for physical
CROSS = (255, 80, 80)
TEXT = (220, 225, 230)
ACTIVE = (60, 220, 100)


class ControllerVisual:
    def __init__(self):
        pygame.init()
        self.width = CONFIG.OVERLAY_WIDTH
        self.height = CONFIG.OVERLAY_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        pygame.display.set_caption("COD AI - Controller Visual")

        # Make it always on top (Windows)
        if CONFIG.OVERLAY_ALWAYS_ON_TOP:
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info()["window"]
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)
            except:
                pass

        self.clock = pygame.time.Clock()
        self.running = True

        # State (updated from main thread)
        self.left_stick = (0.0, 0.0)      # physical
        self.right_stick_physical = (0.0, 0.0)
        self.right_stick_ai = (0.0, 0.0)  # correction only
        self.active = False
        self.detections = 0
        self.fps = 0.0
        self.lock = threading.Lock()

    def update_state(self, left: Tuple[float, float], right_phys: Tuple[float, float],
                     right_ai: Tuple[float, float], active: bool, dets: int, fps: float):
        with self.lock:
            self.left_stick = left
            self.right_stick_physical = right_phys
            self.right_stick_ai = right_ai
            self.active = active
            self.detections = dets
            self.fps = fps

    def _draw_stick(self, cx: int, cy: int, radius: int, sx: float, sy: float, color, label: str, ai_offset=None):
        # Background circle
        pygame.draw.circle(self.screen, STICK_BG, (cx, cy), radius, 0)
        pygame.draw.circle(self.screen, STICK_OUTLINE, (cx, cy), radius, 2)

        # Stick position
        px = int(cx + sx * (radius * 0.85))
        py = int(cy + sy * (radius * 0.85))
        pygame.draw.circle(self.screen, color, (px, py), 12, 0)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 12, 2)

        # AI correction indicator (small arrow or second dot)
        if ai_offset is not None and (abs(ai_offset[0]) > 0.01 or abs(ai_offset[1]) > 0.01):
            aix = int(cx + (sx + ai_offset[0]) * (radius * 0.85))
            aiy = int(cy + (sy + ai_offset[1]) * (radius * 0.85))
            pygame.draw.circle(self.screen, AI_COLOR, (aix, aiy), 7, 0)
            # Line from physical to AI
            pygame.draw.line(self.screen, AI_COLOR, (px, py), (aix, aiy), 2)

        # Label
        font = pygame.font.SysFont("Arial", 14, bold=True)
        txt = font.render(label, True, TEXT)
        self.screen.blit(txt, (cx - txt.get_width()//2, cy + radius + 6))

    def draw(self):
        self.screen.fill(BG)

        # Title bar
        font = pygame.font.SysFont("Arial", 15, bold=True)
        title = font.render("COD AI AIM ASSIST — Controller Visual", True, TEXT)
        self.screen.blit(title, (12, 6))

        # Status
        status_color = ACTIVE if self.active else (200, 80, 80)
        status = font.render("ACTIVE" if self.active else "INACTIVE", True, status_color)
        self.screen.blit(status, (self.width - status.get_width() - 12, 6))

        # Sticks
        left_cx, left_cy = 95, 115
        right_cx, right_cy = self.width - 95, 115
        radius = 52

        with self.lock:
            l = self.left_stick
            rp = self.right_stick_physical
            ra = self.right_stick_ai

        # Left stick (physical only)
        self._draw_stick(left_cx, left_cy, radius, l[0], l[1], PHYSICAL_COLOR, "LEFT (Physical)")

        # Right stick: physical + AI overlay
        self._draw_stick(right_cx, right_cy, radius, rp[0], rp[1], PHYSICAL_COLOR, "RIGHT (Phys + AI)", ai_offset=ra)

        # Info line
        small = pygame.font.SysFont("Arial", 12)
        info = small.render(f"Detections: {self.detections}   FPS: {self.fps:.0f}   Strength: {CONFIG.CONTROLLER_AIM_STRENGTH:.2f}", True, TEXT)
        self.screen.blit(info, (12, self.height - 22))

        # Legend
        leg = small.render("Orange = Physical   Cyan = AI correction", True, (170, 175, 180))
        self.screen.blit(leg, (12, self.height - 40))

        pygame.display.flip()

    def run(self):
        """Blocking loop for the overlay. Call in a thread."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            try:
                self.draw()
            except Exception as e:
                # Prevent the thread from dying on transient errors
                print(f"[Overlay] draw error (non-fatal): {e}")
            self.clock.tick(60)  # 60 FPS overlay is plenty

        pygame.quit()

    def stop(self):
        self.running = False


def start_overlay_thread() -> Optional[ControllerVisual]:
    """Returns the visual object (so main can call update_state on it)."""
    if not CONFIG.SHOW_CONTROLLER_VISUAL:
        return None
    try:
        visual = ControllerVisual()
        t = threading.Thread(target=visual.run, daemon=True, name="controller_visual")
        t.start()
        return visual
    except Exception as e:
        print(f"[Overlay] Failed to start controller visual: {e}")
        return None

"""
config.py
Central configuration for the BEST AI Aim Assist for Call of Duty.

Supports:
- Local screen capture
- HDMI Capture Card (Video Device)
- Chiaki (PS Remote Play window)
- Xbox Remote Play / Cloud / Share Play (window or app capture)

Controller mode with virtual gamepad (non-blocking physical controller) + on-screen controller visual.

For maximum performance use TensorRT .engine + small ROI.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import os


@dataclass
class AimAssistConfig:
    # ==========================================
    # MODEL - BEST PRACTICES
    # ==========================================
    # For best accuracy + speed trade-off: YOLO11n or YOLO11s exported to TensorRT engine.
    # The included cod_yolo11n.pt is a solid starting point (MW2 body).
    # For "best of the best":
    #   1. Collect fresh screenshots from current COD season.
    #   2. Fine-tune on largest recent Roboflow COD dataset (search warzone/black-ops).
    #   3. Export with imgsz=320 or 384, half=True.
    # Prefer .engine for 200-400+ FPS inference.
    MODEL_PATH: str = os.path.join("models", "cod_yolo11n.pt")
    FALLBACK_MODEL_PATHS: Tuple[str, ...] = (
        os.path.join("models", "best.engine"),
        os.path.join("models", "best.pt"),
        os.path.join("models", "cod_yolo11n.pt"),
        os.path.join("yolo-training", "model", "cod_yolo11n.pt"),
        os.path.join("yolo-training", "CallOfDutyBO7Ghostaim.onnx"),  # convertible
        "yolo11n.pt",
    )

    IMGSZ: int = 320
    CONF: float = 0.38
    IOU: float = 0.45
    MAX_DET: int = 8
    # Note: 'half' handled inside inference for compatibility

    ENEMY_CLASS: int = 0   # Adjust if your model uses different indexing (e.g. 0=player)
    HEAD_CLASS: Optional[int] = None

    # ==========================================
    # CAPTURE SOURCE (choose one)
    # ==========================================
    # "screen"     - Local PC game (original mss)
    # "video"      - HDMI Capture Card (best for console + low added latency)
    # "window"     - Specific window title (Chiaki-ng, Xbox app, browser remote play)
    CAPTURE_MODE: str = "screen"   # "screen" | "video" | "window"

    # For CAPTURE_MODE="video" (capture card)
    VIDEO_DEVICE_INDEX: int = 0     # Usually 0 or 1. Run helper to list: python -c "import cv2; [print(i, cap.getBackendName()) for i,cap in enumerate([cv2.VideoCapture(i) for i in range(5)] if cap.isOpened())]"
    VIDEO_WIDTH: int = 1280         # Match your capture card output resolution
    VIDEO_HEIGHT: int = 720
    VIDEO_FPS: int = 60

    # For CAPTURE_MODE="window" (Chiaki or Xbox Remote Play)
    WINDOW_TITLE_CONTAINS: str = "Chiaki"   # e.g. "Chiaki", "Xbox", "Remote Play", "Google Chrome" etc.
    # Falls back to full primary monitor if window not found

    # ROI (relative to the *captured frame* center). Keep small for speed.
    ROI_WIDTH: int = 360
    ROI_HEIGHT: int = 360
    ROI_OFFSET_X: int = 0
    ROI_OFFSET_Y: int = 0

    # ==========================================
    # INPUT / AIMING MODE
    # ==========================================
    # "mouse"     - Original mouse emulation (KBM)
    # "controller" - Virtual Xbox 360 controller (best for console remote play / many titles)
    INPUT_MODE: str = "controller"   # "mouse" | "controller"

    # Controller-specific
    # How strongly AI corrects the right stick (0.1 = subtle assist, 0.6 = strong)
    CONTROLLER_AIM_STRENGTH: float = 0.55
    CONTROLLER_DEADZONE: float = 0.08
    # Max right stick change per frame (prevents obvious snapping)
    CONTROLLER_MAX_DELTA: float = 0.18

    # Head offset for controller too
    HEAD_OFFSET: float = 0.18

    # ==========================================
    # SMOOTHING (shared)
    # ==========================================
    SMOOTH_FACTOR: float = 0.42
    SMOOTH_FACTOR_Y: float = 0.36
    # For mouse mode
    MAX_MOVE_X: int = 26
    MAX_MOVE_Y: int = 20
    MIN_MOVE: int = 1
    JITTER: float = 0.5

    # ==========================================
    # ACTIVATION
    # ==========================================
    ACTIVATION_VK: int = 0x02      # RMB
    TOGGLE_VK: Optional[int] = 0x71  # F2

    # For controller mode, you can also activate on a physical controller button
    # (LT = 4, RT=5, etc in pygame). 0 disables.
    CONTROLLER_ACTIVATION_BUTTON: int = 0   # e.g. 6 for LB or experiment

    # ==========================================
    # VISUAL OVERLAY (Controller Visual)
    # ==========================================
    SHOW_CONTROLLER_VISUAL: bool = True
    OVERLAY_WIDTH: int = 420
    OVERLAY_HEIGHT: int = 220
    OVERLAY_ALWAYS_ON_TOP: bool = True

    # ==========================================
    # PERFORMANCE
    # ==========================================
    TARGET_FPS: int = 240
    FPS_REPORT_INTERVAL: int = 60
    USE_THREADED_CAPTURE: bool = True   # Strongly recommended for video/window sources

    DEBUG: bool = False


CONFIG = AimAssistConfig()


def get_model_path() -> str:
    cfg = CONFIG
    candidates = [cfg.MODEL_PATH] + list(cfg.FALLBACK_MODEL_PATHS)
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return cfg.MODEL_PATH


def list_video_devices():
    """Helper to discover capture card indices."""
    try:
        import cv2
        print("Available video devices:")
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [{i}] Open: {w}x{h}")
                cap.release()
            else:
                cap.release()
    except Exception as e:
        print("cv2 not available or error listing devices:", e)

"""
capture_sources.py
Best-of-the-best capture backends for AI Aim Assist.

Supported modes (set in config.CAPTURE_MODE):
- "screen"  : Local screen (mss, fast)
- "video"   : HDMI Capture Card via OpenCV (threaded for lowest latency)
- "window"  : Specific application window (Chiaki-ng, Xbox Remote Play, etc.)

All return: (frame_rgb, roi_left, roi_top, frame_center_x, frame_center_y)
Where roi_left/top are relative to the captured frame for coordinate conversion.
"""

import time
import threading
from typing import Tuple, Optional
import numpy as np

import mss
from config import CONFIG

try:
    import cv2
except ImportError:
    cv2 = None


class BaseCapture:
    def grab(self) -> Tuple[np.ndarray, int, int, int, int]:
        """Return frame_rgb, roi_left, roi_top, center_x, center_y"""
        raise NotImplementedError

    def close(self):
        pass

    def get_frame_size(self) -> Tuple[int, int]:
        return CONFIG.ROI_WIDTH, CONFIG.ROI_HEIGHT


class ThreadedFrameGrabber:
    """Always keeps the most recent frame from a source in a background thread.
    Critical for low-latency with capture cards and streams.
    """
    def __init__(self, source_func, name="grabber"):
        self.source_func = source_func
        self.latest_frame = None
        self.latest_meta = (0, 0, 0, 0)  # left, top, cx, cy
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                frame, *meta = self.source_func()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_meta = meta
            except Exception:
                time.sleep(0.005)
            time.sleep(0.001)  # yield

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, (0,0,0,0)
            return self.latest_frame.copy(), self.latest_meta

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class ScreenCapture(BaseCapture):
    """Original high-speed local screen capture around center."""
    def __init__(self):
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        self.sw, self.sh = mon["width"], mon["height"]
        self.cx = self.sw // 2 + CONFIG.ROI_OFFSET_X
        self.cy = self.sh // 2 + CONFIG.ROI_OFFSET_Y
        self._update_region()
        print(f"[Capture] SCREEN mode | {self.sw}x{self.sh} | ROI {CONFIG.ROI_WIDTH}x{CONFIG.ROI_HEIGHT}")

    def _update_region(self):
        rw, rh = CONFIG.ROI_WIDTH, CONFIG.ROI_HEIGHT
        left = max(0, self.cx - rw // 2)
        top = max(0, self.cy - rh // 2)
        self.region = {"left": left, "top": top, "width": rw, "height": rh}
        self.roi_left = left
        self.roi_top = top

    def grab(self):
        img = self.sct.grab(self.region)
        frame = np.asarray(img, dtype=np.uint8)[:, :, :3]
        frame = frame[:, :, ::-1]  # BGRA -> RGB
        return frame, self.roi_left, self.roi_top, self.cx, self.cy

    def close(self):
        try:
            self.sct.close()
        except:
            pass


class VideoDeviceCapture(BaseCapture):
    """HDMI Capture Card. Uses threaded grabber for latest-frame-only (low latency)."""
    def __init__(self):
        if cv2 is None:
            raise RuntimeError("opencv-python required for video device capture. pip install opencv-python")

        self.cap = cv2.VideoCapture(CONFIG.VIDEO_DEVICE_INDEX, cv2.CAP_DSHOW)  # dshow often lower latency on Windows
        if not self.cap.isOpened():
            # Fallback without dshow
            self.cap = cv2.VideoCapture(CONFIG.VIDEO_DEVICE_INDEX)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG.VIDEO_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # critical for low latency

        # Warm up
        for _ in range(5):
            self.cap.read()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = w
        self.frame_h = h
        self.cx = w // 2 + CONFIG.ROI_OFFSET_X
        self.cy = h // 2 + CONFIG.ROI_OFFSET_Y

        print(f"[Capture] VIDEO (capture card) mode | device={CONFIG.VIDEO_DEVICE_INDEX} | {w}x{h}")

        # Threaded grabber
        if CONFIG.USE_THREADED_CAPTURE:
            self.grabber = ThreadedFrameGrabber(self._raw_grab, "video_grabber")
        else:
            self.grabber = None

    def _raw_grab(self):
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            # Return black frame on failure
            frame_bgr = np.zeros((self.frame_h or 720, self.frame_w or 1280, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rw, rh = CONFIG.ROI_WIDTH, CONFIG.ROI_HEIGHT
        left = max(0, self.cx - rw // 2)
        top = max(0, self.cy - rh // 2)
        return frame_rgb, left, top, self.cx, self.cy

    def grab(self):
        if self.grabber:
            frame, meta = self.grabber.get_latest()
            if frame is not None:
                return frame, *meta
        return self._raw_grab()

    def close(self):
        if hasattr(self, 'grabber') and self.grabber:
            self.grabber.stop()
        if self.cap:
            self.cap.release()


class WindowCapture(BaseCapture):
    """Capture from a specific window (Chiaki, Xbox remote play, etc).
    Uses mss on the window rectangle. Falls back to full screen.
    """
    def __init__(self):
        self.sct = mss.mss()
        self.window_rect = None
        self._find_window()

        if self.window_rect:
            print(f"[Capture] WINDOW mode | title contains '{CONFIG.WINDOW_TITLE_CONTAINS}' | rect={self.window_rect}")
        else:
            mon = self.sct.monitors[1]
            self.window_rect = {"left": 0, "top": 0, "width": mon["width"], "height": mon["height"]}
            print(f"[Capture] WINDOW fallback to full screen (window '{CONFIG.WINDOW_TITLE_CONTAINS}' not found)")

        self.cx = self.window_rect["width"] // 2 + CONFIG.ROI_OFFSET_X
        self.cy = self.window_rect["height"] // 2 + CONFIG.ROI_OFFSET_Y

        if CONFIG.USE_THREADED_CAPTURE:
            self.grabber = ThreadedFrameGrabber(self._raw_grab, "window_grabber")
        else:
            self.grabber = None

    def _find_window(self):
        try:
            import pygetwindow as gw
            titles = gw.getAllTitles()
            for t in titles:
                if t and CONFIG.WINDOW_TITLE_CONTAINS.lower() in t.lower():
                    win = gw.getWindowsWithTitle(t)[0]
                    if win and win.visible:
                        self.window_rect = {
                            "left": win.left,
                            "top": win.top,
                            "width": win.width,
                            "height": win.height
                        }
                        return
        except ImportError:
            print("[WindowCapture] pygetwindow not installed. Falling back to full screen or manual region.")
        except Exception as e:
            print(f"[WindowCapture] Error finding window: {e}")

    def _raw_grab(self):
        rect = self.window_rect or self.sct.monitors[1]
        img = self.sct.grab(rect)
        frame = np.asarray(img, dtype=np.uint8)[:, :, :3][:, :, ::-1]  # RGB
        rw, rh = CONFIG.ROI_WIDTH, CONFIG.ROI_HEIGHT
        left = max(0, self.cx - rw // 2)
        top = max(0, self.cy - rh // 2)
        return frame, left, top, self.cx, self.cy

    def grab(self):
        if self.grabber:
            f, m = self.grabber.get_latest()
            if f is not None:
                return f, *m
        return self._raw_grab()

    def close(self):
        if self.grabber:
            self.grabber.stop()
        try:
            self.sct.close()
        except:
            pass


def get_capture() -> BaseCapture:
    """Factory for the configured capture source. Best entry point."""
    mode = CONFIG.CAPTURE_MODE.lower()
    if mode == "video":
        return VideoDeviceCapture()
    elif mode == "window":
        return WindowCapture()
    else:
        return ScreenCapture()


# Quick device lister
if __name__ == "__main__":
    from config import list_video_devices
    list_video_devices()
    print("\nTesting capture factory with current config...")
    cap = get_capture()
    for _ in range(3):
        f, rl, rt, cx, cy = cap.grab()
        print("Frame shape:", f.shape, "center:", cx, cy)
    cap.close()

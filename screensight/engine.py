"""Worker loop for screen-only vision and opt-in, ADS-gated Windows mouse input."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import sys
import threading
import time

import cv2
import numpy as np

from .core import Detection, MovementFilter, TargetTracker
from .detector import PersonDetector
from .windows import WindowArea, VK_F8, VK_RBUTTON, foreground_game, key_down, move_mouse

# Pinned source: https://huggingface.co/webnn/yolo11n/blob/main/onnx/yolo11n.onnx
DEFAULT_MODEL_SHA256 = '7d8fd1717d9d5bbab6986cd134afb620649c7a394303d55b1e09fc00804cc5c1'


def bundled_model() -> Path:
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
    return base / 'models' / 'person-yolo11n.onnx'


@dataclass(frozen=True)
class Options:
    model: str = ''
    window_titles: str = 'Call of Duty|Black Ops 7'
    roi: int = 480
    confidence: float = 0.46
    aim_height: float = 0.21
    strength: float = 0.22
    max_step: int = 23
    preview: bool = True
    overlay: bool = True


@dataclass(frozen=True)
class FrameState:
    state: str = 'STOPPED'
    message: str = 'Not started'
    provider: str = '-'
    fps: float = 0.0
    inference_ms: float = 0.0
    window: str = ''
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    boxes: tuple[Detection, ...] = ()
    target: tuple[float, float] | None = None
    ads: bool = False
    input_enabled: bool = False
    control_active: bool = False


class Engine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._options = Options(model=str(bundled_model()))
        self._latest = FrameState()
        self._control_enabled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def options(self) -> Options:
        with self._lock:
            return self._options

    def configure(self, **kwargs: object) -> None:
        with self._lock:
            self._options = replace(self._options, **kwargs)

    def set_control(self, enabled: bool) -> None:
        with self._lock:
            self._control_enabled = bool(enabled)

    def snapshot(self) -> FrameState:
        with self._lock:
            return self._latest

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        with self._lock:
            self._control_enabled = False  # always begin in read-only mode
        self._thread = threading.Thread(target=self._work, name='VisionWorker', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.set_control(False)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.8)
        self._set_frame(FrameState(state='STOPPED', message='Stopped by user'))

    def _set_frame(self, state: FrameState) -> None:
        with self._lock:
            self._latest = state

    def _work(self) -> None:
        self._set_frame(FrameState(state='LOADING', message='Validating local model'))
        with self._lock:
            model = Path(self._options.model).expanduser()
        try:
            if model.resolve() == bundled_model().resolve():
                digest = hashlib.sha256(model.read_bytes()).hexdigest()
                if digest != DEFAULT_MODEL_SHA256:
                    raise ValueError('Bundled model hash mismatch. Do not run a modified EXE.')
            detector = PersonDetector(model)
        except Exception as ex:
            self._set_frame(FrameState(state='ERROR', message=str(ex)[:240]))
            return
        self._set_frame(FrameState(state='READY', message='Hold right mouse while game is focused',
                                   provider=detector.provider))

        import mss
        tracker = TargetTracker()
        mover = MovementFilter()
        prev_f8 = False
        last_tick = time.perf_counter()
        fps_smooth = 0.0
        try:
            with mss.mss() as grabber:
                while not self._stop.is_set():
                    now = time.perf_counter()
                    f8 = key_down(VK_F8)
                    if f8 and not prev_f8:
                        self.set_control(False)
                    prev_f8 = f8
                    with self._lock:
                        opts, enabled = self._options, self._control_enabled
                    filters = tuple(x.strip() for x in opts.window_titles.split('|') if len(x.strip()) >= 5)
                    win = foreground_game(filters) if filters else None
                    ads = key_down(VK_RBUTTON)
                    if win is None:
                        tracker.reset()
                        mover.reset()
                        self._set_frame(FrameState(state='WAITING', message='Focus a matching game window',
                                                   provider=detector.provider, input_enabled=enabled))
                        self._stop.wait(0.13)
                        continue
                    if not ads and not opts.preview:
                        tracker.reset()
                        mover.reset()
                        self._set_frame(FrameState(state='READY', message='Hold right mouse to detect',
                                                   provider=detector.provider, window=win.title,
                                                   input_enabled=enabled))
                        self._stop.wait(0.06)
                        continue

                    side = max(256, min(800, int(opts.roi)))
                    rw = min(side, win.width)
                    rh = min(side, win.height)
                    left = win.left + (win.width - rw) // 2
                    top = win.top + (win.height - rh) // 2
                    try:
                        shot = grabber.grab({'left': left, 'top': top, 'width': rw, 'height': rh})
                        rgb = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2RGB)
                        boxes, infer_ms = detector.detect(rgb, opts.confidence)
                    except Exception as ex:
                        self._set_frame(FrameState(state='ERROR', message=f'Capture/inference: {ex}'[:240],
                                                   provider=detector.provider))
                        self._stop.wait(0.18)
                        continue

                    target = tracker.choose(boxes, rw, rh, opts.aim_height, now)
                    actual_enabled = bool(ads and enabled and target and not self._stop.is_set())
                    if actual_enabled and target:
                        tx, ty = target[1]
                        dx, dy = mover.compute(tx - rw / 2, ty - rh / 2,
                                               opts.strength, max(3, int(opts.max_step)))
                        if (dx or dy) and not move_mouse(dx, dy):
                            self.set_control(False)
                            enabled = False
                            actual_enabled = False
                    else:
                        mover.reset()
                        if not ads:
                            tracker.reset()
                    tick = time.perf_counter()
                    inst_fps = 1.0 / max(1e-5, tick - last_tick)
                    last_tick = tick
                    fps_smooth = fps_smooth * .82 + inst_fps * .18 if fps_smooth else inst_fps
                    self._set_frame(FrameState(state='ACTIVE' if actual_enabled else 'PREVIEW',
                                               message='Mouse movement enabled' if actual_enabled else
                                                       ('Detecting (no mouse input)' if not enabled else 'Mouse gated by ADS'),
                                               provider=detector.provider,
                                               fps=fps_smooth, inference_ms=infer_ms,
                                               window=win.title, region=(left, top, rw, rh),
                                               boxes=tuple(boxes), target=target[1] if target else None,
                                               ads=ads, input_enabled=enabled,
                                               control_active=actual_enabled))
                    # Yield to the game and keep background CPU use bounded.
                    self._stop.wait(.006 if ads else .035)
        except Exception as ex:
            self._set_frame(FrameState(state='ERROR', message=f'Worker: {ex}'[:240],
                                       provider=detector.provider))
        finally:
            self.set_control(False)

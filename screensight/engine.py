"""Single-worker capture/inference with fresh-frame input guards and diagnostics."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import sys
import threading
import time

import cv2
import numpy as np
from .core import Detection, MovementFilter, TargetTracker
from .detector import PersonDetector
from .settings import Options, VERSION, input_permitted, load_settings
from .windows import VK_F8, VK_RBUTTON, foreground_game, key_down, move_mouse

DEFAULT_MODEL_SHA256 = '7d8fd1717d9d5bbab6986cd134afb620649c7a394303d55b1e09fc00804cc5c1'


def bundled_model() -> Path:
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
    return base / 'models' / 'person-yolo11n.onnx'


@dataclass(frozen=True)
class FrameState:
    state: str = 'STOPPED'
    message: str = 'Not started'
    provider: str = '-'
    fps: float = 0.0
    inference_ms: float = 0.0
    capture_ms: float = 0.0
    frame_age_ms: float = 0.0
    last_success: float = 0.0
    frames: int = 0
    stale_frames: int = 0
    track_state: str = 'SEARCHING'
    track_id: int = 0
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
        self._options = replace(load_settings(), model=str(bundled_model()))
        self._latest = FrameState()
        self._control_enabled = False
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def options(self) -> Options:
        with self._lock:
            return self._options

    def configure(self, **kwargs: object) -> None:
        with self._lock:
            updated = replace(self._options, **kwargs)
            if updated != self._options:
                self._generation += 1
            self._options = updated

    def set_control(self, enabled: bool) -> None:
        with self._lock:
            self._control_enabled = bool(enabled) and not self._stop.is_set()
            self._generation += 1
            self._latest = replace(self._latest, input_enabled=self._control_enabled,
                                   control_active=False)

    def snapshot(self) -> FrameState:
        with self._lock:
            return self._latest

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        # Never clear a shared stop event while an old inference call is alive.
        if self.is_running():
            return False
        self._stop.clear()
        self.set_control(False)
        self._latest = FrameState()
        self._thread = threading.Thread(target=self._work, name='VisionWorker', daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        self.set_control(False)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=.15)
        self._pause('STOPPING' if self.is_running() else 'STOPPED',
                    'Waiting for the current inference call' if self.is_running() else 'Stopped')

    def _publish(self, state: FrameState) -> None:
        with self._lock:
            self._latest = replace(state, input_enabled=self._control_enabled,
                                   control_active=state.control_active and self._control_enabled)

    def _pause(self, state: str, message: str) -> None:
        # Retain measurements, explicitly marked as LAST by the UI, instead of false zeroes.
        with self._lock:
            self._latest = replace(self._latest, state=state, message=message,
                                   control_active=False, input_enabled=self._control_enabled,
                                   boxes=(), target=None, region=(0, 0, 0, 0))

    def report(self) -> dict:
        snapshot = asdict(self.snapshot())
        for key in ('window', 'boxes', 'target', 'region', 'last_success'):
            snapshot.pop(key, None)
        options = asdict(self.options())
        for key in ('model', 'window_titles'):
            options.pop(key, None)
        return {'version': VERSION, 'metrics': snapshot, 'settings': options,
                'note': 'User-requested local export. No screenshots, window titles, or paths.'}

    def _work(self) -> None:
        tracker, mover = TargetTracker(), MovementFilter()
        try:
            self._pause('LOADING', 'Validating the local ONNX model')
            model = Path(self.options().model).expanduser()
            if model.resolve() == bundled_model().resolve():
                if hashlib.sha256(model.read_bytes()).hexdigest() != DEFAULT_MODEL_SHA256:
                    raise ValueError('Bundled model hash mismatch')
            detector = PersonDetector(model)
            if self._stop.is_set():
                return
            self._publish(FrameState(state='READY', provider=detector.provider,
                                     message='Focus a matching window; mouse control starts OFF'))
            import mss
            previous_region = None
            previous_time = None
            previous_track = None
            previous_ads = False
            fps_smooth = 0.0
            frames = stale_frames = 0
            consecutive_errors = 0
            with mss.mss() as grabber:
                while not self._stop.is_set():
                    cycle_start = time.perf_counter()
                    if key_down(VK_F8):
                        self.set_control(False)
                    with self._lock:
                        opts = self._options
                        generation = self._generation
                    filters = tuple(v.strip() for v in opts.window_titles.split('|') if len(v.strip()) >= 5)
                    win = foreground_game(filters) if filters else None
                    ads = key_down(VK_RBUTTON)
                    if win is None or (not ads and not opts.preview):
                        tracker.reset()
                        mover.reset()
                        previous_time = previous_region = previous_track = None
                        previous_ads = False
                        self._pause('WAITING' if win is None else 'READY',
                                    'No matching foreground window - last metrics retained' if win is None
                                    else 'Preview disabled - hold right mouse to process frames')
                        self._stop.wait(.08)
                        continue
                    rw, rh = min(opts.roi, win.width), min(opts.roi, win.height)
                    left, top = win.left + (win.width-rw)//2, win.top + (win.height-rh)//2
                    region = (win.hwnd, left, top, rw, rh, opts.aim_height)
                    if region != previous_region or ads != previous_ads:
                        tracker.reset()
                        mover.reset()
                        previous_time = previous_track = None
                    previous_region, previous_ads = region, ads
                    capture_start = time.perf_counter()
                    try:
                        image = grabber.grab(dict(left=left, top=top, width=rw, height=rh))
                        rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGRA2RGB)
                        capture_ms = (time.perf_counter()-capture_start)*1000
                        boxes, inference_ms = detector.detect(rgb, opts.confidence)
                        consecutive_errors = 0
                    except Exception as error:
                        self.set_control(False)
                        tracker.reset()
                        mover.reset()
                        consecutive_errors += 1
                        self._pause('ERROR', f'Capture/inference: {error}'[:240])
                        if consecutive_errors >= 3:
                            break
                        self._stop.wait(.2)
                        continue
                    now = time.perf_counter()
                    dt = now-previous_time if previous_time is not None else 1/max(20, opts.fps_limit)
                    previous_time = now
                    age = now-capture_start
                    frames += 1
                    fps_smooth = (1/dt if not fps_smooth else .8*fps_smooth + .2/dt)
                    if age > .12:
                        stale_frames += 1
                        tracker.reset()
                        target = None
                    else:
                        target = tracker.choose(boxes, rw, rh, opts.aim_height, capture_start,
                                                opts.lead_ms / 1000.)
                    if tracker.track_id != previous_track:
                        mover.reset()
                    previous_track = tracker.track_id
                    active = False
                    # Capture-time checks alone are insufficient: focus/RMB may change in inference.
                    current = foreground_game(filters) if filters else None
                    matching = current is not None and (current.hwnd, current.left, current.top, current.width, current.height) == (win.hwnd, win.left, win.top, win.width, win.height)
                    f8 = key_down(VK_F8)
                    rmb = key_down(VK_RBUTTON)
                    with self._lock:
                        if f8:
                            self._control_enabled = False
                            self._generation += 1
                        permitted = input_permitted(
                            armed=self._control_enabled, confirmed=bool(target and tracker.confirmed),
                            same_window=matching, rmb=rmb, f8=f8, stopping=self._stop.is_set(),
                            same_generation=generation == self._generation,
                            frame_age=time.perf_counter()-capture_start)
                        if permitted and target:
                            x, y = target[1]
                            dx, dy = mover.compute(x-rw/2, y-rh/2, opts.strength,
                                                   opts.max_step, opts.deadzone, dt)
                            active = True
                            if (dx or dy) and not move_mouse(dx, dy):
                                self._control_enabled = False
                                self._generation += 1
                                active = False
                        else:
                            mover.reset()
                    reason = ('Frame too old - input paused' if age > .12 else
                              'Focus changed - input paused' if not matching else
                              'Movement active' if active else
                              'Mouse disabled / right-click not held / target unconfirmed')
                    self._publish(FrameState(
                        state='ACTIVE' if active else 'PREVIEW', message=reason,
                        provider=detector.provider, fps=fps_smooth, inference_ms=inference_ms,
                        capture_ms=capture_ms, frame_age_ms=age*1000, last_success=now,
                        frames=frames, stale_frames=stale_frames, track_state=tracker.state,
                        track_id=tracker.track_id, window=win.title,
                        region=(left, top, rw, rh) if matching else (0, 0, 0, 0),
                        boxes=tuple(boxes), target=target[1] if target else None,
                        ads=rmb, control_active=active))
                    fps_limit = opts.fps_limit if ads else min(30, opts.fps_limit)
                    self._stop.wait(max(.001, 1/fps_limit-(time.perf_counter()-cycle_start)))
        except Exception as error:
            self._pause('ERROR', f'{type(error).__name__}: {error}'[:240])
        finally:
            self.set_control(False)
            if self._stop.is_set():
                self._pause('STOPPED', 'Stopped')

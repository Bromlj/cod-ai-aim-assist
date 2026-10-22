"""Deterministic regression tests. These are not Black Ops 7 gameplay tests."""
from dataclasses import replace
import json
import math
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from screensight.core import Detection, Letterbox, MovementFilter, TargetTracker, parse_yolo_people
from screensight.settings import Options, input_permitted, load_settings, save_settings
from screensight.windows import normalized_window_title, window_title_matches


def person(x, y=190, width=30, height=100, score=.9):
    return Detection(x, y, x+width, y+height, score)


def test_invisible_title_characters_remain_supported():
    title = '\u200b'.join('Call of Duty®: Black Ops 7')
    assert window_title_matches(title, ('Call of Duty',))
    assert normalized_window_title(title) == 'call of duty®: black ops 7'
    assert not window_title_matches(title, ('Notepad',))
    assert not window_title_matches(title, ('', '   '))


def test_confirmed_lock_survives_more_central_distractor():
    tracker = TargetTracker()
    a, b = person(280), person(215)
    assert tracker.choose([a], 480, 480, .21, 1)[0] == a
    assert not tracker.confirmed
    assert tracker.choose([a, b], 480, 480, .21, 1.016)[0] == a
    assert tracker.confirmed
    assert tracker.track_id == 1


def test_lost_track_does_not_return_guessed_movement_or_switch_instantly():
    tracker = TargetTracker()
    a, b = person(280), person(150)
    tracker.choose([a], 480, 480, .21, 1)
    tracker.choose([a], 480, 480, .21, 1.016)
    assert tracker.choose([b], 480, 480, .21, 1.032) is None
    assert not tracker.confirmed
    assert tracker.choose([b], 480, 480, .21, 1.2)[0] == b
    assert tracker.track_id == 2
    assert not tracker.confirmed


def test_two_frames_required_again_after_a_detection_gap():
    tracker = TargetTracker()
    p = person(250)
    for stamp in (1, 1.016):
        tracker.choose([p], 480, 480, .21, stamp)
    assert tracker.confirmed
    assert tracker.choose([], 480, 480, .21, 1.032) is None
    tracker.choose([p], 480, 480, .21, 1.048)
    assert not tracker.confirmed
    tracker.choose([p], 480, 480, .21, 1.064)
    assert tracker.confirmed


def test_circular_zone_rejects_crop_corners():
    tracker = TargetTracker()
    assert tracker.choose([person(0, 0)], 480, 480, .21, 1) is None


def test_prediction_never_exceeds_the_box_relative_lead_cap():
    tracker = TargetTracker()
    tracker.choose([person(240)], 480, 480, .21, 1)
    result = tracker.choose([person(250)], 480, 480, .21, 1.016, .035)
    assert result
    assert abs(result[1][0]-tracker.last[0]) <= 6.000001


def test_deadzone_has_no_residual_momentum():
    movement = MovementFilter()
    movement.compute(100, -100, .65, 32)
    assert movement.compute(0, 0, .65, 32) == (0, 0)
    assert movement.compute(1, -1, .65, 32) == (0, 0)


def test_reversing_direction_does_not_keep_old_momentum():
    movement = MovementFilter()
    assert movement.compute(100, 0, .6, 32)[0] > 0
    assert movement.compute(-20, 0, .6, 32)[0] < 0


@pytest.mark.parametrize('dt', [float('nan'), 0, -.01, .5])
def test_invalid_or_stale_timestep_emits_no_input(dt):
    assert MovementFilter().compute(100, 100, .6, 32, dt=dt) == (0, 0)


def test_response_is_similar_across_inference_rates():
    # Idealized plant: one mouse count = one screen pixel. Not game calibration.
    remaining = []
    for fps in (30, 60, 120):
        controller = MovementFilter()
        error = 150
        for _ in range(round(.1*fps)):
            dx, _ = controller.compute(error, 0, .3, 32, 0, 1/fps)
            error -= dx
        remaining.append(error)
    assert max(remaining)-min(remaining) <= 3, remaining


def test_nonfinite_model_coordinates_are_discarded_before_clipping():
    raw = np.array([[float('inf'), 10, 100, 200, .9, 0],
                    [10, 10, 100, 200, .9, 0]], dtype=np.float32)
    result = parse_yolo_people(raw, Letterbox(1, 0, 0, 480, 480))
    assert len(result) == 1
    assert result[0].x1 == 10


BASE_GUARD = dict(armed=True, confirmed=True, same_window=True, rmb=True,
                  f8=False, stopping=False, frame_age=.02, same_generation=True)

@pytest.mark.parametrize('field,value', [('armed', False), ('confirmed', False),
    ('same_window', False), ('rmb', False), ('f8', True), ('stopping', True),
    ('frame_age', .121), ('frame_age', float('nan')), ('frame_age', -.01),
    ('same_generation', False)])
def test_guard_rejects_each_unsafe_state(field, value):
    assert input_permitted(**BASE_GUARD)
    assert not input_permitted(**dict(BASE_GUARD, **{field: value}))


def test_settings_round_trip_excludes_arm_model_and_overlay(tmp_path):
    path = tmp_path/'settings.json'
    original = Options(strength=.6, overlay=True, model='temporary-model.onnx')
    save_settings(original, path)
    raw = json.loads(path.read_text())
    assert not {'model', 'overlay', 'armed', 'input_enabled'} & raw.keys()
    loaded = load_settings(path)
    assert loaded.strength == .6 and loaded.model == '' and loaded.overlay is False

@pytest.mark.parametrize('text', ['not json', '[]', '{"strength": 3}', '{"confidence": NaN}'])
def test_corrupt_preferences_fail_closed_to_defaults(tmp_path, text):
    path = tmp_path/'settings.json'
    path.write_text(text)
    assert load_settings(path) == Options()


def test_engine_refuses_restart_while_old_worker_alive():
    from screensight.engine import Engine
    engine = Engine()
    engine._thread = SimpleNamespace(is_alive=lambda: True)
    engine._stop.set()
    assert engine.start() is False
    assert engine._stop.is_set()


@pytest.mark.parametrize('change,expected', [('none', 1), ('release', 0), ('focus', 0), ('f8', 0), ('rearm', 0)])
def test_engine_rechecks_input_after_inference(monkeypatch, tmp_path, change, expected):
    import screensight.engine as module
    from screensight.windows import WindowArea
    state = {'rmb': True, 'f8': False, 'hwnd': 10}
    calls = []
    engine = module.Engine()
    engine.configure(model=str(tmp_path/'mock.onnx'))
    engine.set_control(True)
    class OneIteration:
        stopped = False
        def is_set(self): return self.stopped
        def wait(self, timeout): self.stopped = True
    engine._stop = OneIteration()
    class Capture:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def grab(self, area): return np.zeros((area['height'], area['width'], 4), np.uint8)
    class Detector:
        provider = 'TEST'
        def __init__(self, model): pass
        def detect(self, frame, threshold):
            if change == 'release': state['rmb'] = False
            if change == 'focus': state['hwnd'] = 11
            if change == 'f8': state['f8'] = True
            if change == 'rearm':
                engine.set_control(False)
                engine.set_control(True)
            return [person(320)], .01
    class Tracker:
        confirmed = True
        track_id = 1
        state = 'LOCKED'
        def reset(self): pass
        def choose(self, *args): return person(320), (335, 211)
    monkeypatch.setitem(sys.modules, 'mss', SimpleNamespace(mss=Capture))
    monkeypatch.setattr(module, 'PersonDetector', Detector)
    monkeypatch.setattr(module, 'TargetTracker', Tracker)
    monkeypatch.setattr(module, 'foreground_game', lambda filters: WindowArea(0, 0, 1000, 800, 'Call of Duty', state['hwnd']))
    monkeypatch.setattr(module, 'key_down', lambda key: state['f8'] if key == module.VK_F8 else state['rmb'])
    monkeypatch.setattr(module, 'move_mouse', lambda x, y: calls.append((x, y)) or True)
    engine._work()
    assert len(calls) == expected

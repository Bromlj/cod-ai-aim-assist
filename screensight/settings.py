"""Validated local preferences; never persist the armed state or a temp model path."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = '0.2.0'

@dataclass(frozen=True)
class Options:
    model: str = ''
    window_titles: str = 'Call of Duty|Black Ops 7'
    roi: int = 480
    confidence: float = .45
    aim_height: float = .21
    strength: float = .42
    max_step: int = 24
    deadzone: float = 2.0
    lead_ms: float = 0.0
    fps_limit: int = 90
    preview: bool = True
    overlay: bool = False

    def __post_init__(self) -> None:
        ranges = {'roi': (320, 720), 'confidence': (.2, .85), 'aim_height': (.1, .65),
                  'strength': (.05, .65), 'max_step': (4, 32), 'deadzone': (0, 8),
                  'lead_ms': (0, 35), 'fps_limit': (20, 120)}
        for key, (lo, hi) in ranges.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be between {lo} and {hi}')
        for key in ('roi', 'max_step', 'fps_limit'):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key} must be an integer')
        if not isinstance(self.window_titles, str) or len(self.window_titles) > 512:
            raise ValueError('Invalid window filter')
        if not isinstance(self.model, str) or len(self.model) > 4096:
            raise ValueError('Invalid model path')
        if type(self.preview) is not bool or type(self.overlay) is not bool:
            raise ValueError('Invalid checkbox preference')

PRESETS = {
    'Balanced': dict(confidence=.45, strength=.42, roi=480, max_step=24, deadzone=2., lead_ms=0.),
    'Responsive': dict(confidence=.45, strength=.60, roi=600, max_step=32, deadzone=2., lead_ms=0.),
    'Precise': dict(confidence=.55, strength=.28, roi=400, max_step=16, deadzone=3., lead_ms=0.),
}

def settings_path() -> Path:
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'ScreenSight' / 'settings.json'

def load_settings(path: Path | None = None) -> Options:
    try:
        p = path or settings_path()
        if p.stat().st_size > 32768:
            return Options()
        raw = json.loads(p.read_text(encoding='utf-8'))
        allowed = {f.name for f in fields(Options)} - {'model', 'overlay'}
        return Options(**{k: v for k, v in raw.items() if k in allowed})
    except (OSError, ValueError, TypeError, AttributeError):
        return Options()

def save_settings(options: Options, path: Path | None = None) -> None:
    p = path or settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(options)
    data.pop('model')
    data.pop('overlay')  # Known compatibility risk; always OFF after restart.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=p.parent, encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
        os.replace(temporary, p)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def input_permitted(*, armed: bool, confirmed: bool, same_window: bool,
                    rmb: bool, f8: bool, stopping: bool, frame_age: float,
                    same_generation: bool = True) -> bool:
    """Rechecked after inference, immediately before Windows input is emitted."""
    return bool(armed and confirmed and same_window and rmb and not f8 and
                not stopping and same_generation and math.isfinite(frame_age) and
                0 <= frame_age <= .12)

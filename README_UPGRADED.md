# COD AI Aim Assist — BEST VERSION (Capture Card / Remote Play + Controller)

This is the upgraded, professional-grade version supporting:

- **Local screen**
- **HDMI Capture Card** (best low-latency for consoles)
- **Chiaki-ng** (PS5 Remote Play)
- **Xbox Remote Play / Cloud Gaming** (window capture)

**Controller support with visual overlay** — physical controller remains fully usable. AI only assists aim on the virtual pad.

## Quick Setup

```powershell
# In admin PowerShell
cd "C:\Users\maor\Desktop\ai tool"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Critical for Controller Mode:**
1. `pip install vgamepad` will trigger ViGEmBus driver install — accept and finish as admin.
2. Restart PC after driver install the first time.
3. In your remote play client (Chiaki or Xbox), select the **virtual Xbox 360 controller** as input device.

## Configuration (config.py)

Key settings:

```python
CAPTURE_MODE = "screen"          # "screen" | "video" | "window"
VIDEO_DEVICE_INDEX = 0           # For capture card
WINDOW_TITLE_CONTAINS = "Chiaki" # or "Xbox", "Remote Play"

INPUT_MODE = "controller"        # "controller" (recommended) or "mouse"
CONTROLLER_AIM_STRENGTH = 0.55   # 0.3 subtle — 0.7 strong

SHOW_CONTROLLER_VISUAL = True
```

Run `python -c "from config import list_video_devices; list_video_devices()"` to find your capture card index.

## How to Use Different Sources

### 1. Capture Card (Recommended for consoles)
- Connect console HDMI → Capture Card → PC
- Set `CAPTURE_MODE = "video"`
- Set correct `VIDEO_DEVICE_INDEX`
- Play on console. AI sees the feed on PC.
- For input: Either use controller passthrough hardware or (easier) switch to remote play.

### 2. Chiaki (PS5)
- Run Chiaki-ng (best quality remote play).
- Make it windowed.
- Set `CAPTURE_MODE = "window"` and `WINDOW_TITLE_CONTAINS = "Chiaki"`
- AI captures the Chiaki window directly.
- Use virtual controller mode.

### 3. Xbox Remote Play / Cloud
- Use xbox.com/remoteplay or Xbox app.
- Set window title match accordingly (try "Xbox" or the browser title).
- Same as Chiaki.

**Note on latency**: Remote play adds 30-80ms+. Capture card + local console play is lower latency for vision but input return path is harder without extra hardware.

## Controller Visual

When `SHOW_CONTROLLER_VISUAL = True`:
- A small always-on-top overlay appears.
- Orange = your physical stick input.
- Cyan = AI correction being added.
- You keep full control of movement on physical controller.
- Game sees the merged virtual controller.

This is the key feature that "won't block my access to the controller".

## Best Model Recommendations (2026)

Current included model is good for starting.

**For the absolute best performance/accuracy**:
1. Go to Roboflow Universe → search "warzone", "black ops 6", "mw3", "call of duty".
2. Choose the largest/most recent dataset.
3. Export as YOLOv11 format.
4. Fine-tune YOLO11n or YOLO11s using the training scripts in `yolo-training/`.
5. Export to TensorRT:
   ```python
   from ultralytics import YOLO
   YOLO("best.pt").export(format="engine", imgsz=320, half=True, device=0)
   ```
6. Put `best.engine` in `models/` and update `config.MODEL_PATH`.

The included `CallOfDutyBO7Ghostaim.onnx` can also be loaded (ultralytics can sometimes handle it) or converted.

## Running

```powershell
# As Administrator
.\.venv\Scripts\Activate.ps1
python main.py
```

Hold **Right Mouse Button** to activate.

## Troubleshooting

- **No video from capture card**: Wrong index. Use the list helper. Try without `CAP_DSHOW`.
- **Chiaki/Xbox window not captured**: Make sure it's not fullscreen. Title match is case-insensitive substring.
- **Virtual controller not appearing**: Install ViGEmBus properly (re-run pip or download from nefarius/ViGEmBus).
- **Physical controller not read**: Make sure pygame sees it (`pygame.joystick.get_count()`). Try different USB port / wired controller.
- **High latency**: Use threaded capture (on by default), smaller ROI, TensorRT engine, lower stream quality on remote play if needed.
- **AI too strong/weak**: Tune `CONTROLLER_AIM_STRENGTH` and `SMOOTH_FACTOR`.

## Architecture Highlights (Best of the Best)

- Threaded capture grabber (latest frame only, minimal buffering).
- Virtual gamepad with physical + AI merge.
- Clean modular capture sources.
- Real-time on-screen controller visualization.
- Same high-quality human-like smoothing as before.
- Optimized for 144+ FPS even with remote sources.

This is currently one of the most complete and flexible public AI aim assist codebases for COD that supports both PC and console streaming/capture setups without completely hijacking your controller.

Enjoy responsibly (private matches, research, etc.).

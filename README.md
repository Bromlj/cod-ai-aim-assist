# COD AI Aim Assist (Real-time YOLO + TensorRT)

High-performance, low-latency AI aim assist for Call of Duty.

**Core features**
- Captures tiny ROI around crosshair (screen center)
- TensorRT YOLO (or CUDA .pt) inference
- Smooth, human-like head tracking
- **Only active while holding Right Mouse Button** (RMB / ADS)
- Designed for 144–300+ FPS

## Project files
- `main.py` – Main loop
- `config.py` – All tunables (ROI size, smoothing, model path, etc.)
- `capture.py` – Fast mss ROI capture
- `inference.py` – YOLO wrapper + target selection + head estimation
- `aiming.py` – Smoothing & humanization logic
- `input_emulation.py` – RMB detection + SendInput mouse moves
- `activation.py` – Activation state machine
- `models/` – Put your `best.engine` or `best.pt` here

## 1. Installation

```powershell
cd "C:\Users\maor\Desktop\ai tool"

# Create venv (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt
```

**For maximum performance (TensorRT):**
1. Have a recent NVIDIA GPU + latest Game Ready driver.
2. Install CUDA Toolkit + cuDNN (matching versions).
3. `pip install tensorrt` (or download the correct wheel from NVIDIA).
4. See export instructions below.

## 2. Get / Prepare a Model

### Recommended public sources (2025-2026)
- **Roboflow Universe** – Search "warzone", "black ops 6", "mw3", "call of duty"
  - Larger datasets perform better: look for 5k–14k+ images (e.g. "warzone-detection" projects).
  - Example: https://universe.roboflow.com (filter by recent COD titles)
- Small public BO6 models exist (jerome, BO6 AI Training) but are limited (~164 images).
- The project already contains a small trained model:
  - `yolo-training/model/cod_yolo11n.pt` (trained on MW2 body data, 1 class).

**Best practice:** Collect your own recent screenshots from the current season/patch and fine-tune. Game visuals change often.

### Convert to TensorRT .engine (strongly recommended)
```python
from ultralytics import YOLO

model = YOLO("yolo-training/model/cod_yolo11n.pt")   # or your downloaded .pt
model.export(
    format="engine",
    imgsz=320,        # or 288 / 256 for more speed
    half=True,
    device=0,
    # workspace=4,    # lower if you get OOM during export
)
# Produces cod_yolo11n.engine (or best.engine)
```

Move the `.engine` into `models/best.engine`.

You can also start with a `.pt` file for testing (slower but easier).

Update `config.py` → `MODEL_PATH` or place it in one of the fallback locations.

## 3. Configuration (config.py)

Key parameters to tune:
- `ROI_WIDTH / ROI_HEIGHT`: 280–400 is typical sweet spot.
- `IMGSZ`: Should be close to ROI size (320 recommended).
- `CONF`: 0.30–0.55. Higher = fewer false triggers.
- `HEAD_OFFSET`: 0.12–0.25. Lower = higher on the body (head).
- `SMOOTH_FACTOR` / `SMOOTH_FACTOR_Y`: 0.30–0.55 range feels human.
- `MAX_MOVE_X/Y`: 20–35 prevents snaps.

## 4. Running

```powershell
# As Administrator (very important for input reliability)
# Right-click PowerShell → Run as administrator, then:
cd "C:\Users\maor\Desktop\ai tool"
.\.venv\Scripts\Activate.ps1
python main.py
```

**In game:**
- Use **Borderless Windowed** or **Fullscreen Windowed**.
- Disable mouse acceleration / raw input if the assist feels off.
- Hold **Right Mouse Button** (aim/ADS) → assist activates.
- Crosshair must be roughly in center of monitor.

## 5. Testing Steps (without getting banned)

1. **Dry run (no game):** Run `python main.py` and hold RMB on desktop. Watch console for FPS.
   - You should see high average FPS (200+ possible on good GPU).
2. **Capture test:** Temporarily set `DEBUG=True` or add prints. Or run `python capture.py`.
3. **Inference test:** `python inference.py` (will run on dummy data).
4. **Input test:** `python input_emulation.py` – should move mouse slightly.
5. **In game (safe):** 
   - Private match / vs bots / firing range.
   - Lower your confidence / smoothing first.
   - Watch if it tracks heads when you hold RMB.
6. Measure real FPS in console output.

## 6. Troubleshooting

| Problem                        | Likely Cause / Fix |
|--------------------------------|--------------------|
| Low FPS (<80)                  | Using .pt instead of .engine. ROI too big. No CUDA. Update to TensorRT. |
| No detections                  | Wrong class index (see `ENEMY_CLASS`). Low CONF. Model trained on old game version. Increase ROI or lower CONF temporarily. |
| Jerky / snapping movement      | SMOOTH_FACTOR too high, MAX_MOVE too high, or no smoothing. Reduce both. |
| Assist works on desktop but not in game | Game is fullscreen exclusive (use borderless). Not running as admin. Mouse acceleration on. |
| Model load error               | TensorRT version mismatch. Re-export on same machine. Use .pt first. |
| Capture is black / wrong area  | Multi-monitor setup (change `sct.monitors[1]`). Overlay / HDR issues. |
| High CPU usage when inactive   | Normal for tight loop. Add sleep when !active (already done). |
| Windows Defender / false positive | Common with input emulation tools. Add folder exclusion. |
| Game detects / bans            | This is against ToS on all modern COD titles. Use at own risk, preferably only on private servers or for research. |

## 7. Performance Tips for 144+ FPS

- Use **TensorRT .engine + half=True**
- ROI 320×320 or smaller + imgsz=320/288
- Nano model (n or 11n)
- No prints / debug inside the main loop
- Game at 1080p or 1440p (4K is heavier on capture)
- Close overlays (Discord, NVIDIA, etc.)

## 8. Disclaimer

This is a technical demonstration / research project.
Using AI aim assist in online multiplayer is against the terms of service of Call of Duty / Activision and will likely result in bans (including hardware bans via Ricochet).

Use responsibly:
- Private matches / local play / singleplayer
- Educational / reverse engineering purposes only

## Next steps / improvements

- Add support for head keypoints (YOLO-pose)
- Kalman filter on target position
- Different behavior for close vs long range
- Auto-adjust smoothing based on enemy velocity
- ONNX Runtime GPU path as fallback

Good luck. Tune carefully and measure everything.

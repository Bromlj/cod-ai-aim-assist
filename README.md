# ScreenSight — local vision lab (experimental)

**A fresh source-code implementation** drawing on architectural ideas found in `Devloop7/cod-ai-aim-assist` and `SunOner/sunone_aimbot_2`. No third-party aimbot Python source or compiled device DLLs have been copied into this codebase.

ScreenSight is a small Windows 10/11 (x64) application for experimenting with computer vision on live screen content. A YOLO11 ONNX model detects **people**, a bounded tracker estimates a target, and an optional Windows SendInput module can make mouse movements **only while right-click is held AND a permitted game window is foreground**. It does not read game memory, install drivers, hook a process, access a network at runtime, use keyloggers, save screenshots, collect credentials, or click/shoot for you.

## Current capabilities

- Live, center-window capture with `mss`; no game process access.
- Local ONNX YOLO11 people detection (GPU via DirectML when available, CPU fallback).
- Position tracking with inertia and short lookahead; centre-biased subject selection.
- Click-through bounding-box overlay, FPS and inference-time stats.
- Explicit **Enable mouse movement** control, *off by default*. Also requires ADS/right-click plus foreground window title match. **F8 immediately disables mouse control**.
- Tunable detection confidence, area of interest, aim height, smoothing/strength, and maximum movement per frame.
- Unbundled custom `.onnx` file option; model change requires stop and restart.

### IMPORTANT LIMITATIONS

The included model is a generic COCO `person` detector, **not trained to distinguish Black Ops 7 enemies from teammates**. It may miss stylised targets, track friendly players, move inconsistently or not work in-game at all. We have not tested the executable in Black Ops 7; Windows synthetic input may be blocked. It is not designed to evade anti-cheat controls. Use of aim assistance may breach game rules and lead to account restrictions. Source review and tests cannot guarantee a program is malware-free.

## Download and run

1. On the repository's **Actions → Build ScreenSight Windows EXE**, open a successful Windows build.
2. Download the **ScreenSight-Windows-x64** artifact and unzip it.
3. Launch `ScreenSight.exe` as a regular user. No administrator rights are required or recommended.
4. Click **START VISION**. The default mode is display-only; it will look at a foreground window containing “Call of Duty” or “Black Ops 7”.
5. If permitted for your controlled experiment, **Enable mouse movement**; it operates **only while holding right-click** with a matching foreground window.
6. Press **F8** for emergency mouse-disable, or click **STOP**.

No user account, installer, antivirus exclusion or special mouse driver is needed. Windows SmartScreen can warn on an unsigned hobbyist binary; validate the downloaded SHA-256 hash against the GitHub Actions build log and scan it before launch.

## Reproducible build

The Windows Actions workflow installs versions from `requirements-win.txt`, downloads one pinned ONNX model and confirms the SHA-256 hash, runs Python unit tests and an ONNX smoke test, bundles Python and dependencies with PyInstaller, then performs an EXE self-test and publishes the build as a GitHub artifact. On a Windows workstation with Python 3.11, run the same commands in `.github/workflows/windows-exe.yml`.

The *app never downloads the model*. The CI build fetches it once, verifies it and embeds it into `ScreenSight.exe`. Alternatively, you can bring your own ONNX model via the Browse control. Never load unknown `.pt`/`.pkl` checkpoints just to use ScreenSight.

## Security overview

| Area | Design |
| --- | --- |
| Network | Application code contains no HTTP, socket, webhook or telemetry clients. Build dependencies/model download occur in CI only. |
| Input | User-mode `SendInput` movement; never click or type. No WinRing0, drivers, KMBOX, Razer or GHub DLLs. |
| Permissions | Ordinary Windows user, no UAC prompt, no registry writes or scheduled tasks. |
| Data | In-memory detection results only, no automatic screenshot saving or upload. |
| Model | ONNX (no PyTorch pickle loading), hash-checked against the known bundled model. |
| Safety | Starts read-only, foreground-title guard, RMB requirement, F8 emergency disable, movement step cap. |

### Dependency and license notices

Source application code is MIT-licensed (see `LICENSE`). The bundled YOLO11n model is from [webnn/yolo11n](https://huggingface.co/webnn/yolo11n), itself based on Ultralytics YOLO11 pretrained COCO models; the model is distributed under **AGPL-3.0** according to its model card. Respect those terms when distributing. The model SHA-256 is `7d8fd1717d9d5bbab6986cd134afb620649c7a394303d55b1e09fc00804cc5c1`.

The original upstream aimbot repositories remain available separately; this replacement does not redistribute their binary artifacts.

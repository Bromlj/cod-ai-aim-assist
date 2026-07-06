"""
main.py - BEST VERSION
Real-time AI Aim Assist for Call of Duty (Console + PC)

Features:
- Multiple capture sources: Screen / HDMI Capture Card / Chiaki Window / Xbox Remote Play
- Mouse OR Virtual Controller output (non-blocking physical controller)
- On-screen controller visual overlay
- Threaded low-latency capture
- Smooth human-like aim assist (only on RMB hold by default)

REQUIREMENTS:
- Run as Administrator
- For controller mode: vgamepad + ViGEmBus
- For capture card: proper device connected and drivers
- Chiaki or Xbox app running in windowed mode for remote play

Usage:
    python main.py
"""

import time
import sys
import traceback
import threading
from typing import Tuple

import numpy as np

from config import CONFIG, get_model_path, list_video_devices
from capture_sources import get_capture, BaseCapture
from inference import YOLOInference
from aiming import AimController
from activation import is_aim_assist_active
from input_emulation import move_mouse_relative
from virtual_controller import VirtualXboxController, pixel_delta_to_stick_delta
from overlay import start_overlay_thread

try:
    import pygame
except ImportError:
    pygame = None


def print_banner():
    print("=" * 68)
    print("   COD AI AIM ASSIST  —  BEST VERSION (Capture Card / Chiaki / Xbox)")
    print("=" * 68)
    print(f"Capture Mode : {CONFIG.CAPTURE_MODE.upper()}")
    if CONFIG.CAPTURE_MODE == "video":
        print(f"  Video Dev  : {CONFIG.VIDEO_DEVICE_INDEX}")
    elif CONFIG.CAPTURE_MODE == "window":
        print(f"  Window     : contains '{CONFIG.WINDOW_TITLE_CONTAINS}'")
    print(f"Model        : {get_model_path()}")
    print(f"Input Mode   : {CONFIG.INPUT_MODE.upper()}")
    print(f"ROI          : {CONFIG.ROI_WIDTH}x{CONFIG.ROI_HEIGHT}")
    print(f"IMGSZ        : {CONFIG.IMGSZ} | Strength: {CONFIG.CONTROLLER_AIM_STRENGTH if CONFIG.INPUT_MODE=='controller' else 'N/A (mouse)'}")
    print(f"Activation   : Hold RMB (or configured key)")
    print(f"Overlay      : {'ON' if CONFIG.SHOW_CONTROLLER_VISUAL else 'OFF'}")
    print("Press Ctrl+C to exit.")
    print("=" * 68)


def get_physical_controller_state(joystick) -> Tuple[float, float, float, float]:
    """Read left and right sticks from physical controller using pygame."""
    if joystick is None:
        return 0.0, 0.0, 0.0, 0.0
    # Axis 0/1 = left, 2/3 = right (typical for Xbox)
    try:
        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        rx = joystick.get_axis(2)
        ry = joystick.get_axis(3)
        return lx, ly, rx, ry
    except:
        return 0.0, 0.0, 0.0, 0.0


def main():
    print_banner()

    if CONFIG.CAPTURE_MODE == "video":
        list_video_devices()

    # === INIT CAPTURE ===
    try:
        capture: BaseCapture = get_capture()
    except Exception as e:
        print(f"[FATAL] Capture init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # === INIT INFERENCE ===
    try:
        inference = YOLOInference()
    except Exception as e:
        print(f"[FATAL] Model load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    aim = AimController()

    # === INIT VIRTUAL CONTROLLER (if needed) ===
    virtual_pad = None
    joystick = None   # physical
    if CONFIG.INPUT_MODE == "controller":
        try:
            virtual_pad = VirtualXboxController()
        except Exception as e:
            print(f"[Warning] Virtual controller failed: {e}")
            print("Falling back to mouse mode or install vgamepad + ViGEmBus.")
            CONFIG.INPUT_MODE = "mouse"

        # Try to init pygame joystick for physical controller reading
        if pygame:
            try:
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    print(f"[Controller] Physical controller detected: {joystick.get_name()}")
                else:
                    print("[Controller] No physical gamepad detected. AI will drive virtual pad alone.")
            except Exception as e:
                print(f"[Controller] Physical joystick init warning: {e}")

    # === CONTROLLER VISUAL OVERLAY ===
    overlay = start_overlay_thread()

    # === MAIN LOOP VARS ===
    frame_count = 0
    last_report = time.perf_counter()
    fps_sum = 0.0
    fps_n = 0
    last_dets = 0

    crosshair_x = crosshair_y = 0   # will be set from capture

    print("\n[Main] Running. Hold RMB to activate assist.")
    print("       For controller mode: Ensure game/remote client is using the VIRTUAL controller.")
    print("       Physical controller remains fully usable for movement etc.\n")

    try:
        while True:
            loop_start = time.perf_counter()

            # Activation (RMB or toggle). Controller button support can be added here.
            active = is_aim_assist_active()

            if not active:
                aim.reset()
                if virtual_pad:
                    # Keep feeding physical input to virtual when inactive (pass-through)
                    if joystick:
                        lx, ly, rx, ry = get_physical_controller_state(joystick)
                        virtual_pad.update_left_stick(lx, ly)
                        virtual_pad.update_right_stick(rx, ry)
                        virtual_pad.update()
                    else:
                        virtual_pad.reset()
                if overlay:
                    overlay.update_state((0,0), (0,0), (0,0), False, last_dets, 0)
                time.sleep(0.004)
                continue

            # === CAPTURE (latest frame) ===
            try:
                frame_rgb, roi_left, roi_top, cx, cy = capture.grab()
                crosshair_x, crosshair_y = cx, cy
            except Exception as cap_err:
                print("[Capture]", cap_err)
                time.sleep(0.005)
                continue

            # === INFERENCE ===
            try:
                detections = inference.infer(frame_rgb)
            except Exception as inf_err:
                print("[Inference]", inf_err)
                time.sleep(0.005)
                continue

            last_dets = len(detections)

            # === PROCESS TARGET ===
            move_x = move_y = 0
            stick_x = stick_y = 0.0
            phys_l = (0.0, 0.0)
            phys_r = (0.0, 0.0)

            if detections:
                best = inference.select_best_target(detections, CONFIG.ROI_WIDTH, CONFIG.ROI_HEIGHT)
                if best:
                    head_x, head_y = inference.get_head_position(best)
                    dx, dy = aim.calculate_target_delta(head_x, head_y, roi_left, roi_top, crosshair_x, crosshair_y)

                    if CONFIG.INPUT_MODE == "controller":
                        # Controller correction (normalized)
                        stick_x, stick_y = aim.compute_controller_correction(dx, dy, CONFIG.ROI_WIDTH)

                        # Merge with physical if available
                        if joystick:
                            plx, ply, prx, pry = get_physical_controller_state(joystick)
                            phys_l = (plx, ply)
                            phys_r = (prx, pry)

                            # Apply AI assist on top of physical right stick
                            final_rx = max(min(prx + stick_x, 1.0), -1.0)
                            final_ry = max(min(pry + stick_y, 1.0), -1.0)

                            if virtual_pad:
                                virtual_pad.update_left_stick(plx, ply)
                                virtual_pad.update_right_stick(final_rx, final_ry)
                                virtual_pad.update()
                        else:
                            # Pure AI drive on virtual
                            if virtual_pad:
                                virtual_pad.update_right_stick(stick_x, stick_y)
                                virtual_pad.update()

                    else:
                        # Mouse mode
                        move_x, move_y = aim.compute_move(dx, dy)
                        if move_x or move_y:
                            move_mouse_relative(move_x, move_y)

            # === UPDATE OVERLAY ===
            if overlay:
                # For display: show physical right + the AI delta we added
                overlay.update_state(
                    phys_l,
                    phys_r if joystick else (stick_x * 0.6, stick_y * 0.6),  # approximate
                    (stick_x, stick_y),
                    active,
                    last_dets,
                    (fps_sum / max(fps_n, 1)) if fps_n > 0 else 0
                )

            # === FPS ===
            frame_count += 1
            dt = time.perf_counter() - loop_start
            fps_sum += (1.0 / dt) if dt > 0 else 0
            fps_n += 1

            if frame_count % CONFIG.FPS_REPORT_INTERVAL == 0:
                avg_fps = fps_sum / max(fps_n, 1)
                print(f"[FPS] {avg_fps:.1f} | Dets: {last_dets} | Mode: {CONFIG.INPUT_MODE} | Capture: {CONFIG.CAPTURE_MODE}")
                fps_sum = 0.0
                fps_n = 0
                last_report = time.perf_counter()

            # Optional micro-sleep
            # time.sleep(max(0, 0.001 - dt))

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    except Exception as e:
        print(f"[Main ERROR] {e}")
        traceback.print_exc()
    finally:
        capture.close()
        if virtual_pad:
            virtual_pad.close()
        if overlay:
            overlay.stop()
        if pygame:
            try:
                pygame.quit()
            except:
                pass
        print("[Main] Clean exit.")


if __name__ == "__main__":
    main()

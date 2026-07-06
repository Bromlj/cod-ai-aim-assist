"""
virtual_controller.py
Best virtual gamepad support using vgamepad (ViGEmBus).

- Creates a virtual Xbox 360 controller.
- Allows merging physical controller input + AI aim correction.
- Does NOT block your physical controller (you read physical separately).
- Game / remote play client (Chiaki, Xbox) should see the virtual pad.

Install ViGEmBus driver (happens automatically on first `pip install vgamepad` - run as admin).
"""

from typing import Tuple, Optional
import time

try:
    import vgamepad as vg
except ImportError:
    vg = None

from config import CONFIG


class VirtualXboxController:
    def __init__(self):
        if vg is None:
            raise RuntimeError(
                "vgamepad is required for controller mode.\n"
                "pip install vgamepad\n"
                "Then (as admin) let it install the ViGEmBus driver."
            )
        self.pad = vg.VX360Gamepad()
        print("[VirtualController] Virtual Xbox 360 pad created. "
              "Make sure your remote play / game is using this controller.")
        # Small delay for OS to register the device
        time.sleep(0.4)

    def update_right_stick(self, x: float, y: float):
        """
        Set right stick. Values in -1.0 ... +1.0
        x positive = right, y positive = down (typical gamepad convention)
        """
        x = max(min(x, 1.0), -1.0)
        y = max(min(y, 1.0), -1.0)
        self.pad.right_joystick_float(x, y)

    def update_left_stick(self, x: float, y: float):
        x = max(min(x, 1.0), -1.0)
        y = max(min(y, 1.0), -1.0)
        self.pad.left_joystick_float(x, y)

    def set_button(self, button, pressed: bool):
        # For future expansion. Common: vg.XUSB_BUTTON.XUSB_GAMEPAD_A etc.
        pass

    def update(self):
        """Send the current state to the virtual device."""
        self.pad.update()

    def reset(self):
        self.pad.reset()
        self.update()

    def close(self):
        try:
            self.reset()
        except:
            pass


def pixel_delta_to_stick_delta(dx: float, dy: float, frame_width: int, strength: float = None) -> Tuple[float, float]:
    """
    Convert pixel error (from crosshair to target head) into normalized right-stick correction.
    This is tuned for "assist" feel, not full aimbot.
    """
    if strength is None:
        strength = CONFIG.CONTROLLER_AIM_STRENGTH

    # Normalize by half the ROI/frame dimension
    norm_x = dx / (frame_width * 0.5)
    norm_y = dy / (frame_width * 0.5)   # use width for both to keep aspect

    sx = norm_x * strength
    sy = norm_y * strength

    # Clamp
    max_d = CONFIG.CONTROLLER_MAX_DELTA
    sx = max(min(sx, max_d), -max_d)
    sy = max(min(sy, max_d), -max_d)

    # Apply deadzone to result
    dz = CONFIG.CONTROLLER_DEADZONE
    if abs(sx) < dz: sx = 0.0
    if abs(sy) < dz: sy = 0.0

    return sx, sy


# Convenience for direct use
if __name__ == "__main__":
    print("Testing virtual controller (will create device)...")
    vc = VirtualXboxController()
    print("Moving right stick slowly for 3 seconds...")
    for i in range(30):
        vc.update_right_stick(0.3 * (i % 3 - 1), 0.0)
        vc.update()
        time.sleep(0.1)
    vc.reset()
    print("Done. Check your game controller list.")

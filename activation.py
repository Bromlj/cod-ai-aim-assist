"""
activation.py
Handles activation logic for the aim assist.

Current design:
- Primary: Hold Right Mouse Button (RMB) to enable.
- Optional: Toggle key (F2 by default in config).

Keeps state for toggle mode.
"""

from typing import Optional
import time

from config import CONFIG
from input_emulation import is_rmb_down, is_key_down


class ActivationManager:
    def __init__(self):
        self.toggle_state = False
        self.last_toggle_press = 0.0
        self.toggle_vk = CONFIG.TOGGLE_VK
        self.activation_vk = CONFIG.ACTIVATION_VK
        self._debounce = 0.18  # seconds between toggle flips

    def is_active(self) -> bool:
        """
        Returns True if the aim assist should be active this frame.
        RMB hold is primary. Toggle supported. Controller button handled in main if desired.
        """
        if is_rmb_down():
            return True

        if self.toggle_vk is not None:
            if is_key_down(self.toggle_vk):
                now = time.time()
                if now - self.last_toggle_press > self._debounce:
                    self.toggle_state = not self.toggle_state
                    self.last_toggle_press = now
                    print(f"[Activation] Toggle -> {'ON' if self.toggle_state else 'OFF'}")
            if self.toggle_state:
                return True
        return False

    def force_off(self):
        self.toggle_state = False


# Simple functional API if you don't want the class
_activation = ActivationManager()


def is_aim_assist_active() -> bool:
    return _activation.is_active()


if __name__ == "__main__":
    act = ActivationManager()
    print("Activation test. Hold RMB or press F2 (toggle). Ctrl+C to exit.")
    try:
        while True:
            print(f"Active: {act.is_active()}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone")

"""
input_emulation.py
Low-level Windows input for:
- Checking if Right Mouse Button (or other keys) is held
- Relative mouse movement using SendInput (most reliable for games)

No external input libraries needed (uses ctypes + win32).
Run the script as Administrator for best results.
"""

import ctypes
from ctypes import wintypes
import time

# Virtual key for Right Mouse Button
VK_RBUTTON = 0x02

# SendInput structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]
    _anonymous_ = ("_i",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_i", _I),
    ]


# Constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000


_user32 = ctypes.windll.user32


def is_rmb_down() -> bool:
    """Return True if Right Mouse Button is currently held."""
    state = _user32.GetAsyncKeyState(VK_RBUTTON)
    return bool(state & 0x8000)


def is_key_down(vk_code: int) -> bool:
    """Generic key check. Pass virtual key codes (e.g. 0x71 for F2)."""
    if vk_code is None:
        return False
    state = _user32.GetAsyncKeyState(vk_code)
    return bool(state & 0x8000)


def move_mouse_relative(dx: int, dy: int) -> None:
    """Move the mouse by relative pixels using SendInput. Most game-compatible method."""
    if dx == 0 and dy == 0:
        return

    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(
        dx=int(dx),
        dy=int(dy),
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE,
        time=0,
        dwExtraInfo=ctypes.pointer(extra)
    )
    inp = INPUT(type=INPUT_MOUSE, mi=mi)

    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# Optional: absolute move (rarely needed for aim assist)
def move_mouse_absolute(screen_x: int, screen_y: int, screen_width: int = 1920, screen_height: int = 1080) -> None:
    """Move to absolute screen position (0-65535 normalized). Use sparingly."""
    # Convert to absolute 0-65535 range
    abs_x = int(screen_x * 65535 / screen_width)
    abs_y = int(screen_y * 65535 / screen_height)

    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
        time=0,
        dwExtraInfo=ctypes.pointer(extra)
    )
    inp = INPUT(type=INPUT_MOUSE, mi=mi)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# Test
if __name__ == "__main__":
    print("Testing input (move mouse slightly + RMB detection). Press RMB or wait 3s...")
    for i in range(6):
        rmb = is_rmb_down()
        print(f"  RMB held: {rmb}")
        move_mouse_relative(3, 1)
        time.sleep(0.5)
    print("Input test done.")

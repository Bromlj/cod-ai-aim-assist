"""Narrow, auditable Windows API surface. No driver, process injection, or hooks."""
from __future__ import annotations
import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
import unicodedata

VK_RBUTTON = 0x02
VK_F8 = 0x77


@dataclass(frozen=True)
class WindowArea:
    left: int
    top: int
    width: int
    height: int
    title: str
    hwnd: int = 0


def key_down(key: int) -> bool:
    return sys.platform == 'win32' and bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000)


def normalized_window_title(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text)
    return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Cf').casefold()


def window_title_matches(title: str, title_filters: tuple[str, ...]) -> bool:
    candidate = normalized_window_title(title)
    return bool(candidate and any(pattern in candidate for item in title_filters
                if (pattern := normalized_window_title(item).strip())))


def foreground_game(title_filters: tuple[str, ...]) -> WindowArea | None:
    if sys.platform != 'win32':
        return None
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    u.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    u.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    hwnd = u.GetForegroundWindow()
    if not hwnd:
        return None
    buffer = ctypes.create_unicode_buffer(512)
    u.GetWindowTextW(hwnd, buffer, len(buffer))
    title = buffer.value
    if not window_title_matches(title, title_filters):
        return None
    rect = wintypes.RECT()
    if not u.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not u.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 240 or height < 240:
        return None
    return WindowArea(origin.x, origin.y, width, height, title, int(hwnd))


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG),
                ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
                ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD), ('wParamH', wintypes.WORD)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [('type', wintypes.DWORD), ('data', INPUT_UNION)]


def move_mouse(dx: int, dy: int) -> bool:
    """Plain SendInput: no elevated privileges, anti-cheat bypass, or kernel driver."""
    if sys.platform != 'win32' or (not dx and not dy):
        return False
    event = INPUT(type=0, data=INPUT_UNION(mi=MOUSEINPUT(int(dx), int(dy), 0, 0x0001, 0, 0)))
    u = ctypes.windll.user32
    u.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    u.SendInput.restype = wintypes.UINT
    return u.SendInput(1, ctypes.byref(event), ctypes.sizeof(event)) == 1


def make_overlay_clickthrough(hwnd: int) -> None:
    """Ignore clicks and exclude only *our own* overlay from capture if supported."""
    if sys.platform != 'win32':
        return
    u = ctypes.windll.user32
    # Tk's widget ID can be the child, not the native top-level wrapper.
    u.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    u.GetAncestor.restype = wintypes.HWND
    hwnd = u.GetAncestor(hwnd, 2) or hwnd
    u.SetWindowDisplayAffinity.argtypes = (wintypes.HWND, wintypes.DWORD)
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT = 0x20
    WS_EX_LAYERED = 0x80000
    WS_EX_TOOLWINDOW = 0x80
    WS_EX_NOACTIVATE = 0x08000000
    try:
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            u.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
            u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            u.SetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
            old = u.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, old | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        else:
            u.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
            u.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.LONG)
            old = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, old | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        # Prevent our overlay from becoming feedback in our own screen capture.
        # Fails harmlessly on older Windows builds or unsupported configurations.
        u.SetWindowDisplayAffinity(hwnd, 0x11)
    except (OSError, AttributeError):
        pass

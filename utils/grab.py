import ctypes
import numpy as np
import mss

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_sct = mss.MSS()
_monitor_index = None


def set_monitor(index):
    global _monitor_index
    _monitor_index = index


def active_monitor():
    mons = _sct.monitors[1:]
    if _monitor_index is not None and 0 <= _monitor_index < len(mons):
        return mons[_monitor_index]
    for mon in mons:
        if mon.get("is_primary"):
            return mon
    return mons[0]


def list_monitors():
    return [{"index": i, "left": m["left"], "top": m["top"],
             "width": m["width"], "height": m["height"],
             "primary": bool(m.get("is_primary"))}
            for i, m in enumerate(_sct.monitors[1:])]


def screen(region=None):
    mon = active_monitor()
    if region:
        left, top, x2, y2 = region
        monitor = {"left": mon["left"] + left, "top": mon["top"] + top,
                   "width": x2 - left, "height": y2 - top}
    else:
        monitor = mon

    shot = _sct.grab(monitor)
    img = np.array(shot)
    return img[:, :, :3]


def get_screen_size():
    mon = active_monitor()
    return mon["width"], mon["height"]


def get_monitor_origin():
    mon = active_monitor()
    return mon["left"], mon["top"]


def get_virtual_size():
    v = _sct.monitors[0]
    return v["width"], v["height"]


def close():
    _sct.close()

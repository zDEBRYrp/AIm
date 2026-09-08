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


def screen(region=None):
    if region:
        left, top, x2, y2 = region
        monitor = {"left": left, "top": top, "width": x2 - left, "height": y2 - top}
    else:
        monitor = _sct.monitors[1]

    shot = _sct.grab(monitor)
    img = np.array(shot)
    return img[:, :, :3]


def get_screen_size():
    mon = _sct.monitors[1]
    return mon["width"], mon["height"]


def close():
    _sct.close()

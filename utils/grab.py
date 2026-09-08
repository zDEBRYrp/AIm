import numpy as np
import mss


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

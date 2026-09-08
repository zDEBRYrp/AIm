import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import subprocess
import sys
import os

subprocess.call([sys.executable, "-m", "pip", "install", "-r", "assets/requirements.txt"])

os.system("cls" if os.name == "nt" else "clear")

print("\033[1;36m" + r"""
 █████╗ ██╗███╗   ███╗
██╔══██╗██║████╗ ████║
███████║██║██╔████╔██║
██╔══██║██║██║╚██╔╝██║
██║  ██║██║██║ ╚═╝ ██║
╚═╝  ╚═╝╚═╝╚═╝     ╚═╝
     AI + m = AIM
""" + "\033[0m")
print("\033[1;35m[Control] ->\033[1;34m [F1] switch mode: Always On / Hold")
print("\033[1;35m          ->\033[1;34m [Mouse4] aim ONLY in Hold mode (hold to aim)")
print("\033[1;35m          ->\033[1;34m [8] settings (when console focused)")
print("\033[1;35m          ->\033[1;34m [0] Exit")
print("\033[1;33m\n[Tips] Press F1 to set mode\n")

from lib.detect import aimbot

aimbot(ENABLE_AIMBOT=True)

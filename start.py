import subprocess
import sys
import os

subprocess.call([sys.executable, "-m", "pip", "install", "-r", "assets/requirements.txt"])

os.system("cls" if os.name == "nt" else "clear")

print(r"""
    _    ____   _____
   / \  |  _ \ / ____|
  / _ \ | |_) | |
 / ___ \|  _ <| |___
/_/   \_\_| \_\\_____|
   AI + m = AIm
""")
print("\033[1;35m[Control] ->\033[1;34m [F1] Aimbot: Always On/Hold Mode")
print("\033[1;35m          ->\033[1;34m [Mouse4] Hold Mode: Press/Release")
print("\033[1;35m          ->\033[1;34m [0] Exit")
print("\033[1;33m\n[Tips] Press F1 to set mode")
print("\033[1;33m[Config] Edit config.json to change settings\n")

from lib.detect import aimbot

aimbot(ENABLE_AIMBOT=True)

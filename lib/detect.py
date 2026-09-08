import sys
import time
import signal
import ctypes

import cv2
import numpy as np
import keyboard
import winsound
import pynput._util.win32 as _pynput_win32
from pynput.mouse import Listener
from ultralytics import YOLO

from utils.grab import screen, get_screen_size, close

SendInput = ctypes.windll.user32.SendInput
Wd, Hd = get_screen_size()


def position(x, y):
    x = 1 + int(x * 65536.0 / Wd)
    y = 1 + int(y * 65536.0 / Hd)
    extra = ctypes.c_ulong(0)
    ii_ = _pynput_win32.INPUT_union()
    ii_.mi = _pynput_win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0,
                                       ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command = _pynput_win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


def aimbot(ENABLE_AIMBOT=True):
    CONFIDENCE = 0.36
    ACTIVATION_RANGE = 250
    aimbot_paused = True

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    cx = Wd // 2
    cy = Hd // 2
    origbox = (
        cx - ACTIVATION_RANGE // 2,
        cy - ACTIVATION_RANGE // 2,
        cx + ACTIVATION_RANGE // 2,
        cy + ACTIVATION_RANGE // 2,
    )

    def signal_handler(sig, frame):
        print("\n[Exit] cleaning up...")
        close()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("\033[1;36m[Status] Loading YOLOv8n model (first run downloads ~6MB)...")
    model = YOLO("yolov8n.pt")
    names = model.names

    if not ENABLE_AIMBOT:
        print("\033[1;91m[Status] Aimbot disabled, only objects detector works...")
    else:
        print("\033[1;92m[AI] Aimbot enabled..")

    def toggle_aimbot():
        nonlocal aimbot_paused
        aimbot_paused = not aimbot_paused
        status = "hold mode" if aimbot_paused else "always on"
        color = RED if aimbot_paused else GREEN
        print("\nAimbot : " + color + status + RESET)
        if not aimbot_paused:
            winsound.Beep(440, 100)

    keyboard.add_hotkey("F1", toggle_aimbot)

    def on_click(x, y, button, pressed):
        if button == button.x2:
            nonlocal aimbot_paused
            aimbot_paused = not aimbot_paused

    prev_time = time.perf_counter()

    with Listener(on_click=on_click) as listener:
        while True:
            if aimbot_paused:
                time.sleep(0.05)
                continue

            frame = np.array(screen(region=origbox))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame, verbose=False, conf=CONFIDENCE)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                best_idx = boxes.conf.argmax()
                best_conf = float(boxes.conf[best_idx])

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    label = names.get(cls, str(cls))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 3), 5, (0, 0, 255), -1)
                    text = f"{label} {int(conf * 100)}%"
                    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if ENABLE_AIMBOT and i == best_idx:
                        mouseX = origbox[0] + (x1 + x2) / 2
                        mouseY = origbox[1] + (y1 + y2) / 3
                        position(mouseX, mouseY)

            cv2.imshow("AIm - Objects Detector", frame)

            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now
            fps = int(1 / elapsed) if elapsed > 0 else 0
            sys.stdout.write(f"\033[1;33m\rFPS: {fps}  MS: {int(elapsed * 1000)}\t")
            sys.stdout.flush()

            if cv2.waitKey(1) & 0xFF == ord("0"):
                break

    signal_handler(0, 0)
    listener.join()

import sys
import os
import time
import signal
import ctypes
import subprocess
import threading

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import numpy as np
import keyboard
import winsound
import onnxruntime as ort
from pynput.mouse import Listener
import pynput._util.win32 as _pynput_win32

from utils.grab import screen, get_screen_size, close, set_monitor, get_monitor_origin, get_virtual_size, list_monitors

SendInput = ctypes.windll.user32.SendInput
Wd, Hd = 0, 0
VWd, VHd = 0, 0
MON_X, MON_Y = 0, 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_MODEL = os.path.join(ROOT_DIR, "models", "yolov8n.onnx")
INPUT_SIZE = 256
PERSON_CLASS = 0


def load_config():
    defaults = {
        "confidence": 0.30,
        "iou_threshold": 0.45,
        "activation_range": 250,
        "aim_speed": 0.8,
        "deadzone": 2,
        "aim_height_ratio": 0.2,
        "hold_button": "x2",
        "toggle_hotkey": "F1",
        "monitor": "primary",
    }
    config_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            user = json.load(f)
        defaults.update({k: v for k, v in user.items() if v is not None})
    return defaults


CFG = load_config()


def _select_monitor():
    global Wd, Hd, VWd, VHd, MON_X, MON_Y
    choice = CFG.get("monitor", "primary")
    if isinstance(choice, int):
        set_monitor(choice)
    for m in list_monitors():
        tag = " [PRIMARY]" if m["primary"] else ""
        print(f"\033[1;36m[Monitor {m['index']}] {m['width']}x{m['height']} at ({m['left']},{m['top']}){tag}")
    Wd, Hd = get_screen_size()
    MON_X, MON_Y = get_monitor_origin()
    VWd, VHd = get_virtual_size()
    print(f"\033[1;32m[Status] Capture: {Wd}x{Hd} at ({MON_X},{MON_Y}); virtual screen {VWd}x{VHd}.")


_select_monitor()


def export_to_onnx():
    if os.path.exists(ONNX_MODEL):
        return
    print("\033[1;36m[Status] First run — downloading YOLOv8n (~6MB) and exporting to ONNX...")
    subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from ultralytics import YOLO
    pt_path = os.path.join(ROOT_DIR, "yolov8n.pt")
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=INPUT_SIZE, half=False)
    exported = os.path.join(ROOT_DIR, "yolov8n.onnx")
    os.makedirs(os.path.dirname(ONNX_MODEL), exist_ok=True)
    os.replace(exported, ONNX_MODEL)
    if os.path.exists(pt_path):
        os.remove(pt_path)
    print("\033[1;32m[Status] ONNX export done.")


def postprocess(output, img_w, img_h, conf_thresh, iou_thresh):
    output = output[0]
    if output.shape[0] == 84:
        output = output.T

    boxes_xywh = output[:, :4].astype(np.float32)
    class_scores = output[:, 4:]

    max_scores = class_scores.max(axis=1)
    mask = max_scores > conf_thresh
    if not mask.any():
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_scores[mask].argmax(axis=1)

    boxes_x1y1x2y2 = np.zeros_like(boxes_xywh)
    boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    scale_x = img_w / INPUT_SIZE
    scale_y = img_h / INPUT_SIZE
    boxes_x1y1x2y2[:, [0, 2]] *= scale_x
    boxes_x1y1x2y2[:, [1, 3]] *= scale_y

    bboxes_for_nms = boxes_x1y1x2y2.tolist()
    scores_for_nms = max_scores.tolist()
    indices = cv2.dnn.NMSBoxes(bboxes_for_nms, scores_for_nms, conf_thresh, iou_thresh)

    if len(indices) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    indices = indices.flatten()
    return boxes_x1y1x2y2[indices], max_scores[indices], class_ids[indices]


def move_relative(dx, dy):
    dx, dy = int(dx), int(dy)
    if dx == 0 and dy == 0:
        return
    extra = ctypes.c_ulong(0)
    ii_ = _pynput_win32.INPUT_union()
    ii_.mi = _pynput_win32.MOUSEINPUT(dx, dy, 0, 0x0001, 0,
                                       ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command = _pynput_win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


class ThreadedCapture:
    def __init__(self, region):
        self.region = region
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self.running:
            frame = np.array(screen(region=self.region))
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)


def aimbot(ENABLE_AIMBOT=True):
    mode = "hold"
    mouse_held = False
    last_tx, last_ty = None, None
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    FULLSCREEN = CFG.get("fullscreen", False)
    ACTIVATION_RANGE = CFG["activation_range"]
    cx = Wd // 2
    cy = Hd // 2

    if FULLSCREEN:
        capture_region = None
        aim_box = (
            cx - ACTIVATION_RANGE // 2,
            cy - ACTIVATION_RANGE // 2,
            cx + ACTIVATION_RANGE // 2,
            cy + ACTIVATION_RANGE // 2,
        )
    else:
        capture_region = (
            cx - ACTIVATION_RANGE // 2,
            cy - ACTIVATION_RANGE // 2,
            cx + ACTIVATION_RANGE // 2,
            cy + ACTIVATION_RANGE // 2,
        )
        aim_box = capture_region

    def signal_handler(sig, frame):
        print("\n[Exit] cleaning up...")
        cap.stop()
        close()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    export_to_onnx()

    print("\033[1;36m[Status] Loading YOLOv8n ONNX model...")
    sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print("\033[1;32m[Status] Model loaded (onnxruntime CPU).")

    if not ENABLE_AIMBOT:
        print("\033[1;91m[Status] Aimbot disabled, only objects detector works...")
    else:
        print("\033[1;92m[AI] Aimbot enabled..")
    print(f"\033[1;33m[Mode] hold — hold {CFG.get('hold_button', 'x2')} to aim, {CFG.get('toggle_hotkey', 'F1')} for always-on.")

    def is_active():
        return mode == "always" or mouse_held

    def toggle_aimbot():
        nonlocal mode, last_tx, last_ty
        mode = "always" if mode == "hold" else "hold"
        last_tx, last_ty = None, None
        if mode == "always":
            print("\nAimbot : " + GREEN + "always on" + RESET)
            winsound.Beep(440, 100)
        else:
            print("\nAimbot : " + RED + "hold mode" + RESET)

    keyboard.add_hotkey(CFG["toggle_hotkey"], toggle_aimbot)

    hold_button = getattr(__import__("pynput.mouse", fromlist=["Button"]), "Button").x2
    button_names = {"x2": "x2", "left": "left", "right": "right"}
    btn_name = CFG.get("hold_button", "x2")
    if btn_name in ("left", "right"):
        from pynput.mouse import Button as _Btn
        hold_button = getattr(_Btn, btn_name)

    def on_click(x, y, button, pressed):
        if button == hold_button:
            nonlocal mouse_held, last_tx, last_ty
            mouse_held = pressed
            if pressed:
                last_tx, last_ty = None, None

    cap = ThreadedCapture(capture_region)
    cap.start()

    WINDOW_NAME = "AIm - Objects Detector"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, ACTIVATION_RANGE, ACTIVATION_RANGE)

    prev_time = time.perf_counter()
    window_moved = False

    with Listener(on_click=on_click) as listener:
        while True:
            if not is_active():
                time.sleep(0.05)
                continue

            frame = cap.read()
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
            blob = blob.astype(np.float32)

            output = sess.run(None, {inp_name: blob})[0]
            boxes, scores, class_ids = postprocess(output, img_w, img_h, CFG["confidence"], CFG["iou_threshold"])

            if len(boxes) > 0:
                aim_h = CFG.get("aim_height_ratio", 0.2)
                targets = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    conf = scores[i]
                    cls = int(class_ids[i])

                    ax = (x1 + x2) // 2
                    ay = int(y1 + (y2 - y1) * aim_h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (ax, ay), 5, (0, 0, 255), -1)
                    label = "person" if cls == PERSON_CLASS else str(cls)
                    text = f"{label} {int(conf * 100)}%"
                    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if cls == PERSON_CLASS:
                        if FULLSCREEN:
                            tx, ty = float(ax), float(ay)
                        else:
                            tx = aim_box[0] + ax
                            ty = aim_box[1] + ay
                        targets.append((tx, ty))

                if ENABLE_AIMBOT and targets:
                    cross_x, cross_y = Wd // 2, Hd // 2
                    if last_tx is not None:
                        sticky = min(targets, key=lambda p: (p[0] - last_tx) ** 2 + (p[1] - last_ty) ** 2)
                        d = ((sticky[0] - last_tx) ** 2 + (sticky[1] - last_ty) ** 2) ** 0.5
                        if d < 150:
                            target = sticky
                        else:
                            target = min(targets, key=lambda p: (p[0] - cross_x) ** 2 + (p[1] - cross_y) ** 2)
                    else:
                        target = min(targets, key=lambda p: (p[0] - cross_x) ** 2 + (p[1] - cross_y) ** 2)
                    last_tx, last_ty = target
                    dx = target[0] - cross_x
                    dy = target[1] - cross_y
                    dz = CFG.get("deadzone", 2)
                    if abs(dx) >= dz or abs(dy) >= dz:
                        spd = CFG.get("aim_speed", 0.8)
                        move_relative(dx * spd, dy * spd)
            else:
                last_tx, last_ty = None, None

            cv2.imshow(WINDOW_NAME, frame)

            if not window_moved and not FULLSCREEN:
                window_moved = True
                try:
                    import win32gui
                    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
                    if hwnd:
                        win_x = (Wd - ACTIVATION_RANGE) // 2
                        win_y = (Hd - ACTIVATION_RANGE) // 2
                        HWND_TOPMOST = -1
                        SWP_NOSIZE = 0x0001
                        SWP_SHOWWINDOW = 0x0040
                        ctypes.windll.user32.SetWindowPos(
                            hwnd, HWND_TOPMOST,
                            win_x, win_y, 0, 0,
                            SWP_NOSIZE | SWP_SHOWWINDOW
                        )
                except Exception:
                    pass

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

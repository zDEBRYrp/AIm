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

from utils.grab import screen, get_screen_size, close

SendInput = ctypes.windll.user32.SendInput
Wd, Hd = get_screen_size()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_MODEL = os.path.join(ROOT_DIR, "models", "yolov8n.onnx")
INPUT_SIZE = 256
PERSON_CLASS = 0


def load_config():
    defaults = {
        "confidence": 0.36,
        "iou_threshold": 0.45,
        "activation_range": 250,
        "smoothing": 0.35,
        "hold_button": "x2",
        "toggle_hotkey": "F1",
    }
    config_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            user = json.load(f)
        defaults.update({k: v for k, v in user.items() if v is not None})
    return defaults


CFG = load_config()


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


def position(x, y):
    x = 1 + int(x * 65536.0 / Wd)
    y = 1 + int(y * 65536.0 / Hd)
    extra = ctypes.c_ulong(0)
    ii_ = _pynput_win32.INPUT_union()
    ii_.mi = _pynput_win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0,
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


class SmoothAim:
    def __init__(self, factor=0.35):
        self.factor = factor
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.initialized = False

    def update(self, target_x, target_y):
        if not self.initialized:
            self.smooth_x = target_x
            self.smooth_y = target_y
            self.initialized = True
            return self.smooth_x, self.smooth_y
        alpha = self.factor
        self.smooth_x += alpha * (target_x - self.smooth_x)
        self.smooth_y += alpha * (target_y - self.smooth_y)
        return self.smooth_x, self.smooth_y

    def reset(self):
        self.initialized = False


def aimbot(ENABLE_AIMBOT=True):
    aimbot_paused = True
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

    def toggle_aimbot():
        nonlocal aimbot_paused
        aimbot_paused = not aimbot_paused
        if aimbot_paused:
            smooth.reset()
            print("\nAimbot : " + RED + "hold mode" + RESET)
        else:
            print("\nAimbot : " + GREEN + "always on" + RESET)
            winsound.Beep(440, 100)

    keyboard.add_hotkey(CFG["toggle_hotkey"], toggle_aimbot)

    hold_button = getattr(__import__("pynput.mouse", fromlist=["Button"]), "Button").x2
    button_names = {"x2": "x2", "left": "left", "right": "right"}
    btn_name = CFG.get("hold_button", "x2")
    if btn_name in ("left", "right"):
        from pynput.mouse import Button as _Btn
        hold_button = getattr(_Btn, btn_name)

    def on_click(x, y, button, pressed):
        if button == hold_button and pressed:
            nonlocal aimbot_paused
            aimbot_paused = not aimbot_paused
            if aimbot_paused:
                smooth.reset()
                print("\nAimbot : " + RED + "hold mode" + RESET)
            else:
                print("\nAimbot : " + GREEN + "always on" + RESET)
                winsound.Beep(440, 100)

    cap = ThreadedCapture(capture_region)
    cap.start()
    smooth = SmoothAim(factor=CFG["smoothing"])

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
            if aimbot_paused:
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
                best_idx = scores.argmax()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    conf = scores[i]
                    cls = int(class_ids[i])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 3), 5, (0, 0, 255), -1)
                    label = "person" if cls == PERSON_CLASS else str(cls)
                    text = f"{label} {int(conf * 100)}%"
                    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if ENABLE_AIMBOT and i == best_idx and cls == PERSON_CLASS:
                        if FULLSCREEN:
                            raw_x = (x1 + x2) / 2
                            raw_y = (y1 + y2) / 3
                        else:
                            raw_x = aim_box[0] + (x1 + x2) / 2
                            raw_y = aim_box[1] + (y1 + y2) / 3
                        sx, sy = smooth.update(raw_x, raw_y)
                        position(sx, sy)

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

import sys
import os
import time
import signal
import ctypes
import subprocess

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

ONNX_MODEL = "models/yolov8n.onnx"
PT_MODEL = "yolov8n.pt"
INPUT_SIZE = 256
CONFIDENCE = 0.36
IOU_THRESHOLD = 0.45
PERSON_CLASS = 0


def export_to_onnx():
    if os.path.exists(ONNX_MODEL):
        return
    if not os.path.exists(PT_MODEL):
        print("\033[1;36m[Status] Downloading YOLOv8n model (~6MB)...")
        subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("\033[1;36m[Status] Exporting model to ONNX...")
    from ultralytics import YOLO
    model = YOLO(PT_MODEL)
    model.export(format="onnx", imgsz=INPUT_SIZE, half=False)
    os.makedirs("models", exist_ok=True)
    src = os.path.join(os.path.dirname(PT_MODEL) or ".", "yolov8n.onnx")
    os.replace(src, ONNX_MODEL)
    if os.path.exists(PT_MODEL):
        os.remove(PT_MODEL)
    print("\033[1;32m[Status] ONNX export done.")


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(output, img_w, img_h, conf_thresh, iou_thresh):
    output = output[0]
    if output.shape[0] == 84:
        output = output.T

    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]

    boxes_x1y1x2y2 = np.zeros_like(boxes_xywh)
    boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    max_scores = class_scores.max(axis=1)
    mask = max_scores > conf_thresh
    boxes_x1y1x2y2 = boxes_x1y1x2y2[mask]
    max_scores = max_scores[mask]
    class_ids = class_scores[mask].argmax(axis=1)

    boxes_x1y1x2y2[:, [0, 2]] *= img_w / INPUT_SIZE
    boxes_x1y1x2y2[:, [1, 3]] *= img_h / INPUT_SIZE

    keep = nms(boxes_x1y1x2y2, max_scores, iou_thresh)
    return boxes_x1y1x2y2[keep], max_scores[keep], class_ids[keep]


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
    aimbot_paused = True
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    ACTIVATION_RANGE = 250
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
            img_h, img_w = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
            blob = blob.astype(np.float32)

            output = sess.run(None, {inp_name: blob})[0]
            boxes, scores, class_ids = postprocess(output, img_w, img_h, CONFIDENCE, IOU_THRESHOLD)

            if len(boxes) > 0:
                best_idx = scores.argmax()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    conf = scores[i]
                    cls = class_ids[i]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 3), 5, (0, 0, 255), -1)
                    text = f"{'person' if cls == PERSON_CLASS else cls} {int(conf * 100)}%"
                    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if ENABLE_AIMBOT and i == best_idx and cls == PERSON_CLASS:
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

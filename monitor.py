import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
import mss
import time
import signal
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
ONNX_MODEL = os.path.join(ROOT_DIR, "models", "yolov8n.onnx")
INPUT_SIZE = 256
PERSON_CLASS = 0
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

SHOW_FPS = "--no-fps" not in sys.argv

COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 255, 0)]

BANNER = r"""
 █████╗ ██╗███╗   ███╗
██╔══██╗██║████╗ ████║
███████║██║██╔████╔██║
██╔══██║██║██║╚██╔╝██║
██║  ██║██║██║ ╚═╝ ██║
╚═╝  ╚═╝╚═╝╚═╝     ╚═╝
     AI + m = AIM
     МОНИТОРИНГ
"""


def export_to_onnx():
    if os.path.exists(ONNX_MODEL):
        return
    print("\033[1;36m[Монитор] Первый запуск — скачиваю YOLOv8n (~6MB) и экспортирую в ONNX...")
    subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from ultralytics import YOLO
    pt_path = os.path.join(ROOT_DIR, "yolov8n.pt")
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=INPUT_SIZE, half=False)
    exported = os.path.join(ROOT_DIR, "yolov8n.onnx")
    os.makedirs(os.path.dirname(ONNX_MODEL), exist_ok=True)
    os.replace(exported, ONNX_MODEL)
    if os.path.exists(pt_path):
        os.remove(pt_path)
    print("\033[1;32m[Монитор] Экспорт ONNX готов.")


def postprocess(output, img_w, img_h, conf_thresh, iou_thresh):
    output = output[0]
    pred = output.T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

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


def main():
    print("\033[1;36m" + BANNER + "\033[0m")

    sct = mss.mss()
    monitors = sct.monitors[1:]
    if not monitors:
        print("\033[1;91m[Монитор] Мониторы не найдены!")
        return

    print("\033[1;35m[Монитор] Найденные мониторы:\033[0m")
    for i, m in enumerate(monitors):
        w, h = m["width"], m["height"]
        x, y = m["left"], m["top"]
        print(f"  [\033[1;36m{i}\033[0m] {w}x{h} at ({x},{y})")
    print(f"  [\033[1;36mf\033[0m] Все мониторы (виртуальный экран)")

    while True:
        choice = input("\033[1;33mВыбери номер монитора (или f для всех) > \033[0m").strip().lower()
        if choice == "f":
            region = sct.monitors[0]
            break
        try:
            idx = int(choice)
            if 0 <= idx < len(monitors):
                region = monitors[idx]
                break
        except ValueError:
            pass
        print("\033[1;91mНеверный выбор, попробуй снова.")

    print(f"\033[1;32m[Монитор] Захват: {region['width']}x{region['height']} at ({region['left']},{region['top']})")

    export_to_onnx()

    print("\033[1;36m[Монитор] Загружаю модель YOLOv8n ONNX...")
    sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print("\033[1;32m[Монитор] Модель загружена.")
    print("\033[1;32m[Монитор] Запуск... (нажми 'q' в окне для выхода)\n")

    capture = sct

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    cv2.namedWindow("AIm Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AIm Monitor", min(region["width"], 1280), min(region["height"], 720))

    prev_time = time.perf_counter()
    fps = 0

    while True:
        frame = np.array(capture.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img_h, img_w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        blob = blob.astype(np.float32)

        output = sess.run(None, {inp_name: blob})[0]
        boxes, scores, class_ids = postprocess(output, img_w, img_h, 0.25, 0.45)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = scores[i]
            cls = int(class_ids[i])
            color = COLORS[cls % len(COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            name = COCO_NAMES.get(cls, str(cls))
            label = f"{name} {int(conf * 100)}%"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if cls == PERSON_CLASS:
                cx, cy = (x1 + x2) // 2, y1
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if SHOW_FPS:
            fps_text = f"FPS: {fps}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_PLAIN, 1.2, 2)
            fx = img_w - tw - 12
            fy = th + 10
            cv2.putText(frame, fps_text, (fx, fy), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 0), 2, cv2.LINE_AA)

        cv2.imshow("AIm Monitor", frame)

        now = time.perf_counter()
        elapsed = now - prev_time
        prev_time = now
        fps = int(1 / elapsed) if elapsed > 0 else 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

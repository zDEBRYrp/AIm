"""
Визуальный тест моделей YOLO.
Нажимай 1/2/3 чтобы переключать модели и смотри разницу сам.
q — выход
"""
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
import time
import subprocess
import mss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
INPUT_SIZE = 256

MODELS = [
    ("1", "YOLOv8n-COCO", "yolov8n"),
    ("2", "YOLO11n-COCO", "yolo11n"),
    ("3", "YOLO26n-COCO", "yolo26n"),
    ("4", "PHD-Head", "yolov11_phd_s"),
    ("5", "YOLOv8n-Crowd", "yolov8n_crowdhuman"),
    ("6", "YOLOv5m-Crowd", "crowdhuman_yolov5m"),
]

PHD_URL = "https://huggingface.co/Sharath33/Person/resolve/main/yolov11_phd_s.onnx"
CROWD_V8_URL = "https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.onnx"
CROWD_V5_URL = "https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx"

PHD_NAMES = {0: "person", 1: "head"}
CROWD_NAMES = {0: "person", 1: "head"}

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

COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 255, 0)]


def export_to_onnx(pt_name):
    onnx_path = os.path.join(MODELS_DIR, f"{pt_name}.onnx")
    if os.path.exists(onnx_path):
        return onnx_path

    if pt_name == "yolov11_phd_s":
        print(f"  Скачиваю PHD Head Detection с Hugging Face...")
        import urllib.request
        urllib.request.urlretrieve(PHD_URL, onnx_path)
        return onnx_path

    if pt_name == "yolov8n_crowdhuman":
        print(f"  Скачиваю YOLOv8n CrowdHuman...")
        import urllib.request
        urllib.request.urlretrieve(CROWD_V8_URL, onnx_path)
        return onnx_path

    if pt_name == "crowdhuman_yolov5m":
        print(f"  Скачиваю YOLOv5m CrowdHuman...")
        import urllib.request
        urllib.request.urlretrieve(CROWD_V5_URL, onnx_path)
        return onnx_path

    print(f"  Скачиваю и экспортирую {pt_name}...")
    subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from ultralytics import YOLO
    pt_path = os.path.join(MODELS_DIR, f"{pt_name}.pt")
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=INPUT_SIZE, half=False)
    exported = os.path.join(SCRIPT_DIR, f"{pt_name}.onnx")
    if os.path.exists(exported):
        os.replace(exported, onnx_path)
    if os.path.exists(pt_path):
        os.remove(pt_path)
    return onnx_path


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
    b = np.zeros_like(boxes_xywh)
    b[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    b[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    b[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    b[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    sx, sy = img_w / INPUT_SIZE, img_h / INPUT_SIZE
    b[:, [0, 2]] *= sx
    b[:, [1, 3]] *= sy
    indices = cv2.dnn.NMSBoxes(b.tolist(), max_scores.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    indices = indices.flatten()
    return b[indices], max_scores[indices], class_ids[indices]


def main():
    print("Тест моделей YOLO — визуальное сравнение")
    print("Переключение: 1-3=COCO  4=PHD-Head  5=YOLOv8-Crowd  6=YOLOv5-Crowd")
    print("q — выход\n")

    sct = mss.MSS()
    monitors = sct.monitors[1:]
    if not monitors:
        print("Мониторы не найдены!")
        return

    for i, m in enumerate(monitors):
        tag = " [ОСНОВНОЙ]" if m.get("is_primary") else ""
        print(f"  [{i}] {m['width']}x{m['height']}{tag}")

    choice = input("Монитор > ").strip()
    try:
        region = monitors[int(choice)]
    except (ValueError, IndexError):
        region = next((m for m in monitors if m.get("is_primary")), monitors[0])

    print(f"\nЗагрузка моделей...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    sessions = {}
    for key, name, pt_name in MODELS:
        try:
            onnx_path = export_to_onnx(pt_name)
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0].name
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            sessions[key] = {"name": name, "sess": sess, "inp": inp, "size": size_mb}
            print(f"  [{key}] {name} — {size_mb:.1f} MB OK")
        except Exception as e:
            print(f"  [{key}] {name} — ОШИБКА: {e}")

    if not sessions:
        print("Ни одна модель не загружена!")
        return

    active = list(sessions.keys())[0]
    conf = 0.25
    iou = 0.45

    print(f"\nГотово! Нажимай 1/2/3 для переключения моделей.")
    print(f"c — confidence (сейчас {conf}), i — IoU (сейчас {iou})\n")

    cv2.namedWindow("YOLO Test", cv2.WINDOW_NORMAL)
    prev_time = time.perf_counter()
    fps = 0

    while True:
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img_h, img_w = frame.shape[:2]

        m = sessions[active]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        blob = blob.astype(np.float32)

        t0 = time.perf_counter()
        output = m["sess"].run(None, {m["inp"]: blob})[0]
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        boxes, scores, class_ids = postprocess(output, img_w, img_h, conf, iou)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            sc = scores[i]
            cls = int(class_ids[i])
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            name = COCO_NAMES.get(cls, str(cls))
            if m["name"] in ("PHD-Head", "YOLOv8-Crowd", "YOLOv5-Crowd"):
                name = CROWD_NAMES.get(cls, str(cls))
            label = f"{name} {int(sc * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        fps = int(1 / dt) if dt > 0 else 0

        info = f"{m['name']} | {m['size']:.1f}MB | {infer_ms:.0f}ms | FPS:{fps} | det:{len(boxes)} | conf:{conf}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        hints = "1/2/3=model  c=conf  i=iou  q=quit"
        cv2.putText(frame, hints, (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("YOLO Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            k = chr(key)
            if k in sessions:
                active = k
                print(f"Модель: {sessions[active]['name']}")
        elif key == ord("c"):
            conf = min(0.95, conf + 0.05)
            print(f"confidence = {conf:.2f}")
        elif key == ord("C"):
            conf = max(0.05, conf - 0.05)
            print(f"confidence = {conf:.2f}")
        elif key == ord("i"):
            iou = min(0.9, iou + 0.05)
            print(f"iou = {iou:.2f}")
        elif key == ord("I"):
            iou = max(0.1, iou - 0.05)
            print(f"iou = {iou:.2f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

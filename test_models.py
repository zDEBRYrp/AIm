"""
Test different YOLO models: download, export to ONNX, measure inference speed.
Compares YOLOv8n, YOLO11n, YOLO26n on real screen capture.
"""
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import sys
import time
import subprocess
import numpy as np
import cv2
import mss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
INPUT_SIZE = 256

MODELS = {
    "YOLOv8n": "yolov8n",
    "YOLO11n": "yolo11n",
    "YOLO26n": "yolo26n",
}


def ensure_model(name, pt_name):
    onnx_path = os.path.join(MODELS_DIR, f"{pt_name}.onnx")
    if os.path.exists(onnx_path):
        print(f"  [OK] {pt_name}.onnx уже есть")
        return onnx_path

    pt_path = os.path.join(MODELS_DIR, f"{pt_name}.pt")
    if not os.path.exists(pt_path):
        print(f"  Скачиваю {pt_name}.pt...")
        subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(format="onnx", imgsz=INPUT_SIZE, half=False)
        exported = os.path.join(SCRIPT_DIR, f"{pt_name}.onnx")
        if os.path.exists(exported):
            os.replace(exported, onnx_path)
        if os.path.exists(pt_path):
            os.remove(pt_path)
    else:
        print(f"  Экспортирую {pt_name}.pt -> ONNX...")
        from ultralytics import YOLO
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
        return 0
    return mask.sum()


def measure(onnx_path, sct, region, frames=100):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    frame = np.array(sct.grab(region))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # warmup
    for _ in range(10):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        sess.run(None, {inp_name: blob.astype(np.float32)})

    # measure
    times = []
    detections = 0
    for _ in range(frames):
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        blob = blob.astype(np.float32)

        t0 = time.perf_counter()
        output = sess.run(None, {inp_name: blob})[0]
        t1 = time.perf_counter()

        times.append(t1 - t0)
        detections += postprocess(output, frame.shape[1], frame.shape[0], 0.25, 0.45)

    avg_ms = (sum(times) / len(times)) * 1000
    fps = 1000 / avg_ms if avg_ms > 0 else 0
    avg_dets = detections / frames
    return avg_ms, fps, avg_dets


def main():
    print("=" * 50)
    print("  ТЕСТ МОДЕЛЕЙ YOLO")
    print("=" * 50)

    sct = mss.mss()
    monitors = sct.monitors[1:]
    if not monitors:
        print("Мониторы не найдены!")
        return

    print("\nМониторы:")
    for i, m in enumerate(monitors):
        tag = " [ОСНОВНОЙ]" if m["primary"] else ""
        print(f"  [{i}] {m['width']}x{m['height']}{tag}")

    choice = input("\nВыбери монитор (номер) > ").strip()
    try:
        region = monitors[int(choice)]
    except (ValueError, IndexError):
        region = next((m for m in monitors if m["primary"]), monitors[0])

    print(f"\nЗахват: {region['width']}x{region['height']}")
    print(f"Размер входа: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Кадров для замера: 100\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []
    for display_name, pt_name in MODELS.items():
        print(f"--- {display_name} ---")
        try:
            onnx_path = ensure_model(display_name, pt_name)
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            avg_ms, fps, avg_dets = measure(onnx_path, sct, region)
            results.append((display_name, size_mb, avg_ms, fps, avg_dets))
            print(f"  Размер: {size_mb:.1f} MB")
            print(f"  Среднее: {avg_ms:.1f} мс | {fps:.0f} FPS")
            print(f"  Детекций/кадр: {avg_dets:.1f}\n")
        except Exception as e:
            print(f"  ОШИБКА: {e}\n")

    if results:
        print("=" * 60)
        print(f"{'Модель':<12} {'Размер':>8} {'мс':>8} {'FPS':>8} {'Детекций':>10}")
        print("-" * 60)
        for name, size_mb, ms, fps, dets in results:
            print(f"{name:<12} {size_mb:>7.1f}M {ms:>7.1f} {fps:>7.0f} {dets:>9.1f}")
        print("=" * 60)

        best = max(results, key=lambda r: r[3])
        print(f"\nБыстрая модель: {best[0]} ({best[3]:.0f} FPS)")


if __name__ == "__main__":
    main()

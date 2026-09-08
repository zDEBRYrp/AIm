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


DEFAULT_CONFIG = {
    "confidence": 0.30,
    "iou_threshold": 0.45,
    "activation_range": 250,
    "aim_speed": 0.35,
    "max_step": 25,
    "deadzone": 2,
    "aim_height_ratio": 0.2,
    "hold_button": "x2",
    "toggle_hotkey": "F1",
    "fullscreen": False,
    "detector_only": False,
    "monitor": "primary",
}

SETTING_ITEMS = [
    ("1", "confidence", "Уверенность (0.05-0.95)"),
    ("2", "iou_threshold", "NMS IoU (0.1-0.9)"),
    ("3", "activation_range", "FOV размер px или 'full'"),
    ("4", "aim_speed", "Скорость аима (0.05-2.0)"),
    ("5", "max_step", "Макс шаг px/кадр (1-200)"),
    ("6", "deadzone", "Мёртвая зона px (0-50)"),
    ("7", "aim_height_ratio", "Высота прицела 0-1 (0.2=голова)"),
    ("H", "hold_button", "Кнопка hold (x2/left/right)"),
    ("T", "toggle_hotkey", "Хоткей переключения (напр. F1)"),
    ("F", "fullscreen", "Захват всего экрана (y/n)"),
    ("D", "detector_only", "Только детектор без аима (y/n)"),
    ("M", "monitor", "Монитор (primary или 0/1/...)"),
]


def load_config():
    import copy
    import json
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            user = json.load(f)
        cfg.update({k: v for k, v in user.items() if v is not None})
    return cfg


def save_config(cfg=None):
    import json
    cfg = cfg if cfg is not None else CFG
    with open(os.path.join(ROOT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)


def coerce_value(key, text):
    t = text.strip()
    try:
        if key == "confidence":
            v = float(t)
            return (True, v, "") if 0.05 <= v <= 0.95 else (False, None, "Диапазон 0.05-0.95")
        if key == "iou_threshold":
            v = float(t)
            return (True, v, "") if 0.1 <= v <= 0.9 else (False, None, "Диапазон 0.1-0.9")
        if key == "activation_range":
            if t.lower() == "full":
                return (True, "full", "")
            v = int(float(t))
            return (True, v, "") if 64 <= v <= 1000 else (False, None, "Диапазон 64-1000 or 'full'")
        if key == "aim_speed":
            v = float(t)
            return (True, v, "") if 0.05 <= v <= 2.0 else (False, None, "Диапазон 0.05-2.0")
        if key == "max_step":
            v = int(float(t))
            return (True, v, "") if 1 <= v <= 200 else (False, None, "Диапазон 1-200")
        if key == "deadzone":
            v = int(float(t))
            return (True, v, "") if 0 <= v <= 50 else (False, None, "Диапазон 0-50")
        if key == "aim_height_ratio":
            v = float(t)
            return (True, v, "") if 0.0 <= v <= 1.0 else (False, None, "Диапазон 0.0-1.0")
        if key == "hold_button":
            v = t.lower()
            return (True, v, "") if v in ("x2", "left", "right") else (False, None, "Используй x2 / left / right")
        if key == "toggle_hotkey":
            if not t:
                return (False, None, "Пустой хоткей")
            keyboard.parse_hotkey(t.lower())
            return (True, t.lower(), "")
        if key == "fullscreen":
            v = t.lower()
            if v in ("y", "yes", "true", "1", "on"):
                return (True, True, "")
            if v in ("n", "no", "false", "0", "off"):
                return (True, False, "")
            return (False, None, "Введи y/n")
        if key == "detector_only":
            v = t.lower()
            if v in ("y", "yes", "true", "1", "on"):
                return (True, True, "")
            if v in ("n", "no", "false", "0", "off"):
                return (True, False, "")
            return (False, None, "Введи y/n")
        if key == "monitor":
            if t.lower() == "primary":
                return (True, "primary", "")
            v = int(t)
            n = len(list_monitors())
            return (True, v, "") if 0 <= v < n else (False, None, f"Монитор 0-{n - 1} или 'primary'")
    except ValueError:
        return (False, None, "Не число")
    except Exception as e:
        return (False, None, f"Неверное значение: {e}")
    return (False, None, "Неизвестная настройка")


def console_focused():
    try:
        import win32gui
        return win32gui.GetForegroundWindow() == win32gui.GetConsoleWindow()
    except Exception:
        return True


CFG = load_config()


def _select_monitor():
    global Wd, Hd, VWd, VHd, MON_X, MON_Y
    choice = CFG.get("monitor", "primary")
    if isinstance(choice, int):
        set_monitor(choice)
    for m in list_monitors():
        tag = " [ОСНОВНОЙ]" if m["primary"] else ""
        print(f"\033[1;36m[Монитор {m['index']}] {m['width']}x{m['height']} at ({m['left']},{m['top']}){tag}")
    Wd, Hd = get_screen_size()
    MON_X, MON_Y = get_monitor_origin()
    VWd, VHd = get_virtual_size()
    print(f"\033[1;32m[Статус] Захват: {Wd}x{Hd} at ({MON_X},{MON_Y}); виртуальный экран {VWd}x{VHd}.")


_select_monitor()


def export_to_onnx():
    if os.path.exists(ONNX_MODEL):
        return
    print("\033[1;36m[Статус] Первый запуск — скачиваю YOLOv8n (~6MB) и экспортирую в ONNX...")
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
    print("\033[1;32m[Статус] Экспорт ONNX готов.")


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

    def set_region(self, region):
        with self.lock:
            self.region = region


def aimbot(ENABLE_AIMBOT=True):
    mode = "hold"
    mouse_held = False
    last_tx, last_ty = None, None
    in_menu = False
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def is_full():
        return bool(CFG.get("fullscreen", False)) or CFG.get("activation_range") == "full"

    def current_fov():
        ar = CFG.get("activation_range", 250)
        return 250 if ar == "full" else int(ar)

    cx = Wd // 2
    cy = Hd // 2
    ar0 = current_fov()
    box0 = (cx - ar0 // 2, cy - ar0 // 2, cx + ar0 // 2, cy + ar0 // 2)
    if is_full():
        capture_region = None
    else:
        capture_region = box0
    aim_box = box0

    def signal_handler(sig, frame):
        print("\n[Выход] завершаю...")
        cap.stop()
        close()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    export_to_onnx()

    print("\033[1;36m[Статус] Загружаю модель YOLOv8n ONNX...")
    sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print("\033[1;32m[Статус] Модель загружена (onnxruntime CPU).")

    if not ENABLE_AIMBOT:
        print("\033[1;91m[Статус] Аим выключен, только детектор...")
    else:
        print("\033[1;92m[AI] Аим включён..")
    print(f"\033[1;33m[Режим] hold — держи {CFG.get('hold_button', 'x2')} для аима, {CFG.get('toggle_hotkey', 'F1')} для always-on.")

    def is_active():
        return mode == "always" or mouse_held

    def toggle_aimbot():
        nonlocal mode, last_tx, last_ty
        mode = "always" if mode == "hold" else "hold"
        last_tx, last_ty = None, None
        if mode == "always":
            print("\nАим : " + GREEN + "всегда вкл" + RESET)
            winsound.Beep(440, 100)
        else:
            print("\nАим : " + RED + "режим hold" + RESET)

    def current_hold_button():
        from pynput.mouse import Button as _Btn
        return getattr(_Btn, CFG.get("hold_button", "x2"), _Btn.x2)

    def setup_hotkeys():
        try:
            keyboard.clear_all_hotkeys()
        except Exception:
            pass
        keyboard.add_hotkey(CFG.get("toggle_hotkey", "F1"), toggle_aimbot)
        keyboard.add_hotkey("8", open_settings)

    def place_window():
        ar = current_fov()
        if is_full():
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, ar, ar)
            try:
                import win32gui
                hwnd = win32gui.FindWindow(None, WINDOW_NAME)
                if hwnd:
                    win_x = (Wd - ar) // 2
                    win_y = (Hd - ar) // 2
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, win_x, win_y, 0, 0, 0x0001 | 0x0040)
            except Exception:
                pass

    def rebuild_geometry():
        nonlocal capture_region, aim_box
        ar = current_fov()
        cx_, cy_ = Wd // 2, Hd // 2
        box = (cx_ - ar // 2, cy_ - ar // 2, cx_ + ar // 2, cy_ + ar // 2)
        capture_region = None if is_full() else box
        aim_box = box
        cap.set_region(capture_region)
        place_window()

    def apply_setting(ckey):
        if ckey in ("activation_range", "fullscreen"):
            rebuild_geometry()
        elif ckey == "toggle_hotkey":
            setup_hotkeys()
        elif ckey == "monitor":
            _select_monitor()
            rebuild_geometry()

    def apply_all():
        _select_monitor()
        rebuild_geometry()
        setup_hotkeys()

    def open_settings():
        nonlocal in_menu, capture_region, aim_box
        if in_menu or not console_focused():
            return
        in_menu = True
        try:
            import msvcrt
            import questionary
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print("======== Настройки AIm ========")
                for mkey, ckey, label in SETTING_ITEMS:
                    v = CFG.get(ckey)
                    if v is True:
                        v = "вкл"
                    elif v is False:
                        v = "выкл"
                    print(f"  [{mkey}] {label}: {v}")
                print("  [S] Сохранить и выйти")
                print("  [Esc] Сбросить настройки")
                print("  [Q/8] Назад")
                ch = msvcrt.getch()
                if ch == b"\x1b":
                    import copy
                    CFG.clear()
                    CFG.update(copy.deepcopy(DEFAULT_CONFIG))
                    save_config(CFG)
                    apply_all()
                    print("Сброшено к дефолтам. Нажми любую клавишу...")
                    msvcrt.getch()
                elif ch in (b"q", b"Q", b"8"):
                    save_config(CFG)
                    break
                elif ch in (b"s", b"S"):
                    save_config(CFG)
                    print("Сохранено в config.json. Нажми любую клавишу...")
                    msvcrt.getch()
                else:
                    try:
                        sel = ch.decode("utf-8", "ignore").upper()
                    except Exception:
                        continue
                    item = next((it for it in SETTING_ITEMS if it[0] == sel), None)
                    if item is None:
                        continue
                    edit_setting(questionary, item)
        finally:
            in_menu = False

    def edit_setting(questionary, item):
        _, ckey, label = item
        cur = CFG.get(ckey)
        try:
            if ckey in ("fullscreen", "detector_only"):
                val = questionary.confirm(f"{label}? (now {cur})", default=bool(cur)).ask()
                if val is None:
                    return
                CFG[ckey] = bool(val)
            elif ckey == "hold_button":
                val = questionary.select(f"{label} (now {cur}):",
                                         choices=["x2", "left", "right"]).ask()
                if val is None:
                    return
                CFG[ckey] = val
            elif ckey == "monitor":
                from questionary import Choice
                opts = [Choice("primary (auto)", value="primary")]
                for m in list_monitors():
                    tag = "PRIMARY" if m["primary"] else "secondary"
                    opts.append(Choice(f"{m['index']}: {m['width']}x{m['height']} ({tag})", value=m["index"]))
                val = questionary.select(f"{label} (now {cur}):", choices=opts).ask()
                if val is None:
                    return
                CFG[ckey] = val
            else:
                def _v(t):
                    ok, _, err = coerce_value(ckey, t)
                    return True if ok else err
                t = questionary.text(f"{label} (now {cur}):", validate=_v).ask()
                if t is None:
                    return
                ok, val, err = coerce_value(ckey, t)
                if not ok:
                    print(err)
                    return
                CFG[ckey] = val
            save_config(CFG)
            apply_setting(ckey)
            print(f"{label} = {CFG[ckey]}")
        except KeyboardInterrupt:
            return

    setup_hotkeys()

    def on_click(x, y, button, pressed):
        if button == current_hold_button():
            nonlocal mouse_held, last_tx, last_ty
            mouse_held = pressed
            if pressed:
                last_tx, last_ty = None, None

    cap = ThreadedCapture(capture_region)
    cap.start()

    WINDOW_NAME = "AIm - Детектор"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    window_placed = False

    with Listener(on_click=on_click) as listener:
        while True:
            if in_menu:
                time.sleep(0.2)
                continue
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
                        if is_full():
                            tx, ty = float(ax), float(ay)
                        else:
                            tx = aim_box[0] + ax
                            ty = aim_box[1] + ay
                        targets.append((tx, ty))

                if ENABLE_AIMBOT and not CFG.get("detector_only", False) and targets:
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
                        spd = CFG.get("aim_speed", 0.35)
                        mx = CFG.get("max_step", 25)
                        sx = max(-mx, min(mx, dx * spd))
                        sy = max(-mx, min(mx, dy * spd))
                        move_relative(sx, sy)
            else:
                last_tx, last_ty = None, None

            if CFG.get("detector_only", False):
                cv2.putText(frame, "ТОЛЬКО ДЕТЕКТОР - без аима", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            if not window_placed:
                window_placed = True
                place_window()

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

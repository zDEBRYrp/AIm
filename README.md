<h1 align="center">AIm - AI Aimbot</h1>
<p align="center">
    <a href="https://github.com/zDEBRYrp/AIm">
        <img src="https://github.com/McDaived/AIMi/assets/18085492/56684d14-9573-403e-bb06-6c323d475ebc" alt="Logo" width="500" height="150">
    </a>

**AIm** = **AI** + **m** (от AIM) — нейросетевой аимбот, работающий в реальном времени на основе object detection.

<h4 align="center">AIm uses YOLOv8n for real-time person detection. Screen capture via mss, inference via onnxruntime — lightweight, fast, no GPU required.</h4>

<h6 align="center">Fork of <a href="https://github.com/McDaived/AIMi">McDaived/AIMi</a> — modernized and improved.</h6>

  <p align="center">
<img src="https://github.com/McDaived/AIMi/assets/18085492/9e53d002-80ec-472b-9156-5416a061790e" alt="Your Image Description" width="500">
<img src="https://github.com/McDaived/AIMi/assets/18085492/c430ab48-99e4-466b-833f-77879a5a01e9" alt="Your Image Description" width="260">


## About the name

**AIm** — composite name: **AI** (neural network) + **m** (makes it AIM).
> If you want to clarify: "AI Aim - AIm"


## Architecture (v2 - 2026)
- **YOLOv8n** (ultralytics) → ONNX export → **onnxruntime** for inference
- **mss** for screen capture (fast, minimal CPU overhead)
- No PyTorch at runtime — only used once for model export
- Auto-downloads and exports model on first run (~6MB download)


## Features
- (F1) Switch mode: Always On / Hold Mode
- (Mouse4) Aim while held — works ONLY in Hold Mode
- (8) Settings menu in console — FOV (or `full`), aim speed, buttons, monitor, detector-only video mode; Esc resets to defaults
- (0) Exit
- GUI objects detector — see how the model recognizes targets in real time
- All settings in `config.json` (confidence, aim speed, max step, deadzone, aim height, hotkeys, monitor)


## Requirements
- Python 3.10+
- Windows (uses Win32 API for mouse input)
- No RTX required — works on CPU


## How to use
1. Download [Python](https://www.python.org/) (3.10+)
2. Clone or download this repo
3. **Disable** Enhance Pointer Precision: `Mouse Properties` → `Pointer Options` → uncheck
4. **Disable RAW INPUT** in your game (if available)
5. Run:
```
python start.py
```
Dependencies install automatically. YOLOv8n model exports to ONNX on first launch.


## Settings

### Optimizing for CS2
Use a **dot crosshair** — the neural network works better without crosshair lines interfering with detection.

Import this crosshair code in CS2 (Settings → Game → Crosshair → Import):
```
CSGO-FZUXv-VVbLD-EDy3c-kYqf3-UMkTL
```
Or set manually: enable **"Точка в центре"**, set Length to 0, Thickness to 0, Gap to 0, Outline off.

### Change Hotkey
Edit `lib/detect.py`, the `on_click` function.


## Known Issues

**Aiming at the ground** — Disable raw input + enhance pointer precision.

**Valorant** — Requires a kernel driver to bypass mouse input restrictions.


## How it works

### YOLOv8n (ONNX)
Trained on COCO dataset (80 classes). Exported to ONNX for lightweight CPU inference via onnxruntime. No PyTorch overhead at runtime.

### Screen Capture
Uses `mss` for fast screen region capture with minimal CPU overhead.

### Mouse Control
Uses Win32 `SendInput` API via pynput for precise mouse movement.


## Credits
- Original project: [McDaived/AIMi](https://github.com/McDaived/AIMi) — MIT License
- Fork & modernization: **zDEBRYrp**


## License
MIT License — see [LICENSE](LICENSE)

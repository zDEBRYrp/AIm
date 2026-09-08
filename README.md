<h1 align="center">AIm - AI Aimbot</h1>
<p align="center">
    <a href="https://github.com/zDEBRYrp/AIm">
        <img src="https://github.com/McDaived/AIMi/assets/18085492/56684d14-9573-403e-bb06-6c323d475ebc" alt="Logo" width="500" height="150">
    </a>

**AIm** = **AI** + **m** (от AIM) — нейросетевой аимбот, работающий в реальном времени на основе object detection.

<h4 align="center">AIm uses real-time object detection with neural networks to recognize human-like patterns and aim at targets. It never accesses game memory — only captures the screen and controls the mouse on behalf of the user.</h4>

<h6 align="center">Fork of <a href="https://github.com/McDaived/AIMi">McDaived/AIMi</a> — modernized and improved.</h6>

  <p align="center">
<img src="https://github.com/McDaived/AIMi/assets/18085492/9e53d002-80ec-472b-9156-5416a061790e" alt="Your Image Description" width="500">
<img src="https://github.com/McDaived/AIMi/assets/18085492/c430ab48-99e4-466b-833f-77879a5a01e9" alt="Your Image Description" width="260">


## About the name

**AIm** — composite name: **AI** (neural network) + **m** (makes it AIM).
> If you want to clarify: "AI Aim - AIm"


## What's new (v2 - 2026)
- **YOLOv8n** instead of YOLOv3-tiny — faster, more accurate, 80 object classes
- **ultralytics** pipeline — auto NMS, auto model download, GPU/CPU auto-detect
- **mss** screen capture — replaced raw Win32 API calls
- Removed dead code and deprecated libraries
- Fixed multiple bugs (deprecations, missing imports)
- Python 3.10+ support


## Features
- (F1) Aimbot: Always On / Hold Mode
- (Mouse4) Hold Mode: Press / Release
- (0) Exit
- GUI objects detector — see how the model recognizes targets in real time


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
Dependencies install automatically on first run. YOLOv8n model (~6MB) downloads on first launch.


## Settings

### Optimizing for CS2
Use a dot crosshair for best results:
```
CSGO-YE93T-V6tTU-Cxa9r-jCf7s-2XJaA
```

### Stretch Screen (Optional)
Edit `lib/detect.py`, the `ACTIVATION_RANGE` and `origbox` calculation.

### Change Hotkey
Edit `lib/detect.py`, the `on_click` function (line ~79).


## Known Issues

**Aiming at the ground** — Disable raw input + enhance pointer precision.

**Valorant** — Requires a kernel driver to bypass mouse input restrictions. See [this](https://www.unknowncheats.me/forum/3912497-post139.html).


## How it works

### Neural Network
- Never accesses game memory — invisible to most anti-cheat
- Abstracts capabilities to many FPS games without code modifications

### YOLOv8 (ultralytics)
Trained on COCO dataset (80 classes including person). Optimized for real-time detection.

### Screen Capture
Uses `mss` for fast screen region capture with minimal CPU overhead.


## Credits
- Original project: [McDaived/AIMi](https://github.com/McDaived/AIMi) — MIT License
- Fork & modernization: **zDEBRYrp**


## License
MIT License — see [LICENSE](LICENSE)

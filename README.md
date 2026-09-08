<h1 align="center">AIm - AI Aimbot</h1>
<p align="center">
    <a href="https://github.com/zDEBRYrp/AIm">
        <img src="https://github.com/McDaived/AIMi/assets/18085492/56684d14-9573-403e-bb06-6c323d475ebc" alt="Logo" width="500" height="150">
    </a>

**AIm** = **AI** + **m** (от AIM) — нейросетевой аимбот, работающий в реальном времени на основе object detection.

<h4 align="center">AIm uses real-time object detection with neural networks to recognize human-like patterns and aim at targets. It never accesses game memory — only captures the screen and controls the mouse on behalf of the user.</h4>

<h6 align="center">This is a fork of <a href="https://github.com/McDaived/AIMi">McDaived/AIMi</a> — improved and renamed.</h6>

  <p align="center">
<img src="https://github.com/McDaived/AIMi/assets/18085492/9e53d002-80ec-472b-9156-5416a061790e" alt="Your Image Description" width="500">
<img src="https://github.com/McDaived/AIMi/assets/18085492/c430ab48-99e4-466b-833f-77879a5a01e9" alt="Your Image Description" width="260">


## About the name

**AIm** — composite name: **AI** (neural network) + **m** (makes it AIM).
> If you want to clarify: "AI Aim - AIm"

## Features
- (F1) Aimbot: Always On / Hold Mode
- (Mouse4) Hold Mode: Press / Release
- (0) Exit
- GUI objects detector — see how the model recognizes targets in real time


## Requirements
- Python 3.8+
- Windows (uses Win32 API for screen capture and mouse input)
- No RTX required — works on CPU (OpenCL accelerated)


## How to use
1. Download [Python](https://www.python.org/) (latest version recommended)
2. Clone or download this repo
3. **Disable** Enhance Pointer Precision: `Mouse Properties` → `Pointer Options` → uncheck
4. **Disable RAW INPUT** in your game (if available)
5. Run:
```
python start.py
```
Dependencies install automatically on first run.


## Settings

### Optimizing for CS2
Use a dot crosshair for best results:
```
CSGO-YE93T-V6tTU-Cxa9r-jCf7s-2XJaA
```

### Stretch Screen (Optional)
Edit `lib/detect.py`, line 70:
```py
origbox = (int(Wd/3.1 - ACTIVATION_RANGE/4),
           int(Hd/2.5 - ACTIVATION_RANGE/4),
           int(Wd/4 + ACTIVATION_RANGE/1),
           int(Hd/2 + ACTIVATION_RANGE/2))
```

### Change Hotkey
Edit `lib/detect.py`, line 118:
```py
if button == button.x2:  # Change to button.left for left click
```


## Known Issues

**[WARN:0@x.xxx]** — OpenCV warning about CUDA. It switches to CPU automatically. Safe to ignore.

**Aiming at the ground** — Disable raw input + enhance pointer precision.

**Valorant** — Requires a kernel driver to bypass mouse input restrictions. See [this](https://www.unknowncheats.me/forum/3912497-post139.html).


## How it works

### Neural Network
- Never accesses game memory — invisible to most anti-cheat
- Abstracts capabilities to many FPS games without code modifications

### YOLOv3-tiny
Trained on a combination of video game images and the **COCO** dataset. Optimized to recognize human-like objects quickly.

### OpenCV
Screen capture and GPU acceleration via CUDA / OpenCL.


## Credits
- Original project: [McDaived/AIMi](https://github.com/McDaived/AIMi) — MIT License
- Fork & improvements: **zDEBRYrp**


## License
MIT License — see [LICENSE](LICENSE)

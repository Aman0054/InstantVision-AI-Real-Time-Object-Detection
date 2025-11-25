# 🚀 Live Object Detection using YOLOv8

A simple real-time object detection system using **YOLOv8** and **webcam**. Works on **CPU** and **GPU** (if available).

---

## 📦 Setup

### 1. Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

(or)

```powershell
pip install ultralytics opencv-python numpy
```

---

## ▶️ Run Live Detection

```powershell
python detect_live.py --show-fps
```

### Optional:

* Different camera:
  `python detect_live.py --camera 1`
* Custom model:
  `python detect_live.py --model path/to/best.pt`
* Force CPU:
  `python detect_live.py --device cpu`

---

## ⚠️ Common Fixes

* Missing numpy → `pip install numpy`
* Camera not working → try `--camera 1`, close other apps
* GPU not used → install CUDA version of PyTorch

---

## 📁 Files

* `detect_live.py` → main script
* `requirements.txt` → dependencies

---

If you want, I can compress this even more or add emojis/titles.

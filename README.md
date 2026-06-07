# 🎯 Real-Time Occupancy Monitoring System

> **YOLOv8 · Deep SORT · OpenCV · CustomTkinter · Pandas · ReportLab**

A production-ready desktop application that detects and tracks people in video footage, counts entries and exits across a virtual line, maintains real-time occupancy, and generates analytics graphs + PDF reports — all from a polished GUI.

---

## 📸 Screenshots

| Live Feed with Tracking | Statistics Panel | Analytics Dashboard |
|:-:|:-:|:-:|
| *(run the app to see)* | *(run the app to see)* | *(run the app to see)* |

---

## ✨ Features

| Category | Details |
|---|---|
| **Detection** | YOLOv8n (swappable to s/m/l/x), person-class only, configurable confidence |
| **Tracking** | Deep SORT with MobileNet Re-ID embedder, persistent unique IDs |
| **Counting** | Horizontal virtual line, entry/exit direction detection, hysteresis buffer |
| **GUI** | CustomTkinter dark-mode UI, live video feed, real-time stats |
| **Analytics** | Entries/hour, exits/hour, occupancy-over-time graphs via Matplotlib + Pandas |
| **Reporting** | PDF report (cover, graphs, event table, conclusions) + CSV export |
| **Logging** | Per-session CSV event log with timestamps |

---

## 🗂️ Project Structure

```
project/
├── main.py                      ← Entry point
├── config/
│   └── settings.py              ← All configurable constants
├── detection/
│   └── detector.py              ← YOLOv8 wrapper
├── tracking/
│   └── tracker.py               ← Deep SORT wrapper
├── counting/
│   └── occupancy_counter.py     ← Virtual line + event logger
├── analytics/
│   └── analytics.py             ← Graph generation (Matplotlib + Pandas)
├── reports/
│   └── report_generator.py      ← PDF report (ReportLab)
├── gui/
│   └── app.py                   ← CustomTkinter application
├── logs/                        ← CSV event logs
├── graphs/                      ← Generated PNG graphs
├── videos/                      ← Place your input videos here
├── assets/                      ← Optional logo
├── documentation/               ← Full project documentation
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 – 3.11
- pip
- *(Optional)* NVIDIA GPU with CUDA 11.8+ for faster inference

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/occupancy-monitoring-system.git
cd occupancy-monitoring-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Enable CUDA – replace torch lines in requirements.txt:
#    torch>=2.0.0+cu118
#    torchvision>=0.15.0+cu118
```

The YOLOv8 model (`yolov8n.pt`) is downloaded automatically on first run.

---

## 🚀 Usage

```bash
python main.py
```

1. Click **📂 Select Video** and choose an `.mp4` / `.avi` file.
2. Click **▶ Start** – detection, tracking, and counting begin immediately.
3. Use **⏸ Pause** / **▶ Resume** to freeze/continue processing.
4. Click **⏹ Stop** when done.
5. Click **📊 Export Report** to choose an output directory.  
   The system generates:
   - `events_<timestamp>.csv`
   - `graphs_<timestamp>/entries_per_hour.png`
   - `graphs_<timestamp>/exits_per_hour.png`
   - `graphs_<timestamp>/occupancy_over_time.png`
   - `graphs_<timestamp>/dashboard.png`
   - `report_<timestamp>.pdf`

---

## ⚙️ Configuration

All tuneable constants are in `config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolov8n.pt` | YOLO model file |
| `YOLO_CONF_THRESH` | `0.40` | Detection confidence threshold |
| `LINE_POSITION_RATIO` | `0.55` | Counting line as fraction of frame height |
| `CROSSING_BUFFER_PX` | `8` | Hysteresis pixels past line to count |
| `YOLO_DEVICE` | `"cpu"` | `"cpu"`, `"cuda"`, or `"mps"` |
| `DEEPSORT_MAX_AGE` | `30` | Frames a lost track is retained |

---

## 🧪 Testing

```bash
# Run a quick smoke test with a sample video
python -c "
from detection.detector import PersonDetector
import cv2, numpy as np
d = PersonDetector()
frame = np.zeros((480, 640, 3), dtype=np.uint8)
results = d.detect(frame)
print('Detector OK –', len(results), 'detections on blank frame')
"
```

---

## 📊 Technologies

| Library | Version | Role |
|---|---|---|
| `ultralytics` | ≥ 8.0 | YOLOv8 detection |
| `deep-sort-realtime` | ≥ 1.3 | Multi-object tracking |
| `opencv-python` | ≥ 4.8 | Frame capture & annotation |
| `customtkinter` | ≥ 5.2 | Modern Tkinter GUI |
| `pandas` | ≥ 2.0 | Event log aggregation |
| `matplotlib` | ≥ 3.7 | Graph generation |
| `reportlab` | ≥ 4.0 | PDF report generation |
| `Pillow` | ≥ 10.0 | PIL/Tkinter image bridge |

---

## 📈 Results

- Accurate real-time detection at 15–30 FPS on CPU (depending on hardware)
- Persistent track IDs across temporary occlusions
- Duplicate-count prevention via hysteresis buffer
- Full session audit trail in CSV

---

## 🔮 Future Enhancements

- [ ] Multi-camera support
- [ ] RTSP live-stream input
- [ ] Re-ID model fine-tuning for specific environments
- [ ] REST API / MQTT dashboard integration
- [ ] Capacity-limit alerts (push notification)
- [ ] Heatmap generation

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

*Built with ❤️ using Python, YOLOv8, and Deep SORT*
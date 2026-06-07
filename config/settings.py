# ─────────────────────────────────────────────────────────────────────────────
# config/settings.py
# Central configuration for the Real-Time Occupancy Monitoring System.
# All tuneable constants live here so no magic numbers scatter across modules.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
LOGS_DIR   = ROOT_DIR / "logs"
GRAPHS_DIR = ROOT_DIR / "graphs"
VIDEOS_DIR = ROOT_DIR / "videos"
ASSETS_DIR = ROOT_DIR / "assets"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure runtime directories exist
for _d in (LOGS_DIR, GRAPHS_DIR, VIDEOS_DIR, ASSETS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── YOLOv8 Detection ─────────────────────────────────────────────────────────
YOLO_MODEL_NAME   = "yolov8n.pt"   # nano is fastest; swap for yolov8s/m/l/x
YOLO_CONF_THRESH  = 0.40           # minimum detection confidence
YOLO_IOU_THRESH   = 0.45           # NMS IoU threshold
YOLO_PERSON_CLASS = 0              # COCO class-id for "person"
YOLO_IMG_SIZE     = 512            # slightly faster than 640, still stable
YOLO_DEVICE       = "cpu"          # "cpu" | "cuda" | "mps"

# ── Deep SORT Tracking ───────────────────────────────────────────────────────
DEEPSORT_MAX_AGE      = 30         # frames a track is kept without detection
DEEPSORT_N_INIT       = 3          # detections needed to confirm a track
DEEPSORT_MAX_IOU_DIST = 0.7        # max IoU distance for association
DEEPSORT_MAX_COSINE_DIST = 0.3     # max cosine distance for Re-ID

# ── Counting Line ─────────────────────────────────────────────────────────────
# Defined as a fraction of the frame height/width so it scales with any video.
# A horizontal line is drawn at LINE_POSITION_RATIO * frame_height.
# Persons crossing upward  (y decreasing) → EXIT
# Persons crossing downward (y increasing) → ENTRY
LINE_POSITION_RATIO = 0.55         # 0.0 = top, 1.0 = bottom
LINE_COLOR_NORMAL   = (0, 200, 255)   # BGR – cyan when idle
LINE_COLOR_ENTRY    = (0, 255, 0)     # BGR – green flash on entry
LINE_COLOR_EXIT     = (0, 0, 255)     # BGR – red flash on exit
LINE_THICKNESS      = 2

# Vertical offset (pixels) that a track's centroid must travel past the line
# before it is counted (hysteresis to avoid jitter re-counts).
CROSSING_BUFFER_PX  = 8

# ── Bounding Box Visualisation ───────────────────────────────────────────────
BBOX_COLOR     = (0, 255, 0)       # BGR green
BBOX_THICKNESS = 2
LABEL_FONT     = 0                 # cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE    = 0.55
LABEL_THICKNESS = 1

# ── GUI ───────────────────────────────────────────────────────────────────────
APP_TITLE        = "Real-Time Occupancy Monitoring System"
APP_WIDTH        = 1280
APP_HEIGHT       = 780
CTK_APPEARANCE   = "dark"          # "dark" | "light" | "system"
CTK_COLOR_THEME  = "blue"          # built-in theme

VIDEO_PANEL_W    = 720             # a bit smaller = less rendering work
VIDEO_PANEL_H    = 405

STATS_UPDATE_MS  = 500             # how often the stats panel refreshes (ms)
FPS_AVERAGE_FRAMES = 30            # rolling window for FPS calculation
FRAME_SKIP       = 3               # process every 3rd frame for speed

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_CSV_FILENAME = "occupancy_log.csv"
LOG_CSV_COLUMNS  = ["timestamp", "event", "track_id", "occupancy"]

# ── Analytics ────────────────────────────────────────────────────────────────
GRAPH_DPI        = 120
GRAPH_STYLE      = "dark_background"
GRAPH_ACCENT     = "#00C8FF"
GRAPH_ENTRY_CLR  = "#00FF88"
GRAPH_EXIT_CLR   = "#FF4455"
GRAPH_OCC_CLR    = "#FFAA00"

# ── PDF Report ────────────────────────────────────────────────────────────────
PDF_TITLE        = "Occupancy Monitoring – Session Report"
PDF_AUTHOR       = "Occupancy Monitoring System v1.0"
PDF_LOGO_FILE    = ASSETS_DIR / "logo.png"   # optional; skipped if missing
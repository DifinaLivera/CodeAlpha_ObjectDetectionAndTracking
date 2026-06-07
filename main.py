#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# main.py
#
# Entry-point for the Real-Time Occupancy Monitoring System.
#
# Usage:
#   python main.py
#
# The script configures logging, checks runtime dependencies, then launches
# the CustomTkinter GUI application.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
from pathlib import Path

# ── Ensure project root is on the path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging configuration ─────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
LOG_DATE   = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / "app.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_dependencies() -> bool:
    """
    Verify that mandatory runtime packages are importable.
    Prints a friendly error and returns False if any are missing.
    """
    required = {
        "cv2":              "opencv-python",
        "ultralytics":      "ultralytics",
        "deep_sort_realtime": "deep-sort-realtime",
        "customtkinter":    "customtkinter",
        "PIL":              "Pillow",
        "pandas":           "pandas",
        "numpy":            "numpy",
        "matplotlib":       "matplotlib",
        "reportlab":        "reportlab",
    }
    missing = []
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        logger.error(
            "Missing dependencies: %s\n"
            "Run:  pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Real-Time Occupancy Monitoring System – starting up")
    logger.info("=" * 60)

    if not _check_dependencies():
        sys.exit(1)

    # Ensure runtime directories exist (belt-and-suspenders)
    from config import settings  # noqa: F401

    logger.info("Launching GUI …")
    from gui.app import OccupancyApp

    app = OccupancyApp()
    app.mainloop()

    logger.info("Application closed.")


if __name__ == "__main__":
    main()
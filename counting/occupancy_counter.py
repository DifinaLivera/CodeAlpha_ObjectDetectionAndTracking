# ─────────────────────────────────────────────────────────────────────────────
# counting/occupancy_counter.py
#
# Virtual-line occupancy counter.
#
# A horizontal counting line is placed at a configurable fraction of the frame
# height.  For each confirmed track we remember the last y-position of its
# foot-point (bottom-centre of bounding box).  When the foot-point crosses the
# line in a downward direction (y increasing) the person is counted as ENTRY;
# upward (y decreasing) as EXIT.
#
# Duplicate counting is prevented by recording the IDs that have already
# triggered each direction.  A hysteresis buffer (CROSSING_BUFFER_PX pixels)
# prevents jitter from causing multiple counts for the same crossing event.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tracking.tracker import Track

logger = logging.getLogger(__name__)


class CrossingState:
    """Per-track state machine that detects a single line-crossing."""

    ABOVE = "above"    # foot-point was above the line last frame
    BELOW = "below"    # foot-point was below the line last frame
    NONE  = "none"     # not yet observed

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.side: str = CrossingState.NONE
        self.last_y: Optional[float] = None
        self.entry_counted: bool = False
        self.exit_counted: bool  = False


class OccupancyCounter:
    """
    Counts entries, exits and maintains real-time occupancy.

    Parameters
    ----------
    line_ratio      : Horizontal line at *line_ratio* × frame_height.
    buffer_px       : Min pixel distance past the line required to count.
    log_dir         : Directory where the CSV event log is written.
    log_filename    : Name of the CSV file.
    """

    def __init__(
        self,
        line_ratio: float = 0.55,
        buffer_px: int = 8,
        log_dir: Path | str = "logs",
        log_filename: str = "occupancy_log.csv",
    ) -> None:
        self.line_ratio  = line_ratio
        self.buffer_px   = buffer_px
        self.log_path    = Path(log_dir) / log_filename

        # Counters
        self._entries   = 0
        self._exits     = 0
        self._occupancy = 0

        # Per-track state
        self._states: Dict[int, CrossingState] = {}

        # Flash indicator for UI (set to "entry"/"exit"/None for one frame)
        self.last_event: Optional[str] = None

        # Full event log kept in memory for analytics
        self._event_log: List[Dict] = []

        # Sets of IDs that completed a crossing (for duplicate-guard)
        self._entered_ids: Set[int] = set()
        self._exited_ids:  Set[int] = set()

        # Prepare CSV
        self._init_csv()

        logger.info("OccupancyCounter ready. Log → %s", self.log_path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def entries(self) -> int:
        return self._entries

    @property
    def exits(self) -> int:
        return self._exits

    @property
    def occupancy(self) -> int:
        return max(0, self._occupancy)

    @property
    def active_ids(self) -> Set[int]:
        return set(self._states.keys())

    @property
    def event_log(self) -> List[Dict]:
        return list(self._event_log)

    # ── Core update ───────────────────────────────────────────────────────────

    def update(
        self,
        tracks: List[Track],
        frame_height: int,
        frame_width: int,
    ) -> Tuple[int, int, int]:
        """
        Process a list of confirmed tracks for the current frame.

        Returns
        -------
        (entries, exits, occupancy)
        """
        self.last_event = None
        line_y = int(frame_height * self.line_ratio)

        active_ids = {t.track_id for t in tracks}

        # Prune states for tracks that disappeared
        gone = set(self._states.keys()) - active_ids
        for tid in gone:
            del self._states[tid]

        for track in tracks:
            tid = track.track_id
            _, foot_y = track.center_bottom  # bottom-centre y

            # Initialise state on first sighting
            if tid not in self._states:
                st = CrossingState(tid)
                st.side   = CrossingState.ABOVE if foot_y < line_y else CrossingState.BELOW
                st.last_y = foot_y
                self._states[tid] = st
                continue

            st = self._states[tid]
            prev_y = st.last_y
            st.last_y = foot_y

            # ── Crossing detection ──────────────────────────────────────────
            # DOWN crossing (above → below)  → ENTRY
            if (
                prev_y is not None
                and prev_y <= line_y
                and foot_y > line_y + self.buffer_px
                and tid not in self._entered_ids
            ):
                self._entries   += 1
                self._occupancy += 1
                self._entered_ids.add(tid)
                self.last_event = "entry"
                self._log_event("ENTRY", tid)
                logger.debug("ENTRY  track_id=%d  entries=%d  occ=%d", tid, self._entries, self._occupancy)

            # UP crossing (below → above)  → EXIT
            elif (
                prev_y is not None
                and prev_y >= line_y
                and foot_y < line_y - self.buffer_px
                and tid not in self._exited_ids
            ):
                self._exits     += 1
                self._occupancy  = max(0, self._occupancy - 1)
                self._exited_ids.add(tid)
                self.last_event = "exit"
                self._log_event("EXIT", tid)
                logger.debug("EXIT   track_id=%d  exits=%d  occ=%d", tid, self._exits, self._occupancy)

        return self.entries, self.exits, self.occupancy

    # ── Line drawing ──────────────────────────────────────────────────────────

    def draw_line(
        self,
        frame: np.ndarray,
        color_normal=(0, 200, 255),
        color_entry=(0, 255, 0),
        color_exit=(0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Overlay the counting line on *frame* (in-place)."""
        import cv2

        h, w = frame.shape[:2]
        line_y = int(h * self.line_ratio)

        if self.last_event == "entry":
            colour = color_entry
        elif self.last_event == "exit":
            colour = color_exit
        else:
            colour = color_normal

        cv2.line(frame, (0, line_y), (w, line_y), colour, thickness)

        # Labels
        cv2.putText(
            frame, "IN ↓",
            (w - 80, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_entry, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "OUT ↑",
            (w - 90, line_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_exit, 1, cv2.LINE_AA,
        )
        return frame

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all counters (called on new video load)."""
        self._entries   = 0
        self._exits     = 0
        self._occupancy = 0
        self._states.clear()
        self._entered_ids.clear()
        self._exited_ids.clear()
        self._event_log.clear()
        self.last_event = None
        self._init_csv()
        logger.info("OccupancyCounter reset.")

    # ── CSV logging ──────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["timestamp", "event", "track_id", "occupancy"]
            )
            writer.writeheader()

    def _log_event(self, event: str, track_id: int) -> None:
        now = datetime.now()
        row = {
            "timestamp": now.isoformat(timespec="seconds"),
            "event":     event,
            "track_id":  track_id,
            "occupancy": self.occupancy,
        }
        self._event_log.append(row)
        try:
            with open(self.log_path, "a", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["timestamp", "event", "track_id", "occupancy"]
                )
                writer.writerow(row)
        except OSError as exc:
            logger.warning("Could not write to CSV log: %s", exc)

    # ── Export helpers ────────────────────────────────────────────────────────

    def export_session_csv(self, out_path: Path | str) -> Path:
        """Write a copy of the in-memory event log to *out_path*."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["timestamp", "event", "track_id", "occupancy"]
            )
            writer.writeheader()
            writer.writerows(self._event_log)
        logger.info("Session CSV exported → %s", out_path)
        return out_path
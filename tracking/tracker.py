# ─────────────────────────────────────────────────────────────────────────────
# tracking/tracker.py
#
# Deep SORT wrapper that converts DetectionResult objects into persistent
# Track objects with unique integer IDs.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detection.detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A confirmed Deep SORT track."""
    track_id: int
    bbox: List[float]          # [x1, y1, x2, y2]
    confidence: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def center_bottom(self):
        """Foot-point – more stable for line-crossing detection."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


class DeepSORTTracker:
    """
    Thin wrapper around deep_sort_realtime.DeepSort.

    Parameters mirror config/settings.py; all have sensible defaults so the
    class can be used standalone in unit tests.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        embedder: str = "mobilenet",        # light Re-ID model
    ) -> None:
        self.max_age            = max_age
        self.n_init             = n_init
        self.max_iou_distance   = max_iou_distance
        self.max_cosine_distance = max_cosine_distance

        logger.info("Initialising Deep SORT tracker …")
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                embedder=embedder,
                half=False,
                bgr=True,                   # our frames are BGR (OpenCV)
                embedder_gpu=False,
            )
            logger.info("Deep SORT initialised.")
        except Exception as exc:
            logger.exception("Failed to initialise Deep SORT: %s", exc)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self, detections: List[DetectionResult], frame: np.ndarray
    ) -> List[Track]:
        """
        Feed new detections into the tracker and return all confirmed tracks
        for the current frame.

        Parameters
        ----------
        detections : Raw detections from PersonDetector.detect()
        frame      : The BGR frame the detections came from (needed by Re-ID).

        Returns
        -------
        List[Track]  – only *confirmed* (n_init satisfied) tracks.
        """
        if frame is None or frame.size == 0:
            return []

        # deep_sort_realtime expects raw_detections as list of
        # ([left, top, w, h], confidence, class_name)
        raw = [
            (det.tlwh, det.confidence, "person")
            for det in detections
        ]

        try:
            ds_tracks = self._tracker.update_tracks(raw, frame=frame)
        except Exception as exc:
            logger.error("Deep SORT update error: %s", exc)
            return []

        tracks: List[Track] = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            try:
                ltrb = t.to_ltrb()          # [x1, y1, x2, y2]
                tracks.append(
                    Track(
                        track_id=int(t.track_id),
                        bbox=[float(v) for v in ltrb],
                        confidence=float(t.det_conf) if t.det_conf is not None else 1.0,
                    )
                )
            except Exception as inner:
                logger.warning("Skipping malformed track: %s", inner)

        return tracks

    # ── Draw helper ───────────────────────────────────────────────────────────

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: List[Track],
        color=(0, 255, 0),
        thickness: int = 2,
        label_scale: float = 0.55,
    ) -> np.ndarray:
        """
        Draw track bounding boxes + ID labels on *frame* (in-place).
        """
        import cv2

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track.track_id}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return frame
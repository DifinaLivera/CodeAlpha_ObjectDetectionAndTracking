# ─────────────────────────────────────────────────────────────────────────────
# detection/detector.py
#
# YOLOv8-based person detector.
# Returns a list of DetectionResult objects – each holding the bounding box,
# confidence score, and class id of every detected person in a frame.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """A single person detection produced by the detector."""
    bbox: List[float]          # [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_id: int = 0          # always 0 (person) for this system

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def tlwh(self) -> List[float]:
        """Return [top-left-x, top-left-y, width, height]."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]

    @property
    def xyxy(self) -> List[float]:
        return self.bbox

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PersonDetector:
    """
    Wraps YOLOv8 to provide single-call person detection.

    Parameters
    ----------
    model_name   : YOLO model file (downloaded automatically on first run).
    conf_thresh  : Minimum confidence to keep a detection.
    iou_thresh   : NMS IoU threshold.
    device       : Inference device – "cpu", "cuda", or "mps".
    img_size     : Square inference resolution.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
        person_class_id: int = 0,
    ) -> None:
        self.conf_thresh     = conf_thresh
        self.iou_thresh      = iou_thresh
        self.device          = device
        self.img_size        = img_size
        self.person_class_id = person_class_id

        logger.info("Loading YOLO model: %s on device=%s", model_name, device)
        try:
            from ultralytics import YOLO  # imported here to keep startup fast
            self._model = YOLO(model_name)
            self._model.to(device)
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on *frame* (BGR numpy array, HxWxC).

        Returns
        -------
        List[DetectionResult]
            Only person detections above conf_thresh.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                classes=[self.person_class_id],
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            logger.error("YOLO inference error: %s", exc)
            return []

        detections: List[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf             = float(box.conf[0])
                    cls              = int(box.cls[0])
                    detections.append(
                        DetectionResult(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=cls,
                        )
                    )
                except Exception as inner:
                    logger.warning("Skipping malformed box: %s", inner)

        return detections

    # ── Static draw helper ────────────────────────────────────────────────────

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[DetectionResult],
        color=(0, 255, 0),
        thickness: int = 2,
        label_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay bounding boxes + confidence labels on *frame* (in-place).
        Returns the same frame for convenience chaining.
        """
        import cv2

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"person {det.confidence:.2f}"
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
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from config import settings
from detection.detector import PersonDetector
from tracking.tracker import DeepSORTTracker
from counting.occupancy_counter import OccupancyCounter
from analytics.analytics import generate_all_graphs
from reports.report_generator import generate_pdf_report

logger = logging.getLogger(__name__)

VIDEO_POLL_MS = 30
STATS_POLL_MS = 500
FRAME_SKIP = settings.FRAME_SKIP if hasattr(settings, "FRAME_SKIP") else 2

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_STOPPED = "stopped"

_STOP_SENTINEL = None


class _FrameData:
    """Payload pushed from the video thread into the queue."""

    __slots__ = ("frame", "entries", "exits", "occupancy", "active_ids", "fps", "timestamp")

    def __init__(self, frame, entries, exits, occupancy, active_ids, fps, timestamp):
        self.frame = frame
        self.entries = entries
        self.exits = exits
        self.occupancy = occupancy
        self.active_ids = active_ids
        self.fps = fps
        self.timestamp = timestamp


class VideoThread(threading.Thread):
    """
    Background thread that reads video frames, runs detection + tracking,
    and pushes annotated frames into frame_queue.
    """

    def __init__(
        self,
        video_path: str,
        frame_queue: queue.Queue,
        counter: OccupancyCounter,
        detector: PersonDetector,
        tracker: DeepSORTTracker,
        display_w: int = settings.VIDEO_PANEL_W,
        display_h: int = settings.VIDEO_PANEL_H,
    ):
        super().__init__(daemon=True, name="VideoThread")
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.counter = counter
        self.detector = detector
        self.tracker = tracker
        self.display_w = display_w
        self.display_h = display_h

        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.pause_event.set()

        self._fps_history: list[float] = []

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", self.video_path)
            self.frame_queue.put(_STOP_SENTINEL)
            return

        logger.info("VideoThread started: %s", self.video_path)

        frame_idx = 0

        while not self.stop_event.is_set():
            self.pause_event.wait()
            if self.stop_event.is_set():
                break

            t0 = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                logger.info("Video ended or read error - stopping thread.")
                break

            frame_idx += 1

            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                continue

            h, w = frame.shape[:2]

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)

            entries, exits, occupancy = self.counter.update(tracks, h, w)

            annotated = frame.copy()
            self.counter.draw_line(
                annotated,
                color_normal=settings.LINE_COLOR_NORMAL,
                color_entry=settings.LINE_COLOR_ENTRY,
                color_exit=settings.LINE_COLOR_EXIT,
                thickness=settings.LINE_THICKNESS,
            )
            DeepSORTTracker.draw_tracks(
                annotated,
                tracks,
                color=settings.BBOX_COLOR,
                thickness=settings.BBOX_THICKNESS,
            )

            self._draw_hud(annotated, entries, exits, occupancy, len(tracks))

            annotated = cv2.resize(
                annotated,
                (self.display_w, self.display_h),
                interpolation=cv2.INTER_LINEAR,
            )

            elapsed = time.perf_counter() - t0
            self._fps_history.append(elapsed)
            if len(self._fps_history) > settings.FPS_AVERAGE_FRAMES:
                self._fps_history.pop(0)
            avg_fps = 1.0 / (sum(self._fps_history) / len(self._fps_history))

            payload = _FrameData(
                frame=annotated,
                entries=entries,
                exits=exits,
                occupancy=occupancy,
                active_ids=len(tracks),
                fps=avg_fps,
                timestamp=datetime.now(),
            )

            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                pass

        cap.release()
        self.frame_queue.put(_STOP_SENTINEL)
        logger.info("VideoThread finished.")

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        entries: int,
        exits: int,
        occupancy: int,
        active: int,
    ) -> None:
        """Lightweight on-frame HUD in the top-left corner."""
        lines = [
            f"Entries: {entries}",
            f"Exits: {exits}",
            f"Occupancy: {occupancy}",
            f"Active IDs: {active}",
        ]
        for i, line in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(
                frame,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.set()


class OccupancyApp(ctk.CTk):
    """
    Main CustomTkinter window.
    """

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode(settings.CTK_APPEARANCE)
        ctk.set_default_color_theme(settings.CTK_COLOR_THEME)

        self.title(settings.APP_TITLE)
        self.geometry(f"{settings.APP_WIDTH}x{settings.APP_HEIGHT}")
        self.resizable(True, True)
        self.minsize(900, 600)

        self._state = STATE_IDLE
        self._video_path: Optional[str] = None
        self._thread: Optional[VideoThread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._peak_occupancy = 0
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        self._detector = self._init_detector()
        self._tracker = self._init_tracker()
        self._counter = OccupancyCounter(
            line_ratio=settings.LINE_POSITION_RATIO,
            buffer_px=settings.CROSSING_BUFFER_PX,
            log_dir=settings.LOGS_DIR,
            log_filename=settings.LOG_CSV_FILENAME,
        )

        self._build_ui()
        self._update_button_states()
        self._tick_clock()

        logger.info("OccupancyApp ready.")

    def _init_detector(self) -> Optional[PersonDetector]:
        try:
            return PersonDetector(
                model_name=settings.YOLO_MODEL_NAME,
                conf_thresh=settings.YOLO_CONF_THRESH,
                iou_thresh=settings.YOLO_IOU_THRESH,
                device=settings.YOLO_DEVICE,
                img_size=settings.YOLO_IMG_SIZE,
                person_class_id=settings.YOLO_PERSON_CLASS,
            )
        except Exception as exc:
            logger.error("Detector init failed: %s", exc)
            messagebox.showerror("Model Error", f"Could not load YOLO model:\n{exc}")
            return None

    def _init_tracker(self) -> Optional[DeepSORTTracker]:
        try:
            return DeepSORTTracker(
                max_age=settings.DEEPSORT_MAX_AGE,
                n_init=settings.DEEPSORT_N_INIT,
                max_iou_distance=settings.DEEPSORT_MAX_IOU_DIST,
                max_cosine_distance=settings.DEEPSORT_MAX_COSINE_DIST,
            )
        except Exception as exc:
            logger.error("Tracker init failed: %s", exc)
            messagebox.showerror("Tracker Error", f"Could not initialise Deep SORT:\n{exc}")
            return None

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=260)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_video_panel()

    def _build_left_panel(self) -> None:
        self._left = ctk.CTkFrame(self, width=260, corner_radius=0)
        self._left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self._left.grid_propagate(False)
        self._left.grid_columnconfigure(0, weight=1)

        row = 0

        ctk.CTkLabel(
            self._left,
            text="Occupancy Monitor",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=row, column=0, padx=12, pady=(14, 4), sticky="ew")
        row += 1

        ctk.CTkLabel(
            self._left,
            text="YOLOv8 + Deep SORT",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        ).grid(row=row, column=0, padx=12, pady=(0, 12), sticky="ew")
        row += 1

        ctk.CTkFrame(self._left, height=1, fg_color="gray30").grid(
            row=row, column=0, sticky="ew", padx=8, pady=4
        )
        row += 1

        ctk.CTkLabel(
            self._left,
            text="CONTROLS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray",
        ).grid(row=row, column=0, padx=16, pady=(10, 4), sticky="w")
        row += 1

        btn_cfg = dict(corner_radius=8, height=38, font=ctk.CTkFont(size=13))

        self._btn_select = ctk.CTkButton(
            self._left,
            text="Select Video",
            command=self._on_select_video,
            **btn_cfg,
        )
        self._btn_select.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        self._btn_start = ctk.CTkButton(
            self._left,
            text="Start",
            command=self._on_start,
            fg_color="#2ECC71",
            hover_color="#27AE60",
            **btn_cfg,
        )
        self._btn_start.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        self._btn_pause = ctk.CTkButton(
            self._left,
            text="Pause",
            command=self._on_pause,
            fg_color="#E67E22",
            hover_color="#D35400",
            **btn_cfg,
        )
        self._btn_pause.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        self._btn_resume = ctk.CTkButton(
            self._left,
            text="Resume",
            command=self._on_resume,
            fg_color="#3498DB",
            hover_color="#2980B9",
            **btn_cfg,
        )
        self._btn_resume.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        self._btn_stop = ctk.CTkButton(
            self._left,
            text="Stop",
            command=self._on_stop,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            **btn_cfg,
        )
        self._btn_stop.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        self._btn_export = ctk.CTkButton(
            self._left,
            text="Export Report",
            command=self._on_export,
            fg_color="#9B59B6",
            hover_color="#8E44AD",
            **btn_cfg,
        )
        self._btn_export.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        row += 1

        ctk.CTkFrame(self._left, height=1, fg_color="gray30").grid(
            row=row, column=0, sticky="ew", padx=8, pady=(12, 4)
        )
        row += 1

        ctk.CTkLabel(
            self._left,
            text="STATISTICS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray",
        ).grid(row=row, column=0, padx=16, pady=(6, 4), sticky="w")
        row += 1

        self._stats_frame = ctk.CTkFrame(
            self._left,
            corner_radius=8,
            fg_color=("gray90", "gray17"),
        )
        self._stats_frame.grid(row=row, column=0, padx=12, pady=4, sticky="ew")
        row += 1
        self._stats_frame.grid_columnconfigure((0, 1), weight=1)

        stat_items = [
            ("Date", "date_val"),
            ("Time", "time_val"),
            ("Entries", "entries_val"),
            ("Exits", "exits_val"),
            ("Occupancy", "occ_val"),
            ("Active IDs", "ids_val"),
            ("FPS", "fps_val"),
        ]

        self._stat_vars: dict = {}
        for i, (label, key) in enumerate(stat_items):
            ctk.CTkLabel(
                self._stats_frame,
                text=label + ":",
                font=ctk.CTkFont(size=11),
                anchor="w",
            ).grid(row=i, column=0, padx=(10, 2), pady=3, sticky="w")

            var = ctk.StringVar(value="-")
            self._stat_vars[key] = var
            ctk.CTkLabel(
                self._stats_frame,
                textvariable=var,
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="e",
            ).grid(row=i, column=1, padx=(2, 10), pady=3, sticky="e")

        ctk.CTkFrame(self._left, height=1, fg_color="gray30").grid(
            row=row, column=0, sticky="ew", padx=8, pady=(12, 4)
        )
        row += 1

        self._video_label = ctk.CTkLabel(
            self._left,
            text="No video selected",
            font=ctk.CTkFont(size=9),
            text_color="gray",
            wraplength=230,
        )
        self._video_label.grid(row=row, column=0, padx=12, pady=(4, 12), sticky="ew")

    def _build_video_panel(self) -> None:
        self._right = ctk.CTkFrame(self, corner_radius=8)
        self._right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self._right.grid_columnconfigure(0, weight=1)
        self._right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self._right,
            text="LIVE FEED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray",
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self._canvas = ctk.CTkLabel(
            self._right,
            text="Select a video and press Start",
            font=ctk.CTkFont(size=14),
            fg_color=("gray85", "gray10"),
            corner_radius=6,
            width=settings.VIDEO_PANEL_W,
            height=settings.VIDEO_PANEL_H,
        )
        self._canvas.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")

        self._status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(
            self._right,
            textvariable=self._status_var,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w",
        ).grid(row=2, column=0, padx=12, pady=(0, 8), sticky="ew")

    def _on_select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=str(settings.VIDEOS_DIR),
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self._video_path = path
        short = Path(path).name
        self._video_label.configure(text=f"Video: {short}")
        self._status_var.set(f"Loaded: {short}")
        self._state = STATE_IDLE
        self._update_button_states()

    def _on_start(self) -> None:
        if not self._video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        if self._detector is None or self._tracker is None:
            messagebox.showerror("Init Error", "Detector or Tracker not initialised.")
            return

        self._counter.reset()
        self._peak_occupancy = 0

        self._tracker = self._init_tracker()
        if self._tracker is None:
            return

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._thread = VideoThread(
            video_path=self._video_path,
            frame_queue=self._frame_queue,
            counter=self._counter,
            detector=self._detector,
            tracker=self._tracker,
            display_w=settings.VIDEO_PANEL_W,
            display_h=settings.VIDEO_PANEL_H,
        )
        self._thread.start()

        self._state = STATE_RUNNING
        self._update_button_states()
        self._status_var.set("Running...")
        self._poll_frames()

    def _on_pause(self) -> None:
        if self._thread and self._state == STATE_RUNNING:
            self._thread.pause()
            self._state = STATE_PAUSED
            self._update_button_states()
            self._status_var.set("Paused")

    def _on_resume(self) -> None:
        if self._thread and self._state == STATE_PAUSED:
            self._thread.resume()
            self._state = STATE_RUNNING
            self._update_button_states()
            self._status_var.set("Running...")
            self._poll_frames()

    def _on_stop(self) -> None:
        if self._thread:
            self._thread.stop()
            self._thread.join(timeout=3)
        self._state = STATE_STOPPED
        self._update_button_states()
        self._status_var.set("Stopped")

    def _on_export(self) -> None:
        if not self._counter.event_log:
            messagebox.showinfo(
                "No Data",
                "No events recorded yet. Run the system first to collect data.",
            )
            return

        out_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=str(settings.REPORTS_DIR),
        )
        if not out_dir:
            return

        out_dir = Path(out_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        graphs_dir = out_dir / f"graphs_{ts}"
        pdf_path = out_dir / f"report_{ts}.pdf"
        csv_path = out_dir / f"events_{ts}.csv"

        self._status_var.set("Generating report...")
        self.update_idletasks()

        try:
            self._counter.export_session_csv(csv_path)
            generate_all_graphs(self._counter.event_log, graphs_dir)
            generate_pdf_report(
                event_log=self._counter.event_log,
                entries=self._counter.entries,
                exits=self._counter.exits,
                peak_occupancy=self._peak_occupancy,
                graphs_dir=graphs_dir,
                out_path=pdf_path,
                logo_path=settings.PDF_LOGO_FILE,
            )

            self._status_var.set(f"Report saved -> {out_dir.name}/")
            messagebox.showinfo(
                "Export Complete",
                f"Report successfully exported!\n\n"
                f"PDF   : {pdf_path.name}\n"
                f"CSV   : {csv_path.name}\n"
                f"Graphs: {graphs_dir.name}/",
            )
        except Exception as exc:
            logger.exception("Export failed: %s", exc)
            self._status_var.set("Export failed")
            messagebox.showerror("Export Error", str(exc))

    def _poll_frames(self) -> None:
        if self._state not in (STATE_RUNNING,):
            return

        payload = None
        while True:
            try:
                payload = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if payload is None:
            self.after(VIDEO_POLL_MS, self._poll_frames)
            return

        if payload is _STOP_SENTINEL:
            self._state = STATE_STOPPED
            self._update_button_states()
            self._status_var.set("Video finished")
            return

        frame_rgb = cv2.cvtColor(payload.frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_img)
        self._canvas.configure(image=photo, text="")
        self._current_photo = photo

        self._stat_vars["entries_val"].set(str(payload.entries))
        self._stat_vars["exits_val"].set(str(payload.exits))
        self._stat_vars["occ_val"].set(str(payload.occupancy))
        self._stat_vars["ids_val"].set(str(payload.active_ids))
        self._stat_vars["fps_val"].set(f"{payload.fps:.1f}")

        if payload.occupancy > self._peak_occupancy:
            self._peak_occupancy = payload.occupancy

        self.after(VIDEO_POLL_MS, self._poll_frames)

    def _tick_clock(self) -> None:
        now = datetime.now()
        self._stat_vars["date_val"].set(now.strftime("%d %b %Y"))
        self._stat_vars["time_val"].set(now.strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _update_button_states(self) -> None:
        s = self._state
        self._btn_select.configure(state="normal")
        self._btn_start.configure(state="normal" if s in (STATE_IDLE, STATE_STOPPED) else "disabled")
        self._btn_pause.configure(state="normal" if s == STATE_RUNNING else "disabled")
        self._btn_resume.configure(state="normal" if s == STATE_PAUSED else "disabled")
        self._btn_stop.configure(state="normal" if s in (STATE_RUNNING, STATE_PAUSED) else "disabled")
        self._btn_export.configure(state="normal" if s in (STATE_STOPPED, STATE_IDLE) else "disabled")

    def destroy(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.stop()
            self._thread.join(timeout=2)
        super().destroy()


def main() -> None:
    app = OccupancyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
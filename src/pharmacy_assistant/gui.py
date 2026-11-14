"""Tkinter desktop dashboard for live prescription scanning."""

import base64
from dataclasses import dataclass
from datetime import datetime
import queue
import shutil
import sqlite3
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from pharmacy_assistant.database import connect
from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.prescription import (
    PrescriptionError,
    PrescriptionScanResult,
    preprocess_image,
    scan_image_array,
)
from pharmacy_assistant.repository import MedicineRepository

NAVY = "#172B4D"
BLUE = "#2563EB"
BACKGROUND = "#F5F7FA"
WHITE = "#FFFFFF"
TEXT = "#172B4D"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SUCCESS = "#16803C"
WARNING = "#A05A00"
ERROR = "#B42318"


@dataclass
class ScannerState:
    """Keep live preview and captured scan data in independent fields."""

    live_frame: np.ndarray | None = None
    captured_image: np.ndarray | None = None
    captured_at: datetime | None = None
    result: PrescriptionScanResult | None = None
    generation: int = 0
    status: str = "connecting"

    def update_live(self, frame: np.ndarray) -> None:
        self.live_frame = frame

    def begin_capture(self, captured_at: datetime) -> int:
        self.generation += 1
        self.captured_image = None
        self.captured_at = captured_at
        self.result = None
        self.status = "processing"
        return self.generation

    def set_captured_image(self, generation: int, image: np.ndarray) -> bool:
        if generation != self.generation:
            return False
        self.captured_image = image
        return True

    def set_result(
        self, generation: int, result: PrescriptionScanResult
    ) -> bool:
        if generation != self.generation:
            return False
        self.result = result
        self.status = "result"
        return True

    def set_error(self, generation: int) -> bool:
        if generation != self.generation:
            return False
        self.status = "error"
        return True

    def clear(self) -> None:
        self.generation += 1
        self.captured_image = None
        self.captured_at = None
        self.result = None
        self.status = "ready"


class CameraWorker(threading.Thread):
    """Read webcam frames off the UI thread, retaining only the latest frame."""

    def __init__(
        self,
        camera_index: int,
        frame_queue: queue.Queue[np.ndarray],
        event_queue: queue.Queue[tuple[str, object]],
    ) -> None:
        super().__init__(name="pharmacy-camera", daemon=True)
        self.camera_index = camera_index
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def _publish_frame(self, frame: np.ndarray) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def run(self) -> None:
        camera = None
        try:
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                self.event_queue.put(
                    ("camera_error", f"Could not open camera {self.camera_index}.")
                )
                return

            while not self.stop_event.is_set():
                success, frame = camera.read()
                if not success or frame is None or frame.size == 0:
                    if not self.stop_event.is_set():
                        self.event_queue.put(
                            (
                                "camera_error",
                                f"Camera {self.camera_index} stopped providing frames.",
                            )
                        )
                    return
                self._publish_frame(frame)
        except cv2.error as error:
            self.event_queue.put(("camera_error", f"Camera error: {error}"))
        finally:
            if camera is not None:
                camera.release()


class PharmacyDashboard:
    """Professional scanner dashboard with isolated live/captured image panels."""

    def __init__(
        self,
        root: tk.Tk,
        database_path: str | Path,
        camera_index: int = 0,
        ocr_model: str | None = None,
    ) -> None:
        self.root = root
        self.database_path = Path(database_path).expanduser()
        self.camera_index = camera_index
        self.ocr_model = ocr_model
        self.state = ScannerState()
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.camera_worker: CameraWorker | None = None
        self.scan_in_progress = False
        self.live_photo: tk.PhotoImage | None = None
        self.captured_photo: tk.PhotoImage | None = None
        self._closed = False

        self.root.title("Pharmacy Assistant | Prescription Scanner")
        self.root.geometry("1480x920")
        self.root.minsize(1180, 760)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_styles()
        self._build_layout()
        self._check_database()
        self._check_ocr_engine()
        self._start_camera()
        self.root.after(30, self._poll_camera)
        self.root.after(60, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure(
            "Card.TFrame",
            background=WHITE,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Primary.TButton",
            background=BLUE,
            foreground=WHITE,
            padding=(18, 12),
            borderwidth=0,
            font=("TkDefaultFont", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("disabled", "#94A3B8"), ("active", "#1D4ED8")],
            foreground=[("disabled", WHITE)],
        )
        style.configure(
            "Secondary.TButton",
            background=WHITE,
            foreground=TEXT,
            padding=(16, 11),
            borderwidth=1,
            font=("TkDefaultFont", 10, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#EFF6FF"), ("disabled", "#F1F5F9")],
            foreground=[("disabled", "#94A3B8")],
        )

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0, minsize=375)

        self.sidebar = tk.Frame(self.root, bg=NAVY, width=228)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        self.main_area = ttk.Frame(self.root, padding=(26, 24, 14, 24))
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)
        self._build_main_panel()

        self.results_area = ttk.Frame(self.root, padding=(12, 24, 24, 24))
        self.results_area.grid(row=0, column=2, sticky="nsew")
        self.results_area.grid_rowconfigure(1, weight=1)
        self.results_area.grid_columnconfigure(0, weight=1)
        self._build_results_panel()

    def _build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=NAVY)
        brand.pack(fill="x", padx=20, pady=(26, 35))
        mark = tk.Label(
            brand,
            text="PA",
            bg="#24436E",
            fg=WHITE,
            font=("TkDefaultFont", 13, "bold"),
            width=3,
            height=2,
        )
        mark.pack(side="left")
        brand_text = tk.Frame(brand, bg=NAVY)
        brand_text.pack(side="left", padx=(10, 0))
        tk.Label(
            brand_text,
            text="Pharmacy",
            bg=NAVY,
            fg=WHITE,
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand_text,
            text="ASSISTANT",
            bg=NAVY,
            fg="#B9C9DF",
            font=("TkDefaultFont", 8, "bold"),
        ).pack(anchor="w")

        tk.Label(
            self.sidebar,
            text="WORKSPACE",
            bg=NAVY,
            fg="#91A6C3",
            font=("TkDefaultFont", 8, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 12))
        self._nav_item("Dashboard", active=False)
        self._nav_item("Prescription Scanner", active=True)

        tk.Frame(self.sidebar, bg="#314A6A", height=1).pack(
            fill="x", padx=20, pady=(26, 18)
        )
        tk.Label(
            self.sidebar,
            text="SYSTEM STATUS",
            bg=NAVY,
            fg="#91A6C3",
            font=("TkDefaultFont", 8, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 13))

        self.database_status = self._status_row("Database", "Checking")
        self.camera_status = self._status_row("Camera", "Connecting")
        self.ocr_status = self._status_row("OCR engine", "Checking")

        tk.Label(
            self.sidebar,
            text="Review-only workflow\nNo medicine is dispensed.",
            bg=NAVY,
            fg="#B9C9DF",
            justify="left",
            font=("TkDefaultFont", 9),
        ).pack(side="bottom", anchor="w", padx=22, pady=22)

    def _nav_item(self, text: str, active: bool) -> None:
        frame = tk.Frame(
            self.sidebar,
            bg="#24436E" if active else NAVY,
            highlightthickness=0,
        )
        frame.pack(fill="x", padx=12, pady=3)
        indicator = "●" if active else "○"
        tk.Label(
            frame,
            text=indicator,
            bg=frame["bg"],
            fg="#78A8FF" if active else "#A9B8CB",
            font=("TkDefaultFont", 10),
        ).pack(side="left", padx=(12, 10), pady=11)
        tk.Label(
            frame,
            text=text,
            bg=frame["bg"],
            fg=WHITE if active else "#D3DCE8",
            font=("TkDefaultFont", 9, "bold" if active else "normal"),
        ).pack(side="left", pady=11)

    def _status_row(self, label: str, initial: str) -> tk.Label:
        row = tk.Frame(self.sidebar, bg=NAVY)
        row.pack(fill="x", padx=22, pady=6)
        tk.Label(
            row,
            text=label,
            bg=NAVY,
            fg="#D3DCE8",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w")
        value = tk.Label(
            row,
            text=f"●  {initial}",
            bg=NAVY,
            fg="#F5C76A",
            font=("TkDefaultFont", 8),
        )
        value.pack(anchor="w", pady=(3, 0))
        return value

    def _build_main_panel(self) -> None:
        header = ttk.Frame(self.main_area)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Prescription Scanner",
            bg=BACKGROUND,
            fg=TEXT,
            font=("TkDefaultFont", 22, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Capture printed medicine details for careful review.",
            bg=BACKGROUND,
            fg=MUTED,
            font=("TkDefaultFont", 10),
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.state_badge = tk.Label(
            header,
            text="●  CONNECTING",
            bg="#FFF7E6",
            fg=WARNING,
            padx=12,
            pady=8,
            font=("TkDefaultFont", 8, "bold"),
        )
        self.state_badge.grid(row=0, column=1, rowspan=2, sticky="e")

        card = ttk.Frame(self.main_area, style="Card.TFrame", padding=18)
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        top = tk.Frame(card, bg=WHITE)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            top,
            text="LIVE CAMERA",
            bg=WHITE,
            fg=TEXT,
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left")
        self.live_indicator = tk.Label(
            top,
            text="●  Connecting",
            bg=WHITE,
            fg=WARNING,
            font=("TkDefaultFont", 9),
        )
        self.live_indicator.pack(side="right")

        self.live_image_label = tk.Label(
            card,
            text="Connecting to camera…",
            bg="#E9EEF5",
            fg=MUTED,
            font=("TkDefaultFont", 12),
            compound="center",
            anchor="center",
        )
        self.live_image_label.grid(row=1, column=0, sticky="nsew")
        self.live_image_label.bind("<Configure>", self._on_live_resize)

        guide = tk.Frame(card, bg=WHITE)
        guide.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        tk.Label(
            guide,
            text="Tip: keep the medicine name in focus and fill the frame.",
            bg=WHITE,
            fg=MUTED,
            font=("TkDefaultFont", 9),
        ).pack(anchor="w")

        controls = ttk.Frame(self.main_area)
        controls.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.capture_button = ttk.Button(
            controls,
            text="Capture image",
            style="Primary.TButton",
            command=self.capture,
        )
        self.capture_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.retake_button = ttk.Button(
            controls,
            text="Retake",
            style="Secondary.TButton",
            command=self.retake,
        )
        self.retake_button.grid(row=0, column=1, padx=4)
        self.clear_button = ttk.Button(
            controls,
            text="Clear scan",
            style="Secondary.TButton",
            command=self.clear_scan,
        )
        self.clear_button.grid(row=0, column=2, padx=(4, 0))
        self.capture_button.state(["disabled"])

    def _build_results_panel(self) -> None:
        title = tk.Label(
            self.results_area,
            text="Captured scan",
            bg=BACKGROUND,
            fg=TEXT,
            font=("TkDefaultFont", 15, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        panel = ttk.Frame(self.results_area, style="Card.TFrame", padding=14)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_rowconfigure(4, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        tk.Label(
            panel,
            text="PROCESSED GRAYSCALE",
            bg=WHITE,
            fg=TEXT,
            font=("TkDefaultFont", 8, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.captured_image_label = tk.Label(
            panel,
            text="Your captured image will appear here.",
            bg="#E9EEF5",
            fg=MUTED,
            justify="center",
            wraplength=300,
            font=("TkDefaultFont", 10),
        )
        self.captured_image_label.grid(
            row=1, column=0, sticky="nsew", pady=(9, 8)
        )
        self.captured_image_label.bind("<Configure>", self._on_captured_resize)
        self.timestamp_label = tk.Label(
            panel,
            text="No image captured",
            bg=WHITE,
            fg=MUTED,
            font=("TkDefaultFont", 8),
        )
        self.timestamp_label.grid(row=2, column=0, sticky="w", pady=(0, 14))

        tk.Frame(panel, bg=BORDER, height=1).grid(
            row=3, column=0, sticky="ew", pady=(0, 12)
        )
        results_header = tk.Frame(panel, bg=WHITE)
        results_header.grid(row=4, column=0, sticky="new")
        results_header.grid_columnconfigure(0, weight=1)
        tk.Label(
            results_header,
            text="SCAN RESULTS",
            bg=WHITE,
            fg=TEXT,
            font=("TkDefaultFont", 8, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.result_status = tk.Label(
            results_header,
            text="Waiting for capture",
            bg="#F1F5F9",
            fg=MUTED,
            padx=8,
            pady=5,
            font=("TkDefaultFont", 8, "bold"),
        )
        self.result_status.grid(row=0, column=1, sticky="e")
        self.results_text = tk.Text(
            panel,
            height=12,
            wrap="word",
            bg=WHITE,
            fg=TEXT,
            relief="flat",
            borderwidth=0,
            padx=0,
            pady=12,
            font=("TkDefaultFont", 9),
            state="disabled",
        )
        self.results_text.grid(row=5, column=0, sticky="nsew")
        results_scrollbar = ttk.Scrollbar(
            panel, orient="vertical", command=self.results_text.yview
        )
        results_scrollbar.grid(row=5, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        panel.grid_rowconfigure(5, weight=2)
        self.results_text.tag_configure("section", font=("TkDefaultFont", 9, "bold"))
        self.results_text.tag_configure("muted", foreground=MUTED)
        self.results_text.tag_configure("success", foreground=SUCCESS)
        self.results_text.tag_configure("warning", foreground=WARNING)
        self.results_text.tag_configure("error", foreground=ERROR)

    def _check_database(self) -> None:
        if not self.database_path.is_file():
            self._set_system_status(
                self.database_status, "Not found", ERROR
            )
            self.result_status.configure(text="Database unavailable")
            return
        try:
            with connect(self.database_path) as connection:
                connection.execute("SELECT id FROM medicines LIMIT 0")
            self._set_system_status(self.database_status, "Connected", SUCCESS)
        except sqlite3.Error as error:
            self._set_system_status(self.database_status, "Error", ERROR)
            self._show_error(f"Database could not be opened: {error}")

    def _check_ocr_engine(self) -> None:
        def check() -> None:
            executable = shutil.which("tesseract")
            if executable is None:
                self.event_queue.put(
                    ("ocr_status", ("Unavailable", ERROR))
                )
                return
            try:
                from pytesseract import get_tesseract_version

                version = str(get_tesseract_version()).splitlines()[0]
            except (OSError, RuntimeError) as error:
                self.event_queue.put(
                    ("ocr_status", ("Unavailable", ERROR))
                )
                self.event_queue.put(
                    ("system_error", f"Tesseract could not start: {error}")
                )
                return
            self.event_queue.put(("ocr_status", (f"Ready · {version}", SUCCESS)))

        threading.Thread(target=check, name="pharmacy-ocr-check", daemon=True).start()

    def _set_system_status(
        self, widget: tk.Label, message: str, color: str
    ) -> None:
        widget.configure(text=f"●  {message}", fg=color)

    def _start_camera(self) -> None:
        self.camera_worker = CameraWorker(
            self.camera_index, self.frame_queue, self.event_queue
        )
        self.camera_worker.start()

    def _poll_camera(self) -> None:
        if self._closed:
            return
        newest_frame = None
        while True:
            try:
                newest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if newest_frame is not None:
            self.state.update_live(newest_frame)
            self._render_live()
            self._set_system_status(self.camera_status, "Connected", SUCCESS)
            self.live_indicator.configure(text="●  Camera ready", fg=SUCCESS)
            if not self.scan_in_progress:
                self.capture_button.state(["!disabled"])
            if self.state.status in ("connecting", "ready"):
                self.state.status = "ready"
                self._set_workflow_status("READY", SUCCESS)
        self.root.after(30, self._poll_camera)

    def _poll_events(self) -> None:
        if self._closed:
            return
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if event == "camera_error":
                self._set_system_status(self.camera_status, "Unavailable", ERROR)
                self.live_indicator.configure(text="●  Camera unavailable", fg=ERROR)
                self.live_image_label.configure(
                    image="",
                    text="Camera unavailable\n\nCheck the connection and permissions.",
                )
                self._set_workflow_status("CAMERA ERROR", ERROR)
                self.capture_button.state(["disabled"])
                self._show_error(str(payload))
            elif event == "ocr_status":
                message, color = payload
                self._set_system_status(self.ocr_status, message, color)
            elif event == "system_error":
                self._show_error(str(payload))
            elif event == "image_ready":
                generation, image = payload
                if self.state.set_captured_image(generation, image):
                    self._render_captured()
            elif event == "scan_done":
                generation, result = payload
                self.scan_in_progress = False
                if self.state.set_result(generation, result):
                    self._display_result(result)
                    self._set_workflow_status("REVIEW RESULT", BLUE)
                if self.state.live_frame is not None:
                    self.capture_button.state(["!disabled"])
            elif event == "scan_error":
                generation, message = payload
                self.scan_in_progress = False
                if self.state.set_error(generation):
                    self._show_error(message)
                    self._set_workflow_status("REVIEW REQUIRED", ERROR)
                    self._set_result_badge("Scan failed", "#FEF0EF", ERROR)
                    self._write_results(
                        "The image remains available for review.\n\n" + message,
                        "error",
                    )
                if self.state.live_frame is not None:
                    self.capture_button.state(["!disabled"])
        self.root.after(50, self._poll_events)

    def _render_live(
        self, width: int | None = None, height: int | None = None
    ) -> None:
        if self.state.live_frame is None:
            return
        self.live_photo = self._photo_for(
            self.state.live_frame,
            width or self.live_image_label.winfo_width(),
            height or self.live_image_label.winfo_height(),
        )
        self.live_image_label.configure(image=self.live_photo, text="")

    def _render_captured(
        self, width: int | None = None, height: int | None = None
    ) -> None:
        if self.state.captured_image is None:
            self.captured_photo = None
            self.captured_image_label.configure(
                image="", text="Your captured image will appear here."
            )
            return
        self.captured_photo = self._photo_for(
            self.state.captured_image,
            width or self.captured_image_label.winfo_width(),
            height or self.captured_image_label.winfo_height(),
        )
        self.captured_image_label.configure(image=self.captured_photo, text="")
        if self.state.captured_at is not None:
            self.timestamp_label.configure(
                text=f"Captured {self.state.captured_at.strftime('%d %b %Y · %H:%M:%S')}"
            )

    @staticmethod
    def _photo_for(
        image: np.ndarray, max_width: int, max_height: int
    ) -> tk.PhotoImage:
        height, width = image.shape[:2]
        max_width = max(max_width, 100)
        max_height = max(max_height, 100)
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1:
            resized = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image
        extension = ".png"
        success, encoded = cv2.imencode(extension, resized)
        if not success:
            raise RuntimeError("Could not encode the image for display.")
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return tk.PhotoImage(data=data, format="png")

    def _on_live_resize(self, event: tk.Event) -> None:
        if self.state.live_frame is not None:
            self._render_live(event.width, event.height)

    def _on_captured_resize(self, event: tk.Event) -> None:
        if self.state.captured_image is not None:
            self._render_captured(event.width, event.height)

    def capture(self) -> None:
        if self.scan_in_progress:
            return
        if self.state.live_frame is None:
            self._show_error("Camera is not ready. Check the camera connection.")
            return
        if not self.database_path.is_file():
            self._show_error(
                f"Database not found: {self.database_path}\n"
                "Initialize a database before scanning."
            )
            return

        frame = self.state.live_frame.copy()
        captured_at = datetime.now()
        generation = self.state.begin_capture(captured_at)
        self.scan_in_progress = True
        self._render_captured()
        self.timestamp_label.configure(text="Preparing captured image…")
        self.result_status.configure(text="Processing", bg="#EFF6FF", fg=BLUE)
        self._write_results(
            "Image captured. Preparing the OCR image and checking the medicine catalog…",
            "muted",
        )
        self._set_workflow_status("PROCESSING", BLUE)
        self.capture_button.state(["disabled"])
        self._start_scan_worker(generation, frame)

    def _start_scan_worker(self, generation: int, frame: np.ndarray) -> None:
        def scan() -> None:
            try:
                grayscale = preprocess_image(frame)["grayscale"].copy()
                self.event_queue.put(("image_ready", (generation, grayscale)))
                with connect(self.database_path) as connection:
                    matcher = MedicineMatcher(connection)
                    repository = MedicineRepository(connection)
                    result = scan_image_array(
                        frame,
                        matcher,
                        repository,
                        ocr_model=self.ocr_model,
                    )
                self.event_queue.put(("scan_done", (generation, result)))
            except (PrescriptionError, sqlite3.Error, OSError, cv2.error) as error:
                self.event_queue.put(("scan_error", (generation, str(error))))

        threading.Thread(
            target=scan,
            name=f"pharmacy-scan-{generation}",
            daemon=True,
        ).start()

    def retake(self) -> None:
        self._clear_capture("Ready for a new capture.")

    def clear_scan(self) -> None:
        self._clear_capture("Scan cleared. Capture an image to begin.")

    def _clear_capture(self, message: str) -> None:
        self.state.clear()
        self._render_captured()
        self.timestamp_label.configure(text="No image captured")
        self._set_result_badge("Waiting for capture", MUTED)
        self._write_results(message, "muted")
        if self.state.live_frame is not None:
            self._set_workflow_status("READY", SUCCESS)
            if not self.scan_in_progress:
                self.capture_button.state(["!disabled"])
        else:
            self.state.status = "connecting"
            self._set_workflow_status("CONNECTING", WARNING)

    def _display_result(self, result: PrescriptionScanResult) -> None:
        if not result.raw_text.strip():
            self._set_result_badge("No text recognized", ERROR)
            self._write_results(
                "No readable text was returned by OCR.\n"
                "Improve focus and lighting, then retake the image.",
                "error",
            )
            return

        segments: list[tuple[str, str]] = []
        medicine_statuses: list[str] = []
        for index, scanned in enumerate(result.items, start=1):
            candidate = scanned.candidate
            name = candidate.name or "Medicine name unclear"
            segments.append((f"Medicine {index}: {name}\n", "section"))
            if candidate.strength:
                segments.append((f"Strength: {candidate.strength}\n", ""))
            if candidate.dosage_form:
                segments.append((f"Dosage form: {candidate.dosage_form}\n", ""))
            status, tag = self._match_status(scanned.database_status)
            medicine_statuses.append(scanned.database_status)
            segments.append((f"Database: {status}\n", tag))
            if scanned.medicine is not None:
                medicine = scanned.medicine
                segments.append((f"Generic: {medicine.generic_name}\n", ""))
                if medicine.brand_name:
                    segments.append((f"Brand: {medicine.brand_name}\n", ""))
                segments.append((f"Strength: {medicine.strength}\n", ""))
                segments.append((f"Dosage form: {medicine.dosage_form}\n", ""))
                if medicine.common_uses:
                    segments.append(
                        (f"Common uses: {medicine.common_uses}\n", "")
                    )
                if medicine.important_notes:
                    segments.append(
                        (f"Important note: {medicine.important_notes}\n", "warning")
                    )
                if medicine.information_source:
                    segments.append(
                        (f"Source: {medicine.information_source}\n", "muted")
                    )
                    if medicine.information_reviewed:
                        segments.append(
                            (f"Information reviewed: {medicine.information_reviewed}\n", "muted")
                        )
            elif scanned.reference is not None:
                reference = scanned.reference
                segments.append(
                    (
                        f"Generic ingredient (product details unconfirmed): "
                        f"{reference.generic_name}\n",
                        "warning",
                    )
                )
                segments.append((f"Common uses: {reference.common_uses}\n", ""))
                segments.append(
                    (f"Important note: {reference.important_notes}\n", "warning")
                )
                segments.append((f"Source: {reference.information_source}\n", "muted"))
            if scanned.location is not None:
                segments.append((f"Location: {scanned.location.code}\n", ""))
            if scanned.quantity_available is not None:
                segments.append((f"Available: {scanned.quantity_available}\n", ""))
            if scanned.review_fields:
                segments.append(
                    (
                        f"Verify: {', '.join(scanned.review_fields)}\n",
                        "warning",
                    )
                )
            segments.append(("\n", ""))

        if not result.items:
            segments.append(("No medicine candidate was identified.\n", "warning"))
            medicine_statuses.append("not_found")

        segments.append(("Raw OCR output\n", "section"))
        segments.append((result.raw_text.strip() + "\n", ""))
        segments.append(("\nReview only. No inventory or dispensing action was taken.", "muted"))
        overall_status = self._overall_status(medicine_statuses)
        self._set_result_badge(*overall_status)
        self._write_results(*self._flatten_segments(segments))

    @staticmethod
    def _flatten_segments(
        segments: list[tuple[str, str]]
    ) -> tuple[str, list[tuple[str, int, int]]]:
        text = ""
        ranges: list[tuple[str, int, int]] = []
        for content, tag in segments:
            start = len(text)
            text += content
            if tag:
                ranges.append((tag, start, len(text)))
        return text, ranges

    def _write_results(
        self,
        text: str,
        tags: str | list[tuple[str, int, int]] = "",
    ) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        if isinstance(tags, list):
            for tag, start, end in tags:
                self.results_text.tag_add(
                    tag, f"1.0+{start}c", f"1.0+{end}c"
                )
        elif tags:
            self.results_text.tag_add(tags, "1.0", "end")
        self.results_text.configure(state="disabled")

    @staticmethod
    def _match_status(status: str) -> tuple[str, str]:
        if status == "matched":
            return "Matched", "success"
        if status == "not_found":
            return "No reliable match", "error"
        if status == "matched_no_inventory":
            return "Matched; inventory not recorded", "warning"
        return "Requires verification", "warning"

    @staticmethod
    def _overall_status(statuses: list[str]) -> tuple[str, str, str]:
        if statuses and all(status == "matched" for status in statuses):
            return "Database match", "#EAF7EE", SUCCESS
        if statuses and all(status == "not_found" for status in statuses):
            return "No reliable match", "#FEF0EF", ERROR
        return "Review required", "#FFF7E6", WARNING

    def _set_result_badge(
        self, message: str, color: str, foreground: str | None = None
    ) -> None:
        self.result_status.configure(
            text=message,
            bg=color,
            fg=foreground or color,
        )

    def _set_workflow_status(self, label: str, color: str) -> None:
        background = {
            SUCCESS: "#EAF7EE",
            WARNING: "#FFF7E6",
            ERROR: "#FEF0EF",
            BLUE: "#EFF6FF",
        }.get(color, "#F1F5F9")
        self.state_badge.configure(
            text=f"●  {label}", bg=background, fg=color
        )

    def _show_error(self, message: str) -> None:
        self._write_results(message, "error")
        self._set_result_badge("Review required", "#FEF0EF", ERROR)

    def close(self) -> None:
        self._closed = True
        if self.camera_worker is not None:
            self.camera_worker.stop()
        self.root.destroy()


def launch_dashboard(
    database_path: str | Path,
    camera_index: int = 0,
    ocr_model: str | None = None,
) -> None:
    """Open the desktop scanner without changing the existing CLI workflows."""
    root = tk.Tk()
    PharmacyDashboard(root, database_path, camera_index, ocr_model)
    root.mainloop()

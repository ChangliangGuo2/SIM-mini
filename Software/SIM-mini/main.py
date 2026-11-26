import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QRect, QPoint, QObject, QThread, Signal, Slot
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QRubberBand,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.camera import (
    CAMERA_STREAM_FORMAT_MJPEG,
    CAMERA_STREAM_FORMAT_YUYV422,
    FFmpegCameraThread,
    Gray16TiffAcquisitionThread,
    apply_fixed_camera_controls,
    camera_driver_name,
    camera_stream_format_label,
    default_camera_device_name,
    enumerate_video_devices,
    is_linux,
    normalize_camera_stream_format,
    opencv_cuda_available,
    opencv_cuda_device_count,
    query_video_device_modes,
    resolve_ffmpeg_exe,
    select_default_video_device,
)
from core.config import load_config, save_config_to_file
from core.constants import (
    CAMERA_RESOLUTIONS,
    CONFIG_PATH,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_H,
    DEFAULT_CAMERA_W,
    DEFAULT_DISPLAY_AUTO_CONTRAST,
    DEFAULT_DISPLAY_BLACK_LEVEL,
    DEFAULT_DISPLAY_WHITE_LEVEL,
    DEFAULT_FFMPEG_EXE,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_PREVIEW_MAX_W,
    DEFAULT_PREVIEW_MAX_H,
    DEFAULT_ACQ_OUTPUT_DIR,
    DEFAULT_ACQ_FILENAME_PREFIX,
    DEFAULT_ACQ_MODE,
    DEFAULT_ACQ_DURATION_S,
    DEFAULT_ACQ_FRAME_COUNT,
    DEFAULT_ACQ_INTERVAL_MS,
    DEFAULT_ACQ_USE_GPU,
    PROJECT_H,
    PROJECT_W,
)
from core.image_io import (
    auto_contrast_levels,
    compute_histogram_from_bgr,
)
from core.projection import ProjectionWindow
from core.save_worker import SaveWorker
from core.widgets import HistogramContrastWidget

from modes.widefield_mode import WidefieldMode
from modes.sim_mode import SIMMode, SparseSIMMode
from modes.vessel_mode import VesselMode
from modes.neuron_mode import NeuronMode


class ROISelectionLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber_band.setStyleSheet(
            "border: 2px solid yellow; background: rgba(255, 255, 0, 50);"
        )
        self._origin = QPoint()
        self.selection_callback = None
        self._pick_enabled = False

    def set_pick_enabled(self, enabled: bool):
        self._pick_enabled = bool(enabled)
        if not self._pick_enabled:
            self._rubber_band.hide()

    def mousePressEvent(self, event):
        if (
            event.button() != Qt.LeftButton
            or self.pixmap() is None
            or not self._pick_enabled
        ):
            return super().mousePressEvent(event)

        self._origin = event.pos()
        self._rubber_band.setGeometry(QRect(self._origin, QSize()))
        self._rubber_band.show()

    def _four_three_selection_rect(self, current: QPoint) -> QRect:
        bounds = self.contentsRect()
        current = QPoint(
            max(bounds.left(), min(bounds.right(), current.x())),
            max(bounds.top(), min(bounds.bottom(), current.y())),
        )
        dx = current.x() - self._origin.x()
        dy = current.y() - self._origin.y()
        scale = min(abs(dx) // 4, abs(dy) // 3)
        if scale <= 0:
            return QRect(self._origin, QSize())

        width = 4 * scale
        height = 3 * scale
        x = self._origin.x() if dx >= 0 else self._origin.x() - width
        y = self._origin.y() if dy >= 0 else self._origin.y() - height
        return QRect(x, y, width, height)

    def mouseMoveEvent(self, event):
        if self._rubber_band.isVisible():
            self._rubber_band.setGeometry(
                self._four_three_selection_rect(event.pos())
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._rubber_band.isVisible():
            return super().mouseReleaseEvent(event)

        rect = self._four_three_selection_rect(event.pos())
        self._rubber_band.hide()
        if rect.width() > 5 and rect.height() > 5 and callable(self.selection_callback):
            self.selection_callback(rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._rubber_band.isVisible():
            self._rubber_band.hide()


class HistogramWorker(QObject):
    histogram_ready = Signal(object)
    auto_contrast_ready = Signal(int, int)
    calculate_histogram = Signal(object)
    calculate_auto_contrast = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculate_histogram.connect(self.on_calculate_histogram)
        self.calculate_auto_contrast.connect(self.on_calculate_auto_contrast)

    @Slot(object)
    def on_calculate_histogram(self, frame_bgr):
        hist = compute_histogram_from_bgr(frame_bgr)
        self.histogram_ready.emit(hist)

    @Slot(object)
    def on_calculate_auto_contrast(self, frame_bgr):
        black, white = auto_contrast_levels(frame_bgr)
        self.auto_contrast_ready.emit(int(black), int(white))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.config = load_config()

        self.setObjectName("MainWindowClass")
        self.setWindowTitle("miniSIM-Scope Pattern Projection Controller")
        self.setMinimumSize(900, 600)

        self.camera_thread = None
        self.latest_frame = None
        self.latest_frame_seq = 0
        self._last_rendered_seq = 0
        self._last_histogram_update = 0.0
        self._current_display_pixmap = None
        self._last_lut_black = None
        self._last_lut_white = None
        self._contrast_lut = None
        self.roi_has_selection = False
        self.roi_pick_active = False
        self._histogram_busy = False
        # render profiler storage (per-frame step timings)
        self._render_timing_samples = deque(maxlen=120)
        self._camera_preview_timing = {}
        self._last_render_probe_log = 0.0
        self._last_timing_label_update = 0.0

        self._preview_render_timer = QTimer(self)
        self._preview_render_timer.setTimerType(Qt.PreciseTimer)
        self._preview_render_timer.setInterval(33)
        self._preview_render_timer.timeout.connect(self._render_pending_preview)

        self._histogram_thread = QThread(self)
        self._histogram_worker = HistogramWorker()
        self._histogram_worker.moveToThread(self._histogram_thread)
        self._histogram_worker.histogram_ready.connect(self._on_histogram_ready)
        self._histogram_worker.auto_contrast_ready.connect(self._on_auto_contrast_ready)
        self._histogram_thread.start()

        # Must be initialized before _build_ui(), because Image Acquisition is
        # placed above Camera Control and also needs the GPU status.
        self.gpu_available = opencv_cuda_available()
        self.gpu_device_count = opencv_cuda_device_count()

        self.projection = ProjectionWindow()
        self.target_screen = None
        self.save_workers = []
        self.acquisition_thread = None
        self.camera_control_report = None
        self.supported_camera_modes = []
        self.roi_enabled = False
        self.roi_rect = None

        self._build_ui()

        self.recording_blink_state = False
        self.recording_blink_timer = QTimer(self)
        self.recording_blink_timer.timeout.connect(self.toggle_recording_indicator)

        self.apply_loaded_config_to_widgets()
        self.refresh_camera_devices()
        self.refresh_screens()
        self._resize_to_available_screen()

        QTimer.singleShot(700, self.auto_open_default_camera)

    def _resize_to_available_screen(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1300, 800)
            return

        geo = screen.availableGeometry()
        w = min(1500, max(1000, geo.width() - 120))
        h = min(900, max(650, geo.height() - 120))

        self.resize(w, h)
        self.move(
            geo.x() + max(0, (geo.width() - w) // 2),
            geo.y() + max(0, (geo.height() - h) // 2),
        )

    def _build_colormap_preview_pixmap(self, cmap_name: str, width: int = 140, height: int = 16) -> QPixmap:
        gray = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gray, (height, 1))
        cmap_name = cmap_name.lower()

        cv_map = {
            "gray": None,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "cividis": cv2.COLORMAP_CIVIDIS,
            "turbo": cv2.COLORMAP_TURBO,
            "hot": cv2.COLORMAP_HOT,
            "jet": cv2.COLORMAP_JET,
        }

        if cmap_name not in cv_map:
            cmap_name = "gray"

        if cv_map[cmap_name] is not None:
            color = cv2.applyColorMap(gradient, cv_map[cmap_name])
        else:
            color = np.stack([gradient, gradient, gradient], axis=2)

        qimg = QImage(color.data, width, height, color.strides[0], QImage.Format_BGR888)
        return QPixmap.fromImage(qimg.copy())

    def _populate_colormap_combo(self):
        self.colormap_combo.clear()
        self.colormap_combo.setIconSize(QSize(140, 16))
        cmap_items = [
            ("Gray", "gray"),
            ("Viridis", "viridis"),
            ("Magma", "magma"),
            ("Plasma", "plasma"),
            ("Inferno", "inferno"),
            ("Cividis", "cividis"),
            ("Turbo", "turbo"),
            ("Hot", "hot"),
            ("Jet", "jet"),
        ]

        for label, data in cmap_items:
            pixmap = self._build_colormap_preview_pixmap(data)
            self.colormap_combo.addItem(QIcon(pixmap), label, data)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(410)
        left_scroll.setMaximumWidth(540)
        left_scroll.setWidget(left_content)

        left_layout.addWidget(self._build_histogram_group())
        left_layout.addWidget(self._build_acquisition_group())
        left_layout.addWidget(self._build_camera_group())
        left_layout.addWidget(self._build_projection_group())

        self.tabs = QTabWidget()

        self.widefield_mode = WidefieldMode(self)
        self.sim_mode = SIMMode(self)
        self.sparse_sim_mode = SparseSIMMode(self)
        self.vessel_mode = VesselMode(self)
        self.neuron_mode = NeuronMode(self)

        self.tabs.addTab(self.widefield_mode, "Widefield Mode")
        self.tabs.addTab(self.sim_mode, "SIM Mode")
        self.tabs.addTab(self.sparse_sim_mode, "Sparse SIM Mode")
        self.tabs.addTab(self.vessel_mode, "Vessel Illumination")
        self.tabs.addTab(self.neuron_mode, "Neuron Illumination")

        left_layout.addWidget(self.tabs, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        self.image_label = ROISelectionLabel("Waiting for camera preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet(
            "background: #202020; color: #cfcfcf; border: 1px solid #555;"
        )
        self.image_label.selection_callback = self.on_preview_roi_selected
        right_layout.addWidget(self.image_label, stretch=1)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1000])

        status = QStatusBar()
        self.setStatusBar(status)

        self.status_msg_label = QLabel("Ready")
        self.camera_status_label = QLabel("Camera: --")
        self.resolution_label = QLabel("Resolution: -- x --")
        self.fps_label = QLabel("Stream FPS: --")
        self.latency_label = QLabel("Latency: -- ms")
        self.pipeline_timing_label = QLabel("Pipeline: --")
        self.pipeline_timing_label.setToolTip(
            "Camera and UI timing: decode, resize, contrast, color map, QImage, QPixmap and paint/update."
        )

        self.acq_progress_bar = QProgressBar()
        self.acq_progress_bar.setRange(0, 100)
        self.acq_progress_bar.setValue(0)
        self.acq_progress_bar.setFixedWidth(260)
        self.acq_progress_bar.setFixedHeight(20)
        self.acq_progress_bar.setFormat("Idle")

        self.recording_indicator_label = QLabel("● REC")
        self.recording_indicator_label.setFixedWidth(58)
        self.recording_indicator_label.setAlignment(Qt.AlignCenter)
        self.recording_indicator_label.setStyleSheet(
            "color: transparent; font-weight: bold;"
        )

        status.addWidget(self.status_msg_label, stretch=1)
        status.addPermanentWidget(self.acq_progress_bar)
        status.addPermanentWidget(self.recording_indicator_label)
        status.addPermanentWidget(self.camera_status_label)
        status.addPermanentWidget(self.resolution_label)
        status.addPermanentWidget(self.fps_label)
        status.addPermanentWidget(self.latency_label)
        status.addPermanentWidget(self.pipeline_timing_label)

    def _build_histogram_group(self):
        group = QGroupBox("Histogram / Display Contrast")
        layout = QVBoxLayout(group)

        self.histogram_widget = HistogramContrastWidget()
        self.histogram_widget.levelsChanged.connect(self.on_histogram_levels_changed)
        layout.addWidget(self.histogram_widget)

        self.auto_contrast_checkbox = QCheckBox("Auto display contrast")
        self.auto_contrast_checkbox.setChecked(DEFAULT_DISPLAY_AUTO_CONTRAST)
        self.auto_contrast_checkbox.toggled.connect(self.on_auto_contrast_toggled)
        layout.addWidget(self.auto_contrast_checkbox)

        # Optional render profiler to collect per-step timings for diagnosis.
        # Enabled by default while diagnosing preview latency; it only updates
        # the text label a few times per second, so overhead is tiny.
        self.render_profiler_checkbox = QCheckBox("Show decode/display timing")
        self.render_profiler_checkbox.setChecked(True)
        layout.addWidget(self.render_profiler_checkbox)

        self.render_timing_label = QLabel("Timing: --")
        self.render_timing_label.setWordWrap(True)
        self.render_timing_label.setMinimumHeight(42)
        layout.addWidget(self.render_timing_label)

        self.black_level_spin = QSpinBox(group)
        self.black_level_spin.setRange(0, 254)
        self.black_level_spin.setValue(DEFAULT_DISPLAY_BLACK_LEVEL)
        self.black_level_spin.valueChanged.connect(self.on_level_spin_changed)
        self.black_level_spin.hide()

        self.white_level_spin = QSpinBox(group)
        self.white_level_spin.setRange(1, 255)
        self.white_level_spin.setValue(DEFAULT_DISPLAY_WHITE_LEVEL)
        self.white_level_spin.valueChanged.connect(self.on_level_spin_changed)
        self.white_level_spin.hide()

        self.enable_colormap_checkbox = QCheckBox("Enable colormap preview")
        self.enable_colormap_checkbox.setChecked(False)
        self.enable_colormap_checkbox.toggled.connect(self.on_colormap_toggled)

        self.colormap_combo = QComboBox()
        self.colormap_combo.setEnabled(False)
        self._populate_colormap_combo()
        self.colormap_combo.currentIndexChanged.connect(self.on_colormap_selection_changed)

        colormap_row = QHBoxLayout()
        colormap_row.addWidget(self.enable_colormap_checkbox)
        colormap_row.addWidget(self.colormap_combo)
        layout.addLayout(colormap_row)

        self.update_contrast_control_state()
        return group


    def _build_acquisition_group(self):
        group = QGroupBox("Image Acquisition")
        layout = QVBoxLayout(group)

        form = QFormLayout()

        self.acq_output_dir_edit = QLineEdit()
        self.acq_output_dir_edit.setPlaceholderText("Default: ./acquisitions/timestamp")

        self.acq_choose_dir_btn = QPushButton("Browse")
        self.acq_choose_dir_btn.clicked.connect(self.choose_acq_output_dir)

        out_row = QHBoxLayout()
        out_row.addWidget(self.acq_output_dir_edit, stretch=1)
        out_row.addWidget(self.acq_choose_dir_btn)

        self.acq_filename_prefix_edit = QLineEdit()
        self.acq_filename_prefix_edit.setPlaceholderText("Default: None")
        self.acq_filename_prefix_edit.setToolTip(
            "Optional prefix for the TIFF stack filename. Leave empty or type None to keep acquisition_gray16_stack.tif."
        )

        self.acq_mode_combo = QComboBox()
        self.acq_mode_combo.addItem("Time Mode", "Time")
        self.acq_mode_combo.addItem("Sequence Mode", "Sequence")
        self.acq_mode_combo.addItem("Manual Start/Stop", "Manual")
        self.acq_mode_combo.currentIndexChanged.connect(self.update_acq_mode_controls)

        self.acq_duration_spin = QSpinBox()
        self.acq_duration_spin.setRange(1, 24 * 3600)
        self.acq_duration_spin.setValue(DEFAULT_ACQ_DURATION_S)
        self.acq_duration_spin.setSuffix(" s")

        self.acq_frame_count_spin = QSpinBox()
        self.acq_frame_count_spin.setRange(1, 1000000)
        self.acq_frame_count_spin.setValue(DEFAULT_ACQ_FRAME_COUNT)
        self.acq_frame_count_spin.setSuffix(" frames")

        self.acq_interval_spin = QSpinBox()
        self.acq_interval_spin.setRange(0, 600000)
        self.acq_interval_spin.setValue(DEFAULT_ACQ_INTERVAL_MS)
        self.acq_interval_spin.setSuffix(" ms")
        self.acq_interval_spin.setToolTip("0 ms means save every new camera frame.")

        self.acq_gpu_checkbox = QCheckBox("Use GPU for grayscale conversion")
        self.acq_gpu_checkbox.setChecked(self.gpu_available)
        self.acq_gpu_checkbox.setEnabled(self.gpu_available)
        self.acq_gpu_checkbox.setVisible(False)

        form.addRow("Save Folder", out_row)
        form.addRow("File Prefix", self.acq_filename_prefix_edit)
        form.addRow("Capture Mode", self.acq_mode_combo)
        form.addRow("Duration", self.acq_duration_spin)
        form.addRow("Frame Count", self.acq_frame_count_spin)
        form.addRow("Interval", self.acq_interval_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()

        self.acq_start_btn = QPushButton("Start Acquisition")
        self.acq_start_btn.clicked.connect(self.start_image_acquisition)

        self.acq_stop_btn = QPushButton("Stop Acquisition")
        self.acq_stop_btn.clicked.connect(self.stop_image_acquisition)
        self.acq_stop_btn.setEnabled(False)

        btn_row.addWidget(self.acq_start_btn)
        btn_row.addWidget(self.acq_stop_btn)
        layout.addLayout(btn_row)

        self.acq_status_label = QLabel("Output: multi-page 16-bit uncompressed grayscale TIFF stack")
        self.acq_status_label.setWordWrap(True)
        layout.addWidget(self.acq_status_label)

        self.update_acq_mode_controls()
        return group

    def _build_camera_group(self):
        group = QGroupBox("Camera Control")
        layout = QVBoxLayout(group)

        self.camera_control_tabs = QTabWidget()
        self.camera_control_tabs.addTab(self._build_camera_base_tab(), "Base")
        self.camera_control_tabs.addTab(self._build_camera_advance_tab(), "Advance")
        layout.addWidget(self.camera_control_tabs)

        row = QHBoxLayout()

        self.reload_devices_btn = QPushButton("Reload Devices")
        self.reload_devices_btn.clicked.connect(self.refresh_camera_devices)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_current_config)

        self.camera_start_btn = QPushButton("Open Camera")
        self.camera_start_btn.clicked.connect(self.toggle_camera)

        row.addWidget(self.reload_devices_btn)
        row.addWidget(self.save_config_btn)
        row.addWidget(self.camera_start_btn)

        layout.addLayout(row)
        return group

    def _build_camera_base_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.camera_device_combo = QComboBox()
        self.camera_device_combo.setEditable(False)

        configured_device = str(
            self.config.get("camera_device_name", default_camera_device_name())
        ).strip() or default_camera_device_name()
        self.camera_device_combo.addItem(configured_device)
        self.camera_device_combo.setCurrentText(configured_device)
        self.camera_device_combo.currentIndexChanged.connect(
            self.on_camera_device_changed
        )

        self.ffmpeg_exe_edit = QLineEdit(DEFAULT_FFMPEG_EXE)

        self.stream_format_combo = QComboBox()
        self.stream_format_combo.addItem("MJPEG", CAMERA_STREAM_FORMAT_MJPEG)
        self.stream_format_combo.addItem("YUYV422", CAMERA_STREAM_FORMAT_YUYV422)
        self.stream_format_combo.currentIndexChanged.connect(
            self.on_camera_stream_format_changed
        )

        self.resolution_combo = QComboBox()
        self._populate_resolution_combo()

        self.gpu_preview_checkbox = QCheckBox("Use GPU preview resize (OpenCV CUDA)")
        self.gpu_preview_checkbox.setChecked(self.gpu_available)
        self.gpu_preview_checkbox.setEnabled(self.gpu_available)
        self.gpu_preview_checkbox.setVisible(False)

        self.device_info_label = QLabel(f"Driver: {camera_driver_name()} / MJPEG + YUYV422")
        self.device_info_label.setWordWrap(True)

        form.addRow("V4L2 Device" if is_linux() else "DirectShow Device", self.camera_device_combo)
        form.addRow("FFmpeg EXE", self.ffmpeg_exe_edit)
        form.addRow("Video Format", self.stream_format_combo)
        form.addRow("Resolution / FPS", self.resolution_combo)
        form.addRow("Device Info", self.device_info_label)

        return tab

    def current_camera_stream_format(self) -> str:
        if not hasattr(self, "stream_format_combo"):
            return CAMERA_STREAM_FORMAT_MJPEG
        value = self.stream_format_combo.currentData()
        return normalize_camera_stream_format(value)

    def _apply_configured_fixed_camera_controls(self):
        enabled = bool(self.fixed_camera_controls_checkbox.isChecked())
        if not enabled:
            self.camera_control_report = {
                "enabled": False,
                "verified": False,
                "errors": ["Fixed manual exposure/gain is disabled"],
            }
            self.fixed_camera_controls_status.setText("Disabled; SIM acquisition will be blocked")
            return self.camera_control_report

        device_name = self.camera_device_combo.currentText().strip() or default_camera_device_name()
        report = apply_fixed_camera_controls(
            device_name=device_name,
            device_index=int(self.camera_device_index_spin.value()),
            exposure=float(self.manual_exposure_spin.value()),
            gain=float(self.manual_gain_spin.value()),
        )
        report["enabled"] = True
        report["applied_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.camera_control_report = report

        if report.get("verified"):
            readback = report.get("readback", {})
            warning_suffix = (
                " (manual-mode command accepted; mode readback unsupported)"
                if report.get("auto_exposure_verification")
                == "set_accepted_readback_unsupported"
                else ""
            )
            self.fixed_camera_controls_status.setText(
                "Verified: manual exposure="
                f"{readback.get('exposure')}, gain={readback.get('gain')}"
                f"{warning_suffix}"
            )
        else:
            errors = "; ".join(str(item) for item in report.get("errors", []))
            self.fixed_camera_controls_status.setText(f"Verification failed: {errors}")
        return report

    def ensure_fixed_camera_controls_for_sim(self):
        if not self.fixed_camera_controls_checkbox.isChecked():
            return False, "Fixed manual exposure/gain is disabled in Camera Control > Advance."
        report = self.camera_control_report
        if not isinstance(report, dict) or not report.get("verified"):
            errors = "; ".join(str(item) for item in (report or {}).get("errors", []))
            readback = (report or {}).get("readback", {})
            return (
                False,
                "Manual exposure/gain was not verified before the current camera stream opened. "
                "Close the camera, correct the values or device index, and open it again.\n\n"
                f"Details: {errors or 'no control report'}\nReadback: {readback}",
            )
        return True, ""

    def camera_acquisition_metadata(self):
        camera_mode = self.resolution_combo.currentData() or (0, 0, 0)
        if self.camera_thread is not None:
            width = int(self.camera_thread.width)
            height = int(self.camera_thread.height)
            fps = int(self.camera_thread.fps)
            roi = list(self.camera_thread.roi) if self.camera_thread.roi is not None else None
        else:
            width, height, fps = (int(value) for value in camera_mode)
            roi = list(self.roi_rect) if self.roi_enabled and self.roi_rect is not None else None

        return {
            "driver": camera_driver_name(),
            "device_name": self.camera_device_combo.currentText().strip()
            or default_camera_device_name(),
            "ffmpeg_executable": self.ffmpeg_exe_edit.text().strip()
            or DEFAULT_FFMPEG_EXE,
            "directshow_device_index": (
                int(self.camera_device_index_spin.value()) if not is_linux() else None
            ),
            "stream_format": self.current_camera_stream_format(),
            "width": width,
            "height": height,
            "fps": fps,
            "preview_fps_limit": int(DEFAULT_PREVIEW_FPS),
            "gpu_preview_enabled": bool(self.gpu_preview_checkbox.isChecked()),
            "roi_enabled": bool(roi is not None),
            "roi": roi,
            "fixed_exposure_gain": self.camera_control_report,
        }

    def _mode_stream_format(self, mode) -> str:
        if isinstance(mode, dict):
            return normalize_camera_stream_format(
                mode.get("stream_format", mode.get("format", CAMERA_STREAM_FORMAT_MJPEG))
            )
        return CAMERA_STREAM_FORMAT_MJPEG

    def _count_modes_by_format(self):
        counts = {
            CAMERA_STREAM_FORMAT_MJPEG: 0,
            CAMERA_STREAM_FORMAT_YUYV422: 0,
        }
        seen = set()
        for mode in self.supported_camera_modes:
            try:
                fmt = self._mode_stream_format(mode)
                width = int(mode.get("width", 0))
                height = int(mode.get("height", 0))
                fps = int(round(float(mode.get("fps", 0))))
            except Exception:
                continue
            key = (fmt, width, height, fps)
            if width <= 0 or height <= 0 or fps <= 0 or key in seen:
                continue
            seen.add(key)
            counts[fmt] = counts.get(fmt, 0) + 1
        return counts

    def _update_device_info_label(self, device_name: str = ""):
        if not hasattr(self, "device_info_label"):
            return
        counts = self._count_modes_by_format()
        selected_format = self.current_camera_stream_format()
        selected_count = counts.get(selected_format, 0)
        selected_label = camera_stream_format_label(selected_format)

        if counts.get(CAMERA_STREAM_FORMAT_MJPEG, 0) or counts.get(CAMERA_STREAM_FORMAT_YUYV422, 0):
            self.device_info_label.setText(
                f"Driver: {camera_driver_name()}; "
                f"MJPEG {counts.get(CAMERA_STREAM_FORMAT_MJPEG, 0)} modes, "
                f"YUYV422 {counts.get(CAMERA_STREAM_FORMAT_YUYV422, 0)} modes; "
                f"selected {selected_label}: {selected_count} modes"
            )
        else:
            suffix = f" for {device_name}" if device_name else ""
            self.device_info_label.setText(
                f"Driver: {camera_driver_name()} / MJPEG + YUYV422; "
                f"mode query failed{suffix}, using fallback list for {selected_label}"
            )

    def _populate_resolution_combo(self, preferred_mode=None):
        selected_format = self.current_camera_stream_format()
        current_selection = preferred_mode or self.resolution_combo.currentData()

        if isinstance(current_selection, (tuple, list)) and len(current_selection) >= 3:
            current_selection = (
                int(current_selection[0]),
                int(current_selection[1]),
                int(current_selection[2]),
            )
        else:
            current_selection = None

        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()

        seen = set()
        modes = []
        for mode in self.supported_camera_modes:
            try:
                mode_format = self._mode_stream_format(mode)
                width = int(mode.get("width", 0))
                height = int(mode.get("height", 0))
                fps = int(round(float(mode.get("fps", 0))))
            except (TypeError, ValueError):
                continue

            if mode_format != selected_format:
                continue
            if width <= 0 or height <= 0 or fps <= 0:
                continue

            key = (width, height, fps)
            if key in seen:
                continue
            seen.add(key)
            modes.append(key)

        modes.sort(
            key=lambda item: (
                item[0] * item[1],
                item[0],
                item[1],
                item[2],
            ),
            reverse=True,
        )

        if not modes:
            for _label, width, height in CAMERA_RESOLUTIONS:
                key = (int(width), int(height), int(DEFAULT_CAMERA_FPS))
                if key not in seen:
                    seen.add(key)
                    modes.append(key)

        for width, height, fps in modes:
            label = f"{width} x {height} @ {fps} FPS"
            self.resolution_combo.addItem(label, (width, height, fps))

        if self.resolution_combo.count() > 0:
            selected_index = 0
            if current_selection is not None:
                for index in range(self.resolution_combo.count()):
                    if self.resolution_combo.itemData(index) == current_selection:
                        selected_index = index
                        break
            self.resolution_combo.setCurrentIndex(selected_index)

        self.resolution_combo.blockSignals(False)
        if hasattr(self, "device_info_label"):
            self._update_device_info_label()

    def _refresh_modes_for_device(self, device_name: str, preferred_mode=None):
        device_name = str(device_name or "").strip()
        if not device_name:
            self.supported_camera_modes = []
            self._populate_resolution_combo(preferred_mode)
            return

        self.set_status(f"Reading supported MJPEG/YUYV422 modes for {device_name}...")
        mode_list = query_video_device_modes(
            self.ffmpeg_exe_edit.text().strip() or DEFAULT_FFMPEG_EXE,
            device_name,
        )

        self.supported_camera_modes = mode_list or []
        self._populate_resolution_combo(preferred_mode)
        self._update_device_info_label(device_name)

        counts = self._count_modes_by_format()
        if mode_list:
            self.set_status(
                f"Loaded {counts.get(CAMERA_STREAM_FORMAT_MJPEG, 0)} MJPEG modes and "
                f"{counts.get(CAMERA_STREAM_FORMAT_YUYV422, 0)} YUYV422 modes for {device_name}."
            )
        else:
            self.set_status(
                f"No MJPEG/YUYV422 mode list was returned for {device_name}; using fallback modes."
            )

    def on_camera_device_changed(self, index: int):
        if index < 0:
            return

        device_name = self.camera_device_combo.itemText(index).strip()
        if not device_name:
            return

        self._refresh_modes_for_device(device_name)

    def on_camera_stream_format_changed(self, index: int):
        if index < 0:
            return
        self._populate_resolution_combo()
        selected_label = camera_stream_format_label(self.current_camera_stream_format())
        self.set_status(f"Camera format switched to {selected_label}; resolution/FPS list updated.")

    def _build_camera_advance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hint = QLabel(
            (
                "Exposure and gain below are applied and verified with v4l2-ctl "
                "before the FFmpeg stream opens. Use v4l2-ctl directly for other "
                "controls such as focus and anti-flicker."
            )
            if is_linux()
            else (
                "Exposure and gain below are applied and verified through OpenCV "
                "DirectShow before the FFmpeg stream opens. Use the native property "
                "panel for other controls such as white balance, focus and anti-flicker."
            )
        )
        hint.setWordWrap(True)

        self.native_driver_panel_btn = QPushButton(
            "Show Linux UVC Control Hint" if is_linux() else "Open Native DirectShow Property Panel"
        )
        self.native_driver_panel_btn.clicked.connect(self.open_native_camera_property_panel)

        layout.addWidget(hint)
        layout.addWidget(self.native_driver_panel_btn)

        manual_group = QGroupBox("Fixed Exposure / Gain for SIM")
        manual_layout = QFormLayout(manual_group)

        self.fixed_camera_controls_checkbox = QCheckBox(
            "Require verified manual exposure and gain for SIM acquisition"
        )
        self.fixed_camera_controls_checkbox.setChecked(True)

        self.manual_exposure_spin = QDoubleSpinBox()
        self.manual_exposure_spin.setRange(-20.0, 1000000.0)
        self.manual_exposure_spin.setDecimals(2)
        self.manual_exposure_spin.setValue(100.0 if is_linux() else -1.0)
        self.manual_exposure_spin.setToolTip(
            "Windows DirectShow normally uses log2(seconds), for example -1 is 0.5 s. "
            "Linux uses the integer unit reported by v4l2-ctl."
        )

        self.manual_gain_spin = QDoubleSpinBox()
        self.manual_gain_spin.setRange(0.0, 1000000.0)
        self.manual_gain_spin.setDecimals(2)
        self.manual_gain_spin.setValue(0.0)

        self.camera_device_index_spin = QSpinBox()
        self.camera_device_index_spin.setRange(0, 99)
        self.camera_device_index_spin.setValue(0)
        self.camera_device_index_spin.setToolTip(
            "OpenCV DirectShow device index used to set controls before FFmpeg opens the camera."
        )
        self.camera_device_index_spin.setVisible(not is_linux())

        self.fixed_camera_controls_status = QLabel("Not applied")
        self.fixed_camera_controls_status.setWordWrap(True)

        manual_layout.addRow(self.fixed_camera_controls_checkbox)
        manual_layout.addRow("Manual Exposure", self.manual_exposure_spin)
        manual_layout.addRow("Manual Gain", self.manual_gain_spin)
        if not is_linux():
            manual_layout.addRow("DirectShow Device Index", self.camera_device_index_spin)
        manual_layout.addRow("Verification", self.fixed_camera_controls_status)
        layout.addWidget(manual_group)

        roi_group = QGroupBox("ROI / Fullframe")
        roi_layout = QFormLayout(roi_group)

        self.roi_checkbox = QCheckBox("Enable ROI crop")
        self.roi_checkbox.toggled.connect(self.on_roi_toggled)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 9999)
        self.roi_x_spin.setValue(0)
        self.roi_x_spin.valueChanged.connect(
            lambda value: self.on_roi_parameters_changed("x", value)
        )

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 9999)
        self.roi_y_spin.setValue(0)
        self.roi_y_spin.valueChanged.connect(
            lambda value: self.on_roi_parameters_changed("y", value)
        )

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 9999)
        self.roi_w_spin.setValue(640)
        self.roi_w_spin.setSingleStep(4)
        self.roi_w_spin.valueChanged.connect(
            lambda value: self.on_roi_parameters_changed("width", value)
        )

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 9999)
        self.roi_h_spin.setValue(480)
        self.roi_h_spin.setSingleStep(3)
        self.roi_h_spin.valueChanged.connect(
            lambda value: self.on_roi_parameters_changed("height", value)
        )

        self.roi_pick_button = QPushButton("Start ROI Pick")
        self.roi_pick_button.setEnabled(False)
        self.roi_pick_button.clicked.connect(self.on_roi_pick_button_clicked)

        roi_layout.addRow(self.roi_checkbox)
        roi_layout.addRow("X", self.roi_x_spin)
        roi_layout.addRow("Y", self.roi_y_spin)
        roi_layout.addRow("Width", self.roi_w_spin)
        roi_layout.addRow("Height", self.roi_h_spin)
        roi_layout.addRow("Aspect Ratio", QLabel("4:3 (locked)"))
        roi_layout.addRow(self.roi_pick_button)

        layout.addWidget(roi_group)
        layout.addStretch(1)

        self._set_roi_controls_enabled(False)
        return tab

    @staticmethod
    def _roi_is_valid(rect) -> bool:
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            return False
        try:
            x, y, width, height = (int(value) for value in rect)
        except (TypeError, ValueError):
            return False
        return (
            x >= 0
            and y >= 0
            and width >= 4
            and height >= 3
            and width * 3 == height * 4
        )

    def _camera_frame_size(self):
        if self.camera_thread is not None:
            return int(self.camera_thread.width), int(self.camera_thread.height)
        camera_mode = self.resolution_combo.currentData() or (0, 0)
        if not isinstance(camera_mode, (tuple, list)) or len(camera_mode) < 2:
            return 0, 0
        return int(camera_mode[0]), int(camera_mode[1])

    def _normalize_roi_4_3(self, x, y, width, height, changed_field="width"):
        x = max(0, int(x))
        y = max(0, int(y))
        width = max(1, int(width))
        height = max(1, int(height))

        camera_width, camera_height = self._camera_frame_size()
        if camera_width >= 4 and camera_height >= 3:
            x = min(x, camera_width - 4)
            y = min(y, camera_height - 3)

        if changed_field == "height":
            scale = max(1, int(round(height / 3.0)))
        elif changed_field == "fit":
            scale = max(1, min(width // 4, height // 3))
        elif width * 3 == height * 4:
            scale = max(1, width // 4)
        else:
            scale = max(1, int(round(width / 4.0)))

        max_scale = min(9999 // 4, 9999 // 3)
        if camera_width >= 4 and camera_height >= 3:
            max_scale = min(
                max_scale,
                (camera_width - x) // 4,
                (camera_height - y) // 3,
            )
        scale = max(1, min(scale, max_scale))
        return x, y, 4 * scale, 3 * scale

    def _set_roi_spin_values(self, rect):
        for spin, value in zip(
            (
                self.roi_x_spin,
                self.roi_y_spin,
                self.roi_w_spin,
                self.roi_h_spin,
            ),
            rect,
        ):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)

    def on_roi_toggled(self, checked: bool):
        self.roi_enabled = bool(checked)
        self._set_roi_controls_enabled(self.roi_enabled)
        self.roi_pick_button.setEnabled(self.roi_enabled)
        if self.roi_enabled and self._roi_is_valid(self.roi_rect):
            self.roi_rect = self._normalize_roi_4_3(
                *self.roi_rect,
                changed_field="width",
            )
            self.roi_has_selection = True
            self._set_roi_spin_values(self.roi_rect)
            self.set_roi_pick_active(False)
            x, y, width, height = self.roi_rect
            self.set_status(
                f"ROI restored: x={x}, y={y}, w={width}, h={height} (4:3)"
            )
        elif self.roi_enabled:
            self.roi_has_selection = False
            self.set_status("ROI enabled. Click Start ROI Pick or edit ROI values.")
        else:
            self.set_roi_pick_active(False)
            self.set_status("ROI disabled; saved ROI is retained.")
        self._apply_roi_settings()

    def on_roi_pick_button_clicked(self):
        self.set_roi_pick_active(not self.roi_pick_active)

    def set_roi_pick_active(self, active: bool):
        self.roi_pick_active = bool(active)
        self.image_label.set_pick_enabled(self.roi_pick_active)
        self.roi_pick_button.setText(
            "Stop ROI Pick" if self.roi_pick_active else "Start ROI Pick"
        )
        if self.roi_pick_active:
            self.set_status("ROI pick enabled: draw a yellow 4:3 box on preview.")
        else:
            self.set_status("ROI pick disabled.")

    def on_roi_parameters_changed(self, changed_field=None, _value=None):
        rect = self._normalize_roi_4_3(
            self.roi_x_spin.value(),
            self.roi_y_spin.value(),
            self.roi_w_spin.value(),
            self.roi_h_spin.value(),
            changed_field=changed_field or "width",
        )
        self._set_roi_spin_values(rect)
        self.roi_rect = rect
        self.roi_has_selection = True
        self._apply_roi_settings()

    def on_preview_roi_selected(self, rect: QRect):
        if not self.roi_enabled or not self.roi_pick_active:
            return

        label_rect = self.image_label.contentsRect()
        display_rect = label_rect

        # ROISelectionLabel uses QLabel.setScaledContents(True), so the preview is
        # stretched to the label contents rect.  The old code assumed the raw
        # pixmap was centered at its original size, which made ROI Pick coordinates
        # wrong after the preview widget was resized.  Use the actual displayed
        # label area for proportional mapping.
        selection = rect.intersected(display_rect)
        if selection.isEmpty():
            return

        proportional_x = (selection.x() - display_rect.x()) / max(1, display_rect.width())
        proportional_y = (selection.y() - display_rect.y()) / max(1, display_rect.height())
        proportional_w = selection.width() / max(1, display_rect.width())
        proportional_h = selection.height() / max(1, display_rect.height())

        if self.camera_thread is not None:
            cam_w, cam_h = int(self.camera_thread.width), int(self.camera_thread.height)
        else:
            cam_size = self.resolution_combo.currentData() or (0, 0)
            cam_w, cam_h = int(cam_size[0]), int(cam_size[1])

        if cam_w <= 0 or cam_h <= 0:
            return

        x = int(round(proportional_x * cam_w))
        y = int(round(proportional_y * cam_h))
        w = int(round(proportional_w * cam_w))
        h = int(round(proportional_h * cam_h))
        x, y, w, h = self._normalize_roi_4_3(
            x,
            y,
            w,
            h,
            changed_field="fit",
        )
        self._set_roi_spin_values((x, y, w, h))

        self.roi_rect = (x, y, w, h)
        self.roi_has_selection = True
        self._apply_roi_settings()
        self.set_roi_pick_active(False)
        self.set_status(f"ROI crop applied: x={x}, y={y}, w={w}, h={h}")

    def _set_roi_controls_enabled(self, enabled: bool):
        self.roi_x_spin.setEnabled(enabled)
        self.roi_y_spin.setEnabled(enabled)
        self.roi_w_spin.setEnabled(enabled)
        self.roi_h_spin.setEnabled(enabled)

    def _apply_roi_settings(self):
        if (
            self.roi_enabled
            and self.roi_has_selection
            and self._roi_is_valid(self.roi_rect)
        ):
            x, y, w, h = self.roi_rect
            roi = (x, y, w, h)
        else:
            roi = None

        if self.camera_thread is not None:
            self.camera_thread.set_roi(roi)

        # Force preview refresh if we already have a frame
        if self.latest_frame is not None:
            self._last_rendered_seq = 0
            self.on_camera_preview(
                self.latest_frame,
                self.latest_frame_seq,
                getattr(self, "_camera_preview_timestamp", time.perf_counter()),
                getattr(self, "_camera_preview_timing", {}),
            )

    def _build_projection_group(self):
        group = QGroupBox("Projection Screen")
        layout = QVBoxLayout(group)

        self.screen_test_mode_checkbox = QCheckBox(
            "Screen test mode: allow secondary screen larger than 640x480"
        )
        self.screen_test_mode_checkbox.stateChanged.connect(self.refresh_screens)

        self.screen_info_label = QLabel("Not detected")
        self.screen_info_label.setWordWrap(True)

        self.refresh_screen_btn = QPushButton("Refresh Screens")
        self.refresh_screen_btn.clicked.connect(self.refresh_screens)

        layout.addWidget(self.screen_test_mode_checkbox)
        layout.addWidget(self.screen_info_label)
        layout.addWidget(self.refresh_screen_btn)

        return group

    # ---------------- Shared helpers for mode modules ----------------

    def set_status(self, text: str):
        self.status_msg_label.setText(str(text))

    def is_camera_running(self) -> bool:
        return self.camera_thread is not None and self.camera_thread.isRunning()

    def is_global_test_mode(self) -> bool:
        """Backward-compatible alias for vessel-local test mode."""
        return hasattr(self, "vessel_mode") and self.vessel_mode.is_test_mode_enabled()

    def has_latest_camera_frame(self) -> bool:
        return self.is_camera_running() and self.get_latest_sequence() > 0

    def get_latest_sequence(self) -> int:
        if self.camera_thread is None:
            return 0
        return self.camera_thread.latest_sequence()

    def get_latest_jpeg(self):
        if self.camera_thread is None:
            return None, 0
        return self.camera_thread.get_latest_jpeg()

    def get_latest_full_bgr(self):
        """Return the latest full decoded camera frame and its sequence."""
        if self.camera_thread is None:
            return None, 0
        return self.camera_thread.get_latest_full_bgr()

    def get_latest_frame_bgr(self, full: bool = True):
        if full and self.camera_thread is not None:
            frame, _seq = self.camera_thread.get_latest_full_bgr()
            if frame is not None:
                return frame
        return None if self.latest_frame is None else self.latest_frame.copy()

    def resolve_output_base_dir(self) -> Path:
        text = self.acq_output_dir_edit.text().strip()
        if text:
            return Path(text)
        return Path.cwd() / "captures"

    def start_save_worker(self, tasks, label="Save", on_finished=None):
        worker = SaveWorker(tasks, label=label, parent=self)
        self.save_workers.append(worker)

        def cleanup(_label=None):
            try:
                self.save_workers.remove(worker)
            except ValueError:
                pass

            if callable(on_finished):
                on_finished()

        worker.progress.connect(lambda pct, msg: self.set_status(msg))
        worker.finished.connect(cleanup)
        worker.error.connect(lambda msg: QMessageBox.critical(self, "Save Error", msg))
        worker.error.connect(lambda _msg: cleanup(label))
        worker.start()
        return worker



    # ---------------- Status-bar recording indicator ----------------

    def start_recording_indicator(self, manual_mode: bool = False):
        self.recording_blink_state = False
        self.recording_indicator_label.setText("● REC")
        self.recording_indicator_label.setStyleSheet("color: red; font-weight: bold;")

        if manual_mode:
            self.acq_progress_bar.setRange(0, 0)
            self.acq_progress_bar.setFormat("Recording")
        else:
            self.acq_progress_bar.setRange(0, 100)
            self.acq_progress_bar.setValue(0)
            self.acq_progress_bar.setFormat("%p%")

        self.recording_blink_timer.start(500)

    def stop_recording_indicator(self):
        self.recording_blink_timer.stop()
        self.recording_indicator_label.setText("● REC")
        self.recording_indicator_label.setStyleSheet(
            "color: transparent; font-weight: bold;"
        )
        self.acq_progress_bar.setRange(0, 100)
        self.acq_progress_bar.setValue(0)
        self.acq_progress_bar.setFormat("Idle")

    def toggle_recording_indicator(self):
        self.recording_blink_state = not self.recording_blink_state
        color = "red" if self.recording_blink_state else "transparent"
        self.recording_indicator_label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )

    # ---------------- Image acquisition ----------------

    def choose_acq_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Acquisition Save Folder")
        if directory:
            self.acq_output_dir_edit.setText(directory)

    def update_acq_mode_controls(self):
        mode = self.acq_mode_combo.currentData()
        self.acq_duration_spin.setEnabled(mode == "Time")
        self.acq_frame_count_spin.setEnabled(mode == "Sequence")
        self.acq_interval_spin.setEnabled(True)

    def resolve_acq_output_dir(self) -> Path:
        text = self.acq_output_dir_edit.text().strip()
        if text:
            base = Path(text)
        else:
            base = Path.cwd() / "acquisitions"

        return base / time.strftime("gray16_%Y%m%d_%H%M%S")

    def start_image_acquisition(self):
        if self.acquisition_thread is not None and self.acquisition_thread.isRunning():
            QMessageBox.information(self, "Acquisition", "Acquisition is already running.")
            return

        if not self.is_camera_running():
            QMessageBox.warning(self, "Acquisition Error", "Please open the camera first.")
            return

        if not self.has_latest_camera_frame():
            QMessageBox.warning(self, "Acquisition Error", "No camera frame has been received yet.")
            return

        output_dir = self.resolve_acq_output_dir()
        mode = self.acq_mode_combo.currentData()

        self.acquisition_thread = Gray16TiffAcquisitionThread(
            camera_source=self.camera_thread,
            output_dir=output_dir,
            mode=mode,
            duration_s=float(self.acq_duration_spin.value()),
            frame_count=int(self.acq_frame_count_spin.value()),
            interval_ms=int(self.acq_interval_spin.value()),
            use_gpu=bool(self.acq_gpu_checkbox.isChecked()),
            filename_prefix=self.acq_filename_prefix_edit.text().strip(),
            parent=self,
        )
        self.acquisition_thread.progress.connect(self.on_acquisition_progress)
        self.acquisition_thread.finished.connect(self.on_acquisition_finished)
        self.acquisition_thread.error.connect(self.on_acquisition_error)
        self.acquisition_thread.start()

        self.acq_start_btn.setEnabled(False)
        self.acq_stop_btn.setEnabled(True)
        self.start_recording_indicator(manual_mode=(mode == "Manual"))
        self.acq_status_label.setText(f"Saving multi-page TIFF stack to: {output_dir}")
        self.set_status(f"Image acquisition started: {output_dir}")

    def stop_image_acquisition(self):
        if self.acquisition_thread is not None:
            self.acquisition_thread.stop()
            self.acq_stop_btn.setEnabled(False)
            self.set_status("Stopping image acquisition and closing multi-page TIFF stack...")

    def on_acquisition_progress(self, pct: int, message: str):
        self.acq_status_label.setText(message)
        self.set_status(message)

        if self.acq_progress_bar.maximum() != 0:
            self.acq_progress_bar.setValue(max(0, min(100, int(pct))))

    def on_acquisition_finished(self, output_dir: str):
        self.acq_start_btn.setEnabled(True)
        self.acq_stop_btn.setEnabled(False)
        self.stop_recording_indicator()
        self.acq_progress_bar.setValue(100)
        self.acq_progress_bar.setFormat("Done")
        self.acq_status_label.setText(f"Finished: {output_dir}")
        self.set_status(f"Image acquisition finished: {output_dir}")
        self.acquisition_thread = None

    def on_acquisition_error(self, message: str):
        self.acq_start_btn.setEnabled(True)
        self.acq_stop_btn.setEnabled(False)
        self.stop_recording_indicator()
        self.acq_status_label.setText(f"Error: {message}")
        self.acquisition_thread = None
        QMessageBox.critical(self, "Acquisition Error", message)

    # ---------------- Config ----------------

    def apply_loaded_config_to_widgets(self):
        cfg = self.config

        self.ffmpeg_exe_edit.setText(str(cfg.get("ffmpeg_exe", DEFAULT_FFMPEG_EXE)))
        configured_device = str(
            cfg.get("camera_device_name", default_camera_device_name())
        ).strip() or default_camera_device_name()
        self.camera_device_combo.setCurrentText(configured_device)

        target_format = normalize_camera_stream_format(
            cfg.get("camera_stream_format", CAMERA_STREAM_FORMAT_MJPEG)
        )
        fmt_index = self.stream_format_combo.findData(target_format)
        self.stream_format_combo.blockSignals(True)
        self.stream_format_combo.setCurrentIndex(fmt_index if fmt_index >= 0 else 0)
        self.stream_format_combo.blockSignals(False)

        target_w, target_h = cfg.get(
            "resolution", [DEFAULT_CAMERA_W, DEFAULT_CAMERA_H]
        )
        target_fps = int(cfg.get("fps", DEFAULT_CAMERA_FPS))
        self._pending_camera_mode = (
            int(target_w),
            int(target_h),
            target_fps,
        )
        self.gpu_preview_checkbox.setChecked(self.gpu_available)
        self.acq_gpu_checkbox.setChecked(self.gpu_available)

        self.acq_output_dir_edit.setText(str(cfg.get("acq_output_dir", DEFAULT_ACQ_OUTPUT_DIR)))
        self.acq_filename_prefix_edit.setText(str(cfg.get("acq_filename_prefix", DEFAULT_ACQ_FILENAME_PREFIX)))

        self.enable_colormap_checkbox.setChecked(bool(cfg.get("display_enable_colormap", False)))
        self.colormap_combo.setEnabled(self.enable_colormap_checkbox.isChecked())
        if self.enable_colormap_checkbox.isChecked():
            cmap_name = str(cfg.get("display_colormap", "gray"))
            idx = self.colormap_combo.findData(cmap_name)
            self.colormap_combo.setCurrentIndex(idx if idx >= 0 else 0)

        acq_mode = str(cfg.get("acq_mode", DEFAULT_ACQ_MODE))
        acq_index = self.acq_mode_combo.findData(acq_mode)
        self.acq_mode_combo.setCurrentIndex(acq_index if acq_index >= 0 else 1)

        self.acq_duration_spin.setValue(int(cfg.get("acq_duration_s", DEFAULT_ACQ_DURATION_S)))
        self.acq_frame_count_spin.setValue(int(cfg.get("acq_frame_count", DEFAULT_ACQ_FRAME_COUNT)))
        self.acq_interval_spin.setValue(int(cfg.get("acq_interval_ms", DEFAULT_ACQ_INTERVAL_MS)))
        self.acq_gpu_checkbox.setChecked(bool(cfg.get("acq_use_gpu", DEFAULT_ACQ_USE_GPU)) and self.gpu_available)
        self.update_acq_mode_controls()

        self.supported_camera_modes = cfg.get("supported_camera_modes", [])
        self._populate_resolution_combo(self._pending_camera_mode)

        self.fixed_camera_controls_checkbox.setChecked(
            bool(cfg.get("fixed_camera_controls_enabled", True))
        )
        self.manual_exposure_spin.setValue(
            float(cfg.get("manual_exposure", 100.0 if is_linux() else -1.0))
        )
        self.manual_gain_spin.setValue(float(cfg.get("manual_gain", 0.0)))
        self.camera_device_index_spin.setValue(
            int(cfg.get("directshow_device_index", 0))
        )

        self.roi_enabled = bool(cfg.get("roi_enabled", False))
        stored_roi = cfg.get("roi_rect")
        self.roi_rect = None
        if isinstance(stored_roi, (tuple, list)) and len(stored_roi) == 4:
            try:
                self.roi_rect = self._normalize_roi_4_3(
                    *stored_roi,
                    changed_field="width",
                )
            except (TypeError, ValueError):
                self.roi_rect = None
        stored_has_selection = bool(
            cfg.get("roi_has_selection", self.roi_rect is not None)
        )
        self.roi_has_selection = (
            stored_has_selection and self._roi_is_valid(self.roi_rect)
        )
        self.roi_checkbox.blockSignals(True)
        self.roi_checkbox.setChecked(self.roi_enabled)
        self.roi_checkbox.blockSignals(False)
        self._set_roi_spin_values(self.roi_rect or (0, 0, 640, 480))
        self.roi_pick_button.setEnabled(self.roi_enabled)
        self.set_roi_pick_active(False)
        self._set_roi_controls_enabled(self.roi_enabled)
        self._apply_roi_settings()

        self.screen_test_mode_checkbox.setChecked(bool(cfg.get("test_mode", False)))
        self.vessel_mode.set_test_mode_enabled(bool(cfg.get("vessel_test_mode", False)))
        self.neuron_mode.set_test_mode_enabled(bool(cfg.get("neuron_test_mode", False)))

        self.auto_contrast_checkbox.setChecked(
            bool(cfg.get("display_auto_contrast", DEFAULT_DISPLAY_AUTO_CONTRAST))
        )
        self.black_level_spin.setValue(
            int(cfg.get("display_black_level", DEFAULT_DISPLAY_BLACK_LEVEL))
        )
        self.white_level_spin.setValue(
            int(cfg.get("display_white_level", DEFAULT_DISPLAY_WHITE_LEVEL))
        )
        self.histogram_widget.set_levels(
            self.black_level_spin.value(),
            self.white_level_spin.value(),
        )

        vessel_test_image_path = cfg.get("vessel_test_image_path", "")
        if vessel_test_image_path:
            self.vessel_mode.set_test_image_path(vessel_test_image_path)

        neuron_test_image_path = cfg.get("neuron_test_image_path", "")
        if neuron_test_image_path:
            self.neuron_mode.set_test_image_path(neuron_test_image_path)

        self.update_contrast_control_state()

    def save_current_config(self):
        camera_mode = self.resolution_combo.currentData()

        if not isinstance(camera_mode, (tuple, list)) or len(camera_mode) != 3:
            QMessageBox.warning(
                self,
                "Config Error",
                "No valid resolution/FPS mode is selected.",
            )
            return

        w, h, fps = camera_mode

        config = {
            "camera_driver": camera_driver_name(),
            "camera_device_name": self.camera_device_combo.currentText().strip()
            or default_camera_device_name(),
            "ffmpeg_exe": self.ffmpeg_exe_edit.text().strip() or DEFAULT_FFMPEG_EXE,
            "camera_stream_format": self.current_camera_stream_format(),
            "fps": int(fps),
            "use_gpu_preview": bool(self.gpu_preview_checkbox.isChecked()),
            "acq_output_dir": self.acq_output_dir_edit.text().strip(),
            "acq_filename_prefix": self.acq_filename_prefix_edit.text().strip(),
            "acq_mode": self.acq_mode_combo.currentData(),
            "acq_duration_s": int(self.acq_duration_spin.value()),
            "acq_frame_count": int(self.acq_frame_count_spin.value()),
            "acq_interval_ms": int(self.acq_interval_spin.value()),
            "acq_use_gpu": bool(self.acq_gpu_checkbox.isChecked()),
            "resolution": [int(w), int(h)],
            "test_mode": bool(self.screen_test_mode_checkbox.isChecked()),
            "vessel_test_mode": bool(self.vessel_mode.is_test_mode_enabled()),
            "neuron_test_mode": bool(self.neuron_mode.is_test_mode_enabled()),
            "display_auto_contrast": bool(self.auto_contrast_checkbox.isChecked()),
            "display_black_level": int(self.black_level_spin.value()),
            "display_white_level": int(self.white_level_spin.value()),
            "display_enable_colormap": bool(self.enable_colormap_checkbox.isChecked()),
            "display_colormap": str(self.colormap_combo.currentData() or "gray"),
            "vessel_test_image_path": self.vessel_mode.get_test_image_path(),
            "neuron_test_image_path": self.neuron_mode.get_test_image_path(),
            "supported_camera_modes": self.supported_camera_modes,
            "fixed_camera_controls_enabled": bool(
                self.fixed_camera_controls_checkbox.isChecked()
            ),
            "manual_exposure": float(self.manual_exposure_spin.value()),
            "manual_gain": float(self.manual_gain_spin.value()),
            "directshow_device_index": int(self.camera_device_index_spin.value()),
            "roi_enabled": bool(self.roi_enabled),
            "roi_has_selection": bool(
                self.roi_has_selection and self._roi_is_valid(self.roi_rect)
            ),
            "roi_rect": [
                int(self.roi_rect[0]) if self.roi_rect is not None else 0,
                int(self.roi_rect[1]) if self.roi_rect is not None else 0,
                int(self.roi_rect[2]) if self.roi_rect is not None else 640,
                int(self.roi_rect[3]) if self.roi_rect is not None else 480,
            ],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        ok = save_config_to_file(config)

        if not ok:
            QMessageBox.critical(self, "Config Error", f"Failed to save config:\n{CONFIG_PATH}")
            return

        self.config = config
        self.supported_camera_modes = config.get("supported_camera_modes", [])
        self._populate_resolution_combo()
        self._apply_roi_settings()
        self.set_status(f"Config saved: {CONFIG_PATH}")

    # ---------------- Histogram / preview ----------------

    def update_contrast_control_state(self):
        manual = not self.auto_contrast_checkbox.isChecked()
        self.black_level_spin.setEnabled(manual)
        self.white_level_spin.setEnabled(manual)
        self.histogram_widget.set_draggable(manual)

    def on_colormap_toggled(self, checked: bool):
        self.colormap_combo.setEnabled(checked)
        # refresh preview after toggling colormap
        if self.latest_frame is not None:
            self._last_rendered_seq = 0
            self.on_camera_preview(
                self.latest_frame,
                self.latest_frame_seq,
                getattr(self, "_camera_preview_timestamp", time.perf_counter()),
                getattr(self, "_camera_preview_timing", {}),
            )

    def on_colormap_selection_changed(self, index: int):
        if self.enable_colormap_checkbox.isChecked() and self.latest_frame is not None:
            self.on_camera_preview(
                self.latest_frame,
                self.latest_frame_seq,
                getattr(self, "_camera_preview_timestamp", time.perf_counter()),
            )

    def on_auto_contrast_toggled(self, checked: bool):
        self.update_contrast_control_state()

    def on_histogram_levels_changed(self, black: int, white: int):
        self.auto_contrast_checkbox.blockSignals(True)
        self.auto_contrast_checkbox.setChecked(False)
        self.auto_contrast_checkbox.blockSignals(False)

        self.update_contrast_control_state()

        self.black_level_spin.blockSignals(True)
        self.white_level_spin.blockSignals(True)
        self.black_level_spin.setValue(black)
        self.white_level_spin.setValue(white)
        self.black_level_spin.blockSignals(False)
        self.white_level_spin.blockSignals(False)

    def on_level_spin_changed(self, _value=None):
        black = int(self.black_level_spin.value())
        white = int(self.white_level_spin.value())

        if black >= white:
            if self.sender() is self.black_level_spin:
                white = min(255, black + 1)
                self.white_level_spin.blockSignals(True)
                self.white_level_spin.setValue(white)
                self.white_level_spin.blockSignals(False)
            else:
                black = max(0, white - 1)
                self.black_level_spin.blockSignals(True)
                self.black_level_spin.setValue(black)
                self.black_level_spin.blockSignals(False)

        self.histogram_widget.set_levels(black, white)

    def on_camera_preview(self, preview_bgr, seq: int, timestamp: float, timing=None):
        # Keep only the newest preview frame. This prevents old frames from
        # queueing up when the UI is busy and keeps the displayed image real-time.
        self.latest_frame = preview_bgr
        self.latest_frame_seq = int(seq)
        self._camera_preview_timestamp = float(timestamp)
        self._camera_preview_timing = timing if isinstance(timing, dict) else {}

    def _update_latency_label(self):
        if getattr(self, "_camera_preview_timestamp", None) is None:
            self.latency_label.setText("Latency: -- ms")
            if hasattr(self, "pipeline_timing_label"):
                self.pipeline_timing_label.setText("Pipeline: --")
            if hasattr(self, "render_timing_label"):
                self.render_timing_label.setText("Timing: --")
            return

        latency_ms = int((time.perf_counter() - self._camera_preview_timestamp) * 1000)
        self.latency_label.setText(f"Latency: {latency_ms} ms")

    def _render_pending_preview(self):
        if self.latest_frame is None or self.latest_frame_seq == self._last_rendered_seq:
            return
        self._last_rendered_seq = self.latest_frame_seq
        self._do_render_preview(self.latest_frame)
        self._update_latency_label()

    def _do_render_preview(self, preview_bgr):
        frame_start = time.perf_counter()
        timings = {}

        t0 = time.perf_counter()
        actual_h, actual_w = preview_bgr.shape[:2]
        self.resolution_label.setText(f"Preview: {actual_w} x {actual_h}")
        timings["label_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        now = time.perf_counter()
        hist_queued = False
        if now - self._last_histogram_update > 0.1:
            # Copy only once and reuse it for histogram + auto contrast.
            # The histogram worker only reads the array, so this avoids one
            # unnecessary full-frame memory copy on every histogram update.
            hist_frame = preview_bgr.copy()
            self._request_histogram(hist_frame)
            if self.auto_contrast_checkbox.isChecked():
                self._request_auto_contrast(hist_frame)
            self._last_histogram_update = now
            hist_queued = True
        timings["hist_queue_ms"] = (time.perf_counter() - t0) * 1000.0
        timings["hist_queued"] = hist_queued

        t0 = time.perf_counter()
        black = self.black_level_spin.value()
        white = self.white_level_spin.value()
        if black != self._last_lut_black or white != self._last_lut_white or self._contrast_lut is None:
            lut = np.arange(256, dtype=np.float32)
            lut = (lut - black) * 255.0 / max(1, white - black)
            lut = np.clip(lut, 0, 255).astype(np.uint8)
            self._contrast_lut = lut
            self._last_lut_black = black
            self._last_lut_white = white
        timings["lut_build_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        display_bgr = cv2.LUT(preview_bgr, self._contrast_lut)
        timings["contrast_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.enable_colormap_checkbox.isChecked():
            cmap_name = str(self.colormap_combo.currentData() or "gray").lower()
            cv_map = {
                "gray": None,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "magma": cv2.COLORMAP_MAGMA,
                "plasma": cv2.COLORMAP_PLASMA,
                "inferno": cv2.COLORMAP_INFERNO,
                "cividis": cv2.COLORMAP_CIVIDIS,
                "turbo": cv2.COLORMAP_TURBO,
                "hot": cv2.COLORMAP_HOT,
                "jet": cv2.COLORMAP_JET,
            }
            gray8 = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY)
            if cv_map.get(cmap_name) is not None:
                display_bgr = cv2.applyColorMap(gray8, cv_map[cmap_name])
            else:
                display_bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        timings["colormap_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        qimg = self._bgr_to_qimage_fast(display_bgr)
        timings["qimage_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pix = QPixmap.fromImage(qimg)
        timings["qpixmap_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._current_display_pixmap = pix
        self.image_label.setPixmap(pix)
        timings["setpixmap_ms"] = (time.perf_counter() - t0) * 1000.0

        # Acknowledge preview consumption to camera thread. The camera thread now
        # checks this before decoding/resizing the next preview, so a slow UI
        # does not waste CPU preparing frames that will be dropped.
        t0 = time.perf_counter()
        try:
            if self.camera_thread is not None:
                self.camera_thread.preview_ack()
        except Exception:
            pass
        timings["ack_ms"] = (time.perf_counter() - t0) * 1000.0

        timings["ui_total_ms"] = (time.perf_counter() - frame_start) * 1000.0
        timings["seq"] = int(self.latest_frame_seq)
        timings["latency_ms"] = (
            (time.perf_counter() - self._camera_preview_timestamp) * 1000.0
            if getattr(self, "_camera_preview_timestamp", None) is not None
            else 0.0
        )

        camera_timing = dict(getattr(self, "_camera_preview_timing", {}) or {})
        combined_timing = {**camera_timing, **timings}
        self._render_timing_samples.append(combined_timing)
        self._maybe_update_timing_display(combined_timing)

    def _bgr_to_qimage_fast(self, frame_bgr):
        # Avoid cv2.cvtColor(BGR->RGB) for every preview frame. Qt supports
        # Format_BGR888, so this path saves one full-frame color conversion.
        if frame_bgr is None:
            return QImage()

        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        h, w = frame_bgr.shape[:2]
        bytes_per_line = int(frame_bgr.strides[0])
        return QImage(
            frame_bgr.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_BGR888,
        ).copy()

    def _avg_timing(self, key: str, default: float = 0.0) -> float:
        samples = list(self._render_timing_samples)
        if not samples:
            return float(default)
        values = [float(item.get(key, 0.0)) for item in samples if key in item]
        if not values:
            return float(default)
        return sum(values) / max(1, len(values))

    def _maybe_update_timing_display(self, timing: dict):
        now = time.perf_counter()
        if now - self._last_timing_label_update < 0.35:
            return
        self._last_timing_label_update = now

        if not getattr(self, "render_profiler_checkbox", None) or not self.render_profiler_checkbox.isChecked():
            if hasattr(self, "render_timing_label"):
                self.render_timing_label.setText("Timing: disabled")
            if hasattr(self, "pipeline_timing_label"):
                self.pipeline_timing_label.setText("Pipeline: --")
            return

        def ms(key):
            return float(timing.get(key, 0.0))

        avg_total = self._avg_timing("ui_total_ms")
        avg_latency = self._avg_timing("latency_ms")
        avg_decode = self._avg_timing("decode_ms")
        avg_resize = self._avg_timing("resize_ms")

        short_text = (
            f"Pipeline: lag {avg_latency:.0f} ms | "
            f"cam dec {avg_decode:.1f} / resize {avg_resize:.1f} ms | "
            f"UI {avg_total:.1f} ms"
        )
        self.pipeline_timing_label.setText(short_text)

        payload_kb = float(timing.get('jpeg_kb', timing.get('raw_kb', 0.0)))
        payload_label = "raw" if "raw_kb" in timing and "jpeg_kb" not in timing else "jpeg"

        detail_text = (
            "Timing: "
            f"seq {int(timing.get('seq', 0))}, "
            f"{payload_label} {payload_kb:.0f} KB, "
            f"decode {ms('decode_ms'):.1f} ms, "
            f"crop {ms('crop_ms'):.1f} ms, "
            f"resize {ms('resize_ms'):.1f} ms, "
            f"LUT {ms('contrast_ms'):.1f} ms, "
            f"cmap {ms('colormap_ms'):.1f} ms, "
            f"QImage {ms('qimage_ms'):.1f} ms, "
            f"QPixmap {ms('qpixmap_ms'):.1f} ms, "
            f"paint {ms('setpixmap_ms'):.1f} ms, "
            f"UI total {ms('ui_total_ms'):.1f} ms, "
            f"lag {ms('latency_ms'):.0f} ms, "
            f"skip {int(timing.get('preview_skip_pending_total', 0))}"
        )
        self.render_timing_label.setText(detail_text)
        self.set_status(detail_text)

    def _request_histogram(self, preview_bgr):
        if self._histogram_busy or self._histogram_worker is None:
            return
        self._histogram_busy = True
        self._histogram_worker.calculate_histogram.emit(preview_bgr)

    def _request_auto_contrast(self, preview_bgr):
        if self._histogram_worker is None:
            return
        self._histogram_worker.calculate_auto_contrast.emit(preview_bgr)

    @Slot(object)
    def _on_histogram_ready(self, hist):
        self._histogram_busy = False
        if hist is None:
            return
        self.histogram_widget.set_histogram(hist)

    @Slot(int, int)
    def _on_auto_contrast_ready(self, black, white):
        if not self.auto_contrast_checkbox.isChecked():
            return
        self.histogram_widget.set_levels(black, white)
        self.black_level_spin.blockSignals(True)
        self.white_level_spin.blockSignals(True)
        self.black_level_spin.setValue(black)
        self.white_level_spin.setValue(white)
        self.black_level_spin.blockSignals(False)
        self.white_level_spin.blockSignals(False)

    # ---------------- Camera ----------------

    def refresh_camera_devices(self, _checked=False):
        raw_devices = enumerate_video_devices(
            self.ffmpeg_exe_edit.text().strip() or DEFAULT_FFMPEG_EXE
        )
        devices = []
        seen_devices = set()
        for raw_name in raw_devices or []:
            name = str(raw_name or "").strip()
            if name and name not in seen_devices:
                seen_devices.add(name)
                devices.append(name)
        configured_name = str(
            self.config.get("camera_device_name", "")
        ).strip()
        current = (
            configured_name
            or self.camera_device_combo.currentText().strip()
            or default_camera_device_name()
        )

        self.camera_device_combo.blockSignals(True)
        self.camera_device_combo.clear()

        if devices:
            for name in devices:
                self.camera_device_combo.addItem(name)

            default_name = (
                current if current in devices else select_default_video_device(devices)
            )
            index = self.camera_device_combo.findText(default_name)
            self.camera_device_combo.setCurrentIndex(index if index >= 0 else 0)
            selected_name = self.camera_device_combo.currentText().strip()

            preferred_mode = getattr(self, "_pending_camera_mode", None)
            self._refresh_modes_for_device(selected_name, preferred_mode)
            self._pending_camera_mode = None
        else:
            self.camera_device_combo.addItem(current)
            self.camera_device_combo.setCurrentText(current)
            self.supported_camera_modes = []
            self._populate_resolution_combo(
                getattr(self, "_pending_camera_mode", None)
            )
            self._pending_camera_mode = None
            self.device_info_label.setText(
                f"Driver: {camera_driver_name()} / MJPEG + YUYV422; no video device found"
            )
            self.set_status(f"No video devices found for {camera_driver_name()}")

        self.camera_device_combo.blockSignals(False)

    def toggle_camera(self):
        if self.is_camera_running():
            self.stop_camera()
            return

        device_name = self.camera_device_combo.currentText().strip() or default_camera_device_name()
        ffmpeg_exe = self.ffmpeg_exe_edit.text().strip() or DEFAULT_FFMPEG_EXE
        camera_mode = self.resolution_combo.currentData()
        if not isinstance(camera_mode, (tuple, list)) or len(camera_mode) != 3:
            QMessageBox.warning(
                self,
                "Camera Error",
                "Please select a valid resolution/FPS mode.",
            )
            return

        width, height, fps = (int(value) for value in camera_mode)
        stream_format = self.current_camera_stream_format()
        self._apply_configured_fixed_camera_controls()

        self.camera_thread = FFmpegCameraThread(
            ffmpeg_exe=ffmpeg_exe,
            device_name=device_name,
            width=width,
            height=height,
            fps=fps,
            stream_format=stream_format,
            preview_fps=DEFAULT_PREVIEW_FPS,
            preview_max_w=DEFAULT_PREVIEW_MAX_W,
            preview_max_h=DEFAULT_PREVIEW_MAX_H,
            use_gpu_preview=bool(self.gpu_preview_checkbox.isChecked()),
            roi=(
                tuple(self.roi_rect)
                if self.roi_enabled
                and self.roi_has_selection
                and self._roi_is_valid(self.roi_rect)
                else None
            ),
            parent=self,
        )

        self.camera_thread.preview_ready.connect(self.on_camera_preview)
        self.camera_thread.fps_ready.connect(self.on_camera_fps)
        self.camera_thread.settings_ready.connect(self.on_camera_settings_ready)
        self.camera_thread.error.connect(self.on_camera_error)
        self.camera_thread.start()
        self._preview_render_timer.start()

        self.camera_start_btn.setText("Close Camera")
        self.camera_status_label.setText(
            f"Camera: {device_name} / {camera_stream_format_label(stream_format)}"
        )
        self.stream_format_combo.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.camera_device_combo.setEnabled(False)
        self.ffmpeg_exe_edit.setEnabled(False)
        self.reload_devices_btn.setEnabled(False)
        self.fixed_camera_controls_checkbox.setEnabled(False)
        self.manual_exposure_spin.setEnabled(False)
        self.manual_gain_spin.setEnabled(False)
        self.camera_device_index_spin.setEnabled(False)
        self.set_status(
            f"Opening USB Camera through {camera_driver_name()}: {device_name} "
            f"({camera_stream_format_label(stream_format)})"
        )

    def auto_open_default_camera(self):
        if not self.is_camera_running():
            self.toggle_camera()

    def stop_camera(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

        self._preview_render_timer.stop()
        self._last_rendered_seq = 0

        if self._histogram_thread is not None:
            self._histogram_thread.quit()
            self._histogram_thread.wait(500)

        self.camera_start_btn.setText("Open Camera")
        self.stream_format_combo.setEnabled(True)
        self.resolution_combo.setEnabled(True)
        self.camera_device_combo.setEnabled(True)
        self.ffmpeg_exe_edit.setEnabled(True)
        self.reload_devices_btn.setEnabled(True)
        self.fixed_camera_controls_checkbox.setEnabled(True)
        self.manual_exposure_spin.setEnabled(True)
        self.manual_gain_spin.setEnabled(True)
        self.camera_device_index_spin.setEnabled(True)
        self.camera_control_report = None
        self.fixed_camera_controls_status.setText("Not applied")
        self.camera_status_label.setText("Camera: --")
        self.fps_label.setText("Stream FPS: --")
        self.resolution_label.setText("Resolution: -- x --")
        self.latency_label.setText("Latency: -- ms")
        if hasattr(self, "pipeline_timing_label"):
            self.pipeline_timing_label.setText("Pipeline: --")
        if hasattr(self, "render_timing_label"):
            self.render_timing_label.setText("Timing: --")
        self.set_status("Camera closed")

    def on_camera_fps(self, fps: float):
        self.fps_label.setText(f"Stream FPS: {fps:.1f}")

    def on_camera_settings_ready(self, message: str):
        self.device_info_label.setText(message)
        self.set_status(message)

    def on_camera_error(self, message: str):
        QMessageBox.critical(self, "Camera Error", message)
        self.stop_camera()

    def open_native_camera_property_panel(self):
        if is_linux():
            device_name = self.camera_device_combo.currentText().strip() or default_camera_device_name()
            QMessageBox.information(
                self,
                "Linux UVC Controls",
                "Use v4l2-ctl from the terminal to inspect and set camera controls.\n\n"
                f"List controls:\n  v4l2-ctl -d {device_name} --list-ctrls\n\n"
                "Set one control:\n  v4l2-ctl -d "
                f"{device_name} --set-ctrl exposure_auto=1",
            )
            return

        device_name = self.camera_device_combo.currentText().strip() or default_camera_device_name()
        ffmpeg_exe = resolve_ffmpeg_exe(self.ffmpeg_exe_edit.text().strip() or DEFAULT_FFMPEG_EXE)

        cmd = [
            ffmpeg_exe,
            "-f",
            "dshow",
            "-show_video_device_dialog",
            "true",
            "-i",
            f"video={device_name}",
        ]

        try:
            import subprocess

            subprocess.Popen(cmd)
            self.set_status(f"Opened native DirectShow property panel: {device_name}")
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Native Panel Error",
                "Failed to open native panel.\n\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"Error: {exc}",
            )

    # ---------------- Screen / Projection ----------------

    def refresh_screens(self, _state=None):
        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
        test_mode = self.screen_test_mode_checkbox.isChecked()

        target = None
        lines = [f"Detected {len(screens)} screen(s):"]

        for i, screen in enumerate(screens):
            geo = screen.geometry()
            available = screen.availableGeometry()
            is_primary = screen == primary
            role = "Primary" if is_primary else "Secondary"

            is_target_candidate = False
            if not is_primary:
                is_target_candidate = (
                    geo.width() >= PROJECT_W and geo.height() >= PROJECT_H
                    if test_mode
                    else geo.width() == PROJECT_W and geo.height() == PROJECT_H
                )

            mark = ""
            if is_target_candidate and target is None:
                target = screen
                mark = (
                    "  <- Target screen, test popup mode"
                    if test_mode
                    else "  <- Target screen, fullscreen mode"
                )

            lines.append(
                f"{i}: {role}, geometry={geo.width()}x{geo.height()}+{geo.x()}+{geo.y()}, "
                f"available={available.width()}x{available.height()}+{available.x()}+{available.y()}{mark}"
            )

        if target is None and test_mode and primary is not None:
            target = primary
            lines.append("Using primary screen for test popup mode.")

        self.target_screen = target

        if target is not None:
            self.projection.open_on_screen(target, test_mode=test_mode)
            self.set_status("Projection screen ready")
        else:
            self.projection.hide()
            lines.append("No valid secondary projection screen was found.")
            self.set_status("Projection screen not found")

        self.screen_info_label.setText("\n".join(lines))

    def ensure_projection_window(self, warn: bool = True) -> bool:
        """Make sure the projection window is available before showing any mask/pattern.

        In Projection Screen test mode this opens/reopens a 640 x 480 popup on
        the secondary screen.  This is used by all white/mask/pattern display
        paths so a manually closed popup is restored automatically.
        """
        try:
            test_mode = bool(self.screen_test_mode_checkbox.isChecked())
        except Exception:
            test_mode = False

        if self.target_screen is None:
            self.refresh_screens()

        if self.target_screen is None:
            if warn:
                QMessageBox.warning(self, "Screen Error", "No valid projection screen was detected.")
            return False

        if (
            not self.projection.isVisible()
            or self.projection.test_mode != test_mode
        ):
            self.projection.open_on_screen(self.target_screen, test_mode=test_mode)

        return True

    def show_projection_pattern(self, image_rgb, warn: bool = True) -> bool:
        if not self.ensure_projection_window(warn=warn):
            return False
        self.projection.show_pattern(image_rgb)
        return True

    def show_projection_white(self, warn: bool = True) -> bool:
        if not self.ensure_projection_window(warn=warn):
            return False
        self.projection.show_white()
        return True

    def show_projection_black(self, warn: bool = False) -> bool:
        if not self.ensure_projection_window(warn=warn):
            return False
        self.projection.show_black()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.latest_frame is not None:
            self._last_rendered_seq = 0
            self.on_camera_preview(
                self.latest_frame,
                self.latest_frame_seq,
                getattr(self, "_camera_preview_timestamp", time.perf_counter()),
                getattr(self, "_camera_preview_timing", {}),
            )

    def closeEvent(self, event):

        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.stop_image_acquisition()
        self.sim_mode.abort_if_active()
        self.sparse_sim_mode.abort_if_active()
        self.vessel_mode.abort_if_active()
        self.neuron_mode.abort_if_active()

        self.save_current_config()
        self.stop_camera()
        self.projection.close()

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

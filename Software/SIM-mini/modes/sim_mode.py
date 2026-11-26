import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.constants import PROJECT_H, PROJECT_W


SIM_START_ANGLE_CHOICES = (0, 5, 10, 15, 20, 25, 45)
MIN_FRAME_WAIT_TIMEOUT_S = 5.0
FRAME_WAIT_TIMEOUT_PER_FRAME_S = 2.0


@dataclass
class SIMPattern:
    angle_deg: int
    phase_index: int
    shift_px: int
    image_rgb: np.ndarray


class PatternGenerator:
    @staticmethod
    def _coordinate_grid(width, height, angle_deg):
        y, x = np.indices((height, width), dtype=np.float32)
        theta = np.deg2rad(angle_deg)
        return -np.sin(theta) * x + np.cos(theta) * y

    @staticmethod
    def generate_stripe_pattern(
        width,
        height,
        stripe_width,
        angle_deg,
        shift_px,
    ):
        coord = PatternGenerator._coordinate_grid(width, height, angle_deg)
        band_index = np.floor((coord + shift_px) / stripe_width).astype(np.int32)
        bright = (band_index % 2) == 0
        gray = np.where(bright, 255, 0).astype(np.uint8)
        return np.dstack([gray, gray, gray])

    @staticmethod
    def generate_sim_sequence(
        stripe_width,
        start_angle_deg=15,
        width=PROJECT_W,
        height=PROJECT_H,
    ):
        if stripe_width not in (3, 6):
            raise ValueError("stripe_width must be 3 or 6")

        angles = [
            (int(start_angle_deg) + offset) % 360
            for offset in (0, 120, 240)
        ]
        period = 2 * stripe_width
        shift_step = period // 3
        shifts = [0, shift_step, 2 * shift_step]

        patterns = []
        for angle in angles:
            for phase_index, shift in enumerate(shifts):
                image = PatternGenerator.generate_stripe_pattern(
                    width=width,
                    height=height,
                    stripe_width=stripe_width,
                    angle_deg=angle,
                    shift_px=shift,
                )
                patterns.append(
                    SIMPattern(angle, phase_index, shift, image)
                )
        return patterns

    @staticmethod
    def generate_sparse_stripe_pattern(
        width,
        height,
        period_px,
        bright_width_px,
        angle_deg,
        shift_px,
    ):
        if period_px < 3:
            raise ValueError("period_px must be at least 3")
        if not 1 <= bright_width_px < period_px:
            raise ValueError(
                "bright_width_px must be at least 1 and less than period_px"
            )

        coord = PatternGenerator._coordinate_grid(width, height, angle_deg)
        phase = np.mod(
            np.floor(coord + shift_px).astype(np.int32),
            period_px,
        )
        bright = phase < bright_width_px
        gray = np.where(bright, 255, 0).astype(np.uint8)
        return np.dstack([gray, gray, gray])

    @staticmethod
    def generate_sparse_sim_sequence(
        period_px=12,
        bright_width_px=1,
        width=PROJECT_W,
        height=PROJECT_H,
    ):
        angles = [0, 90]

        patterns = []
        for angle in angles:
            for phase_index, shift in enumerate(range(period_px)):
                image = PatternGenerator.generate_sparse_stripe_pattern(
                    width=width,
                    height=height,
                    period_px=period_px,
                    bright_width_px=bright_width_px,
                    angle_deg=angle,
                    shift_px=shift,
                )
                patterns.append(
                    SIMPattern(angle, phase_index, shift, image)
                )
        return patterns


class PatternAcquisitionMode(QWidget):
    display_name = "SIM"
    output_prefix = "sim"

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.patterns = []
        self.index = 0
        self.output_dir = None
        self.raw_widefield_dir = None
        self.active = False
        self.is_white_capture_active = False
        self.frame_seq_at_pattern = 0
        self.frame_seq_after_settle = 0
        self.frames_to_discard_for_capture = 0
        self.target_frame_sequence = 1
        self.pattern_show_t0 = 0.0
        self.capture_wait_t0 = 0.0
        self.capture_window_armed = False
        self.source_dtype = None
        self.acquisition_created_at = None
        self.acquisition_start_t0 = 0.0
        self.raw_widefield_record = None
        self.captured_pattern_records = []
        self.parameter_widgets = []
        self._build_ui()

    def _build_ui(self):
        raise NotImplementedError

    def _pattern_request(self):
        raise NotImplementedError

    def _set_controls_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        for widget in self.parameter_widgets:
            widget.setEnabled(enabled)

    def _pattern_discard_frames(self):
        return self.pattern_transition_discard_frames_spin.value()

    def _mode_acquisition_parameters(self):
        return {}

    def _arm_pattern_capture(self, discard_frames=0):
        self.frame_seq_at_pattern = self.app.get_latest_sequence()
        self.frames_to_discard_for_capture = max(0, int(discard_frames))
        self.frame_seq_after_settle = self.frame_seq_at_pattern
        self.target_frame_sequence = self.frame_seq_at_pattern + 1
        self.pattern_show_t0 = time.perf_counter()
        self.capture_wait_t0 = 0.0
        self.capture_window_armed = False

    @staticmethod
    def _pattern_filename(index, pattern):
        return (
            f"pattern_{index + 1:02d}_"
            f"angle_{pattern.angle_deg:03d}_"
            f"phase_{pattern.phase_index:02d}_"
            f"shift_{pattern.shift_px}px.png"
        )

    @staticmethod
    def _pattern_tiff_filename(index, pattern):
        """Return the uncompressed TIFF name used for acquisition records."""
        return (
            f"pattern_{index + 1:02d}_"
            f"angle_{pattern.angle_deg:03d}_"
            f"phase_{pattern.phase_index:02d}_"
            f"shift_{pattern.shift_px}px.tif"
        )

    @staticmethod
    def _frame_to_tiff_data(frame_bgr):
        """Convert camera BGR to TIFF RGB without changing dtype/bit depth."""
        frame = np.asarray(frame_bgr)
        if frame.ndim == 2:
            return np.ascontiguousarray(frame)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raise ValueError(f"Unsupported camera frame shape: {frame.shape}")

    @classmethod
    def _tiff_task(cls, path, frame_bgr):
        data = cls._frame_to_tiff_data(frame_bgr)
        photometric = "minisblack" if data.ndim == 2 else "rgb"
        return {
            "kind": "tiff",
            "path": path,
            "data": data,
            "kwargs": {
                "photometric": photometric,
                "compression": None,
            },
        }

    def _write_acquisition_metadata(self, status=None):
        if self.output_dir is None:
            return

        try:
            stream_format = self.app.current_camera_stream_format()
        except Exception:
            stream_format = "unknown"

        if stream_format == "mjpeg":
            transport = "MJPEG (lossy compressed camera transport)"
        elif stream_format == "yuyv422":
            transport = "YUYV422 (uncompressed camera transport)"
        else:
            transport = "unknown"

        try:
            camera = self.app.camera_acquisition_metadata()
        except Exception as exc:
            camera = {"metadata_error": str(exc), "stream_format": stream_format}

        patterns = [
            {
                "frame_index": index,
                "angle_deg": int(pattern.angle_deg),
                "phase_index": int(pattern.phase_index),
                "shift_px": int(pattern.shift_px),
            }
            for index, pattern in enumerate(self.patterns, start=1)
        ]

        metadata = {
            "metadata_version": 2,
            "mode": self.display_name,
            "output_dir": str(self.output_dir.resolve()),
            "camera_stream_format": stream_format,
            "camera_transport": transport,
            "camera": camera,
            "saved_image_format": "TIFF",
            "tiff_compression": "none",
            "saved_color_order": "RGB",
            "source_dtype": self.source_dtype,
            "source_bit_depth": (
                int(np.dtype(self.source_dtype).itemsize * 8)
                if self.source_dtype else None
            ),
            "status": status or ("running" if self.active else "finished"),
            "saved_frame_count": int(self.index),
            "save_projection_patterns": bool(
                getattr(self, "save_projection_patterns_check", None)
                and self.save_projection_patterns_check.isChecked()
            ),
            "acquisition_parameters": {
                "projection_width_px": int(PROJECT_W),
                "projection_height_px": int(PROJECT_H),
                "projection_settling_ms": int(self.settle_spin.value()),
                "startup_warmup_frames": int(
                    self.startup_warmup_frames_spin.value()
                ),
                "pattern_transition_discard_frames": int(
                    self.pattern_transition_discard_frames_spin.value()
                ),
                "first_pattern_extra_discard_frames": (
                    int(self.first_pattern_extra_discard_frames_spin.value())
                    if hasattr(self, "first_pattern_extra_discard_frames_spin")
                    else 0
                ),
                "expected_pattern_frame_count": len(self.patterns),
                "widefield_frame_count": 1,
                "frame_wait_timeout_min_s": float(MIN_FRAME_WAIT_TIMEOUT_S),
                "frame_wait_timeout_per_frame_s": float(
                    FRAME_WAIT_TIMEOUT_PER_FRAME_S
                ),
                **self._mode_acquisition_parameters(),
            },
            "pattern_sequence": patterns,
            "raw_widefield_capture": self.raw_widefield_record,
            "pattern_frame_captures": self.captured_pattern_records,
            "created_at": self.acquisition_created_at
            or time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.output_dir / "acquisition_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)

    def _load_or_generate_patterns(
        self,
        patterns_lib_dir,
        expected_count,
        generator,
        save_images=True,
    ):
        save_images = bool(save_images)
        metadata_path = patterns_lib_dir / "patterns.json"

        loaded_patterns = []
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
                items = (
                    metadata.get("patterns", [])
                    if isinstance(metadata, dict)
                    else metadata
                )
                for item in items:
                    pattern_path = patterns_lib_dir / item.get("filename", "")
                    if not pattern_path.exists():
                        raise FileNotFoundError(
                            f"Pattern file missing: {pattern_path}"
                        )
                    bgr = cv2.imread(str(pattern_path))
                    if bgr is None:
                        raise IOError(
                            f"Failed to read pattern image: {pattern_path}"
                        )
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (PROJECT_H, PROJECT_W):
                        raise ValueError(
                            f"Unexpected pattern size: {rgb.shape[:2]}"
                        )
                    loaded_patterns.append(
                        SIMPattern(
                            int(item.get("angle_deg", 0)),
                            int(item.get("phase_index", 0)),
                            int(item.get("shift_px", 0)),
                            rgb,
                        )
                    )
                if len(loaded_patterns) != expected_count:
                    raise ValueError(
                        f"Expected {expected_count} cached patterns, "
                        f"found {len(loaded_patterns)}"
                    )
            except Exception:
                loaded_patterns = []

        if loaded_patterns:
            return loaded_patterns

        patterns = generator()
        if len(patterns) != expected_count:
            raise ValueError(
                f"Pattern generator returned {len(patterns)} images; "
                f"expected {expected_count}"
            )

        if save_images:
            patterns_lib_dir.mkdir(parents=True, exist_ok=True)
            items = []
            for index, pattern in enumerate(patterns):
                filename = self._pattern_filename(index, pattern)
                target = patterns_lib_dir / filename
                if not cv2.imwrite(
                    str(target),
                    cv2.cvtColor(pattern.image_rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise IOError(f"Failed to save pattern image: {target}")
                items.append(
                    {
                        "filename": filename,
                        "angle_deg": pattern.angle_deg,
                        "phase_index": pattern.phase_index,
                        "shift_px": pattern.shift_px,
                    }
                )

            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(
                    {"version": 2, "patterns": items},
                    file,
                    indent=2,
                )
        return patterns

    def start(self):
        if self.active:
            return
        if not self.app.ensure_projection_window(warn=True):
            return
        if not self.app.is_camera_running():
            QMessageBox.warning(
                self,
                "Camera Error",
                "Please open the camera first.",
            )
            return
        if not self.app.has_latest_camera_frame():
            QMessageBox.warning(
                self,
                "Camera Error",
                "No camera frame has been received yet.",
            )
            return

        controls_check = getattr(
            self.app,
            "ensure_fixed_camera_controls_for_sim",
            None,
        )
        if callable(controls_check):
            controls_ok, controls_error = controls_check()
            if not controls_ok:
                QMessageBox.warning(
                    self,
                    "Camera Control Error",
                    controls_error,
                )
                return

        try:
            cache_dir, expected_count, generator = self._pattern_request()
            self.patterns = self._load_or_generate_patterns(
                cache_dir,
                expected_count,
                generator,
                save_images=self.save_projection_patterns_check.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Pattern Error",
                f"Unable to prepare {self.display_name} patterns:\n{exc}",
            )
            return

        self.output_dir = (
            self.app.resolve_output_base_dir()
            / time.strftime(f"{self.output_prefix}_%Y%m%d_%H%M%S")
        )
        (self.output_dir / "camera_original_tiff").mkdir(
            parents=True,
            exist_ok=True,
        )
        if self.save_projection_patterns_check.isChecked():
            (self.output_dir / "patterns").mkdir(parents=True, exist_ok=True)
        self.raw_widefield_dir = self.output_dir / "raw_widefield_tiff"
        self.raw_widefield_dir.mkdir(parents=True, exist_ok=True)

        self.index = 0
        self.active = True
        self.is_white_capture_active = True
        self.source_dtype = None
        self.acquisition_created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.acquisition_start_t0 = time.perf_counter()
        self.raw_widefield_record = None
        self.captured_pattern_records = []
        self._write_acquisition_metadata(status="running")
        self._set_controls_enabled(False)
        self.app.set_status(
            f"{self.display_name} acquisition started. "
            f"Discarding {self.startup_warmup_frames_spin.value()} startup "
            f"frame(s), then capturing one white-field reference before "
            f"{len(self.patterns)} pattern frames. Output: {self.output_dir}"
        )

        if not self.app.show_projection_white():
            self._abort("Unable to show white-field illumination.")
            return
        self._arm_pattern_capture(self.startup_warmup_frames_spin.value())
        QTimer.singleShot(10, self._try_capture_current_pattern)

    def _show_next_pattern(self):
        if not self.active:
            return
        if self.index >= len(self.patterns):
            self._finish()
            return

        pattern = self.patterns[self.index]
        if not self.app.show_projection_pattern(pattern.image_rgb):
            self._abort("Unable to show the next projection pattern.")
            return

        if self.save_projection_patterns_check.isChecked():
            pattern_name = self._pattern_tiff_filename(self.index, pattern)
            self.app.start_save_worker(
                [
                    self._tiff_task(
                        self.output_dir / "patterns" / pattern_name,
                        pattern.image_rgb,
                    )
                ],
                label=f"{self.display_name} pattern save",
            )

        transition_discard_frames = self._pattern_discard_frames()
        self._arm_pattern_capture(transition_discard_frames)
        self.app.set_status(
            f"Projecting {self.index + 1}/{len(self.patterns)}: "
            f"angle={pattern.angle_deg} deg, "
            f"phase={pattern.phase_index}, shift={pattern.shift_px}px; "
            f"discarding {transition_discard_frames} transition frame(s)"
        )
        QTimer.singleShot(10, self._try_capture_current_pattern)

    def _try_capture_current_pattern(self):
        if not self.active:
            return

        settle_s = self.settle_spin.value() / 1000.0
        elapsed = time.perf_counter() - self.pattern_show_t0

        if elapsed < settle_s:
            QTimer.singleShot(10, self._try_capture_current_pattern)
            return

        # Start counting discarded frames only after the projection has had
        # time to settle. Frames received during the settling interval can
        # still belong to the previous illumination because FFmpeg and the
        # camera both buffer frames.
        if not self.capture_window_armed:
            self.frame_seq_after_settle = self.app.get_latest_sequence()
            self.target_frame_sequence = (
                self.frame_seq_after_settle
                + self.frames_to_discard_for_capture
                + 1
            )
            self.capture_wait_t0 = time.perf_counter()
            self.capture_window_armed = True
            QTimer.singleShot(10, self._try_capture_current_pattern)
            return

        latest_sequence = self.app.get_latest_sequence()
        if latest_sequence < self.target_frame_sequence:
            discard_frames = self.frames_to_discard_for_capture
            required_frames = discard_frames + 1
            frame_timeout_s = max(
                MIN_FRAME_WAIT_TIMEOUT_S,
                required_frames * FRAME_WAIT_TIMEOUT_PER_FRAME_S,
            )
            capture_wait_elapsed = time.perf_counter() - self.capture_wait_t0
            if capture_wait_elapsed < frame_timeout_s:
                QTimer.singleShot(10, self._try_capture_current_pattern)
                return
            received_frames = max(
                0,
                latest_sequence - self.frame_seq_after_settle,
            )
            self._abort(
                f"Only {received_frames} new camera frame(s) were received "
                f"within {frame_timeout_s:.1f} s after projection settling "
                f"completed; {required_frames} are required "
                f"({discard_frames} discarded + 1 captured)."
            )
            return

        frame_bgr, seq = self.app.get_latest_full_bgr()
        if frame_bgr is None:
            self._abort("Camera frame is empty during acquisition.")
            return

        self.source_dtype = str(np.asarray(frame_bgr).dtype)

        if self.is_white_capture_active:
            frame_name = f"raw_widefield_seq_{seq}.tif"
            save_path = self.raw_widefield_dir / frame_name
            self.is_white_capture_active = False
            self.raw_widefield_record = {
                "filename": str(Path("raw_widefield_tiff") / frame_name),
                "captured_sequence": int(seq),
                "projection_sequence_at_show": int(self.frame_seq_at_pattern),
                "sequence_after_settle": int(self.frame_seq_after_settle),
                "target_sequence": int(self.target_frame_sequence),
                "discard_frames": int(self.frames_to_discard_for_capture),
                "captured_elapsed_ms": round(
                    (time.perf_counter() - self.acquisition_start_t0) * 1000.0,
                    3,
                ),
            }
            self.app.start_save_worker(
                [self._tiff_task(save_path, frame_bgr)],
                label=f"{self.display_name} raw widefield TIFF save",
            )
            self.app.set_status(
                f"Captured white-field frame seq={seq}. "
                f"Starting {len(self.patterns)} pattern frames..."
            )
            self._write_acquisition_metadata(status="running")
            QTimer.singleShot(20, self._show_next_pattern)
            return

        pattern = self.patterns[self.index]
        frame_name = (
            f"frame_{self.index + 1:02d}_"
            f"angle_{pattern.angle_deg:03d}_"
            f"phase_{pattern.phase_index:02d}_"
            f"shift_{pattern.shift_px}px_"
            f"seq_{seq}.tif"
        )
        self.captured_pattern_records.append(
            {
                "frame_index": int(self.index + 1),
                "filename": str(Path("camera_original_tiff") / frame_name),
                "angle_deg": int(pattern.angle_deg),
                "phase_index": int(pattern.phase_index),
                "shift_px": int(pattern.shift_px),
                "captured_sequence": int(seq),
                "projection_sequence_at_show": int(self.frame_seq_at_pattern),
                "sequence_after_settle": int(self.frame_seq_after_settle),
                "target_sequence": int(self.target_frame_sequence),
                "discard_frames": int(self.frames_to_discard_for_capture),
                "captured_elapsed_ms": round(
                    (time.perf_counter() - self.acquisition_start_t0) * 1000.0,
                    3,
                ),
            }
        )
        self.app.start_save_worker(
            [
                self._tiff_task(
                    self.output_dir / "camera_original_tiff" / frame_name,
                    frame_bgr,
                )
            ],
            label=f"{self.display_name} camera TIFF save",
        )

        self.index += 1
        self._write_acquisition_metadata(status="running")
        QTimer.singleShot(20, self._show_next_pattern)

    def _finish(self):
        self.active = False
        self._set_controls_enabled(True)
        self.app.show_projection_black()
        self._write_acquisition_metadata(status="finished")
        self.app.set_status(
            f"{self.display_name} acquisition finished: {self.output_dir}"
        )
        QMessageBox.information(
            self,
            "Acquisition Finished",
            f"One widefield reference and {len(self.patterns)} pattern "
            f"frames have been saved to:\n{self.output_dir}",
        )

    def _abort(self, reason):
        self.active = False
        self._set_controls_enabled(True)
        self.app.show_projection_black()
        self._write_acquisition_metadata(status=f"aborted: {reason}")
        self.app.set_status(
            f"{self.display_name} acquisition aborted: {reason}"
        )
        QMessageBox.critical(self, "Acquisition Aborted", reason)

    def abort_if_active(self):
        if self.active:
            self._abort("Window closed. Acquisition aborted.")


class SIMMode(PatternAcquisitionMode):
    display_name = "SIM"
    output_prefix = "sim"

    def _build_ui(self):
        layout = QVBoxLayout(self)
        sim_group = QGroupBox("SIM Acquisition Parameters")
        form = QFormLayout(sim_group)

        self.stripe_width_combo = QComboBox()
        self.stripe_width_combo.addItems(["3", "6"])

        self.start_angle_combo = QComboBox()
        for angle in SIM_START_ANGLE_CHOICES:
            label = f"{angle} deg"
            if angle == 15:
                label += " (recommended)"
            self.start_angle_combo.addItem(label, angle)
        self.start_angle_combo.setCurrentIndex(
            SIM_START_ANGLE_CHOICES.index(15)
        )
        self.start_angle_combo.setToolTip(
            "Choose a start angle away from horizontal/vertical pixel axes. "
            "15 deg is the recommended default."
        )
        self.start_angle_combo.currentIndexChanged.connect(
            self._update_angle_summary
        )

        self.angle_summary_label = QLabel()

        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(0, 2000)
        self.settle_spin.setValue(120)
        self.settle_spin.setSuffix(" ms")

        self.startup_warmup_frames_spin = QSpinBox()
        self.startup_warmup_frames_spin.setRange(0, 30)
        self.startup_warmup_frames_spin.setValue(2)
        self.startup_warmup_frames_spin.setToolTip(
            "Discard camera frames once, after the initial white-field "
            "illumination turns on. Use 2 for the typical YUYV422 "
            "startup pipeline delay; use 0 if no warm-up is needed."
        )

        self.pattern_transition_discard_frames_spin = QSpinBox()
        self.pattern_transition_discard_frames_spin.setRange(0, 30)
        self.pattern_transition_discard_frames_spin.setValue(2)
        self.pattern_transition_discard_frames_spin.setToolTip(
            "Discard stale camera frames after every projected pattern "
            "change. Use 2 for the typical camera/display pipeline delay."
        )

        self.first_pattern_extra_discard_frames_spin = QSpinBox()
        self.first_pattern_extra_discard_frames_spin.setRange(0, 30)
        self.first_pattern_extra_discard_frames_spin.setValue(4)
        self.first_pattern_extra_discard_frames_spin.setToolTip(
            "Discard additional frames only when changing from the white-field "
            "reference to the first SIM pattern. Start with 4 for YUYV422."
        )

        form.addRow("Stripe Width", self.stripe_width_combo)
        form.addRow("Start Angle", self.start_angle_combo)
        form.addRow("Pattern Angles", self.angle_summary_label)
        form.addRow("Projection Settling Time", self.settle_spin)
        form.addRow(
            "Startup Warm-up Frames",
            self.startup_warmup_frames_spin,
        )
        form.addRow(
            "Pattern-change Discard Frames",
            self.pattern_transition_discard_frames_spin,
        )
        form.addRow(
            "First-pattern Extra Discard Frames",
            self.first_pattern_extra_discard_frames_spin,
        )
        self.save_projection_patterns_check = QCheckBox(
            "Save projected pattern TIFFs"
        )
        self.save_projection_patterns_check.setChecked(False)
        self.save_projection_patterns_check.setToolTip(
            "When enabled, save each projected SIM pattern as an uncompressed TIFF."
        )
        form.addRow("Projection Image Output", self.save_projection_patterns_check)
        layout.addWidget(sim_group)

        self.start_btn = QPushButton("Start 9-Frame SIM Acquisition")
        self.start_btn.clicked.connect(self.start)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)

        self.parameter_widgets = [
            self.stripe_width_combo,
            self.start_angle_combo,
            self.settle_spin,
            self.startup_warmup_frames_spin,
            self.pattern_transition_discard_frames_spin,
            self.first_pattern_extra_discard_frames_spin,
            self.save_projection_patterns_check,
        ]
        self._update_angle_summary()

    def _update_angle_summary(self):
        start = int(self.start_angle_combo.currentData())
        angles = [(start + offset) % 360 for offset in (0, 120, 240)]
        self.angle_summary_label.setText(
            ", ".join(str(angle) for angle in angles) + " deg"
        )

    def _pattern_discard_frames(self):
        discard_frames = super()._pattern_discard_frames()
        if self.index == 0:
            discard_frames += self.first_pattern_extra_discard_frames_spin.value()
        return discard_frames

    def _mode_acquisition_parameters(self):
        stripe_width = int(self.stripe_width_combo.currentText())
        start_angle = int(self.start_angle_combo.currentData())
        return {
            "sim_type": "conventional_three_phase",
            "stripe_width_px": stripe_width,
            "stripe_period_px": 2 * stripe_width,
            "phase_count_per_angle": 3,
            "start_angle_deg": start_angle,
            "angles_deg": [
                (start_angle + offset) % 360 for offset in (0, 120, 240)
            ],
            "phase_shifts_px": [0, (2 * stripe_width) // 3, 2 * ((2 * stripe_width) // 3)],
        }

    def _pattern_request(self):
        stripe_width = int(self.stripe_width_combo.currentText())
        start_angle = int(self.start_angle_combo.currentData())
        cache_dir = (
            Path.cwd()
            / "sim_patterns"
            / (
                f"regular_width_{stripe_width}_"
                f"start_{start_angle:03d}"
            )
        )
        return (
            cache_dir,
            9,
            lambda: PatternGenerator.generate_sim_sequence(
                stripe_width=stripe_width,
                start_angle_deg=start_angle,
            ),
        )


class SparseSIMMode(PatternAcquisitionMode):
    display_name = "Sparse SIM"
    output_prefix = "sparse_sim"

    def _build_ui(self):
        layout = QVBoxLayout(self)
        sim_group = QGroupBox("Sparse SIM Acquisition Parameters")
        form = QFormLayout(sim_group)

        self.period_spin = QSpinBox()
        self.period_spin.setRange(3, 64)
        self.period_spin.setValue(12)
        self.period_spin.setSuffix(" px")
        self.period_spin.valueChanged.connect(
            self._update_sparse_controls
        )

        self.bright_width_spin = QSpinBox()
        self.bright_width_spin.setRange(1, 11)
        self.bright_width_spin.setValue(1)
        self.bright_width_spin.setSuffix(" px")

        self.angle_summary_label = QLabel("0, 90 deg (fixed)")

        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(0, 2000)
        self.settle_spin.setValue(120)
        self.settle_spin.setSuffix(" ms")

        self.startup_warmup_frames_spin = QSpinBox()
        self.startup_warmup_frames_spin.setRange(0, 30)
        self.startup_warmup_frames_spin.setValue(2)
        self.startup_warmup_frames_spin.setToolTip(
            "Discard camera frames once, after the initial white-field "
            "illumination turns on. Use 2 for the typical YUYV422 "
            "startup pipeline delay; use 0 if no warm-up is needed."
        )

        self.pattern_transition_discard_frames_spin = QSpinBox()
        self.pattern_transition_discard_frames_spin.setRange(0, 30)
        self.pattern_transition_discard_frames_spin.setValue(2)
        self.pattern_transition_discard_frames_spin.setToolTip(
            "Discard stale camera frames after every projected pattern "
            "change. Use 2 for the typical camera/display pipeline delay."
        )

        self.first_pattern_extra_discard_frames_spin = QSpinBox()
        self.first_pattern_extra_discard_frames_spin.setRange(0, 30)
        self.first_pattern_extra_discard_frames_spin.setValue(4)
        self.first_pattern_extra_discard_frames_spin.setToolTip(
            "Discard additional frames only for the first sparse pattern, "
            "allowing camera exposure to recover from the white field."
        )

        form.addRow("Sparse Period", self.period_spin)
        form.addRow("Bright Stripe Width", self.bright_width_spin)
        form.addRow("Pattern Angles", self.angle_summary_label)
        form.addRow("Projection Settling Time", self.settle_spin)
        form.addRow(
            "Startup Warm-up Frames",
            self.startup_warmup_frames_spin,
        )
        form.addRow(
            "Pattern-change Discard Frames",
            self.pattern_transition_discard_frames_spin,
        )
        form.addRow(
            "First-pattern Extra Discard Frames",
            self.first_pattern_extra_discard_frames_spin,
        )
        self.save_projection_patterns_check = QCheckBox(
            "Save projected pattern TIFFs"
        )
        self.save_projection_patterns_check.setChecked(False)
        self.save_projection_patterns_check.setToolTip(
            "When enabled, save each projected sparse-SIM pattern as an uncompressed TIFF."
        )
        form.addRow("Projection Image Output", self.save_projection_patterns_check)
        layout.addWidget(sim_group)

        self.start_btn = QPushButton()
        self.start_btn.clicked.connect(self.start)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)

        self.parameter_widgets = [
            self.period_spin,
            self.bright_width_spin,
            self.settle_spin,
            self.startup_warmup_frames_spin,
            self.pattern_transition_discard_frames_spin,
            self.first_pattern_extra_discard_frames_spin,
            self.save_projection_patterns_check,
        ]
        self._update_sparse_controls()

    def _pattern_discard_frames(self):
        discard_frames = super()._pattern_discard_frames()
        if self.index == 0:
            discard_frames += self.first_pattern_extra_discard_frames_spin.value()
        return discard_frames

    def _update_sparse_controls(self):
        period = int(self.period_spin.value())
        self.bright_width_spin.setMaximum(max(1, period - 1))
        if self.bright_width_spin.value() >= period:
            self.bright_width_spin.setValue(period - 1)
        self.start_btn.setText(
            f"Start {2 * period}-Frame Sparse SIM Acquisition"
        )

    def _mode_acquisition_parameters(self):
        period = int(self.period_spin.value())
        return {
            "sim_type": "sparse",
            "period_px": period,
            "bright_width_px": int(self.bright_width_spin.value()),
            "phase_count_per_angle": period,
            "angles_deg": [0, 90],
            "phase_shifts_px": list(range(period)),
        }

    def _pattern_request(self):
        period = int(self.period_spin.value())
        bright_width = int(self.bright_width_spin.value())
        cache_dir = (
            Path.cwd()
            / "sim_patterns"
            / f"sparse_period_{period}_bright_{bright_width}"
        )
        return (
            cache_dir,
            2 * period,
            lambda: PatternGenerator.generate_sparse_sim_sequence(
                period_px=period,
                bright_width_px=bright_width,
            ),
        )

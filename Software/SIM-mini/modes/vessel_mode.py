import os
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology
from skimage.morphology import disk
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.constants import PROJECT_H
from core.image_io import (
    add_panel_title,
    camera_frame_to_gray_640_float,
    gray_to_rgb_u8,
    load_local_image_as_bgr,
    mask_to_rgb_u8,
    normalize01,
    percentile_normalize,
)
from core.widgets import MaskReviewDialog


@dataclass
class VesselParams:
    algorithm: str = "Frangi vesselness"
    invert: bool = False
    background_radius: int = 80
    denoise_sigma: float = 1.5
    clahe_clip: float = 0.003
    min_sigma: float = 1.2
    max_sigma: float = 6.0
    sigma_count: int = 7
    high_percentile: float = 98.0
    low_ratio: float = 0.65
    original_fusion_weight: float = 0.38
    min_object_size: int = 300
    closing_radius: int = 1


DEFAULT_VESSEL_PARAMS = VesselParams()


def preprocess_vessel_image(img: np.ndarray, p: VesselParams) -> np.ndarray:
    x = percentile_normalize(img)

    if p.invert:
        x = 1.0 - x

    if p.background_radius > 0:
        background = morphology.opening(x, footprint=disk(int(p.background_radius)))
        x = normalize01(x - background)

    if p.denoise_sigma > 0:
        x = filters.gaussian(x, sigma=float(p.denoise_sigma), preserve_range=True, mode="reflect")
        x = normalize01(x)

    if p.clahe_clip > 0:
        x = exposure.equalize_adapthist(x, clip_limit=float(p.clahe_clip))
        x = normalize01(x)

    return x.astype(np.float32)


def get_vessel_sigmas(p: VesselParams) -> np.ndarray:
    mn = max(0.2, float(p.min_sigma))
    mx = max(mn, float(p.max_sigma))
    if p.sigma_count <= 1:
        return np.array([mn], dtype=np.float32)
    return np.linspace(mn, mx, int(p.sigma_count)).astype(np.float32)


def vesselness_filter(img: np.ndarray, p: VesselParams) -> np.ndarray:
    method = p.algorithm.lower()
    sigmas = get_vessel_sigmas(p)
    kwargs = dict(sigmas=sigmas, black_ridges=False, mode="reflect")

    if method == "frangi vesselness":
        response = filters.frangi(img, **kwargs)
    elif method == "sato tubeness":
        response = filters.sato(img, **kwargs)
    elif method == "meijering curvilinear":
        response = filters.meijering(img, **kwargs)
    else:
        response = filters.frangi(img, **kwargs)

    return normalize01(np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0))


def hysteresis_on_response(response: np.ndarray, high_percentile: float, low_ratio: float) -> np.ndarray:
    r = normalize01(response)
    vals = r[np.isfinite(r) & (r > 0)]
    if vals.size < 10:
        return np.zeros_like(r, dtype=bool)

    high = float(np.percentile(vals, high_percentile))
    low = float(max(0.0, min(high * low_ratio, high)))
    if high <= 0:
        return np.zeros_like(r, dtype=bool)

    return filters.apply_hysteresis_threshold(r, low, high).astype(bool)


def clean_vessel_mask(mask: np.ndarray, p: VesselParams) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)

    if p.min_object_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(p.min_object_size))

    if p.closing_radius > 0:
        mask = morphology.binary_closing(mask, footprint=disk(int(p.closing_radius)))

    mask = ndi.binary_fill_holes(mask)

    if p.min_object_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(p.min_object_size))

    return mask.astype(bool)


def auto_contrast_original_for_fusion(raw_gray: np.ndarray) -> np.ndarray:
    """Auto-contrast the original image before fusing it with the enhanced response."""
    return percentile_normalize(raw_gray, p_low=0.5, p_high=99.5).astype(np.float32)


def make_vessel_mask_source(raw_gray: np.ndarray, response: np.ndarray, p: VesselParams) -> np.ndarray:
    """Fuse enhanced vessel response with auto-contrast original intensity.

    The fusion coefficient controls how much auto-contrast original image is
    added to the enhanced vesselness response. This helps keep thick vessel
    interiors from becoming hollow while vesselness still provides the main
    structural cue.
    """
    resp = normalize01(response)
    original_ac = auto_contrast_original_for_fusion(raw_gray)
    w_original = float(max(0.0, min(1.0, p.original_fusion_weight)))
    return normalize01((1.0 - w_original) * resp + w_original * original_ac).astype(np.float32)


def mask_from_enhanced_result(raw_gray: np.ndarray, preprocessed: np.ndarray, response: np.ndarray, p: VesselParams) -> np.ndarray:
    response_mask = hysteresis_on_response(response, p.high_percentile, p.low_ratio)
    fused = make_vessel_mask_source(raw_gray, response, p)
    fused_mask = hysteresis_on_response(fused, max(50.0, float(p.high_percentile) - 4.0), p.low_ratio)

    # Only borrow fused/original support close to response-positive structures.
    # This prevents bright background from flooding the mask while filling thick
    # vessel interiors that are bounded by strong vesselness edges.
    support_radius = max(2, int(p.closing_radius) * 2 + 2)
    if np.count_nonzero(response_mask) > 0:
        support = morphology.binary_dilation(response_mask, footprint=disk(support_radius))
        mask = response_mask | (fused_mask & support)
    else:
        mask = fused_mask

    return clean_vessel_mask(mask, p)


def run_vessel_pipeline(frame_bgr: np.ndarray, p: VesselParams) -> dict:
    raw_gray = camera_frame_to_gray_640_float(frame_bgr)
    preprocessed = preprocess_vessel_image(raw_gray, p)
    response = vesselness_filter(preprocessed, p)
    mask_source = make_vessel_mask_source(raw_gray, response, p)
    mask = mask_from_enhanced_result(raw_gray, preprocessed, response, p)

    return {
        "raw_gray": raw_gray,
        "preprocessed": preprocessed,
        "response": response,
        "mask_source": mask_source,
        "mask": mask,
    }


def make_vessel_mosaic(result: dict) -> np.ndarray:
    original_rgb = add_panel_title(gray_to_rgb_u8(result["raw_gray"]), "Original")
    enhanced_rgb = add_panel_title(gray_to_rgb_u8(result.get("mask_source", result["response"])), "Enhanced + Original")
    mask_rgb = add_panel_title(mask_to_rgb_u8(result["mask"]), "Mask")
    separator = np.full((PROJECT_H, 6, 3), 40, dtype=np.uint8)
    return np.hstack([original_rgb, separator, enhanced_rgb, separator, mask_rgb])


class VesselProcessingThread(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, frame_bgr: np.ndarray, params: VesselParams, parent=None):
        super().__init__(parent)
        self.frame_bgr = frame_bgr.copy()
        self.params = params

    def run(self):
        try:
            result = run_vessel_pipeline(self.frame_bgr, self.params)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class VesselMode(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.processing_thread = None
        self.result_dialog = None
        self.active = False
        self.output_dir = None
        self.frame_seq_at_mask = 0
        self.mask_show_t0 = 0.0
        self.last_frame_bgr = None
        self.last_result = None
        self.last_mosaic = None
        self.current_params = VesselParams(**DEFAULT_VESSEL_PARAMS.__dict__)
        self._live_update_busy = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        capture_group = QGroupBox("Vessel Mask Calculation")
        form = QFormLayout(capture_group)

        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 5000)
        self.delay_spin.setValue(300)
        self.delay_spin.setSuffix(" ms")

        self.mask_capture_delay_spin = QSpinBox()
        self.mask_capture_delay_spin.setRange(0, 5000)
        self.mask_capture_delay_spin.setValue(300)
        self.mask_capture_delay_spin.setSuffix(" ms")

        self.test_mode_checkbox = QCheckBox("Test mode (use local image)")
        self.test_mode_checkbox.setChecked(False)
        self.test_mode_checkbox.toggled.connect(self._update_test_image_controls)

        self.test_image_edit = QLineEdit()
        self.test_image_edit.setPlaceholderText("Local vessel image for Test Mode")

        self.test_browse_btn = QPushButton("Browse")
        self.test_browse_btn.clicked.connect(self.choose_test_image)

        test_image_row = QHBoxLayout()
        test_image_row.addWidget(self.test_image_edit)
        test_image_row.addWidget(self.test_browse_btn)

        form.addRow("White illumination delay", self.delay_spin)
        form.addRow("Mask capture delay", self.mask_capture_delay_spin)
        form.addRow("Test mode", self.test_mode_checkbox)
        form.addRow("Test image", test_image_row)
        layout.addWidget(capture_group)
        self._update_test_image_controls()

        hint = QLabel("Mask processing parameters are edited in the Vessel Mask Result / Parameters window after calculation.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Calculate Mask")
        self.run_btn.clicked.connect(self.start)
        self.show_mask_btn = QPushButton("Show Mask")
        self.show_mask_btn.setEnabled(False)
        self.show_mask_btn.clicked.connect(self.show_last_mask)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.show_mask_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def is_test_mode_enabled(self):
        return bool(self.test_mode_checkbox.isChecked())

    def set_test_mode_enabled(self, enabled: bool):
        self.test_mode_checkbox.setChecked(bool(enabled))
        self._update_test_image_controls()

    def _update_test_image_controls(self, _checked=None):
        enabled = not self.active if hasattr(self, "active") else True
        use_test = bool(self.test_mode_checkbox.isChecked())
        self.test_image_edit.setEnabled(enabled and use_test)
        self.test_browse_btn.setEnabled(enabled and use_test)

    def choose_test_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Local Vessel Image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if path:
            self.test_image_edit.setText(path)
            self.app.set_status(f"Selected test vessel image: {path}")

    def get_test_image_path(self):
        return self.test_image_edit.text().strip()

    def set_test_image_path(self, path):
        self.test_image_edit.setText(str(path or ""))

    def collect_params(self) -> VesselParams:
        return VesselParams(**self.current_params.__dict__)

    def validate_params(self, p: VesselParams) -> bool:
        if p.max_sigma < p.min_sigma:
            QMessageBox.warning(self, "Invalid Parameters", "Max sigma must be >= Min sigma.")
            return False
        return True

    def set_controls_enabled(self, enabled: bool):
        for w in [self.run_btn, self.delay_spin, self.mask_capture_delay_spin, self.test_mode_checkbox]:
            w.setEnabled(enabled)
        self._update_test_image_controls()

    def start(self):
        if self.active:
            return

        vessel_test_mode = self.is_test_mode_enabled()

        if not vessel_test_mode:
            if not self.app.is_camera_running():
                QMessageBox.warning(self, "Camera Error", "Please open the camera first.")
                return
            if not self.app.has_latest_camera_frame():
                QMessageBox.warning(self, "Camera Error", "No camera frame has been received yet.")
                return
            if not self.app.show_projection_white(warn=True):
                return
        else:
            test_image_path = self.get_test_image_path()
            if not test_image_path:
                QMessageBox.warning(self, "Missing Test Image", "Please choose a local vessel image.")
                return
            if not os.path.exists(test_image_path):
                QMessageBox.warning(self, "Invalid Test Image", "The selected local vessel image does not exist.")
                return
            self.app.show_projection_white(warn=False)

        p = self.collect_params()
        if not self.validate_params(p):
            return

        self.active = True
        self.set_controls_enabled(False)
        self.show_mask_btn.setEnabled(False)

        delay_ms = int(self.delay_spin.value())
        if vessel_test_mode:
            self.app.set_status(f"Vessel test mode: calculating vessel mask from local image after {delay_ms} ms...")
        else:
            self.app.set_status(f"Vessel mask calculation: white image projected, waiting {delay_ms} ms...")

        QTimer.singleShot(delay_ms, self.capture_and_process)

    def capture_and_process(self):
        if not self.active:
            return

        try:
            if self.is_test_mode_enabled():
                frame = load_local_image_as_bgr(self.get_test_image_path())
            else:
                frame = self.app.get_latest_frame_bgr(full=True)
                if frame is None:
                    raise RuntimeError("No camera frame is available.")
        except Exception as e:
            self.active = False
            self.set_controls_enabled(True)
            QMessageBox.warning(self, "Input Error", str(e))
            self.app.set_status("Vessel input failed")
            return

        self.last_frame_bgr = frame.copy()
        self.app.set_status("Processing vessel mask...")
        self.processing_thread = VesselProcessingThread(frame, self.collect_params(), self)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_failed)
        self.processing_thread.start()

    def _ensure_output_dir(self):
        if self.output_dir is None:
            self.output_dir = self.app.resolve_output_base_dir() / time.strftime("vessel_%Y%m%d_%H%M%S")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _make_save_tasks(self, result: dict, mosaic: np.ndarray):
        raw_u8 = (normalize01(result["raw_gray"]) * 255).astype(np.uint8)
        pre_u8 = (normalize01(result["preprocessed"]) * 255).astype(np.uint8)
        response_u8 = (normalize01(result["response"]) * 255).astype(np.uint8)
        fused_u8 = (normalize01(result.get("mask_source", result["response"])) * 255).astype(np.uint8)
        mask_u8 = result["mask"].astype(np.uint8) * 255
        mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)

        out = self._ensure_output_dir()
        return [
            {"kind": "image", "path": out / "vessel_original_640x480.png", "data": raw_u8},
            {"kind": "image", "path": out / "vessel_preprocessed_640x480.png", "data": pre_u8},
            {"kind": "image", "path": out / "vessel_response_640x480.png", "data": response_u8},
            {"kind": "image", "path": out / "vessel_enhanced_plus_original_640x480.png", "data": fused_u8},
            {"kind": "image", "path": out / "vessel_mask_640x480.png", "data": mask_u8},
            {"kind": "image", "path": out / "vessel_compare.png", "data": mosaic_bgr},
        ]

    def save_last_result(self, show_message: bool = False):
        if self.last_result is None or self.last_mosaic is None:
            QMessageBox.information(self, "Save Mask", "No vessel mask has been calculated yet.")
            return

        self.apply_review_params_to_main_if_available()
        out = self._ensure_output_dir()
        self.app.start_save_worker(
            self._make_save_tasks(self.last_result, self.last_mosaic),
            label="Vessel result save",
        )
        self.app.set_status(f"Vessel mask saved to: {out}")
        if show_message:
            QMessageBox.information(self, "Save Mask", f"Vessel mask and comparison images were queued for saving to:\n{out}")

    def on_processing_finished(self, result: dict):
        self.active = False
        self.set_controls_enabled(True)

        self.last_result = result
        self.last_mosaic = make_vessel_mosaic(result)
        self.show_mask_btn.setEnabled(True)

        self.save_last_result(show_message=False)
        self.show_result_dialog()

        mask_area = int(np.count_nonzero(result["mask"]))
        self.app.set_status(f"Vessel mask calculated and saved. Mask area: {mask_area} px. Use 'Show Mask' to project it.")

    def show_last_mask(self):
        if self.last_result is None:
            QMessageBox.information(self, "Show Mask", "Please calculate a vessel mask first.")
            return

        mask_rgb = mask_to_rgb_u8(self.last_result["mask"])
        if not self.app.show_projection_pattern(mask_rgb, warn=True):
            return

        mask_area = int(np.count_nonzero(self.last_result["mask"]))
        if self.is_test_mode_enabled():
            self.app.set_status(f"Vessel mask shown in test mode popup. Mask area: {mask_area} px.")
            return

        if self.app.is_camera_running() and self.app.has_latest_camera_frame():
            self.frame_seq_at_mask = self.app.get_latest_sequence()
            self.mask_show_t0 = time.perf_counter()
            delay_ms = int(self.mask_capture_delay_spin.value())
            self.app.set_status(f"Vessel mask projected. Capturing mask-illumination frame after {delay_ms} ms...")
            QTimer.singleShot(delay_ms, self.capture_mask_illumination_frame)
        else:
            self.app.set_status(f"Vessel mask projected. Mask area: {mask_area} px.")

    def show_result_dialog(self):
        if isinstance(self.result_dialog, MaskReviewDialog) and self.result_dialog.isVisible():
            self.result_dialog.set_image(self.last_mosaic)
            return

        panel = self._build_review_params_panel()
        self.result_dialog = MaskReviewDialog(
            self.last_mosaic,
            "Vessel Mask Result / Parameters",
            params_widget=panel,
            parent=self,
        )
        self.result_dialog.recalculate_btn.clicked.connect(self.recalculate_from_review_dialog)
        self.result_dialog.save_btn.clicked.connect(lambda: self.save_last_result(show_message=True))
        self.result_dialog.show()

    def _new_review_double_spin(self, value, minimum, maximum, decimals=2, step=0.1):
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        return spin

    def _add_spin_slider_row(self, form, label, spin, slider_min, slider_max, scale, live_update=True):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(slider_min), int(slider_max))
        slider.setValue(int(round(float(spin.value()) * scale)))

        row = QHBoxLayout()
        row.addWidget(spin)
        row.addWidget(slider, stretch=1)
        form.addRow(label, row)

        def spin_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(int(round(float(value) * scale)))
            slider.blockSignals(False)
            if live_update:
                self.update_vessel_mask_live_from_review()

        def slider_to_spin(value):
            spin.blockSignals(True)
            if isinstance(spin, QDoubleSpinBox):
                spin.setValue(float(value) / float(scale))
            else:
                spin.setValue(int(round(float(value) / float(scale))))
            spin.blockSignals(False)
            if live_update:
                self.update_vessel_mask_live_from_review()

        spin.valueChanged.connect(spin_to_slider)
        slider.valueChanged.connect(slider_to_spin)
        return slider

    def _build_review_params_panel(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        p = self.current_params

        enhance_group = QGroupBox("Enhancement Parameters")
        enhance_form = QFormLayout(enhance_group)

        self.review_algorithm_combo = QComboBox()
        self.review_algorithm_combo.addItems(["Frangi vesselness", "Sato tubeness", "Meijering curvilinear"])
        self.review_algorithm_combo.setCurrentText(p.algorithm)

        self.review_invert_check = QCheckBox("Invert image")
        self.review_invert_check.setChecked(bool(p.invert))

        self.review_bg_radius_spin = QSpinBox()
        self.review_bg_radius_spin.setRange(0, 300)
        self.review_bg_radius_spin.setValue(int(p.background_radius))

        self.review_denoise_sigma_spin = self._new_review_double_spin(p.denoise_sigma, 0.0, 10.0, 2, 0.1)
        self.review_clahe_clip_spin = self._new_review_double_spin(p.clahe_clip, 0.0, 0.2, 4, 0.002)
        self.review_min_sigma_spin = self._new_review_double_spin(p.min_sigma, 0.2, 50.0, 2, 0.2)
        self.review_max_sigma_spin = self._new_review_double_spin(p.max_sigma, 0.2, 80.0, 2, 0.5)

        self.review_sigma_count_spin = QSpinBox()
        self.review_sigma_count_spin.setRange(1, 30)
        self.review_sigma_count_spin.setValue(int(p.sigma_count))

        enhance_form.addRow("Algorithm", self.review_algorithm_combo)
        enhance_form.addRow("Invert", self.review_invert_check)
        enhance_form.addRow("Background radius", self.review_bg_radius_spin)
        enhance_form.addRow("Gaussian denoise sigma", self.review_denoise_sigma_spin)
        enhance_form.addRow("CLAHE clip", self.review_clahe_clip_spin)
        enhance_form.addRow("Min sigma", self.review_min_sigma_spin)
        enhance_form.addRow("Max sigma", self.review_max_sigma_spin)
        enhance_form.addRow("Scale count", self.review_sigma_count_spin)

        mask_group = QGroupBox("Mask From Enhanced Result")
        mask_form = QFormLayout(mask_group)

        self.review_high_pct_spin = self._new_review_double_spin(p.high_percentile, 50.0, 99.9, 1, 0.5)
        self.review_low_ratio_spin = self._new_review_double_spin(p.low_ratio, 0.05, 0.99, 2, 0.05)
        self.review_fusion_weight_spin = self._new_review_double_spin(p.original_fusion_weight, 0.0, 1.0, 2, 0.05)
        self.review_min_obj_spin = QSpinBox()
        self.review_min_obj_spin.setRange(0, 1000000)
        self.review_min_obj_spin.setValue(int(p.min_object_size))
        self.review_close_radius_spin = QSpinBox()
        self.review_close_radius_spin.setRange(0, 50)
        self.review_close_radius_spin.setValue(int(p.closing_radius))

        self.review_high_pct_slider = self._add_spin_slider_row(mask_form, "High percentile", self.review_high_pct_spin, 500, 999, 10.0)
        self.review_low_ratio_slider = self._add_spin_slider_row(mask_form, "Low ratio", self.review_low_ratio_spin, 5, 99, 100.0)
        self.review_fusion_weight_slider = self._add_spin_slider_row(mask_form, "Original fusion coefficient", self.review_fusion_weight_spin, 0, 100, 100.0)
        self.review_min_obj_slider = self._add_spin_slider_row(mask_form, "Min object size", self.review_min_obj_spin, 0, 10000, 1.0)
        self.review_close_radius_slider = self._add_spin_slider_row(mask_form, "Closing radius", self.review_close_radius_spin, 0, 50, 1.0)

        help_label = QLabel("Dragging these sliders updates the Enhanced + Original and Mask panels immediately; use Recalculate after changing enhancement parameters.")
        help_label.setWordWrap(True)
        mask_form.addRow(help_label)

        outer.addWidget(enhance_group)
        outer.addWidget(mask_group)
        outer.addStretch(1)
        return container

    def collect_review_params(self) -> VesselParams:
        return VesselParams(
            algorithm=self.review_algorithm_combo.currentText(),
            invert=self.review_invert_check.isChecked(),
            background_radius=int(self.review_bg_radius_spin.value()),
            denoise_sigma=float(self.review_denoise_sigma_spin.value()),
            clahe_clip=float(self.review_clahe_clip_spin.value()),
            min_sigma=float(self.review_min_sigma_spin.value()),
            max_sigma=float(self.review_max_sigma_spin.value()),
            sigma_count=int(self.review_sigma_count_spin.value()),
            high_percentile=float(self.review_high_pct_spin.value()),
            low_ratio=float(self.review_low_ratio_spin.value()),
            original_fusion_weight=float(self.review_fusion_weight_spin.value()),
            min_object_size=int(self.review_min_obj_spin.value()),
            closing_radius=int(self.review_close_radius_spin.value()),
        )

    def apply_params_to_main(self, p: VesselParams):
        self.current_params = VesselParams(**p.__dict__)

    def apply_review_params_to_main_if_available(self):
        if not hasattr(self, "review_algorithm_combo"):
            return
        try:
            self.apply_params_to_main(self.collect_review_params())
        except RuntimeError:
            pass

    def update_vessel_mask_live_from_review(self):
        if self._live_update_busy or self.last_result is None or not hasattr(self, "review_high_pct_spin"):
            return
        try:
            p = self.collect_review_params()
        except RuntimeError:
            return
        if p.max_sigma < p.min_sigma:
            return

        self._live_update_busy = True
        try:
            updated = dict(self.last_result)
            updated["mask_source"] = make_vessel_mask_source(updated["raw_gray"], updated["response"], p)
            updated["mask"] = mask_from_enhanced_result(updated["raw_gray"], updated["preprocessed"], updated["response"], p)
            self.last_result = updated
            self.last_mosaic = make_vessel_mosaic(updated)
            if isinstance(self.result_dialog, MaskReviewDialog):
                self.result_dialog.set_image(self.last_mosaic)
            self.show_mask_btn.setEnabled(True)
            self.app.set_status(f"Vessel mask live-updated. Mask area: {int(np.count_nonzero(updated['mask']))} px.")
        finally:
            self._live_update_busy = False

    def recalculate_from_review_dialog(self):
        if self.last_frame_bgr is None:
            QMessageBox.warning(self, "Recalculate Mask", "No source image is available for recalculation.")
            return

        p = self.collect_review_params()
        if not self.validate_params(p):
            return

        self.apply_params_to_main(p)
        self.active = True
        self.set_controls_enabled(False)
        self.show_mask_btn.setEnabled(False)
        self.app.set_status("Recalculating vessel mask from dialog parameters...")
        self.processing_thread = VesselProcessingThread(self.last_frame_bgr, p, self)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_failed)
        self.processing_thread.start()

    def capture_mask_illumination_frame(self):
        if self.is_test_mode_enabled():
            return

        if not self.app.is_camera_running() or not self.app.has_latest_camera_frame():
            self.app.set_status("Mask-illumination capture skipped: camera is not running or no frame")
            return

        elapsed = time.perf_counter() - self.mask_show_t0
        if self.app.get_latest_sequence() <= self.frame_seq_at_mask and elapsed < 2.0:
            QTimer.singleShot(20, self.capture_mask_illumination_frame)
            return

        jpeg, seq = self.app.get_latest_jpeg()
        frame = self.app.get_latest_frame_bgr(full=True)
        out = self._ensure_output_dir()

        tasks = []
        if jpeg is not None:
            tasks.append({"kind": "bytes", "path": out / f"mask_illumination_camera_frame_seq_{seq}.jpg", "data": jpeg})
        if frame is not None:
            tasks.append({"kind": "image", "path": out / "mask_illumination_camera_frame_decoded.png", "data": frame})
        if not tasks:
            self.app.set_status("Mask-illumination capture skipped: no camera frame")
            return
        self.app.start_save_worker(tasks, label="Mask illumination frame save")
        self.app.set_status(f"Mask-illumination camera frame queued for saving: {out}")

    def on_processing_failed(self, error_text: str):
        self.active = False
        self.set_controls_enabled(True)
        self.show_mask_btn.setEnabled(self.last_result is not None)
        QMessageBox.critical(self, "Vessel Processing Failed", error_text)
        self.app.set_status("Vessel processing failed")

    def abort_if_active(self):
        if self.active:
            self.active = False
            self.set_controls_enabled(True)
            self.show_mask_btn.setEnabled(self.last_result is not None)
            self.app.show_projection_black(warn=False)
            self.app.set_status("Vessel mask calculation aborted.")

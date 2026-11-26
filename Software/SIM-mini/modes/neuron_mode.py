import os
import time
import traceback
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import tifffile as tiff
from skimage import feature, filters, measure, morphology
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.image_io import (
    add_panel_title,
    imread_unicode,
    camera_frame_to_gray_u8,
    gray_to_rgb_u8,
    mask_to_rgb_u8,
    normalize01,
    resize_mask_to_projector,
    rgb_to_qimage,
)
from core.widgets import MaskReviewDialog, ReviewImageLabel


@dataclass
class NeuronParams:
    frame_count: int = 100
    capture_interval_ms: int = 100
    frame_rate_hz: float = 10.0

    # Pre-processing
    do_motion_correction: bool = True
    max_motion_shift_px: int = 20
    do_temporal_min_background: bool = True
    do_spatial_background: bool = True
    spatial_background_sigma: float = 25.0

    # Projection / score-map parameters
    projection_percentile: float = 95.0
    gaussian_sigma: float = 1.2
    min_distance: int = 8
    threshold_percentile: float = 97.0
    max_rois: int = 300
    roi_radius_px: int = 6

    # Sensitivity / candidate seed generation.
    # Sensitive mode deliberately proposes more candidates; users can delete false positives
    # in Manual ROI Mark. Balanced/Conservative progressively reduce weak candidates.
    detection_sensitivity: str = "Sensitive"  # Sensitive / Balanced / Conservative
    enable_multithreshold_seed: bool = True
    enable_log_seed: bool = True
    log_min_sigma: float = 1.5
    log_max_sigma: float = 6.0
    log_num_sigma: int = 6
    log_threshold: float = 0.035

    # Spatial-temporal ROI refinement
    region_corr_threshold: float = 0.22
    min_roi_area_px: int = 8
    max_roi_area_px: int = 500
    min_roi_snr: float = 0.70
    max_neuropil_corr: float = 0.92

    # Memory / speed guardrails for large local Test stacks.
    # The original stack can easily be 1000+ px wide and hundreds of frames; automatic ROI
    # detection works on a down-sampled working copy so z-score/correlation maps do not
    # create multi-GB temporary arrays.
    max_processing_frames: int = 200
    max_processing_side_px: int = 640
    max_score_frames: int = 120


DEFAULT_NEURON_PARAMS = NeuronParams()


def _to_uint8_gray_stack(arr: np.ndarray) -> np.ndarray:
    """Convert an image / stack / hyperstack-like array to uint8 T x Y x X.

    This helper is intentionally permissive for Test mode:
    - 2-D images become one frame.
    - RGB/RGBA images become one grayscale frame.
    - T x Y x X stacks are preserved.
    - Y x X x T raw arrays are detected heuristically and moved to T x Y x X.
    - Higher-dimensional non-RGB arrays are flattened across all non-spatial axes.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        stack = arr[None, :, :]

    elif arr.ndim == 3:
        # Single RGB/RGBA image: H x W x C.
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                rgb = (normalize01(rgb.astype(np.float32)) * 255).astype(np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            stack = gray[None, :, :]
        # Raw NumPy-style H x W x T stack. Keep TIFF axes handling in _read_tiff_stack_as_tyx,
        # but make npy/npz Test mode robust when users save data as HWT.
        elif arr.shape[0] > 16 and arr.shape[1] > 16 and arr.shape[-1] < min(arr.shape[0], arr.shape[1]):
            stack = np.moveaxis(arr, -1, 0)
        else:
            stack = arr

    elif arr.ndim > 3:
        # If the last axis is RGB/RGBA, convert first, then flatten all non-spatial axes into time.
        if arr.shape[-1] in (3, 4):
            leading = arr.shape[:-3]
            h, w, c = arr.shape[-3:]
            rgb = arr.reshape((-1, h, w, c))[..., :3]
            if rgb.dtype != np.uint8:
                rgb = (normalize01(rgb.astype(np.float32)) * 255).astype(np.uint8)
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
            stack = np.stack(gray_frames, axis=0)
        else:
            h, w = arr.shape[-2], arr.shape[-1]
            stack = arr.reshape((-1, h, w))
    else:
        raise ValueError(f"Unsupported test image shape: {arr.shape}")

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Failed to convert test image to T x Y x X stack: {stack.shape}")

    if stack.dtype == np.uint8:
        return np.ascontiguousarray(stack)

    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        if info.max <= info.min:
            return np.zeros(stack.shape, dtype=np.uint8)
        out = np.empty(stack.shape, dtype=np.uint8)
        scale = 255.0 / float(info.max - info.min)
        offset = float(info.min)
        # Convert one frame at a time to avoid a full-stack float32 temporary for uint16 TIFFs.
        for i in range(stack.shape[0]):
            frame_f = (stack[i].astype(np.float32) - offset) * scale
            out[i] = np.clip(frame_f, 0, 255).astype(np.uint8)
        return out

    stack_f = stack.astype(np.float32)
    finite = stack_f[np.isfinite(stack_f)]
    if finite.size and float(finite.max()) <= 1.0 and float(finite.min()) >= 0.0:
        pass
    else:
        stack_f = normalize01(stack_f)

    return (np.clip(stack_f, 0, 1) * 255).astype(np.uint8)


def _read_tiff_pages_as_stack(tif):
    """Fallback reader for classic multi-page TIFFs where tif.series[0] is one YX frame.

    Pre-allocates the stack instead of collecting page arrays in a Python list and then
    stacking them, avoiding a second full-size copy for large Test TIFFs.
    """
    n_pages = len(tif.pages)
    if n_pages <= 1:
        return None

    first = np.asarray(tif.pages[0].asarray())
    first_shape = first.shape
    if len(first_shape) < 2:
        return None

    out = np.empty((n_pages,) + first_shape, dtype=first.dtype)
    out[0] = first
    for i in range(1, n_pages):
        frame = np.asarray(tif.pages[i].asarray())
        if frame.shape != first_shape:
            return None
        out[i] = frame
    return out

def _read_tiff_stack_as_tyx(path: str) -> np.ndarray:
    """Read TIFF/ImageJ/OME/classic multipage data as a T x Y x X uint8 stack.

    For Neuron Test mode, preserving multiple frames is more important than selecting one
    channel/Z plane. Therefore non-spatial axes are flattened into the time dimension unless
    they clearly represent RGB/RGBA samples.
    """
    with tiff.TiffFile(str(path)) as tif:
        if not tif.series:
            raise ValueError("TIFF file does not contain any image series.")
        series = tif.series[0]
        axes = series.axes or ""
        arr = np.asarray(series.asarray())
        page_stack = _read_tiff_pages_as_stack(tif)

    original_shape = arr.shape
    original_axes = axes

    # Some classic multi-page TIFFs expose series[0] as one YX image although pages contains T frames.
    if arr.ndim == 2 and page_stack is not None:
        return _to_uint8_gray_stack(page_stack)

    if axes and len(axes) == arr.ndim:
        keep = [i for i, size in enumerate(arr.shape) if size != 1]
        if len(keep) != arr.ndim:
            arr = np.squeeze(arr)
            axes = "".join(axes[i] for i in keep)
    else:
        arr = np.squeeze(arr)
        axes = ""

    if arr.ndim == 2:
        return _to_uint8_gray_stack(arr[None, :, :])

    if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        non_spatial = [i for i in range(arr.ndim) if i not in (y_axis, x_axis)]
        order = non_spatial + [y_axis, x_axis]
        arr = np.transpose(arr, order)
        axes = "".join(axes[i] for i in order)
        prefix_axes = axes[:-2]

        if not prefix_axes:
            if page_stack is not None:
                return _to_uint8_gray_stack(page_stack)
            return _to_uint8_gray_stack(arr[None, :, :])

        # Sample axis with 3/4 values usually means RGB/RGBA rather than time.
        sample_axis_pos = None
        for i, ax in enumerate(prefix_axes):
            if ax == "S" and arr.shape[i] in (3, 4):
                sample_axis_pos = i
                break

        if sample_axis_pos is not None:
            prefix_len = len(prefix_axes)
            other_prefix = [i for i in range(prefix_len) if i != sample_axis_pos]
            # Convert prefix..., S, Y, X -> other_prefix..., Y, X, S, then _to_uint8 flattens any remaining prefix.
            rgb_like = np.transpose(arr, other_prefix + [prefix_len, prefix_len + 1, sample_axis_pos])
            return _to_uint8_gray_stack(rgb_like)

        # Put explicit time axes first when available, then flatten all remaining non-spatial axes.
        time_axis_pos = None
        for time_char in ("T", "I", "Q"):
            if time_char in prefix_axes:
                time_axis_pos = prefix_axes.index(time_char)
                break

        prefix_len = len(prefix_axes)
        if time_axis_pos is not None:
            reorder_prefix = [time_axis_pos] + [i for i in range(prefix_len) if i != time_axis_pos]
            arr = np.transpose(arr, reorder_prefix + [prefix_len, prefix_len + 1])

        h, w = arr.shape[-2], arr.shape[-1]
        movie = arr.reshape((-1, h, w))
        if movie.ndim != 3:
            raise ValueError(
                f"Failed to parse TIFF as T x Y x X. shape={original_shape}, axes={original_axes}, parsed={movie.shape}"
            )
        return _to_uint8_gray_stack(movie)

    # No usable axes metadata. If series collapsed to one frame but pages has many frames, use pages.
    if page_stack is not None and np.asarray(arr).ndim == 2:
        return _to_uint8_gray_stack(page_stack)

    return _to_uint8_gray_stack(arr)

def load_neuron_test_stack(path: str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _read_tiff_stack_as_tyx(path)
    elif suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".npz":
        data = np.load(str(path))
        if not data.files:
            raise ValueError("NPZ file does not contain any arrays.")
        arr = data[data.files[0]]
    else:
        img = imread_unicode(path)
        if img is None:
            raise ValueError(f"Failed to read test image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img
    return _to_uint8_gray_stack(arr)


def sample_stack_time(stack: np.ndarray, max_frames=None) -> np.ndarray:
    """Uniformly sample a T x Y x X stack without copying when no sampling is needed."""
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected T x Y x X stack, got {stack.shape}")
    if max_frames is None or int(max_frames) <= 0 or stack.shape[0] <= int(max_frames):
        return stack
    idx = np.linspace(0, stack.shape[0] - 1, int(max_frames)).astype(np.int64)
    return stack[idx]


def resize_stack_max_side(stack: np.ndarray, max_side_px=None) -> np.ndarray:
    """Downsample a stack so max(H, W) <= max_side_px while preserving aspect ratio."""
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected T x Y x X stack, got {stack.shape}")
    if max_side_px is None or int(max_side_px) <= 0:
        return stack

    T, H, W = stack.shape
    max_side = int(max_side_px)
    if max(H, W) <= max_side:
        return stack

    scale = float(max_side) / float(max(H, W))
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    out = np.empty((T, new_h, new_w), dtype=stack.dtype)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    for i in range(T):
        out[i] = cv2.resize(stack[i], (new_w, new_h), interpolation=interpolation)
    return out


def prepare_neuron_stack_for_processing(stack: np.ndarray, max_frames: int = 200, max_side_px: int = 640) -> np.ndarray:
    """Create a memory-safe working stack for automatic ROI detection and manual review.

    The raw local Test TIFF may be very large, e.g. 346 x 1303 x 1839. Running local
    correlation on that full stack requires several GB of RAM. This function samples and
    resizes frames in one streaming loop, so it does not create a large sampled copy first.
    """
    stack = _to_uint8_gray_stack(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected T x Y x X stack, got {stack.shape}")

    T, H, W = stack.shape
    max_frames = int(max_frames) if max_frames is not None else 0
    if max_frames > 0 and T > max_frames:
        frame_indices = np.linspace(0, T - 1, max_frames).astype(np.int64)
    else:
        frame_indices = np.arange(T, dtype=np.int64)

    max_side = int(max_side_px) if max_side_px is not None else 0
    if max_side > 0 and max(H, W) > max_side:
        scale = float(max_side) / float(max(H, W))
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    else:
        new_h, new_w = H, W
        interpolation = None

    # Fast path: no frame sampling, no resize, already contiguous uint8.
    if len(frame_indices) == T and new_h == H and new_w == W:
        return np.ascontiguousarray(stack, dtype=np.uint8)

    out = np.empty((len(frame_indices), new_h, new_w), dtype=np.uint8)
    for out_i, src_i in enumerate(frame_indices):
        frame = stack[int(src_i)]
        if new_h == H and new_w == W:
            out[out_i] = frame
        else:
            out[out_i] = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    return out

def percentile_normalize01(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(img, [float(p_low), float(p_high)])
    if hi <= lo:
        return normalize01(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_projection(stack: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    return np.percentile(stack.astype(np.float32), float(percentile), axis=0).astype(np.float32)


def subtract_temporal_min_background(stack: np.ndarray):
    stack = np.asarray(stack, dtype=np.float32)
    background = np.nanmin(stack, axis=0).astype(np.float32)
    corrected = stack - background[None, :, :]
    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), background


def apply_rigid_motion_correction(stack: np.ndarray, max_shift_px: int = 20):
    """Rigid x/y motion correction using phase correlation.

    This is intentionally lightweight and dependency-free beyond OpenCV. Frames with
    implausible shifts or low phase-correlation response are left unchanged.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        return stack.astype(np.float32), np.zeros((int(stack.shape[0]), 3), dtype=np.float32)

    T, H, W = stack.shape
    sample_step = max(1, T // 50)
    reference = np.median(stack[::sample_step], axis=0).astype(np.float32)
    ref_norm = percentile_normalize01(reference, 1.0, 99.5).astype(np.float32)

    corrected = np.empty_like(stack, dtype=np.float32)
    shifts = np.zeros((T, 3), dtype=np.float32)  # dx, dy, response
    max_shift_px = int(max(0, max_shift_px))

    for i in range(T):
        frame = stack[i].astype(np.float32)
        frame_norm = percentile_normalize01(frame, 1.0, 99.5).astype(np.float32)
        try:
            (dx, dy), response = cv2.phaseCorrelate(ref_norm, frame_norm)
        except Exception:
            dx, dy, response = 0.0, 0.0, 0.0

        if max_shift_px <= 0 or abs(dx) > max_shift_px or abs(dy) > max_shift_px or response < 0.02:
            dx, dy = 0.0, 0.0

        # phaseCorrelate estimates how the current frame is shifted relative to the reference;
        # apply the inverse shift to align it back to the reference.
        M = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]], dtype=np.float32)
        corrected[i] = cv2.warpAffine(
            frame,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        shifts[i] = [float(dx), float(dy), float(response)]

    return corrected.astype(np.float32), shifts


def subtract_spatial_background(stack: np.ndarray, sigma: float = 25.0):
    """Remove slow spatial background/neuropil-like haze frame-by-frame.

    This implementation stores only the corrected stack and a running mean background.
    The previous version kept a full 3-D background stack, which doubled memory usage.
    """
    stack = np.asarray(stack, dtype=np.float32)
    sigma = float(max(0.1, sigma))
    corrected = np.empty_like(stack, dtype=np.float32)
    background_sum = np.zeros(stack.shape[1:], dtype=np.float64)

    for i, frame in enumerate(stack):
        frame_f = frame.astype(np.float32, copy=False)
        bg = cv2.GaussianBlur(frame_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        corr = frame_f - bg
        corr -= np.percentile(corr, 1.0)
        corr[corr < 0] = 0
        corrected[i] = corr.astype(np.float32, copy=False)
        background_sum += bg.astype(np.float64, copy=False)

    mean_background = (background_sum / max(1, stack.shape[0])).astype(np.float32)
    return corrected, mean_background

def compute_dff_movie(stack: np.ndarray, baseline_percentile: float = 20.0) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"dF/F expects T x Y x X stack, got {stack.shape}")

    if stack.shape[0] < 2:
        return np.zeros_like(stack, dtype=np.float32)

    f0 = np.percentile(stack, float(baseline_percentile), axis=0).astype(np.float32)
    eps = max(float(np.percentile(f0, 5.0)), 1.0)
    dff = (stack - f0[None, :, :]) / (f0[None, :, :] + eps)
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(dff, [0.5, 99.5])
    if hi > lo:
        dff = np.clip(dff, lo, hi)
    return dff.astype(np.float32)


def zscore_movie(movie: np.ndarray) -> np.ndarray:
    """Return temporal z-score without np.nan_to_num temporary bool arrays.

    np.nan_to_num on a large 3-D array internally creates bool masks the same size as
    the movie. For 346 x 1303 x 1839, that single mask is about 791 MiB. The movie is
    already finite after compute_dff_movie(), so we avoid that extra allocation here.
    """
    movie = np.asarray(movie, dtype=np.float32)
    mean = movie.mean(axis=0, keepdims=True, dtype=np.float32)
    std = movie.std(axis=0, keepdims=True, dtype=np.float32)
    std += np.float32(1e-6)
    z = np.empty_like(movie, dtype=np.float32)
    np.subtract(movie, mean, out=z)
    np.divide(z, std, out=z)
    return z

def local_correlation_image(movie: np.ndarray) -> np.ndarray:
    """Average temporal correlation between each pixel and its 8-neighborhood."""
    movie = np.asarray(movie, dtype=np.float32)
    if movie.ndim != 3 or movie.shape[0] < 3:
        return np.zeros(movie.shape[-2:], dtype=np.float32)

    _, H, W = movie.shape
    z = zscore_movie(movie)
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for dy, dx in shifts:
        y0 = max(0, dy)
        y1 = H + min(0, dy)
        x0 = max(0, dx)
        x1 = W + min(0, dx)

        yy0 = max(0, -dy)
        yy1 = H - max(0, dy)
        xx0 = max(0, -dx)
        xx1 = W - max(0, dx)

        corr = np.mean(z[:, y0:y1, x0:x1] * z[:, yy0:yy1, xx0:xx1], axis=0)
        acc[y0:y1, x0:x1] += corr
        cnt[y0:y1, x0:x1] += 1.0

    corr_img = acc / np.maximum(cnt, 1.0)
    corr_img = np.clip(corr_img, 0.0, 1.0)
    return filters.gaussian(corr_img, sigma=1.0, preserve_range=True).astype(np.float32)


def temporal_skewness_image(movie: np.ndarray) -> np.ndarray:
    """Positive temporal skewness image computed frame-by-frame to reduce memory."""
    movie = np.asarray(movie, dtype=np.float32)
    if movie.ndim != 3 or movie.shape[0] < 3:
        return np.zeros(movie.shape[-2:], dtype=np.float32)
    mean = movie.mean(axis=0, dtype=np.float32)
    std = movie.std(axis=0, dtype=np.float32)
    std += np.float32(1e-6)
    acc = np.zeros(movie.shape[1:], dtype=np.float32)
    for frame in movie:
        z = (frame - mean) / std
        acc += z * z * z
    skew = acc / float(movie.shape[0])
    skew[skew < 0] = 0
    return skew.astype(np.float32, copy=False)

def build_neuron_score_image(processed_stack: np.ndarray, dff: np.ndarray, projection_percentile: float, sigma: float):
    projection = make_projection(processed_stack, percentile=float(projection_percentile))
    projection_n = percentile_normalize01(projection, 1.0, 99.5)

    if dff.shape[0] >= 3:
        corr = local_correlation_image(dff)
        transient = np.percentile(dff, 99.0, axis=0) - np.percentile(dff, 50.0, axis=0)
        temporal_std = np.std(dff, axis=0)
        skew = temporal_skewness_image(dff)
    else:
        corr = np.zeros_like(projection_n, dtype=np.float32)
        transient = projection_n.copy()
        temporal_std = projection_n.copy()
        skew = np.zeros_like(projection_n, dtype=np.float32)

    corr_n = percentile_normalize01(corr, 1.0, 99.0)
    transient_n = percentile_normalize01(transient, 1.0, 99.5)
    std_n = percentile_normalize01(temporal_std, 1.0, 99.5)
    skew_n = percentile_normalize01(skew, 1.0, 99.5)

    # Score map: spatial brightness still matters, but temporal consistency/transients dominate.
    score = 0.35 * corr_n + 0.30 * transient_n + 0.20 * projection_n + 0.10 * std_n + 0.05 * skew_n
    score = filters.gaussian(score, sigma=float(sigma), preserve_range=True)
    score = percentile_normalize01(score, 1.0, 99.7)
    return projection.astype(np.float32), score.astype(np.float32), corr.astype(np.float32), transient.astype(np.float32)


def _disk_boolean(shape, cy: int, cx: int, radius: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((yy - int(cy)) ** 2 + (xx - int(cx)) ** 2 <= int(radius) ** 2)


def trace_snr(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size < 3:
        return 0.0
    med = float(np.median(trace))
    mad = float(np.median(np.abs(trace - med))) + 1e-6
    noise = 1.4826 * mad
    signal = float(np.percentile(trace, 99.0) - np.percentile(trace, 50.0))
    return float(signal / max(noise, 1e-6))


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size < 3 or b.size != a.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) + 1e-6
    return float(np.sum(a * b) / denom)


def _clone_neuron_params(p: NeuronParams) -> NeuronParams:
    return NeuronParams(**p.__dict__)


def apply_detection_sensitivity(p: NeuronParams) -> NeuronParams:
    """Return an effective parameter copy for seed generation and ROI filtering.

    The review-panel spin boxes still work; the sensitivity preset only relaxes/strengthens
    the most important gates so users can switch between 'more candidates' and 'fewer false positives'.
    """
    q = _clone_neuron_params(p)
    mode = str(getattr(q, "detection_sensitivity", "Balanced") or "Balanced").lower()

    if mode.startswith("sens"):
        q.enable_multithreshold_seed = True
        q.enable_log_seed = True
        q.threshold_percentile = min(float(q.threshold_percentile), 97.0)
        q.min_distance = min(int(q.min_distance), 8)
        q.region_corr_threshold = min(float(q.region_corr_threshold), 0.22)
        q.min_roi_snr = min(float(q.min_roi_snr), 0.70)
        q.max_neuropil_corr = max(float(q.max_neuropil_corr), 0.92)
        q.min_roi_area_px = min(int(q.min_roi_area_px), 8)
        q.max_rois = max(int(q.max_rois), 300)
        q.log_threshold = min(float(q.log_threshold), 0.035)

    elif mode.startswith("cons"):
        q.threshold_percentile = max(float(q.threshold_percentile), 98.5)
        q.region_corr_threshold = max(float(q.region_corr_threshold), 0.32)
        q.min_roi_snr = max(float(q.min_roi_snr), 1.20)
        q.max_neuropil_corr = min(float(q.max_neuropil_corr), 0.82)
        q.min_roi_area_px = max(int(q.min_roi_area_px), 12)
        q.log_threshold = max(float(q.log_threshold), 0.060)

    else:  # Balanced
        q.threshold_percentile = min(max(float(q.threshold_percentile), 97.5), 98.2)
        q.region_corr_threshold = min(max(float(q.region_corr_threshold), 0.25), 0.30)
        q.min_roi_snr = min(max(float(q.min_roi_snr), 0.85), 1.00)
        q.max_neuropil_corr = min(max(float(q.max_neuropil_corr), 0.86), 0.90)
        q.min_roi_area_px = min(max(int(q.min_roi_area_px), 10), 14)
        q.log_threshold = min(max(float(q.log_threshold), 0.040), 0.055)

    return q


def seed_percentiles_for_params(p: NeuronParams):
    base = float(p.threshold_percentile)
    mode = str(getattr(p, "detection_sensitivity", "Balanced") or "Balanced").lower()

    if not bool(getattr(p, "enable_multithreshold_seed", True)):
        return [base]

    if mode.startswith("sens"):
        vals = [base, base - 0.8, base - 1.6, 95.5]
    elif mode.startswith("cons"):
        vals = [base, base + 0.5]
    else:
        vals = [base, base - 0.8, 96.8]

    vals = [float(np.clip(v, 80.0, 99.9)) for v in vals]
    out = []
    for v in vals:
        if not any(abs(v - u) < 0.05 for u in out):
            out.append(v)
    return out


def collect_multithreshold_seeds(img: np.ndarray, p: NeuronParams):
    """Find peak seeds at several score thresholds and keep source metadata."""
    img = percentile_normalize01(img, 1.0, 99.7)
    seeds = []

    for pct in seed_percentiles_for_params(p):
        thr = float(np.percentile(img, pct))
        coords = feature.peak_local_max(
            img,
            min_distance=max(1, int(p.min_distance)),
            threshold_abs=thr,
            exclude_border=False,
        )
        for y, x in coords:
            seeds.append({
                "y": int(y),
                "x": int(x),
                "score": float(img[int(y), int(x)]),
                "source": f"peak@{pct:.1f}",
                "sigma": 0.0,
            })

    return seeds


def collect_log_blob_seeds(img: np.ndarray, p: NeuronParams):
    """LoG spot detector for dim, round neuron-like candidates missed by hard peak thresholds."""
    if not bool(getattr(p, "enable_log_seed", True)):
        return []

    img = percentile_normalize01(img, 1.0, 99.7)
    min_sigma = max(0.8, float(getattr(p, "log_min_sigma", 1.5)))
    max_sigma = max(min_sigma + 0.1, float(getattr(p, "log_max_sigma", 6.0)))
    num_sigma = max(2, int(getattr(p, "log_num_sigma", 6)))
    threshold = max(0.001, float(getattr(p, "log_threshold", 0.035)))

    try:
        blobs = feature.blob_log(
            img,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=0.45,
        )
    except Exception:
        blobs = np.empty((0, 3), dtype=np.float32)

    seeds = []
    h, w = img.shape
    # LoG can return weak background blobs in stripe-like areas. Require a modest score floor,
    # relaxed in Sensitive mode but still above broad background.
    score_floor = float(np.percentile(img, 90.0))
    if str(getattr(p, "detection_sensitivity", "")).lower().startswith("sens"):
        score_floor = float(np.percentile(img, 84.0))

    for row in blobs:
        y, x, sigma = float(row[0]), float(row[1]), float(row[2])
        yi, xi = int(round(y)), int(round(x))
        if yi < 0 or yi >= h or xi < 0 or xi >= w:
            continue
        score = float(img[yi, xi])
        if score < score_floor:
            continue
        seeds.append({
            "y": yi,
            "x": xi,
            "score": score + 0.02,  # small bonus so true round spots are not lost after merge
            "source": "LoG",
            "sigma": float(sigma),
        })

    return seeds


def merge_seed_candidates(seeds, min_distance: int, max_candidates: int):
    """Sort candidate seeds by score/source priority and remove near-duplicates."""
    if not seeds:
        return np.empty((0, 2), dtype=np.int64), []

    # Prefer high score, but allow LoG seeds to survive because they often rescue dim cells.
    def seed_key(s):
        source_bonus = 0.035 if str(s.get("source", "")).startswith("LoG") else 0.0
        return float(s.get("score", 0.0)) + source_bonus

    ordered = sorted(seeds, key=seed_key, reverse=True)
    keep = []
    keep_meta = []
    dedup_radius = max(2.0, float(min_distance) * 0.65)
    dedup_r2 = dedup_radius * dedup_radius

    for s in ordered:
        y, x = int(s["y"]), int(s["x"])
        duplicate = False
        for yy, xx in keep:
            if (y - yy) ** 2 + (x - xx) ** 2 <= dedup_r2:
                duplicate = True
                break
        if duplicate:
            continue
        keep.append((y, x))
        keep_meta.append(dict(s))
        if len(keep) >= int(max_candidates):
            break

    return np.asarray(keep, dtype=np.int64), keep_meta

def detect_neuron_rois_from_score(
    score: np.ndarray,
    dff: np.ndarray,
    p: NeuronParams,
):
    """Detect ROI masks using Sensitive/Balanced/Conservative seeds + LoG rescue + temporal growing.

    v6 changes:
    - Multi-threshold peak seeds prevent a single high percentile from suppressing weak cells.
    - LoG blob seeds recover dim, round cell bodies that have moderate score-map intensity.
    - The merged candidate list is then filtered by temporal region growing, SNR, and neuropil correlation.
    """
    p = apply_detection_sensitivity(p)
    score = np.asarray(score, dtype=np.float32)
    dff = np.asarray(dff, dtype=np.float32)
    H, W = score.shape
    labels = np.zeros((H, W), dtype=np.int32)
    valid_centers = []

    img = percentile_normalize01(score, 1.0, 99.7)

    seed_candidates = []
    seed_candidates.extend(collect_multithreshold_seeds(img, p))
    seed_candidates.extend(collect_log_blob_seeds(img, p))

    max_candidates = max(int(p.max_rois) * 5, int(p.max_rois) + 50)
    coordinates, seed_meta = merge_seed_candidates(
        seed_candidates,
        min_distance=int(p.min_distance),
        max_candidates=max_candidates,
    )

    if len(coordinates) == 0:
        return labels, labels > 0, valid_centers

    # If there are too few frames for temporal correlation, fall back to circular ROIs.
    if dff.ndim != 3 or dff.shape[0] < 5:
        centers = [(int(y), int(x)) for y, x in coordinates[: int(p.max_rois)]]
        return make_circular_neuron_mask(score.shape, centers, radius_px=int(p.roi_radius_px))

    z = zscore_movie(dff)
    roi_id = 1
    radius = int(max(2, p.roi_radius_px))
    seed_radius = int(max(1, round(radius * 0.45)))
    grow_radius = int(max(radius * 2.8, p.min_distance * 1.8, 9))
    min_area = int(max(3, p.min_roi_area_px))
    max_area = int(max(min_area + 1, p.max_roi_area_px))
    max_rois = int(max(1, p.max_rois))
    mode = str(getattr(p, "detection_sensitivity", "Balanced") or "Balanced").lower()
    sensitive = mode.startswith("sens")

    local_score_percentile = 34.0 if sensitive else (42.0 if not mode.startswith("cons") else 50.0)
    seed_score_factor = 0.18 if sensitive else (0.25 if not mode.startswith("cons") else 0.35)
    fallback_score_factor = 0.25 if sensitive else (0.35 if not mode.startswith("cons") else 0.45)

    for seed_idx, (cy, cx) in enumerate(coordinates):
        if roi_id > max_rois:
            break
        cy = int(cy)
        cx = int(cx)
        if labels[cy, cx] > 0:
            continue

        seed_source = seed_meta[seed_idx].get("source", "") if seed_idx < len(seed_meta) else ""

        seed_mask = _disk_boolean((H, W), cy, cx, seed_radius) & (labels == 0)
        if np.count_nonzero(seed_mask) < 2:
            seed_mask = np.zeros((H, W), dtype=bool)
            seed_mask[cy, cx] = True

        seed_trace = dff[:, seed_mask].mean(axis=1)
        seed_trace_z = seed_trace - seed_trace.mean()
        seed_std = float(seed_trace_z.std())
        if seed_std < 1e-6:
            continue
        seed_trace_z = seed_trace_z / (seed_std + 1e-6)

        y0 = max(0, cy - grow_radius)
        y1 = min(H, cy + grow_radius + 1)
        x0 = max(0, cx - grow_radius)
        x1 = min(W, cx + grow_radius + 1)

        z_win = z[:, y0:y1, x0:x1]
        corr_win = np.mean(z_win * seed_trace_z[:, None, None], axis=0)

        local_score = img[y0:y1, x0:x1]
        local_unassigned = labels[y0:y1, x0:x1] == 0
        local_disk = _disk_boolean((y1 - y0, x1 - x0), cy - y0, cx - x0, grow_radius)
        local_threshold = max(
            float(np.percentile(local_score, local_score_percentile)),
            float(img[cy, cx]) * seed_score_factor,
        )

        corr_threshold = float(p.region_corr_threshold)
        # LoG seeds are often dim cells; relax correlation slightly so the candidate can grow.
        if str(seed_source).startswith("LoG"):
            corr_threshold = max(0.10, corr_threshold - (0.05 if sensitive else 0.03))

        candidate_local = (
            (corr_win >= corr_threshold)
            & (local_score >= local_threshold)
            & local_unassigned
            & local_disk
        )

        if np.count_nonzero(candidate_local) < min_area:
            # Compact rescue fallback. This is intentionally more permissive in Sensitive mode,
            # because false positives can be removed in the manual ROI popup.
            fallback = _disk_boolean((y1 - y0, x1 - x0), cy - y0, cx - x0, radius)
            candidate_local = (
                fallback
                & (local_score >= max(float(np.percentile(local_score, 40.0 if sensitive else 50.0)),
                                      float(img[cy, cx]) * fallback_score_factor))
                & local_unassigned
            )

        candidate_local = morphology.opening(candidate_local, morphology.disk(1))
        candidate_local = morphology.closing(candidate_local, morphology.disk(1))

        cc_pre = measure.label(candidate_local.astype(np.uint8), connectivity=2)
        cleaned = np.zeros_like(candidate_local, dtype=bool)
        for prop in measure.regionprops(cc_pre):
            if prop.area >= min_area:
                cleaned[cc_pre == prop.label] = True
        candidate_local = cleaned

        if np.count_nonzero(candidate_local) < min_area:
            continue

        cc = measure.label(candidate_local.astype(np.uint8), connectivity=2)
        seed_label = cc[cy - y0, cx - x0]
        if seed_label == 0:
            props = measure.regionprops(cc)
            if not props:
                continue
            props = sorted(props, key=lambda r: (r.centroid[0] - (cy - y0)) ** 2 + (r.centroid[1] - (cx - x0)) ** 2)
            seed_label = props[0].label

        roi_local = cc == seed_label
        area = int(np.count_nonzero(roi_local))
        if area < min_area or area > max_area:
            continue

        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[y0:y1, x0:x1] = roi_local
        roi_trace = dff[:, roi_mask].mean(axis=1)
        roi_snr = trace_snr(roi_trace)

        # Sensitive mode keeps weaker ROIs so the user can review/delete them manually.
        min_snr = float(p.min_roi_snr)
        if sensitive and str(seed_source).startswith("LoG"):
            min_snr = min(min_snr, 0.55)

        if roi_snr < min_snr:
            continue

        ring_outer = _disk_boolean((H, W), cy, cx, int(max(radius * 3, grow_radius)))
        ring_inner = _disk_boolean((H, W), cy, cx, int(max(radius * 1.4, radius + 2)))
        ring = ring_outer & (~ring_inner) & (labels == 0)
        if np.count_nonzero(ring) >= min_area:
            neuropil_trace = dff[:, ring].mean(axis=1)
            neuropil_corr = safe_corrcoef(roi_trace, neuropil_trace)
            # Sensitive mode only rejects if neuropil correlation is high AND the ROI is very weak.
            reject_factor = 1.4 if sensitive else 2.0
            if neuropil_corr > float(p.max_neuropil_corr) and roi_snr < min_snr * reject_factor:
                continue

        labels[roi_mask] = roi_id
        coords = np.argwhere(roi_mask)
        centroid_yx = coords.mean(axis=0)
        valid_centers.append((roi_id, int(round(float(centroid_yx[0]))), int(round(float(centroid_yx[1])))))
        roi_id += 1

    binary_mask = labels > 0
    return labels, binary_mask, valid_centers


def enhance_neuron_projection(projection: np.ndarray, sigma: float) -> np.ndarray:
    img_norm = percentile_normalize01(projection, 1.0, 99.5)
    img_smooth = filters.gaussian(img_norm, sigma=float(sigma), preserve_range=True)
    return percentile_normalize01(img_smooth, 1.0, 99.7).astype(np.float32)


def detect_neuron_centers_from_enhanced(
    enhanced: np.ndarray,
    min_distance: int = 10,
    threshold_percentile: float = 97.5,
    max_rois: int = 200,
):
    img = percentile_normalize01(enhanced, 1.0, 99.7)
    threshold = np.percentile(img, float(threshold_percentile))

    coordinates = feature.peak_local_max(
        img,
        min_distance=int(min_distance),
        threshold_abs=threshold,
        exclude_border=False,
    )

    if len(coordinates) > 0:
        intensities = img[coordinates[:, 0], coordinates[:, 1]]
        order = np.argsort(intensities)[::-1]
        coordinates = coordinates[order]

    if max_rois is not None and len(coordinates) > int(max_rois):
        coordinates = coordinates[:int(max_rois)]

    return [(int(y), int(x)) for y, x in coordinates]


def detect_neuron_centers_from_projection(
    projection: np.ndarray,
    sigma: float = 1.2,
    min_distance: int = 10,
    threshold_percentile: float = 97.5,
    max_rois: int = 200,
):
    enhanced = enhance_neuron_projection(projection, sigma=float(sigma))
    centers = detect_neuron_centers_from_enhanced(
        enhanced,
        min_distance=int(min_distance),
        threshold_percentile=float(threshold_percentile),
        max_rois=int(max_rois),
    )
    return centers, enhanced


def make_circular_neuron_mask(image_shape, centers_yx, radius_px: int = 6):
    h, w = image_shape
    yy, xx = np.ogrid[:h, :w]
    labels = np.zeros((h, w), dtype=np.int32)
    valid_centers = []
    roi_id = 1

    for cy, cx in centers_yx:
        if cy < 0 or cy >= h or cx < 0 or cx >= w:
            continue
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= int(radius_px) ** 2
        circle = circle & (labels == 0)
        if np.count_nonzero(circle) == 0:
            continue
        labels[circle] = roi_id
        valid_centers.append((roi_id, int(cy), int(cx)))
        roi_id += 1

    binary_mask = labels > 0
    return labels, binary_mask, valid_centers


def rebuild_neuron_mask_from_centers(result: dict, centers_yx, radius_px: int) -> dict:
    updated = dict(result)
    projection = updated["projection"]
    labels, mask, valid_centers = make_circular_neuron_mask(
        image_shape=projection.shape,
        centers_yx=centers_yx,
        radius_px=int(radius_px),
    )
    updated["labels"] = labels
    updated["mask"] = mask
    updated["mask_projector"] = resize_mask_to_projector(mask)
    updated["centers"] = valid_centers
    return updated


def run_neuron_pipeline(stack_gray_u8: np.ndarray, p: NeuronParams) -> dict:
    stack_gray_u8 = np.asarray(stack_gray_u8)
    if stack_gray_u8.ndim != 3:
        raise ValueError(f"Neuron stack must be T x Y x X, got {stack_gray_u8.shape}")

    original_stack_shape = tuple(int(v) for v in stack_gray_u8.shape)
    # Defensive guard: callers should already do this, but keep the pipeline safe if called directly.
    stack_gray_u8 = prepare_neuron_stack_for_processing(
        stack_gray_u8,
        max_frames=int(p.max_processing_frames),
        max_side_px=int(p.max_processing_side_px),
    )
    working_stack_shape = tuple(int(v) for v in stack_gray_u8.shape)

    processed_stack = stack_gray_u8.astype(np.float32, copy=False)

    motion_shifts = None
    if p.do_motion_correction:
        processed_stack, motion_shifts = apply_rigid_motion_correction(
            processed_stack,
            max_shift_px=int(p.max_motion_shift_px),
        )

    spatial_background = None
    if p.do_spatial_background:
        processed_stack, spatial_background = subtract_spatial_background(
            processed_stack,
            sigma=float(p.spatial_background_sigma),
        )

    temporal_background = None
    if p.do_temporal_min_background:
        processed_stack, temporal_background = subtract_temporal_min_background(processed_stack)

    dff = compute_dff_movie(processed_stack)
    # Use a smaller temporal sample for correlation/region-growing. It preserves activity
    # diversity while avoiding a second full-size z-score movie during detection.
    dff_for_roi = sample_stack_time(dff, int(p.max_score_frames)).astype(np.float32, copy=False)

    projection, score, corr_img, transient_img = build_neuron_score_image(
        processed_stack,
        dff_for_roi,
        projection_percentile=float(p.projection_percentile),
        sigma=float(p.gaussian_sigma),
    )

    # processed_stack is no longer needed after projection/score construction; freeing the
    # reference helps Windows reclaim memory before z-scoring in detect_neuron_rois_from_score().
    del processed_stack
    if dff is not dff_for_roi:
        del dff

    labels, mask, valid_centers = detect_neuron_rois_from_score(score, dff_for_roi, p)

    background = temporal_background
    if background is None:
        background = spatial_background

    return {
        "original_stack_shape": original_stack_shape,
        "stack_shape": working_stack_shape,
        "projection": projection,
        "enhanced": score,
        "score": score,
        "local_correlation": corr_img,
        "transient": transient_img,
        "background": background,
        "spatial_background": spatial_background,
        "temporal_background": temporal_background,
        "motion_shifts": motion_shifts,
        "dff": dff_for_roi.astype(np.float32, copy=False),
        "labels": labels,
        "mask": mask,
        "mask_projector": resize_mask_to_projector(mask),
        "centers": valid_centers,
    }

def add_neuron_centers_to_rgb(img_rgb: np.ndarray, centers, radius_px: int = 6):
    out = img_rgb.copy()
    for roi_id, cy, cx in centers:
        cv2.circle(
            out,
            center=(int(cx), int(cy)),
            radius=int(radius_px),
            color=(255, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(roi_id),
            (int(cx) + 3, int(cy) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def make_neuron_mosaic(result: dict, roi_radius_px: int) -> np.ndarray:
    projection_rgb = add_panel_title(gray_to_rgb_u8(result["projection"]), "Processed Projection")
    score_rgb = add_panel_title(gray_to_rgb_u8(result["enhanced"]), "Score Map + Centers")
    score_rgb = add_neuron_centers_to_rgb(score_rgb, result["centers"], radius_px=roi_radius_px)
    mask_rgb = add_panel_title(mask_to_rgb_u8(result["mask"]), "Region-Grown Mask")
    h = projection_rgb.shape[0]
    separator = np.full((h, 6, 3), 40, dtype=np.uint8)
    return np.hstack([projection_rgb, separator, score_rgb, separator, mask_rgb])


class NeuronProcessingThread(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, stack_gray_u8: np.ndarray, params: NeuronParams, parent=None):
        super().__init__(parent)
        # Keep a reference instead of copying; _start_processing_stack already owns a
        # compact working stack. Copying here can add hundreds of MB for large Test files.
        self.stack_gray_u8 = stack_gray_u8
        self.params = params

    def run(self):
        try:
            result = run_neuron_pipeline(self.stack_gray_u8, self.params)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class NeuronManualRoiDialog(QDialog):
    centersChanged = Signal(object)

    def __init__(self, stack_gray_u8: np.ndarray, centers_yx, radius_px: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Neuron ROI Review")
        self.resize(980, 760)
        self.setMinimumSize(720, 520)

        self.stack = _to_uint8_gray_stack(stack_gray_u8)
        self.centers_yx = [(int(y), int(x)) for y, x in centers_yx]
        self.radius_px = int(radius_px)
        self.frame_index = 0

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(150)
        self.play_timer.timeout.connect(self.next_frame)

        self.label = ReviewImageLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(520, 390)
        self.label.setStyleSheet("background: #202020; color: #dddddd; border: 1px solid #555;")
        self.label.imageClicked.connect(self.on_image_clicked)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.label)

        self.prev_btn = QPushButton("Previous Frame")
        self.prev_btn.clicked.connect(self.previous_frame)
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self.toggle_playback)
        self.next_btn = QPushButton("Next Frame")
        self.next_btn.clicked.connect(self.next_frame)
        self.frame_label = QLabel()

        frame_row = QHBoxLayout()
        frame_row.addWidget(self.prev_btn)
        frame_row.addWidget(self.play_btn)
        frame_row.addWidget(self.next_btn)
        frame_row.addWidget(self.frame_label, stretch=1)

        self.black_spin = QSpinBox()
        self.black_spin.setRange(0, 254)
        self.black_spin.setValue(0)
        self.black_spin.valueChanged.connect(self.on_contrast_changed)
        self.white_spin = QSpinBox()
        self.white_spin.setRange(1, 255)
        self.white_spin.setValue(255)
        self.white_spin.valueChanged.connect(self.on_contrast_changed)
        self.auto_contrast_btn = QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self.auto_contrast_current_frame)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Black"))
        contrast_row.addWidget(self.black_spin)
        contrast_row.addWidget(QLabel("White"))
        contrast_row.addWidget(self.white_spin)
        contrast_row.addWidget(self.auto_contrast_btn)
        contrast_row.addStretch(1)

        help_label = QLabel("Left-click to add ROI; right-click to delete nearest ROI. Use frame controls and contrast to inspect dim or transient neurons before confirming ROI coverage.")
        help_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area, stretch=1)
        layout.addLayout(frame_row)
        layout.addLayout(contrast_row)
        layout.addWidget(help_label)

        self._qimage = None
        self._pixmap = None
        self.auto_contrast_current_frame()

    def set_stack(self, stack_gray_u8: np.ndarray):
        self.stack = _to_uint8_gray_stack(stack_gray_u8)
        self.frame_index = max(0, min(self.frame_index, self.stack.shape[0] - 1))
        self.render_frame()

    def set_centers_and_radius(self, centers_yx, radius_px: int):
        self.centers_yx = [(int(y), int(x)) for y, x in centers_yx]
        self.radius_px = int(radius_px)
        self.render_frame()

    def current_frame(self) -> np.ndarray:
        return self.stack[int(self.frame_index)]

    def on_contrast_changed(self, _value=None):
        if self.white_spin.value() <= self.black_spin.value():
            self.white_spin.blockSignals(True)
            self.white_spin.setValue(self.black_spin.value() + 1)
            self.white_spin.blockSignals(False)
        self.render_frame()

    def auto_contrast_current_frame(self):
        frame = self.current_frame().astype(np.float32)
        lo, hi = np.percentile(frame, [1.0, 99.5])
        lo = int(max(0, min(254, round(lo))))
        hi = int(max(lo + 1, min(255, round(hi))))
        self.black_spin.blockSignals(True)
        self.white_spin.blockSignals(True)
        self.black_spin.setValue(lo)
        self.white_spin.setValue(hi)
        self.black_spin.blockSignals(False)
        self.white_spin.blockSignals(False)
        self.render_frame()

    def toggle_playback(self, checked: bool):
        if checked:
            self.play_btn.setText("Pause")
            self.play_timer.start()
        else:
            self.play_timer.stop()
            self.play_btn.setText("Play")

    def previous_frame(self):
        self.frame_index = (int(self.frame_index) - 1) % max(1, self.stack.shape[0])
        self.render_frame()

    def next_frame(self):
        self.frame_index = (int(self.frame_index) + 1) % max(1, self.stack.shape[0])
        self.render_frame()

    def render_frame(self):
        frame = self.current_frame().astype(np.float32)
        black = int(self.black_spin.value())
        white = int(self.white_spin.value())
        view = np.clip((frame - black) * 255.0 / max(1, white - black), 0, 255).astype(np.uint8)
        rgb = np.dstack([view, view, view])

        for idx, (cy, cx) in enumerate(self.centers_yx, start=1):
            cv2.circle(rgb, (int(cx), int(cy)), int(self.radius_px), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb, str(idx), (int(cx) + 3, int(cy) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(
            rgb,
            f"Frame {int(self.frame_index) + 1}/{self.stack.shape[0]}  ROIs: {len(self.centers_yx)}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self._qimage = rgb_to_qimage(rgb)
        pixmap = QPixmap.fromImage(self._qimage)
        self._pixmap = pixmap
        self.label.set_source_size(pixmap.size())
        self.update_pixmap()
        self.frame_label.setText(f"Frame {int(self.frame_index) + 1}/{self.stack.shape[0]}    ROIs: {len(self.centers_yx)}")

    def update_pixmap(self):
        if self._pixmap is None:
            return
        target_size = self.label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.scroll_area.viewport().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)
        x = max(0, (self.label.width() - scaled.width()) // 2)
        y = max(0, (self.label.height() - scaled.height()) // 2)
        self.label.set_display_rect(QRect(x, y, scaled.width(), scaled.height()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update_pixmap)

    def on_image_clicked(self, x: int, y: int, button: int):
        cy, cx = int(y), int(x)
        if button == int(Qt.RightButton.value):
            if self.centers_yx:
                d2 = [(cy - ey) ** 2 + (cx - ex) ** 2 for ey, ex in self.centers_yx]
                idx = int(np.argmin(d2))
                if d2[idx] <= max(25, (int(self.radius_px) * 3) ** 2):
                    self.centers_yx.pop(idx)
        else:
            self.centers_yx.append((cy, cx))

        self.render_frame()
        self.centersChanged.emit(list(self.centers_yx))

    def closeEvent(self, event):
        self.play_timer.stop()
        self.play_btn.setChecked(False)
        super().closeEvent(event)


class NeuronMode(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.active = False
        self.stack_frames = []
        self.frame_index = 0
        self.processing_thread = None
        self.result_dialog = None
        self.output_dir = None
        self.pending_stack = None
        self.last_stack_source = None
        self.last_result = None
        self.last_mosaic = None
        self.current_params = NeuronParams(**DEFAULT_NEURON_PARAMS.__dict__)
        self._live_update_busy = False
        self.manual_roi_mode = False
        self.manual_roi_dialog = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        acquisition_group = QGroupBox("Neuron Mask Acquisition")
        form = QFormLayout(acquisition_group)

        self.frame_count_spin = QSpinBox()
        self.frame_count_spin.setRange(1, 5000)
        self.frame_count_spin.setValue(self.current_params.frame_count)

        self.capture_interval_spin = QSpinBox()
        self.capture_interval_spin.setRange(0, 5000)
        self.capture_interval_spin.setValue(self.current_params.capture_interval_ms)
        self.capture_interval_spin.setSuffix(" ms")

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.01, 1000.0)
        self.frame_rate_spin.setDecimals(3)
        self.frame_rate_spin.setValue(self.current_params.frame_rate_hz)

        self.test_mode_checkbox = QCheckBox("Test mode (use local stack/image)")
        self.test_mode_checkbox.setChecked(False)
        self.test_mode_checkbox.toggled.connect(self._update_test_image_controls)

        self.test_image_edit = QLineEdit()
        self.test_image_edit.setPlaceholderText("Local neuron TIFF stack / image for Test Mode")

        self.test_browse_btn = QPushButton("Browse")
        self.test_browse_btn.clicked.connect(self.choose_test_image)

        test_image_row = QHBoxLayout()
        test_image_row.addWidget(self.test_image_edit)
        test_image_row.addWidget(self.test_browse_btn)

        form.addRow("Frame count", self.frame_count_spin)
        form.addRow("Capture interval", self.capture_interval_spin)
        form.addRow("Saved TIFF frame rate", self.frame_rate_spin)
        form.addRow("Test mode", self.test_mode_checkbox)
        form.addRow("Test image", test_image_row)
        layout.addWidget(acquisition_group)
        self._update_test_image_controls()

        hint = QLabel("Neuron detection and Mask parameters are edited in the Neuron Mask Result / Parameters window after acquisition.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Calculate Mask")
        self.run_btn.clicked.connect(self.start)
        self.show_mask_btn = QPushButton("Show Mask")
        self.show_mask_btn.setEnabled(False)
        self.show_mask_btn.clicked.connect(self.show_last_mask)

        self.manual_roi_btn = QPushButton("Manual ROI Mark")
        self.manual_roi_btn.setEnabled(False)
        self.manual_roi_btn.clicked.connect(self.open_manual_neuron_roi_dialog)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.show_mask_btn)
        btn_row.addWidget(self.manual_roi_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def is_test_mode_enabled(self):
        return bool(self.test_mode_checkbox.isChecked())

    def set_test_mode_enabled(self, enabled: bool):
        self.test_mode_checkbox.setChecked(bool(enabled))
        self._update_test_image_controls()

    def get_test_image_path(self):
        return self.test_image_edit.text().strip()

    def set_test_image_path(self, path):
        self.test_image_edit.setText(str(path or ""))

    def choose_test_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Local Neuron Stack / Image",
            "",
            "Data/Image Files (*.tif *.tiff *.npy *.npz *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if path:
            self.test_image_edit.setText(path)
            self.app.set_status(f"Selected test neuron image/stack: {path}")

    def _update_test_image_controls(self, _checked=None):
        enabled = not self.active if hasattr(self, "active") else True
        use_test = bool(self.test_mode_checkbox.isChecked())
        self.test_image_edit.setEnabled(enabled and use_test)
        self.test_browse_btn.setEnabled(enabled and use_test)
        self.frame_count_spin.setEnabled(enabled and not use_test)
        self.capture_interval_spin.setEnabled(enabled and not use_test)

    def set_controls_enabled(self, enabled: bool):
        widgets = [self.run_btn, self.frame_count_spin, self.capture_interval_spin, self.frame_rate_spin, self.test_mode_checkbox]
        for w in widgets:
            w.setEnabled(enabled)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(bool(enabled) and self.last_result is not None)
        self._update_test_image_controls()

    def collect_params(self) -> NeuronParams:
        p = NeuronParams(**self.current_params.__dict__)
        p.frame_count = int(self.frame_count_spin.value())
        p.capture_interval_ms = int(self.capture_interval_spin.value())
        p.frame_rate_hz = float(self.frame_rate_spin.value())
        return p

    def _start_processing_stack(self, stack: np.ndarray, params: NeuronParams, status_prefix: str):
        stack_u8 = _to_uint8_gray_stack(stack)
        raw_shape = tuple(int(v) for v in stack_u8.shape)
        stack = prepare_neuron_stack_for_processing(
            stack_u8,
            max_frames=int(params.max_processing_frames),
            max_side_px=int(params.max_processing_side_px),
        )
        self.stack_frames = []
        self.pending_stack = stack
        self.frame_index = int(stack.shape[0])
        self.active = True
        self.set_controls_enabled(False)
        self.show_mask_btn.setEnabled(False)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(False)
        self.app.set_status(
            f"{status_prefix}: raw stack {raw_shape} -> working stack {stack.shape}. "
            "Processing neuron mask..."
        )
        self.processing_thread = NeuronProcessingThread(self.pending_stack, params, self)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_failed)
        self.processing_thread.start()

    def _load_neuron_test_stack_for_calculate(self) -> np.ndarray:
        test_image_path = self.get_test_image_path()
        if not test_image_path:
            raise ValueError("Please choose a local neuron Test image/stack before Calculate Mask.")
        if not os.path.exists(test_image_path):
            raise ValueError("The selected local neuron Test image/stack does not exist.")
        stack = load_neuron_test_stack(test_image_path)
        if stack.ndim != 3:
            raise ValueError(f"Local neuron Test image/stack was not parsed as T x Y x X: {stack.shape}")
        self.last_stack_source = str(test_image_path)
        return stack

    def start(self):
        if self.active:
            return

        params = self.collect_params()

        if self.is_test_mode_enabled():
            try:
                stack = self._load_neuron_test_stack_for_calculate()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Neuron Test Image", str(exc))
                return

            self.output_dir = self.app.resolve_output_base_dir() / time.strftime("neuron_test_%Y%m%d_%H%M%S")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._start_processing_stack(
                stack,
                params,
                f"Neuron Test mode Calculate Mask using local file: {self.last_stack_source}",
            )
            return

        if params.frame_count <= 0:
            QMessageBox.warning(self, "Neuron Error", "Frame count must be greater than 0.")
            return

        self.last_stack_source = "camera"
        self.output_dir = self.app.resolve_output_base_dir() / time.strftime("neuron_%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.app.is_camera_running():
            QMessageBox.warning(self, "Camera Error", "Please open the camera first.")
            return
        if not self.app.has_latest_camera_frame():
            QMessageBox.warning(self, "Camera Error", "No camera frame has been received yet.")
            return

        self.stack_frames = []
        self.frame_index = 0
        self.active = True
        self.set_controls_enabled(False)
        self.show_mask_btn.setEnabled(False)
        self.app.set_status(f"Neuron mask acquisition started: 0/{params.frame_count}")
        self._capture_next_frame()

    def _capture_next_frame(self):
        if not self.active:
            return

        params = self.collect_params()
        frame = self.app.get_latest_frame_bgr(full=True)
        if frame is None:
            self._abort("Camera frame is empty during neuron acquisition.")
            return

        frame_gray = camera_frame_to_gray_u8(frame)
        self.stack_frames.append(frame_gray)
        self.frame_index += 1
        self.app.set_status(f"Neuron mask acquisition: {self.frame_index}/{params.frame_count}")

        if self.frame_index >= params.frame_count:
            self._finish_acquisition_and_save_raw()
            return

        QTimer.singleShot(int(params.capture_interval_ms), self._capture_next_frame)

    def _finish_acquisition_and_save_raw(self):
        if not self.active:
            return

        try:
            stack = np.stack(self.stack_frames, axis=0).astype(np.uint8)
        except Exception as e:
            self._abort(str(e))
            return

        self.pending_stack = stack
        self.last_stack_source = "camera"
        params = self.collect_params()
        self.app.set_status(f"Neuron stack acquired: {stack.shape}. Saving uncompressed TIFF...")

        raw_tiff_path = self.output_dir / "neuron_raw_stack_uncompressed.tif"
        self.app.start_save_worker(
            [
                {
                    "kind": "tiff",
                    "path": raw_tiff_path,
                    "data": stack,
                    "kwargs": {
                        "imagej": True,
                        "metadata": {
                            "axes": "TYX",
                            "fps": float(params.frame_rate_hz),
                            "finterval": float(1.0 / float(params.frame_rate_hz)),
                        },
                        "photometric": "minisblack",
                        "compression": None,
                    },
                }
            ],
            label="Neuron raw TIFF save",
            on_finished=self._start_processing_after_raw_save,
        )

    def _start_processing_after_raw_save(self):
        if not self.active or self.pending_stack is None:
            return

        params = self.collect_params()
        self._start_processing_stack(self.pending_stack, params, "Neuron raw TIFF saved")

    def _ensure_output_dir(self):
        if self.output_dir is None:
            self.output_dir = self.app.resolve_output_base_dir() / time.strftime("neuron_%Y%m%d_%H%M%S")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _make_save_tasks(self, result: dict, mosaic: np.ndarray):
        out = self._ensure_output_dir()
        mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        projector_mask_u8 = result["mask_projector"].astype(np.uint8) * 255
        tasks = [
            {
                "kind": "tiff",
                "path": out / "neuron_projection.tif",
                "data": result["projection"].astype(np.float32),
                "kwargs": {"photometric": "minisblack", "compression": None},
            },
            {
                "kind": "tiff",
                "path": out / "neuron_score_map.tif",
                "data": result["enhanced"].astype(np.float32),
                "kwargs": {"photometric": "minisblack", "compression": None},
            },
            {
                "kind": "tiff",
                "path": out / "neuron_labels.tif",
                "data": result["labels"].astype(np.int32),
                "kwargs": {"photometric": "minisblack", "compression": None},
            },
            {"kind": "image", "path": out / "neuron_mask_projector_640x480.png", "data": projector_mask_u8},
            {"kind": "image", "path": out / "neuron_compare.png", "data": mosaic_bgr},
            {
                "kind": "csv_rows",
                "path": out / "neuron_centers_coordinates.csv",
                "header": ["roi_id", "center_y", "center_x"],
                "rows": result["centers"],
            },
        ]

        if result.get("local_correlation") is not None:
            tasks.append(
                {
                    "kind": "tiff",
                    "path": out / "neuron_local_correlation.tif",
                    "data": result["local_correlation"].astype(np.float32),
                    "kwargs": {"photometric": "minisblack", "compression": None},
                }
            )

        if result.get("transient") is not None:
            tasks.append(
                {
                    "kind": "tiff",
                    "path": out / "neuron_transient_map.tif",
                    "data": result["transient"].astype(np.float32),
                    "kwargs": {"photometric": "minisblack", "compression": None},
                }
            )

        if result.get("motion_shifts") is not None:
            motion_rows = [
                [i, float(dx), float(dy), float(resp)]
                for i, (dx, dy, resp) in enumerate(np.asarray(result["motion_shifts"]), start=1)
            ]
            tasks.append(
                {
                    "kind": "csv_rows",
                    "path": out / "neuron_motion_shifts.csv",
                    "header": ["frame", "dx", "dy", "response"],
                    "rows": motion_rows,
                }
            )

        if self.pending_stack is not None and self.last_stack_source not in (None, "camera"):
            tasks.append(
                {
                    "kind": "tiff",
                    "path": out / "neuron_test_input_stack_uncompressed.tif",
                    "data": np.asarray(self.pending_stack).astype(np.uint8),
                    "kwargs": {
                        "imagej": True,
                        "metadata": {
                            "axes": "TYX",
                            "source_path": str(self.last_stack_source),
                        },
                        "photometric": "minisblack",
                        "compression": None,
                    },
                }
            )

        if result.get("background") is not None:
            tasks.append(
                {
                    "kind": "tiff",
                    "path": out / "neuron_temporal_min_background.tif",
                    "data": result["background"].astype(np.float32),
                    "kwargs": {"photometric": "minisblack", "compression": None},
                }
            )
        return tasks

    def save_last_result(self, show_message: bool = False):
        if self.last_result is None or self.last_mosaic is None:
            QMessageBox.information(self, "Save Mask", "No neuron mask has been calculated yet.")
            return
        self.apply_review_params_to_main_if_available()
        out = self._ensure_output_dir()
        self.app.start_save_worker(
            self._make_save_tasks(self.last_result, self.last_mosaic),
            label="Neuron result save",
        )
        self.app.set_status(f"Neuron mask saved to: {out}")
        if show_message:
            QMessageBox.information(self, "Save Mask", f"Neuron mask and comparison images were queued for saving to:\n{out}")

    def on_processing_finished(self, result: dict):
        self.active = False
        self.set_controls_enabled(True)

        self.last_result = result
        self.last_mosaic = make_neuron_mosaic(result, roi_radius_px=int(self.current_params.roi_radius_px))
        self.show_mask_btn.setEnabled(True)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(True)

        self.save_last_result(show_message=False)
        self.show_result_dialog()

        n_cells = len(result["centers"])
        self.app.set_status(
            f"Neuron mask calculated and saved. Cells: {n_cells}. "
            f"Stack: {result.get('original_stack_shape', result.get('stack_shape'))} -> {result.get('stack_shape')}. "
            "Use 'Show Mask' to project it."
        )

    def show_last_mask(self):
        if self.last_result is None:
            QMessageBox.information(self, "Show Mask", "Please calculate a neuron mask first.")
            return
        mask_projector_rgb = mask_to_rgb_u8(self.last_result["mask_projector"])
        if not self.app.show_projection_pattern(mask_projector_rgb, warn=True):
            return
        self.app.set_status(f"Neuron mask shown on projection. Cells: {len(self.last_result['centers'])}.")

    def show_result_dialog(self):
        if isinstance(self.result_dialog, MaskReviewDialog) and self.result_dialog.isVisible():
            self.result_dialog.set_image(self.last_mosaic)
            self._sync_manual_roi_dialog()
            return

        panel = self._build_review_params_panel()
        self.result_dialog = MaskReviewDialog(
            self.last_mosaic,
            "Neuron Mask Result / Parameters",
            params_widget=panel,
            parent=self,
        )
        self.result_dialog.recalculate_btn.clicked.connect(self.recalculate_from_review_dialog)
        self.result_dialog.save_btn.clicked.connect(lambda: self.save_last_result(show_message=True))
        self.result_dialog.imageClicked.connect(self.on_review_image_clicked)
        self.result_dialog.show()
        if self.manual_roi_mode:
            self.open_manual_neuron_roi_dialog()

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
                self.update_neuron_mask_live_from_review()

        def slider_to_spin(value):
            spin.blockSignals(True)
            if isinstance(spin, QDoubleSpinBox):
                spin.setValue(float(value) / float(scale))
            else:
                spin.setValue(int(round(float(value) / float(scale))))
            spin.blockSignals(False)
            if live_update:
                self.update_neuron_mask_live_from_review()

        spin.valueChanged.connect(spin_to_slider)
        slider.valueChanged.connect(slider_to_spin)
        return slider

    def _build_review_params_panel(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        p = self.current_params

        enhance_group = QGroupBox("Enhancement Parameters")
        enhance_form = QFormLayout(enhance_group)

        self.review_motion_check = QCheckBox("Rigid motion correction")
        self.review_motion_check.setChecked(bool(p.do_motion_correction))

        self.review_max_motion_shift_spin = QSpinBox()
        self.review_max_motion_shift_spin.setRange(0, 200)
        self.review_max_motion_shift_spin.setValue(int(p.max_motion_shift_px))

        self.review_spatial_bg_check = QCheckBox("Spatial background / neuropil haze subtraction")
        self.review_spatial_bg_check.setChecked(bool(p.do_spatial_background))

        self.review_spatial_bg_sigma_spin = self._new_review_double_spin(p.spatial_background_sigma, 1.0, 200.0, 1, 1.0)

        self.review_max_processing_frames_spin = QSpinBox()
        self.review_max_processing_frames_spin.setRange(1, 5000)
        self.review_max_processing_frames_spin.setValue(int(p.max_processing_frames))

        self.review_max_processing_side_spin = QSpinBox()
        self.review_max_processing_side_spin.setRange(128, 4096)
        self.review_max_processing_side_spin.setValue(int(p.max_processing_side_px))

        self.review_max_score_frames_spin = QSpinBox()
        self.review_max_score_frames_spin.setRange(10, 2000)
        self.review_max_score_frames_spin.setValue(int(p.max_score_frames))

        self.review_bg_check = QCheckBox("Temporal-min background subtraction")
        self.review_bg_check.setChecked(bool(p.do_temporal_min_background))

        self.review_projection_pct_spin = self._new_review_double_spin(p.projection_percentile, 1.0, 100.0, 1, 0.5)
        self.review_gaussian_sigma_spin = self._new_review_double_spin(p.gaussian_sigma, 0.1, 20.0, 2, 0.1)

        enhance_form.addRow("Motion correction", self.review_motion_check)
        enhance_form.addRow("Max motion shift px", self.review_max_motion_shift_spin)
        enhance_form.addRow("Spatial background", self.review_spatial_bg_check)
        enhance_form.addRow("Background sigma px", self.review_spatial_bg_sigma_spin)
        enhance_form.addRow("Max processing frames", self.review_max_processing_frames_spin)
        enhance_form.addRow("Max processing side px", self.review_max_processing_side_spin)
        enhance_form.addRow("Max score/correlation frames", self.review_max_score_frames_spin)
        enhance_form.addRow("Temporal background", self.review_bg_check)
        enhance_form.addRow("Projection percentile", self.review_projection_pct_spin)
        enhance_form.addRow("Score smoothing sigma", self.review_gaussian_sigma_spin)

        mask_group = QGroupBox("Mask From Enhanced Result")
        mask_form = QFormLayout(mask_group)

        self.review_sensitivity_combo = QComboBox()
        self.review_sensitivity_combo.addItems(["Sensitive", "Balanced", "Conservative"])
        sensitivity_value = str(getattr(p, "detection_sensitivity", "Sensitive") or "Sensitive")
        idx = self.review_sensitivity_combo.findText(sensitivity_value)
        self.review_sensitivity_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.review_multithreshold_check = QCheckBox("Multi-threshold seed proposal")
        self.review_multithreshold_check.setChecked(bool(getattr(p, "enable_multithreshold_seed", True)))

        self.review_log_seed_check = QCheckBox("LoG blob rescue seeds")
        self.review_log_seed_check.setChecked(bool(getattr(p, "enable_log_seed", True)))

        self.review_log_threshold_spin = self._new_review_double_spin(getattr(p, "log_threshold", 0.035), 0.001, 0.500, 3, 0.005)
        self.review_log_min_sigma_spin = self._new_review_double_spin(getattr(p, "log_min_sigma", 1.5), 0.5, 20.0, 1, 0.5)
        self.review_log_max_sigma_spin = self._new_review_double_spin(getattr(p, "log_max_sigma", 6.0), 0.5, 50.0, 1, 0.5)

        self.review_threshold_pct_spin = self._new_review_double_spin(p.threshold_percentile, 0.0, 100.0, 2, 0.5)
        self.review_min_distance_spin = QSpinBox()
        self.review_min_distance_spin.setRange(1, 200)
        self.review_min_distance_spin.setValue(int(p.min_distance))

        self.review_max_rois_spin = QSpinBox()
        self.review_max_rois_spin.setRange(1, 10000)
        self.review_max_rois_spin.setValue(int(p.max_rois))

        self.review_roi_radius_spin = QSpinBox()
        self.review_roi_radius_spin.setRange(1, 100)
        self.review_roi_radius_spin.setValue(int(p.roi_radius_px))

        mask_form.addRow("Detection sensitivity", self.review_sensitivity_combo)
        mask_form.addRow("", self.review_multithreshold_check)
        mask_form.addRow("", self.review_log_seed_check)
        mask_form.addRow("LoG threshold", self.review_log_threshold_spin)
        mask_form.addRow("LoG min sigma px", self.review_log_min_sigma_spin)
        mask_form.addRow("LoG max sigma px", self.review_log_max_sigma_spin)

        self.review_threshold_slider = self._add_spin_slider_row(mask_form, "Seed threshold percentile", self.review_threshold_pct_spin, 0, 10000, 100.0)
        self.review_min_distance_slider = self._add_spin_slider_row(mask_form, "Min seed distance px", self.review_min_distance_spin, 1, 200, 1.0)
        self.review_max_rois_slider = self._add_spin_slider_row(mask_form, "Max neurons", self.review_max_rois_spin, 1, 1000, 1.0)
        self.review_roi_radius_slider = self._add_spin_slider_row(mask_form, "Approx. ROI radius px", self.review_roi_radius_spin, 1, 100, 1.0)

        self.review_region_corr_spin = self._new_review_double_spin(p.region_corr_threshold, 0.00, 1.00, 2, 0.05)
        self.review_min_area_spin = QSpinBox()
        self.review_min_area_spin.setRange(1, 10000)
        self.review_min_area_spin.setValue(int(p.min_roi_area_px))

        self.review_max_area_spin = QSpinBox()
        self.review_max_area_spin.setRange(1, 100000)
        self.review_max_area_spin.setValue(int(p.max_roi_area_px))

        self.review_min_snr_spin = self._new_review_double_spin(p.min_roi_snr, 0.0, 20.0, 2, 0.1)
        self.review_neuropil_corr_spin = self._new_review_double_spin(p.max_neuropil_corr, 0.0, 1.0, 2, 0.05)

        self.review_region_corr_slider = self._add_spin_slider_row(mask_form, "Region corr threshold", self.review_region_corr_spin, 0, 100, 100.0)
        self.review_min_area_slider = self._add_spin_slider_row(mask_form, "Min ROI area px", self.review_min_area_spin, 1, 10000, 1.0)
        self.review_max_area_slider = self._add_spin_slider_row(mask_form, "Max ROI area px", self.review_max_area_spin, 1, 100000, 1.0)
        self.review_min_snr_slider = self._add_spin_slider_row(mask_form, "Min ROI SNR", self.review_min_snr_spin, 0, 2000, 100.0)
        self.review_neuropil_corr_slider = self._add_spin_slider_row(mask_form, "Max neuropil corr", self.review_neuropil_corr_spin, 0, 100, 100.0)

        self.review_sensitivity_combo.currentTextChanged.connect(self.on_detection_sensitivity_changed)
        self.review_multithreshold_check.toggled.connect(lambda _checked: self.update_neuron_mask_live_from_review())
        self.review_log_seed_check.toggled.connect(lambda _checked: self.update_neuron_mask_live_from_review())
        self.review_log_threshold_spin.valueChanged.connect(lambda _value: self.update_neuron_mask_live_from_review())
        self.review_log_min_sigma_spin.valueChanged.connect(lambda _value: self.update_neuron_mask_live_from_review())
        self.review_log_max_sigma_spin.valueChanged.connect(lambda _value: self.update_neuron_mask_live_from_review())

        roi_btn_row = QHBoxLayout()

        self.review_auto_detect_btn = QPushButton("Auto Calculate ROIs")
        self.review_auto_detect_btn.clicked.connect(self.auto_detect_neuron_centers_from_review)

        self.review_manual_mark_btn = QPushButton("Manual ROI Mark")
        self.review_manual_mark_btn.setCheckable(True)
        self.review_manual_mark_btn.clicked.connect(self.toggle_manual_neuron_roi_mode)

        self.review_clear_rois_btn = QPushButton("Clear ROIs")
        self.review_clear_rois_btn.clicked.connect(self.clear_neuron_centers_from_review)

        roi_btn_row.addWidget(self.review_auto_detect_btn)
        roi_btn_row.addWidget(self.review_manual_mark_btn)
        roi_btn_row.addWidget(self.review_clear_rois_btn)
        mask_form.addRow("ROI operations", roi_btn_row)

        help_label = QLabel("Auto Calculate uses multi-threshold score-map seeds plus LoG blob rescue seeds, then temporal-correlation region growing. Sensitive mode proposes more weak candidates so you can remove false positives in Manual ROI Mark.")
        help_label.setWordWrap(True)
        mask_form.addRow(help_label)

        outer.addWidget(enhance_group)
        outer.addWidget(mask_group)
        outer.addStretch(1)
        return container

    def collect_review_params(self) -> NeuronParams:
        return NeuronParams(
            frame_count=int(self.frame_count_spin.value()),
            capture_interval_ms=int(self.capture_interval_spin.value()),
            frame_rate_hz=float(self.frame_rate_spin.value()),
            do_motion_correction=self.review_motion_check.isChecked(),
            max_motion_shift_px=int(self.review_max_motion_shift_spin.value()),
            do_temporal_min_background=self.review_bg_check.isChecked(),
            do_spatial_background=self.review_spatial_bg_check.isChecked(),
            spatial_background_sigma=float(self.review_spatial_bg_sigma_spin.value()),
            projection_percentile=float(self.review_projection_pct_spin.value()),
            gaussian_sigma=float(self.review_gaussian_sigma_spin.value()),
            min_distance=int(self.review_min_distance_spin.value()),
            threshold_percentile=float(self.review_threshold_pct_spin.value()),
            max_rois=int(self.review_max_rois_spin.value()),
            roi_radius_px=int(self.review_roi_radius_spin.value()),
            detection_sensitivity=str(self.review_sensitivity_combo.currentText()),
            enable_multithreshold_seed=bool(self.review_multithreshold_check.isChecked()),
            enable_log_seed=bool(self.review_log_seed_check.isChecked()),
            log_min_sigma=float(self.review_log_min_sigma_spin.value()),
            log_max_sigma=float(self.review_log_max_sigma_spin.value()),
            log_num_sigma=int(getattr(DEFAULT_NEURON_PARAMS, "log_num_sigma", 6)),
            log_threshold=float(self.review_log_threshold_spin.value()),
            region_corr_threshold=float(self.review_region_corr_spin.value()),
            min_roi_area_px=int(self.review_min_area_spin.value()),
            max_roi_area_px=int(self.review_max_area_spin.value()),
            min_roi_snr=float(self.review_min_snr_spin.value()),
            max_neuropil_corr=float(self.review_neuropil_corr_spin.value()),
            max_processing_frames=int(self.review_max_processing_frames_spin.value()),
            max_processing_side_px=int(self.review_max_processing_side_spin.value()),
            max_score_frames=int(self.review_max_score_frames_spin.value()),
        )

    def on_detection_sensitivity_changed(self, text: str):
        """Apply practical defaults when the sensitivity preset changes."""
        if not hasattr(self, "review_threshold_pct_spin"):
            return

        text = str(text or "Balanced")
        presets = {
            "Sensitive": {
                "threshold": 97.0,
                "min_distance": 8,
                "region_corr": 0.22,
                "min_area": 8,
                "min_snr": 0.70,
                "neuropil_corr": 0.92,
                "max_rois": 300,
                "log_threshold": 0.035,
                "multithreshold": True,
                "log": True,
            },
            "Balanced": {
                "threshold": 97.8,
                "min_distance": 9,
                "region_corr": 0.27,
                "min_area": 10,
                "min_snr": 0.90,
                "neuropil_corr": 0.88,
                "max_rois": 250,
                "log_threshold": 0.045,
                "multithreshold": True,
                "log": True,
            },
            "Conservative": {
                "threshold": 98.7,
                "min_distance": 11,
                "region_corr": 0.34,
                "min_area": 14,
                "min_snr": 1.20,
                "neuropil_corr": 0.82,
                "max_rois": 200,
                "log_threshold": 0.065,
                "multithreshold": False,
                "log": False,
            },
        }
        v = presets.get(text, presets["Balanced"])

        controls = [
            self.review_threshold_pct_spin,
            self.review_min_distance_spin,
            self.review_region_corr_spin,
            self.review_min_area_spin,
            self.review_min_snr_spin,
            self.review_neuropil_corr_spin,
            self.review_max_rois_spin,
            self.review_log_threshold_spin,
            self.review_multithreshold_check,
            self.review_log_seed_check,
        ]
        for w in controls:
            w.blockSignals(True)

        self.review_threshold_pct_spin.setValue(float(v["threshold"]))
        self.review_min_distance_spin.setValue(int(v["min_distance"]))
        self.review_region_corr_spin.setValue(float(v["region_corr"]))
        self.review_min_area_spin.setValue(int(v["min_area"]))
        self.review_min_snr_spin.setValue(float(v["min_snr"]))
        self.review_neuropil_corr_spin.setValue(float(v["neuropil_corr"]))
        self.review_max_rois_spin.setValue(int(v["max_rois"]))
        self.review_log_threshold_spin.setValue(float(v["log_threshold"]))
        self.review_multithreshold_check.setChecked(bool(v["multithreshold"]))
        self.review_log_seed_check.setChecked(bool(v["log"]))

        for w in controls:
            w.blockSignals(False)

        self.update_neuron_mask_live_from_review()


    def apply_params_to_main(self, p: NeuronParams):
        self.current_params = NeuronParams(**p.__dict__)
        self.frame_count_spin.setValue(int(p.frame_count))
        self.capture_interval_spin.setValue(int(p.capture_interval_ms))
        self.frame_rate_spin.setValue(float(p.frame_rate_hz))

    def apply_review_params_to_main_if_available(self):
        if not hasattr(self, "review_bg_check"):
            return
        try:
            self.apply_params_to_main(self.collect_review_params())
        except RuntimeError:
            pass

    def _current_centers_yx(self):
        if self.last_result is None:
            return []
        return [(int(cy), int(cx)) for _roi_id, cy, cx in self.last_result.get("centers", [])]

    def update_neuron_result_from_centers(self, centers_yx):
        if self.last_result is None:
            return
        radius = int(self.review_roi_radius_spin.value()) if hasattr(self, "review_roi_radius_spin") else int(self.current_params.roi_radius_px)
        self.last_result = rebuild_neuron_mask_from_centers(self.last_result, centers_yx, radius_px=radius)
        self.last_mosaic = make_neuron_mosaic(self.last_result, roi_radius_px=radius)
        if isinstance(self.result_dialog, MaskReviewDialog):
            self.result_dialog.set_image(self.last_mosaic)
        self._sync_manual_roi_dialog()
        self.show_mask_btn.setEnabled(True)

    def update_neuron_result_from_auto_detection(self, p: NeuronParams):
        if self.last_result is None:
            return
        dff = self.last_result.get("dff")
        score = self.last_result.get("score", self.last_result.get("enhanced"))
        if dff is None or score is None:
            centers_yx = detect_neuron_centers_from_enhanced(
                self.last_result["enhanced"],
                min_distance=int(p.min_distance),
                threshold_percentile=float(p.threshold_percentile),
                max_rois=int(p.max_rois),
            )
            self.last_result = rebuild_neuron_mask_from_centers(self.last_result, centers_yx, radius_px=int(p.roi_radius_px))
        else:
            labels, mask, valid_centers = detect_neuron_rois_from_score(score, dff, p)
            updated = dict(self.last_result)
            updated["labels"] = labels
            updated["mask"] = mask
            updated["mask_projector"] = resize_mask_to_projector(mask)
            updated["centers"] = valid_centers
            self.last_result = updated

        self.last_mosaic = make_neuron_mosaic(self.last_result, roi_radius_px=int(p.roi_radius_px))
        if isinstance(self.result_dialog, MaskReviewDialog):
            self.result_dialog.set_image(self.last_mosaic)
        self._sync_manual_roi_dialog()
        self.show_mask_btn.setEnabled(True)

    def update_neuron_mask_live_from_review(self):
        if self._live_update_busy or self.last_result is None or not hasattr(self, "review_threshold_pct_spin"):
            return
        if self.manual_roi_mode:
            # Manual mode preserves clicked centers; radius slider still updates mask size.
            self.update_neuron_result_from_centers(self._current_centers_yx())
            return
        try:
            p = self.collect_review_params()
        except RuntimeError:
            return

        self._live_update_busy = True
        try:
            self.update_neuron_result_from_auto_detection(p)
            self.app.set_status(f"Neuron mask live-updated by Sensitive/Multi-seed/LoG temporal growing. Cells: {len(self.last_result['centers'])}.")
        finally:
            self._live_update_busy = False

    def _map_mosaic_click_to_projection(self, x: int, y: int):
        if self.last_result is None:
            return None
        h, w = self.last_result["projection"].shape[:2]
        sep = 6
        if y < 0 or y >= h:
            return None
        # Projection panel
        if 0 <= x < w:
            return int(y), int(x)
        # Enhanced panel
        if w + sep <= x < 2 * w + sep:
            return int(y), int(x - w - sep)
        # Mask panel
        if 2 * (w + sep) <= x < 3 * w + 2 * sep:
            return int(y), int(x - 2 * (w + sep))
        return None

    def on_review_image_clicked(self, x: int, y: int, button: int):
        if self.manual_roi_mode:
            self.open_manual_neuron_roi_dialog()

    def _manual_dialog_stack(self) -> np.ndarray:
        if self.last_result is None or self.last_result.get("projection") is None:
            raise RuntimeError(
                "No processed projection is available for manual ROI review."
            )
        return _to_uint8_gray_stack(self.last_result["projection"])

    def _sync_manual_roi_dialog(self):
        if not isinstance(self.manual_roi_dialog, NeuronManualRoiDialog) or not self.manual_roi_dialog.isVisible():
            return
        try:
            radius = int(self.review_roi_radius_spin.value()) if hasattr(self, "review_roi_radius_spin") else int(self.current_params.roi_radius_px)
            self.manual_roi_dialog.set_stack(self._manual_dialog_stack())
            self.manual_roi_dialog.set_centers_and_radius(self._current_centers_yx(), radius)
        except Exception:
            pass

    def open_manual_neuron_roi_dialog(self):
        if self.last_result is None:
            QMessageBox.information(self, "Manual ROI Mark", "Please calculate a neuron mask before manual ROI review.")
            return
        try:
            stack = self._manual_dialog_stack()
        except Exception as exc:
            QMessageBox.warning(self, "Manual ROI Mark", str(exc))
            return

        radius = int(self.review_roi_radius_spin.value()) if hasattr(self, "review_roi_radius_spin") else int(self.current_params.roi_radius_px)
        centers = self._current_centers_yx()

        if isinstance(self.manual_roi_dialog, NeuronManualRoiDialog):
            self.manual_roi_dialog.set_stack(stack)
            self.manual_roi_dialog.set_centers_and_radius(centers, radius)
            self.manual_roi_dialog.show()
            self.manual_roi_dialog.raise_()
            self.manual_roi_dialog.activateWindow()
            return

        self.manual_roi_dialog = NeuronManualRoiDialog(stack, centers, radius_px=radius, parent=self)
        self.manual_roi_dialog.centersChanged.connect(self.on_manual_roi_dialog_centers_changed)
        self.manual_roi_dialog.show()

    def on_manual_roi_dialog_centers_changed(self, centers_yx):
        self.update_neuron_result_from_centers(centers_yx)
        self.app.set_status(f"Manual neuron ROIs updated. Cells: {len(self.last_result['centers'])}.")

    def toggle_manual_neuron_roi_mode(self, checked: bool = None):
        self.manual_roi_mode = bool(self.review_manual_mark_btn.isChecked()) if hasattr(self, "review_manual_mark_btn") else bool(checked)
        if hasattr(self, "review_manual_mark_btn"):
            self.review_manual_mark_btn.setText("Manual ROI Mark: ON" if self.manual_roi_mode else "Manual ROI Mark")
        if self.manual_roi_mode:
            self.open_manual_neuron_roi_dialog()
        self.app.set_status(
            "Manual neuron ROI mode enabled in a separate popup: left-click to add, right-click to remove nearest."
            if self.manual_roi_mode
            else "Manual neuron ROI mode disabled."
        )

    def auto_detect_neuron_centers_from_review(self):
        if self.last_result is None or not hasattr(self, "review_threshold_pct_spin"):
            return
        if hasattr(self, "review_manual_mark_btn"):
            self.review_manual_mark_btn.setChecked(False)
            self.review_manual_mark_btn.setText("Manual ROI Mark")
        self.manual_roi_mode = False
        p = self.collect_review_params()
        self.update_neuron_result_from_auto_detection(p)
        self.app.set_status(f"Auto-calculated neuron ROIs by Sensitive/Multi-seed/LoG temporal growing. Cells: {len(self.last_result['centers'])}.")

    def clear_neuron_centers_from_review(self):
        if self.last_result is None:
            return
        if hasattr(self, "review_manual_mark_btn"):
            self.review_manual_mark_btn.setChecked(False)
            self.review_manual_mark_btn.setText("Manual ROI Mark")
        self.manual_roi_mode = False
        self.update_neuron_result_from_centers([])
        self.app.set_status("Neuron ROIs cleared in review window.")

    def recalculate_from_review_dialog(self):
        if self.pending_stack is None:
            QMessageBox.warning(self, "Recalculate Mask", "No captured neuron stack is available for recalculation.")
            return

        p = self.collect_review_params()
        self.apply_params_to_main(p)
        self.active = True
        self.set_controls_enabled(False)
        self.show_mask_btn.setEnabled(False)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(False)
        source_note = self.last_stack_source or "captured stack"
        self.app.set_status(f"Recalculating neuron mask from existing stack: {source_note}...")
        self.processing_thread = NeuronProcessingThread(self.pending_stack, p, self)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_failed)
        self.processing_thread.start()

    def on_processing_failed(self, error_text: str):
        self.active = False
        self.set_controls_enabled(True)
        self.show_mask_btn.setEnabled(self.last_result is not None)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(self.last_result is not None)
        QMessageBox.critical(self, "Neuron Processing Failed", error_text)
        self.app.set_status("Neuron processing failed")

    def _abort(self, reason: str):
        self.active = False
        self.set_controls_enabled(True)
        self.show_mask_btn.setEnabled(self.last_result is not None)
        if hasattr(self, "manual_roi_btn"):
            self.manual_roi_btn.setEnabled(self.last_result is not None)
        QMessageBox.critical(self, "Neuron Acquisition Aborted", reason)
        self.app.set_status(f"Neuron acquisition aborted: {reason}")

    def abort_if_active(self):
        if self.active:
            self._abort("Window closed. Acquisition aborted.")

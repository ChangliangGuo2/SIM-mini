import os
import json
import glob
import sys
import traceback
from datetime import datetime
import time


def _fit_window_geometry_to_screens(x, y, width, height, screen_rects, fallback_rect=None):
    """Return a window rectangle that is fully visible on an available screen."""
    requested = (int(x), int(y), max(1, int(width)), max(1, int(height)))

    def normalize_rect(rect):
        if rect is None or len(rect) != 4:
            return None
        rx, ry, rw, rh = (int(value) for value in rect)
        if rw <= 0 or rh <= 0:
            return None
        return rx, ry, rw, rh

    screens = [normalized for rect in screen_rects if (normalized := normalize_rect(rect)) is not None]
    if not screens:
        return requested

    fallback = normalize_rect(fallback_rect) or screens[0]
    x, y, width, height = requested

    def intersection_area(screen):
        sx, sy, sw, sh = screen
        overlap_w = max(0, min(x + width, sx + sw) - max(x, sx))
        overlap_h = max(0, min(y + height, sy + sh) - max(y, sy))
        return overlap_w * overlap_h

    target = max(screens, key=intersection_area)
    if intersection_area(target) == 0:
        target = fallback
        place_at_center = True
    else:
        place_at_center = False

    sx, sy, sw, sh = target
    width = min(width, sw)
    height = min(height, sh)

    if place_at_center:
        x = sx + (sw - width) // 2
        y = sy + (sh - height) // 2
    else:
        x = min(max(x, sx), sx + sw - width)
        y = min(max(y, sy), sy + sh - height)

    return int(x), int(y), int(width), int(height)

def _startup_log_path():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.join(base_dir, "sim_gui_startup_error.log")


def _write_startup_error_log(details):
    """Write startup/runtime errors to a file that can be found after double-click launch failures."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{timestamp}] SIM GUI error\n{details}\n"
    candidates = [_startup_log_path(), os.path.join(os.getcwd(), "sim_gui_startup_error.log")]
    last_error = None
    for path in candidates:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n" + "=" * 80 + "\n")
            return path
        except Exception as exc:
            last_error = exc
    try:
        print(text, file=sys.stderr)
        if last_error is not None:
            print(f"Could not write log file: {last_error}", file=sys.stderr)
    except Exception:
        pass
    return None


def _show_startup_error_and_exit(title, message, details=None, exit_code=1):
    """Show a simple system dialog before Qt is available, then exit."""
    full_details = details or message
    log_path = _write_startup_error_log(full_details)
    display_message = message
    if log_path:
        display_message += f"\n\nError log:\n{log_path}"
    try:
        import tkinter as _tk
        from tkinter import messagebox as _tk_messagebox
        root = _tk.Tk()
        root.withdraw()
        _tk_messagebox.showerror(title, display_message)
        root.destroy()
    except Exception:
        try:
            print(f"{title}: {display_message}\n\n{full_details}", file=sys.stderr)
        except Exception:
            pass
    raise SystemExit(exit_code)


def _require_startup_import(import_statement_name, install_hint):
    try:
        return __import__(import_statement_name)
    except Exception:
        details = traceback.format_exc()
        _show_startup_error_and_exit(
            "SIM GUI startup error",
            "A required Python package could not be imported.\n\n"
            f"Missing/import-failed module: {import_statement_name}\n"
            f"Install/repair it with:\n{install_hint}",
            details=details,
        )


try:
    import cv2
except Exception:
    _show_startup_error_and_exit(
        "SIM GUI startup error",
        "OpenCV could not be imported. Install it with:\npip install opencv-python",
        details=traceback.format_exc(),
    )

try:
    import numpy as np
except Exception:
    _show_startup_error_and_exit(
        "SIM GUI startup error",
        "NumPy could not be imported. Install it with:\npip install numpy",
        details=traceback.format_exc(),
    )

try:
    import pywt
except Exception:
    _show_startup_error_and_exit(
        "SIM GUI startup error",
        "PyWavelets could not be imported. Install it with:\npip install PyWavelets",
        details=traceback.format_exc(),
    )

try:
    import bm3d as _bm3d_package
except Exception:
    _bm3d_package = None
try:
    import tifffile
except Exception:
    tifffile = None

# =========================================================
# Optional GPU acceleration backend
# =========================================================
# Usage:
#   - Default: CPU-safe startup (CuPy is not imported automatically).
#   - Disable: SIM_USE_GPU=0     -> force CPU path.
#   - Force:   SIM_USE_GPU=1     -> explicitly load and use CuPy/CUDA.
#   - Small arrays are kept on CPU by default to avoid PCIe transfer overhead.
#     Tune with SIM_GPU_MIN_PIXELS, e.g. SIM_GPU_MIN_PIXELS=65536.
# CuPy can block for a long time while loading its CUDA/native backend. Keep
# startup CPU-safe: only an explicit SIM_USE_GPU=1/true request loads CuPy.
_gpu_mode_at_startup = os.environ.get("SIM_USE_GPU", "auto").strip().lower()
_gpu_explicitly_requested_at_startup = _gpu_mode_at_startup in (
    "1", "true", "yes", "on", "gpu", "cuda", "force", "forced"
)

if not _gpu_explicitly_requested_at_startup:
    _cp = None
    _cupyx_gaussian_filter = None
    _cupyx_uniform_filter = None
    _cupy_import_error = "deferred; set SIM_USE_GPU=1 to enable CUDA"
else:
    try:
        import cupy as _cp
        from cupyx.scipy.ndimage import gaussian_filter as _cupyx_gaussian_filter
        from cupyx.scipy.ndimage import uniform_filter as _cupyx_uniform_filter
        _cupy_import_error = None
    except Exception as _gpu_import_exc:
        _cp = None
        _cupyx_gaussian_filter = None
        _cupyx_uniform_filter = None
        _cupy_import_error = str(_gpu_import_exc)

_SIM_GPU_RUNTIME_AVAILABLE = None
_SIM_GPU_LAST_ERROR = None


def _gpu_mode():
    return os.environ.get("SIM_USE_GPU", "auto").strip().lower()


def _gpu_env_enabled():
    return _gpu_mode() not in ("0", "false", "no", "off", "cpu", "disable", "disabled")


def _gpu_env_forced():
    return _gpu_mode() in ("1", "true", "yes", "on", "gpu", "cuda", "force", "forced")


def _gpu_min_pixels(default=262144):
    try:
        return max(1, int(os.environ.get("SIM_GPU_MIN_PIXELS", str(default))))
    except Exception:
        return int(default)


def gpu_runtime_available():
    """Return True when CuPy and a CUDA device are usable."""
    global _SIM_GPU_RUNTIME_AVAILABLE, _SIM_GPU_LAST_ERROR
    if _SIM_GPU_RUNTIME_AVAILABLE is not None:
        return bool(_SIM_GPU_RUNTIME_AVAILABLE)
    if not _gpu_env_enabled():
        _SIM_GPU_RUNTIME_AVAILABLE = False
        _SIM_GPU_LAST_ERROR = "disabled by SIM_USE_GPU"
        return False
    if _cp is None:
        _SIM_GPU_RUNTIME_AVAILABLE = False
        _SIM_GPU_LAST_ERROR = _cupy_import_error or "cupy is not installed"
        return False
    try:
        count = int(_cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            _SIM_GPU_RUNTIME_AVAILABLE = False
            _SIM_GPU_LAST_ERROR = "no CUDA device was detected"
            return False
        _cp.cuda.Device(0).use()
        _SIM_GPU_RUNTIME_AVAILABLE = True
        _SIM_GPU_LAST_ERROR = None
        return True
    except Exception as exc:
        _SIM_GPU_RUNTIME_AVAILABLE = False
        _SIM_GPU_LAST_ERROR = str(exc)
        return False


def gpu_backend_status_text():
    """Human-readable GPU backend status for the GUI log."""
    if gpu_runtime_available():
        try:
            dev_id = int(_cp.cuda.Device().id)
            props = _cp.cuda.runtime.getDeviceProperties(dev_id)
            name = props.get("name", b"CUDA device")
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            total_mem_gb = float(props.get("totalGlobalMem", 0)) / (1024 ** 3)
            return (
                f"GPU acceleration: enabled via CuPy on device {dev_id} "
                f"({name}, {total_mem_gb:.1f} GB). "
                f"Set SIM_USE_GPU=0 to force CPU."
            )
        except Exception:
            return "GPU acceleration: enabled via CuPy. Set SIM_USE_GPU=0 to force CPU."
    reason = _SIM_GPU_LAST_ERROR or "not available"
    return (
        "GPU acceleration: CPU fallback active "
        f"({reason}). Set SIM_USE_GPU=1 to explicitly enable CUDA (and install a matching CuPy package)."
    )


def _should_use_gpu_array(*arrays, min_pixels=None):
    """Use GPU only for large 2D numeric arrays unless forced by SIM_USE_GPU=1."""
    if not gpu_runtime_available():
        return False
    if min_pixels is None:
        min_pixels = 1 if _gpu_env_forced() else _gpu_min_pixels()
    try:
        max_size = 0
        for arr in arrays:
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.ndim < 2:
                continue
            max_size = max(max_size, int(a.size))
        return max_size >= int(min_pixels)
    except Exception:
        return False


def _gpu_fail_and_fallback(exc):
    """Remember a GPU failure and continue with the CPU implementation."""
    global _SIM_GPU_RUNTIME_AVAILABLE, _SIM_GPU_LAST_ERROR
    _SIM_GPU_RUNTIME_AVAILABLE = False
    _SIM_GPU_LAST_ERROR = str(exc)


def _as_numpy_float32(arr):
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr).astype(np.float32, copy=False)
    return np.asarray(arr, dtype=np.float32)


def _gaussian_blur_accel(img, sigma):
    """Gaussian blur: CuPy/CUDA for large arrays, OpenCV CPU fallback."""
    arr = np.asarray(img, dtype=np.float32)
    sigma = float(sigma)
    if arr.size == 0 or sigma <= 0:
        return arr.copy()
    if _should_use_gpu_array(arr):
        try:
            x = _cp.asarray(arr, dtype=_cp.float32)
            y = _cupyx_gaussian_filter(x, sigma=(sigma, sigma), mode="reflect")
            return _cp.asnumpy(y).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)


def _uniform_filter_accel(img, kernel_size):
    """Box/uniform filter: CuPy/CUDA for large arrays, OpenCV CPU fallback."""
    arr = np.asarray(img, dtype=np.float32)
    k = _odd_kernel_size(kernel_size)
    if arr.size == 0:
        return arr.copy()
    if _should_use_gpu_array(arr):
        try:
            x = _cp.asarray(arr, dtype=_cp.float32)
            y = _cupyx_uniform_filter(x, size=(k, k), mode="reflect")
            return _cp.asnumpy(y).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)
    return cv2.blur(arr, (k, k)).astype(np.float32)

try:
    from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QUrl
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QMainWindow,
        QFileDialog,
        QMessageBox,
        QLabel,
        QPushButton,
        QLineEdit,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QVBoxLayout,
        QHBoxLayout,
        QFormLayout,
        QGroupBox,
        QTextEdit,
        QSizePolicy,
        QSplitter,
        QSlider,
        QTabWidget,
        QScrollArea,
        QProgressDialog,
    )
    from PySide6.QtGui import QDesktopServices
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:
    _show_startup_error_and_exit(
        "SIM GUI startup error",
        "GUI dependencies could not be imported. Install/repair them with:\n"
        "pip install PySide6 matplotlib",
        details=traceback.format_exc(),
    )


# =========================================================
# 基础图像处理函数
# =========================================================

def _to_gray_float32(img):
    """
    Convert an image array to grayscale float32 while preserving the original
    numeric dynamic range.

    Important for microscopy data:
    - Do not normalize 12/14/16-bit data to 8-bit.
    - Do not convert via cv2.COLOR_* on an already-normalized 8-bit copy.
    - For multi-channel images, average the first 3 channels in float32 so the
      original scale is retained regardless of RGB/BGR channel order.
    """
    arr = np.asarray(img)

    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float32, copy=False)
        if arr.shape[-1] in (3, 4):
            return np.mean(arr[..., :3].astype(np.float32), axis=-1).astype(np.float32)
        # Treat a non-color 3D array as a stack and use its first plane. Folder
        # input should contain single images; multi-page stacks should be loaded
        # through load_multipage_tiff().
        return arr[0].astype(np.float32, copy=False)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.mean(arr[..., :3].astype(np.float32), axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported image shape: {np.asarray(img).shape}")


def load_image(path):
    """
    Stable image reader compatible with Windows Chinese paths.

    Returns a grayscale float32 image while preserving the original intensity
    range, including 12/14/16-bit TIFF/PNG data. This replaces the older
    cv2.IMREAD_COLOR path, which could silently reduce scientific images to
    8-bit before reconstruction.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()

    # Prefer tifffile for TIFF because it keeps scientific metadata and bit depth.
    if ext in (".tif", ".tiff") and tifffile is not None:
        arr = tifffile.imread(path)
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            # A folder input TIFF may accidentally be multi-page. Use the first
            # page here; the dedicated stack loader handles full 6/9-page TIFFs.
            arr = arr[0]
        return _to_gray_float32(arr)

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return _to_gray_float32(img)


def load_multipage_tiff(path):
    """
    读取单个多页 TIFF 文件，返回灰度 float32 图像列表。

    支持:
        - 单页灰度: (H, W)
        - 多页灰度: (N, H, W)
        - 彩色单页: (H, W, 3/4)
        - 彩色多页: (N, H, W, 3/4)

    本程序重建需要 6 或 9 页。
    """
    if tifffile is None:
        raise RuntimeError("tifffile is not available. Please install it with: pip install tifffile")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    data = tifffile.imread(path)
    arr = np.asarray(data)

    if arr.ndim == 2:
        return [_to_gray_float32(arr)]

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return [_to_gray_float32(arr)]
        return [_to_gray_float32(arr[i]) for i in range(arr.shape[0])]

    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        return [_to_gray_float32(arr[i]) for i in range(arr.shape[0])]

    raise ValueError(f"Unsupported multi-page TIFF shape: {arr.shape}")


def reorder_stripe_sequence(items, order_mode="orientation_major"):
    """
    将输入的 6/9 张相位图重排为程序内部统一使用的 orientation-major 顺序。

    内部顺序要求：
        [ori1_p0, ori1_p1, ori1_p2,
         ori2_p0, ori2_p1, ori2_p2,
         ori3_p0, ori3_p1, ori3_p2]

    order_mode:
        - orientation_major / keep:
            输入已经是内部顺序，不重排。
        - reverse_phase:
            每个方向内部相位反转：
            [p0, p1, p2] -> [p2, p1, p0]。
        - phase_major:
            输入按相位优先排列，例如 9 张：
            [ori1_p0, ori2_p0, ori3_p0,
             ori1_p1, ori2_p1, ori3_p1,
             ori1_p2, ori2_p2, ori3_p2]
            重排为 orientation-major。
        - 147258369_to_orientation_major:
            适合文件/页序为 1,4,7,2,5,8,3,6,9 的情况。
            对 9 张图等价于把列优先顺序改成每个方向连续 3 相。
            对 6 张图对应 1,3,5,2,4,6。
    """
    items = list(items)
    if len(items) not in (6, 9):
        raise ValueError(f"Expected 6 or 9 images/items, got {len(items)}")

    orientation_count = len(items) // 3
    order_mode = str(order_mode).lower()

    if order_mode in ("keep", "orientation_major"):
        return items

    if order_mode == "reverse_phase":
        reordered = []
        for ori_idx in range(orientation_count):
            group = items[ori_idx * 3:(ori_idx + 1) * 3]
            reordered.extend(group[::-1])
        return reordered

    if order_mode in ("phase_major", "147258369_to_orientation_major"):
        reordered = []
        for ori_idx in range(orientation_count):
            for phase_idx in range(3):
                src_idx = phase_idx * orientation_count + ori_idx
                reordered.append(items[src_idx])
        return reordered

    raise ValueError(f"Unsupported stripe order mode: {order_mode}")


def crop_roi(img, x, y, w, h):
    """
    按给定 ROI 裁剪图像。
    NumPy / OpenCV 切片顺序：
        img[y:y+h, x:x+w]
    """
    img_h, img_w = img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(
            f"ROI out of bounds. Image size: ({img_w}, {img_h}), "
            f"ROI: x={x}, y={y}, w={w}, h={h}"
        )
    return img[y:y + h, x:x + w]


def preprocess(img, background_percentile=5.0):
    """
    预处理：
    1) 用低百分位近似背景值
    2) 做背景扣除
    3) 小于 0 的值截断为 0

    GPU acceleration:
    - For large ROI arrays, CuPy computes percentile/subtract/clip on CUDA.
    - If CuPy/CUDA is unavailable, the original NumPy CPU path is used.
    """
    arr = np.asarray(img, dtype=np.float32)
    if _should_use_gpu_array(arr):
        try:
            x = _cp.asarray(arr, dtype=_cp.float32)
            bg = _cp.percentile(x, float(background_percentile))
            out = _cp.maximum(x - bg, 0)
            return _cp.asnumpy(out).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    out = arr.astype(np.float32, copy=True)
    bg = np.percentile(out, background_percentile)
    out = out - bg
    out[out < 0] = 0
    return out.astype(np.float32, copy=False)

def _normalize_preprocess_mode(mode):
    mode = str(mode).strip().lower().replace(" ", "_").replace("+", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "shared_percentile": "shared_percentile",
        "shared": "shared_percentile",
        "group_shared_percentile": "shared_percentile",
        "dark_only": "dark_only",
        "dark": "dark_only",
        "dark_flat": "dark_flat",
        "dark+flat": "dark_flat",
        "flat": "dark_flat",
    }
    return aliases.get(mode, "shared_percentile")


def preprocess_phase_group(
    group,
    background_percentile=5.0,
    mode="shared_percentile",
    dark_image=None,
    flat_image=None,
    flat_eps=1e-6,
):
    """
    Preprocess one orientation's 3 phase images without breaking phase relations.

    mode:
        - shared_percentile: subtract one common percentile background per group.
        - dark_only: subtract the same dark frame from each phase.
        - dark_flat: subtract dark, then divide by normalized (flat - dark).
        - none: return float32 copies without correction.

    GPU acceleration:
        CuPy accelerates the large per-pixel arithmetic and percentile/mean operations
        when a CUDA device is available. Returned arrays remain NumPy float32 so the
        existing GUI and PyWavelets code keep working unchanged.
    """
    group = [np.asarray(img, dtype=np.float32) for img in group]
    if len(group) != 3:
        raise ValueError(f"Expected 3 phase images in one orientation group, got {len(group)}")

    mode = _normalize_preprocess_mode(mode)
    info = {
        "mode": mode,
        "background_percentile": float(background_percentile),
        "shared_background": 0.0,
        "per_phase_percentile_backgrounds": [],
        "dark_mean": None,
        "flat_mean": None,
        "flat_corrected_mean": None,
    }

    if mode == "none":
        return [img.copy() for img in group], info

    use_gpu = _should_use_gpu_array(*group)
    if use_gpu:
        try:
            xs = [_cp.asarray(img, dtype=_cp.float32) for img in group]

            if mode == "shared_percentile":
                bgs = [float(_cp.asnumpy(_cp.percentile(x, float(background_percentile)))) for x in xs]
                common_bg = float(np.median(bgs))
                out = [_cp.asnumpy(_cp.maximum(x - common_bg, 0)).astype(np.float32, copy=False) for x in xs]
                info["per_phase_percentile_backgrounds"] = bgs
                info["shared_background"] = common_bg
                return out, info

            if mode in ("dark_only", "dark_flat"):
                if dark_image is None:
                    raise ValueError("Preprocess mode requires a dark image, but Dark image path is empty.")
                dark = np.asarray(dark_image, dtype=np.float32)
                if dark.shape != group[0].shape:
                    raise ValueError(f"Dark image shape {dark.shape} does not match ROI shape {group[0].shape}")
                dark_gpu = _cp.asarray(dark, dtype=_cp.float32)
                out_gpu = [_cp.maximum(x - dark_gpu, 0) for x in xs]
                info["dark_mean"] = float(np.mean(dark))

                if mode == "dark_flat":
                    if flat_image is None:
                        raise ValueError("Preprocess mode 'dark_flat' requires a flat-field image, but Flat image path is empty.")
                    flat = np.asarray(flat_image, dtype=np.float32)
                    if flat.shape != group[0].shape:
                        raise ValueError(f"Flat image shape {flat.shape} does not match ROI shape {group[0].shape}")
                    flat_gpu = _cp.asarray(flat, dtype=_cp.float32)
                    flat_corr = _cp.maximum(flat_gpu - dark_gpu, 0)
                    flat_threshold = max(float(_cp.asnumpy(_cp.percentile(flat_corr, 5.0))), float(flat_eps))
                    valid = flat_corr > flat_threshold
                    flat_mean_gpu = _cp.mean(flat_corr[valid]) if bool(_cp.asnumpy(_cp.any(valid))) else _cp.mean(flat_corr)
                    flat_mean = float(_cp.asnumpy(flat_mean_gpu))
                    if not np.isfinite(flat_mean) or flat_mean <= flat_eps:
                        raise ValueError("Flat-field image is too dark or invalid after dark subtraction.")
                    flat_norm = _cp.maximum(flat_corr / flat_mean, float(flat_eps))
                    out_gpu = [x / flat_norm for x in out_gpu]
                    info["flat_mean"] = float(np.mean(flat))
                    info["flat_corrected_mean"] = flat_mean

                out = [_cp.asnumpy(x).astype(np.float32, copy=False) for x in out_gpu]
                return out, info
        except ValueError:
            raise
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    if mode == "shared_percentile":
        bgs = [float(np.percentile(img, background_percentile)) for img in group]
        common_bg = float(np.median(bgs))
        out = []
        for img in group:
            x = img - common_bg
            x[x < 0] = 0
            out.append(x.astype(np.float32))
        info["per_phase_percentile_backgrounds"] = bgs
        info["shared_background"] = common_bg
        return out, info

    if mode in ("dark_only", "dark_flat"):
        if dark_image is None:
            raise ValueError("Preprocess mode requires a dark image, but Dark image path is empty.")
        dark = np.asarray(dark_image, dtype=np.float32)
        if dark.shape != group[0].shape:
            raise ValueError(f"Dark image shape {dark.shape} does not match ROI shape {group[0].shape}")

        out = []
        for img in group:
            x = img - dark
            x[x < 0] = 0
            out.append(x.astype(np.float32))
        info["dark_mean"] = float(np.mean(dark))

        if mode == "dark_flat":
            if flat_image is None:
                raise ValueError("Preprocess mode 'dark_flat' requires a flat-field image, but Flat image path is empty.")
            flat = np.asarray(flat_image, dtype=np.float32)
            if flat.shape != group[0].shape:
                raise ValueError(f"Flat image shape {flat.shape} does not match ROI shape {group[0].shape}")

            flat_corr = flat - dark
            flat_corr[flat_corr < 0] = 0
            valid = flat_corr > max(float(np.percentile(flat_corr, 5.0)), flat_eps)
            flat_mean = float(np.mean(flat_corr[valid])) if np.any(valid) else float(np.mean(flat_corr))
            if not np.isfinite(flat_mean) or flat_mean <= flat_eps:
                raise ValueError("Flat-field image is too dark or invalid after dark subtraction.")
            flat_norm = np.maximum(flat_corr / flat_mean, flat_eps)
            out = [(img / flat_norm).astype(np.float32) for img in out]
            info["flat_mean"] = float(np.mean(flat))
            info["flat_corrected_mean"] = flat_mean

        return out, info

    return preprocess_phase_group(
        group,
        background_percentile=background_percentile,
        mode="shared_percentile",
        dark_image=dark_image,
        flat_image=flat_image,
        flat_eps=flat_eps,
    )

def estimate_dynamic_range_max(img):
    """Estimate likely camera/file maximum for saturation diagnostics."""
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return 1.0
    vmax = float(np.nanmax(img))
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    candidates = [1.0, 255.0, 1023.0, 4095.0, 8191.0, 16383.0, 32767.0, 65535.0]
    for c in candidates:
        if vmax <= c:
            return c
    return vmax


def compute_saturation_ratio(img, dynamic_max=None, threshold_fraction=0.999):
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return 0.0, 1.0
    if dynamic_max is None:
        dynamic_max = estimate_dynamic_range_max(img)
    threshold = float(dynamic_max) * float(threshold_fraction)
    ratio = float(np.mean(img >= threshold)) if threshold > 0 else 0.0
    return ratio, float(dynamic_max)


def compute_fft_peak_diagnostics(img, central_exclusion_fraction=0.04):
    """Find strongest non-DC FFT peak in a phase-difference image."""
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return {
            "peak_x_offset_px": 0,
            "peak_y_offset_px": 0,
            "frequency_cycles_per_pixel": 0.0,
            "period_pixels": float("inf"),
            "angle_degrees": 0.0,
            "peak_to_median": 0.0,
        }

    if _should_use_gpu_array(arr):
        try:
            x = _cp.asarray(arr, dtype=_cp.float32)
            x = x - _cp.mean(x)
            wy = _cp.hanning(h).astype(_cp.float32)
            wx = _cp.hanning(w).astype(_cp.float32)
            x = x * wy[:, None] * wx[None, :]
            spec = _cp.abs(_cp.fft.fftshift(_cp.fft.fft2(x))).astype(_cp.float32)
            cy, cx = h // 2, w // 2
            yy, xx = _cp.indices((h, w))
            rr = _cp.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            exclude_radius = max(4.0, min(h, w) * float(central_exclusion_fraction))
            spec = _cp.where(rr <= exclude_radius, 0, spec)
            peak_index = int(_cp.asnumpy(_cp.argmax(spec)))
            py, px = np.unravel_index(peak_index, (h, w))
            peak_value = float(_cp.asnumpy(spec[py, px]))
            valid = spec[spec > 0]
            median_value = float(_cp.asnumpy(_cp.median(valid))) if int(valid.size) else 0.0
            peak_to_median = peak_value / (median_value + 1e-12)
            dx = float(px - cx)
            dy = float(py - cy)
            fx = dx / float(w)
            fy = dy / float(h)
            freq = float(np.sqrt(fx ** 2 + fy ** 2))
            period = float(1.0 / freq) if freq > 1e-12 else float("inf")
            angle = float(np.degrees(np.arctan2(dy, dx)))
            return {
                "peak_x_offset_px": int(dx),
                "peak_y_offset_px": int(dy),
                "frequency_cycles_per_pixel": freq,
                "period_pixels": period,
                "angle_degrees": angle,
                "peak_to_median": float(peak_to_median),
            }
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    arr = arr - float(np.mean(arr))
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    arr = arr * wy[:, None] * wx[None, :]

    spec = np.abs(np.fft.fftshift(np.fft.fft2(arr))).astype(np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    exclude_radius = max(4.0, min(h, w) * float(central_exclusion_fraction))
    spec_masked = spec.copy()
    spec_masked[rr <= exclude_radius] = 0

    peak_index = int(np.argmax(spec_masked))
    py, px = np.unravel_index(peak_index, spec.shape)
    peak_value = float(spec_masked[py, px])

    valid = spec_masked[spec_masked > 0]
    median_value = float(np.median(valid)) if valid.size else 0.0
    peak_to_median = peak_value / (median_value + 1e-12)

    dx = float(px - cx)
    dy = float(py - cy)
    fx = dx / float(w)
    fy = dy / float(h)
    freq = float(np.sqrt(fx ** 2 + fy ** 2))
    period = float(1.0 / freq) if freq > 1e-12 else float("inf")
    angle = float(np.degrees(np.arctan2(dy, dx)))

    return {
        "peak_x_offset_px": int(dx),
        "peak_y_offset_px": int(dy),
        "frequency_cycles_per_pixel": freq,
        "period_pixels": period,
        "angle_degrees": angle,
        "peak_to_median": float(peak_to_median),
    }

def compute_phase_group_diagnostics(group, phase_model="0/120/240"):
    """Diagnostics for one orientation's three phase images."""
    group = [np.asarray(img, dtype=np.float32) for img in group]
    phase_stats = []
    sat_ratios = []
    dynamic_max_values = []
    for img in group:
        sat, dyn = compute_saturation_ratio(img)
        phase_stats.append({
            "min": float(np.min(img)),
            "max": float(np.max(img)),
            "mean": float(np.mean(img)),
            "median": float(np.median(img)),
            "p99": float(np.percentile(img, 99.0)),
        })
        sat_ratios.append(float(sat))
        dynamic_max_values.append(float(dyn))

    wf = pseudo_widefield(group[0], group[1], group[2])
    amp = sim_reconstruct(group[0], group[1], group[2], phase_model=phase_model)
    eps = 1e-6
    wf_threshold = float(np.percentile(wf, 20.0))
    valid = wf > max(wf_threshold, eps)
    if np.any(valid):
        modulation_map = amp[valid] / np.maximum(wf[valid], eps)
    else:
        modulation_map = amp.ravel() / np.maximum(wf.ravel(), eps)

    diffs = [
        group[0] - group[1],
        group[1] - group[2],
        group[2] - group[0],
    ]
    diff_stds = [float(np.std(d)) for d in diffs]
    best_diff = diffs[int(np.argmax(diff_stds))]
    fft_peak = compute_fft_peak_diagnostics(best_diff)

    return {
        "phase_model": str(phase_model),
        "phase_stats": phase_stats,
        "saturation_ratios": sat_ratios,
        "dynamic_max_values": dynamic_max_values,
        "modulation_median": float(np.median(modulation_map)) if modulation_map.size else 0.0,
        "modulation_mean": float(np.mean(modulation_map)) if modulation_map.size else 0.0,
        "modulation_p95": float(np.percentile(modulation_map, 95.0)) if modulation_map.size else 0.0,
        "difference_std_values": diff_stds,
        "fft_peak": fft_peak,
    }


def format_phase_group_diagnostics(label, diag):
    sat_text = ", ".join(f"{100.0 * r:.4f}%" for r in diag.get("saturation_ratios", []))
    means = ", ".join(f"{s.get('mean', 0):.3f}" for s in diag.get("phase_stats", []))
    maxima = ", ".join(f"{s.get('max', 0):.3f}" for s in diag.get("phase_stats", []))
    fft = diag.get("fft_peak", {})
    return (
        f"{label}: phase_model={diag.get('phase_model', '')}; "
        f"phase_mean=[{means}]; phase_max=[{maxima}]; "
        f"saturation=[{sat_text}]; "
        f"modulation median/mean/p95="
        f"{diag.get('modulation_median', 0):.4f}/"
        f"{diag.get('modulation_mean', 0):.4f}/"
        f"{diag.get('modulation_p95', 0):.4f}; "
        f"FFT peak: freq={fft.get('frequency_cycles_per_pixel', 0):.5f} cyc/pix, "
        f"period={fft.get('period_pixels', float('inf')):.2f} pix, "
        f"angle={fft.get('angle_degrees', 0):.2f} deg, "
        f"peak/median={fft.get('peak_to_median', 0):.2f}"
    )


def normalize_for_display(img):
    """
    归一化到 0~1，用于 matplotlib 显示。
    """
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img

def get_percentile_range(img, low=1.0, high=99.0):
    """
    计算显示用分位数范围。
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(img, low))
    vmax = float(np.percentile(img, high))

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def prepare_display_image(img, auto_contrast=True, low=1.0, high=99.0, manual_min=0.0, manual_max=1.0):
    """
    为 imshow 准备显示参数，返回:
        disp_img, vmin, vmax
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img, 0.0, 1.0

    disp_img = img.copy()

    if auto_contrast:
        vmin, vmax = get_percentile_range(disp_img, low=low, high=high)
    else:
        vmin = float(manual_min)
        vmax = float(manual_max)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

    return disp_img, vmin, vmax


def compute_intensity_stat(img, method="median"):
    """
    计算图像强度统计量，用于方向间强度匹配。
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return 1.0

    method = str(method).lower()
    if method == "mean":
        value = float(np.mean(img))
    elif method == "p95":
        value = float(np.percentile(img, 95.0))
    else:
        value = float(np.median(img))

    if not np.isfinite(value) or abs(value) < 1e-12:
        value = 1.0
    return value

def compute_match_intensity_value(img, method="p95"):
    """
    Compute a robust intensity value for final output brightness matching.

    method:
        - mean
        - median
        - p90 / p95 / p99
        - max
    """
    img = np.asarray(img, dtype=np.float32)
    img = img[np.isfinite(img)]

    if img.size == 0:
        return 0.0

    img = img[img > 0]
    if img.size == 0:
        return 0.0

    method = str(method).strip().lower()

    if method == "mean":
        value = float(np.mean(img))
    elif method == "median":
        value = float(np.median(img))
    elif method == "p90":
        value = float(np.percentile(img, 90.0))
    elif method == "p99":
        value = float(np.percentile(img, 99.0))
    elif method == "max":
        value = float(np.max(img))
    else:
        value = float(np.percentile(img, 95.0))

    if not np.isfinite(value):
        value = 0.0

    return value


def match_output_intensity_to_reference(
    img,
    reference,
    method="p95",
    strength=1.0,
    max_scale=5.0,
    clip_output=True,
):
    """
    Match final output intensity range to a reference image.

    This is intended for preserving brightness after RMS fusion,
    HiLo frequency splitting, Fourier filtering, Wiener/NLM denoising, etc.

    Formula:
        scale = statistic(reference) / statistic(img)
        final_scale = 1 + strength * (scale - 1)
        output = img * final_scale

    strength:
        0.0 = keep original output
        1.0 = fully match reference statistic
    """
    img = np.asarray(img, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if img.shape != reference.shape:
        raise ValueError(
            f"Intensity match reference shape {reference.shape} does not match image shape {img.shape}"
        )

    method = str(method).strip().lower()
    strength = float(np.clip(strength, 0.0, 1.0))
    max_scale = float(max(max_scale, 1.0))

    target_value = compute_match_intensity_value(img, method=method)
    reference_value = compute_match_intensity_value(reference, method=method)

    info = {
        "method": method,
        "target_value": float(target_value),
        "reference_value": float(reference_value),
        "raw_scale": 1.0,
        "applied_scale": 1.0,
        "strength": float(strength),
        "max_scale": float(max_scale),
    }

    if target_value <= 1e-12 or reference_value <= 1e-12:
        out = img.copy()
        if clip_output:
            out = np.clip(out, 0, None)
        return out.astype(np.float32), info

    raw_scale = reference_value / target_value
    raw_scale = float(np.clip(raw_scale, 1.0 / max_scale, max_scale))

    applied_scale = 1.0 + strength * (raw_scale - 1.0)

    out = img * applied_scale
    if clip_output:
        out = np.clip(out, 0, None)

    info["raw_scale"] = float(raw_scale)
    info["applied_scale"] = float(applied_scale)

    return out.astype(np.float32), info

def match_image_sequence_intensity(images, method="median"):
    """
    将多张图像按统一统计量进行强度匹配。
    返回:
        matched_images, reference_value, scales
    """
    if not images:
        return [], 1.0, []

    stats = [compute_intensity_stat(img, method=method) for img in images]
    valid_stats = [s for s in stats if np.isfinite(s) and s > 1e-12]
    if not valid_stats:
        return [np.asarray(img, dtype=np.float32).copy() for img in images], 1.0, [1.0] * len(images)

    reference_value = float(np.median(valid_stats))
    if abs(reference_value) < 1e-12:
        reference_value = 1.0

    matched_images = []
    scales = []
    for img, stat in zip(images, stats):
        scale = reference_value / stat if abs(stat) > 1e-12 else 1.0
        matched_images.append((np.asarray(img, dtype=np.float32) * scale).astype(np.float32))
        scales.append(float(scale))

    return matched_images, reference_value, scales


def normalize_phase_groups(groups, mode="per_orientation", method="mean"):
    """
    对原始/预处理后的相位图做统一强度归一化。
    mode:
        - per_orientation: 每个 orientation 内 3 相共享一个参考值
        - global: 所有 orientation 和 phase 共用一个参考值
    method:
        - mean / median / p95
    返回:
        normalized_groups, reference_values, scales_by_group
    """
    if not groups:
        return [], [], []

    mode = str(mode).lower()
    method = str(method).lower()

    def _stat(img):
        return compute_intensity_stat(img, method=method)

    normalized_groups = []
    reference_values = []
    scales_by_group = []

    if mode == "global":
        flat_images = [img for group in groups for img in group]
        global_stats = [_stat(img) for img in flat_images]
        valid_stats = [s for s in global_stats if np.isfinite(s) and s > 1e-12]
        global_ref = float(np.mean(valid_stats)) if valid_stats else 1.0
        if abs(global_ref) < 1e-12:
            global_ref = 1.0

        for group in groups:
            norm_group = []
            scale_group = []
            for img in group:
                s = _stat(img)
                scale = global_ref / s if abs(s) > 1e-12 else 1.0
                norm_group.append((np.asarray(img, dtype=np.float32) * scale).astype(np.float32))
                scale_group.append(float(scale))
            normalized_groups.append(norm_group)
            reference_values.append(float(global_ref))
            scales_by_group.append(scale_group)
    else:
        for group in groups:
            group_stats = [_stat(img) for img in group]
            valid_stats = [s for s in group_stats if np.isfinite(s) and s > 1e-12]
            group_ref = float(np.mean(valid_stats)) if valid_stats else 1.0
            if abs(group_ref) < 1e-12:
                group_ref = 1.0

            norm_group = []
            scale_group = []
            for img, s in zip(group, group_stats):
                scale = group_ref / s if abs(s) > 1e-12 else 1.0
                norm_group.append((np.asarray(img, dtype=np.float32) * scale).astype(np.float32))
                scale_group.append(float(scale))

            normalized_groups.append(norm_group)
            reference_values.append(float(group_ref))
            scales_by_group.append(scale_group)

    return normalized_groups, reference_values, scales_by_group


def correct_low_frequency_background(img, sigma=25.0, mode="subtract"):
    """
    单方向 SIM 低频背景校正：
    - subtract: 减法校正
    - divide:   除法校正
    - hybrid:   减法和除法混合

    Gaussian low-pass uses GPU/CuPy for large arrays when available.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img.copy(), np.zeros_like(img, dtype=np.float32)

    sigma = float(sigma)
    if sigma <= 0:
        return img.copy(), np.zeros_like(img, dtype=np.float32)

    mode = str(mode).lower()
    background = _gaussian_blur_accel(img, sigma).astype(np.float32)
    eps = 1e-6

    if mode == "divide":
        bg_safe = np.maximum(background, eps)
        bg_mean = float(np.mean(bg_safe))
        corrected = img / bg_safe * bg_mean
    elif mode == "hybrid":
        subtract_part = img - background
        subtract_part[subtract_part < 0] = 0
        bg_safe = np.maximum(background, eps)
        bg_mean = float(np.mean(bg_safe))
        divide_part = img / bg_safe * bg_mean
        corrected = 0.5 * subtract_part + 0.5 * divide_part
    else:
        corrected = img - background
        corrected[corrected < 0] = 0

    corrected = np.clip(corrected, 0, None).astype(np.float32)
    return corrected, background.astype(np.float32)

def enhance_final_sim_structure(
    img,
    sigma=35.0,
    amount=1.0,
    keep_original=0.15,
    mode="highpass_mix",
):
    """
    最终 SIM 融合结果的结构增强 / 低频背景抑制。

    Gaussian low-pass uses GPU/CuPy for large arrays when available.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        z = np.zeros_like(img, dtype=np.float32)
        return img.copy(), z, z

    sigma = float(sigma)
    amount = float(amount)
    keep_original = float(keep_original)
    mode = str(mode).lower()

    if sigma <= 0:
        z = np.zeros_like(img, dtype=np.float32)
        return np.clip(img, 0, None).astype(np.float32), z, z

    background = _gaussian_blur_accel(img, sigma).astype(np.float32)

    highpass = img - background
    highpass[highpass < 0] = 0

    eps = 1e-6
    if mode == "divide":
        bg_safe = np.maximum(background, eps)
        bg_mean = float(np.mean(bg_safe))
        enhanced = img / bg_safe * bg_mean
    elif mode == "hybrid":
        subtract_part = img - amount * background
        subtract_part[subtract_part < 0] = 0

        bg_safe = np.maximum(background, eps)
        bg_mean = float(np.mean(bg_safe))
        divide_part = img / bg_safe * bg_mean

        enhanced = 0.5 * subtract_part + 0.5 * divide_part
    elif mode == "subtract":
        enhanced = img - amount * background
        enhanced[enhanced < 0] = 0
    else:
        enhanced = keep_original * img + amount * highpass
        enhanced[enhanced < 0] = 0

    return (
        np.clip(enhanced, 0, None).astype(np.float32),
        background.astype(np.float32),
        highpass.astype(np.float32),
    )

def save_image_tif(path, img):
    """
    保存为 16-bit TIFF。
    每张图单独归一化到 0~65535，适合显示型保存。
    兼容 Windows 中文路径。
    """
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max() * 65535.0
    img = img.astype(np.uint16)

    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Failed to encode image: {path}")
    encoded.tofile(path)


# =========================================================
# 核心公式函数
# =========================================================

def demodulate_three_phase(i0, i1, i2, phase_model="0/120/240"):
    """
    Demodulate one 3-phase orientation into cosine/sine quadrature components
    and modulation amplitude.

    Returns:
        c_component, s_component, amplitude

    GPU acceleration:
        Large arrays use CuPy for the elementwise quadrature math; results are
        returned as NumPy float32 arrays for compatibility with the rest of the GUI.
    """
    phase_model_norm = str(phase_model).lower().replace(" ", "")

    if _should_use_gpu_array(i0, i1, i2):
        try:
            x0 = _cp.asarray(i0, dtype=_cp.float32)
            x1 = _cp.asarray(i1, dtype=_cp.float32)
            x2 = _cp.asarray(i2, dtype=_cp.float32)
            if "90" in phase_model_norm:
                c = (x2 - x0).astype(_cp.float32)
                s = (2.0 * x1 - x2 - x0).astype(_cp.float32)
                amp = (0.5 * _cp.sqrt(c ** 2 + s ** 2)).astype(_cp.float32)
            else:
                c = ((2.0 / 3.0) * (x0 - 0.5 * x1 - 0.5 * x2)).astype(_cp.float32)
                s = ((2.0 / 3.0) * ((_cp.sqrt(_cp.float32(3.0)) / 2.0) * (x2 - x1))).astype(_cp.float32)
                amp = _cp.sqrt(c ** 2 + s ** 2).astype(_cp.float32)
            return (
                _cp.asnumpy(c).astype(np.float32, copy=False),
                _cp.asnumpy(s).astype(np.float32, copy=False),
                _cp.asnumpy(amp).astype(np.float32, copy=False),
            )
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    i0 = np.asarray(i0, dtype=np.float32)
    i1 = np.asarray(i1, dtype=np.float32)
    i2 = np.asarray(i2, dtype=np.float32)

    if "90" in phase_model_norm:
        c = (i2 - i0).astype(np.float32)
        s = (2.0 * i1 - i2 - i0).astype(np.float32)
        amp = (0.5 * np.sqrt(c ** 2 + s ** 2)).astype(np.float32)
        return c, s, amp

    c = ((2.0 / 3.0) * (i0 - 0.5 * i1 - 0.5 * i2)).astype(np.float32)
    s = ((2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (i2 - i1))).astype(np.float32)
    amp = np.sqrt(c ** 2 + s ** 2).astype(np.float32)
    return c, s, amp

def sim_reconstruct(i0, i1, i2, phase_model="0/120/240"):
    """
    Single-orientation OS-SIM demodulation.

    phase_model:
        - "0/90/180": legacy 3-frame formula used by the original script.
        - "0/120/240": standard equally spaced 3-phase demodulation.

    The returned image is the modulation-amplitude / optical-sectioning image,
    not a full frequency-domain SR-SIM reconstruction.
    """
    _, _, sim = demodulate_three_phase(i0, i1, i2, phase_model=phase_model)
    return sim.astype(np.float32)


def _frequency_radius_map(shape):
    """Return normalized FFT radius map in cycles/pixel for a 2D image shape."""
    h, w = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(xx ** 2 + yy ** 2).astype(np.float32)


def apply_fourier_frequency_filter(
    img,
    cutoff_cycles_per_pixel=0.18,
    filter_mode="Gaussian low-pass",
    butterworth_order=2,
    amount=1.0,
):
    """
    Frequency-domain low-pass filtering for the Fourier-OS-SIM branch.

    GPU acceleration:
        Large FFTs use CuPy/CUDA. CPU NumPy FFT is retained as a fallback.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img.copy(), 0.0

    cutoff = float(cutoff_cycles_per_pixel)
    if not np.isfinite(cutoff) or cutoff <= 0:
        return img.copy(), cutoff

    cutoff = float(np.clip(cutoff, 1e-4, 0.5))
    amount = float(np.clip(amount, 0.0, 1.0))
    mode = str(filter_mode).strip().lower()
    if mode in ("none", "off", "disabled") or amount <= 0:
        return img.copy(), cutoff

    if _should_use_gpu_array(img):
        try:
            x = _cp.asarray(img, dtype=_cp.float32)
            h, w = int(img.shape[0]), int(img.shape[1])
            fy = _cp.fft.fftfreq(h).astype(_cp.float32)
            fx = _cp.fft.fftfreq(w).astype(_cp.float32)
            yy, xx = _cp.meshgrid(fy, fx, indexing="ij")
            radius = _cp.sqrt(xx ** 2 + yy ** 2).astype(_cp.float32)
            if "butter" in mode:
                order = max(1, int(round(float(butterworth_order))))
                transfer = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))
            else:
                transfer = _cp.exp(-0.5 * (radius / cutoff) ** 2)
            fft = _cp.fft.fft2(x)
            filtered_gpu = _cp.real(_cp.fft.ifft2(fft * transfer)).astype(_cp.float32)
            if amount < 1.0:
                filtered_gpu = ((1.0 - amount) * x + amount * filtered_gpu).astype(_cp.float32)
            filtered = _cp.asnumpy(filtered_gpu).astype(np.float32, copy=False)
            return filtered.astype(np.float32), cutoff
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    radius = _frequency_radius_map(img.shape)
    if "butter" in mode:
        order = max(1, int(round(float(butterworth_order))))
        transfer = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))
    else:
        transfer = np.exp(-0.5 * (radius / cutoff) ** 2)

    fft = np.fft.fft2(img)
    filtered = np.real(np.fft.ifft2(fft * transfer)).astype(np.float32)
    if amount < 1.0:
        filtered = ((1.0 - amount) * img + amount * filtered).astype(np.float32)
    return filtered.astype(np.float32), cutoff

def estimate_fourier_os_sim_cutoff_from_fft_diagnostics(phase_diagnostics, fallback_cutoff=0.18):
    """
    Suggest a Fourier-OS-SIM low-pass cutoff from the illumination stripe FFT.

    The demodulated OS-SIM envelope can usually be filtered below a frequency
    comparable to the stripe carrier. This heuristic keeps a conservative band
    of roughly 2 / stripe_period cycles/pixel while clamping to useful limits.
    """
    periods = []
    peak_ratios = []
    for diag in phase_diagnostics or []:
        fft = diag.get("fft_peak", {}) if isinstance(diag, dict) else {}
        period = float(fft.get("period_pixels", 0.0) or 0.0)
        peak_ratio = float(fft.get("peak_to_median", 0.0) or 0.0)
        if np.isfinite(period) and 4.0 <= period <= 512.0 and peak_ratio >= 3.0:
            periods.append(period)
            peak_ratios.append(peak_ratio)

    if not periods:
        return {
            "used": False,
            "reason": "No reliable FFT stripe peak was found; fallback Fourier cutoff was kept.",
            "stripe_period_pixels": None,
            "peak_to_median_median": None,
            "cutoff_cycles_per_pixel": float(fallback_cutoff),
        }

    stripe_period = float(np.median(periods))
    peak_to_median = float(np.median(peak_ratios)) if peak_ratios else None
    cutoff = float(np.clip(2.0 / max(stripe_period, 1e-6), 0.035, 0.30))
    return {
        "used": True,
        "reason": "Estimated from median reliable FFT stripe period.",
        "stripe_period_pixels": stripe_period,
        "peak_to_median_median": peak_to_median,
        "cutoff_cycles_per_pixel": cutoff,
    }


def fourier_os_sim_reconstruct(
    i0,
    i1,
    i2,
    phase_model="0/120/240",
    cutoff_cycles_per_pixel=0.18,
    filter_mode="Gaussian low-pass",
    butterworth_order=2,
    filter_amount=1.0,
):
    """
    Optional Fourier-OS-SIM branch.

    It demodulates the 3 phase frames into quadrature components, filters the
    quadrature components in the Fourier domain, then recomputes the OS-SIM
    modulation amplitude. This is intended for optical-sectioning denoising /
    background-noise suppression, not SR-SIM frequency shifting.

    Returns:
        filtered_os_sim, info_dict
    """
    c, s, spatial_amp = demodulate_three_phase(i0, i1, i2, phase_model=phase_model)
    c_f, actual_cutoff = apply_fourier_frequency_filter(
        c,
        cutoff_cycles_per_pixel=cutoff_cycles_per_pixel,
        filter_mode=filter_mode,
        butterworth_order=butterworth_order,
        amount=filter_amount,
    )
    s_f, _ = apply_fourier_frequency_filter(
        s,
        cutoff_cycles_per_pixel=cutoff_cycles_per_pixel,
        filter_mode=filter_mode,
        butterworth_order=butterworth_order,
        amount=filter_amount,
    )
    amp = np.sqrt(c_f ** 2 + s_f ** 2).astype(np.float32)
    if str(phase_model).lower().replace(" ", "").find("90") >= 0:
        amp = (0.5 * amp).astype(np.float32)
    amp = np.clip(amp, 0, None).astype(np.float32)
    return amp, {
        "mode": "Fourier-OS-SIM filtered demodulation",
        "filter_mode": str(filter_mode),
        "cutoff_cycles_per_pixel": float(actual_cutoff),
        "butterworth_order": int(max(1, round(float(butterworth_order)))),
        "filter_amount": float(np.clip(filter_amount, 0.0, 1.0)),
        "spatial_mean_before_filter": float(np.mean(spatial_amp)),
        "spatial_mean_after_filter": float(np.mean(amp)),
    }


def pseudo_widefield(i0, i90, i180):
    """
    单个方向的 pseudo-widefield (p-WF)：
        pWF = (I0 + I90 + I180) / 3
    """
    if _should_use_gpu_array(i0, i90, i180):
        try:
            x0 = _cp.asarray(i0, dtype=_cp.float32)
            x1 = _cp.asarray(i90, dtype=_cp.float32)
            x2 = _cp.asarray(i180, dtype=_cp.float32)
            out = (x0 + x1 + x2) / 3.0
            return _cp.asnumpy(out).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)
    return ((np.asarray(i0, dtype=np.float32) + np.asarray(i90, dtype=np.float32) + np.asarray(i180, dtype=np.float32)) / 3.0).astype(np.float32)

def fuse_coeffs_maxabs(arr1, arr2):
    """
    逐点比较绝对值，保留绝对值更大的系数。
    """
    return np.where(np.abs(arr1) >= np.abs(arr2), arr1, arr2)


def fuse_coeffs_mean(arr1, arr2):
    """
    更柔和的细节融合：
    直接对高频细节系数取平均，减少浮雕/边缘过冲感。
    """
    return (arr1 + arr2) / 2.0


def fuse_coeffs(arr1, arr2, detail_mode="maxabs"):
    """
    统一高频细节融合接口。
    detail_mode:
        - maxabs: 保留绝对值更大的细节，锐利但更容易有浮雕感
        - mean:   对细节系数求平均，更柔和
    """
    detail_mode = str(detail_mode).lower()
    if detail_mode == "mean":
        return fuse_coeffs_mean(arr1, arr2)
    return fuse_coeffs_maxabs(arr1, arr2)


def dwt_wavelet_fusion(img1, img2, wavelet="db2", level=2, detail_mode="maxabs"):
    """
    DWT 版小波融合：
    - 低频近似系数取平均
    - 高频细节系数取绝对值更大者
    """
    max_level = pywt.dwtn_max_level(img1.shape, wavelet)
    level = max(1, min(level, max_level)) if max_level > 0 else 1

    coeffs1 = pywt.wavedec2(img1, wavelet=wavelet, level=level)
    coeffs2 = pywt.wavedec2(img2, wavelet=wavelet, level=level)

    fused_coeffs = []

    cA1 = coeffs1[0]
    cA2 = coeffs2[0]
    fused_cA = (cA1 + cA2) / 2.0
    fused_coeffs.append(fused_cA)

    for i in range(1, len(coeffs1)):
        cH1, cV1, cD1 = coeffs1[i]
        cH2, cV2, cD2 = coeffs2[i]

        fused_cH = fuse_coeffs(cH1, cH2, detail_mode=detail_mode)
        fused_cV = fuse_coeffs(cV1, cV2, detail_mode=detail_mode)
        fused_cD = fuse_coeffs(cD1, cD2, detail_mode=detail_mode)

        fused_coeffs.append((fused_cH, fused_cV, fused_cD))

    fused = pywt.waverec2(fused_coeffs, wavelet=wavelet)
    fused = fused[:img1.shape[0], :img1.shape[1]]
    return np.clip(fused, 0, None).astype(np.float32), level


def swt_wavelet_fusion(img1, img2, wavelet="db2", level=2, detail_mode="maxabs"):
    """
    SWT 版小波融合（Stationary Wavelet Transform）：
    - 不下采样，shift-invariant，更适合融合
    - 每层 cA 取平均
    - 每层细节系数取绝对值更大者
    """
    max_level = pywt.swt_max_level(min(img1.shape))
    if max_level < 1:
        raise ValueError(
            f"SWT requires image dimensions with sufficient factor-of-2 depth. "
            f"Current ROI shape: {img1.shape}, max SWT level = {max_level}"
        )

    level = max(1, min(level, max_level))

    coeffs1 = pywt.swt2(img1, wavelet=wavelet, level=level)
    coeffs2 = pywt.swt2(img2, wavelet=wavelet, level=level)

    fused_coeffs = []
    for (cA1, (cH1, cV1, cD1)), (cA2, (cH2, cV2, cD2)) in zip(coeffs1, coeffs2):
        fused_cA = (cA1 + cA2) / 2.0
        fused_cH = fuse_coeffs(cH1, cH2, detail_mode=detail_mode)
        fused_cV = fuse_coeffs(cV1, cV2, detail_mode=detail_mode)
        fused_cD = fuse_coeffs(cD1, cD2, detail_mode=detail_mode)
        fused_coeffs.append((fused_cA, (fused_cH, fused_cV, fused_cD)))

    fused = pywt.iswt2(fused_coeffs, wavelet=wavelet)
    fused = fused[:img1.shape[0], :img1.shape[1]]
    return np.clip(fused, 0, None).astype(np.float32), level


def wavelet_fusion(img1, img2, wavelet="db2", level=2, method="DWT", detail_mode="maxabs"):
    """
    统一接口：
    method = "DWT" 或 "SWT"
    返回：
        fused_image, actual_level
    """
    method = method.upper()
    if method == "DWT":
        return dwt_wavelet_fusion(img1, img2, wavelet=wavelet, level=level, detail_mode=detail_mode)
    if method == "SWT":
        return swt_wavelet_fusion(img1, img2, wavelet=wavelet, level=level, detail_mode=detail_mode)
    raise ValueError(f"Unsupported fusion method: {method}")


def fuse_image_sequence(images, wavelet="db2", level=2, method="DWT", detail_mode="maxabs"):
    """
    顺序融合多张图像。
    至少需要 2 张图，返回:
        fused_image, final_actual_level
    """
    if len(images) < 2:
        raise ValueError("At least 2 images are required for wavelet fusion.")

    fused = images[0]
    actual_level = level
    for img in images[1:]:
        fused, actual_level = wavelet_fusion(
            fused, img, wavelet=wavelet, level=level, method=method, detail_mode=detail_mode
        )
    return fused, actual_level


def fuse_sim_orientations(images, mode="RMS"):
    """
    多方向 SIM 结果的最终融合。

    mode:
        - RMS: sqrt(mean(I_i^2))，比简单平均更不容易直接保留单方向残余条纹
        - Quadrature: sqrt(sum(I_i^2))，能量型融合，亮度会更高
        - Mean: arithmetic mean，原始简单平均方式
    """
    if not images:
        raise ValueError("No SIM images to fuse.")

    mode = str(mode).lower()
    if _should_use_gpu_array(*images):
        try:
            stack = _cp.stack([_cp.asarray(img, dtype=_cp.float32) for img in images], axis=0)
            if mode == "quadrature":
                fused = _cp.sqrt(_cp.sum(stack ** 2, axis=0))
            elif mode == "mean":
                fused = _cp.mean(stack, axis=0)
            else:
                fused = _cp.sqrt(_cp.mean(stack ** 2, axis=0))
            fused = _cp.maximum(fused, 0)
            return _cp.asnumpy(fused).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    stack = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)

    if mode == "quadrature":
        fused = np.sqrt(np.sum(stack ** 2, axis=0))
    elif mode == "mean":
        fused = np.mean(stack, axis=0)
    else:
        fused = np.sqrt(np.mean(stack ** 2, axis=0))

    return np.clip(fused, 0, None).astype(np.float32)

def extract_hilo_high_frequency(widefield, sigma=2.0):
    """
    HiLo 的 Hi 分量：从宽场/伪宽场图中提取高频细节。
    high = widefield - GaussianLowPass(widefield)
    """
    widefield = np.asarray(widefield, dtype=np.float32)
    if widefield.ndim != 2:
        raise ValueError(f"HiLo widefield image must be 2D, got shape {widefield.shape}")

    sigma = float(sigma)
    if sigma <= 0:
        return widefield.copy(), np.zeros_like(widefield, dtype=np.float32)

    lowpass = _gaussian_blur_accel(widefield, sigma).astype(np.float32)
    high_signed = (widefield - lowpass).astype(np.float32)
    return high_signed, lowpass

def extract_hilo_mid_high_frequency(widefield, high_sigma=1.0, mid_sigma=3.0):
    """
    将 HiLo 宽场源拆成中频和高频两段，用于更细的结构调节。

    Gaussian low-pass operations use GPU/CuPy for large arrays when available.
    """
    widefield = np.asarray(widefield, dtype=np.float32)
    if widefield.ndim != 2:
        raise ValueError(f"HiLo widefield image must be 2D, got shape {widefield.shape}")

    high_sigma = float(high_sigma)
    mid_sigma = float(mid_sigma)

    if high_sigma <= 0 and mid_sigma <= 0:
        z = np.zeros_like(widefield, dtype=np.float32)
        return z, widefield.copy(), z, z

    if high_sigma <= 0:
        high_lowpass = widefield.astype(np.float32).copy()
        high_signed = np.zeros_like(widefield, dtype=np.float32)
    else:
        high_lowpass = _gaussian_blur_accel(widefield, high_sigma).astype(np.float32)
        high_signed = (widefield - high_lowpass).astype(np.float32)

    if mid_sigma <= 0:
        mid_lowpass = np.zeros_like(widefield, dtype=np.float32)
        mid_signed = high_lowpass.copy().astype(np.float32)
    else:
        if high_sigma > 0 and mid_sigma <= high_sigma:
            mid_sigma = high_sigma * 1.5
        mid_lowpass = _gaussian_blur_accel(widefield, mid_sigma).astype(np.float32)
        mid_signed = (high_lowpass - mid_lowpass).astype(np.float32)

    return (
        mid_signed.astype(np.float32),
        high_signed.astype(np.float32),
        mid_lowpass.astype(np.float32),
        high_lowpass.astype(np.float32),
    )

def extract_hilo_low_frequency(sectioned_image, sigma=2.0):
    """
    HiLo 的 Lo 分量：从 OS-SIM 光学切片图中取低频焦内信息。
    Gaussian low-pass uses GPU/CuPy for large arrays when available.
    """
    sectioned_image = np.asarray(sectioned_image, dtype=np.float32)
    if sectioned_image.ndim != 2:
        raise ValueError(f"HiLo sectioned image must be 2D, got shape {sectioned_image.shape}")

    sigma = float(sigma)
    if sigma <= 0:
        return sectioned_image.copy()

    return _gaussian_blur_accel(sectioned_image, sigma).astype(np.float32)

def create_hilo_hybrid(
    widefield,
    sectioned_image,
    sigma=2.0,
    lo_weight=1.0,
    mid_weight=0.0,
    hi_weight=1.0,
    high_sigma=None,
    mid_sigma=None,
    clip_output=True,
):
    """
    生成 OS-SIM + HiLo 混合结果。

    Lo = LowPass(OS-SIM sectioned image, sigma)
    Mid = Gaussian(Widefield, high_sigma) - Gaussian(Widefield, mid_sigma)
    High = Widefield - Gaussian(Widefield, high_sigma)
    Hybrid = lo_weight * Lo + mid_weight * Mid + hi_weight * High

    当 mid_weight=0 且 high_sigma=None 时，等价于旧版：
    Hybrid = lo_weight * Lo + hi_weight * (Widefield - Gaussian(Widefield, sigma))

    返回:
        low_component,
        high_signed_component,
        high_abs_component,
        widefield_lowpass,
        hybrid,
        mid_signed_component,
        mid_abs_component,
        combined_mid_high_component,
        high_lowpass
    """
    widefield = np.asarray(widefield, dtype=np.float32)
    sectioned_image = np.asarray(sectioned_image, dtype=np.float32)

    if widefield.shape != sectioned_image.shape:
        raise ValueError(
            f"HiLo widefield shape {widefield.shape} does not match sectioned image shape {sectioned_image.shape}"
        )

    sigma = float(sigma)
    if high_sigma is None:
        high_sigma = sigma
    if mid_sigma is None:
        mid_sigma = sigma

    low_component = extract_hilo_low_frequency(sectioned_image, sigma=sigma)
    (
        mid_signed,
        high_signed,
        widefield_lowpass,
        high_lowpass,
    ) = extract_hilo_mid_high_frequency(
        widefield,
        high_sigma=high_sigma,
        mid_sigma=mid_sigma,
    )

    combined_mid_high = float(mid_weight) * mid_signed + float(hi_weight) * high_signed
    hybrid = float(lo_weight) * low_component + combined_mid_high
    if clip_output:
        hybrid = np.clip(hybrid, 0, None)

    return (
        low_component.astype(np.float32),
        high_signed.astype(np.float32),
        np.abs(high_signed).astype(np.float32),
        widefield_lowpass.astype(np.float32),
        hybrid.astype(np.float32),
        mid_signed.astype(np.float32),
        np.abs(mid_signed).astype(np.float32),
        combined_mid_high.astype(np.float32),
        high_lowpass.astype(np.float32),
    )


def _odd_kernel_size(value, minimum=3):
    """Return a positive odd kernel size for local filters."""
    k = int(round(float(value)))
    k = max(int(minimum), k)
    if k % 2 == 0:
        k += 1
    return k


def apply_local_wiener_filter(img, kernel_size=5, noise_variance=0.0, amount=1.0):
    """
    简单局部维纳滤波，用于后处理降噪。

    GPU acceleration:
        The local mean/variance and Wiener gain are computed on CUDA for large
        arrays when CuPy is available. CPU OpenCV blur remains the fallback.

    返回：
        filtered, estimated_noise_variance, actual_kernel_size
    """
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img.copy(), 0.0, _odd_kernel_size(kernel_size)

    amount = float(np.clip(amount, 0.0, 1.0))
    k = _odd_kernel_size(kernel_size)
    noise_variance = float(noise_variance)

    if _should_use_gpu_array(img):
        try:
            x = _cp.asarray(img, dtype=_cp.float32)
            mean = _cupyx_uniform_filter(x, size=(k, k), mode="reflect").astype(_cp.float32)
            mean_sq = _cupyx_uniform_filter((x * x).astype(_cp.float32), size=(k, k), mode="reflect").astype(_cp.float32)
            local_var = mean_sq - mean * mean
            local_var = _cp.maximum(local_var, 0)

            nv = noise_variance
            if nv <= 0 or not np.isfinite(nv):
                valid = local_var[_cp.isfinite(local_var)]
                nv = float(_cp.asnumpy(_cp.median(valid))) if int(valid.size) else 0.0

            eps = 1e-12
            gain = _cp.maximum(local_var - nv, 0.0) / _cp.maximum(local_var, eps)
            filtered_gpu = mean + gain * (x - mean)
            filtered_gpu = _cp.maximum(filtered_gpu, 0).astype(_cp.float32)

            if amount < 1.0:
                filtered_gpu = ((1.0 - amount) * x + amount * filtered_gpu).astype(_cp.float32)
                filtered_gpu = _cp.maximum(filtered_gpu, 0).astype(_cp.float32)

            return _cp.asnumpy(filtered_gpu).astype(np.float32, copy=False), float(nv), int(k)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)

    mean = cv2.blur(img, (k, k)).astype(np.float32)
    mean_sq = cv2.blur((img * img).astype(np.float32), (k, k)).astype(np.float32)
    local_var = mean_sq - mean * mean
    local_var[local_var < 0] = 0

    if noise_variance <= 0 or not np.isfinite(noise_variance):
        valid = local_var[np.isfinite(local_var)]
        noise_variance = float(np.median(valid)) if valid.size else 0.0

    eps = 1e-12
    gain = np.maximum(local_var - noise_variance, 0.0) / np.maximum(local_var, eps)
    filtered = mean + gain * (img - mean)
    filtered = np.clip(filtered, 0, None).astype(np.float32)

    if amount < 1.0:
        filtered = ((1.0 - amount) * img + amount * filtered).astype(np.float32)
        filtered = np.clip(filtered, 0, None).astype(np.float32)

    return filtered, float(noise_variance), int(k)

def estimate_hilo_sigmas_from_fft_diagnostics(
    phase_diagnostics,
    fallback_lo_sigma=2.0,
    fallback_mid_sigma=3.0,
    fallback_high_sigma=1.0,
):
    """
    根据相位组 FFT 条纹峰值给出 HiLo Lo/Mid/High sigma 的稳健建议。

    这不是严格的物理标定，而是把条纹周期转成适合 GUI 初始调参的
    空间域 Gaussian sigma：
        - high_sigma 控制最细高频边缘，通常约为条纹周期的 1/8；
        - mid_sigma 控制中频带宽，通常约为条纹周期的 1/3；
        - lo_sigma 控制 OS-SIM 低频焦内信息，通常约为条纹周期的 1/4。

    若 FFT 峰值不可靠，则返回用户界面中的 fallback 值。
    """
    periods = []
    peak_ratios = []
    for diag in phase_diagnostics or []:
        fft = diag.get("fft_peak", {}) if isinstance(diag, dict) else {}
        period = float(fft.get("period_pixels", 0.0) or 0.0)
        peak_ratio = float(fft.get("peak_to_median", 0.0) or 0.0)
        if np.isfinite(period) and 4.0 <= period <= 512.0 and peak_ratio >= 3.0:
            periods.append(period)
            peak_ratios.append(peak_ratio)

    if not periods:
        return {
            "used": False,
            "reason": "No reliable FFT stripe peak was found; fallback UI sigmas were kept.",
            "stripe_period_pixels": None,
            "peak_to_median_median": None,
            "lo_sigma": float(fallback_lo_sigma),
            "mid_sigma": float(fallback_mid_sigma),
            "high_sigma": float(fallback_high_sigma),
            "lo_cutoff_cycles_per_pixel": float(0.1325 / max(float(fallback_lo_sigma), 1e-6)),
            "mid_cutoff_cycles_per_pixel": float(0.1325 / max(float(fallback_mid_sigma), 1e-6)),
            "high_cutoff_cycles_per_pixel": float(0.1325 / max(float(fallback_high_sigma), 1e-6)),
        }

    stripe_period = float(np.median(periods))
    peak_to_median = float(np.median(peak_ratios)) if peak_ratios else None

    high_sigma = float(np.clip(stripe_period / 8.0, 0.6, 3.0))
    mid_sigma = float(np.clip(stripe_period / 3.0, max(high_sigma * 1.5, 1.2), 15.0))
    lo_sigma = float(np.clip(stripe_period / 4.0, 1.0, 12.0))

    # Gaussian sigma -> approximate -3 dB cutoff frequency.
    # For a Gaussian low-pass exp(-2*pi^2*sigma^2*f^2), f_3db ~= 0.1325/sigma.
    high_cutoff = float(0.1325 / max(high_sigma, 1e-6))
    mid_cutoff = float(0.1325 / max(mid_sigma, 1e-6))
    lo_cutoff = float(0.1325 / max(lo_sigma, 1e-6))

    # 保证 Lo/Mid/High 的尺度顺序稳定：high < mid，Lo 位于中低频。
    if mid_sigma <= high_sigma:
        mid_sigma = high_sigma * 1.5
    if lo_sigma < high_sigma:
        lo_sigma = high_sigma

    return {
        "used": True,
        "reason": "Estimated from median reliable FFT stripe period.",
        "stripe_period_pixels": stripe_period,
        "stripe_frequency_cycles_per_pixel": float(1.0 / max(stripe_period, 1e-6)),
        "peak_to_median_median": peak_to_median,
        "lo_sigma": lo_sigma,
        "mid_sigma": mid_sigma,
        "high_sigma": high_sigma,
        "lo_cutoff_cycles_per_pixel": lo_cutoff,
        "mid_cutoff_cycles_per_pixel": mid_cutoff,
        "high_cutoff_cycles_per_pixel": high_cutoff,
    }



def _robust_normalize_to_uint8(img, low=0.5, high=99.5):
    """Normalize a float image to uint8 and return inverse scaling metadata."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8), 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8), 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not np.isfinite(vmin):
        vmin = float(np.min(finite))
    if not np.isfinite(vmax):
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0 + 0.5).astype(np.uint8), vmin, vmax


def _restore_uint8_to_float(img_u8, vmin, vmax):
    return (np.asarray(img_u8, dtype=np.float32) / 255.0 * (float(vmax) - float(vmin)) + float(vmin)).astype(np.float32)


def estimate_noise_sigma_mad(img):
    """Robust high-frequency noise sigma estimate using a Laplacian MAD."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=3)
    med = float(np.median(lap))
    mad = float(np.median(np.abs(lap - med)))
    sigma = mad / 0.6745 if mad > 0 else float(np.std(lap))
    return float(max(sigma, 0.0))


def apply_nlm_denoise_float(img, h=8.0, template_window=7, search_window=21, amount=0.75):
    """Apply OpenCV non-local means to a float image while preserving its scale."""
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img.copy(), {"method": "NLM", "applied": False, "reason": "empty image"}
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return img.copy(), {"method": "NLM", "applied": False, "reason": "amount <= 0"}
    template_window = _odd_kernel_size(template_window, minimum=3)
    search_window = _odd_kernel_size(search_window, minimum=max(template_window + 2, 5))
    img_u8, vmin, vmax = _robust_normalize_to_uint8(img)
    den_u8 = cv2.fastNlMeansDenoising(
        img_u8,
        None,
        h=float(max(h, 0.0)),
        templateWindowSize=int(template_window),
        searchWindowSize=int(search_window),
    )
    den = _restore_uint8_to_float(den_u8, vmin, vmax)
    out = ((1.0 - amount) * img + amount * den).astype(np.float32)
    out = np.clip(out, 0, None).astype(np.float32)
    return out, {
        "method": "NLM",
        "applied": True,
        "h": float(h),
        "template_window": int(template_window),
        "search_window": int(search_window),
        "amount": float(amount),
        "scale_vmin": float(vmin),
        "scale_vmax": float(vmax),
    }


def apply_wavelet_shrinkage_denoise(img, sigma=0.0, wavelet="db2", level=2, amount=0.75):
    """BM3D-like fallback: multiscale wavelet soft-threshold denoising."""
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        return img.copy(), {"method": "Wavelet shrinkage", "applied": False, "reason": "empty image"}
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return img.copy(), {"method": "Wavelet shrinkage", "applied": False, "reason": "amount <= 0"}
    max_level = pywt.dwtn_max_level(img.shape, wavelet)
    level = max(1, min(int(level), max_level)) if max_level > 0 else 1
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    sigma = float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        detail = coeffs[-1][-1]
        sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745) if detail.size else estimate_noise_sigma_mad(img)
    threshold = float(sigma * np.sqrt(2.0 * np.log(max(img.size, 2))))
    den_coeffs = [coeffs[0]]
    for detail_tuple in coeffs[1:]:
        den_coeffs.append(tuple(pywt.threshold(c, threshold, mode="soft") for c in detail_tuple))
    den = pywt.waverec2(den_coeffs, wavelet=wavelet)
    den = den[:img.shape[0], :img.shape[1]].astype(np.float32)
    out = ((1.0 - amount) * img + amount * den).astype(np.float32)
    out = np.clip(out, 0, None).astype(np.float32)
    return out, {
        "method": "Wavelet shrinkage (BM3D-like fallback)",
        "applied": True,
        "sigma": float(sigma),
        "threshold": float(threshold),
        "wavelet": str(wavelet),
        "level": int(level),
        "amount": float(amount),
    }


def apply_bm3d_or_fallback_denoise(img, sigma=0.0, amount=0.75):
    """Use bm3d package if available; otherwise use wavelet-shrinkage fallback."""
    img = np.asarray(img, dtype=np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    if img.size == 0 or amount <= 0:
        return img.copy(), {"method": "BM3D", "applied": False, "reason": "empty image or amount <= 0"}

    sigma = float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = estimate_noise_sigma_mad(img)

    if _bm3d_package is not None:
        finite = img[np.isfinite(img)]
        vmin = float(np.percentile(finite, 0.5)) if finite.size else 0.0
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
        sigma_norm = float(np.clip(sigma / max(vmax - vmin, 1e-6), 1e-6, 0.5))
        try:
            den_norm = _bm3d_package.bm3d(norm, sigma_psd=sigma_norm)
            den = (np.asarray(den_norm, dtype=np.float32) * (vmax - vmin) + vmin).astype(np.float32)
            out = ((1.0 - amount) * img + amount * den).astype(np.float32)
            out = np.clip(out, 0, None).astype(np.float32)
            return out, {
                "method": "BM3D package",
                "applied": True,
                "sigma": float(sigma),
                "sigma_normalized": float(sigma_norm),
                "amount": float(amount),
            }
        except Exception as e:
            den, info = apply_wavelet_shrinkage_denoise(img, sigma=sigma, amount=amount)
            info["bm3d_package_error"] = str(e)
            return den, info

    den, info = apply_wavelet_shrinkage_denoise(img, sigma=sigma, amount=amount)
    info["bm3d_package_available"] = False
    return den, info


def apply_hilo_special_denoise(
    img,
    method="NLM",
    nlm_h=8.0,
    nlm_template_window=7,
    nlm_search_window=21,
    bm3d_sigma=0.0,
    amount=0.75,
):
    """HiLo-only denoising wrapper for NLM / BM3D-like methods."""
    method_norm = str(method).strip().lower()
    if method_norm in ("none", "off", "disabled"):
        return np.asarray(img, dtype=np.float32).copy(), {"method": method, "applied": False, "reason": "disabled"}
    if "bm3d" in method_norm:
        return apply_bm3d_or_fallback_denoise(img, sigma=bm3d_sigma, amount=amount)
    if "wavelet" in method_norm:
        return apply_wavelet_shrinkage_denoise(img, sigma=bm3d_sigma, amount=amount)
    return apply_nlm_denoise_float(
        img,
        h=nlm_h,
        template_window=nlm_template_window,
        search_window=nlm_search_window,
        amount=amount,
    )

def fuse_signed_image_sequence(images, mode="Mean"):
    """
    融合可能带符号的 HiLo Mid/High 分量。

    RMS/Quadrature 对符号分量会丢失正负方向，因此默认 Mean；
    对最终非负强度图仍使用 fuse_sim_orientations。
    """
    if not images:
        raise ValueError("No images to fuse.")
    mode = str(mode).lower()
    if _should_use_gpu_array(*images):
        try:
            stack = _cp.stack([_cp.asarray(img, dtype=_cp.float32) for img in images], axis=0)
            if mode == "sum":
                out = _cp.sum(stack, axis=0)
            else:
                out = _cp.mean(stack, axis=0)
            return _cp.asnumpy(out).astype(np.float32, copy=False)
        except Exception as exc:
            _gpu_fail_and_fallback(exc)
    stack = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)
    if mode == "sum":
        return np.sum(stack, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.float32)

def normalize_profile(profile, method="minmax"):
    """
    对线剖面做归一化，便于多条曲线放在相近范围内比较。
    当前支持：
        - minmax: 归一化到 0~1
        - max:    按最大值归一化到 0~1
    """
    profile = np.asarray(profile, dtype=np.float32)

    if profile.size == 0:
        return profile

    method = str(method).lower()

    if method == "max":
        vmax = float(np.max(profile))
        if vmax > 1e-12:
            return profile / vmax
        return np.zeros_like(profile, dtype=np.float32)

    vmin = float(np.min(profile))
    vmax = float(np.max(profile))
    if vmax - vmin > 1e-12:
        return (profile - vmin) / (vmax - vmin)
    return np.zeros_like(profile, dtype=np.float32)


def compute_profile_metrics(profile):
    """
    基于线剖面计算简单指标，用于论文风格的定量比较。
    """
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size == 0:
        return {
            "peak": 0.0,
            "background": 0.0,
            "contrast_michelson": 0.0,
            "cnr_like": 0.0,
        }

    peak = float(np.max(profile))
    background = float(np.min(profile))
    denom = peak + background
    contrast_michelson = (peak - background) / denom if abs(denom) > 1e-12 else 0.0

    profile_std = float(np.std(profile))
    cnr_like = (peak - background) / profile_std if profile_std > 1e-12 else 0.0

    return {
        "peak": peak,
        "background": background,
        "contrast_michelson": contrast_michelson,
        "cnr_like": cnr_like,
    }


def format_metrics_text(metrics_dict):
    lines = []
    for name, metrics in metrics_dict.items():
        lines.append(
            f"{name}: "
            f"Peak={metrics['peak']:.4f}, "
            f"Background={metrics['background']:.4f}, "
            f"Michelson Contrast={metrics['contrast_michelson']:.4f}, "
            f"CNR-like={metrics['cnr_like']:.4f}"
        )
    return "\n".join(lines)


def sample_line_profile(img, p0, p1, num=None):
    """
    在图像上沿直线采样强度线剖面。
    使用双线性插值。
    p0, p1: (x, y)
    返回:
        distance, profile
    """
    x0, y0 = p0
    x1, y1 = p1

    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([float(img[int(round(y0)), int(round(x0))])], dtype=np.float32)

    if num is None:
        num = max(2, int(np.ceil(length)) + 1)

    xs = np.linspace(x0, x1, num).astype(np.float32)
    ys = np.linspace(y0, y1, num).astype(np.float32)

    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    x0i = np.floor(xs).astype(np.int32)
    x1i = np.clip(x0i + 1, 0, w - 1)
    y0i = np.floor(ys).astype(np.int32)
    y1i = np.clip(y0i + 1, 0, h - 1)

    dx = xs - x0i
    dy = ys - y0i

    Ia = img[y0i, x0i]
    Ib = img[y0i, x1i]
    Ic = img[y1i, x0i]
    Id = img[y1i, x1i]

    profile = (
        Ia * (1 - dx) * (1 - dy) +
        Ib * dx * (1 - dy) +
        Ic * (1 - dx) * dy +
        Id * dx * dy
    ).astype(np.float32)

    distance = np.linspace(0, length, num).astype(np.float32)
    return distance, profile


# =========================================================
# Matplotlib 画布
# =========================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(10, 7), dpi=100):
        self.figure = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)


# =========================================================
# 线剖面对比窗口
# =========================================================

class ProfileComparisonWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile Window")
        self.resize(1300, 980)

        self.images = {}
        self.current_line = None
        self.press_xy = None
        self.release_xy = None
        self.display_config = {
            "enabled": False,
            "colormap": "gray",
            "auto_contrast": True,
            "auto_low": 1.0,
            "auto_high": 99.0,
            "manual_min": 0.0,
            "manual_max": 1.0,
        }

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.info_label = QLabel(
            "Draw a line on any image panel below to compare line profiles. "
            "HiLo Hybrid will be included when HiLo mode is enabled and computed."
        )
        self.info_label.setWordWrap(True)

        control_row = QHBoxLayout()

        self.comparison_mode_combo = QComboBox()
        self.comparison_mode_combo.addItems([
            "Main comparison: Widefield / SIM Wavelet / Final SIM / HiLo Hybrid",
            "Add raw widefield curve",
        ])
        self.comparison_mode_combo.currentIndexChanged.connect(self.redraw)

        self.show_raw_widefield_check = QCheckBox("Show raw widefield")
        self.show_raw_widefield_check.setChecked(False)
        self.show_raw_widefield_check.toggled.connect(self.redraw)

        self.show_hilo_check = QCheckBox("Show HiLo Hybrid")
        self.show_hilo_check.setChecked(True)
        self.show_hilo_check.toggled.connect(self.redraw)

        self.normalize_profiles_check = QCheckBox("Normalize profiles")
        self.normalize_profiles_check.setChecked(True)
        self.normalize_profiles_check.toggled.connect(self.redraw)

        self.profile_norm_method_combo = QComboBox()
        self.profile_norm_method_combo.addItems(["Min-Max (0-1)", "Max only (0-1)"])
        self.profile_norm_method_combo.currentIndexChanged.connect(self.redraw)

        control_row.addWidget(QLabel("Comparison"))
        control_row.addWidget(self.comparison_mode_combo)
        control_row.addSpacing(12)
        control_row.addWidget(self.show_raw_widefield_check)
        control_row.addWidget(self.show_hilo_check)
        control_row.addSpacing(12)
        control_row.addWidget(self.normalize_profiles_check)
        control_row.addWidget(QLabel("Method"))
        control_row.addWidget(self.profile_norm_method_combo)
        control_row.addStretch()

        self.content_splitter = QSplitter(Qt.Vertical)

        self.canvas = MplCanvas(self, figsize=(12, 8), dpi=100)

        metrics_container = QWidget()
        metrics_layout = QVBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        self.metrics_box = QTextEdit()
        self.metrics_box.setReadOnly(True)
        self.metrics_box.setMinimumHeight(140)
        self.metrics_box.setPlaceholderText("Quantitative metrics will appear here after drawing a line.")

        metrics_layout.addWidget(QLabel("Quantitative metrics"))
        metrics_layout.addWidget(self.metrics_box)

        self.content_splitter.addWidget(self.canvas)
        self.content_splitter.addWidget(metrics_container)
        self.content_splitter.setSizes([700, 220])

        layout.addWidget(self.info_label)
        layout.addLayout(control_row)
        layout.addWidget(self.content_splitter)

        self.image_axes = []
        self.ax_profile = None
        self.line_artists = []

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def get_ui_state(self):
        return {
            "comparison_mode": self.comparison_mode_combo.currentText(),
            "show_raw_widefield": self.show_raw_widefield_check.isChecked(),
            "show_hilo": self.show_hilo_check.isChecked(),
            "normalize_profiles": self.normalize_profiles_check.isChecked(),
            "profile_norm_method": self.profile_norm_method_combo.currentText(),
            "content_splitter_sizes": self.content_splitter.sizes(),
        }

    def apply_ui_state(self, state):
        if not state:
            return
        comparison_mode = state.get("comparison_mode")
        if comparison_mode:
            idx = self.comparison_mode_combo.findText(comparison_mode)
            if idx >= 0:
                self.comparison_mode_combo.setCurrentIndex(idx)

        self.show_raw_widefield_check.setChecked(
            bool(state.get("show_raw_widefield", self.show_raw_widefield_check.isChecked()))
        )
        self.show_hilo_check.setChecked(
            bool(state.get("show_hilo", self.show_hilo_check.isChecked()))
        )
        self.normalize_profiles_check.setChecked(
            bool(state.get("normalize_profiles", self.normalize_profiles_check.isChecked()))
        )

        profile_norm_method = state.get("profile_norm_method")
        if profile_norm_method:
            idx = self.profile_norm_method_combo.findText(profile_norm_method)
            if idx >= 0:
                self.profile_norm_method_combo.setCurrentIndex(idx)

        splitter_sizes = state.get("content_splitter_sizes")
        if splitter_sizes and hasattr(self, "content_splitter"):
            try:
                self.content_splitter.setSizes([int(v) for v in splitter_sizes])
            except Exception:
                pass

        if self.images:
            self.redraw()

    def set_display_config(
        self,
        enabled=False,
        colormap="gray",
        auto_contrast=True,
        auto_low=1.0,
        auto_high=99.0,
        manual_min=0.0,
        manual_max=1.0,
    ):
        self.display_config = {
            "enabled": bool(enabled),
            "colormap": str(colormap),
            "auto_contrast": bool(auto_contrast),
            "auto_low": float(auto_low),
            "auto_high": float(auto_high),
            "manual_min": float(manual_min),
            "manual_max": float(manual_max),
        }
        if self.images:
            self.redraw()

    def _get_profile_window_display_params(self, img):
        if self.display_config.get("enabled", False):
            return prepare_display_image(
                img,
                auto_contrast=self.display_config.get("auto_contrast", True),
                low=self.display_config.get("auto_low", 1.0),
                high=self.display_config.get("auto_high", 99.0),
                manual_min=self.display_config.get("manual_min", 0.0),
                manual_max=self.display_config.get("manual_max", 1.0),
            )
        disp_img = normalize_for_display(img)
        return disp_img, 0.0, 1.0

    def set_images(
        self,
        widefield_img,
        sim_wavelet_img,
        final_sim_img,
        raw_widefield_img=None,
        fusion_method_label="Final SIM Fused",
        hilo_hybrid_img=None,
    ):
        self.images = {
            "WFused": widefield_img,
            "SIMWavelet": sim_wavelet_img,
            "FinalSIM": final_sim_img,
            "RawWF": raw_widefield_img,
            "HiLoHybrid": hilo_hybrid_img,
            "FinalSIM_label": fusion_method_label,
            "HiLo_label": "HiLo Hybrid",
        }
        self.current_line = None
        self.press_xy = None
        self.release_xy = None
        if hasattr(self, "metrics_box"):
            self.metrics_box.clear()
        self.redraw()

    def clear_line_artists(self):
        for line in self.line_artists:
            try:
                line.remove()
            except Exception:
                pass
        self.line_artists = []

    def _build_profile_image_panels(self):
        panels = []
        has_raw_wf = bool(self.images) and (self.images.get("RawWF") is not None)
        has_hilo = (
            bool(self.images)
            and self.show_hilo_check.isChecked()
            and (self.images.get("HiLoHybrid") is not None)
        )

        if has_raw_wf and self.show_raw_widefield_check.isChecked():
            panels.append(("Raw Widefield", self.images["RawWF"]))

        if self.images:
            panels.extend([
                ("Widefield Fused", self.images["WFused"]),
                ("SIM Wavelet Fused", self.images["SIMWavelet"]),
                (self.images.get("FinalSIM_label", "Final SIM Fused"), self.images["FinalSIM"]),
            ])

        if has_hilo:
            panels.append(("HiLo Hybrid", self.images["HiLoHybrid"]))

        return panels

    def redraw(self):
        self.canvas.figure.clear()

        panels = self._build_profile_image_panels()
        img_cols = max(1, len(panels))
        gs = self.canvas.figure.add_gridspec(2, img_cols, height_ratios=[3, 2])

        self.image_axes = [self.canvas.figure.add_subplot(gs[0, i]) for i in range(img_cols)]
        self.ax_profile = self.canvas.figure.add_subplot(gs[1, :])

        if panels:
            cmap_name = self.display_config.get("colormap", "gray") if self.display_config.get("enabled", False) else "gray"
            for ax, (title, img) in zip(self.image_axes, panels):
                disp_img, vmin, vmax = self._get_profile_window_display_params(img)
                ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")
        else:
            self.image_axes[0].text(
                0.5, 0.5,
                "Run reconstruction first.",
                ha="center", va="center", fontsize=12
            )
            self.image_axes[0].axis("off")

        self.ax_profile.set_title("Line Profile Comparison")
        self.ax_profile.set_xlabel("Distance (pixels)")
        self.ax_profile.set_ylabel("Intensity")
        self.ax_profile.grid(True, alpha=0.3)

        self.clear_line_artists()

        if self.current_line is not None and panels:
            self.draw_line_on_images(*self.current_line)
            self.update_profiles(*self.current_line)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def get_valid_axes(self):
        return list(self.image_axes)

    def on_press(self, event):
        if not self.images:
            return
        if event.inaxes not in self.get_valid_axes():
            return
        if event.xdata is None or event.ydata is None:
            return
        self.press_xy = (float(event.xdata), float(event.ydata))

    def on_release(self, event):
        if not self.images or self.press_xy is None:
            return
        if event.inaxes not in self.get_valid_axes():
            self.press_xy = None
            return
        if event.xdata is None or event.ydata is None:
            self.press_xy = None
            return

        self.release_xy = (float(event.xdata), float(event.ydata))
        self.current_line = (self.press_xy, self.release_xy)
        self.press_xy = None

        self.redraw()

    def draw_line_on_images(self, p0, p1):
        self.clear_line_artists()
        for ax in self.get_valid_axes():
            line, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=1.8)
            self.line_artists.append(line)

    def _normalize_profile_if_needed(self, profile):
        if not self.normalize_profiles_check.isChecked():
            return profile
        norm_text = self.profile_norm_method_combo.currentText()
        norm_method = "max" if norm_text.startswith("Max only") else "minmax"
        return normalize_profile(profile, method=norm_method)

    def update_profiles(self, p0, p1):
        self.ax_profile.clear()
        self.ax_profile.set_title("Line Profile Comparison")
        self.ax_profile.set_xlabel("Distance (pixels)")
        self.ax_profile.grid(True, alpha=0.3)

        curves = []

        dist_wf, prof_wf_raw = sample_line_profile(self.images["WFused"], p0, p1)
        curves.append(("Widefield Fused", dist_wf, prof_wf_raw, {"linewidth": 2.0, "color": "tab:blue"}))

        dist_simw, prof_simw_raw = sample_line_profile(self.images["SIMWavelet"], p0, p1)
        curves.append(("SIM Wavelet Fused", dist_simw, prof_simw_raw, {"linewidth": 2.0, "color": "tab:orange"}))

        final_label = self.images.get("FinalSIM_label", "Final SIM Fused")
        dist_final, prof_final_raw = sample_line_profile(self.images["FinalSIM"], p0, p1)
        curves.append((final_label, dist_final, prof_final_raw, {"linewidth": 2.0, "color": "tab:green"}))

        show_hilo = self.show_hilo_check.isChecked() and self.images.get("HiLoHybrid") is not None
        if show_hilo:
            hilo_label = self.images.get("HiLo_label", "HiLo Hybrid")
            dist_hilo, prof_hilo_raw = sample_line_profile(self.images["HiLoHybrid"], p0, p1)
            curves.append((hilo_label, dist_hilo, prof_hilo_raw, {"linewidth": 2.0, "color": "tab:purple"}))

        show_raw_wf = self.show_raw_widefield_check.isChecked() and self.images.get("RawWF") is not None
        if show_raw_wf:
            dist_rawwf, prof_rawwf_raw = sample_line_profile(self.images["RawWF"], p0, p1)
            curves.append(("Raw Widefield", dist_rawwf, prof_rawwf_raw, {"linewidth": 1.8, "linestyle": "--", "color": "tab:red"}))

        metrics_dict = {}
        for label, _, raw_profile, _ in curves:
            metrics_dict[label] = compute_profile_metrics(raw_profile)

        if self.normalize_profiles_check.isChecked():
            self.ax_profile.set_ylabel("Normalized Intensity")
        else:
            self.ax_profile.set_ylabel("Intensity")

        for label, dist, raw_profile, style in curves:
            profile = self._normalize_profile_if_needed(raw_profile.copy())
            self.ax_profile.plot(dist, profile, label=label, **style)

        self.ax_profile.legend()

        metric_text = [
            "Quantitative summary (computed from raw, non-normalized profiles):",
            format_metrics_text(metrics_dict),
            "",
            "Notes:",
            "- Draw a line on any image panel to update all curves.",
            "- HiLo Hybrid is included when HiLo mode is enabled and computed.",
            "- Enable 'Show raw widefield' to add the raw widefield image curve.",
            "- Michelson Contrast = (peak - background) / (peak + background)",
            "- CNR-like = (peak - background) / std(profile)",
        ]
        self.metrics_box.setPlainText("\n".join(metric_text))

    def save_current_view(self, output_path):
        self.canvas.figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )


# =========================================================
# Background processing worker
# =========================================================

def _supported_image_patterns(input_dir, input_type):
    mode = str(input_type or "all supported")
    if mode == "jpg/jpeg":
        patterns = [
            os.path.join(input_dir, "*.jpg"),
            os.path.join(input_dir, "*.jpeg"),
            os.path.join(input_dir, "*.JPG"),
            os.path.join(input_dir, "*.JPEG"),
        ]
    elif mode == "png":
        patterns = [
            os.path.join(input_dir, "*.png"),
            os.path.join(input_dir, "*.PNG"),
        ]
    elif mode == "tif/tiff":
        patterns = [
            os.path.join(input_dir, "*.tif"),
            os.path.join(input_dir, "*.tiff"),
            os.path.join(input_dir, "*.TIF"),
            os.path.join(input_dir, "*.TIFF"),
        ]
    else:
        patterns = [
            os.path.join(input_dir, "*.jpg"),
            os.path.join(input_dir, "*.jpeg"),
            os.path.join(input_dir, "*.JPG"),
            os.path.join(input_dir, "*.JPEG"),
            os.path.join(input_dir, "*.png"),
            os.path.join(input_dir, "*.PNG"),
            os.path.join(input_dir, "*.tif"),
            os.path.join(input_dir, "*.tiff"),
            os.path.join(input_dir, "*.TIF"),
            os.path.join(input_dir, "*.TIFF"),
        ]
    return patterns


def _collect_image_paths_from_config(input_dir, input_type):
    input_dir = str(input_dir or "").strip()
    if not input_dir:
        raise ValueError("Input folder is empty.")
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input folder does not exist: {input_dir}")

    image_paths = []
    for pattern in _supported_image_patterns(input_dir, input_type):
        image_paths.extend(glob.glob(pattern))

    image_paths = sorted(os.path.normpath(p) for p in set(image_paths))
    if len(image_paths) not in (6, 9):
        raise ValueError(
            f"Expected 6 or 9 images in the input folder, but found {len(image_paths)}."
        )
    return image_paths


def _collect_input_items_from_config(config):
    single_tiff_path = str(config.get("single_tiff_path", "") or "").strip()
    if single_tiff_path:
        if not single_tiff_path.lower().endswith((".tif", ".tiff")):
            raise ValueError("Single TIFF stack input must be a .tif or .tiff file.")
        pages = load_multipage_tiff(single_tiff_path)
        if len(pages) not in (6, 9):
            raise ValueError(
                f"Expected a 6-page or 9-page TIFF stack, but found {len(pages)} pages."
            )
        items = [
            {
                "source": single_tiff_path,
                "page_index": i,
                "image": page,
                "name": f"{os.path.basename(single_tiff_path)} page {i + 1}",
            }
            for i, page in enumerate(pages)
        ]
        return items, "single_tiff_stack"

    return _collect_image_paths_from_config(
        config.get("input_dir", ""),
        config.get("input_type", "all supported"),
    ), "folder_images"


def _load_input_item_image(item):
    if isinstance(item, dict):
        return np.asarray(item["image"], dtype=np.float32)
    return load_image(item)


def _input_item_display_name(item):
    if isinstance(item, dict):
        return str(item.get("name", f"page {int(item.get('page_index', 0)) + 1}"))
    return os.path.basename(str(item))


def _input_item_source_name(item):
    if isinstance(item, dict):
        return f"{os.path.basename(item.get('source', 'stack.tif'))}#page{int(item.get('page_index', 0)) + 1}"
    return os.path.basename(str(item))


def _load_optional_correction_image(path, name, full_shape, roi_x, roi_y, roi_w, roi_h):
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise ValueError(f"{name} image does not exist: {path}")
    img = load_image(path)
    roi_shape = (int(roi_h), int(roi_w))
    if img.shape == tuple(full_shape):
        img = crop_roi(img, roi_x, roi_y, roi_w, roi_h)
    elif img.shape == roi_shape:
        img = img.astype(np.float32)
    else:
        raise ValueError(
            f"{name} image shape {img.shape} must match either full input shape {tuple(full_shape)} "
            f"or ROI shape {roi_shape}."
        )
    return np.asarray(img, dtype=np.float32)


def _load_optional_raw_widefield(path, roi_enabled, roi_x, roi_y, roi_w, roi_h):
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise ValueError(f"Raw widefield image does not exist: {path}")

    img = load_image(path)
    if roi_enabled:
        img = crop_roi(img, roi_x, roi_y, roi_w, roi_h)
    else:
        img = np.asarray(img, dtype=np.float32)
    return img


def run_sim_processing_core(config, progress_cb=None, log_cb=None):
    def log(message):
        if log_cb is not None:
            log_cb(str(message))

    def progress(message, value=None):
        if progress_cb is not None:
            progress_cb(str(message), int(value) if value is not None else -1)

    progress("Collecting input images/pages...", 5)

    input_items_original, input_source_type = _collect_input_items_from_config(config)
    image_order_mode = config.get("image_order_mode") or "orientation_major"
    image_order_text = config.get("image_order_text") or str(image_order_mode)
    input_items = reorder_stripe_sequence(input_items_original, order_mode=image_order_mode)

    single_tiff_path = config.get("single_tiff_path", "")
    raw_widefield_path = config.get("raw_widefield_path", "")
    dark_image_path = config.get("dark_image_path", "")
    flat_image_path = config.get("flat_image_path", "")

    log(f"Input source: {input_source_type}")
    if input_source_type == "single_tiff_stack":
        log(f"Single TIFF stack: {single_tiff_path}")
    log(f"Image/page order mode: {image_order_text}")
    if input_items != input_items_original:
        log("Input order adjusted before grouping orientations/phases.")
    for idx, item in enumerate(input_items, 1):
        log(f"{idx}: {_input_item_display_name(item)}")

    roi_enabled = bool(config.get("roi_enabled", True))
    roi_x = int(config.get("roi_x", 0))
    roi_y = int(config.get("roi_y", 0))
    roi_w = int(config.get("roi_w", 0))
    roi_h = int(config.get("roi_h", 0))

    enable_preprocess = bool(config.get("enable_preprocess", True))
    phase_model = config.get("phase_model", "0/120/240")
    preprocess_mode = config.get("preprocess_mode", "shared_percentile")
    bg_percentile = float(config.get("bg_percentile", 5.0))
    phase_norm_mode = config.get("phase_norm_mode", "per_orientation")
    phase_norm_stat = config.get("phase_norm_stat", "mean")
    os_sim_recon_mode = config.get("os_sim_recon_mode", "Spatial demodulation (standard OS-SIM)")
    fourier_filter_mode = config.get("fourier_filter_mode", "Gaussian low-pass")
    fourier_cutoff_input = float(config.get("fourier_cutoff_input", 0.0))
    fourier_order = int(config.get("fourier_order", 2))
    fourier_amount = float(config.get("fourier_amount", 1.0))

    fusion_method = config.get("fusion_method", "SWT")
    wavelet_name = config.get("wavelet_name", "db2")
    wavelet_level = int(config.get("wavelet_level", 2))
    wavelet_detail_mode = config.get("wavelet_detail_mode", "maxabs")
    enable_orientation_match = bool(config.get("enable_orientation_match", True))
    orientation_match_method = config.get("orientation_match_method", "median")
    final_fusion_mode = config.get("final_fusion_mode", "RMS")

    sigma = float(config.get("gaussian_sigma", 1.0))
    enable_smoothing = bool(config.get("enable_smoothing", True))
    enable_wiener_filter = bool(config.get("enable_wiener_filter", False))
    wiener_target = config.get("wiener_target", "Final SIM only")
    wiener_kernel = int(config.get("wiener_kernel", 5))
    wiener_noise_variance = float(config.get("wiener_noise_variance", 0.0))
    wiener_amount = float(config.get("wiener_amount", 1.0))

    enable_output_intensity_match = bool(config.get("enable_output_intensity_match", True))
    output_intensity_target = config.get("output_intensity_target", "Final SIM + Wavelet fused + HiLo Hybrid")
    output_intensity_reference = config.get("output_intensity_reference", "Widefield fused")
    output_intensity_method = config.get("output_intensity_method", "P95")
    output_intensity_strength = float(config.get("output_intensity_strength", 0.85))
    output_intensity_max_scale = float(config.get("output_intensity_max_scale", 5.0))

    enable_sim_bg_correction = bool(config.get("enable_sim_bg_correction", True))
    sim_bg_mode = config.get("sim_bg_mode", "subtract")
    sim_bg_sigma = float(config.get("sim_bg_sigma", 25.0))

    enable_final_structure_enhance = bool(config.get("enable_final_structure_enhance", True))
    final_structure_target = config.get("final_structure_target", "Final SIM only")
    final_structure_mode = config.get("final_structure_mode", "highpass_mix")
    final_structure_sigma = float(config.get("final_structure_sigma", 35.0))
    final_structure_amount = float(config.get("final_structure_amount", 1.0))
    final_structure_keep_original = float(config.get("final_structure_keep_original", 0.15))

    enable_hilo_mode = bool(config.get("enable_hilo_mode", True))
    hilo_strategy = config.get("hilo_strategy", "Per-orientation OS-SIM/pWF then RMS fuse (recommended)")
    hilo_auto_sigma = bool(config.get("hilo_auto_sigma", True))
    hilo_widefield_source = config.get("hilo_widefield_source", "Pseudo-widefield fused")
    hilo_low_source = config.get("hilo_low_source", "Final SIM fused")
    hilo_sigma = float(config.get("hilo_sigma", 2.0))
    hilo_mid_sigma = float(config.get("hilo_mid_sigma", 3.0))
    hilo_high_sigma = float(config.get("hilo_high_sigma", 1.0))
    hilo_lo_weight = float(config.get("hilo_lo_weight", 1.0))
    hilo_mid_weight = float(config.get("hilo_mid_weight", 0.50))
    hilo_hi_weight = float(config.get("hilo_hi_weight", 1.0))
    hilo_clip_output = bool(config.get("hilo_clip_output", True))

    enable_hilo_denoise = bool(config.get("enable_hilo_denoise", False))
    hilo_denoise_method = config.get("hilo_denoise_method", "NLM")
    hilo_nlm_h = float(config.get("hilo_nlm_h", 8.0))
    hilo_nlm_template_window = int(config.get("hilo_nlm_template_window", 7))
    hilo_nlm_search_window = int(config.get("hilo_nlm_search_window", 21))
    hilo_bm3d_sigma = float(config.get("hilo_bm3d_sigma", 0.0))
    hilo_denoise_amount = float(config.get("hilo_denoise_amount", 0.60))

    orientation_count = len(input_items) // 3
    if orientation_count == 2:
        default_labels = ["Vertical", "Horizontal"]
    else:
        default_labels = ["0deg", "120deg", "240deg"]

    orientation_labels = default_labels[:orientation_count]
    if len(orientation_labels) < orientation_count:
        orientation_labels.extend(
            [f"Orientation {i + 1}" for i in range(len(orientation_labels), orientation_count)]
        )

    log(f"Detected {len(input_items)} images/pages -> {orientation_count} orientations x 3 phases.")
    progress("Loading and checking input image data...", 12)
    log(f"Phase model: {phase_model}")
    log("Internal grouping after order adjustment: each orientation uses 3 consecutive phase images.")
    if orientation_count == 2:
        log("Using 6-image mode: first 3 = vertical stripes, last 3 = horizontal stripes.")
    else:
        log("Using 9-image three-orientation mode (0deg, 120deg, 240deg).")

    raw_images = [_load_input_item_image(item) for item in input_items]
    all_shapes = [img.shape for img in raw_images]
    if len(set(all_shapes)) != 1:
        raise ValueError(f"All images must have the same size, got: {all_shapes}")

    input_h, input_w = raw_images[0].shape[:2]
    log(f"Input image size: {input_w} x {input_h}")
    progress("Applying ROI and preprocessing settings...", 22)

    if roi_enabled:
        log(f"ROI enabled: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
        log("Cropping ROI...")
        cropped_images = [crop_roi(img, roi_x, roi_y, roi_w, roi_h) for img in raw_images]
    else:
        roi_x, roi_y, roi_w, roi_h = 0, 0, int(input_w), int(input_h)
        log("ROI disabled. Using the full input image for processing.")
        cropped_images = [np.asarray(img, dtype=np.float32).copy() for img in raw_images]

    raw_cropped_images = [img.copy() for img in cropped_images]

    raw_orientation_groups = []
    for ori_idx in range(orientation_count):
        raw_group = cropped_images[ori_idx * 3:(ori_idx + 1) * 3]
        raw_orientation_groups.append(raw_group)

    raw_widefield_images = [
        pseudo_widefield(group[0], group[1], group[2])
        for group in raw_orientation_groups
    ]
    raw_widefield_fused = np.mean(np.stack(raw_widefield_images, axis=0), axis=0).astype(np.float32)

    dark_roi = None
    flat_roi = None
    if enable_preprocess:
        log(f"Preprocess enabled. mode={preprocess_mode}")
        full_shape = raw_images[0].shape
        dark_roi = _load_optional_correction_image(
            dark_image_path, "Dark", full_shape, roi_x, roi_y, roi_w, roi_h
        )
        flat_roi = _load_optional_correction_image(
            flat_image_path, "Flat", full_shape, roi_x, roi_y, roi_w, roi_h
        )
        if dark_roi is not None:
            log(f"Loaded dark image ROI: mean={float(np.mean(dark_roi)):.3f}, max={float(np.max(dark_roi)):.3f}")
        if flat_roi is not None:
            log(f"Loaded flat image ROI: mean={float(np.mean(flat_roi)):.3f}, max={float(np.max(flat_roi)):.3f}")

        processed_orientation_groups = []
        preprocess_infos = []
        for ori_idx, raw_group in enumerate(raw_orientation_groups, 1):
            proc_group, info = preprocess_phase_group(
                raw_group,
                background_percentile=bg_percentile,
                mode=preprocess_mode,
                dark_image=dark_roi,
                flat_image=flat_roi,
            )
            processed_orientation_groups.append(proc_group)
            preprocess_infos.append(info)
            if info.get("mode") == "shared_percentile":
                log(
                    f"Orientation {ori_idx} shared background = {info.get('shared_background', 0):.3f} "
                    f"from per-phase percentiles "
                    + ", ".join(f"{v:.3f}" for v in info.get("per_phase_percentile_backgrounds", []))
                )
            else:
                log(f"Orientation {ori_idx} preprocess info: {info}")
        processed_images = [img for group in processed_orientation_groups for img in group]
    else:
        log("Preprocess disabled. Using current processing images.")
        processed_orientation_groups = [[img.copy() for img in group] for group in raw_orientation_groups]
        preprocess_infos = [
            {"mode": "disabled", "background_percentile": float(bg_percentile), "shared_background": 0.0}
            for _ in raw_orientation_groups
        ]
        processed_images = [img for group in processed_orientation_groups for img in group]

    progress("Loading optional reference images and preparing groups...", 34)

    external_raw_widefield = None
    raw_widefield_path = raw_widefield_path
    if raw_widefield_path:
        log(f"Loading external raw widefield: {os.path.basename(raw_widefield_path)}")
        external_raw_widefield = _load_optional_raw_widefield(raw_widefield_path, roi_enabled, roi_x, roi_y, roi_w, roi_h)

    orientation_groups = []
    for ori_idx in range(orientation_count):
        group = processed_images[ori_idx * 3:(ori_idx + 1) * 3]
        if len(group) != 3:
            raise ValueError(f"Orientation group {ori_idx + 1} does not contain exactly 3 phase images.")
        orientation_groups.append(group)

    progress("Computing phase diagnostics...", 40)

    raw_phase_diagnostics = []
    processed_phase_diagnostics = []
    for ori_idx, (raw_group, proc_group) in enumerate(zip(raw_orientation_groups, orientation_groups), 1):
        raw_diag = compute_phase_group_diagnostics(raw_group, phase_model=phase_model)
        proc_diag = compute_phase_group_diagnostics(proc_group, phase_model=phase_model)
        raw_phase_diagnostics.append(raw_diag)
        processed_phase_diagnostics.append(proc_diag)
        label = orientation_labels[ori_idx - 1] if ori_idx - 1 < len(orientation_labels) else f"Orientation {ori_idx}"
        log("[Raw diagnostics] " + format_phase_group_diagnostics(label, raw_diag))
        log("[Processed diagnostics] " + format_phase_group_diagnostics(label, proc_diag))

    hilo_auto_sigma_info = estimate_hilo_sigmas_from_fft_diagnostics(
        processed_phase_diagnostics,
        fallback_lo_sigma=hilo_sigma,
        fallback_mid_sigma=hilo_mid_sigma,
        fallback_high_sigma=hilo_high_sigma,
    )
    fourier_cutoff_info = estimate_fourier_os_sim_cutoff_from_fft_diagnostics(
        processed_phase_diagnostics,
        fallback_cutoff=(fourier_cutoff_input if fourier_cutoff_input > 0 else 0.18),
    )
    fourier_actual_cutoff = float(fourier_cutoff_input)
    if fourier_actual_cutoff <= 0:
        fourier_actual_cutoff = float(fourier_cutoff_info.get("cutoff_cycles_per_pixel", 0.18))

    if hilo_auto_sigma_info.get("used", False):
        log(
            "Recommended HiLo cutoff/sigma from processed phase FFT: "
            f"stripe_period={hilo_auto_sigma_info.get('stripe_period_pixels'):.2f} px, "
            f"stripe_freq={hilo_auto_sigma_info.get('stripe_frequency_cycles_per_pixel'):.5f} cyc/pix, "
            f"peak/median={hilo_auto_sigma_info.get('peak_to_median_median'):.2f}, "
            f"lo_sigma={hilo_auto_sigma_info.get('lo_sigma'):.2f}, "
            f"mid_sigma={hilo_auto_sigma_info.get('mid_sigma'):.2f}, "
            f"high_sigma={hilo_auto_sigma_info.get('high_sigma'):.2f}, "
            f"lo/mid/high cutoff={hilo_auto_sigma_info.get('lo_cutoff_cycles_per_pixel'):.5f}/"
            f"{hilo_auto_sigma_info.get('mid_cutoff_cycles_per_pixel'):.5f}/"
            f"{hilo_auto_sigma_info.get('high_cutoff_cycles_per_pixel'):.5f} cyc/pix"
        )
    else:
        log(
            "HiLo cutoff/sigma recommendation unavailable; keeping manual values. "
            f"Reason: {hilo_auto_sigma_info.get('reason', '')}"
        )

    if enable_hilo_mode and hilo_auto_sigma and hilo_auto_sigma_info.get("used", False):
        hilo_sigma = float(hilo_auto_sigma_info["lo_sigma"])
        hilo_mid_sigma = float(hilo_auto_sigma_info["mid_sigma"])
        hilo_high_sigma = float(hilo_auto_sigma_info["high_sigma"])
        log("Applied recommended HiLo Lo/Mid/High sigmas to this run.")
    elif enable_hilo_mode and hilo_auto_sigma:
        log("Auto HiLo sigma application requested, but recommendation was unreliable; manual values were used.")

    if "fourier" in os_sim_recon_mode.lower():
        log(
            "Fourier-OS-SIM branch selected: "
            f"filter={fourier_filter_mode}, cutoff={fourier_actual_cutoff:.5f} cyc/pix "
            f"({'auto' if fourier_cutoff_input <= 0 else 'manual'}), "
            f"order={fourier_order}, amount={fourier_amount}"
        )

    progress("Preparing reconstruction parameters...", 48)
    log(f"Applying unified phase normalization: mode={phase_norm_mode}, stat={phase_norm_stat}")
    normalized_orientation_groups, phase_norm_refs, phase_norm_scales = normalize_phase_groups(
        orientation_groups,
        mode=phase_norm_mode,
        method=phase_norm_stat,
    )
    for ori_idx, scales in enumerate(phase_norm_scales, 1):
        log(
            f"Orientation {ori_idx} phase scales = "
            + ", ".join(f"{s:.4f}" for s in scales)
        )

    progress("Computing pseudo-widefield images...", 55)
    log("Computing pseudo-widefield...")
    pwf_images_raw = [
        pseudo_widefield(group[0], group[1], group[2])
        for group in normalized_orientation_groups
    ]

    progress("Computing SIM reconstruction branch...", 62)
    log(f"Computing SIM reconstruction branch: {os_sim_recon_mode}...")
    sim_images_raw = []
    fourier_os_sim_infos = []
    if "fourier" in os_sim_recon_mode.lower():
        for idx, group in enumerate(normalized_orientation_groups, 1):
            sim_img, f_info = fourier_os_sim_reconstruct(
                group[0], group[1], group[2],
                phase_model=phase_model,
                cutoff_cycles_per_pixel=fourier_actual_cutoff,
                filter_mode=fourier_filter_mode,
                butterworth_order=fourier_order,
                filter_amount=fourier_amount,
            )
            sim_images_raw.append(sim_img)
            fourier_os_sim_infos.append(f_info)
            log(
                f"Fourier-OS-SIM orientation {idx}: cutoff={f_info.get('cutoff_cycles_per_pixel', 0):.5f} cyc/pix, "
                f"mean before/after={f_info.get('spatial_mean_before_filter', 0):.4f}/"
                f"{f_info.get('spatial_mean_after_filter', 0):.4f}"
            )
    else:
        sim_images_raw = [
            sim_reconstruct(group[0], group[1], group[2], phase_model=phase_model)
            for group in normalized_orientation_groups
        ]
        fourier_os_sim_infos = []
    sim_images_before_bg_correction = [img.copy() for img in sim_images_raw]

    sim_bg_estimates = []
    if enable_sim_bg_correction:
        log(
            f"Applying single-orientation SIM low-frequency background correction, "
            f"mode={sim_bg_mode}, sigma={sim_bg_sigma}"
        )
        corrected_sim_images = []
        for idx, sim_img in enumerate(sim_images_raw, 1):
            corrected_img, bg_img = correct_low_frequency_background(
                sim_img, sigma=sim_bg_sigma, mode=sim_bg_mode
            )
            corrected_sim_images.append(corrected_img)
            sim_bg_estimates.append(bg_img)
            log(f"Orientation {idx} SIM background corrected.")
        sim_images_raw = corrected_sim_images
    else:
        sim_bg_estimates = [np.zeros_like(img, dtype=np.float32) for img in sim_images_raw]

    sim_images_after_bg_correction = [img.copy() for img in sim_images_raw]

    if enable_orientation_match:
        log(f"Matching orientation intensity before fusion using {orientation_match_method}...")
        sim_images, sim_match_ref, sim_match_scales = match_image_sequence_intensity(
            sim_images_raw, method=orientation_match_method
        )
        pwf_images, pwf_match_ref, pwf_match_scales = match_image_sequence_intensity(
            pwf_images_raw, method=orientation_match_method
        )
        for idx, scale in enumerate(sim_match_scales, 1):
            log(f"Orientation {idx} match scale = {scale:.4f}")
    else:
        sim_images = [img.copy() for img in sim_images_raw]
        pwf_images = [img.copy() for img in pwf_images_raw]
        sim_match_ref = 1.0
        sim_match_scales = [1.0] * len(sim_images)
        pwf_match_ref = 1.0
        pwf_match_scales = [1.0] * len(pwf_images)

    pwf_fused = np.mean(np.stack(pwf_images, axis=0), axis=0).astype(np.float32)
    sim_fused_arithmetic_mean = fuse_sim_orientations(sim_images, mode="Mean")
    sim_fused_final = fuse_sim_orientations(sim_images, mode=final_fusion_mode)
    progress("Fusing orientations...", 70)
    log(f"Computing final SIM fusion, mode={final_fusion_mode}.")

    progress("Computing wavelet fusion...", 74)
    log(f"Computing {fusion_method} wavelet fusion, detail_mode={wavelet_detail_mode}...")
    sim_wavelet_fused, actual_level = fuse_image_sequence(
        sim_images,
        wavelet=wavelet_name,
        level=wavelet_level,
        method=fusion_method,
        detail_mode=wavelet_detail_mode,
    )

    if actual_level != wavelet_level:
        log(
            f"Requested wavelet level = {wavelet_level}, "
            f"but actual usable level = {actual_level} for current ROI/method."
        )

    # =========================================================
    # Optimized OS-SIM + HiLo pipeline
    # 1) OS-SIM / pWF are computed per orientation above.
    # 2) HiLo is computed before final postprocessing so Lo/Mid/High
    #    fusion is not biased by later smoothing or denoising.
    # 3) Postprocess is then applied in a controlled order:
    #    low-frequency structure/background correction -> optional
    #    Gaussian smoothing -> optional local Wiener filtering.
    # =========================================================
    sim_fused_before_postprocess = sim_fused_final.copy()
    sim_wavelet_before_postprocess = sim_wavelet_fused.copy()

    hilo_widefield_source_image = None
    hilo_sectioned_source_image = None
    hilo_widefield_lowpass = None
    hilo_low_component = None
    hilo_high_component = None
    hilo_high_abs_component = None
    hilo_mid_component = None
    hilo_mid_abs_component = None
    hilo_mid_high_component = None
    hilo_high_lowpass = None
    hilo_hybrid_before_postprocess = None
    hilo_hybrid_before_final_structure_enhance = None
    hilo_hybrid_before_smoothing = None
    hilo_hybrid_before_wiener = None
    hilo_hybrid = None
    hilo_wiener_noise_variance = None
    hilo_wiener_kernel = None
    hilo_lowfreq_bg = None
    hilo_highpass = None
    hilo_hybrid_before_hilo_denoise = None
    hilo_denoise_info = {"enabled": bool(enable_hilo_denoise), "applied": False}
    hilo_orientation_hybrids = []
    hilo_orientation_components = []
    hilo_info = {
        "enabled": bool(enable_hilo_mode),
        "strategy": hilo_strategy,
        "auto_sigma": bool(hilo_auto_sigma),
        "auto_sigma_info": hilo_auto_sigma_info,
        "widefield_source": hilo_widefield_source,
        "low_source": hilo_low_source,
        "lo_sigma": float(hilo_sigma),
        "mid_sigma": float(hilo_mid_sigma),
        "high_sigma": float(hilo_high_sigma),
        "lo_weight": float(hilo_lo_weight),
        "mid_weight": float(hilo_mid_weight),
        "hi_weight": float(hilo_hi_weight),
        "clip_output": bool(hilo_clip_output),
        "actual_widefield_source": "",
        "actual_low_source": "",
        "fusion_note": "",
    }

    progress("Computing HiLo components...", 80)

    if enable_hilo_mode:
        use_per_orientation_hilo = hilo_strategy.startswith("Per-orientation")
        if use_per_orientation_hilo:
            log(
                "Computing optimized HiLo before postprocess: per-orientation pWF + OS-SIM, then RMS fusion. "
                f"lo_sigma={hilo_sigma}, mid_sigma={hilo_mid_sigma}, high_sigma={hilo_high_sigma}, "
                f"lo_weight={hilo_lo_weight}, mid_weight={hilo_mid_weight}, hi_weight={hilo_hi_weight}, "
                f"clip={hilo_clip_output}"
            )
            hilo_info["fusion_note"] = "HiLo was computed per orientation from matched pWF/raw WF and matched OS-SIM, then RMS-fused."

            per_low = []
            per_mid = []
            per_mid_abs = []
            per_high = []
            per_high_abs = []
            per_mid_high = []
            per_widefield_lowpass = []
            per_high_lowpass = []
            per_widefield_sources = []
            per_sectioned_sources = []

            for ori_idx in range(orientation_count):
                if hilo_widefield_source == "External raw widefield if available":
                    if external_raw_widefield is not None:
                        wf_src = external_raw_widefield
                        actual_wf = "external_raw_widefield shared"
                    else:
                        wf_src = pwf_images[ori_idx]
                        actual_wf = "orientation_pwf fallback"
                elif hilo_widefield_source == "Raw widefield from ROI":
                    wf_src = raw_widefield_images[ori_idx]
                    actual_wf = "orientation_raw_pwf"
                else:
                    wf_src = pwf_images[ori_idx]
                    actual_wf = "orientation_pwf"

                if hilo_low_source == "SIM wavelet fused":
                    # Wavelet fusion is already a multi-orientation image. In per-orientation mode
                    # we keep the HiLo Lo source physically consistent with each orientation OS-SIM.
                    section_src = sim_images[ori_idx]
                    actual_low = "orientation_os_sim (wavelet requested; per-orientation fallback)"
                else:
                    section_src = sim_images[ori_idx]
                    actual_low = "orientation_os_sim"

                (
                    low_i,
                    high_i,
                    high_abs_i,
                    wf_low_i,
                    hybrid_i,
                    mid_i,
                    mid_abs_i,
                    mid_high_i,
                    high_low_i,
                ) = create_hilo_hybrid(
                    widefield=wf_src,
                    sectioned_image=section_src,
                    sigma=hilo_sigma,
                    lo_weight=hilo_lo_weight,
                    mid_weight=hilo_mid_weight,
                    hi_weight=hilo_hi_weight,
                    high_sigma=hilo_high_sigma,
                    mid_sigma=hilo_mid_sigma,
                    clip_output=hilo_clip_output,
                )

                hilo_orientation_hybrids.append(hybrid_i)
                hilo_orientation_components.append({
                    "widefield_source": wf_src,
                    "sectioned_source": section_src,
                    "low": low_i,
                    "mid": mid_i,
                    "mid_abs": mid_abs_i,
                    "high": high_i,
                    "high_abs": high_abs_i,
                    "weighted_mid_high": mid_high_i,
                    "widefield_lowpass": wf_low_i,
                    "high_lowpass": high_low_i,
                    "actual_widefield_source": actual_wf,
                    "actual_low_source": actual_low,
                })
                per_low.append(low_i)
                per_mid.append(mid_i)
                per_mid_abs.append(mid_abs_i)
                per_high.append(high_i)
                per_high_abs.append(high_abs_i)
                per_mid_high.append(mid_high_i)
                per_widefield_lowpass.append(wf_low_i)
                per_high_lowpass.append(high_low_i)
                per_widefield_sources.append(wf_src)
                per_sectioned_sources.append(section_src)
                log(f"HiLo orientation {ori_idx + 1}: WF source={actual_wf}, Lo source={actual_low}.")

            hilo_hybrid = fuse_sim_orientations(hilo_orientation_hybrids, mode="RMS")
            hilo_low_component = fuse_sim_orientations(per_low, mode="RMS")
            hilo_mid_component = fuse_signed_image_sequence(per_mid, mode="Mean")
            hilo_mid_abs_component = np.mean(np.stack(per_mid_abs, axis=0), axis=0).astype(np.float32)
            hilo_high_component = fuse_signed_image_sequence(per_high, mode="Mean")
            hilo_high_abs_component = np.mean(np.stack(per_high_abs, axis=0), axis=0).astype(np.float32)
            hilo_mid_high_component = fuse_signed_image_sequence(per_mid_high, mode="Mean")
            hilo_widefield_lowpass = np.mean(np.stack(per_widefield_lowpass, axis=0), axis=0).astype(np.float32)
            hilo_high_lowpass = np.mean(np.stack(per_high_lowpass, axis=0), axis=0).astype(np.float32)
            hilo_widefield_source_image = np.mean(np.stack(per_widefield_sources, axis=0), axis=0).astype(np.float32)
            hilo_sectioned_source_image = fuse_sim_orientations(per_sectioned_sources, mode="RMS")
            hilo_info["actual_widefield_source"] = "per-orientation sources; displayed image is their mean"
            hilo_info["actual_low_source"] = "per-orientation OS-SIM; displayed image is RMS fusion"
        else:
            if hilo_widefield_source == "External raw widefield if available":
                if external_raw_widefield is not None:
                    hilo_widefield_source_image = external_raw_widefield
                    hilo_info["actual_widefield_source"] = "external_raw_widefield"
                else:
                    hilo_widefield_source_image = pwf_fused
                    hilo_info["actual_widefield_source"] = "pwf_fused fallback"
                    log("HiLo requested external raw widefield, but none was loaded. Falling back to pseudo-widefield fused.")
            elif hilo_widefield_source == "Raw widefield from ROI":
                hilo_widefield_source_image = raw_widefield_fused
                hilo_info["actual_widefield_source"] = "raw_widefield_fused"
            else:
                hilo_widefield_source_image = pwf_fused
                hilo_info["actual_widefield_source"] = "pwf_fused"

            if hilo_low_source == "SIM wavelet fused":
                hilo_sectioned_source_image = sim_wavelet_before_postprocess
                hilo_info["actual_low_source"] = "sim_wavelet_fused_before_postprocess"
            else:
                hilo_sectioned_source_image = sim_fused_before_postprocess
                hilo_info["actual_low_source"] = "sim_fused_final_before_postprocess"

            log(
                "Computing legacy fused-source HiLo before postprocess: "
                f"Hi source={hilo_info['actual_widefield_source']}, "
                f"Lo source={hilo_info['actual_low_source']}, "
                f"lo_sigma={hilo_sigma}, mid_sigma={hilo_mid_sigma}, high_sigma={hilo_high_sigma}, "
                f"lo_weight={hilo_lo_weight}, mid_weight={hilo_mid_weight}, hi_weight={hilo_hi_weight}, "
                f"clip={hilo_clip_output}"
            )
            (
                hilo_low_component,
                hilo_high_component,
                hilo_high_abs_component,
                hilo_widefield_lowpass,
                hilo_hybrid,
                hilo_mid_component,
                hilo_mid_abs_component,
                hilo_mid_high_component,
                hilo_high_lowpass,
            ) = create_hilo_hybrid(
                widefield=hilo_widefield_source_image,
                sectioned_image=hilo_sectioned_source_image,
                sigma=hilo_sigma,
                lo_weight=hilo_lo_weight,
                mid_weight=hilo_mid_weight,
                hi_weight=hilo_hi_weight,
                high_sigma=hilo_high_sigma,
                mid_sigma=hilo_mid_sigma,
                clip_output=hilo_clip_output,
            )
            hilo_info["fusion_note"] = "HiLo was computed once from already-fused widefield and OS-SIM sources."

        hilo_hybrid_before_postprocess = hilo_hybrid.copy()
        hilo_hybrid_before_hilo_denoise = hilo_hybrid.copy()
        if enable_hilo_denoise:
            log(
                "Applying HiLo-only denoise after Lo/Mid/High fusion: "
                f"method={hilo_denoise_method}, nlm_h={hilo_nlm_h}, "
                f"template={hilo_nlm_template_window}, search={hilo_nlm_search_window}, "
                f"bm3d_sigma={hilo_bm3d_sigma}, amount={hilo_denoise_amount}"
            )
            hilo_hybrid, hilo_denoise_info = apply_hilo_special_denoise(
                hilo_hybrid,
                method=hilo_denoise_method,
                nlm_h=hilo_nlm_h,
                nlm_template_window=hilo_nlm_template_window,
                nlm_search_window=hilo_nlm_search_window,
                bm3d_sigma=hilo_bm3d_sigma,
                amount=hilo_denoise_amount,
            )
            hilo_denoise_info["enabled"] = True
            log(f"HiLo-only denoise info: {hilo_denoise_info}")

    # 08 Postprocess A: final structure / low-frequency background correction
    sim_fused_before_final_structure_enhance = sim_fused_final.copy()
    sim_fused_lowfreq_bg = np.zeros_like(sim_fused_final, dtype=np.float32)
    sim_fused_highpass = np.zeros_like(sim_fused_final, dtype=np.float32)

    sim_wavelet_before_final_structure_enhance = sim_wavelet_fused.copy()
    sim_wavelet_lowfreq_bg = np.zeros_like(sim_wavelet_fused, dtype=np.float32)
    sim_wavelet_highpass = np.zeros_like(sim_wavelet_fused, dtype=np.float32)

    if hilo_hybrid is not None:
        hilo_hybrid_before_final_structure_enhance = hilo_hybrid.copy()
        hilo_lowfreq_bg = np.zeros_like(hilo_hybrid, dtype=np.float32)
        hilo_highpass = np.zeros_like(hilo_hybrid, dtype=np.float32)

    if enable_final_structure_enhance:
        log(
            "Applying postprocess low-frequency structure/background correction: "
            f"target={final_structure_target}, mode={final_structure_mode}, "
            f"sigma={final_structure_sigma}, amount={final_structure_amount}, "
            f"keep_original={final_structure_keep_original}"
        )
        sim_fused_final, sim_fused_lowfreq_bg, sim_fused_highpass = enhance_final_sim_structure(
            sim_fused_final,
            sigma=final_structure_sigma,
            amount=final_structure_amount,
            keep_original=final_structure_keep_original,
            mode=final_structure_mode,
        )

        if "Wavelet fused" in final_structure_target:
            sim_wavelet_fused, sim_wavelet_lowfreq_bg, sim_wavelet_highpass = enhance_final_sim_structure(
                sim_wavelet_fused,
                sigma=final_structure_sigma,
                amount=final_structure_amount,
                keep_original=final_structure_keep_original,
                mode=final_structure_mode,
            )

        if "HiLo Hybrid" in final_structure_target and hilo_hybrid is not None:
            hilo_hybrid, hilo_lowfreq_bg, hilo_highpass = enhance_final_sim_structure(
                hilo_hybrid,
                sigma=final_structure_sigma,
                amount=final_structure_amount,
                keep_original=final_structure_keep_original,
                mode=final_structure_mode,
            )

    # 08 Postprocess B: optional Gaussian smoothing
    sim_fused_before_smoothing = sim_fused_final.copy()
    sim_wavelet_before_smoothing = sim_wavelet_fused.copy()
    if hilo_hybrid is not None:
        hilo_hybrid_before_smoothing = hilo_hybrid.copy()

    if enable_smoothing and sigma > 0:
        log(f"Applying Gaussian smoothing to Final SIM and Wavelet fused, sigma={sigma}")
        sim_wavelet_fused = cv2.GaussianBlur(sim_wavelet_fused, (0, 0), sigma)
        sim_fused_final = cv2.GaussianBlur(sim_fused_final, (0, 0), sigma)

    # 08 Postprocess C: optional local Wiener filtering
    sim_fused_before_wiener = sim_fused_final.copy()
    sim_wavelet_before_wiener = sim_wavelet_fused.copy()
    if hilo_hybrid is not None:
        hilo_hybrid_before_wiener = hilo_hybrid.copy()

    wiener_actual_noise_variance = None
    wiener_actual_kernel = _odd_kernel_size(wiener_kernel)
    wiener_applied_to = []

    if enable_wiener_filter:
        log(
            "Applying Wiener filter after OS-SIM/HiLo fusion and background correction: "
            f"target={wiener_target}, kernel={wiener_kernel}, "
            f"noise_variance={wiener_noise_variance}, amount={wiener_amount}"
        )
        sim_fused_final, noise_est, wiener_actual_kernel = apply_local_wiener_filter(
            sim_fused_final,
            kernel_size=wiener_kernel,
            noise_variance=wiener_noise_variance,
            amount=wiener_amount,
        )
        wiener_actual_noise_variance = float(noise_est)
        wiener_applied_to.append("Final SIM")

        if "Wavelet fused" in wiener_target:
            sim_wavelet_fused, noise_est_w, _ = apply_local_wiener_filter(
                sim_wavelet_fused,
                kernel_size=wiener_kernel,
                noise_variance=wiener_noise_variance,
                amount=wiener_amount,
            )
            wiener_applied_to.append("Wavelet fused")
            if wiener_noise_variance > 0:
                wiener_actual_noise_variance = float(noise_est_w)

        if "HiLo Hybrid" in wiener_target and hilo_hybrid is not None:
            hilo_hybrid, hilo_noise_est, hilo_kernel = apply_local_wiener_filter(
                hilo_hybrid,
                kernel_size=wiener_kernel,
                noise_variance=wiener_noise_variance,
                amount=wiener_amount,
            )
            hilo_wiener_noise_variance = float(hilo_noise_est)
            hilo_wiener_kernel = int(hilo_kernel)
            wiener_applied_to.append("HiLo Hybrid")
            if wiener_noise_variance > 0:
                wiener_actual_noise_variance = float(hilo_noise_est)
                        # 08 Postprocess D: optional output intensity preservation / brightness matching
    sim_fused_before_intensity_match = sim_fused_final.copy()
    sim_wavelet_before_intensity_match = sim_wavelet_fused.copy()
    hilo_hybrid_before_intensity_match = hilo_hybrid.copy() if hilo_hybrid is not None else None

    progress("Applying final postprocessing...", 88)

    output_intensity_match_info = {
        "enabled": bool(enable_output_intensity_match),
        "applied_to": [],
        "reference_source": str(output_intensity_reference),
        "method": str(output_intensity_method),
        "strength": float(output_intensity_strength),
        "max_scale": float(output_intensity_max_scale),
        "details": {},
    }

    if enable_output_intensity_match:
        # Choose reference image
        if output_intensity_reference == "External raw widefield if available" and external_raw_widefield is not None:
            intensity_reference_img = external_raw_widefield
            intensity_reference_name = "external_raw_widefield"
        elif output_intensity_reference == "Raw widefield fused":
            intensity_reference_img = raw_widefield_fused
            intensity_reference_name = "raw_widefield_fused"
        else:
            intensity_reference_img = pwf_fused
            intensity_reference_name = "pwf_fused"

        output_intensity_match_info["reference_used"] = intensity_reference_name

        log(
            "Applying output intensity preservation: "
            f"target={output_intensity_target}, "
            f"reference={intensity_reference_name}, "
            f"method={output_intensity_method}, "
            f"strength={output_intensity_strength}, "
            f"max_scale={output_intensity_max_scale}"
        )

        sim_fused_final, match_info_final = match_output_intensity_to_reference(
            sim_fused_final,
            intensity_reference_img,
            method=output_intensity_method,
            strength=output_intensity_strength,
            max_scale=output_intensity_max_scale,
            clip_output=True,
        )
        output_intensity_match_info["applied_to"].append("Final SIM")
        output_intensity_match_info["details"]["Final SIM"] = match_info_final

        if "Wavelet fused" in output_intensity_target:
            sim_wavelet_fused, match_info_wavelet = match_output_intensity_to_reference(
                sim_wavelet_fused,
                intensity_reference_img,
                method=output_intensity_method,
                strength=output_intensity_strength,
                max_scale=output_intensity_max_scale,
                clip_output=True,
            )
            output_intensity_match_info["applied_to"].append("Wavelet fused")
            output_intensity_match_info["details"]["Wavelet fused"] = match_info_wavelet

        if "HiLo Hybrid" in output_intensity_target and hilo_hybrid is not None:
            hilo_hybrid, match_info_hilo = match_output_intensity_to_reference(
                hilo_hybrid,
                intensity_reference_img,
                method=output_intensity_method,
                strength=output_intensity_strength,
                max_scale=output_intensity_max_scale,
                clip_output=True,
            )
            output_intensity_match_info["applied_to"].append("HiLo Hybrid")
            output_intensity_match_info["details"]["HiLo Hybrid"] = match_info_hilo

    progress("Packaging results for display...", 92)

    results = {
        "gpu_backend_status": gpu_backend_status_text(),
        "orientation_count": orientation_count,
        "orientation_labels": orientation_labels,
        "input_source_type": input_source_type,
        "single_tiff_path": single_tiff_path,
        "image_order_mode": image_order_mode,
        "image_order_label": image_order_text,
        "source_items_in_processing_order": [_input_item_source_name(item) for item in input_items],
        "roi": {
            "enabled": bool(roi_enabled),
            "x": int(roi_x),
            "y": int(roi_y),
            "width": int(roi_w),
            "height": int(roi_h),
        },
        "raw_cropped_images": raw_cropped_images,
        "raw_orientation_groups": raw_orientation_groups,
        "raw_widefield_images": raw_widefield_images,
        "processed_groups": orientation_groups,
        "preprocess_enabled": enable_preprocess,
        "phase_model": phase_model,
        "preprocess_mode": preprocess_mode,
        "preprocess_infos": preprocess_infos,
        "dark_image_path": dark_image_path,
        "flat_image_path": flat_image_path,
        "dark_roi": dark_roi,
        "flat_roi": flat_roi,
        "background_percentile": bg_percentile,
        "raw_phase_diagnostics": raw_phase_diagnostics,
        "processed_phase_diagnostics": processed_phase_diagnostics,
        "normalized_orientation_groups": normalized_orientation_groups,
        "phase_norm_mode": phase_norm_mode,
        "phase_norm_stat": phase_norm_stat,
        "os_sim_recon_mode": os_sim_recon_mode,
        "fourier_filter_mode": fourier_filter_mode,
        "fourier_cutoff_input": fourier_cutoff_input,
        "fourier_actual_cutoff": fourier_actual_cutoff,
        "fourier_order": fourier_order,
        "fourier_amount": fourier_amount,
        "fourier_cutoff_info": fourier_cutoff_info,
        "fourier_os_sim_infos": fourier_os_sim_infos,
        "phase_norm_reference_values": phase_norm_refs,
        "phase_norm_scales": phase_norm_scales,
        "pwf_images_raw": pwf_images_raw,
        "sim_images_before_bg_correction": sim_images_before_bg_correction,
        "sim_images_raw": sim_images_raw,
        "sim_images_after_bg_correction": sim_images_after_bg_correction,
        "sim_bg_estimates": sim_bg_estimates,
        "pwf_images": pwf_images,
        "sim_images": sim_images,
        "pwf_fused": pwf_fused,
        "raw_widefield_fused": raw_widefield_fused,
        "external_raw_widefield": external_raw_widefield,
        "sim_fused_mean": sim_fused_final,
        "sim_fused_arithmetic_mean": sim_fused_arithmetic_mean,
        "sim_fused_final": sim_fused_final,
        "sim_fused_before_postprocess": sim_fused_before_postprocess,
        "sim_fused_before_final_structure_enhance": sim_fused_before_final_structure_enhance,
        "sim_fused_lowfreq_bg": sim_fused_lowfreq_bg,
        "sim_fused_highpass": sim_fused_highpass,
        "sim_fused_before_smoothing": sim_fused_before_smoothing,
        "final_fusion_mode": final_fusion_mode,
        "sim_wavelet_fused": sim_wavelet_fused,
        "sim_wavelet_before_postprocess": sim_wavelet_before_postprocess,
        "sim_wavelet_before_final_structure_enhance": sim_wavelet_before_final_structure_enhance,
        "sim_wavelet_lowfreq_bg": sim_wavelet_lowfreq_bg,
        "sim_wavelet_highpass": sim_wavelet_highpass,
        "sim_wavelet_before_smoothing": sim_wavelet_before_smoothing,
        "enable_final_structure_enhance": enable_final_structure_enhance,
        "final_structure_target": final_structure_target,
        "final_structure_mode": final_structure_mode,
        "final_structure_sigma": final_structure_sigma,
        "final_structure_amount": final_structure_amount,
        "final_structure_keep_original": final_structure_keep_original,
        "fusion_method": fusion_method,
        "wavelet_name": wavelet_name,
        "wavelet_level": actual_level,
        "wavelet_detail_mode": wavelet_detail_mode,
        "enable_orientation_match": enable_orientation_match,
        "orientation_match_method": orientation_match_method,
        "sim_match_reference": sim_match_ref,
        "sim_match_scales": sim_match_scales,
        "pwf_match_reference": pwf_match_ref,
        "pwf_match_scales": pwf_match_scales,
        "enable_sim_bg_correction": enable_sim_bg_correction,
        "sim_bg_mode": sim_bg_mode,
        "sim_bg_sigma": sim_bg_sigma,
        "enable_smoothing": enable_smoothing,
        "gaussian_sigma": sigma,
        "enable_wiener_filter": enable_wiener_filter,
        "wiener_target": wiener_target,
        "wiener_kernel": wiener_kernel,
        "wiener_actual_kernel": wiener_actual_kernel,
        "wiener_noise_variance": wiener_noise_variance,
        "wiener_actual_noise_variance": wiener_actual_noise_variance,
        "wiener_amount": wiener_amount,
        "wiener_applied_to": wiener_applied_to,
        "sim_fused_before_wiener": sim_fused_before_wiener,
        "sim_wavelet_before_wiener": sim_wavelet_before_wiener,
        "enable_output_intensity_match": enable_output_intensity_match,
        "output_intensity_target": output_intensity_target,
        "output_intensity_reference": output_intensity_reference,
        "output_intensity_method": output_intensity_method,
        "output_intensity_strength": output_intensity_strength,
        "output_intensity_max_scale": output_intensity_max_scale,
        "output_intensity_match_info": output_intensity_match_info,
        "sim_fused_before_intensity_match": sim_fused_before_intensity_match,
        "sim_wavelet_before_intensity_match": sim_wavelet_before_intensity_match,
        "hilo_hybrid_before_intensity_match": hilo_hybrid_before_intensity_match,

        "enable_hilo_mode": enable_hilo_mode,
        "hilo_info": hilo_info,
        "hilo_strategy": hilo_strategy,
        "hilo_auto_sigma": hilo_auto_sigma,
        "hilo_auto_sigma_info": hilo_auto_sigma_info,
        "hilo_widefield_source": hilo_widefield_source,
        "hilo_low_source": hilo_low_source,
        "hilo_sigma": hilo_sigma,
        "hilo_mid_sigma": hilo_mid_sigma,
        "hilo_high_sigma": hilo_high_sigma,
        "hilo_lo_weight": hilo_lo_weight,
        "hilo_mid_weight": hilo_mid_weight,
        "hilo_hi_weight": hilo_hi_weight,
        "hilo_clip_output": hilo_clip_output,
        "hilo_widefield_source_image": hilo_widefield_source_image,
        "hilo_sectioned_source_image": hilo_sectioned_source_image,
        "hilo_widefield_lowpass": hilo_widefield_lowpass,
        "hilo_high_lowpass": hilo_high_lowpass,
        "hilo_low_component": hilo_low_component,
        "hilo_mid_component": hilo_mid_component,
        "hilo_mid_abs_component": hilo_mid_abs_component,
        "hilo_high_component": hilo_high_component,
        "hilo_high_abs_component": hilo_high_abs_component,
        "hilo_mid_high_component": hilo_mid_high_component,
        "hilo_orientation_hybrids": hilo_orientation_hybrids,
        "hilo_orientation_components": hilo_orientation_components,
        "hilo_hybrid_before_postprocess": hilo_hybrid_before_postprocess,
        "hilo_hybrid_before_final_structure_enhance": hilo_hybrid_before_final_structure_enhance,
        "hilo_lowfreq_bg": hilo_lowfreq_bg,
        "hilo_highpass": hilo_highpass,
        "hilo_hybrid_before_smoothing": hilo_hybrid_before_smoothing,
        "hilo_hybrid_before_hilo_denoise": hilo_hybrid_before_hilo_denoise,
        "enable_hilo_denoise": enable_hilo_denoise,
        "hilo_denoise_method": hilo_denoise_method,
        "hilo_nlm_h": hilo_nlm_h,
        "hilo_nlm_template_window": hilo_nlm_template_window,
        "hilo_nlm_search_window": hilo_nlm_search_window,
        "hilo_bm3d_sigma": hilo_bm3d_sigma,
        "hilo_denoise_amount": hilo_denoise_amount,
        "hilo_denoise_info": hilo_denoise_info,
        "hilo_hybrid_before_wiener": hilo_hybrid_before_wiener,
        "hilo_hybrid": hilo_hybrid,
        "hilo_wiener_noise_variance": hilo_wiener_noise_variance,
        "hilo_wiener_kernel": hilo_wiener_kernel,
    }

    if orientation_count >= 1:
        results["pwf_vertical"] = pwf_images[0]
        results["sim_vertical"] = sim_images[0]
        results["v0_p"], results["v90_p"], results["v180_p"] = orientation_groups[0]
    if orientation_count >= 2:
        results["pwf_horizontal"] = pwf_images[1]
        results["sim_horizontal"] = sim_images[1]
        results["h0_p"], results["h90_p"], results["h180_p"] = orientation_groups[1]
    if orientation_count >= 3:
        results["pwf_orientation3"] = pwf_images[2]
        results["sim_orientation3"] = sim_images[2]
        results["o3_0_p"], results["o3_90_p"], results["o3_180_p"] = orientation_groups[2]


    progress("Processing finished.", 100)
    log("Processing finished.")
    return results, input_items


class SIMProcessingWorker(QObject):
    log_message = Signal(str)
    progress_changed = Signal(str, int)
    finished = Signal(object, object, float)
    failed = Signal(str, str)

    def __init__(self, config):
        super().__init__()
        self.config = dict(config or {})

    def _emit_log(self, message):
        self.log_message.emit(str(message))

    def _emit_progress(self, message, value):
        self.progress_changed.emit(str(message), int(value))

    @Slot()
    def run(self):
        started = time.perf_counter()
        try:
            results, image_items = run_sim_processing_core(
                self.config,
                progress_cb=self._emit_progress,
                log_cb=self._emit_log,
            )
            elapsed = time.perf_counter() - started
            self.finished.emit(results, image_items, float(elapsed))
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


# =========================================================
# 主界面
# =========================================================

class SIMGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIM Reconstruction GUI")
        self.resize(1550, 920)

        self.results = {}
        self.image_paths = []
        self.profile_window = None
        self._processing_thread = None
        self._processing_worker = None
        self._processing_progress = None
        self._processing_cursor_set = False
        self._last_run_elapsed_seconds = None

        self._build_ui()
        self._set_default_values()
        self._auto_load_startup_config()

    def _auto_load_startup_config(self):
        try:
            if os.path.exists(getattr(self, "config_path", "")):
                self.load_params_and_layout(show_message=False)
        except Exception as e:
            self.log(f"[WARN] Failed to auto-load startup config: {e}")

    def _restore_main_window_geometry(self, data):
        """Restore saved geometry without leaving the window and its dialogs off-screen."""
        if not data:
            return

        screens = QApplication.screens()
        screen_rects = []
        for screen in screens:
            available = screen.availableGeometry()
            screen_rects.append(
                (available.x(), available.y(), available.width(), available.height())
            )

        primary_screen = QApplication.primaryScreen()
        fallback_rect = None
        if primary_screen is not None:
            available = primary_screen.availableGeometry()
            fallback_rect = (
                available.x(),
                available.y(),
                available.width(),
                available.height(),
            )

        geometry = _fit_window_geometry_to_screens(
            data.get("x", self.x()),
            data.get("y", self.y()),
            data.get("width", self.width()),
            data.get("height", self.height()),
            screen_rects,
            fallback_rect=fallback_rect,
        )
        self.setGeometry(*geometry)

    def closeEvent(self, event):
        if getattr(self, "_processing_thread", None) is not None and self._processing_thread.isRunning():
            QMessageBox.warning(
                self,
                "Processing is running",
                "A reconstruction is still running in the background. Please let it finish before closing the GUI."
            )
            event.ignore()
            return

        try:
            self.save_params_and_layout(show_message=False)
        except Exception as e:
            try:
                self.log(f"[WARN] Failed to auto-save config on exit: {e}")
            except Exception:
                pass
        super().closeEvent(event)

    # -------------------------
    # UI 构建
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        self.main_splitter = QSplitter(Qt.Horizontal)

        # Compact left dock sizing. Keep all left-side frames constrained to this width.
        self._left_panel_min_width = 320
        self._left_panel_default_width = 400
        self._left_panel_max_width = 430

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        left_panel.setMinimumWidth(self._left_panel_min_width)
        left_panel.setMaximumWidth(self._left_panel_max_width)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([self._left_panel_default_width, 1150])

        layout = QHBoxLayout()
        layout.addWidget(self.main_splitter)
        central.setLayout(layout)
        self._build_status_bar()

    def _set_path_tooltip_from_text(self, edit):
        try:
            edit.setToolTip(edit.text())
        except Exception:
            pass

    def _apply_left_dock_compact_layout(
        self,
        groups=None,
        forms=None,
        line_edits=None,
        browse_buttons=None,
        clear_buttons=None,
        combos=None,
        spin_boxes=None,
        labels=None,
        row_widgets=None,
    ):
        """
        Keep all left-dock group boxes and their fields inside the compact dock width.
        This prevents Path / ROI / long ComboBox text from forcing the whole left panel wider.
        """
        dock_max = int(getattr(self, "_left_panel_max_width", 430))
        # Subtract outer margins + tab/scroll/page margins so frames visually fit the dock.
        group_max = max(280, dock_max - 34)
        field_max = max(145, dock_max - 190)
        combo_max = max(150, dock_max - 165)

        for group in groups or []:
            try:
                group.setMaximumWidth(group_max)
                group.setMinimumWidth(0)
                group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            except Exception:
                pass

        for form in forms or []:
            try:
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.setHorizontalSpacing(6)
                form.setVerticalSpacing(5)
            except Exception:
                pass

        for edit in line_edits or []:
            try:
                edit.setMinimumWidth(70)
                edit.setMaximumWidth(field_max)
                edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                edit.setToolTip(edit.text())
                edit.textChanged.connect(lambda _text, e=edit: self._set_path_tooltip_from_text(e))
            except Exception:
                pass

        for btn in browse_buttons or []:
            try:
                btn.setText("Browse")
                btn.setMinimumWidth(54)
                btn.setMaximumWidth(62)
                btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            except Exception:
                pass

        for btn in clear_buttons or []:
            try:
                btn.setText("Clear")
                btn.setMinimumWidth(42)
                btn.setMaximumWidth(48)
                btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            except Exception:
                pass

        for combo in combos or []:
            try:
                combo.setMinimumWidth(80)
                combo.setMaximumWidth(combo_max)
                combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                combo.setMinimumContentsLength(8)
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            except Exception:
                pass

        for spin in spin_boxes or []:
            try:
                spin.setMinimumWidth(72)
                spin.setMaximumWidth(118)
                spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            except Exception:
                pass

        for label in labels or []:
            try:
                label.setWordWrap(True)
                label.setMaximumWidth(group_max - 18)
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            except Exception:
                pass

        for row_widget in row_widgets or []:
            try:
                row_widget.setMinimumWidth(0)
                row_widget.setMaximumWidth(group_max - 115)
                row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                if row_widget.layout() is not None:
                    row_widget.layout().setContentsMargins(0, 0, 0, 0)
                    row_widget.layout().setSpacing(4)
            except Exception:
                pass

    def _build_left_panel(self):
        widget = QWidget()
        widget.setMinimumWidth(getattr(self, "_left_panel_min_width", 320))
        widget.setMaximumWidth(getattr(self, "_left_panel_max_width", 430))
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(widget)

        # ========== 路径 ==========
        path_group = QGroupBox("Paths")
        path_form = QFormLayout()
        path_form.setLabelAlignment(Qt.AlignRight)
        path_form.setFormAlignment(Qt.AlignTop)
        path_form.setHorizontalSpacing(8)
        path_form.setVerticalSpacing(6)

        self.input_dir_edit = QLineEdit()
        self.single_tiff_path_edit = QLineEdit()
        self.raw_widefield_path_edit = QLineEdit()
        self.dark_path_edit = QLineEdit()
        self.flat_path_edit = QLineEdit()
        self.input_dir_edit.setMinimumWidth(80)
        self.single_tiff_path_edit.setMinimumWidth(80)
        self.raw_widefield_path_edit.setMinimumWidth(80)
        self.dark_path_edit.setMinimumWidth(80)
        self.flat_path_edit.setMinimumWidth(80)

        for path_edit in (
            self.input_dir_edit,
            self.single_tiff_path_edit,
            self.raw_widefield_path_edit,
            self.dark_path_edit,
            self.flat_path_edit,
        ):
            path_edit.textChanged.connect(self._update_status_bar)

        input_btn = QPushButton("Browse...")
        input_btn.clicked.connect(self.choose_input_dir)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_dir_edit)
        input_row.addWidget(input_btn)

        single_tiff_btn = QPushButton("Browse...")
        single_tiff_btn.clicked.connect(self.choose_single_tiff_file)
        single_tiff_clear_btn = QPushButton("Clear")
        single_tiff_clear_btn.clicked.connect(self.clear_single_tiff_path)
        single_tiff_row = QHBoxLayout()
        single_tiff_row.addWidget(self.single_tiff_path_edit)
        single_tiff_row.addWidget(single_tiff_btn)
        single_tiff_row.addWidget(single_tiff_clear_btn)

        raw_wf_btn = QPushButton("Browse...")
        raw_wf_btn.clicked.connect(self.choose_raw_widefield_file)
        raw_wf_clear_btn = QPushButton("Clear")
        raw_wf_clear_btn.clicked.connect(lambda: self.raw_widefield_path_edit.clear())
        raw_wf_row = QHBoxLayout()
        raw_wf_row.addWidget(self.raw_widefield_path_edit)
        raw_wf_row.addWidget(raw_wf_btn)
        raw_wf_row.addWidget(raw_wf_clear_btn)

        dark_btn = QPushButton("Browse...")
        dark_btn.clicked.connect(self.choose_dark_file)
        dark_clear_btn = QPushButton("Clear")
        dark_clear_btn.clicked.connect(lambda: self.dark_path_edit.clear())
        dark_row = QHBoxLayout()
        dark_row.addWidget(self.dark_path_edit)
        dark_row.addWidget(dark_btn)
        dark_row.addWidget(dark_clear_btn)

        flat_btn = QPushButton("Browse...")
        flat_btn.clicked.connect(self.choose_flat_file)
        flat_clear_btn = QPushButton("Clear")
        flat_clear_btn.clicked.connect(lambda: self.flat_path_edit.clear())
        flat_row = QHBoxLayout()
        flat_row.addWidget(self.flat_path_edit)
        flat_row.addWidget(flat_btn)
        flat_row.addWidget(flat_clear_btn)

        input_row_widget = QWidget()
        input_row_widget.setLayout(input_row)
        single_tiff_row_widget = QWidget()
        single_tiff_row_widget.setLayout(single_tiff_row)
        raw_wf_row_widget = QWidget()
        raw_wf_row_widget.setLayout(raw_wf_row)
        dark_row_widget = QWidget()
        dark_row_widget.setLayout(dark_row)
        flat_row_widget = QWidget()
        flat_row_widget.setLayout(flat_row)

        path_form.addRow("Input folder", input_row_widget)
        path_form.addRow("Single TIFF stack", single_tiff_row_widget)
        path_form.addRow("Raw widefield", raw_wf_row_widget)
        path_form.addRow("Dark image", dark_row_widget)
        path_form.addRow("Flat image", flat_row_widget)

        self.ext_combo = QComboBox()
        self.ext_combo.addItems(["jpg/jpeg", "png", "tif/tiff", "all supported"])
        path_form.addRow("Input type", self.ext_combo)

        self.image_order_combo = QComboBox()
        self.image_order_combo.addItem(
            "Orientation-major / keep: 123 456 789",
            "orientation_major",
        )
        self.image_order_combo.addItem(
            "Phase-major: 147 258 369 -> 123 456 789",
            "phase_major",
        )
        self.image_order_combo.addItem(
            "147258369 order -> orientation-major",
            "147258369_to_orientation_major",
        )
        self.image_order_combo.addItem(
            "Reverse phase within each orientation: 321 654 987",
            "reverse_phase",
        )

        self.image_order_hint_label = QLabel(
            "Internal order after adjustment is always: "
            "orientation1 phase1/phase2/phase3, orientation2 phase1/phase2/phase3, orientation3 phase1/phase2/phase3. Phase shifts are selected separately in Reconstruct > Phase model. "
            "For a single multi-page TIFF, the same order modes are applied to TIFF pages."
        )
        self.image_order_hint_label.setWordWrap(True)

        self.save_ordered_roi_tiff_btn = QPushButton("Save ordered ROI stack TIFF...")
        self.save_ordered_roi_tiff_btn.setToolTip(
            "Save the current reordered and ROI-cropped 6/9 phase images as one multi-page TIFF."
        )

        path_form.addRow("Image order", self.image_order_combo)
        path_form.addRow("", self.save_ordered_roi_tiff_btn)
        path_form.addRow("Order note", self.image_order_hint_label)

        path_group.setLayout(path_form)

        # ========== 重建模式预设 ==========
        mode_group = QGroupBox("Reconstruction Mode Preset")
        mode_form = QFormLayout()
        mode_form.setLabelAlignment(Qt.AlignRight)
        mode_form.setFormAlignment(Qt.AlignTop)
        mode_form.setHorizontalSpacing(8)
        mode_form.setVerticalSpacing(6)

        self.recon_mode_combo = QComboBox()
        self.recon_mode_combo.addItems([
            "Paper mode",
            "Paper mode (soft fusion)",
            "Enhanced mode",
            "Custom",
        ])

        self.recon_mode_hint_label = QLabel("")
        self.recon_mode_hint_label.setWordWrap(True)

        mode_form.addRow("Preset", self.recon_mode_combo)
        mode_form.addRow("Description", self.recon_mode_hint_label)
        mode_group.setLayout(mode_form)

        # ========== 显示设置 ==========
        display_group = QGroupBox("Display Settings")
        display_form = QFormLayout()
        display_form.setLabelAlignment(Qt.AlignRight)
        display_form.setFormAlignment(Qt.AlignTop)
        display_form.setHorizontalSpacing(8)
        display_form.setVerticalSpacing(6)


        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Final comparison view",
            "Orientation view",
            "HiLo component view",
        ])

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "gray", "viridis", "magma", "plasma", "inferno",
            "cividis", "turbo", "hot", "jet"
        ])

        self.auto_contrast_check = QCheckBox("Enable auto contrast")
        self.auto_contrast_check.setChecked(True)

        self.auto_low_spin = QDoubleSpinBox()
        self.auto_low_spin.setRange(0.0, 50.0)
        self.auto_low_spin.setDecimals(1)
        self.auto_low_spin.setSingleStep(0.5)
        self.auto_low_spin.setValue(1.0)

        self.auto_high_spin = QDoubleSpinBox()
        self.auto_high_spin.setRange(50.0, 100.0)
        self.auto_high_spin.setDecimals(1)
        self.auto_high_spin.setSingleStep(0.5)
        self.auto_high_spin.setValue(99.0)

        self.manual_min_slider = QSlider(Qt.Horizontal)
        self.manual_min_slider.setRange(0, 1000)
        self.manual_min_slider.setValue(0)

        self.manual_max_slider = QSlider(Qt.Horizontal)
        self.manual_max_slider.setRange(0, 1000)
        self.manual_max_slider.setValue(1000)

        self.manual_min_label = QLabel("0.000")
        self.manual_max_label = QLabel("1.000")
        self.display_range_hint_label = QLabel("Display range: [0.000, 1.000]")

        self.sync_profile_display_check = QCheckBox("Sync display settings to profile window")
        self.sync_profile_display_check.setChecked(True)

        manual_min_widget = QWidget()
        manual_min_layout = QHBoxLayout(manual_min_widget)
        manual_min_layout.setContentsMargins(0, 0, 0, 0)
        manual_min_layout.addWidget(self.manual_min_slider)
        manual_min_layout.addWidget(self.manual_min_label)

        manual_max_widget = QWidget()
        manual_max_layout = QHBoxLayout(manual_max_widget)
        manual_max_layout.setContentsMargins(0, 0, 0, 0)
        manual_max_layout.addWidget(self.manual_max_slider)
        manual_max_layout.addWidget(self.manual_max_label)

        display_form.addRow("Main view mode", self.view_mode_combo)
        display_form.addRow("Colormap", self.colormap_combo)
        display_form.addRow("", self.auto_contrast_check)
        display_form.addRow("Auto low percentile", self.auto_low_spin)
        display_form.addRow("Auto high percentile", self.auto_high_spin)
        display_form.addRow("Manual min", manual_min_widget)
        display_form.addRow("Manual max", manual_max_widget)
        display_form.addRow("Image range", self.display_range_hint_label)
        display_form.addRow("", self.sync_profile_display_check)

        display_group.setLayout(display_form)

        # ========== ROI ==========
        roi_group = QGroupBox("ROI")
        roi_form = QFormLayout()
        roi_form.setLabelAlignment(Qt.AlignRight)
        roi_form.setFormAlignment(Qt.AlignTop)
        roi_form.setHorizontalSpacing(8)
        roi_form.setVerticalSpacing(6)


        self.enable_roi_check = QCheckBox("Enable ROI crop")
        self.enable_roi_check.setChecked(True)
        self.enable_roi_check.setToolTip(
            "When enabled, process only the ROI below. "
            "When unchecked, ROI x/y/width/height are disabled and the full input image is processed."
        )

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 100000)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 100000)

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 100000)

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 100000)

        roi_form.addRow("", self.enable_roi_check)
        roi_form.addRow("ROI x", self.roi_x_spin)
        roi_form.addRow("ROI y", self.roi_y_spin)
        roi_form.addRow("ROI width", self.roi_w_spin)
        roi_form.addRow("ROI height", self.roi_h_spin)
        roi_group.setLayout(roi_form)

        # ========== 预处理 ==========
        prep_group = QGroupBox("Preprocess")
        prep_form = QFormLayout()
        prep_form.setLabelAlignment(Qt.AlignRight)
        prep_form.setFormAlignment(Qt.AlignTop)
        prep_form.setHorizontalSpacing(8)
        prep_form.setVerticalSpacing(6)


        self.enable_preprocess_check = QCheckBox("Enable preprocess")
        self.enable_preprocess_check.setChecked(True)

        self.phase_model_combo = QComboBox()
        self.phase_model_combo.addItems([
            "0/120/240",
            "0/90/180",
        ])
        self.phase_model_combo.setToolTip(
            "Choose the actual phase shifts of each three-frame orientation group. "
            "Use 0/120/240 for standard 3-phase SIM/OS-SIM; use 0/90/180 only if your acquisition truly used those phases."
        )

        self.preprocess_mode_combo = QComboBox()
        self.preprocess_mode_combo.addItems([
            "shared_percentile",
            "dark_only",
            "dark_flat",
            "none",
        ])
        self.preprocess_mode_combo.setToolTip(
            "shared_percentile subtracts one common percentile background per 3-phase group; "
            "dark_only subtracts the Dark image; dark_flat subtracts Dark and divides by normalized Flat-Dark."
        )

        self.bg_percentile_spin = QDoubleSpinBox()
        self.bg_percentile_spin.setRange(0.0, 100.0)
        self.bg_percentile_spin.setDecimals(2)
        self.bg_percentile_spin.setSingleStep(1.0)

        self.phase_norm_mode_combo = QComboBox()
        self.phase_norm_mode_combo.addItems(["per_orientation", "global"])

        self.phase_norm_stat_combo = QComboBox()
        self.phase_norm_stat_combo.addItems(["mean", "median", "p95"])

        prep_form.addRow("", self.enable_preprocess_check)
        self.os_sim_recon_mode_combo = QComboBox()
        self.os_sim_recon_mode_combo.addItems([
            "Spatial demodulation (standard OS-SIM)",
            "Fourier-OS-SIM filtered demodulation",
        ])
        self.os_sim_recon_mode_combo.setToolTip(
            "Choose the per-orientation OS-SIM reconstruction branch. Fourier-OS-SIM filters the demodulated quadrature components in frequency space before amplitude reconstruction."
        )

        self.fourier_filter_mode_combo = QComboBox()
        self.fourier_filter_mode_combo.addItems([
            "Gaussian low-pass",
            "Butterworth low-pass",
            "None",
        ])

        self.fourier_cutoff_spin = QDoubleSpinBox()
        self.fourier_cutoff_spin.setRange(0.0, 0.50)
        self.fourier_cutoff_spin.setDecimals(4)
        self.fourier_cutoff_spin.setSingleStep(0.005)
        self.fourier_cutoff_spin.setValue(0.0)
        self.fourier_cutoff_spin.setSpecialValueText("Auto")
        self.fourier_cutoff_spin.setToolTip("Fourier-OS-SIM cutoff in cycles/pixel. Set to 0 / Auto to estimate from stripe FFT.")

        self.fourier_order_spin = QSpinBox()
        self.fourier_order_spin.setRange(1, 8)
        self.fourier_order_spin.setValue(2)
        self.fourier_order_spin.setToolTip("Butterworth order for the Fourier-OS-SIM branch.")

        self.fourier_amount_spin = QDoubleSpinBox()
        self.fourier_amount_spin.setRange(0.0, 1.0)
        self.fourier_amount_spin.setDecimals(2)
        self.fourier_amount_spin.setSingleStep(0.05)
        self.fourier_amount_spin.setValue(1.0)
        self.fourier_amount_spin.setToolTip("Blend strength for Fourier filtering: 0 keeps spatial OS-SIM, 1 uses filtered demodulation.")

        prep_form.addRow("Phase model", self.phase_model_combo)
        prep_form.addRow("Preprocess mode", self.preprocess_mode_combo)
        prep_form.addRow("Background percentile", self.bg_percentile_spin)
        prep_form.addRow("Phase normalization", self.phase_norm_mode_combo)
        prep_form.addRow("Normalization statistic", self.phase_norm_stat_combo)
        prep_form.addRow("OS-SIM branch", self.os_sim_recon_mode_combo)
        prep_form.addRow("Fourier filter", self.fourier_filter_mode_combo)
        prep_form.addRow("Fourier cutoff", self.fourier_cutoff_spin)
        prep_form.addRow("Fourier order", self.fourier_order_spin)
        prep_form.addRow("Fourier amount", self.fourier_amount_spin)
        prep_group.setLayout(prep_form)

        # ========== 小波融合 ==========
        wave_group = QGroupBox("Wavelet Fusion")
        wave_form = QFormLayout()
        wave_form.setLabelAlignment(Qt.AlignRight)
        wave_form.setFormAlignment(Qt.AlignTop)
        wave_form.setHorizontalSpacing(8)
        wave_form.setVerticalSpacing(6)


        self.fusion_method_combo = QComboBox()
        self.fusion_method_combo.addItems(["DWT", "SWT"])

        self.wavelet_name_combo = QComboBox()
        self.wavelet_name_combo.addItems([
            "haar", "db2", "db4", "sym2", "sym4", "coif1", "bior1.3"
        ])

        self.wavelet_level_spin = QSpinBox()
        self.wavelet_level_spin.setRange(1, 10)

        self.wavelet_detail_mode_combo = QComboBox()
        self.wavelet_detail_mode_combo.addItems([
            "maxabs",
            "mean",
        ])

        self.enable_orientation_match_check = QCheckBox("Match orientation intensity before fusion")
        self.enable_orientation_match_check.setChecked(True)

        self.orientation_match_method_combo = QComboBox()
        self.orientation_match_method_combo.addItems(["median", "mean", "p95"])

        self.final_fusion_mode_combo = QComboBox()
        self.final_fusion_mode_combo.addItems([
            "RMS",
            "Quadrature",
            "Mean",
        ])

        wave_form.addRow("Fusion method", self.fusion_method_combo)
        wave_form.addRow("Wavelet name", self.wavelet_name_combo)
        wave_form.addRow("Wavelet level", self.wavelet_level_spin)
        wave_form.addRow("Wavelet detail fusion", self.wavelet_detail_mode_combo)
        wave_form.addRow("", self.enable_orientation_match_check)
        wave_form.addRow("Match statistic", self.orientation_match_method_combo)
        wave_form.addRow("Final SIM fusion", self.final_fusion_mode_combo)
        wave_group.setLayout(wave_form)

        # ========== 后处理 ==========
        smooth_group = QGroupBox("Postprocess")
        smooth_form = QFormLayout()
        smooth_form.setLabelAlignment(Qt.AlignRight)
        smooth_form.setFormAlignment(Qt.AlignTop)
        smooth_form.setHorizontalSpacing(8)
        smooth_form.setVerticalSpacing(6)


        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.0, 20.0)
        self.gaussian_sigma_spin.setDecimals(2)
        self.gaussian_sigma_spin.setSingleStep(0.1)

        self.enable_smoothing_check = QCheckBox("Enable Gaussian smoothing")

        self.enable_wiener_filter_check = QCheckBox("Enable Wiener filter")
        self.enable_wiener_filter_check.setToolTip(
            "Apply a local Wiener denoising filter after structure enhancement and Gaussian smoothing. "
            "Noise variance = 0 uses automatic estimation from local variance."
        )

        self.wiener_target_combo = QComboBox()
        self.wiener_target_combo.addItems([
            "Final SIM only",
            "Final SIM + Wavelet fused",
            "Final SIM + Wavelet fused + HiLo Hybrid",
        ])

        self.wiener_kernel_spin = QSpinBox()
        self.wiener_kernel_spin.setRange(3, 99)
        self.wiener_kernel_spin.setSingleStep(2)
        self.wiener_kernel_spin.setValue(5)
        self.wiener_kernel_spin.setToolTip("Local Wiener window size. Even values are corrected to the next odd value internally.")

        self.wiener_noise_spin = QDoubleSpinBox()
        self.wiener_noise_spin.setRange(0.0, 1e12)
        self.wiener_noise_spin.setDecimals(6)
        self.wiener_noise_spin.setSingleStep(0.001)
        self.wiener_noise_spin.setValue(0.0)
        self.wiener_noise_spin.setSpecialValueText("Auto")
        self.wiener_noise_spin.setToolTip("Noise variance. Set to 0 / Auto to estimate it from the local variance map.")

        self.wiener_amount_spin = QDoubleSpinBox()
        self.wiener_amount_spin.setRange(0.0, 1.0)
        self.wiener_amount_spin.setDecimals(2)
        self.wiener_amount_spin.setSingleStep(0.05)
        self.wiener_amount_spin.setValue(1.0)
        self.wiener_amount_spin.setToolTip("Blend strength: 0 keeps the input, 1 uses the full Wiener-filtered image.")

        self.enable_output_intensity_match_check = QCheckBox("Preserve output intensity / brightness match")
        self.enable_output_intensity_match_check.setChecked(True)
        self.enable_output_intensity_match_check.setToolTip(
            "Match final SIM / HiLo output intensity range back to a reference image. "
            "Useful when RMS fusion or HiLo high-pass splitting makes the final image too dim."
        )

        self.output_intensity_target_combo = QComboBox()
        self.output_intensity_target_combo.addItems([
            "Final SIM only",
            "Final SIM + Wavelet fused",
            "Final SIM + Wavelet fused + HiLo Hybrid",
        ])

        self.output_intensity_reference_combo = QComboBox()
        self.output_intensity_reference_combo.addItems([
            "Widefield fused",
            "Raw widefield fused",
            "External raw widefield if available",
        ])

        self.output_intensity_method_combo = QComboBox()
        self.output_intensity_method_combo.addItems([
            "P95",
            "P99",
            "P90",
            "Median",
            "Mean",
            "Max",
        ])
        self.output_intensity_method_combo.setToolTip(
            "Statistic used for brightness matching. P95 or P99 is usually safer than Max."
        )

        self.output_intensity_strength_spin = QDoubleSpinBox()
        self.output_intensity_strength_spin.setRange(0.0, 1.0)
        self.output_intensity_strength_spin.setDecimals(2)
        self.output_intensity_strength_spin.setSingleStep(0.05)
        self.output_intensity_strength_spin.setValue(0.85)
        self.output_intensity_strength_spin.setToolTip(
            "0 keeps current output; 1 fully matches the reference brightness statistic."
        )

        self.output_intensity_max_scale_spin = QDoubleSpinBox()
        self.output_intensity_max_scale_spin.setRange(1.0, 20.0)
        self.output_intensity_max_scale_spin.setDecimals(1)
        self.output_intensity_max_scale_spin.setSingleStep(0.5)
        self.output_intensity_max_scale_spin.setValue(5.0)
        self.output_intensity_max_scale_spin.setToolTip(
            "Maximum allowed brightness scale to avoid over-amplifying noise."
        )

        self.enable_sim_bg_correction_check = QCheckBox("Enable single-SIM low-frequency background correction")
        self.enable_sim_bg_correction_check.setChecked(True)

        self.sim_bg_mode_combo = QComboBox()
        self.sim_bg_mode_combo.addItems(["subtract", "divide", "hybrid"])

        self.sim_bg_sigma_spin = QDoubleSpinBox()
        self.sim_bg_sigma_spin.setRange(1.0, 200.0)
        self.sim_bg_sigma_spin.setDecimals(1)
        self.sim_bg_sigma_spin.setSingleStep(1.0)
        self.sim_bg_sigma_spin.setValue(25.0)

        self.enable_final_structure_enhance_check = QCheckBox("Enable final SIM structure enhancement")
        self.enable_final_structure_enhance_check.setChecked(True)

        self.final_structure_target_combo = QComboBox()
        self.final_structure_target_combo.addItems([
            "Final SIM only",
            "Final SIM + Wavelet fused",
            "Final SIM + Wavelet fused + HiLo Hybrid",
        ])

        self.final_structure_mode_combo = QComboBox()
        self.final_structure_mode_combo.addItems([
            "highpass_mix",
            "subtract",
            "divide",
            "hybrid",
        ])

        self.final_structure_sigma_spin = QDoubleSpinBox()
        self.final_structure_sigma_spin.setRange(1.0, 300.0)
        self.final_structure_sigma_spin.setDecimals(1)
        self.final_structure_sigma_spin.setSingleStep(1.0)
        self.final_structure_sigma_spin.setValue(35.0)

        self.final_structure_amount_spin = QDoubleSpinBox()
        self.final_structure_amount_spin.setRange(0.0, 5.0)
        self.final_structure_amount_spin.setDecimals(2)
        self.final_structure_amount_spin.setSingleStep(0.05)
        self.final_structure_amount_spin.setValue(1.0)

        self.final_structure_keep_spin = QDoubleSpinBox()
        self.final_structure_keep_spin.setRange(0.0, 1.0)
        self.final_structure_keep_spin.setDecimals(2)
        self.final_structure_keep_spin.setSingleStep(0.05)
        self.final_structure_keep_spin.setValue(0.15)

        self.enable_hilo_mode_check = QCheckBox("Enable HiLo mode / OS-SIM + HiLo hybrid")
        self.enable_hilo_mode_check.setChecked(True)
        self.enable_hilo_mode_check.setToolTip(
            "Create a HiLo-style hybrid image: Lo = low-pass OS-SIM, "
            "Mid/High = band-pass/high-pass widefield, Hybrid = Lo + Mid + High."
        )

        self.hilo_strategy_combo = QComboBox()
        self.hilo_strategy_combo.addItems([
            "Per-orientation OS-SIM/pWF then RMS fuse (recommended)",
            "Fused sources then HiLo (legacy)",
        ])
        self.hilo_strategy_combo.setToolTip(
            "Recommended: compute HiLo per orientation from matched pWF + OS-SIM, "
            "then RMS-fuse the HiLo orientation results. Legacy: compute HiLo once from fused sources."
        )

        self.hilo_auto_sigma_check = QCheckBox("Auto HiLo sigmas from stripe FFT")
        self.hilo_auto_sigma_check.setChecked(True)
        self.hilo_auto_sigma_check.setToolTip(
            "When enabled, estimate Lo/Mid/High sigmas from the processed phase FFT stripe period. "
            "If the stripe peak is unreliable, the manual sigma values below are kept."
        )

        self.hilo_widefield_source_combo = QComboBox()
        self.hilo_widefield_source_combo.addItems([
            "Pseudo-widefield fused",
            "Raw widefield from ROI",
            "External raw widefield if available",
        ])

        self.hilo_low_source_combo = QComboBox()
        self.hilo_low_source_combo.addItems([
            "Final SIM fused",
            "SIM wavelet fused",
        ])

        self.hilo_sigma_spin = QDoubleSpinBox()
        self.hilo_sigma_spin.setRange(0.1, 100.0)
        self.hilo_sigma_spin.setDecimals(2)
        self.hilo_sigma_spin.setSingleStep(0.1)
        self.hilo_sigma_spin.setValue(2.0)
        self.hilo_sigma_spin.setToolTip("Lo component Gaussian sigma applied to the OS-SIM source.")

        self.hilo_mid_sigma_spin = QDoubleSpinBox()
        self.hilo_mid_sigma_spin.setRange(0.1, 100.0)
        self.hilo_mid_sigma_spin.setDecimals(2)
        self.hilo_mid_sigma_spin.setSingleStep(0.1)
        self.hilo_mid_sigma_spin.setValue(3.0)
        self.hilo_mid_sigma_spin.setToolTip("Mid-band upper Gaussian sigma. Larger values include broader structures in the HiLo mid-frequency band.")

        self.hilo_high_sigma_spin = QDoubleSpinBox()
        self.hilo_high_sigma_spin.setRange(0.1, 100.0)
        self.hilo_high_sigma_spin.setDecimals(2)
        self.hilo_high_sigma_spin.setSingleStep(0.1)
        self.hilo_high_sigma_spin.setValue(1.0)
        self.hilo_high_sigma_spin.setToolTip("High-band Gaussian sigma. Smaller values keep only finer details in the HiLo high-frequency band.")

        self.hilo_mid_weight_spin = QDoubleSpinBox()
        self.hilo_mid_weight_spin.setRange(-10.0, 10.0)
        self.hilo_mid_weight_spin.setDecimals(2)
        self.hilo_mid_weight_spin.setSingleStep(0.05)
        self.hilo_mid_weight_spin.setValue(0.50)
        self.hilo_mid_weight_spin.setToolTip("Weight of the HiLo mid-frequency band.")

        self.hilo_lo_weight_spin = QDoubleSpinBox()
        self.hilo_lo_weight_spin.setRange(0.0, 10.0)
        self.hilo_lo_weight_spin.setDecimals(2)
        self.hilo_lo_weight_spin.setSingleStep(0.05)
        self.hilo_lo_weight_spin.setValue(1.0)

        self.hilo_hi_weight_spin = QDoubleSpinBox()
        self.hilo_hi_weight_spin.setRange(0.0, 10.0)
        self.hilo_hi_weight_spin.setDecimals(2)
        self.hilo_hi_weight_spin.setSingleStep(0.05)
        self.hilo_hi_weight_spin.setValue(1.0)

        self.hilo_clip_output_check = QCheckBox("Clip HiLo output to non-negative")
        self.hilo_clip_output_check.setChecked(True)

        self.enable_hilo_denoise_check = QCheckBox("Enable HiLo-only NLM/BM3D denoise")
        self.enable_hilo_denoise_check.setChecked(False)
        self.enable_hilo_denoise_check.setToolTip(
            "Apply a denoiser only to the HiLo Hybrid after Lo/Mid/High fusion and before the global final postprocess."
        )

        self.hilo_denoise_method_combo = QComboBox()
        self.hilo_denoise_method_combo.addItems([
            "NLM",
            "BM3D package if available / BM3D-like fallback",
            "Wavelet shrinkage (BM3D-like fallback)",
        ])

        self.hilo_nlm_h_spin = QDoubleSpinBox()
        self.hilo_nlm_h_spin.setRange(0.0, 100.0)
        self.hilo_nlm_h_spin.setDecimals(1)
        self.hilo_nlm_h_spin.setSingleStep(0.5)
        self.hilo_nlm_h_spin.setValue(8.0)
        self.hilo_nlm_h_spin.setToolTip("NLM denoise strength on an internal 8-bit display-normalized copy. Larger values remove more speckle/noise.")

        self.hilo_nlm_template_spin = QSpinBox()
        self.hilo_nlm_template_spin.setRange(3, 21)
        self.hilo_nlm_template_spin.setSingleStep(2)
        self.hilo_nlm_template_spin.setValue(7)

        self.hilo_nlm_search_spin = QSpinBox()
        self.hilo_nlm_search_spin.setRange(5, 51)
        self.hilo_nlm_search_spin.setSingleStep(2)
        self.hilo_nlm_search_spin.setValue(21)

        self.hilo_bm3d_sigma_spin = QDoubleSpinBox()
        self.hilo_bm3d_sigma_spin.setRange(0.0, 1e9)
        self.hilo_bm3d_sigma_spin.setDecimals(4)
        self.hilo_bm3d_sigma_spin.setSingleStep(0.1)
        self.hilo_bm3d_sigma_spin.setValue(0.0)
        self.hilo_bm3d_sigma_spin.setSpecialValueText("Auto")
        self.hilo_bm3d_sigma_spin.setToolTip("Noise sigma for BM3D / wavelet fallback. Set 0 / Auto for robust MAD estimate.")

        self.hilo_denoise_amount_spin = QDoubleSpinBox()
        self.hilo_denoise_amount_spin.setRange(0.0, 1.0)
        self.hilo_denoise_amount_spin.setDecimals(2)
        self.hilo_denoise_amount_spin.setSingleStep(0.05)
        self.hilo_denoise_amount_spin.setValue(0.60)
        self.hilo_denoise_amount_spin.setToolTip("Blend strength: 0 keeps HiLo input, 1 uses full denoised HiLo output.")

        sim_bg_preset_widget = QWidget()
        sim_bg_preset_layout = QHBoxLayout(sim_bg_preset_widget)
        sim_bg_preset_layout.setContentsMargins(0, 0, 0, 0)
        self.sim_bg_preset_btn_15 = QPushButton("15")
        self.sim_bg_preset_btn_25 = QPushButton("25")
        self.sim_bg_preset_btn_40 = QPushButton("40")
        self.sim_bg_preset_btn_60 = QPushButton("60")
        sim_bg_preset_layout.addWidget(self.sim_bg_preset_btn_15)
        sim_bg_preset_layout.addWidget(self.sim_bg_preset_btn_25)
        sim_bg_preset_layout.addWidget(self.sim_bg_preset_btn_40)
        sim_bg_preset_layout.addWidget(self.sim_bg_preset_btn_60)

        smooth_form.addRow("Gaussian sigma", self.gaussian_sigma_spin)
        smooth_form.addRow("", self.enable_smoothing_check)
        smooth_form.addRow("", self.enable_wiener_filter_check)
        smooth_form.addRow("Wiener target", self.wiener_target_combo)
        smooth_form.addRow("Wiener kernel", self.wiener_kernel_spin)
        smooth_form.addRow("Wiener noise var", self.wiener_noise_spin)
        smooth_form.addRow("Wiener amount", self.wiener_amount_spin)

        smooth_form.addRow("", self.enable_output_intensity_match_check)
        smooth_form.addRow("Intensity target", self.output_intensity_target_combo)
        smooth_form.addRow("Intensity reference", self.output_intensity_reference_combo)
        smooth_form.addRow("Intensity statistic", self.output_intensity_method_combo)
        smooth_form.addRow("Intensity strength", self.output_intensity_strength_spin)
        smooth_form.addRow("Max intensity scale", self.output_intensity_max_scale_spin)

        smooth_form.addRow("", self.enable_sim_bg_correction_check)
        smooth_form.addRow("SIM bg mode", self.sim_bg_mode_combo)
        smooth_form.addRow("SIM bg sigma", self.sim_bg_sigma_spin)
        smooth_form.addRow("SIM bg presets", sim_bg_preset_widget)
        smooth_form.addRow("", self.enable_final_structure_enhance_check)
        smooth_form.addRow("Final enhance target", self.final_structure_target_combo)
        smooth_form.addRow("Final enhance mode", self.final_structure_mode_combo)
        smooth_form.addRow("Final enhance sigma", self.final_structure_sigma_spin)
        smooth_form.addRow("Final enhance amount", self.final_structure_amount_spin)
        smooth_form.addRow("Final keep original", self.final_structure_keep_spin)
        smooth_form.addRow("", self.enable_hilo_mode_check)
        smooth_form.addRow("HiLo strategy", self.hilo_strategy_combo)
        smooth_form.addRow("", self.hilo_auto_sigma_check)
        smooth_form.addRow("HiLo widefield source", self.hilo_widefield_source_combo)
        smooth_form.addRow("HiLo Lo source", self.hilo_low_source_combo)
        smooth_form.addRow("HiLo Lo sigma", self.hilo_sigma_spin)
        smooth_form.addRow("HiLo mid sigma", self.hilo_mid_sigma_spin)
        smooth_form.addRow("HiLo high sigma", self.hilo_high_sigma_spin)
        smooth_form.addRow("HiLo Lo weight", self.hilo_lo_weight_spin)
        smooth_form.addRow("HiLo mid weight", self.hilo_mid_weight_spin)
        smooth_form.addRow("HiLo high weight", self.hilo_hi_weight_spin)
        smooth_form.addRow("", self.hilo_clip_output_check)
        smooth_form.addRow("", self.enable_hilo_denoise_check)
        smooth_form.addRow("HiLo denoise method", self.hilo_denoise_method_combo)
        smooth_form.addRow("HiLo NLM h", self.hilo_nlm_h_spin)
        smooth_form.addRow("NLM template window", self.hilo_nlm_template_spin)
        smooth_form.addRow("NLM search window", self.hilo_nlm_search_spin)
        smooth_form.addRow("BM3D/noise sigma", self.hilo_bm3d_sigma_spin)
        smooth_form.addRow("HiLo denoise amount", self.hilo_denoise_amount_spin)
        smooth_group.setLayout(smooth_form)

        # ========== 左侧 Dock 紧凑宽度约束 ==========
        # Important: long paths and long ComboBox items should not expand the left dock.
        self._apply_left_dock_compact_layout(
            groups=[
                path_group, roi_group, display_group, mode_group,
                prep_group, wave_group, smooth_group,
            ],
            forms=[
                path_form, roi_form, display_form, mode_form,
                prep_form, wave_form, smooth_form,
            ],
            line_edits=[
                self.input_dir_edit,
                self.single_tiff_path_edit,
                self.raw_widefield_path_edit,
                self.dark_path_edit,
                self.flat_path_edit,
            ],
            browse_buttons=[
                input_btn, single_tiff_btn,
                raw_wf_btn, dark_btn, flat_btn,
            ],
            clear_buttons=[
                single_tiff_clear_btn, raw_wf_clear_btn,
                dark_clear_btn, flat_clear_btn,
            ],
            combos=[
                self.ext_combo,
                self.image_order_combo,
                self.recon_mode_combo,
                self.view_mode_combo,
                self.colormap_combo,
                self.phase_model_combo,
                self.preprocess_mode_combo,
                self.phase_norm_mode_combo,
                self.phase_norm_stat_combo,
                self.os_sim_recon_mode_combo,
                self.fourier_filter_mode_combo,
                self.fusion_method_combo,
                self.wavelet_name_combo,
                self.wavelet_detail_mode_combo,
                self.orientation_match_method_combo,
                self.final_fusion_mode_combo,
                self.wiener_target_combo,
                self.output_intensity_target_combo,
                self.output_intensity_reference_combo,
                self.output_intensity_method_combo,
                self.sim_bg_mode_combo,
                self.final_structure_target_combo,
                self.final_structure_mode_combo,
                self.hilo_strategy_combo,
                self.hilo_widefield_source_combo,
                self.hilo_low_source_combo,
                self.hilo_denoise_method_combo,
            ],
            spin_boxes=[
                self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin,
                self.auto_low_spin, self.auto_high_spin,
                self.bg_percentile_spin,
                self.fourier_cutoff_spin, self.fourier_order_spin, self.fourier_amount_spin,
                self.wavelet_level_spin,
                self.gaussian_sigma_spin,
                self.wiener_kernel_spin, self.wiener_noise_spin, self.wiener_amount_spin,
                self.output_intensity_strength_spin, self.output_intensity_max_scale_spin,
                self.sim_bg_sigma_spin,
                self.final_structure_sigma_spin, self.final_structure_amount_spin, self.final_structure_keep_spin,
                self.hilo_sigma_spin, self.hilo_mid_sigma_spin, self.hilo_high_sigma_spin,
                self.hilo_lo_weight_spin, self.hilo_mid_weight_spin, self.hilo_hi_weight_spin,
                self.hilo_nlm_h_spin, self.hilo_nlm_template_spin, self.hilo_nlm_search_spin,
                self.hilo_bm3d_sigma_spin, self.hilo_denoise_amount_spin,
            ],
            labels=[
                self.image_order_hint_label,
                self.recon_mode_hint_label,
                self.display_range_hint_label,
                self.manual_min_label,
                self.manual_max_label,
            ],
            row_widgets=[
                input_row_widget,
                single_tiff_row_widget,
                raw_wf_row_widget,
                dark_row_widget,
                flat_row_widget,
                manual_min_widget,
                manual_max_widget,
                sim_bg_preset_widget,
            ],
        )

        # ========== 按钮 ==========
        btn_layout1 = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_processing)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)

        self.refresh_btn = QPushButton("Refresh Preview")
        self.refresh_btn.clicked.connect(self.refresh_preview)

        self.view_mode_combo.currentIndexChanged.connect(self.update_plot)
        self.colormap_combo.currentIndexChanged.connect(self.update_plot)
        self.auto_contrast_check.toggled.connect(self._on_display_control_changed)
        self.auto_low_spin.valueChanged.connect(self._on_display_control_changed)
        self.auto_high_spin.valueChanged.connect(self._on_display_control_changed)
        self.manual_min_slider.valueChanged.connect(self._on_display_control_changed)
        self.manual_max_slider.valueChanged.connect(self._on_display_control_changed)
        self.sync_profile_display_check.toggled.connect(self._on_display_control_changed)
        self.raw_widefield_path_edit.editingFinished.connect(self._on_raw_widefield_path_changed)
        self.recon_mode_combo.currentTextChanged.connect(self._on_recon_mode_changed)
        self.enable_roi_check.toggled.connect(self._update_roi_control_state)
        self.enable_roi_check.toggled.connect(self._update_status_bar)
        for roi_widget in (self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin):
            roi_widget.valueChanged.connect(self._update_status_bar)
        self.ext_combo.currentTextChanged.connect(self._update_status_bar)
        self.image_order_combo.currentTextChanged.connect(self._update_status_bar)
        self.save_ordered_roi_tiff_btn.clicked.connect(self.save_ordered_roi_stack_tiff)

        self.sim_bg_preset_btn_15.clicked.connect(lambda: self.sim_bg_sigma_spin.setValue(15.0))
        self.sim_bg_preset_btn_25.clicked.connect(lambda: self.sim_bg_sigma_spin.setValue(25.0))
        self.sim_bg_preset_btn_40.clicked.connect(lambda: self.sim_bg_sigma_spin.setValue(40.0))
        self.sim_bg_preset_btn_60.clicked.connect(lambda: self.sim_bg_sigma_spin.setValue(60.0))
        self._connect_reconstruction_param_signals()
        self._connect_optional_control_state_signals()

        for btn in (self.run_btn, self.save_btn, self.refresh_btn):
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(126)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_layout1.addWidget(self.run_btn)
        btn_layout1.addWidget(self.save_btn)
        btn_layout1.addWidget(self.refresh_btn)

        btn_layout2 = QHBoxLayout()
        self.profile_btn = QPushButton("Profile")
        self.profile_btn.setToolTip("Show Profile Tab")
        self.profile_btn.clicked.connect(self.open_profile_window)

        self.save_layout_btn = QPushButton("Save Layout")
        self.save_layout_btn.setToolTip("Save Params/Layout...")
        self.save_layout_btn.clicked.connect(lambda: self.save_params_and_layout(show_message=True))

        self.load_layout_btn = QPushButton("Load Layout")
        self.load_layout_btn.setToolTip("Load Params/Layout...")
        self.load_layout_btn.clicked.connect(lambda: self.load_params_and_layout(show_message=True))

        for btn in (self.profile_btn, self.save_layout_btn, self.load_layout_btn):
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(126)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_layout2.addWidget(self.profile_btn)
        btn_layout2.addWidget(self.save_layout_btn)
        btn_layout2.addWidget(self.load_layout_btn)

        # ========== 日志 ==========
        log_group = QGroupBox("Log / Status")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # ========== Compact tabbed parameter layout ==========
        # The original version stacked all parameter groups vertically, which made
        # the control panel crowded after adding final SIM structure enhancement.
        # Here the same controls are split into tabs while preserving all object names
        # and signal connections used elsewhere in the program.

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        def make_tab(groups):
            page = QWidget()
            page.setMinimumWidth(0)
            page.setMaximumWidth(max(280, int(getattr(self, "_left_panel_max_width", 430)) - 28))
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            page_layout.setSpacing(6)
            for group in groups:
                page_layout.addWidget(group)
            page_layout.addStretch()

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            return scroll

        tabs.addTab(make_tab([path_group, roi_group]), "Input / ROI")
        tabs.addTab(make_tab([display_group]), "Display")
        tabs.addTab(make_tab([mode_group, prep_group, wave_group]), "Reconstruct")
        tabs.addTab(make_tab([smooth_group]), "Postprocess")

        button_panel = QWidget()
        button_panel_layout = QVBoxLayout(button_panel)
        button_panel_layout.setContentsMargins(0, 0, 0, 0)
        button_panel_layout.setSpacing(6)
        button_panel_layout.addLayout(btn_layout1)
        button_panel_layout.addLayout(btn_layout2)

        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(tabs, 3)
        layout.addWidget(button_panel, 0)
        layout.addWidget(log_group, 1)

        return widget

    def _build_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Analysis Tabs")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setDocumentMode(True)

        def make_canvas_tab(canvas):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            page_layout.addWidget(canvas)
            return page

        # Keep self.canvas as the primary/final comparison canvas for backward compatibility.
        self.canvas = MplCanvas(self, figsize=(10, 7), dpi=100)
        self.orientation_canvas = MplCanvas(self, figsize=(10, 7), dpi=100)
        self.hilo_canvas = MplCanvas(self, figsize=(10, 7), dpi=100)

        self.analysis_tabs.addTab(make_canvas_tab(self.canvas), "Final Comparison")
        self.analysis_tabs.addTab(make_canvas_tab(self.orientation_canvas), "Orientation View")
        self.analysis_tabs.addTab(make_canvas_tab(self.hilo_canvas), "HiLo Components")

        self.profile_window = ProfileComparisonWindow(self)
        self.profile_window.setWindowFlags(Qt.Widget)
        self.analysis_tabs.addTab(self.profile_window, "Profile Window")

        self.analysis_tabs.currentChanged.connect(self._on_analysis_tab_changed)

        layout.addWidget(title)
        layout.addWidget(self.analysis_tabs)

        return widget

    def _set_default_values(self):
        self.enable_roi_check.setChecked(True)
        self.roi_x_spin.setValue(832)
        self.roi_y_spin.setValue(52)
        self.roi_w_spin.setValue(1370)
        self.roi_h_spin.setValue(1830)

        self.phase_model_combo.setCurrentText("0/120/240")
        self.preprocess_mode_combo.setCurrentText("shared_percentile")
        self.bg_percentile_spin.setValue(5.0)
        self.phase_norm_mode_combo.setCurrentText("per_orientation")
        self.phase_norm_stat_combo.setCurrentText("mean")
        self.os_sim_recon_mode_combo.setCurrentText("Spatial demodulation (standard OS-SIM)")
        self.fourier_filter_mode_combo.setCurrentText("Gaussian low-pass")
        self.fourier_cutoff_spin.setValue(0.0)
        self.fourier_order_spin.setValue(2)
        self.fourier_amount_spin.setValue(1.0)

        self.view_mode_combo.setCurrentText("Final comparison view")
        self.colormap_combo.setCurrentText("gray")
        self.auto_contrast_check.setChecked(True)
        self.auto_low_spin.setValue(1.0)
        self.auto_high_spin.setValue(99.0)
        self.manual_min_slider.setValue(0)
        self.manual_max_slider.setValue(1000)
        self._update_roi_control_state()
        self.sync_profile_display_check.setChecked(True)
        self._update_display_control_state()
        self._update_manual_labels()

        self.fusion_method_combo.setCurrentText("SWT")
        self.wavelet_name_combo.setCurrentText("db2")
        self.wavelet_level_spin.setValue(2)
        self.wavelet_detail_mode_combo.setCurrentText("maxabs")
        self.enable_orientation_match_check.setChecked(True)
        self.orientation_match_method_combo.setCurrentText("median")
        self.final_fusion_mode_combo.setCurrentText("RMS")

        self.gaussian_sigma_spin.setValue(1.0)
        self.enable_smoothing_check.setChecked(True)
        self.enable_wiener_filter_check.setChecked(False)
        self.wiener_target_combo.setCurrentText("Final SIM only")
        self.wiener_kernel_spin.setValue(5)
        self.wiener_noise_spin.setValue(0.0)
        self.wiener_amount_spin.setValue(1.0)
        self.enable_output_intensity_match_check.setChecked(True)
        self.output_intensity_target_combo.setCurrentText("Final SIM + Wavelet fused + HiLo Hybrid")
        self.output_intensity_reference_combo.setCurrentText("Widefield fused")
        self.output_intensity_method_combo.setCurrentText("P95")
        self.output_intensity_strength_spin.setValue(0.85)
        self.output_intensity_max_scale_spin.setValue(5.0)

        self.enable_sim_bg_correction_check.setChecked(True)
        self.sim_bg_mode_combo.setCurrentText("subtract")
        self.sim_bg_sigma_spin.setValue(25.0)

        self.enable_final_structure_enhance_check.setChecked(True)
        self.final_structure_target_combo.setCurrentText("Final SIM only")
        self.final_structure_mode_combo.setCurrentText("highpass_mix")
        self.final_structure_sigma_spin.setValue(35.0)
        self.final_structure_amount_spin.setValue(1.0)
        self.final_structure_keep_spin.setValue(0.15)
        self.enable_hilo_mode_check.setChecked(True)
        self.hilo_strategy_combo.setCurrentText("Per-orientation OS-SIM/pWF then RMS fuse (recommended)")
        self.hilo_auto_sigma_check.setChecked(True)
        self.hilo_widefield_source_combo.setCurrentText("Pseudo-widefield fused")
        self.hilo_low_source_combo.setCurrentText("Final SIM fused")
        self.hilo_sigma_spin.setValue(2.0)
        self.hilo_mid_sigma_spin.setValue(3.0)
        self.hilo_high_sigma_spin.setValue(1.0)
        self.hilo_lo_weight_spin.setValue(1.0)
        self.hilo_mid_weight_spin.setValue(0.50)
        self.hilo_hi_weight_spin.setValue(1.0)
        self.hilo_clip_output_check.setChecked(True)
        self.enable_hilo_denoise_check.setChecked(False)
        self.hilo_denoise_method_combo.setCurrentText("NLM")
        self.hilo_nlm_h_spin.setValue(8.0)
        self.hilo_nlm_template_spin.setValue(7)
        self.hilo_nlm_search_spin.setValue(21)
        self.hilo_bm3d_sigma_spin.setValue(0.0)
        self.hilo_denoise_amount_spin.setValue(0.60)

        self.input_dir_edit.setText("./images")
        self.single_tiff_path_edit.setText("")
        self.raw_widefield_path_edit.setText("")
        self.dark_path_edit.setText("")
        self.flat_path_edit.setText("")
        self.image_order_combo.setCurrentIndex(0)
        self.config_path = os.path.join(os.getcwd(), "sim_gui_config.json")
        self._last_param_config_dir = os.getcwd()
        self._last_param_config_path = self.config_path

        self._applying_recon_mode = False
        self.recon_mode_combo.setCurrentText("Enhanced mode")
        self.apply_reconstruction_mode("Enhanced mode")
        self._update_optional_control_states()
        self._update_status_bar()

    def _connect_optional_control_state_signals(self):
        widgets = [
            self.enable_preprocess_check,
            self.os_sim_recon_mode_combo,
            self.enable_orientation_match_check,
            self.enable_smoothing_check,
            self.enable_wiener_filter_check,
            self.enable_output_intensity_match_check,
            self.enable_sim_bg_correction_check,
            self.enable_final_structure_enhance_check,
            self.enable_hilo_mode_check,
            self.hilo_auto_sigma_check,
            self.enable_hilo_denoise_check,
            self.hilo_denoise_method_combo,
        ]
        for widget in widgets:
            if hasattr(widget, "toggled"):
                widget.toggled.connect(self._update_optional_control_states)
            elif hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._update_optional_control_states)

    def _update_optional_control_states(self, *args):
        if not hasattr(self, "recon_mode_combo"):
            return

        custom_mode = self.recon_mode_combo.currentText() == "Custom"

        def set_widgets_enabled(widgets, enabled):
            for widget in widgets:
                if widget is not None:
                    widget.setEnabled(bool(custom_mode and enabled))

        preprocess_on = self.enable_preprocess_check.isChecked()
        set_widgets_enabled(
            [
                self.preprocess_mode_combo,
                self.bg_percentile_spin,
                self.phase_norm_mode_combo,
                self.phase_norm_stat_combo,
            ],
            preprocess_on,
        )

        fourier_on = "Fourier" in self.os_sim_recon_mode_combo.currentText()
        set_widgets_enabled(
            [
                self.fourier_filter_mode_combo,
                self.fourier_cutoff_spin,
                self.fourier_order_spin,
                self.fourier_amount_spin,
            ],
            fourier_on,
        )

        set_widgets_enabled([self.orientation_match_method_combo], self.enable_orientation_match_check.isChecked())
        set_widgets_enabled([self.gaussian_sigma_spin], self.enable_smoothing_check.isChecked())
        set_widgets_enabled(
            [
                self.wiener_target_combo,
                self.wiener_kernel_spin,
                self.wiener_noise_spin,
                self.wiener_amount_spin,
            ],
            self.enable_wiener_filter_check.isChecked(),
        )
        set_widgets_enabled(
            [
                self.output_intensity_target_combo,
                self.output_intensity_reference_combo,
                self.output_intensity_method_combo,
                self.output_intensity_strength_spin,
                self.output_intensity_max_scale_spin,
            ],
            self.enable_output_intensity_match_check.isChecked(),
        )
        set_widgets_enabled(
            [
                self.sim_bg_mode_combo,
                self.sim_bg_sigma_spin,
                self.sim_bg_preset_btn_15,
                self.sim_bg_preset_btn_25,
                self.sim_bg_preset_btn_40,
                self.sim_bg_preset_btn_60,
            ],
            self.enable_sim_bg_correction_check.isChecked(),
        )
        set_widgets_enabled(
            [
                self.final_structure_target_combo,
                self.final_structure_mode_combo,
                self.final_structure_sigma_spin,
                self.final_structure_amount_spin,
                self.final_structure_keep_spin,
            ],
            self.enable_final_structure_enhance_check.isChecked(),
        )

        hilo_on = self.enable_hilo_mode_check.isChecked()
        hilo_auto = self.hilo_auto_sigma_check.isChecked()
        set_widgets_enabled(
            [
                self.hilo_strategy_combo,
                self.hilo_auto_sigma_check,
                self.hilo_widefield_source_combo,
                self.hilo_low_source_combo,
                self.hilo_lo_weight_spin,
                self.hilo_mid_weight_spin,
                self.hilo_hi_weight_spin,
                self.hilo_clip_output_check,
                self.enable_hilo_denoise_check,
            ],
            hilo_on,
        )
        set_widgets_enabled(
            [
                self.hilo_sigma_spin,
                self.hilo_mid_sigma_spin,
                self.hilo_high_sigma_spin,
            ],
            hilo_on and not hilo_auto,
        )

        hilo_denoise_on = hilo_on and self.enable_hilo_denoise_check.isChecked()
        denoise_method = self.hilo_denoise_method_combo.currentText()
        nlm_on = hilo_denoise_on and "NLM" in denoise_method
        bm3d_on = hilo_denoise_on and "BM3D" in denoise_method
        set_widgets_enabled([self.hilo_denoise_method_combo, self.hilo_denoise_amount_spin], hilo_denoise_on)
        set_widgets_enabled([self.hilo_nlm_h_spin, self.hilo_nlm_template_spin, self.hilo_nlm_search_spin], nlm_on)
        set_widgets_enabled([self.hilo_bm3d_sigma_spin], bm3d_on)

    def _build_status_bar(self):
        self.status_label = QLabel("Ready")
        self.status_gpu_label = QLabel("")
        self.status_input_label = QLabel("")
        self.status_output_label = QLabel("")
        self.status_roi_label = QLabel("")
        self.status_image_label = QLabel("")
        self.status_elapsed_label = QLabel("")

        for label in (
            self.status_label,
            self.status_gpu_label,
            self.status_input_label,
            self.status_output_label,
            self.status_roi_label,
            self.status_image_label,
            self.status_elapsed_label,
        ):
            label.setMinimumWidth(90)
            label.setToolTip(label.text())

        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.status_gpu_label)
        self.statusBar().addPermanentWidget(self.status_input_label)
        self.statusBar().addPermanentWidget(self.status_output_label)
        self.statusBar().addPermanentWidget(self.status_roi_label)
        self.statusBar().addPermanentWidget(self.status_image_label)
        self.statusBar().addPermanentWidget(self.status_elapsed_label)

    def _short_path_for_status(self, path, max_chars=36):
        path = str(path or "").strip()
        if not path:
            return "(empty)"
        norm = os.path.normpath(path)
        if len(norm) <= max_chars:
            return norm
        base = os.path.basename(norm.rstrip("/\\")) or norm
        parent = os.path.basename(os.path.dirname(norm))
        compact = os.path.join("...", parent, base) if parent else os.path.join("...", base)
        if len(compact) > max_chars:
            compact = "..." + compact[-max_chars + 3:]
        return compact

    def _current_input_for_status(self):
        single_tiff = self.single_tiff_path_edit.text().strip() if hasattr(self, "single_tiff_path_edit") else ""
        if single_tiff:
            return single_tiff
        return self.input_dir_edit.text().strip() if hasattr(self, "input_dir_edit") else ""

    def _set_status_text(self, text):
        if hasattr(self, "status_label"):
            self.status_label.setText(str(text))
            self.status_label.setToolTip(str(text))

    def _update_status_bar(self, *args):
        if not hasattr(self, "status_label"):
            return

        try:
            gpu_text = "GPU: on" if gpu_runtime_available() else "GPU: CPU"
        except Exception:
            gpu_text = "GPU: unknown"

        input_path = self._current_input_for_status()
        output_path = self._get_effective_output_dir()

        if hasattr(self, "enable_roi_check") and self.enable_roi_check.isChecked():
            roi_text = f"ROI: {self.roi_w_spin.value()}x{self.roi_h_spin.value()} @ {self.roi_x_spin.value()},{self.roi_y_spin.value()}"
        else:
            roi_text = "ROI: full"

        image_text = "Images: --"
        if isinstance(getattr(self, "results", None), dict) and self.results:
            orientation_count = int(self.results.get("orientation_count", 0) or 0)
            image_count = len(self.results.get("input_items", []) or [])
            if not image_count:
                image_count = orientation_count * 3 if orientation_count else 0
            if image_count:
                image_text = f"Images: {image_count} ({orientation_count} ori)"

        elapsed_text = ""
        if getattr(self, "_last_run_elapsed_seconds", None) is not None:
            elapsed_text = f"Last run: {self._last_run_elapsed_seconds:.1f}s"

        pairs = [
            (self.status_gpu_label, gpu_text),
            (self.status_input_label, "Input: " + self._short_path_for_status(input_path)),
            (self.status_output_label, "Output: " + self._short_path_for_status(output_path)),
            (self.status_roi_label, roi_text),
            (self.status_image_label, image_text),
            (self.status_elapsed_label, elapsed_text),
        ]
        for label, value in pairs:
            label.setText(value)
            label.setToolTip(value)

    def _collect_processing_config(self):
        return {
            "input_dir": self.input_dir_edit.text().strip(),
            "single_tiff_path": self.single_tiff_path_edit.text().strip(),
            "input_type": self.ext_combo.currentText(),
            "image_order_mode": self.image_order_combo.currentData() or "orientation_major",
            "image_order_text": self.image_order_combo.currentText(),
            "raw_widefield_path": self.raw_widefield_path_edit.text().strip(),
            "dark_image_path": self.dark_path_edit.text().strip(),
            "flat_image_path": self.flat_path_edit.text().strip(),

            "roi_enabled": self.enable_roi_check.isChecked(),
            "roi_x": self.roi_x_spin.value(),
            "roi_y": self.roi_y_spin.value(),
            "roi_w": self.roi_w_spin.value(),
            "roi_h": self.roi_h_spin.value(),

            "enable_preprocess": self.enable_preprocess_check.isChecked(),
            "phase_model": self.phase_model_combo.currentText(),
            "preprocess_mode": self.preprocess_mode_combo.currentText(),
            "bg_percentile": self.bg_percentile_spin.value(),
            "phase_norm_mode": self.phase_norm_mode_combo.currentText(),
            "phase_norm_stat": self.phase_norm_stat_combo.currentText(),
            "os_sim_recon_mode": self.os_sim_recon_mode_combo.currentText(),
            "fourier_filter_mode": self.fourier_filter_mode_combo.currentText(),
            "fourier_cutoff_input": self.fourier_cutoff_spin.value(),
            "fourier_order": self.fourier_order_spin.value(),
            "fourier_amount": self.fourier_amount_spin.value(),

            "fusion_method": self.fusion_method_combo.currentText(),
            "wavelet_name": self.wavelet_name_combo.currentText(),
            "wavelet_level": self.wavelet_level_spin.value(),
            "wavelet_detail_mode": self.wavelet_detail_mode_combo.currentText(),
            "enable_orientation_match": self.enable_orientation_match_check.isChecked(),
            "orientation_match_method": self.orientation_match_method_combo.currentText(),
            "final_fusion_mode": self.final_fusion_mode_combo.currentText(),

            "gaussian_sigma": self.gaussian_sigma_spin.value(),
            "enable_smoothing": self.enable_smoothing_check.isChecked(),
            "enable_wiener_filter": self.enable_wiener_filter_check.isChecked(),
            "wiener_target": self.wiener_target_combo.currentText(),
            "wiener_kernel": self.wiener_kernel_spin.value(),
            "wiener_noise_variance": self.wiener_noise_spin.value(),
            "wiener_amount": self.wiener_amount_spin.value(),

            "enable_output_intensity_match": self.enable_output_intensity_match_check.isChecked(),
            "output_intensity_target": self.output_intensity_target_combo.currentText(),
            "output_intensity_reference": self.output_intensity_reference_combo.currentText(),
            "output_intensity_method": self.output_intensity_method_combo.currentText(),
            "output_intensity_strength": self.output_intensity_strength_spin.value(),
            "output_intensity_max_scale": self.output_intensity_max_scale_spin.value(),

            "enable_sim_bg_correction": self.enable_sim_bg_correction_check.isChecked(),
            "sim_bg_mode": self.sim_bg_mode_combo.currentText(),
            "sim_bg_sigma": self.sim_bg_sigma_spin.value(),

            "enable_final_structure_enhance": self.enable_final_structure_enhance_check.isChecked(),
            "final_structure_target": self.final_structure_target_combo.currentText(),
            "final_structure_mode": self.final_structure_mode_combo.currentText(),
            "final_structure_sigma": self.final_structure_sigma_spin.value(),
            "final_structure_amount": self.final_structure_amount_spin.value(),
            "final_structure_keep_original": self.final_structure_keep_spin.value(),

            "enable_hilo_mode": self.enable_hilo_mode_check.isChecked(),
            "hilo_strategy": self.hilo_strategy_combo.currentText(),
            "hilo_auto_sigma": self.hilo_auto_sigma_check.isChecked(),
            "hilo_widefield_source": self.hilo_widefield_source_combo.currentText(),
            "hilo_low_source": self.hilo_low_source_combo.currentText(),
            "hilo_sigma": self.hilo_sigma_spin.value(),
            "hilo_mid_sigma": self.hilo_mid_sigma_spin.value(),
            "hilo_high_sigma": self.hilo_high_sigma_spin.value(),
            "hilo_lo_weight": self.hilo_lo_weight_spin.value(),
            "hilo_mid_weight": self.hilo_mid_weight_spin.value(),
            "hilo_hi_weight": self.hilo_hi_weight_spin.value(),
            "hilo_clip_output": self.hilo_clip_output_check.isChecked(),

            "enable_hilo_denoise": self.enable_hilo_denoise_check.isChecked(),
            "hilo_denoise_method": self.hilo_denoise_method_combo.currentText(),
            "hilo_nlm_h": self.hilo_nlm_h_spin.value(),
            "hilo_nlm_template_window": self.hilo_nlm_template_spin.value(),
            "hilo_nlm_search_window": self.hilo_nlm_search_spin.value(),
            "hilo_bm3d_sigma": self.hilo_bm3d_sigma_spin.value(),
            "hilo_denoise_amount": self.hilo_denoise_amount_spin.value(),
        }

    def _on_processing_log(self, message):
        self.log(message)

    def _on_processing_progress(self, message, value):
        progress = getattr(self, "_processing_progress", None)
        if value is None or int(value) < 0:
            self._set_busy_progress(progress, message)
        else:
            self._set_busy_progress(progress, message, int(value))
        self._set_status_text(str(message))

    def _finish_processing_ui(self):
        if getattr(self, "_processing_cursor_set", False):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            self._processing_cursor_set = False

        self._set_processing_controls_enabled(True)
        self._close_busy_progress(getattr(self, "_processing_progress", None))
        self._processing_progress = None
        self.run_btn.setText("Run")
        self._update_optional_control_states()
        self._update_status_bar()

    def _on_processing_finished(self, results, image_items, elapsed_seconds):
        try:
            self.results = results if isinstance(results, dict) else {}
            self.image_paths = list(image_items or [])
            self._last_run_elapsed_seconds = float(elapsed_seconds)

            self._set_busy_progress(getattr(self, "_processing_progress", None), "Rendering preview...", 96)
            if hasattr(self, "analysis_tabs"):
                self.analysis_tabs.setCurrentIndex(0)
            self.update_plot()
            self.update_profile_window_data()
            self._set_busy_progress(getattr(self, "_processing_progress", None), "Processing finished.", 100)
            self.log(f"Processing finished in {elapsed_seconds:.2f} s.")
            self._set_status_text("Finished")
        finally:
            self._finish_processing_ui()
            self._cleanup_processing_thread()

    def _on_processing_error(self, message, details):
        try:
            self.log(f"[ERROR] {message}")
            self.log(details)
            self._set_status_text("Error")
            QMessageBox.critical(self, "Error", str(message))
        finally:
            self._finish_processing_ui()
            self._cleanup_processing_thread()

    def _cleanup_processing_thread(self):
        thread = getattr(self, "_processing_thread", None)
        worker = getattr(self, "_processing_worker", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait(3000)
            except Exception:
                pass
        if worker is not None:
            try:
                worker.deleteLater()
            except Exception:
                pass
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._processing_thread = None
        self._processing_worker = None

    def _set_reconstruction_controls_enabled(self, enabled):
        widgets = [
            self.enable_preprocess_check,
            self.preprocess_mode_combo,
            self.bg_percentile_spin,
            self.phase_norm_mode_combo,
            self.phase_norm_stat_combo,
            self.os_sim_recon_mode_combo,
            self.fourier_filter_mode_combo,
            self.fourier_cutoff_spin,
            self.fourier_order_spin,
            self.fourier_amount_spin,
            self.fusion_method_combo,
            self.wavelet_name_combo,
            self.wavelet_level_spin,
            self.wavelet_detail_mode_combo,
            self.enable_orientation_match_check,
            self.orientation_match_method_combo,
            self.final_fusion_mode_combo,
            self.gaussian_sigma_spin,
            self.enable_smoothing_check,
            self.enable_wiener_filter_check,
            self.wiener_target_combo,
            self.wiener_kernel_spin,
            self.wiener_noise_spin,
            self.wiener_amount_spin,
            self.enable_output_intensity_match_check,
            self.output_intensity_target_combo,
            self.output_intensity_reference_combo,
            self.output_intensity_method_combo,
            self.output_intensity_strength_spin,
            self.output_intensity_max_scale_spin,

            self.enable_sim_bg_correction_check,
            self.sim_bg_mode_combo,
            self.sim_bg_sigma_spin,
            self.sim_bg_preset_btn_15,
            self.sim_bg_preset_btn_25,
            self.sim_bg_preset_btn_40,
            self.sim_bg_preset_btn_60,
            self.enable_final_structure_enhance_check,
            self.final_structure_target_combo,
            self.final_structure_mode_combo,
            self.final_structure_sigma_spin,
            self.final_structure_amount_spin,
            self.final_structure_keep_spin,
            self.enable_hilo_mode_check,
            self.hilo_strategy_combo,
            self.hilo_auto_sigma_check,
            self.hilo_widefield_source_combo,
            self.hilo_low_source_combo,
            self.hilo_sigma_spin,
            self.hilo_mid_sigma_spin,
            self.hilo_high_sigma_spin,
            self.hilo_lo_weight_spin,
            self.hilo_mid_weight_spin,
            self.hilo_hi_weight_spin,
            self.hilo_clip_output_check,
            self.enable_hilo_denoise_check,
            self.hilo_denoise_method_combo,
            self.hilo_nlm_h_spin,
            self.hilo_nlm_template_spin,
            self.hilo_nlm_search_spin,
            self.hilo_bm3d_sigma_spin,
            self.hilo_denoise_amount_spin,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def _update_reconstruction_mode_ui(self):
        mode = self.recon_mode_combo.currentText()
        if mode == "Paper mode":
            self.recon_mode_hint_label.setText(
                "Strict paper-style pipeline: disable preprocess, phase normalization, orientation matching, "
                "single-SIM background correction, final structure enhancement, and Gaussian smoothing. "
                "Keep wavelet detail fusion = maxabs."
            )
            self._set_reconstruction_controls_enabled(False)
        elif mode == "Paper mode (soft fusion)":
            self.recon_mode_hint_label.setText(
                "Paper-style pipeline with softer wavelet detail fusion: disable preprocess, phase normalization, "
                "orientation matching, single-SIM background correction, final structure enhancement, and Gaussian smoothing, "
                "while using detail fusion = mean to reduce emboss-like appearance."
            )
            self._set_reconstruction_controls_enabled(False)
        elif mode == "Enhanced mode":
            self.recon_mode_hint_label.setText(
                "Optimized optical-sectioning pipeline: preprocess/QC, selectable spatial/Fourier OS-SIM, "
                "per-orientation HiLo Lo/Mid/High fusion before postprocess, automatic cutoff/sigma recommendation, "
                "and optional HiLo-only NLM/BM3D-style denoising plus final smoothing/Wiener denoising."
            )
            self._set_reconstruction_controls_enabled(False)
        else:
            self.recon_mode_hint_label.setText(
                "Custom pipeline: parameters below can be edited manually."
            )
            self._set_reconstruction_controls_enabled(True)

        self._update_optional_control_states()

    def _mark_recon_mode_custom(self):
        if getattr(self, "_applying_recon_mode", False):
            return
        if self.recon_mode_combo.currentText() != "Custom":
            self.recon_mode_combo.blockSignals(True)
            self.recon_mode_combo.setCurrentText("Custom")
            self.recon_mode_combo.blockSignals(False)
        self._update_reconstruction_mode_ui()

    def _connect_reconstruction_param_signals(self):
        widgets = [
            self.enable_preprocess_check,
            self.phase_model_combo,
            self.preprocess_mode_combo,
            self.bg_percentile_spin,
            self.phase_norm_mode_combo,
            self.phase_norm_stat_combo,
            self.os_sim_recon_mode_combo,
            self.fourier_filter_mode_combo,
            self.fourier_cutoff_spin,
            self.fourier_order_spin,
            self.fourier_amount_spin,
            self.fusion_method_combo,
            self.wavelet_name_combo,
            self.wavelet_level_spin,
            self.wavelet_detail_mode_combo,
            self.enable_orientation_match_check,
            self.orientation_match_method_combo,
            self.final_fusion_mode_combo,
            self.gaussian_sigma_spin,
            self.enable_smoothing_check,
            self.enable_wiener_filter_check,
            self.wiener_target_combo,
            self.wiener_kernel_spin,
            self.wiener_noise_spin,
            self.wiener_amount_spin,
            self.enable_output_intensity_match_check,
            self.output_intensity_target_combo,
            self.output_intensity_reference_combo,
            self.output_intensity_method_combo,
            self.output_intensity_strength_spin,
            self.output_intensity_max_scale_spin,

            self.enable_sim_bg_correction_check,
            self.sim_bg_mode_combo,
            self.sim_bg_sigma_spin,
            self.enable_final_structure_enhance_check,
            self.final_structure_target_combo,
            self.final_structure_mode_combo,
            self.final_structure_sigma_spin,
            self.final_structure_amount_spin,
            self.final_structure_keep_spin,
            self.enable_hilo_mode_check,
            self.hilo_strategy_combo,
            self.hilo_auto_sigma_check,
            self.hilo_widefield_source_combo,
            self.hilo_low_source_combo,
            self.hilo_sigma_spin,
            self.hilo_mid_sigma_spin,
            self.hilo_high_sigma_spin,
            self.hilo_lo_weight_spin,
            self.hilo_mid_weight_spin,
            self.hilo_hi_weight_spin,
            self.hilo_clip_output_check,
            self.enable_hilo_denoise_check,
            self.hilo_denoise_method_combo,
            self.hilo_nlm_h_spin,
            self.hilo_nlm_template_spin,
            self.hilo_nlm_search_spin,
            self.hilo_bm3d_sigma_spin,
            self.hilo_denoise_amount_spin,
        ]
        for widget in widgets:
            signal = None
            if hasattr(widget, "toggled"):
                signal = widget.toggled
            elif hasattr(widget, "valueChanged"):
                signal = widget.valueChanged
            elif hasattr(widget, "currentTextChanged"):
                signal = widget.currentTextChanged
            if signal is not None:
                signal.connect(self._mark_recon_mode_custom)

    def apply_reconstruction_mode(self, mode):
        presets = {
            "Paper mode": {
                "enable_preprocess": False,
                "phase_model": "0/120/240",
                "preprocess_mode": "shared_percentile",
                "bg_percentile": 5.0,
                "phase_norm_mode": "per_orientation",
                "phase_norm_stat": "mean",
                "os_sim_recon_mode": "Spatial demodulation (standard OS-SIM)",
                "fourier_filter_mode": "Gaussian low-pass",
                "fourier_cutoff": 0.0,
                "fourier_order": 2,
                "fourier_amount": 1.0,
                "fusion_method": "SWT",
                "wavelet_name": "db2",
                "wavelet_level": 2,
                "wavelet_detail_mode": "maxabs",
                "enable_orientation_match": False,
                "orientation_match_method": "median",
                "final_fusion_mode": "RMS",
                "gaussian_sigma": 1.0,
                "enable_smoothing": False,
                "enable_wiener_filter": False,
                "wiener_target": "Final SIM only",
                "wiener_kernel": 5,
                "wiener_noise_variance": 0.0,
                "wiener_amount": 1.0,
                "enable_sim_bg_correction": False,
                "sim_bg_mode": "subtract",
                "sim_bg_sigma": 25.0,
                "enable_final_structure_enhance": False,
                "final_structure_target": "Final SIM only",
                "final_structure_mode": "highpass_mix",
                "final_structure_sigma": 35.0,
                "final_structure_amount": 1.0,
                "final_structure_keep_original": 0.15,
                "enable_hilo_mode": True,
                "hilo_strategy": "Per-orientation OS-SIM/pWF then RMS fuse (recommended)",
                "hilo_auto_sigma": False,
                "hilo_widefield_source": "Pseudo-widefield fused",
                "hilo_low_source": "Final SIM fused",
                "hilo_sigma": 2.0,
                "hilo_mid_sigma": 3.0,
                "hilo_high_sigma": 1.0,
                "hilo_lo_weight": 1.0,
                "hilo_mid_weight": 0.50,
                "hilo_hi_weight": 0.8,
                "hilo_clip_output": True,
                "enable_hilo_denoise": False,
                "hilo_denoise_method": "NLM",
                "hilo_nlm_h": 8.0,
                "hilo_nlm_template_window": 7,
                "hilo_nlm_search_window": 21,
                "hilo_bm3d_sigma": 0.0,
                "hilo_denoise_amount": 0.60,
            },
            "Paper mode (soft fusion)": {
                "enable_preprocess": False,
                "phase_model": "0/120/240",
                "preprocess_mode": "shared_percentile",
                "bg_percentile": 5.0,
                "phase_norm_mode": "per_orientation",
                "phase_norm_stat": "mean",
                "os_sim_recon_mode": "Spatial demodulation (standard OS-SIM)",
                "fourier_filter_mode": "Gaussian low-pass",
                "fourier_cutoff": 0.0,
                "fourier_order": 2,
                "fourier_amount": 1.0,
                "fusion_method": "SWT",
                "wavelet_name": "db2",
                "wavelet_level": 2,
                "wavelet_detail_mode": "mean",
                "enable_orientation_match": False,
                "orientation_match_method": "median",
                "final_fusion_mode": "RMS",
                "gaussian_sigma": 1.0,
                "enable_smoothing": False,
                "enable_wiener_filter": False,
                "wiener_target": "Final SIM only",
                "wiener_kernel": 5,
                "wiener_noise_variance": 0.0,
                "wiener_amount": 1.0,
                "enable_sim_bg_correction": False,
                "sim_bg_mode": "subtract",
                "sim_bg_sigma": 25.0,
                "enable_final_structure_enhance": False,
                "final_structure_target": "Final SIM only",
                "final_structure_mode": "highpass_mix",
                "final_structure_sigma": 35.0,
                "final_structure_amount": 1.0,
                "final_structure_keep_original": 0.15,
                "enable_hilo_mode": True,
                "hilo_strategy": "Per-orientation OS-SIM/pWF then RMS fuse (recommended)",
                "hilo_auto_sigma": False,
                "hilo_widefield_source": "Pseudo-widefield fused",
                "hilo_low_source": "Final SIM fused",
                "hilo_sigma": 2.0,
                "hilo_mid_sigma": 3.0,
                "hilo_high_sigma": 1.0,
                "hilo_lo_weight": 1.0,
                "hilo_mid_weight": 0.40,
                "hilo_hi_weight": 0.7,
                "hilo_clip_output": True,
                "enable_hilo_denoise": False,
                "hilo_denoise_method": "NLM",
                "hilo_nlm_h": 8.0,
                "hilo_nlm_template_window": 7,
                "hilo_nlm_search_window": 21,
                "hilo_bm3d_sigma": 0.0,
                "hilo_denoise_amount": 0.60,
            },
            "Enhanced mode": {
                "enable_preprocess": True,
                "phase_model": "0/120/240",
                "preprocess_mode": "shared_percentile",
                "bg_percentile": 5.0,
                "phase_norm_mode": "per_orientation",
                "phase_norm_stat": "mean",
                "os_sim_recon_mode": "Spatial demodulation (standard OS-SIM)",
                "fourier_filter_mode": "Gaussian low-pass",
                "fourier_cutoff": 0.0,
                "fourier_order": 2,
                "fourier_amount": 1.0,
                "fusion_method": "SWT",
                "wavelet_name": "db2",
                "wavelet_level": 2,
                "wavelet_detail_mode": "maxabs",
                "enable_orientation_match": True,
                "orientation_match_method": "median",
                "final_fusion_mode": "RMS",
                "gaussian_sigma": 1.0,
                "enable_smoothing": True,
                "enable_wiener_filter": False,
                "wiener_target": "Final SIM only",
                "wiener_kernel": 5,
                "wiener_noise_variance": 0.0,
                "wiener_amount": 1.0,
                "enable_sim_bg_correction": True,
                "sim_bg_mode": "subtract",
                "sim_bg_sigma": 25.0,
                "enable_final_structure_enhance": True,
                "final_structure_target": "Final SIM only",
                "final_structure_mode": "highpass_mix",
                "final_structure_sigma": 35.0,
                "final_structure_amount": 1.0,
                "final_structure_keep_original": 0.15,
                "enable_hilo_mode": True,
                "hilo_strategy": "Per-orientation OS-SIM/pWF then RMS fuse (recommended)",
                "hilo_auto_sigma": True,
                "hilo_widefield_source": "Pseudo-widefield fused",
                "hilo_low_source": "Final SIM fused",
                "hilo_sigma": 2.0,
                "hilo_mid_sigma": 3.0,
                "hilo_high_sigma": 1.0,
                "hilo_lo_weight": 1.0,
                "hilo_mid_weight": 0.60,
                "hilo_hi_weight": 0.9,
                "hilo_clip_output": True,
                "enable_hilo_denoise": False,
                "hilo_denoise_method": "NLM",
                "hilo_nlm_h": 8.0,
                "hilo_nlm_template_window": 7,
                "hilo_nlm_search_window": 21,
                "hilo_bm3d_sigma": 0.0,
                "hilo_denoise_amount": 0.60,
            },
        }

        if mode == "Custom":
            self._update_reconstruction_mode_ui()
            return

        params = presets.get(mode)
        if params is None:
            return

        self._applying_recon_mode = True
        try:
            self.enable_preprocess_check.setChecked(params["enable_preprocess"])
            self.phase_model_combo.setCurrentText(params.get("phase_model", "0/120/240"))
            self.preprocess_mode_combo.setCurrentText(params.get("preprocess_mode", "shared_percentile"))
            self.bg_percentile_spin.setValue(params["bg_percentile"])
            self.phase_norm_mode_combo.setCurrentText(params["phase_norm_mode"])
            self.phase_norm_stat_combo.setCurrentText(params["phase_norm_stat"])
            self.os_sim_recon_mode_combo.setCurrentText(params.get("os_sim_recon_mode", "Spatial demodulation (standard OS-SIM)"))
            self.fourier_filter_mode_combo.setCurrentText(params.get("fourier_filter_mode", "Gaussian low-pass"))
            self.fourier_cutoff_spin.setValue(float(params.get("fourier_cutoff", 0.0)))
            self.fourier_order_spin.setValue(int(params.get("fourier_order", 2)))
            self.fourier_amount_spin.setValue(float(params.get("fourier_amount", 1.0)))
            self.fusion_method_combo.setCurrentText(params["fusion_method"])
            self.wavelet_name_combo.setCurrentText(params["wavelet_name"])
            self.wavelet_level_spin.setValue(params["wavelet_level"])
            self.wavelet_detail_mode_combo.setCurrentText(params["wavelet_detail_mode"])
            self.enable_orientation_match_check.setChecked(params["enable_orientation_match"])
            self.orientation_match_method_combo.setCurrentText(params["orientation_match_method"])
            self.final_fusion_mode_combo.setCurrentText(params["final_fusion_mode"])
            self.gaussian_sigma_spin.setValue(params["gaussian_sigma"])
            self.enable_smoothing_check.setChecked(params["enable_smoothing"])
            self.enable_wiener_filter_check.setChecked(params.get("enable_wiener_filter", False))
            self.wiener_target_combo.setCurrentText(params.get("wiener_target", "Final SIM only"))
            self.wiener_kernel_spin.setValue(int(params.get("wiener_kernel", 5)))
            self.wiener_noise_spin.setValue(float(params.get("wiener_noise_variance", 0.0)))
            self.wiener_amount_spin.setValue(float(params.get("wiener_amount", 1.0)))
            self.enable_sim_bg_correction_check.setChecked(params["enable_sim_bg_correction"])
            self.sim_bg_mode_combo.setCurrentText(params["sim_bg_mode"])
            self.sim_bg_sigma_spin.setValue(params["sim_bg_sigma"])
            self.enable_final_structure_enhance_check.setChecked(params["enable_final_structure_enhance"])
            self.final_structure_target_combo.setCurrentText(params["final_structure_target"])
            self.final_structure_mode_combo.setCurrentText(params["final_structure_mode"])
            self.final_structure_sigma_spin.setValue(params["final_structure_sigma"])
            self.final_structure_amount_spin.setValue(params["final_structure_amount"])
            self.final_structure_keep_spin.setValue(params["final_structure_keep_original"])
            self.enable_hilo_mode_check.setChecked(params.get("enable_hilo_mode", True))
            self.hilo_strategy_combo.setCurrentText(params.get("hilo_strategy", "Per-orientation OS-SIM/pWF then RMS fuse (recommended)"))
            self.hilo_auto_sigma_check.setChecked(bool(params.get("hilo_auto_sigma", True)))
            self.hilo_widefield_source_combo.setCurrentText(params.get("hilo_widefield_source", "Pseudo-widefield fused"))
            self.hilo_low_source_combo.setCurrentText(params.get("hilo_low_source", "Final SIM fused"))
            self.hilo_sigma_spin.setValue(float(params.get("hilo_sigma", 2.0)))
            self.hilo_mid_sigma_spin.setValue(float(params.get("hilo_mid_sigma", 3.0)))
            self.hilo_high_sigma_spin.setValue(float(params.get("hilo_high_sigma", 1.0)))
            self.hilo_lo_weight_spin.setValue(float(params.get("hilo_lo_weight", 1.0)))
            self.hilo_mid_weight_spin.setValue(float(params.get("hilo_mid_weight", 0.5)))
            self.hilo_hi_weight_spin.setValue(float(params.get("hilo_hi_weight", 1.0)))
            self.hilo_clip_output_check.setChecked(bool(params.get("hilo_clip_output", True)))
            self.enable_hilo_denoise_check.setChecked(bool(params.get("enable_hilo_denoise", False)))
            self.hilo_denoise_method_combo.setCurrentText(params.get("hilo_denoise_method", "NLM"))
            self.hilo_nlm_h_spin.setValue(float(params.get("hilo_nlm_h", 8.0)))
            self.hilo_nlm_template_spin.setValue(int(params.get("hilo_nlm_template_window", 7)))
            self.hilo_nlm_search_spin.setValue(int(params.get("hilo_nlm_search_window", 21)))
            self.hilo_bm3d_sigma_spin.setValue(float(params.get("hilo_bm3d_sigma", 0.0)))
            self.hilo_denoise_amount_spin.setValue(float(params.get("hilo_denoise_amount", 0.60)))
        finally:
            self._applying_recon_mode = False

        self._update_reconstruction_mode_ui()

    def _on_recon_mode_changed(self, mode):
        self.apply_reconstruction_mode(mode)

    def _on_display_control_changed(self):
        self._update_display_control_state()
        self._update_manual_labels()
        self.update_plot()
        self.update_profile_window_data()

    def _on_raw_widefield_path_changed(self):
        try:
            self._load_optional_raw_widefield()
        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

    def _update_display_control_state(self):
        auto_enabled = self.auto_contrast_check.isChecked()
        self.auto_low_spin.setEnabled(auto_enabled)
        self.auto_high_spin.setEnabled(auto_enabled)
        self.manual_min_slider.setEnabled(not auto_enabled)
        self.manual_max_slider.setEnabled(not auto_enabled)

    def _update_roi_control_state(self):
        roi_enabled = self.enable_roi_check.isChecked()
        for widget in (
            self.roi_x_spin,
            self.roi_y_spin,
            self.roi_w_spin,
            self.roi_h_spin,
        ):
            widget.setEnabled(roi_enabled)

        if hasattr(self, "save_ordered_roi_tiff_btn"):
            if roi_enabled:
                self.save_ordered_roi_tiff_btn.setText("Save ordered ROI stack TIFF...")
                self.save_ordered_roi_tiff_btn.setToolTip(
                    "Save the current reordered and ROI-cropped 6/9 phase images as one multi-page TIFF."
                )
            else:
                self.save_ordered_roi_tiff_btn.setText("Save ordered full stack TIFF...")
                self.save_ordered_roi_tiff_btn.setToolTip(
                    "ROI is disabled: save the current reordered full-frame 6/9 phase images as one multi-page TIFF."
                )

    def _get_reference_display_range(self):
        imgs = []

        if "pwf_images" in self.results:
            imgs.extend(self.results.get("pwf_images", []))
        else:
            for key in ["pwf_vertical", "pwf_horizontal", "pwf_fused"]:
                if key in self.results and self.results[key] is not None:
                    imgs.append(self.results[key])

        if "sim_images" in self.results:
            imgs.extend(self.results.get("sim_images", []))
        else:
            for key in ["sim_vertical", "sim_horizontal"]:
                if key in self.results and self.results[key] is not None:
                    imgs.append(self.results[key])

        if "sim_fused_mean" in self.results:
            imgs.append(self.results["sim_fused_mean"])
        if "sim_wavelet_fused" in self.results:
            imgs.append(self.results["sim_wavelet_fused"])

        imgs = [np.asarray(img, dtype=np.float32) for img in imgs if img is not None and np.asarray(img).size > 0]

        if not imgs:
            return 0.0, 1.0

        data_min = min(float(np.min(img)) for img in imgs)
        data_max = max(float(np.max(img)) for img in imgs)

        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max):
            data_max = 1.0
        if data_max <= data_min:
            data_max = data_min + 1e-6
        return data_min, data_max

    def _slider_to_value(self, slider_value, data_min, data_max):
        ratio = float(slider_value) / 1000.0
        return data_min + ratio * (data_max - data_min)

    def _update_manual_labels(self):
        data_min, data_max = self._get_reference_display_range()
        manual_min = self._slider_to_value(self.manual_min_slider.value(), data_min, data_max)
        manual_max = self._slider_to_value(self.manual_max_slider.value(), data_min, data_max)

        if manual_max <= manual_min:
            manual_max = manual_min + 1e-6

        self.manual_min_label.setText(f"{manual_min:.3f}")
        self.manual_max_label.setText(f"{manual_max:.3f}")
        self.display_range_hint_label.setText(f"Display range: [{data_min:.3f}, {data_max:.3f}]")
        return manual_min, manual_max

    def _get_display_params(self, img):
        manual_min, manual_max = self._update_manual_labels()
        return prepare_display_image(
            img,
            auto_contrast=self.auto_contrast_check.isChecked(),
            low=self.auto_low_spin.value(),
            high=self.auto_high_spin.value(),
            manual_min=manual_min,
            manual_max=manual_max,
        )

    def _get_profile_window_display_config(self):
        manual_min, manual_max = self._update_manual_labels()
        return {
            "enabled": self.sync_profile_display_check.isChecked(),
            "colormap": self.colormap_combo.currentText(),
            "auto_contrast": self.auto_contrast_check.isChecked(),
            "auto_low": self.auto_low_spin.value(),
            "auto_high": self.auto_high_spin.value(),
            "manual_min": manual_min,
            "manual_max": manual_max,
        }

    # -------------------------
    # 工具函数
    # -------------------------
    def _begin_busy_progress(self, title, message, minimum=0, maximum=100):
        """Show a modal progress dialog and immediately flush the Qt event queue."""
        progress = QProgressDialog(str(message), "", int(minimum), int(maximum), self)
        progress.setWindowTitle(str(title))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setCancelButton(None)
        progress.setValue(int(minimum))
        progress.show()
        QApplication.processEvents()
        return progress

    def _set_busy_progress(self, progress, message, value=None):
        """Update progress text/value and keep the GUI responsive between heavy steps."""
        if progress is None:
            return
        try:
            if value is not None:
                value = int(value)
                if value > progress.maximum():
                    progress.setMaximum(value)
                progress.setValue(value)
            progress.setLabelText(str(message))
            progress.repaint()
            QApplication.processEvents()
        except Exception:
            pass

    def _close_busy_progress(self, progress, final_message=None):
        if progress is None:
            return
        try:
            if final_message:
                self._set_busy_progress(progress, final_message, progress.maximum())
            progress.close()
            QApplication.processEvents()
        except Exception:
            pass

    def _set_processing_controls_enabled(self, enabled):
        for attr in ("run_btn", "refresh_btn", "save_btn"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(bool(enabled))

    def log(self, text):
        self.log_text.append(text)
        try:
            QApplication.processEvents()
        except Exception:
            pass

    def _default_output_dir_for_path(self, input_path, is_file=False):
        """
        Default output folder:
        - folder input /data/images -> /data/HiLo_OS_SIM
        - TIFF input   /data/stack.tif -> /data/HiLo_OS_SIM
        """
        input_path = str(input_path or "").strip()
        if not input_path:
            base_dir = os.getcwd()
        else:
            expanded = os.path.abspath(os.path.expanduser(input_path))
            if is_file:
                base_dir = os.path.dirname(expanded)
            else:
                normalized = os.path.normpath(expanded)
                base_dir = os.path.dirname(normalized)
                if not base_dir:
                    base_dir = os.getcwd()

        return os.path.normpath(os.path.join(base_dir, "HiLo_OS_SIM"))

    def _current_input_path_for_default_output(self):
        single_tiff_path = self.single_tiff_path_edit.text().strip()
        if single_tiff_path:
            return single_tiff_path, True

        input_dir = self.input_dir_edit.text().strip()
        return input_dir, False

    def _get_effective_output_dir(self):
        input_path, is_file = self._current_input_path_for_default_output()
        return self._default_output_dir_for_path(input_path, is_file=is_file)

    def clear_single_tiff_path(self):
        self.single_tiff_path_edit.clear()

    def choose_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Input Folder")
        if path:
            self.input_dir_edit.setText(path)

    def choose_single_tiff_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Single 6/9-page TIFF Stack",
            "",
            "TIFF files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.single_tiff_path_edit.setText(file_path)

    def choose_raw_widefield_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Raw Widefield Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.raw_widefield_path_edit.setText(file_path)
            self._load_optional_raw_widefield()

    def choose_dark_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Dark Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.dark_path_edit.setText(file_path)

    def choose_flat_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Flat-field Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.flat_path_edit.setText(file_path)

    def _load_optional_correction_image(self, path, name, full_shape, roi_x, roi_y, roi_w, roi_h):
        """Load dark/flat image and return an effective-processing-size float32 image."""
        path = str(path).strip()
        if not path:
            return None
        if not os.path.exists(path):
            raise ValueError(f"{name} image does not exist: {path}")
        img = load_image(path)
        roi_shape = (int(roi_h), int(roi_w))
        if img.shape == tuple(full_shape):
            img = crop_roi(img, roi_x, roi_y, roi_w, roi_h)
        elif img.shape == roi_shape:
            img = img.astype(np.float32)
        else:
            raise ValueError(
                f"{name} image shape {img.shape} must match either full input shape {tuple(full_shape)} "
                f"or ROI shape {roi_shape}."
            )
        return np.asarray(img, dtype=np.float32)

    def save_ordered_roi_stack_tiff(self):
        """
        按当前 Image order 设置重排输入图像，
        再按当前 ROI 裁切，
        最后将 6/9 张裁切后的相位图按内部顺序保存为一个多页 TIFF。

        保存顺序固定为程序内部使用的 orientation-major:
            orientation1 phase1/phase2/phase3,
            orientation2 phase1/phase2/phase3,
            orientation3 phase1/phase2/phase3.
        """
        try:
            if tifffile is None:
                raise RuntimeError(
                    "tifffile is not available. Please install it with: pip install tifffile"
                )

            input_items_original, input_source_type = self.collect_input_items()
            image_order_mode = self.image_order_combo.currentData() or "orientation_major"
            input_items = reorder_stripe_sequence(input_items_original, order_mode=image_order_mode)

            roi_enabled = self.enable_roi_check.isChecked()
            roi_x = self.roi_x_spin.value()
            roi_y = self.roi_y_spin.value()
            roi_w = self.roi_w_spin.value()
            roi_h = self.roi_h_spin.value()

            output_dir = self._get_effective_output_dir()
            os.makedirs(output_dir, exist_ok=True)

            default_name = "ordered_roi_phase_stack.tif" if roi_enabled else "ordered_full_phase_stack.tif"
            default_path = os.path.join(output_dir, default_name)

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save ordered ROI phase stack as multi-page TIFF",
                default_path,
                "TIFF files (*.tif *.tiff);;All Files (*)"
            )
            if not save_path:
                return

            if not save_path.lower().endswith((".tif", ".tiff")):
                save_path += ".tif"

            raw_stack_images = []
            for idx, item in enumerate(input_items, 1):
                img = self.load_input_item_image(item)
                raw_stack_images.append(np.asarray(img, dtype=np.float32))
                self.log(f"Stack page {idx}: {self.input_item_display_name(item)}")

            if not raw_stack_images:
                raise ValueError("No images to save.")

            raw_shapes = [img.shape for img in raw_stack_images]
            if len(set(raw_shapes)) != 1:
                raise ValueError(f"Input images do not have the same shape: {raw_shapes}")

            input_h, input_w = raw_stack_images[0].shape[:2]
            if roi_enabled:
                self.log(f"ROI enabled for ordered stack: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
                cropped_images = [crop_roi(img, roi_x, roi_y, roi_w, roi_h) for img in raw_stack_images]
            else:
                roi_x, roi_y, roi_w, roi_h = 0, 0, int(input_w), int(input_h)
                self.log("ROI disabled for ordered stack. Saving full-frame ordered input images.")
                cropped_images = [img.copy() for img in raw_stack_images]

            shapes = [img.shape for img in cropped_images]
            if len(set(shapes)) != 1:
                raise ValueError(f"Output stack images do not have the same shape: {shapes}")

            stack = np.stack(cropped_images, axis=0).astype(np.float32)

            description = json.dumps(
                {
                    "type": "ordered_roi_phase_stack",
                    "roi_enabled": bool(roi_enabled),
                    "image_order_mode": image_order_mode,
                    "image_order_label": self.image_order_combo.currentText(),
                    "internal_order": (
                        "orientation-major: ori1_p0, ori1_p90, ori1_p180, "
                        "ori2_p0, ori2_p90, ori2_p180, ori3_p0, ori3_p90, ori3_p180"
                    ),
                    "roi": {
                        "x": int(roi_x),
                        "y": int(roi_y),
                        "width": int(roi_w),
                        "height": int(roi_h),
                    },
                    "input_source_type": input_source_type,
                    "source_items_in_saved_order": [self.input_item_source_name(item) for item in input_items],
                },
                ensure_ascii=False,
                indent=2,
            )

            tifffile.imwrite(
                save_path,
                stack,
                photometric="minisblack",
                metadata={"axes": "QYX"},
                description=description,
            )

            self.log(f"Saved ordered ROI multi-page TIFF: {save_path}")
            self.log(f"Saved stack shape: {stack.shape}, dtype: {stack.dtype}")
            QMessageBox.information(
                self,
                "Saved",
                f"Ordered ROI multi-page TIFF saved to:\n{save_path}\n\n"
                f"Pages: {stack.shape[0]}\n"
                f"ROI enabled: {roi_enabled}\n"
                f"Saved image size: {stack.shape[2]} x {stack.shape[1]}"
            )

        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

    def _load_optional_raw_widefield(self):
        path = self.raw_widefield_path_edit.text().strip()
        if not path:
            if isinstance(self.results, dict) and "pwf_fused" in self.results:
                self.results["external_raw_widefield"] = None
                self.update_plot()
                self.update_profile_window_data()
            return None

        if not os.path.exists(path):
            raise ValueError(f"Raw widefield image does not exist: {path}")

        img = load_image(path)
        if self.enable_roi_check.isChecked():
            roi_x = self.roi_x_spin.value()
            roi_y = self.roi_y_spin.value()
            roi_w = self.roi_w_spin.value()
            roi_h = self.roi_h_spin.value()
            img = crop_roi(img, roi_x, roi_y, roi_w, roi_h)
        else:
            img = np.asarray(img, dtype=np.float32)

        if isinstance(self.results, dict) and "pwf_fused" in self.results:
            self.results["external_raw_widefield"] = img
            self.update_plot()
            self.update_profile_window_data()
        return img

    def _get_profile_window_geometry_dict(self):
        if self.profile_window is None:
            return None
        geo = self.profile_window.geometry()
        return {
            "x": int(geo.x()),
            "y": int(geo.y()),
            "width": int(geo.width()),
            "height": int(geo.height()),
            "visible": bool(self.profile_window.isVisible()),
        }

    def _apply_profile_window_geometry_dict(self, data):
        if not data:
            return
        if self.profile_window is None:
            self.profile_window = ProfileComparisonWindow(self)
        try:
            self.profile_window.setGeometry(
                int(data.get("x", self.profile_window.x())),
                int(data.get("y", self.profile_window.y())),
                int(data.get("width", self.profile_window.width())),
                int(data.get("height", self.profile_window.height())),
            )
            if data.get("visible", False) and hasattr(self, "analysis_tabs"):
                idx = self.analysis_tabs.indexOf(self.profile_window)
                if idx >= 0:
                    self.analysis_tabs.setCurrentIndex(idx)
        except Exception:
            pass

    def _collect_config(self):
        main_geo = self.geometry()
        return {
            "paths": {
                "input_dir": self.input_dir_edit.text().strip(),
                "single_tiff_path": self.single_tiff_path_edit.text().strip(),
                "raw_widefield_path": self.raw_widefield_path_edit.text().strip(),
                "dark_path": self.dark_path_edit.text().strip(),
                "flat_path": self.flat_path_edit.text().strip(),
                "input_type": self.ext_combo.currentText(),
                "image_order": self.image_order_combo.currentData(),
                "image_order_label": self.image_order_combo.currentText(),
            },
            "roi": {
                "enabled": self.enable_roi_check.isChecked(),
                "x": self.roi_x_spin.value(),
                "y": self.roi_y_spin.value(),
                "w": self.roi_w_spin.value(),
                "h": self.roi_h_spin.value(),
            },
            "reconstruction_mode": {
                "mode": self.recon_mode_combo.currentText(),
            },
            "preprocess": {
                "enabled": self.enable_preprocess_check.isChecked(),
                "phase_model": self.phase_model_combo.currentText(),
                "preprocess_mode": self.preprocess_mode_combo.currentText(),
                "bg_percentile": self.bg_percentile_spin.value(),
                "phase_norm_mode": self.phase_norm_mode_combo.currentText(),
                "phase_norm_stat": self.phase_norm_stat_combo.currentText(),
                "os_sim_recon_mode": self.os_sim_recon_mode_combo.currentText(),
                "fourier_filter_mode": self.fourier_filter_mode_combo.currentText(),
                "fourier_cutoff": self.fourier_cutoff_spin.value(),
                "fourier_order": self.fourier_order_spin.value(),
                "fourier_amount": self.fourier_amount_spin.value(),
            },
            "display": {
                "view_mode": self.view_mode_combo.currentText(),
                "colormap": self.colormap_combo.currentText(),
                "auto_contrast": self.auto_contrast_check.isChecked(),
                "auto_low": self.auto_low_spin.value(),
                "auto_high": self.auto_high_spin.value(),
                "manual_min_slider": self.manual_min_slider.value(),
                "manual_max_slider": self.manual_max_slider.value(),
                "sync_profile_display": self.sync_profile_display_check.isChecked(),
            },
            "wavelet": {
                "fusion_method": self.fusion_method_combo.currentText(),
                "wavelet_name": self.wavelet_name_combo.currentText(),
                "wavelet_level": self.wavelet_level_spin.value(),
                "wavelet_detail_mode": self.wavelet_detail_mode_combo.currentText(),
                "enable_orientation_match": self.enable_orientation_match_check.isChecked(),
                "orientation_match_method": self.orientation_match_method_combo.currentText(),
                "final_fusion_mode": self.final_fusion_mode_combo.currentText(),
            },
            "postprocess": {
                "gaussian_sigma": self.gaussian_sigma_spin.value(),
                "enable_smoothing": self.enable_smoothing_check.isChecked(),
                "enable_wiener_filter": self.enable_wiener_filter_check.isChecked(),
                "wiener_target": self.wiener_target_combo.currentText(),
                "wiener_kernel": self.wiener_kernel_spin.value(),
                "wiener_noise_variance": self.wiener_noise_spin.value(),
                "wiener_amount": self.wiener_amount_spin.value(),
                "enable_output_intensity_match": self.enable_output_intensity_match_check.isChecked(),
                "output_intensity_target": self.output_intensity_target_combo.currentText(),
                "output_intensity_reference": self.output_intensity_reference_combo.currentText(),
                "output_intensity_method": self.output_intensity_method_combo.currentText(),
                "output_intensity_strength": self.output_intensity_strength_spin.value(),
                "output_intensity_max_scale": self.output_intensity_max_scale_spin.value(),
                "enable_sim_bg_correction": self.enable_sim_bg_correction_check.isChecked(),
                "sim_bg_mode": self.sim_bg_mode_combo.currentText(),
                "sim_bg_sigma": self.sim_bg_sigma_spin.value(),
                "enable_final_structure_enhance": self.enable_final_structure_enhance_check.isChecked(),
                "final_structure_target": self.final_structure_target_combo.currentText(),
                "final_structure_mode": self.final_structure_mode_combo.currentText(),
                "final_structure_sigma": self.final_structure_sigma_spin.value(),
                "final_structure_amount": self.final_structure_amount_spin.value(),
                "final_structure_keep_original": self.final_structure_keep_spin.value(),
                "enable_hilo_mode": self.enable_hilo_mode_check.isChecked(),
                "hilo_strategy": self.hilo_strategy_combo.currentText(),
                "hilo_auto_sigma": self.hilo_auto_sigma_check.isChecked(),
                "hilo_widefield_source": self.hilo_widefield_source_combo.currentText(),
                "hilo_low_source": self.hilo_low_source_combo.currentText(),
                "hilo_sigma": self.hilo_sigma_spin.value(),
                "hilo_mid_sigma": self.hilo_mid_sigma_spin.value(),
                "hilo_high_sigma": self.hilo_high_sigma_spin.value(),
                "hilo_lo_weight": self.hilo_lo_weight_spin.value(),
                "hilo_mid_weight": self.hilo_mid_weight_spin.value(),
                "hilo_hi_weight": self.hilo_hi_weight_spin.value(),
                "hilo_clip_output": self.hilo_clip_output_check.isChecked(),
                "enable_hilo_denoise": self.enable_hilo_denoise_check.isChecked(),
                "hilo_denoise_method": self.hilo_denoise_method_combo.currentText(),
                "hilo_nlm_h": self.hilo_nlm_h_spin.value(),
                "hilo_nlm_template_window": self.hilo_nlm_template_spin.value(),
                "hilo_nlm_search_window": self.hilo_nlm_search_spin.value(),
                "hilo_bm3d_sigma": self.hilo_bm3d_sigma_spin.value(),
                "hilo_denoise_amount": self.hilo_denoise_amount_spin.value(),
            },
            "windows": {
                "main": {
                    "x": int(main_geo.x()),
                    "y": int(main_geo.y()),
                    "width": int(main_geo.width()),
                    "height": int(main_geo.height()),
                    "splitter_sizes": self.main_splitter.sizes() if hasattr(self, "main_splitter") else None,
                },
                "profile": self._get_profile_window_geometry_dict(),
            },
            "profile_window_state": (
                self.profile_window.get_ui_state()
                if self.profile_window is not None else None
            ),
        }

    def _get_config_dialog_dir(self):
        """
        Return a sensible folder for manual parameter preset save/load dialogs.
        The default autosave path remains self.config_path and is not changed by
        manual presets.
        """
        candidates = [
            getattr(self, "_last_param_config_dir", ""),
            os.path.dirname(getattr(self, "_last_param_config_path", "") or ""),
            self._get_effective_output_dir(),
            os.path.dirname(getattr(self, "config_path", "") or ""),
            os.getcwd(),
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return os.getcwd()

    def _ensure_json_config_path(self, path):
        path = str(path).strip()
        if not path:
            return path
        if not path.lower().endswith(".json"):
            path += ".json"
        return os.path.abspath(path)

    def _suggest_config_preset_filename(self):
        """
        Build a readable default filename for manual parameter presets.
        Users can still edit the filename freely in the Save dialog.
        """
        source = ""
        try:
            source = self.single_tiff_path_edit.text().strip()
            if not source:
                source = self.input_dir_edit.text().strip()
            source = os.path.splitext(os.path.basename(source.rstrip("/\\")))[0]
        except Exception:
            source = ""

        if not source:
            source = "sim_dataset"

        try:
            mode = self.recon_mode_combo.currentText()
        except Exception:
            mode = "custom"

        try:
            fusion = self.final_fusion_mode_combo.currentText()
        except Exception:
            fusion = "fusion"

        parts = [
            self._safe_filename_part(source),
            self._safe_filename_part(mode),
            self._safe_filename_part(fusion),
            "params",
        ]
        filename = "_".join([p for p in parts if p]).strip("_")
        if not filename:
            filename = "sim_params"
        return filename + ".json"

    def _add_config_profile_metadata(self, config, config_path):
        """
        Add metadata to custom/default JSON files. Older config files without
        this section remain loadable.
        """
        config_path = self._ensure_json_config_path(config_path)
        preset_name = os.path.splitext(os.path.basename(config_path))[0]
        config["config_profile"] = {
            "preset_name": preset_name,
            "preset_path": config_path,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "version": "custom_param_presets_v1",
            "source_script": os.path.abspath(__file__) if "__file__" in globals() else "",
        }
        return config

    def save_params_and_layout(self, show_message=True, target_path=None):
        """
        Save parameters/layout.

        - show_message=False keeps the original autosave behavior and writes to
          self.config_path without opening a dialog.
        - show_message=True opens a Save dialog so each dataset can have a
          named preset stored in any folder.
        """
        try:
            if target_path is not None:
                config_path = self._ensure_json_config_path(target_path)
            elif show_message:
                default_path = os.path.join(
                    self._get_config_dialog_dir(),
                    self._suggest_config_preset_filename(),
                )
                selected_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Params/Layout Preset",
                    default_path,
                    "SIM parameter presets (*.json);;JSON files (*.json);;All files (*)",
                )
                if not selected_path:
                    self.log("Save Params/Layout canceled.")
                    return
                config_path = self._ensure_json_config_path(selected_path)
            else:
                config_path = self._ensure_json_config_path(getattr(self, "config_path", "sim_gui_config.json"))

            config = self._collect_config()
            config = self._add_config_profile_metadata(config, config_path)

            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            self._last_param_config_path = config_path
            self._last_param_config_dir = os.path.dirname(config_path) or os.getcwd()

            profile = config.get("config_profile", {})
            preset_name = profile.get("preset_name", os.path.splitext(os.path.basename(config_path))[0])
            self.log(f"Saved parameters/layout preset '{preset_name}' to: {config_path}")

            if show_message:
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Parameters and layout preset saved as:\n{preset_name}\n\nPath:\n{config_path}"
                )

        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.log(traceback.format_exc())
            if show_message:
                QMessageBox.critical(self, "Error", str(e))

    def load_params_and_layout(self, show_message=True, source_path=None):
        """
        Load parameters/layout.

        - show_message=False keeps the original startup behavior and reads
          self.config_path without opening a dialog.
        - show_message=True opens a file dialog so a named preset can be
          selected for the current dataset.
        """
        try:
            if source_path is not None:
                config_path = self._ensure_json_config_path(source_path)
            elif show_message:
                selected_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Load Params/Layout Preset",
                    self._get_config_dialog_dir(),
                    "SIM parameter presets (*.json);;JSON files (*.json);;All files (*)",
                )
                if not selected_path:
                    self.log("Load Params/Layout canceled.")
                    return
                config_path = self._ensure_json_config_path(selected_path)
            else:
                config_path = self._ensure_json_config_path(getattr(self, "config_path", "sim_gui_config.json"))

            if not os.path.exists(config_path):
                raise ValueError(f"Config file does not exist: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            paths = config.get("paths", {})
            self.input_dir_edit.setText(paths.get("input_dir", self.input_dir_edit.text()))
            self.single_tiff_path_edit.setText(paths.get("single_tiff_path", self.single_tiff_path_edit.text()))

            self.raw_widefield_path_edit.setText(paths.get("raw_widefield_path", self.raw_widefield_path_edit.text()))
            self.dark_path_edit.setText(paths.get("dark_path", self.dark_path_edit.text()))
            self.flat_path_edit.setText(paths.get("flat_path", self.flat_path_edit.text()))
            if paths.get("input_type"):
                self.ext_combo.setCurrentText(paths.get("input_type"))
            if paths.get("image_order"):
                idx = self.image_order_combo.findData(paths.get("image_order"))
                if idx >= 0:
                    self.image_order_combo.setCurrentIndex(idx)
            elif paths.get("image_order_label"):
                idx = self.image_order_combo.findText(paths.get("image_order_label"))
                if idx >= 0:
                    self.image_order_combo.setCurrentIndex(idx)

            roi = config.get("roi", {})
            self.enable_roi_check.setChecked(bool(roi.get("enabled", True)))
            self.roi_x_spin.setValue(int(roi.get("x", self.roi_x_spin.value())))
            self.roi_y_spin.setValue(int(roi.get("y", self.roi_y_spin.value())))
            self.roi_w_spin.setValue(int(roi.get("w", self.roi_w_spin.value())))
            self.roi_h_spin.setValue(int(roi.get("h", self.roi_h_spin.value())))
            self._update_roi_control_state()

            recon_mode_cfg = config.get("reconstruction_mode", {})
            if recon_mode_cfg.get("mode"):
                self.recon_mode_combo.setCurrentText(recon_mode_cfg.get("mode"))

            preprocess_cfg = config.get("preprocess", {})
            self.enable_preprocess_check.setChecked(bool(preprocess_cfg.get("enabled", self.enable_preprocess_check.isChecked())))
            if preprocess_cfg.get("phase_model"):
                self.phase_model_combo.setCurrentText(preprocess_cfg.get("phase_model"))
            if preprocess_cfg.get("preprocess_mode"):
                self.preprocess_mode_combo.setCurrentText(preprocess_cfg.get("preprocess_mode"))
            self.bg_percentile_spin.setValue(float(preprocess_cfg.get("bg_percentile", self.bg_percentile_spin.value())))
            if preprocess_cfg.get("phase_norm_mode"):
                self.phase_norm_mode_combo.setCurrentText(preprocess_cfg.get("phase_norm_mode"))
            if preprocess_cfg.get("phase_norm_stat"):
                self.phase_norm_stat_combo.setCurrentText(preprocess_cfg.get("phase_norm_stat"))
            if preprocess_cfg.get("os_sim_recon_mode"):
                self.os_sim_recon_mode_combo.setCurrentText(preprocess_cfg.get("os_sim_recon_mode"))
            if preprocess_cfg.get("fourier_filter_mode"):
                self.fourier_filter_mode_combo.setCurrentText(preprocess_cfg.get("fourier_filter_mode"))
            self.fourier_cutoff_spin.setValue(float(preprocess_cfg.get("fourier_cutoff", self.fourier_cutoff_spin.value())))
            self.fourier_order_spin.setValue(int(preprocess_cfg.get("fourier_order", self.fourier_order_spin.value())))
            self.fourier_amount_spin.setValue(float(preprocess_cfg.get("fourier_amount", self.fourier_amount_spin.value())))

            display = config.get("display", {})
            if display.get("view_mode"):
                self.view_mode_combo.setCurrentText(display.get("view_mode"))
            if display.get("colormap"):
                self.colormap_combo.setCurrentText(display.get("colormap"))
            self.auto_contrast_check.setChecked(bool(display.get("auto_contrast", self.auto_contrast_check.isChecked())))
            self.auto_low_spin.setValue(float(display.get("auto_low", self.auto_low_spin.value())))
            self.auto_high_spin.setValue(float(display.get("auto_high", self.auto_high_spin.value())))
            self.manual_min_slider.setValue(int(display.get("manual_min_slider", self.manual_min_slider.value())))
            self.manual_max_slider.setValue(int(display.get("manual_max_slider", self.manual_max_slider.value())))
            self.sync_profile_display_check.setChecked(bool(display.get("sync_profile_display", self.sync_profile_display_check.isChecked())))

            wavelet = config.get("wavelet", {})
            if wavelet.get("fusion_method"):
                self.fusion_method_combo.setCurrentText(wavelet.get("fusion_method"))
            if wavelet.get("wavelet_name"):
                self.wavelet_name_combo.setCurrentText(wavelet.get("wavelet_name"))
            self.wavelet_level_spin.setValue(int(wavelet.get("wavelet_level", self.wavelet_level_spin.value())))
            if wavelet.get("wavelet_detail_mode"):
                self.wavelet_detail_mode_combo.setCurrentText(wavelet.get("wavelet_detail_mode"))
            self.enable_orientation_match_check.setChecked(bool(wavelet.get("enable_orientation_match", self.enable_orientation_match_check.isChecked())))
            if wavelet.get("orientation_match_method"):
                self.orientation_match_method_combo.setCurrentText(wavelet.get("orientation_match_method"))
            if wavelet.get("final_fusion_mode"):
                self.final_fusion_mode_combo.setCurrentText(wavelet.get("final_fusion_mode"))

            postprocess = config.get("postprocess", {})
            self.gaussian_sigma_spin.setValue(float(postprocess.get("gaussian_sigma", self.gaussian_sigma_spin.value())))
            self.enable_smoothing_check.setChecked(bool(postprocess.get("enable_smoothing", self.enable_smoothing_check.isChecked())))
            self.enable_wiener_filter_check.setChecked(bool(postprocess.get("enable_wiener_filter", self.enable_wiener_filter_check.isChecked())))
            if postprocess.get("wiener_target"):
                self.wiener_target_combo.setCurrentText(postprocess.get("wiener_target"))
            self.wiener_kernel_spin.setValue(int(postprocess.get("wiener_kernel", self.wiener_kernel_spin.value())))
            self.wiener_noise_spin.setValue(float(postprocess.get("wiener_noise_variance", self.wiener_noise_spin.value())))
            self.wiener_amount_spin.setValue(float(postprocess.get("wiener_amount", self.wiener_amount_spin.value())))
            self.enable_output_intensity_match_check.setChecked(
                bool(postprocess.get("enable_output_intensity_match", self.enable_output_intensity_match_check.isChecked()))
            )
            if postprocess.get("output_intensity_target"):
                self.output_intensity_target_combo.setCurrentText(postprocess.get("output_intensity_target"))
            if postprocess.get("output_intensity_reference"):
                self.output_intensity_reference_combo.setCurrentText(postprocess.get("output_intensity_reference"))
            if postprocess.get("output_intensity_method"):
                self.output_intensity_method_combo.setCurrentText(postprocess.get("output_intensity_method"))
            self.output_intensity_strength_spin.setValue(
                float(postprocess.get("output_intensity_strength", self.output_intensity_strength_spin.value()))
            )
            self.output_intensity_max_scale_spin.setValue(
                float(postprocess.get("output_intensity_max_scale", self.output_intensity_max_scale_spin.value()))
            )

            self.enable_sim_bg_correction_check.setChecked(bool(postprocess.get("enable_sim_bg_correction", self.enable_sim_bg_correction_check.isChecked())))
            if postprocess.get("sim_bg_mode"):
                self.sim_bg_mode_combo.setCurrentText(postprocess.get("sim_bg_mode"))
            self.sim_bg_sigma_spin.setValue(float(postprocess.get("sim_bg_sigma", self.sim_bg_sigma_spin.value())))

            self.enable_final_structure_enhance_check.setChecked(
                bool(postprocess.get("enable_final_structure_enhance", self.enable_final_structure_enhance_check.isChecked()))
            )
            if postprocess.get("final_structure_target"):
                self.final_structure_target_combo.setCurrentText(postprocess.get("final_structure_target"))
            if postprocess.get("final_structure_mode"):
                self.final_structure_mode_combo.setCurrentText(postprocess.get("final_structure_mode"))
            self.final_structure_sigma_spin.setValue(
                float(postprocess.get("final_structure_sigma", self.final_structure_sigma_spin.value()))
            )
            self.final_structure_amount_spin.setValue(
                float(postprocess.get("final_structure_amount", self.final_structure_amount_spin.value()))
            )
            self.final_structure_keep_spin.setValue(
                float(postprocess.get("final_structure_keep_original", self.final_structure_keep_spin.value()))
            )

            self.enable_hilo_mode_check.setChecked(bool(postprocess.get("enable_hilo_mode", self.enable_hilo_mode_check.isChecked())))
            if postprocess.get("hilo_strategy"):
                self.hilo_strategy_combo.setCurrentText(postprocess.get("hilo_strategy"))
            self.hilo_auto_sigma_check.setChecked(bool(postprocess.get("hilo_auto_sigma", self.hilo_auto_sigma_check.isChecked())))
            if postprocess.get("hilo_widefield_source"):
                self.hilo_widefield_source_combo.setCurrentText(postprocess.get("hilo_widefield_source"))
            if postprocess.get("hilo_low_source"):
                self.hilo_low_source_combo.setCurrentText(postprocess.get("hilo_low_source"))
            self.hilo_sigma_spin.setValue(float(postprocess.get("hilo_sigma", self.hilo_sigma_spin.value())))
            self.hilo_mid_sigma_spin.setValue(float(postprocess.get("hilo_mid_sigma", self.hilo_mid_sigma_spin.value())))
            self.hilo_high_sigma_spin.setValue(float(postprocess.get("hilo_high_sigma", self.hilo_high_sigma_spin.value())))
            self.hilo_lo_weight_spin.setValue(float(postprocess.get("hilo_lo_weight", self.hilo_lo_weight_spin.value())))
            self.hilo_mid_weight_spin.setValue(float(postprocess.get("hilo_mid_weight", self.hilo_mid_weight_spin.value())))
            self.hilo_hi_weight_spin.setValue(float(postprocess.get("hilo_hi_weight", self.hilo_hi_weight_spin.value())))
            self.hilo_clip_output_check.setChecked(bool(postprocess.get("hilo_clip_output", self.hilo_clip_output_check.isChecked())))
            self.enable_hilo_denoise_check.setChecked(bool(postprocess.get("enable_hilo_denoise", self.enable_hilo_denoise_check.isChecked())))
            if postprocess.get("hilo_denoise_method"):
                self.hilo_denoise_method_combo.setCurrentText(postprocess.get("hilo_denoise_method"))
            self.hilo_nlm_h_spin.setValue(float(postprocess.get("hilo_nlm_h", self.hilo_nlm_h_spin.value())))
            self.hilo_nlm_template_spin.setValue(int(postprocess.get("hilo_nlm_template_window", self.hilo_nlm_template_spin.value())))
            self.hilo_nlm_search_spin.setValue(int(postprocess.get("hilo_nlm_search_window", self.hilo_nlm_search_spin.value())))
            self.hilo_bm3d_sigma_spin.setValue(float(postprocess.get("hilo_bm3d_sigma", self.hilo_bm3d_sigma_spin.value())))
            self.hilo_denoise_amount_spin.setValue(float(postprocess.get("hilo_denoise_amount", self.hilo_denoise_amount_spin.value())))

            windows = config.get("windows", {})
            main_win = windows.get("main", {})
            if main_win:
                self._restore_main_window_geometry(main_win)
                splitter_sizes = main_win.get("splitter_sizes")
                if splitter_sizes and hasattr(self, "main_splitter"):
                    try:
                        sizes = [int(v) for v in splitter_sizes]
                        if sizes:
                            default_left = int(getattr(self, "_left_panel_default_width", 400))
                            max_left = int(getattr(self, "_left_panel_max_width", 430))
                            if sizes[0] > max_left:
                                sizes[0] = default_left
                        self.main_splitter.setSizes(sizes)
                    except Exception:
                        pass

            self._apply_profile_window_geometry_dict(windows.get("profile"))

            profile_window_state = config.get("profile_window_state")
            if profile_window_state:
                if self.profile_window is None:
                    self.profile_window = ProfileComparisonWindow(self)
                self.profile_window.apply_ui_state(profile_window_state)

            self._update_display_control_state()
            self._update_manual_labels()
            self._update_reconstruction_mode_ui()

            if self.raw_widefield_path_edit.text().strip():
                self._load_optional_raw_widefield()

            self.update_plot()
            self.update_profile_window_data()

            if profile_window_state and self.profile_window is not None:
                self.profile_window.apply_ui_state(profile_window_state)

            self._last_param_config_path = config_path
            self._last_param_config_dir = os.path.dirname(config_path) or os.getcwd()

            profile = config.get("config_profile", {})
            preset_name = profile.get("preset_name", os.path.splitext(os.path.basename(config_path))[0])
            self.log(f"Loaded parameters/layout preset '{preset_name}' from: {config_path}")

            if show_message:
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Parameters and layout preset loaded:\n{preset_name}\n\nPath:\n{config_path}"
                )

        except Exception as e:
            self.log(f"[ERROR] {e}")
            self.log(traceback.format_exc())
            if show_message:
                QMessageBox.critical(self, "Error", str(e))

    def get_supported_patterns(self):
        input_dir = self.input_dir_edit.text().strip()
        mode = self.ext_combo.currentText()

        if mode == "jpg/jpeg":
            patterns = [
                os.path.join(input_dir, "*.jpg"),
                os.path.join(input_dir, "*.jpeg"),
                os.path.join(input_dir, "*.JPG"),
                os.path.join(input_dir, "*.JPEG"),
            ]
        elif mode == "png":
            patterns = [
                os.path.join(input_dir, "*.png"),
                os.path.join(input_dir, "*.PNG"),
            ]
        elif mode == "tif/tiff":
            patterns = [
                os.path.join(input_dir, "*.tif"),
                os.path.join(input_dir, "*.tiff"),
                os.path.join(input_dir, "*.TIF"),
                os.path.join(input_dir, "*.TIFF"),
            ]
        else:
            patterns = [
                os.path.join(input_dir, "*.jpg"),
                os.path.join(input_dir, "*.jpeg"),
                os.path.join(input_dir, "*.JPG"),
                os.path.join(input_dir, "*.JPEG"),
                os.path.join(input_dir, "*.png"),
                os.path.join(input_dir, "*.PNG"),
                os.path.join(input_dir, "*.tif"),
                os.path.join(input_dir, "*.tiff"),
                os.path.join(input_dir, "*.TIF"),
                os.path.join(input_dir, "*.TIFF"),
            ]
        return patterns

    def collect_image_paths(self):
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            raise ValueError("Input folder is empty.")
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input folder does not exist: {input_dir}")

        image_paths = []
        for pattern in self.get_supported_patterns():
            image_paths.extend(glob.glob(pattern))

        image_paths = sorted(os.path.normpath(p) for p in set(image_paths))

        if len(image_paths) not in (6, 9):
            raise ValueError(
                f"Expected 6 or 9 images in the input folder, but found {len(image_paths)}."
            )

        return image_paths

    def collect_input_items(self):
        """
        收集当前输入源。
        如果 Single TIFF stack 不为空，则优先使用该多页 TIFF；
        否则使用 Input folder 中的 6/9 张图像。
        """
        single_tiff_path = self.single_tiff_path_edit.text().strip()

        if single_tiff_path:
            if not single_tiff_path.lower().endswith((".tif", ".tiff")):
                raise ValueError("Single TIFF stack input must be a .tif or .tiff file.")
            pages = load_multipage_tiff(single_tiff_path)
            if len(pages) not in (6, 9):
                raise ValueError(
                    f"Expected a 6-page or 9-page TIFF stack, but found {len(pages)} pages."
                )
            items = [
                {
                    "source": single_tiff_path,
                    "page_index": i,
                    "image": page,
                    "name": f"{os.path.basename(single_tiff_path)} page {i + 1}",
                }
                for i, page in enumerate(pages)
            ]
            return items, "single_tiff_stack"

        return self.collect_image_paths(), "folder_images"

    def load_input_item_image(self, item):
        """
        将文件夹图像路径或单 TIFF 页对象读取为 float32 图像。
        """
        if isinstance(item, dict):
            return np.asarray(item["image"], dtype=np.float32)
        return load_image(item)

    def input_item_display_name(self, item):
        if isinstance(item, dict):
            return str(item.get("name", f"page {int(item.get('page_index', 0)) + 1}"))
        return os.path.basename(str(item))

    def input_item_source_name(self, item):
        if isinstance(item, dict):
            return f"{os.path.basename(item.get('source', 'stack.tif'))}#page{int(item.get('page_index', 0)) + 1}"
        return os.path.basename(str(item))

    # -------------------------
    # 主处理流程
    # -------------------------
    def run_processing(self):
        if getattr(self, "_processing_thread", None) is not None and self._processing_thread.isRunning():
            QMessageBox.information(self, "Running", "A reconstruction is already running.")
            return

        try:
            config = self._collect_processing_config()
        except Exception as exc:
            self.log(f"[ERROR] {exc}")
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.log_text.clear()
        self.log("Starting processing in background thread...")
        self.log(gpu_backend_status_text())

        self._processing_progress = self._begin_busy_progress(
            "Running Reconstruction",
            "Starting background processing...",
            0,
            100,
        )
        self._set_processing_controls_enabled(False)
        self.run_btn.setText("Running...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._processing_cursor_set = True
        self._set_status_text("Running reconstruction...")

        self._processing_thread = QThread(self)
        self._processing_worker = SIMProcessingWorker(config)
        self._processing_worker.moveToThread(self._processing_thread)

        self._processing_thread.started.connect(self._processing_worker.run)
        self._processing_worker.log_message.connect(self._on_processing_log)
        self._processing_worker.progress_changed.connect(self._on_processing_progress)
        self._processing_worker.finished.connect(self._on_processing_finished)
        self._processing_worker.failed.connect(self._on_processing_error)

        self._processing_thread.start()

    # -------------------------
    # 更新右侧对比图
    # -------------------------
    def _on_analysis_tab_changed(self, index):
        """Keep the embedded Profile Window synchronized when its tab is selected."""
        if hasattr(self, "analysis_tabs") and self.analysis_tabs.widget(index) is getattr(self, "profile_window", None):
            self.update_profile_window_data()

    def _result_canvases(self):
        canvases = []
        for name in ("canvas", "orientation_canvas", "hilo_canvas"):
            canvas = getattr(self, name, None)
            if canvas is not None and canvas not in canvases:
                canvases.append(canvas)
        return canvases

    def _clear_canvas(self, canvas, message=None):
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(1, 1, 1)
        if message:
            ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.axis("off")
        canvas.draw()

    def _draw_view_on_canvas(self, canvas, view_mode):
        canvas.figure.clear()

        cmap_name = self.colormap_combo.currentText()
        external_raw_widefield = self.results.get("external_raw_widefield", None)

        if view_mode == "HiLo component view":
            hilo_hybrid = self.results.get("hilo_hybrid", None)
            if hilo_hybrid is None:
                ax = canvas.figure.add_subplot(1, 1, 1)
                ax.text(
                    0.5, 0.5,
                    "HiLo mode is not enabled or has not been computed.",
                    ha="center", va="center", fontsize=12
                )
                ax.axis("off")
            else:
                panels = [
                    ("Hi source widefield", self.results.get("hilo_widefield_source_image", None)),
                    ("Mid-band low-pass", self.results.get("hilo_widefield_lowpass", None)),
                    ("High-band low-pass", self.results.get("hilo_high_lowpass", None)),
                    ("Mid component |band-pass|", self.results.get("hilo_mid_abs_component", None)),
                    ("High component |high-pass|", self.results.get("hilo_high_abs_component", None)),
                    ("Weighted Mid+High", self.results.get("hilo_mid_high_component", None)),
                    ("Lo component from OS-SIM", self.results.get("hilo_low_component", None)),
                    ("HiLo before denoise", self.results.get("hilo_hybrid_before_hilo_denoise", None)),
                    ("Final HiLo Hybrid", hilo_hybrid),
                ]
                panels = [(title, img) for title, img in panels if img is not None]
                for idx, (title, img) in enumerate(panels):
                    ax = canvas.figure.add_subplot(1, len(panels), idx + 1)
                    disp_img, vmin, vmax = self._get_display_params(img)
                    ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    ax.axis("off")

        elif view_mode == "Orientation view":
            orientation_labels = self.results.get("orientation_labels", [])
            pwf_images = self.results.get("pwf_images", [])
            sim_images = self.results.get("sim_images", [])
            orientation_count = len(pwf_images)
            total_cols = max(1, orientation_count + 1)

            if orientation_count < 1:
                ax = canvas.figure.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, "No orientation images available.", ha="center", va="center", fontsize=12)
                ax.axis("off")
            else:
                for idx in range(orientation_count):
                    ax = canvas.figure.add_subplot(2, total_cols, idx + 1)
                    disp_img, vmin, vmax = self._get_display_params(pwf_images[idx])
                    ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    label = orientation_labels[idx] if idx < len(orientation_labels) else f"orientation{idx + 1}"
                    ax.set_title(f"Widefield {label}")
                    ax.axis("off")

                ax = canvas.figure.add_subplot(2, total_cols, total_cols)
                disp_img, vmin, vmax = self._get_display_params(self.results["pwf_fused"])
                ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax.set_title("Widefield Fused")
                ax.axis("off")

                for idx in range(orientation_count):
                    ax = canvas.figure.add_subplot(2, total_cols, total_cols + idx + 1)
                    disp_img, vmin, vmax = self._get_display_params(sim_images[idx])
                    ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    label = orientation_labels[idx] if idx < len(orientation_labels) else f"orientation{idx + 1}"
                    ax.set_title(f"SIM {label}")
                    ax.axis("off")

                ax = canvas.figure.add_subplot(2, total_cols, 2 * total_cols)
                disp_img, vmin, vmax = self._get_display_params(self.results["sim_wavelet_fused"])
                ax.imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax.set_title("SIM Wavelet Fused")
                ax.axis("off")
        else:
            hilo_hybrid = self.results.get("hilo_hybrid", None)
            panel_count = 3 + (1 if external_raw_widefield is not None else 0) + (1 if hilo_hybrid is not None else 0)
            axes = [canvas.figure.add_subplot(1, panel_count, i + 1) for i in range(panel_count)]
            ax_idx = 0

            if external_raw_widefield is not None:
                disp_img, vmin, vmax = self._get_display_params(external_raw_widefield)
                axes[ax_idx].imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                axes[ax_idx].set_title("Raw Widefield")
                axes[ax_idx].axis("off")
                ax_idx += 1

            disp_img, vmin, vmax = self._get_display_params(self.results["pwf_fused"])
            axes[ax_idx].imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
            axes[ax_idx].set_title("Widefield Fused")
            axes[ax_idx].axis("off")
            ax_idx += 1

            disp_img, vmin, vmax = self._get_display_params(self.results["sim_wavelet_fused"])
            axes[ax_idx].imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
            axes[ax_idx].set_title("SIM Wavelet Fused")
            axes[ax_idx].axis("off")
            ax_idx += 1

            disp_img, vmin, vmax = self._get_display_params(self.results["sim_fused_mean"])
            axes[ax_idx].imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
            axes[ax_idx].set_title(f"Final SIM Fused ({self.results.get('final_fusion_mode', 'RMS')})")
            axes[ax_idx].axis("off")
            ax_idx += 1

            if hilo_hybrid is not None:
                disp_img, vmin, vmax = self._get_display_params(hilo_hybrid)
                axes[ax_idx].imshow(disp_img, cmap=cmap_name, vmin=vmin, vmax=vmax)
                axes[ax_idx].set_title("HiLo Hybrid")
                axes[ax_idx].axis("off")

        canvas.figure.tight_layout()
        canvas.draw()

    def refresh_preview(self, *args):
        self.update_plot(show_progress=True)

    def update_plot(self, *args, show_progress=False):
        progress = None
        if show_progress:
            progress = self._begin_busy_progress("Refreshing Preview", "Preparing preview refresh...", 0, 100)
            self._set_processing_controls_enabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._set_busy_progress(progress, "Checking available reconstruction results...", 10)
            if not self.results or "pwf_fused" not in self.results or "sim_wavelet_fused" not in self.results or "sim_fused_mean" not in self.results:
                self._update_display_control_state()
                self._update_manual_labels()
                for canvas in self._result_canvases():
                    self._clear_canvas(canvas, "Run reconstruction first.")
                if self.profile_window is not None:
                    self.profile_window.redraw()
                self._set_busy_progress(progress, "No reconstruction result is available yet.", 100)
                return

            self._set_busy_progress(progress, "Updating display controls...", 20)
            self._update_display_control_state()
            self._update_manual_labels()

            self._set_busy_progress(progress, "Drawing final comparison view...", 45)
            self._draw_view_on_canvas(self.canvas, "Final comparison view")
            if hasattr(self, "orientation_canvas"):
                self._set_busy_progress(progress, "Drawing orientation view...", 65)
                self._draw_view_on_canvas(self.orientation_canvas, "Orientation view")
            if hasattr(self, "hilo_canvas"):
                self._set_busy_progress(progress, "Drawing HiLo component view...", 82)
                self._draw_view_on_canvas(self.hilo_canvas, "HiLo component view")

            self._set_busy_progress(progress, "Updating profile view...", 94)
            self.update_profile_window_data()
            self._set_busy_progress(progress, "Preview refresh complete.", 100)
        finally:
            if show_progress:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                self._set_processing_controls_enabled(True)
                self._close_busy_progress(progress)

    # -------------------------
    # 线剖面窗口 / Profile Tab
    # -------------------------
    def open_profile_window(self):
        """Switch to the embedded Profile Window tab instead of opening a separate window."""
        self.update_profile_window_data()
        if hasattr(self, "analysis_tabs") and self.profile_window is not None:
            idx = self.analysis_tabs.indexOf(self.profile_window)
            if idx >= 0:
                self.analysis_tabs.setCurrentIndex(idx)

    def update_profile_window_data(self):
        if self.profile_window is None:
            return

        display_cfg = self._get_profile_window_display_config()
        self.profile_window.set_display_config(**display_cfg)

        if not self.results or "pwf_fused" not in self.results or "sim_wavelet_fused" not in self.results or "sim_fused_mean" not in self.results:
            return

        raw_widefield_img = self.results.get("external_raw_widefield", None)
        if raw_widefield_img is None:
            raw_widefield_img = self.results.get("raw_widefield_fused", None)

        self.profile_window.set_images(
            self.results["pwf_fused"],
            self.results["sim_wavelet_fused"],
            self.results["sim_fused_mean"],
            raw_widefield_img=raw_widefield_img,
            fusion_method_label=f"Final SIM Fused ({self.results.get('final_fusion_mode', 'RMS')})",
            hilo_hybrid_img=self.results.get("hilo_hybrid", None),
        )

    # -------------------------
    # 保存结果
    # -------------------------

    def _safe_filename_part(self, text):
        text = str(text).strip().lower().replace(" ", "_")
        keep = []
        for ch in text:
            if ch.isalnum() or ch in ("_", "-", "."):
                keep.append(ch)
            else:
                keep.append("_")
        safe = "".join(keep).strip("_")
        return safe or "item"

    def _phase_name(self, phase_idx):
        phase_names = ["phase0", "phase90", "phase180"]
        if 0 <= phase_idx < len(phase_names):
            return phase_names[phase_idx]
        return f"phase{phase_idx + 1}"

    def _save_step_image(self, records, base_dir, subfolder, filename, img, description):
        folder = os.path.join(base_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        save_image_tif(path, img)
        records.append({
            "folder": subfolder,
            "filename": filename,
            "description": description,
        })

        progress_callback = getattr(self, "_save_progress_callback", None)
        if callable(progress_callback):
            progress_callback(f"{subfolder}/{filename}")

        return path

    def _fft_log_spectrum(self, img):
        """
        生成用于显示的二维 FFT log 幅度谱。
        输出为非负 float32，可直接用 save_image_tif 保存。
        """
        arr = np.asarray(img, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            return arr.astype(np.float32)

        arr = arr - float(np.mean(arr))
        spectrum = np.fft.fftshift(np.fft.fft2(arr))
        mag = np.log1p(np.abs(spectrum))
        return np.asarray(mag, dtype=np.float32)

    def _resize_to_image_shape(self, arr, shape):
        """
        将系数图 resize 到原图大小，仅用于诊断可视化。
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == tuple(shape):
            return arr
        return cv2.resize(arr, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    def _dwt_single_subband_reconstruction(self, coeffs, wavelet, img_shape, detail_index=None, band=None, keep_approx=False):
        """
        从 DWT 系数中只保留某一部分并反变换回原图大小。
        """
        new_coeffs = []

        if keep_approx:
            new_coeffs.append(coeffs[0].copy())
        else:
            new_coeffs.append(np.zeros_like(coeffs[0]))

        for idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[idx]
            zH, zV, zD = np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)

            if idx == detail_index:
                if band == "H":
                    zH = cH.copy()
                elif band == "V":
                    zV = cV.copy()
                elif band == "D":
                    zD = cD.copy()
                elif band == "all":
                    zH, zV, zD = cH.copy(), cV.copy(), cD.copy()

            new_coeffs.append((zH, zV, zD))

        rec = pywt.waverec2(new_coeffs, wavelet=wavelet)
        rec = rec[:img_shape[0], :img_shape[1]]
        return np.asarray(rec, dtype=np.float32)

    def _swt_single_band_map(self, coeffs, img_shape, coeff_level_index=None, band=None, keep_approx=False):
        """
        SWT 不下采样，系数本身通常已经是原图尺寸。
        这里输出系数幅值图，作为该层/方向频带的空间分布诊断图。
        """
        if keep_approx:
            cA = coeffs[-1][0]
            return self._resize_to_image_shape(cA, img_shape)

        cA, (cH, cV, cD) = coeffs[coeff_level_index]
        if band == "H":
            arr = cH
        elif band == "V":
            arr = cV
        elif band == "D":
            arr = cD
        else:
            arr = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)

        return self._resize_to_image_shape(arr, img_shape)

    def _save_wavelet_diagnostics(self, records, steps_dir):
        """
        输出每个方向 SIM 图的小波分解诊断图：
        - 低频近似部分
        - 总高频部分
        - 每层 H/V/D 高频细节
        - 对应空间图的 log FFT 频域分布

        使用 self.results['sim_images']，即方向强度匹配后的单方向 SIM 图，
        也是后续小波融合使用的输入图。
        """
        sim_images = self.results.get("sim_images", []) or []
        if not sim_images:
            return

        method = str(self.results.get("fusion_method", "DWT")).upper()
        wavelet = self.results.get("wavelet_name", "db2")
        requested_level = int(self.results.get("wavelet_level", 2))
        orientation_labels = self.results.get("orientation_labels", [])

        subfolder = "09_wavelet_decomposition_diagnostics"

        def label_for(idx):
            if idx < len(orientation_labels):
                return self._safe_filename_part(orientation_labels[idx])
            return f"orientation{idx + 1}"

        for ori_idx, img in enumerate(sim_images):
            img = np.asarray(img, dtype=np.float32)
            label = label_for(ori_idx)
            base = f"ori{ori_idx + 1:02d}_{label}"

            self._save_step_image(
                records,
                steps_dir,
                subfolder,
                f"{base}_00_wavelet_input_sim.tif",
                img,
                f"Wavelet decomposition input for orientation {ori_idx + 1}: matched single-orientation SIM image.",
            )
            self._save_step_image(
                records,
                steps_dir,
                subfolder,
                f"{base}_00_wavelet_input_sim_fft_log_spectrum.tif",
                self._fft_log_spectrum(img),
                f"FFT log spectrum of wavelet input SIM image for orientation {ori_idx + 1}.",
            )

            if method == "SWT":
                max_level = pywt.swt_max_level(min(img.shape))
                if max_level < 1:
                    self.log(f"[WARN] Skip SWT diagnostics for {base}: max SWT level < 1")
                    continue

                actual_level = max(1, min(requested_level, max_level))
                coeffs = pywt.swt2(img, wavelet=wavelet, level=actual_level)

                low = self._swt_single_band_map(coeffs, img.shape, keep_approx=True)
                high_total_abs = np.abs(img - low).astype(np.float32)

                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_01_swt_lowfreq_cA_level{actual_level:02d}.tif",
                    low,
                    f"SWT low-frequency approximation cA at level {actual_level} for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_01_swt_lowfreq_cA_level{actual_level:02d}_fft_log_spectrum.tif",
                    self._fft_log_spectrum(low),
                    f"FFT log spectrum of SWT low-frequency approximation for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_02_swt_total_highfreq_abs.tif",
                    high_total_abs,
                    f"Absolute total high-frequency component estimated as |SIM - SWT low-frequency approximation| for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_02_swt_total_highfreq_abs_fft_log_spectrum.tif",
                    self._fft_log_spectrum(high_total_abs),
                    f"FFT log spectrum of SWT total high-frequency component for orientation {ori_idx + 1}.",
                )

                for coeff_idx in range(len(coeffs)):
                    scale_label = actual_level - coeff_idx
                    for band, band_name in [("H", "horizontal"), ("V", "vertical"), ("D", "diagonal"), ("all", "energy")]:
                        band_map = self._swt_single_band_map(coeffs, img.shape, coeff_level_index=coeff_idx, band=band)
                        band_img = np.abs(band_map).astype(np.float32)
                        fname = f"{base}_03_swt_level{scale_label:02d}_{band_name}_detail_abs.tif"
                        self._save_step_image(
                            records, steps_dir, subfolder, fname, band_img,
                            f"SWT level {scale_label} {band_name} high-frequency detail absolute map for orientation {ori_idx + 1}.",
                        )
                        self._save_step_image(
                            records, steps_dir, subfolder,
                            fname.replace(".tif", "_fft_log_spectrum.tif"),
                            self._fft_log_spectrum(band_img),
                            f"FFT log spectrum of SWT level {scale_label} {band_name} detail map for orientation {ori_idx + 1}.",
                        )

            else:
                max_level = pywt.dwtn_max_level(img.shape, wavelet)
                actual_level = max(1, min(requested_level, max_level)) if max_level > 0 else 1
                coeffs = pywt.wavedec2(img, wavelet=wavelet, level=actual_level)

                low = self._dwt_single_subband_reconstruction(
                    coeffs, wavelet=wavelet, img_shape=img.shape, keep_approx=True
                )
                high_total_abs = np.abs(img - low).astype(np.float32)

                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_01_dwt_lowfreq_cA_level{actual_level:02d}.tif",
                    low,
                    f"DWT low-frequency reconstruction from cA at level {actual_level} for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_01_dwt_lowfreq_cA_level{actual_level:02d}_fft_log_spectrum.tif",
                    self._fft_log_spectrum(low),
                    f"FFT log spectrum of DWT low-frequency reconstruction for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_02_dwt_total_highfreq_abs.tif",
                    high_total_abs,
                    f"Absolute total high-frequency component estimated as |SIM - DWT low-frequency reconstruction| for orientation {ori_idx + 1}.",
                )
                self._save_step_image(
                    records, steps_dir, subfolder,
                    f"{base}_02_dwt_total_highfreq_abs_fft_log_spectrum.tif",
                    self._fft_log_spectrum(high_total_abs),
                    f"FFT log spectrum of DWT total high-frequency component for orientation {ori_idx + 1}.",
                )

                for coeff_index in range(1, len(coeffs)):
                    scale_label = actual_level - coeff_index + 1
                    for band, band_name in [("H", "horizontal"), ("V", "vertical"), ("D", "diagonal"), ("all", "energy")]:
                        rec = self._dwt_single_subband_reconstruction(
                            coeffs,
                            wavelet=wavelet,
                            img_shape=img.shape,
                            detail_index=coeff_index,
                            band=band,
                        )
                        rec_abs = np.abs(rec).astype(np.float32)
                        fname = f"{base}_03_dwt_level{scale_label:02d}_{band_name}_detail_recon_abs.tif"
                        self._save_step_image(
                            records, steps_dir, subfolder, fname, rec_abs,
                            f"DWT level {scale_label} {band_name} high-frequency detail reconstructed to image size, absolute value, for orientation {ori_idx + 1}.",
                        )
                        self._save_step_image(
                            records, steps_dir, subfolder,
                            fname.replace(".tif", "_fft_log_spectrum.tif"),
                            self._fft_log_spectrum(rec_abs),
                            f"FFT log spectrum of DWT level {scale_label} {band_name} detail reconstruction for orientation {ori_idx + 1}.",
                        )

    def export_processing_steps(self, output_dir):
        """
        将处理流程中的关键中间图像按步骤输出到子文件夹，
        并生成 README_steps.txt 说明每个文件名对应的处理阶段。
        """
        if not self.results:
            raise ValueError("No results to export.")

        steps_dir = os.path.join(output_dir, "processing_steps")
        os.makedirs(steps_dir, exist_ok=True)

        records = []
        orientation_labels = self.results.get("orientation_labels", [])
        phase_names = ["phase0", "phase90", "phase180"]

        def label_for(idx):
            if idx < len(orientation_labels):
                return self._safe_filename_part(orientation_labels[idx])
            return f"orientation{idx + 1}"

        def save_group(subfolder, prefix, groups, description_prefix):
            for ori_idx, group in enumerate(groups or []):
                label = label_for(ori_idx)
                for phase_idx, img in enumerate(group):
                    phase = phase_names[phase_idx] if phase_idx < len(phase_names) else f"phase{phase_idx + 1}"
                    filename = f"{prefix}_{ori_idx + 1:02d}_{label}_{phase}.tif"
                    self._save_step_image(
                        records,
                        steps_dir,
                        subfolder,
                        filename,
                        img,
                        f"{description_prefix}: orientation {ori_idx + 1} ({label}), {phase}.",
                    )

        def save_sequence(subfolder, prefix, images, description_prefix):
            for ori_idx, img in enumerate(images or []):
                label = label_for(ori_idx)
                filename = f"{prefix}_{ori_idx + 1:02d}_{label}.tif"
                self._save_step_image(
                    records,
                    steps_dir,
                    subfolder,
                    filename,
                    img,
                    f"{description_prefix}: orientation {ori_idx + 1} ({label}).",
                )

        # 00 Ordered ROI input images
        raw_cropped = self.results.get("raw_cropped_images", [])
        orientation_count = self.results.get("orientation_count", len(orientation_labels))
        for idx, img in enumerate(raw_cropped):
            ori_idx = idx // 3
            phase_idx = idx % 3
            label = label_for(ori_idx)
            phase = phase_names[phase_idx] if phase_idx < len(phase_names) else f"phase{phase_idx + 1}"
            filename = f"ordered_roi_{idx + 1:02d}_ori{ori_idx + 1:02d}_{label}_{phase}.tif"
            source_items = self.results.get("source_items_in_processing_order", [])
            source_text = source_items[idx] if idx < len(source_items) else "unknown source"
            self._save_step_image(
                records,
                steps_dir,
                "00_ordered_roi_input",
                filename,
                img,
                f"Effective input after image/page order adjustment and optional ROI crop. Source item: {source_text}.",
            )

        # 01 Raw widefield before preprocess
        raw_wf_images = self.results.get("raw_widefield_images", [])
        save_sequence(
            "01_raw_widefield_from_roi",
            "raw_pwf",
            raw_wf_images,
            "Pseudo-widefield computed from ordered effective input before preprocess",
        )
        if self.results.get("raw_widefield_fused", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "01_raw_widefield_from_roi",
                "raw_pwf_fused_all_orientations.tif",
                self.results["raw_widefield_fused"],
                "Average of raw pseudo-widefield images from all orientations before preprocess.",
            )

        # 02 Preprocess
        save_group(
            "02_preprocess_or_raw_used",
            "preprocess",
            self.results.get("processed_groups", []),
            "Phase image used after preprocess; if preprocess is disabled, this is a raw effective-input copy",
        )

        # 03 Phase normalization
        save_group(
            "03_phase_normalization",
            "phase_norm",
            self.results.get("normalized_orientation_groups", []),
            "Phase image after unified phase normalization",
        )

        # 04 Pseudo-widefield from normalized phases
        save_sequence(
            "04_pseudo_widefield",
            "pwf_before_orientation_match",
            self.results.get("pwf_images_raw", []),
            "Pseudo-widefield from normalized phase images before orientation intensity matching",
        )
        save_sequence(
            "04_pseudo_widefield",
            "pwf_after_orientation_match",
            self.results.get("pwf_images", []),
            "Pseudo-widefield after optional orientation intensity matching",
        )
        if self.results.get("pwf_fused", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "04_pseudo_widefield",
                "pwf_fused_after_orientation_match.tif",
                self.results["pwf_fused"],
                "Final fused pseudo-widefield after optional orientation intensity matching.",
            )

        # 05 SIM reconstruction and single-orientation background correction
        save_sequence(
            "05_sim_reconstruction",
            "sim_before_bg_correction",
            self.results.get("sim_images_before_bg_correction", []),
            "Single-orientation SIM reconstruction before low-frequency background correction",
        )
        save_sequence(
            "05_sim_reconstruction",
            "sim_bg_estimate",
            self.results.get("sim_bg_estimates", []),
            "Estimated low-frequency background for each single-orientation SIM image",
        )
        save_sequence(
            "05_sim_reconstruction",
            "sim_after_bg_correction",
            self.results.get("sim_images_after_bg_correction", []),
            "Single-orientation SIM after optional low-frequency background correction",
        )

        # 06 Orientation intensity matching
        save_sequence(
            "06_orientation_intensity_match",
            "sim_after_orientation_match",
            self.results.get("sim_images", []),
            "SIM image after optional orientation intensity matching and before final fusion",
        )

        # 07 Fusion
        if self.results.get("sim_fused_arithmetic_mean", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "07_fusion",
                "final_sim_fused_arithmetic_mean_reference.tif",
                self.results["sim_fused_arithmetic_mean"],
                "Arithmetic-mean fusion of matched SIM orientations, saved as a reference.",
            )
        if self.results.get("sim_fused_before_postprocess", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "07_fusion",
                "final_sim_fused_before_postprocess.tif",
                self.results["sim_fused_before_postprocess"],
                "Final SIM fused by selected mode before HiLo/postprocess.",
            )
        if self.results.get("sim_fused_before_final_structure_enhance", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "07_fusion",
                "final_sim_fused_before_structure_enhance.tif",
                self.results["sim_fused_before_final_structure_enhance"],
                "Final SIM fused by selected mode before final structure enhancement.",
            )
        if self.results.get("sim_wavelet_before_postprocess", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "07_fusion",
                "sim_wavelet_fused_before_postprocess.tif",
                self.results["sim_wavelet_before_postprocess"],
                "Wavelet-fused SIM before HiLo/postprocess.",
            )
        if self.results.get("sim_wavelet_before_final_structure_enhance", None) is not None:
            self._save_step_image(
                records,
                steps_dir,
                "07_fusion",
                "sim_wavelet_fused_before_structure_enhance.tif",
                self.results["sim_wavelet_before_final_structure_enhance"],
                "Wavelet-fused SIM before final structure enhancement.",
            )

        # 08 Final structure enhancement
        for key, filename, desc in [
            ("sim_fused_lowfreq_bg", "final_sim_lowfreq_background.tif", "Low-frequency background estimated from final SIM fused image."),
            ("sim_fused_highpass", "final_sim_highpass_structure.tif", "High-pass structure component extracted from final SIM fused image."),
            ("sim_fused_before_smoothing", "final_sim_after_structure_enhance_before_smoothing.tif", "Final SIM after optional structure enhancement and before Gaussian smoothing."),
            ("sim_wavelet_lowfreq_bg", "sim_wavelet_lowfreq_background.tif", "Low-frequency background estimated from wavelet-fused SIM image."),
            ("sim_wavelet_highpass", "sim_wavelet_highpass_structure.tif", "High-pass structure component extracted from wavelet-fused SIM image."),
            ("sim_wavelet_before_smoothing", "sim_wavelet_after_structure_enhance_before_smoothing.tif", "Wavelet-fused SIM after optional structure enhancement and before Gaussian smoothing."),
            ("sim_fused_before_wiener", "final_sim_before_wiener_filter.tif", "Final SIM before optional Wiener filtering."),
            ("sim_wavelet_before_wiener", "sim_wavelet_before_wiener_filter.tif", "Wavelet-fused SIM before optional Wiener filtering."),
        ]:
            if self.results.get(key, None) is not None:
                self._save_step_image(records, steps_dir, "08_final_structure_enhancement", filename, self.results[key], desc)

        # 09 Wavelet decomposition diagnostics
        self._save_wavelet_diagnostics(records, steps_dir)

        # 10 HiLo per-orientation outputs
        save_sequence(
            "10_hilo_outputs",
            "hilo_orientation_hybrid_before_postprocess",
            self.results.get("hilo_orientation_hybrids", []),
            "Per-orientation HiLo hybrid before final postprocess; these are RMS-fused into the final HiLo Hybrid",
        )

        # 10 Final outputs after postprocess
        final_outputs = [
            ("pwf_fused", "widefield_fused_final.tif", "Final fused widefield output."),
            ("raw_widefield_fused", "raw_widefield_fused_from_ordered_roi.tif", "Raw widefield fused from ordered effective inputs."),
            ("external_raw_widefield", "external_raw_widefield_roi.tif", "Optional external raw widefield ROI."),
            ("sim_fused_mean", "final_sim_fused_after_smoothing.tif", "Final SIM fused output after optional Gaussian smoothing."),
            ("sim_wavelet_fused", "sim_wavelet_fused_after_smoothing.tif", "Wavelet-fused SIM output after optional Gaussian smoothing."),
            ("hilo_widefield_source_image", "hilo_hi_source_widefield.tif", "Widefield image used as the Hi source for HiLo mode."),
            ("hilo_widefield_lowpass", "hilo_mid_band_lowpass.tif", "Low-pass image used to define the HiLo mid-frequency band."),
            ("hilo_high_lowpass", "hilo_high_band_lowpass.tif", "Low-pass image used to define the HiLo high-frequency band."),
            ("hilo_mid_abs_component", "hilo_mid_component_abs.tif", "Absolute mid-frequency HiLo band-pass component."),
            ("hilo_high_abs_component", "hilo_high_component_abs.tif", "Absolute high-frequency HiLo high-pass component."),
            ("hilo_mid_high_component", "hilo_weighted_mid_high_component.tif", "Weighted signed mid-plus-high component added to the HiLo Lo component."),
            ("hilo_low_component", "hilo_low_component_from_os_sim.tif", "Low-pass Lo component from the selected OS-SIM result."),
            ("hilo_hybrid_before_postprocess", "hilo_hybrid_before_postprocess.tif", "HiLo hybrid immediately after Lo/Mid/High fusion and before all final postprocessing."),
            ("hilo_hybrid_before_hilo_denoise", "hilo_hybrid_before_hilo_denoise.tif", "HiLo hybrid before the HiLo-only NLM/BM3D denoise stage."),
            ("hilo_hybrid_before_final_structure_enhance", "hilo_hybrid_before_structure_enhance.tif", "HiLo hybrid before optional low-frequency structure/background correction."),
            ("hilo_lowfreq_bg", "hilo_lowfreq_background.tif", "Low-frequency background estimated from HiLo hybrid when final structure enhancement targets HiLo."),
            ("hilo_highpass", "hilo_highpass_structure.tif", "High-pass structure component from HiLo hybrid when final structure enhancement targets HiLo."),
            ("hilo_hybrid_before_smoothing", "hilo_hybrid_before_smoothing.tif", "HiLo hybrid before optional Gaussian smoothing stage."),
            ("hilo_hybrid_before_wiener", "hilo_hybrid_before_wiener_filter.tif", "HiLo hybrid before optional Wiener filtering."),
            ("hilo_hybrid", "hilo_hybrid_os_sim_plus_hilo.tif", "Final OS-SIM + HiLo hybrid image."),
        ]
        for key, filename, desc in final_outputs:
            if self.results.get(key, None) is not None:
                self._save_step_image(records, steps_dir, "10_final_outputs", filename, self.results[key], desc)

        # 11 Views
        views_dir = os.path.join(steps_dir, "11_views")
        os.makedirs(views_dir, exist_ok=True)
        progress_callback = getattr(self, "_save_progress_callback", None)

        comparison_view_path = os.path.join(views_dir, "comparison_view.png")
        self.canvas.figure.savefig(
            comparison_view_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )
        records.append({
            "folder": "11_views",
            "filename": "comparison_view.png",
            "description": "Current GUI comparison view saved as PNG.",
        })
        if callable(progress_callback):
            progress_callback("11_views/comparison_view.png")

        if self.profile_window is not None and self.profile_window.isVisible():
            profile_view_path = os.path.join(views_dir, "profile_comparison_view.png")
            self.profile_window.save_current_view(profile_view_path)
            records.append({
                "folder": "11_views",
                "filename": "profile_comparison_view.png",
                "description": "Current profile comparison window view saved as PNG.",
            })
            if callable(progress_callback):
                progress_callback("11_views/profile_comparison_view.png")

        readme_path = os.path.join(steps_dir, "README_steps.txt")
        roi = self.results.get("roi", {})
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("SIM processing steps image export\\n")
            f.write("=" * 40 + "\\n\\n")
            f.write("说明：所有 TIFF 图像由脚本保存为 16-bit 显示型 TIFF；每张图会独立归一化到 0~65535，适合查看与对比。\\n")
            f.write("如果要做严格定量，请优先使用程序内 float32 结果或另行保存未归一化数据。\\n\\n")

            f.write("[Input / Order / ROI]\\n")
            f.write(f"Input source type: {self.results.get('input_source_type', '')}\\n")
            f.write(f"Single TIFF path: {self.results.get('single_tiff_path', '')}\\n")
            f.write(f"Image order mode: {self.results.get('image_order_mode', '')}\\n")
            f.write(f"Image order label: {self.results.get('image_order_label', '')}\\n")
            f.write(f"Dark image path: {self.results.get('dark_image_path', '')}\\n")
            f.write(f"Flat image path: {self.results.get('flat_image_path', '')}\\n")
            f.write(f"ROI enabled: {roi.get('enabled', True)}\\n")
            f.write(f"ROI/effective processing region: x={roi.get('x', '')}, y={roi.get('y', '')}, width={roi.get('width', '')}, height={roi.get('height', '')}\\n")
            f.write(f"Orientation count: {orientation_count}\\n")
            f.write("Orientation labels: " + ", ".join(orientation_labels) + "\\n\\n")

            f.write("[Processing parameters]\\n")
            f.write(f"Phase model: {self.results.get('phase_model', '')}\\n")
            f.write(f"Preprocess enabled: {self.results.get('preprocess_enabled', '')}\\n")
            f.write(f"Preprocess mode: {self.results.get('preprocess_mode', '')}\\n")
            f.write(f"Background percentile: {self.results.get('background_percentile', '')}\\n")
            f.write(f"Phase normalization: mode={self.results.get('phase_norm_mode', '')}, stat={self.results.get('phase_norm_stat', '')}\\n")
            f.write(f"Single SIM background correction: enabled={self.results.get('enable_sim_bg_correction', '')}, mode={self.results.get('sim_bg_mode', '')}, sigma={self.results.get('sim_bg_sigma', '')}\\n")
            f.write(f"Orientation matching: enabled={self.results.get('enable_orientation_match', '')}, stat={self.results.get('orientation_match_method', '')}\\n")
            f.write(f"Final SIM fusion mode: {self.results.get('final_fusion_mode', '')}\\n")
            f.write(f"Wavelet fusion: method={self.results.get('fusion_method', '')}, wavelet={self.results.get('wavelet_name', '')}, level={self.results.get('wavelet_level', '')}, detail={self.results.get('wavelet_detail_mode', '')}\\n")
            f.write(f"Final structure enhancement: enabled={self.results.get('enable_final_structure_enhance', '')}, target={self.results.get('final_structure_target', '')}, mode={self.results.get('final_structure_mode', '')}, sigma={self.results.get('final_structure_sigma', '')}, amount={self.results.get('final_structure_amount', '')}, keep_original={self.results.get('final_structure_keep_original', '')}\\n")
            f.write(f"Gaussian smoothing: enabled={self.results.get('enable_smoothing', '')}, sigma={self.results.get('gaussian_sigma', '')}\\n")
            f.write(f"Wiener filter: enabled={self.results.get('enable_wiener_filter', '')}, target={self.results.get('wiener_target', '')}, kernel={self.results.get('wiener_actual_kernel', self.results.get('wiener_kernel', ''))}, noise_variance_input={self.results.get('wiener_noise_variance', '')}, noise_variance_actual={self.results.get('wiener_actual_noise_variance', '')}, amount={self.results.get('wiener_amount', '')}, applied_to={', '.join(self.results.get('wiener_applied_to', []) or [])}\\n")
            f.write(
                f"Output intensity preservation: "
                f"enabled={self.results.get('enable_output_intensity_match', '')}, "
                f"target={self.results.get('output_intensity_target', '')}, "
                f"reference={self.results.get('output_intensity_reference', '')}, "
                f"method={self.results.get('output_intensity_method', '')}, "
                f"strength={self.results.get('output_intensity_strength', '')}, "
                f"max_scale={self.results.get('output_intensity_max_scale', '')}, "
                f"info={self.results.get('output_intensity_match_info', {})}\\n"
            )

            f.write(f"HiLo: enabled={self.results.get('enable_hilo_mode', '')}, strategy={self.results.get('hilo_strategy', '')}, auto_sigma={self.results.get('hilo_auto_sigma', '')}, widefield_source={self.results.get('hilo_widefield_source', '')}, low_source={self.results.get('hilo_low_source', '')}, lo_sigma={self.results.get('hilo_sigma', '')}, mid_sigma={self.results.get('hilo_mid_sigma', '')}, high_sigma={self.results.get('hilo_high_sigma', '')}, lo_weight={self.results.get('hilo_lo_weight', '')}, mid_weight={self.results.get('hilo_mid_weight', '')}, high_weight={self.results.get('hilo_hi_weight', '')}, clip={self.results.get('hilo_clip_output', '')}\n")
            f.write(f"HiLo cutoff/sigma recommendation: {self.results.get('hilo_auto_sigma_info', {})}\n")
            f.write(f"HiLo-only denoise: enabled={self.results.get('enable_hilo_denoise', '')}, method={self.results.get('hilo_denoise_method', '')}, nlm_h={self.results.get('hilo_nlm_h', '')}, template={self.results.get('hilo_nlm_template_window', '')}, search={self.results.get('hilo_nlm_search_window', '')}, bm3d_sigma={self.results.get('hilo_bm3d_sigma', '')}, amount={self.results.get('hilo_denoise_amount', '')}, info={self.results.get('hilo_denoise_info', {})}\n\n")

            f.write("[Source items in processing order]\\n")
            for idx, item in enumerate(self.results.get("source_items_in_processing_order", []), 1):
                f.write(f"{idx:02d}. {item}\\n")
            f.write("\\n")

            f.write("[Phase / stripe diagnostics]\\n")
            raw_diags = self.results.get("raw_phase_diagnostics", []) or []
            proc_diags = self.results.get("processed_phase_diagnostics", []) or []
            for idx, diag in enumerate(raw_diags, 1):
                label = orientation_labels[idx - 1] if idx - 1 < len(orientation_labels) else f"Orientation {idx}"
                f.write("Raw " + format_phase_group_diagnostics(label, diag) + "\\n")
            for idx, diag in enumerate(proc_diags, 1):
                label = orientation_labels[idx - 1] if idx - 1 < len(orientation_labels) else f"Orientation {idx}"
                f.write("Processed " + format_phase_group_diagnostics(label, diag) + "\\n")
            f.write("\\n")

            f.write("[Wavelet decomposition diagnostics]\\n")
            f.write("09_wavelet_decomposition_diagnostics/ contains one diagnostic set for each orientation SIM image used as wavelet-fusion input.\\n")
            f.write("Files containing lowfreq_cA are low-frequency approximation images.\\n")
            f.write("Files containing total_highfreq_abs are absolute high-frequency components estimated from input minus low-frequency approximation.\\n")
            f.write("Files containing horizontal/vertical/diagonal_detail are H/V/D high-frequency detail components for each wavelet level.\\n")
            f.write("Files ending with fft_log_spectrum are log-magnitude FFT frequency-domain distributions of the corresponding spatial-domain image.\\n\\n")

            f.write("[File map]\\n")
            current_folder = None
            for record in records:
                if record["folder"] != current_folder:
                    current_folder = record["folder"]
                    f.write(f"\\n{current_folder}/\\n")
                f.write(f"  {record['filename']} : {record['description']}\\n")

        if callable(progress_callback):
            progress_callback("processing_steps/README_steps.txt")

        self.log(f"Saved processing step folders to: {steps_dir}")
        self.log(f"Saved processing step description to: {readme_path}")
        return steps_dir, readme_path

    def open_output_folder(self, output_dir=None):
        output_dir = output_dir or self._get_effective_output_dir()
        output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Open output folder", f"Output folder does not exist:\n{output_dir}")
            return False
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
        if not ok:
            QMessageBox.warning(self, "Open output folder", f"Could not open output folder:\n{output_dir}")
        return bool(ok)

    def _show_saved_results_message(self, title, message, output_dir):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(message)
        open_button = box.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_button:
            self.open_output_folder(output_dir)

    def save_results(self):
        progress = None
        try:
            if not self.results:
                raise ValueError("No results to save. Please run processing first.")

            output_dir = self._get_effective_output_dir()

            reply = QMessageBox.question(
                self,
                "Save intermediate files?",
                "Do you want to save intermediate processing files?\n\n"
                "Yes: save final results, comparison views, top-level intermediate images, "
                "and the full processing_steps folder.\n"
                "No: save only final result images and comparison views.\n"
                "Cancel: cancel saving.",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                    | QMessageBox.StandardButton.Cancel
                ),
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                self.log("Save canceled before writing files.")
                return

            save_intermediate_files = reply == QMessageBox.StandardButton.Yes

            os.makedirs(output_dir, exist_ok=True)

            orientation_labels = self.results.get("orientation_labels", [])
            processed_groups = self.results.get("processed_groups", [])
            pwf_images = self.results.get("pwf_images", [])
            sim_images = self.results.get("sim_images", [])
            phase_suffixes = ["i0", "i90", "i180"]

            def label_for(idx):
                return orientation_labels[idx] if idx < len(orientation_labels) else f"orientation{idx + 1}"

            def safe_label_for(idx):
                return self._safe_filename_part(label_for(idx))

            def count_optional_images(keys):
                return sum(1 for key in keys if self.results.get(key, None) is not None)

            def estimate_processing_steps_count():
                count = 0
                count += len(self.results.get("raw_cropped_images", []) or [])
                count += len(self.results.get("raw_widefield_images", []) or [])
                count += 1 if self.results.get("raw_widefield_fused", None) is not None else 0
                count += sum(len(group) for group in (self.results.get("processed_groups", []) or []))
                count += sum(len(group) for group in (self.results.get("normalized_orientation_groups", []) or []))
                count += len(self.results.get("pwf_images_raw", []) or [])
                count += len(self.results.get("pwf_images", []) or [])
                count += 1 if self.results.get("pwf_fused", None) is not None else 0
                count += len(self.results.get("sim_images_before_bg_correction", []) or [])
                count += len(self.results.get("sim_bg_estimates", []) or [])
                count += len(self.results.get("sim_images_after_bg_correction", []) or [])
                count += len(self.results.get("sim_images", []) or [])

                fusion_keys = [
                    "sim_fused_arithmetic_mean",
                    "sim_fused_before_final_structure_enhance",
                    "sim_wavelet_before_final_structure_enhance",
                ]
                count += count_optional_images(fusion_keys)

                structure_keys = [
                    "sim_fused_lowfreq_bg",
                    "sim_fused_highpass",
                    "sim_fused_before_smoothing",
                    "sim_wavelet_lowfreq_bg",
                    "sim_wavelet_highpass",
                    "sim_wavelet_before_smoothing",
                    "sim_fused_before_wiener",
                    "sim_wavelet_before_wiener",
                    "hilo_hybrid_before_hilo_denoise",
                ]
                count += count_optional_images(structure_keys)

                sim_diag_count = len(self.results.get("sim_images", []) or [])
                diag_level = max(1, int(self.results.get("wavelet_level", 2) or 2))
                count += sim_diag_count * (6 + diag_level * 8)

                final_output_keys = [
                    "pwf_fused",
                    "raw_widefield_fused",
                    "external_raw_widefield",
                    "sim_fused_mean",
                    "sim_wavelet_fused",
                    "hilo_hybrid",
                ]
                count += count_optional_images(final_output_keys)

                count += 1  # processing_steps/11_views/comparison_view.png
                if self.profile_window is not None and self.profile_window.isVisible():
                    count += 1
                count += 1  # README_steps.txt
                return count

            final_result_exports = [
                ("pwf_fused", "widefield_fused.tif"),
                ("raw_widefield_fused", "raw_widefield_fused.tif"),
                ("external_raw_widefield", "external_raw_widefield.tif"),
                ("sim_fused_mean", "final_sim_fused.tif"),
                ("sim_wavelet_fused", "sim_wavelet_fused.tif"),
                ("hilo_hybrid", "hilo_hybrid_os_sim_plus_hilo.tif"),
            ]

            intermediate_top_level_exports = [
                ("sim_fused_arithmetic_mean", "final_sim_fused_arithmetic_mean.tif"),
                ("sim_fused_before_final_structure_enhance", "final_sim_fused_before_structure_enhance.tif"),
                ("sim_fused_lowfreq_bg", "final_sim_lowfreq_background.tif"),
                ("sim_fused_highpass", "final_sim_highpass_structure.tif"),
                ("sim_wavelet_before_final_structure_enhance", "sim_wavelet_fused_before_structure_enhance.tif"),
                ("sim_wavelet_lowfreq_bg", "sim_wavelet_lowfreq_background.tif"),
                ("sim_wavelet_highpass", "sim_wavelet_highpass_structure.tif"),
                ("sim_fused_before_wiener", "final_sim_before_wiener_filter.tif"),
                ("sim_wavelet_before_wiener", "sim_wavelet_before_wiener_filter.tif"),
                ("hilo_mid_abs_component", "hilo_mid_component_abs.tif"),
                ("hilo_high_abs_component", "hilo_high_component_abs.tif"),
                ("hilo_mid_high_component", "hilo_weighted_mid_high_component.tif"),
                ("hilo_hybrid_before_hilo_denoise", "hilo_hybrid_before_hilo_denoise.tif"),
                ("hilo_hybrid_before_wiener", "hilo_hybrid_before_wiener_filter.tif"),
            ]

            top_level_exports = list(final_result_exports)
            if save_intermediate_files:
                top_level_exports.extend(intermediate_top_level_exports)

            top_level_count = 0
            if save_intermediate_files:
                top_level_count += sum(len(group) for group in processed_groups)
                top_level_count += len(pwf_images)
                top_level_count += len(sim_images)

            top_level_count += count_optional_images([key for key, _ in top_level_exports])
            top_level_count += 1  # comparison_view.png
            if self.profile_window is not None and self.profile_window.isVisible():
                top_level_count += 1

            total_steps = top_level_count
            if save_intermediate_files:
                total_steps += estimate_processing_steps_count()
            total_steps = max(1, total_steps)

            current_step = 0

            progress = QProgressDialog("Preparing save...", "Cancel", 0, total_steps, self)
            progress.setWindowTitle("Saving Results")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            def progress_step(message):
                nonlocal current_step
                current_step += 1
                if current_step > progress.maximum():
                    progress.setMaximum(current_step)
                progress.setLabelText(f"Saving step {current_step}/{progress.maximum()}\n{message}")
                progress.setValue(current_step)
                QApplication.processEvents()
                if progress.wasCanceled():
                    raise InterruptedError("Saving was canceled by user.")

            def save_named_image(filename, img, message=None):
                path = os.path.join(output_dir, filename)
                save_image_tif(path, img)
                progress_step(message or filename)
                return path

            self._save_progress_callback = progress_step

            if save_intermediate_files:
                self.log("Saving final results and intermediate processing files with progress dialog...")
            else:
                self.log("Saving final results only. Intermediate processing files will be skipped.")

            if save_intermediate_files:
                for idx, group in enumerate(processed_groups):
                    safe_label = safe_label_for(idx)
                    for phase_idx, img in enumerate(group):
                        suffix = phase_suffixes[phase_idx] if phase_idx < len(phase_suffixes) else f"i{phase_idx}"
                        filename = f"{safe_label}_{suffix}_preprocessed.tif"
                        save_named_image(filename, img, filename)

                for idx, img in enumerate(pwf_images):
                    safe_label = safe_label_for(idx)
                    filename = f"pwf_{safe_label}.tif"
                    save_named_image(filename, img, filename)

                for idx, img in enumerate(sim_images):
                    safe_label = safe_label_for(idx)
                    filename = f"sim_{safe_label}.tif"
                    save_named_image(filename, img, filename)

            for key, filename in top_level_exports:
                if self.results.get(key, None) is not None:
                    save_named_image(filename, self.results[key], filename)

            comparison_png_path = os.path.join(output_dir, "comparison_view.png")
            self.canvas.figure.savefig(
                comparison_png_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white"
            )
            progress_step("comparison_view.png")

            if self.profile_window is not None and self.profile_window.isVisible():
                profile_png_path = os.path.join(output_dir, "profile_comparison_view.png")
                self.profile_window.save_current_view(profile_png_path)
                self.log(f"Saved profile comparison view to: {profile_png_path}")
                progress_step("profile_comparison_view.png")
            else:
                profile_png_path = None

            steps_dir = None
            readme_path = None

            if save_intermediate_files:
                progress.setLabelText("Exporting processing step images...")
                QApplication.processEvents()
                steps_dir, readme_path = self.export_processing_steps(output_dir)
            else:
                self.log("Skipped processing_steps export because intermediate saving was not selected.")

            progress.setValue(progress.maximum())
            progress.setLabelText("Save complete.")
            QApplication.processEvents()
            progress.close()

            self.log(f"Saved results to: {output_dir}")
            self.log(f"Saved comparison view to: {comparison_png_path}")
            self._set_status_text("Saved results")
            self._update_status_bar()

            if save_intermediate_files:
                self.log(f"Saved processing step images to: {steps_dir}")
                self._show_saved_results_message(
                    "Saved",
                    f"Results saved to:\n{output_dir}\n\n"
                    f"Processing step images:\n{steps_dir}\n\n"
                    f"Step description:\n{readme_path}",
                    output_dir,
                )
            else:
                self._show_saved_results_message(
                    "Saved",
                    f"Final results saved to:\n{output_dir}\n\n"
                    f"Comparison view:\n{comparison_png_path}\n\n"
                    "Intermediate processing files were not saved.",
                    output_dir,
                )

        except InterruptedError as e:
            if progress is not None:
                progress.close()
            self.log(f"[CANCELED] {e}")
            QMessageBox.information(self, "Canceled", str(e))

        except Exception as e:
            if progress is not None:
                progress.close()
            self.log(f"[ERROR] {e}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

        finally:
            if hasattr(self, "_save_progress_callback"):
                self._save_progress_callback = None

def _qt_exception_hook(exc_type, exc_value, exc_traceback):
    """Log uncaught Qt/Python errors instead of failing silently after double-click launch."""
    details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log_path = _write_startup_error_log(details)
    message = str(exc_value) or exc_type.__name__
    if log_path:
        message += f"\n\nError log:\n{log_path}"
    try:
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(None, "SIM GUI error", message)
        else:
            print(details, file=sys.stderr)
    except Exception:
        try:
            print(details, file=sys.stderr)
        except Exception:
            pass


def main():
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        sys.excepthook = _qt_exception_hook
        win = SIMGui()
        win.show()
        return app.exec()
    except SystemExit:
        raise
    except Exception:
        details = traceback.format_exc()
        log_path = _write_startup_error_log(details)
        message = "The SIM GUI failed to start."
        if log_path:
            message += f"\n\nError log:\n{log_path}"
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "SIM GUI startup error", message)
        except Exception:
            try:
                print(message, file=sys.stderr)
                print(details, file=sys.stderr)
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

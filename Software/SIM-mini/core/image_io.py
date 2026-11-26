from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .constants import PROJECT_W, PROJECT_H
from .jpeg_turbo import (
    decode_jpeg_unchanged,
    decode_jpeg_to_bgr,
    encode_image_to_jpeg,
    is_jpeg_bytes,
    is_jpeg_path,
)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_jpeg_path(path):
        encoded = encode_image_to_jpeg(image, quality=95)
        if encoded is None:
            return False
        path.write_bytes(encoded)
        return True

    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False

    buf.tofile(str(path))
    return True


def imread_unicode(path: str) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()

    if is_jpeg_bytes(raw):
        image = decode_jpeg_unchanged(raw)
    else:
        data = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    return image


def load_local_image_as_bgr(path: str) -> np.ndarray:
    image = imread_unicode(path)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(np.min(x))
    mx = float(np.max(x))

    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / (mx - mn)).astype(np.float32)


def percentile_normalize(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.7) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        return normalize01(x)

    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def rgb_to_qimage(frame_rgb: np.ndarray) -> QImage:
    h, w, ch = frame_rgb.shape
    return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def gray_to_rgb_u8(img: np.ndarray) -> np.ndarray:
    g = (normalize01(img) * 255).clip(0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def mask_to_rgb_u8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    return np.dstack([m, m, m])


def add_panel_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = img_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def resize_mask_to_projector(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    resized = cv2.resize(
        mask_u8,
        (PROJECT_W, PROJECT_H),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def camera_frame_to_gray_u8(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr.astype(np.uint8)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def camera_frame_to_gray_640_float(frame_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        frame_bgr,
        (PROJECT_W, PROJECT_H),
        interpolation=cv2.INTER_AREA,
    )
    gray_u8 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return percentile_normalize(gray_u8.astype(np.float32))


def resize_keep_aspect_bgr(frame_bgr: np.ndarray, max_w: int, max_h: int, use_gpu: bool = False) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_w and h <= max_h:
        return frame_bgr

    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if use_gpu:
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame_bgr)
                gpu_resized = cv2.cuda.resize(
                    gpu_frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                return gpu_resized.download()
        except Exception:
            pass

    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_histogram_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return np.bincount(gray.reshape(-1), minlength=256).astype(np.float32)


def auto_contrast_levels(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [0.5, 99.5])
    lo = int(max(0, min(254, round(lo))))
    hi = int(max(lo + 1, min(255, round(hi))))
    return lo, hi


def apply_contrast_lut(frame_bgr: np.ndarray, black_level: int, white_level: int) -> np.ndarray:
    black_level = int(max(0, min(254, black_level)))
    white_level = int(max(black_level + 1, min(255, white_level)))

    lut = np.arange(256, dtype=np.float32)
    lut = (lut - black_level) * 255.0 / max(1, white_level - black_level)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return cv2.LUT(frame_bgr, lut)

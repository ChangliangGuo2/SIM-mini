from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg", ".jpe", ".jfif"})


class TurboJPEGUnavailableError(RuntimeError):
    """Raised when PyTurboJPEG or the native libjpeg-turbo runtime is unavailable."""


_thread_local = threading.local()
_binding_lock = threading.Lock()
_binding: Optional[Dict[str, Any]] = None


def _load_binding() -> Dict[str, Any]:
    """Import PyTurboJPEG lazily so non-image tools can still import the project."""
    global _binding

    if _binding is not None:
        return _binding

    with _binding_lock:
        if _binding is not None:
            return _binding

        try:
            import turbojpeg as tj
        except Exception as exc:
            raise TurboJPEGUnavailableError(
                "PyTurboJPEG is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        required = (
            "TurboJPEG",
            "TJPF_BGR",
            "TJPF_GRAY",
            "TJSAMP_420",
            "TJSAMP_GRAY",
        )
        missing = [name for name in required if not hasattr(tj, name)]
        if missing:
            raise TurboJPEGUnavailableError(
                "The installed `turbojpeg` module is not the PyTurboJPEG binding "
                f"required by this project. Missing symbols: {', '.join(missing)}. "
                "Install `PyTurboJPEG>=1.8.3,<2.0`."
            )

        _binding = {
            "TurboJPEG": tj.TurboJPEG,
            "TJPF_BGR": tj.TJPF_BGR,
            "TJPF_GRAY": tj.TJPF_GRAY,
            "TJSAMP_420": tj.TJSAMP_420,
            "TJSAMP_GRAY": tj.TJSAMP_GRAY,
            "TJCS_GRAY": getattr(tj, "TJCS_GRAY", None),
        }
        return _binding


def _native_library_candidates():
    """Yield explicit native-library locations before automatic discovery."""
    seen = set()

    def emit(path):
        if not path:
            return
        candidate = Path(path).expanduser()
        key = str(candidate).lower()
        if key in seen:
            return
        seen.add(key)
        yield candidate

    for env_name in ("TURBOJPEG_LIB", "LIBTURBOJPEG_PATH"):
        yield from emit(os.environ.get(env_name))

    project_root = Path(__file__).resolve().parents[1]
    bases = (
        project_root,
        project_root / "lib",
        project_root / "bin",
        project_root / "vendor",
        project_root / "vendor" / "libjpeg-turbo",
        project_root / "vendor" / "libjpeg-turbo" / "bin",
        Path.cwd(),
        Path(sys.executable).resolve().parent,
    )
    library_names = (
        "turbojpeg.dll",
        "libturbojpeg.so",
        "libturbojpeg.so.0",
        "libturbojpeg.dylib",
    )

    for base in bases:
        for name in library_names:
            yield from emit(base / name)

    if os.name == "nt":
        for path in (
            r"C:\libjpeg-turbo64\bin\turbojpeg.dll",
            r"C:\libjpeg-turbo\bin\turbojpeg.dll",
            r"C:\Program Files\libjpeg-turbo64\bin\turbojpeg.dll",
            r"C:\Program Files\libjpeg-turbo\bin\turbojpeg.dll",
        ):
            yield from emit(path)


def _create_codec():
    binding = _load_binding()
    codec_type = binding["TurboJPEG"]
    load_errors = []

    for candidate in _native_library_candidates():
        if not candidate.is_file():
            continue
        try:
            return codec_type(str(candidate)), binding, str(candidate)
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")

    try:
        return codec_type(), binding, "automatic library discovery"
    except Exception as exc:
        load_errors.append(f"automatic library discovery: {exc}")

    details = "\n".join(load_errors[-4:])
    raise TurboJPEGUnavailableError(
        "The native libjpeg-turbo library could not be loaded. "
        "Install libjpeg-turbo, or set TURBOJPEG_LIB to turbojpeg.dll/"
        "libturbojpeg.so/libturbojpeg.dylib."
        + (f"\n\nLoader details:\n{details}" if details else "")
    )


def _get_codec():
    state = getattr(_thread_local, "codec_state", None)
    if state is None:
        state = _create_codec()
        _thread_local.codec_state = state
    return state


def ensure_turbojpeg_available() -> str:
    """Initialize the codec for the current thread and return its library source."""
    _codec, _binding_info, source = _get_codec()
    return source


def is_jpeg_path(path) -> bool:
    return Path(path).suffix.lower() in JPEG_EXTENSIONS


def is_jpeg_bytes(data: bytes) -> bool:
    return bool(data) and len(data) >= 3 and data[:3] == b"\xff\xd8\xff"


def _reduction_scaling_factor(reduced: int) -> Optional[Tuple[int, int]]:
    reduction = int(reduced) if reduced in (1, 2, 4, 8) else 1
    return None if reduction == 1 else (1, reduction)


def _decode(
    jpeg_bytes: bytes,
    *,
    reduced: int,
    preserve_grayscale: bool,
) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None

    codec, binding, _source = _get_codec()
    pixel_format = binding["TJPF_BGR"]

    if preserve_grayscale:
        try:
            header = codec.decode_header(jpeg_bytes)
            jpeg_subsample = header[2] if len(header) > 2 else None
            jpeg_colorspace = header[3] if len(header) > 3 else None
            is_grayscale = (
                jpeg_subsample == binding["TJSAMP_GRAY"]
                or (
                    binding["TJCS_GRAY"] is not None
                    and jpeg_colorspace == binding["TJCS_GRAY"]
                )
            )
            if is_grayscale:
                pixel_format = binding["TJPF_GRAY"]
        except Exception:
            # A malformed header will fail in decode() below and return None.
            pass

    kwargs = {"pixel_format": pixel_format}
    scaling_factor = _reduction_scaling_factor(reduced)
    if scaling_factor is not None:
        kwargs["scaling_factor"] = scaling_factor

    try:
        frame = codec.decode(jpeg_bytes, **kwargs)
    except TurboJPEGUnavailableError:
        raise
    except Exception:
        return None

    if frame is None:
        return None

    frame = np.asarray(frame, dtype=np.uint8)

    if pixel_format == binding["TJPF_GRAY"]:
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        if frame.ndim != 2:
            return None
        return np.ascontiguousarray(frame)

    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)

    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    return np.ascontiguousarray(frame)


def decode_jpeg_to_bgr(jpeg_bytes: bytes, reduced: int = 1) -> Optional[np.ndarray]:
    """Decode an in-memory JPEG to a contiguous uint8 BGR image.

    ``reduced`` preserves the previous OpenCV behavior for JPEG-native 1/2,
    1/4, and 1/8 IDCT-scaled preview decoding. Invalid JPEG data returns None.
    """
    frame = _decode(
        jpeg_bytes,
        reduced=reduced,
        preserve_grayscale=False,
    )
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    return np.ascontiguousarray(frame)


def decode_jpeg_unchanged(jpeg_bytes: bytes, reduced: int = 1) -> Optional[np.ndarray]:
    """Decode JPEG while retaining OpenCV IMREAD_UNCHANGED grayscale semantics.

    Grayscale JPEG files produce a 2-D uint8 array. Other JPEG files produce a
    contiguous 3-channel BGR uint8 array. JPEG has no alpha channel.
    """
    return _decode(
        jpeg_bytes,
        reduced=reduced,
        preserve_grayscale=True,
    )


def _to_jpeg_u8(image: np.ndarray) -> Optional[np.ndarray]:
    frame = np.asarray(image)
    if frame.size == 0:
        return None

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] == 4:
        # JPEG has no alpha channel. OpenCV likewise discards alpha on JPEG save.
        frame = frame[:, :, :3]

    if frame.ndim == 2:
        return np.ascontiguousarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    return None


def encode_image_to_jpeg(image: np.ndarray, quality: int = 95) -> Optional[bytes]:
    """Encode a grayscale or BGR/BGRA image to JPEG using libjpeg-turbo.

    Color images use 4:2:0 chroma subsampling, matching the common OpenCV JPEG
    default. Quality defaults to 95, also matching OpenCV's JPEG write default.
    """
    frame = _to_jpeg_u8(image)
    if frame is None:
        return None

    codec, binding, _source = _get_codec()
    quality = max(1, min(100, int(quality)))

    if frame.ndim == 2:
        pixel_format = binding["TJPF_GRAY"]
        jpeg_subsample = binding["TJSAMP_GRAY"]
    else:
        pixel_format = binding["TJPF_BGR"]
        jpeg_subsample = binding["TJSAMP_420"]

    try:
        encoded = codec.encode(
            frame,
            quality=quality,
            pixel_format=pixel_format,
            jpeg_subsample=jpeg_subsample,
        )
    except TurboJPEGUnavailableError:
        raise
    except Exception:
        return None

    if encoded is None:
        return None
    return bytes(encoded)


def encode_bgr_to_jpeg(image_bgr: np.ndarray, quality: int = 95) -> Optional[bytes]:
    """Compatibility wrapper for camera code that always supplies BGR images."""
    return encode_image_to_jpeg(image_bgr, quality=quality)

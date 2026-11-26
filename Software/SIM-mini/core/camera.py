import re
import sys
import shutil
import subprocess
import threading
import time
import json
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tifffile as tiff
from PySide6.QtCore import QThread, Signal

from .constants import (
    DEFAULT_CAMERA_DEVICE_NAME,
    DEFAULT_WINDOWS_CAMERA_DEVICE_NAME,
    DEFAULT_FFMPEG_EXE,
    DEFAULT_PREVIEW_MAX_H,
    DEFAULT_PREVIEW_MAX_W,
)
from .jpeg_turbo import (
    TurboJPEGUnavailableError,
    decode_jpeg_to_bgr,
    encode_bgr_to_jpeg,
    ensure_turbojpeg_available,
)


def resolve_ffmpeg_exe(ffmpeg_exe: str = DEFAULT_FFMPEG_EXE) -> str:
    ffmpeg_exe = str(ffmpeg_exe or "").strip()

    if ffmpeg_exe:
        if Path(ffmpeg_exe).exists():
            return ffmpeg_exe

        found = shutil.which(ffmpeg_exe)
        if found:
            return found

    common = (
        r"C:\Software\ffmpeg-8.1.1-full_build"
        r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
    )
    if Path(common).exists():
        return common

    return "ffmpeg"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def camera_driver_name() -> str:
    return "FFmpeg V4L2" if is_linux() else "FFmpeg DirectShow"


def default_camera_device_name() -> str:
    return DEFAULT_CAMERA_DEVICE_NAME if is_linux() else DEFAULT_WINDOWS_CAMERA_DEVICE_NAME


def _run_v4l2_control(device_name: str, *args: str):
    exe = shutil.which("v4l2-ctl")
    if not exe:
        raise FileNotFoundError("v4l2-ctl was not found in PATH")
    return subprocess.run(
        [exe, "--device", str(device_name), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=8,
    )


def _apply_linux_fixed_camera_controls(device_name: str, exposure: float, gain: float):
    report = {
        "backend": "v4l2-ctl",
        "device_name": str(device_name),
        "requested": {
            "auto_exposure": False,
            "exposure": float(exposure),
            "auto_gain": False,
            "gain": float(gain),
        },
        "control_names": {},
        "set_results": {},
        "readback": {},
        "verified": False,
        "warnings": [],
        "errors": [],
    }

    try:
        listed = _run_v4l2_control(device_name, "--list-ctrls")
    except Exception as exc:
        report["errors"].append(str(exc))
        return report
    if listed.returncode != 0:
        report["errors"].append((listed.stderr or listed.stdout or "v4l2-ctl failed").strip())
        return report

    controls = set(
        re.findall(r"^\s*([A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+", listed.stdout or "", re.MULTILINE)
    )

    def choose(*names):
        return next((name for name in names if name in controls), None)

    selected = {
        "auto_exposure": choose("exposure_auto", "auto_exposure"),
        "exposure": choose("exposure_time_absolute", "exposure_absolute", "exposure"),
        "auto_gain": choose("gain_automatic", "autogain"),
        "gain": choose("gain"),
    }
    report["control_names"] = selected

    required_missing = [key for key in ("auto_exposure", "exposure", "gain") if not selected[key]]
    if required_missing:
        report["errors"].append(
            "Required V4L2 controls are unavailable: " + ", ".join(required_missing)
        )
        return report

    requested_values = {
        "auto_exposure": 1,
        "exposure": int(round(float(exposure))),
        "auto_gain": 0,
        "gain": int(round(float(gain))),
    }
    for key in ("auto_exposure", "auto_gain", "exposure", "gain"):
        control_name = selected[key]
        if not control_name:
            continue
        proc = _run_v4l2_control(
            device_name,
            "--set-ctrl",
            f"{control_name}={requested_values[key]}",
        )
        report["set_results"][key] = proc.returncode == 0
        if proc.returncode != 0:
            report["errors"].append(
                f"Failed to set {control_name}: "
                + (proc.stderr or proc.stdout or "unknown error").strip()
            )

    for key, control_name in selected.items():
        if not control_name:
            continue
        proc = _run_v4l2_control(device_name, "--get-ctrl", control_name)
        match = re.search(r":\s*(-?\d+(?:\.\d+)?)", proc.stdout or "")
        if proc.returncode == 0 and match:
            report["readback"][key] = float(match.group(1))
        else:
            report["errors"].append(f"Unable to read back {control_name}")

    readback = report["readback"]
    checks = [
        readback.get("auto_exposure") == 1,
        readback.get("exposure") == requested_values["exposure"],
        readback.get("gain") == requested_values["gain"],
    ]
    if selected["auto_gain"]:
        checks.append(readback.get("auto_gain") == 0)
    report["verified"] = bool(all(checks) and not report["errors"])
    return report


def _apply_windows_fixed_camera_controls(
    device_name: str,
    device_index: int,
    exposure: float,
    gain: float,
):
    report = {
        "backend": "OpenCV DirectShow",
        "device_name": str(device_name),
        "device_index": int(device_index),
        "requested": {
            "auto_exposure": False,
            "exposure": float(exposure),
            "auto_gain": False,
            "gain": float(gain),
        },
        "set_results": {},
        "readback": {},
        "verified": False,
        "warnings": [],
        "errors": [],
    }

    try:
        capture = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)
    except Exception as exc:
        report["errors"].append(f"Unable to initialize OpenCV DirectShow: {exc}")
        return report
    try:
        if not capture.isOpened():
            report["errors"].append(
                f"OpenCV could not open DirectShow camera index {int(device_index)}"
            )
            return report

        # DirectShow represents manual/auto exposure as 0.25/0.75.
        report["set_results"]["auto_exposure"] = bool(
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        )
        report["set_results"]["exposure"] = bool(
            capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        )
        report["set_results"]["gain"] = bool(
            capture.set(cv2.CAP_PROP_GAIN, float(gain))
        )
        time.sleep(0.15)

        readback = {
            "auto_exposure": float(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "exposure": float(capture.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(capture.get(cv2.CAP_PROP_GAIN)),
        }
        report["readback"] = {
            key: value if np.isfinite(value) else None
            for key, value in readback.items()
        }
    finally:
        capture.release()

    readback = report["readback"]
    auto_exposure = readback.get("auto_exposure")
    actual_exposure = readback.get("exposure")
    actual_gain = readback.get("gain")
    manual_exposure_readback = (
        auto_exposure is not None and abs(auto_exposure - 0.25) <= 0.26
    )
    manual_exposure_set_accepted = bool(
        report["set_results"].get("auto_exposure")
    )
    auto_exposure_readback_unsupported = auto_exposure == -1.0
    manual_exposure = manual_exposure_readback or (
        manual_exposure_set_accepted and auto_exposure_readback_unsupported
    )
    exposure_matches = (
        actual_exposure is not None
        and abs(actual_exposure - float(exposure)) <= 0.51
    )
    gain_matches = actual_gain is not None and abs(actual_gain - float(gain)) <= 1.01
    set_ok = all(report["set_results"].values())
    report["verified"] = bool(manual_exposure and exposure_matches and gain_matches and set_ok)
    if auto_exposure_readback_unsupported and manual_exposure_set_accepted:
        report["warnings"].append(
            "The driver accepted manual exposure mode but reports -1 when that mode is read back"
        )
        report["auto_exposure_verification"] = "set_accepted_readback_unsupported"
    elif manual_exposure_readback:
        report["auto_exposure_verification"] = "readback_confirmed"
    else:
        report["auto_exposure_verification"] = "failed"
    if not report["verified"]:
        report["errors"].append(
            "The DirectShow driver did not confirm manual exposure and the requested exposure/gain values"
        )
    return report


def apply_fixed_camera_controls(
    device_name: str,
    exposure: float,
    gain: float,
    device_index: int = 0,
):
    """Apply and read back fixed exposure/gain before FFmpeg opens the camera."""
    try:
        if is_linux():
            return _apply_linux_fixed_camera_controls(device_name, exposure, gain)
        return _apply_windows_fixed_camera_controls(
            device_name,
            device_index,
            exposure,
            gain,
        )
    except Exception as exc:
        return {
            "backend": "v4l2-ctl" if is_linux() else "OpenCV DirectShow",
            "device_name": str(device_name),
            "device_index": None if is_linux() else int(device_index),
            "requested": {
                "auto_exposure": False,
                "exposure": float(exposure),
                "auto_gain": False,
                "gain": float(gain),
            },
            "readback": {},
            "verified": False,
            "warnings": [],
            "errors": [str(exc)],
        }


def enumerate_dshow_video_devices(ffmpeg_exe: str = DEFAULT_FFMPEG_EXE):
    exe = resolve_ffmpeg_exe(ffmpeg_exe)

    try:
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
    except Exception:
        return []

    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    devices = []
    seen = set()
    section = None

    for line in text.splitlines():
        lower = line.lower()
        if "directshow video devices" in lower:
            section = "video"
            continue
        if "directshow audio devices" in lower:
            section = "audio"
            continue

        # Current FFmpeg builds list video names under the video section
        # without appending "(video)". Keep the old marker as a fallback for
        # builds that do not print section headers.
        is_video_line = section == "video" or "(video)" in lower
        if not is_video_line or "alternative name" in lower:
            continue

        first = line.find('"')
        last = line.rfind('"')
        if first >= 0 and last > first:
            name = line[first + 1:last].strip()
            if name and name not in seen:
                seen.add(name)
                devices.append(name)

    return devices


def enumerate_v4l2_video_devices():
    devices = []
    seen = set()
    for dev in sorted(Path("/dev").glob("video*")):
        path = str(dev)
        try:
            if not dev.exists():
                continue
        except Exception:
            continue
        if path not in seen:
            seen.add(path)
            devices.append(path)
    return devices


def enumerate_video_devices(ffmpeg_exe: str = DEFAULT_FFMPEG_EXE):
    if is_linux():
        return enumerate_v4l2_video_devices()
    return enumerate_dshow_video_devices(ffmpeg_exe)


def select_default_dshow_device(devices):
    if not devices:
        return DEFAULT_WINDOWS_CAMERA_DEVICE_NAME

    for name in devices:
        if name == DEFAULT_WINDOWS_CAMERA_DEVICE_NAME:
            return name

    for name in devices:
        if "usb camera" in name.lower():
            return name

    for name in devices:
        lower = name.lower()
        if "xiaomi" not in lower and "integrated" not in lower and "webcam" not in lower:
            return name

    return devices[0]


def select_default_video_device(devices):
    if is_linux():
        return devices[0] if devices else DEFAULT_CAMERA_DEVICE_NAME
    return select_default_dshow_device(devices)


CAMERA_STREAM_FORMAT_MJPEG = "mjpeg"
CAMERA_STREAM_FORMAT_YUYV422 = "yuyv422"
CAMERA_STREAM_FORMATS = (CAMERA_STREAM_FORMAT_MJPEG, CAMERA_STREAM_FORMAT_YUYV422)


def normalize_camera_stream_format(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if text in {"yuyv422", "yuy2", "yuyv", "raw", "rawvideo"}:
        return CAMERA_STREAM_FORMAT_YUYV422
    return CAMERA_STREAM_FORMAT_MJPEG


def camera_stream_format_label(value: str) -> str:
    fmt = normalize_camera_stream_format(value)
    if fmt == CAMERA_STREAM_FORMAT_YUYV422:
        return "YUYV422"
    return "MJPEG"


def _parse_dshow_line_format(line: str):
    lower = line.lower()
    if "pixel_format=yuyv422" in lower or "pixel_format=yuy2" in lower or "yuyv422" in lower:
        return CAMERA_STREAM_FORMAT_YUYV422
    if "vcodec=mjpeg" in lower or "vcodec=mjpg" in lower or "mjpeg" in lower or "mjpg" in lower:
        return CAMERA_STREAM_FORMAT_MJPEG
    return None


def _parse_dshow_line_max_fps(line: str, default: int = 30) -> int:
    # DirectShow commonly prints ranges, for example:
    #   min s=2592x1944 fps=5 max s=2592x1944 fps=30
    # The UI should expose the usable maximum FPS for the selected mode, not
    # the minimum value that appears first in the line.
    values = re.findall(
        r"(?:fps|framerate)\s*=\s*(\d+(?:\.\d+)?)(?:/\d+)?",
        line,
        flags=re.IGNORECASE,
    )
    if not values:
        # Fallback for text like "30 fps". Avoid matching the height in
        # "1920x1080 fps=30" by requiring that the number is not preceded by
        # x/digit and that "fps" is not followed by "=".
        values = re.findall(
            r"(?<![x\d])(\d+(?:\.\d+)?)\s*fps\b(?!\s*=)",
            line,
            flags=re.IGNORECASE,
        )

    fps_values = []
    for value in values:
        try:
            fps_values.append(int(round(float(value))))
        except Exception:
            pass

    if not fps_values:
        return int(default)

    return max(1, max(fps_values))


def query_dshow_device_modes(ffmpeg_exe: str, device_name: str):
    exe = resolve_ffmpeg_exe(ffmpeg_exe)
    try:
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "info",
                "-f",
                "dshow",
                "-list_options",
                "true",
                "-i",
                f"video={device_name}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return []

    output = (proc.stderr or "") + "\n" + (proc.stdout or "")
    mode_by_key = {}

    for line in output.splitlines():
        stream_format = _parse_dshow_line_format(line)
        if stream_format is None:
            continue
        if "video=" not in line and "size=" not in line and "x" not in line:
            continue

        res_match = re.search(r"(\d{3,4})x(\d{3,4})", line)
        if not res_match:
            continue

        width = int(res_match.group(1))
        height = int(res_match.group(2))
        fps = _parse_dshow_line_max_fps(line, default=30)

        key = (stream_format, width, height)
        mode_by_key[key] = max(int(fps), int(mode_by_key.get(key, 0)))

    modes = [
        {"stream_format": fmt, "format": fmt, "width": w, "height": h, "fps": f}
        for (fmt, w, h), f in mode_by_key.items()
    ]
    modes.sort(
        key=lambda item: (
            0 if item["stream_format"] == CAMERA_STREAM_FORMAT_MJPEG else 1,
            item["width"] * item["height"],
            item["width"],
            item["height"],
            item["fps"],
        ),
        reverse=True,
    )
    return modes


def _parse_v4l2_fps_values(text: str):
    fps_values = []
    for fps_match in re.finditer(r"(\d+(?:\.\d+)?)\s*fps", text, flags=re.IGNORECASE):
        try:
            fps_values.append(int(round(float(fps_match.group(1)))))
        except Exception:
            pass

    for interval_match in re.finditer(r"(\d+(?:\.\d+)?)\s*s\b", text, flags=re.IGNORECASE):
        try:
            seconds = float(interval_match.group(1))
            if seconds > 0:
                fps_values.append(int(round(1.0 / seconds)))
        except Exception:
            pass

    return [fps for fps in fps_values if fps > 0]


def _parse_v4l2_ctl_formats(text: str):
    mode_by_key = {}
    current_format = None
    current_size = None

    for line in text.splitlines():
        upper = line.upper()
        fmt_match = re.search(r"\[\d+\]:\s*'([^']+)'", line)
        if fmt_match:
            fourcc = fmt_match.group(1).upper()
            if fourcc in {"MJPG", "MJPEG"}:
                current_format = CAMERA_STREAM_FORMAT_MJPEG
            elif fourcc in {"YUYV", "YUY2"}:
                current_format = CAMERA_STREAM_FORMAT_YUYV422
            else:
                current_format = None
            current_size = None
            continue

        if "MJPEG" in upper or "MJPG" in upper:
            current_format = CAMERA_STREAM_FORMAT_MJPEG
        elif "YUYV" in upper or "YUY2" in upper:
            current_format = CAMERA_STREAM_FORMAT_YUYV422

        size_match = re.search(r"(\d{3,5})x(\d{3,5})", line)
        if size_match and current_format is not None:
            current_size = (int(size_match.group(1)), int(size_match.group(2)))
            key = (current_format, current_size[0], current_size[1])
            mode_by_key.setdefault(key, 0)

        fps_values = _parse_v4l2_fps_values(line)
        if fps_values and current_format is not None and current_size is not None:
            key = (current_format, current_size[0], current_size[1])
            mode_by_key[key] = max(max(fps_values), int(mode_by_key.get(key, 0)))

    return [
        {"stream_format": fmt, "format": fmt, "width": w, "height": h, "fps": f or 30}
        for (fmt, w, h), f in mode_by_key.items()
    ]


def _sort_camera_modes(modes):
    modes.sort(
        key=lambda item: (
            0 if item["stream_format"] == CAMERA_STREAM_FORMAT_MJPEG else 1,
            item["width"] * item["height"],
            item["width"],
            item["height"],
            item["fps"],
        ),
        reverse=True,
    )
    return modes


def query_v4l2_device_modes(ffmpeg_exe: str, device_name: str):
    v4l2_ctl = shutil.which("v4l2-ctl")
    if v4l2_ctl:
        try:
            proc = subprocess.run(
                [v4l2_ctl, "-d", device_name, "--list-formats-ext"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
            )
            modes = _parse_v4l2_ctl_formats((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if modes:
                return _sort_camera_modes(modes)
        except Exception:
            pass

    exe = resolve_ffmpeg_exe(ffmpeg_exe)
    try:
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "info",
                "-f",
                "v4l2",
                "-list_formats",
                "all",
                "-i",
                device_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return []

    output = (proc.stderr or "") + "\n" + (proc.stdout or "")
    mode_by_key = {}
    current_format = None
    current_size = None

    for line in output.splitlines():
        lower = line.lower()
        parsed_format = _parse_dshow_line_format(line)
        if parsed_format is not None:
            current_format = parsed_format
            current_size = None
            continue

        if "compressed" in lower and ("mjpeg" in lower or "mjpg" in lower):
            current_format = CAMERA_STREAM_FORMAT_MJPEG
            current_size = None
            continue

        if "raw" in lower and ("yuyv" in lower or "yuy2" in lower):
            current_format = CAMERA_STREAM_FORMAT_YUYV422
            current_size = None
            continue

        size_match = re.search(r"(\d{3,5})x(\d{3,5})", line)
        if size_match and current_format is not None:
            current_size = (int(size_match.group(1)), int(size_match.group(2)))
            fps_values = _parse_v4l2_fps_values(line)
            if fps_values:
                key = (current_format, current_size[0], current_size[1])
                mode_by_key[key] = max(max(fps_values), int(mode_by_key.get(key, 0)))
            continue

        fps_values = _parse_v4l2_fps_values(line)
        if fps_values and current_format is not None and current_size is not None:
            key = (current_format, current_size[0], current_size[1])
            mode_by_key[key] = max(max(fps_values), int(mode_by_key.get(key, 0)))

    modes = [
        {"stream_format": fmt, "format": fmt, "width": w, "height": h, "fps": f or 30}
        for (fmt, w, h), f in mode_by_key.items()
    ]
    return _sort_camera_modes(modes)


def query_video_device_modes(ffmpeg_exe: str, device_name: str):
    if is_linux():
        return query_v4l2_device_modes(ffmpeg_exe, device_name)
    return query_dshow_device_modes(ffmpeg_exe, device_name)


def opencv_cuda_device_count() -> int:
    try:
        if not hasattr(cv2, "cuda"):
            return 0
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def opencv_cuda_available() -> bool:
    return opencv_cuda_device_count() > 0


class FFmpegCameraThread(QThread):
    preview_ready = Signal(object, int, float, object)
    fps_ready = Signal(float)
    settings_ready = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        ffmpeg_exe: str,
        device_name: str,
        width: int,
        height: int,
        fps: int,
        stream_format: str = CAMERA_STREAM_FORMAT_MJPEG,
        preview_fps: int = 0,
        preview_max_w: int = DEFAULT_PREVIEW_MAX_W,
        preview_max_h: int = DEFAULT_PREVIEW_MAX_H,
        use_gpu_preview: bool = True,
        roi=None,
        parent=None,
    ):
        super().__init__(parent)

        self.ffmpeg_exe = ffmpeg_exe
        self.device_name = device_name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.stream_format = normalize_camera_stream_format(stream_format)
        self.preview_fps = max(0, int(preview_fps))
        self.preview_max_w = int(preview_max_w)
        self.preview_max_h = int(preview_max_h)
        self.use_gpu_preview = bool(use_gpu_preview and opencv_cuda_available())

        self._running = False
        self._process = None
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_bgr = None
        self._latest_seq = 0
        self._preview_pending = False
        self._preview_skip_pending_total = 0
        self.roi = None
        if roi is not None:
            self.set_roi(roi)

    def _camera_input_args(self):
        if is_linux():
            fmt = "mjpeg" if self.stream_format == CAMERA_STREAM_FORMAT_MJPEG else "yuyv422"
            return [
                "-f",
                "v4l2",
                "-input_format",
                fmt,
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                self.device_name,
            ]

        return [
            "-f",
            "dshow",
            "-rtbufsize",
            "1024M",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
        ]

    def latest_sequence(self) -> int:
        with self._lock:
            return int(self._latest_seq)

    def get_latest_jpeg(self):
        with self._lock:
            seq = int(self._latest_seq)
            jpeg = bytes(self._latest_jpeg) if self._latest_jpeg is not None else None
            frame = self._latest_bgr.copy() if self._latest_bgr is not None else None
            roi = self.roi

        if jpeg is not None:
            if roi is None:
                return jpeg, seq

            frame_from_jpeg = decode_jpeg_to_bgr(jpeg)
            if frame_from_jpeg is None:
                return jpeg, seq

            frame_from_jpeg = self._crop_frame(frame_from_jpeg, roi)
            encoded = encode_bgr_to_jpeg(frame_from_jpeg, quality=95)
            if encoded is None:
                return jpeg, seq
            return encoded, seq

        if frame is None:
            return None, seq

        if roi is not None:
            frame = self._crop_frame(frame, roi)

        encoded = encode_bgr_to_jpeg(frame, quality=95)
        if encoded is None:
            return None, seq

        return encoded, seq

    def get_latest_full_bgr(self):
        with self._lock:
            seq = int(self._latest_seq)
            frame = self._latest_bgr.copy() if self._latest_bgr is not None else None
            jpeg = bytes(self._latest_jpeg) if self._latest_jpeg is not None else None
            roi = self.roi

        if frame is None and jpeg is not None:
            frame = decode_jpeg_to_bgr(jpeg)

        if frame is not None and roi is not None:
            frame = self._crop_frame(frame, roi)

        return frame, seq

    def _preview_decode_reduction(self) -> int:
        if self.stream_format != CAMERA_STREAM_FORMAT_MJPEG:
            return 1

        scale = max(
            self.width / max(1, self.preview_max_w),
            self.height / max(1, self.preview_max_h),
        )

        if scale >= 8:
            return 8
        if scale >= 4:
            return 4
        if scale >= 2:
            return 2
        return 1

    def _build_ffmpeg_command(self):
        exe = resolve_ffmpeg_exe(self.ffmpeg_exe)
        common = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-max_delay",
            "0",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-thread_queue_size",
            "1",
        ] + self._camera_input_args()

        if self.stream_format == CAMERA_STREAM_FORMAT_YUYV422:
            if not is_linux():
                common = common + [
                    "-pixel_format",
                    "yuyv422",
                    "-i",
                    f"video={self.device_name}",
                ]
            return common + [
                "-an",
                "-pix_fmt",
                "bgr24",
                "-vcodec",
                "rawvideo",
                "-f",
                "rawvideo",
                "pipe:1",
            ]

        if not is_linux():
            common = common + ["-vcodec", "mjpeg", "-i", f"video={self.device_name}"]

        return common + ["-an", "-c:v", "copy", "-f", "image2pipe", "pipe:1"]

    def run(self):
        try:
            turbojpeg_source = ensure_turbojpeg_available()
        except TurboJPEGUnavailableError as exc:
            self.error.emit(
                "Failed to initialize the libjpeg-turbo JPEG codec.\n\n"
                f"{exc}"
            )
            return

        cmd = self._build_ffmpeg_command()

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            self.error.emit(
                f"Failed to start {camera_driver_name()} camera.\n\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"Error: {exc}"
            )
            return

        self._running = True
        format_label = camera_stream_format_label(self.stream_format)
        if self.stream_format == CAMERA_STREAM_FORMAT_YUYV422:
            pipeline_text = (
                "YUYV422 input -> raw BGR preview; "
                f"on-demand JPEG encode via libjpeg-turbo ({turbojpeg_source})"
            )
        else:
            pipeline_text = (
                "MJPEG copy stream -> libjpeg-turbo BGR decode "
                f"({turbojpeg_source})"
            )

        self.settings_ready.emit(
            f"{camera_driver_name()} active: "
            f"{self.device_name}, {pipeline_text}, "
            f"{self.width} x {self.height} @ {self.fps} FPS. "
            f"Input format: {format_label}. "
            f"Preview FPS: {self.preview_fps}. "
            "Preview resize is disabled in the camera thread. "
            "Preview frames are prepared only after the UI consumes the previous preview."
        )

        if self.stream_format == CAMERA_STREAM_FORMAT_YUYV422:
            self._run_raw_bgr_loop()
        else:
            self._run_mjpeg_loop()

        self._cleanup_process()

    def _should_emit_preview(self, now: float, last_preview_emit: float, preview_interval: float) -> bool:
        return preview_interval == 0.0 or now - last_preview_emit >= preview_interval

    def _emit_fps_if_needed(self, timestamps, now: float, last_fps_emit: float):
        if len(timestamps) >= 2 and now - last_fps_emit > 0.5:
            stream_fps = (len(timestamps) - 1) / max(timestamps[-1] - timestamps[0], 1e-6)
            self.fps_ready.emit(float(stream_fps))
            return now
        return last_fps_emit

    def _prepare_and_emit_preview(self, frame, seq: int, now: float, read_ms: float, timing_extra=None):
        with self._lock:
            if getattr(self, "_preview_pending", False):
                self._preview_skip_pending_total += 1
                return False
            self._preview_pending = True

        preview_prepare_start = time.perf_counter()
        timing = {
            "seq": int(seq),
            "read_ms": float(read_ms),
            "preview_skip_pending_total": int(self._preview_skip_pending_total),
        }
        if isinstance(timing_extra, dict):
            timing.update(timing_extra)

        if frame is None:
            with self._lock:
                self._preview_pending = False
            return False

        timing["source_w"] = int(frame.shape[1])
        timing["source_h"] = int(frame.shape[0])

        t_crop = time.perf_counter()
        with self._lock:
            roi = self.roi
        if roi is not None:
            frame = self._crop_frame_for_decoded_size(frame, roi)
        timing["crop_ms"] = (time.perf_counter() - t_crop) * 1000.0

        # No CPU/OpenCV preview resize here. QLabel/Qt handles visual scaling.
        t_resize = time.perf_counter()
        preview = frame
        timing["resize_ms"] = 0.0
        timing["resize_check_ms"] = (time.perf_counter() - t_resize) * 1000.0
        timing["preview_w"] = int(preview.shape[1])
        timing["preview_h"] = int(preview.shape[0])
        timing["camera_prepare_ms"] = (time.perf_counter() - preview_prepare_start) * 1000.0

        self.preview_ready.emit(preview, seq, now, timing)
        return True

    def _run_mjpeg_loop(self):
        buffer = bytearray()
        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"
        timestamps = deque(maxlen=180)
        last_fps_emit = 0.0
        last_preview_emit = 0.0
        preview_interval = 0.0 if self.preview_fps == 0 else 1.0 / self.preview_fps
        self._preview_pending = False

        while self._running:
            if self._process is None or self._process.stdout is None:
                break

            read_start = time.perf_counter()
            chunk = self._process.stdout.read(131072)
            if not chunk:
                break

            buffer.extend(chunk)

            while self._running:
                start = buffer.find(soi)
                if start < 0:
                    if len(buffer) > 2:
                        del buffer[:-2]
                    break

                end = buffer.find(eoi, start + 2)
                if end < 0:
                    if start > 0:
                        del buffer[:start]
                    break

                jpeg = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                now = time.perf_counter()

                with self._lock:
                    self._latest_jpeg = jpeg
                    self._latest_bgr = None
                    self._latest_seq += 1
                    seq = self._latest_seq

                timestamps.append(now)
                last_fps_emit = self._emit_fps_if_needed(timestamps, now, last_fps_emit)

                if not self._should_emit_preview(now, last_preview_emit, preview_interval):
                    continue

                t_decode = time.perf_counter()
                reduced = self._preview_decode_reduction()
                frame = decode_jpeg_to_bgr(jpeg, reduced=reduced)
                decode_ms = (time.perf_counter() - t_decode) * 1000.0

                emitted = self._prepare_and_emit_preview(
                    frame,
                    seq,
                    now,
                    (now - read_start) * 1000.0,
                    {
                        "stream_format": self.stream_format,
                        "jpeg_kb": len(jpeg) / 1024.0,
                        "decode_reduced": int(reduced),
                        "decode_ms": decode_ms,
                    },
                )
                if emitted:
                    last_preview_emit = now

    def _run_raw_bgr_loop(self):
        frame_size = max(1, int(self.width) * int(self.height) * 3)
        buffer = bytearray()
        timestamps = deque(maxlen=180)
        last_fps_emit = 0.0
        last_preview_emit = 0.0
        preview_interval = 0.0 if self.preview_fps == 0 else 1.0 / self.preview_fps
        self._preview_pending = False

        while self._running:
            if self._process is None or self._process.stdout is None:
                break

            read_start = time.perf_counter()
            chunk = self._process.stdout.read(min(262144, frame_size))
            if not chunk:
                break

            buffer.extend(chunk)

            while self._running and len(buffer) >= frame_size:
                raw = bytes(buffer[:frame_size])
                del buffer[:frame_size]
                now = time.perf_counter()

                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                except Exception:
                    continue

                with self._lock:
                    self._latest_jpeg = None
                    self._latest_bgr = frame.copy()
                    self._latest_seq += 1
                    seq = self._latest_seq

                timestamps.append(now)
                last_fps_emit = self._emit_fps_if_needed(timestamps, now, last_fps_emit)

                if not self._should_emit_preview(now, last_preview_emit, preview_interval):
                    continue

                emitted = self._prepare_and_emit_preview(
                    frame,
                    seq,
                    now,
                    (now - read_start) * 1000.0,
                    {
                        "stream_format": self.stream_format,
                        "raw_kb": len(raw) / 1024.0,
                        "decode_reduced": 1,
                        "decode_ms": 0.0,
                    },
                )
                if emitted:
                    last_preview_emit = now

    def set_roi(self, roi):
        if roi is None:
            with self._lock:
                self.roi = None
            return

        x, y, w, h = roi
        roi_tuple = (int(max(0, x)), int(max(0, y)), int(max(1, w)), int(max(1, h)))
        with self._lock:
            self.roi = roi_tuple

    def _crop_frame(self, frame, roi):
        if roi is None or frame is None:
            return frame

        x, y, w, h = [int(v) for v in roi]
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return frame[y : y + h, x : x + w]

    def _crop_frame_for_decoded_size(self, frame, roi):
        if roi is None or frame is None:
            return frame

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return frame

        scale_x = frame_w / max(1, int(self.width))
        scale_y = frame_h / max(1, int(self.height))

        x, y, w, h = [int(v) for v in roi]
        scaled_roi = (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )
        return self._crop_frame(frame, scaled_roi)

    def _cleanup_process(self):
        proc = self._process
        self._process = None

        if proc is None:
            return

        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self._cleanup_process()
        self.wait(1500)

    def preview_ack(self):
        """Called by the UI/main thread to acknowledge that the last
        emitted preview has been consumed and it's safe to emit the next one."""
        with self._lock:
            self._preview_pending = False


# Backward-compatible alias for older imports/config notes.
FFmpegDirectShowCameraThread = FFmpegCameraThread


# =============================================================================
# 16-bit grayscale TIFF acquisition
# =============================================================================

def bgr_to_gray16(frame_bgr, use_gpu: bool = False):
    """Convert a BGR frame to 16-bit grayscale.

    The USB camera stream is 8-bit MJPEG or YUYV422. For scientific TIFF
    compatibility, the 8-bit gray values are scaled to 16-bit by value * 257.

    GPU acceleration is optional and only used when OpenCV CUDA is available.
    """
    if frame_bgr is None:
        return None

    if use_gpu and opencv_cuda_available():
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame_bgr)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gray8 = gpu_gray.download()
            return gray8.astype(np.uint16) * 257
        except Exception:
            # Robust fallback: acquisition must keep working even if CUDA is unavailable
            # or the installed OpenCV build lacks a specific CUDA function.
            pass

    gray8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray8.astype(np.uint16) * 257


def write_gray16_tiff_uncompressed(path, gray16, metadata=None):
    """Save one uint16 grayscale image as an uncompressed TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gray16 = np.asarray(gray16, dtype=np.uint16)
    tiff.imwrite(
        str(path),
        gray16,
        photometric="minisblack",
        compression=None,
        metadata=metadata or {},
    )


def _sanitize_acquisition_filename_prefix(prefix: str) -> str:
    """Return a filesystem-safe acquisition filename prefix.

    Empty / None / literal "None" keeps the historical default filename.
    """
    prefix = "" if prefix is None else str(prefix).strip()
    if not prefix or prefix.lower() == "none":
        return ""

    safe_chars = []
    for ch in prefix:
        if ch.isalnum() or ch in ("-", "_"):
            safe_chars.append(ch)
        elif ch.isspace():
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


class Gray16TiffAcquisitionThread(QThread):
    """Acquire frames and save them into one multi-page 16-bit grayscale TIFF.

    mode:
        "Time"     - acquire until duration_s is reached
        "Sequence" - acquire frame_count frames
        "Manual"   - acquire until stop() is called

    Output:
        acquisition_gray16_stack.tif
            Multi-page BigTIFF, uint16 grayscale, uncompressed.
        acquisition_metadata.json
            Acquisition metadata and final saved frame count.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        camera_source,
        output_dir,
        mode: str,
        duration_s: float = 10.0,
        frame_count: int = 100,
        interval_ms: int = 0,
        use_gpu: bool = False,
        filename_prefix: str = "",
        parent=None,
    ):
        super().__init__(parent)

        self.camera_source = camera_source
        self.output_dir = Path(output_dir)
        self.mode = str(mode)
        self.duration_s = max(0.0, float(duration_s))
        self.frame_count = max(1, int(frame_count))
        self.interval_ms = max(0, int(interval_ms))
        self.use_gpu = bool(use_gpu and opencv_cuda_available())
        self.filename_prefix = _sanitize_acquisition_filename_prefix(filename_prefix)

        self._running = False
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        self._running = False

    def _make_initial_metadata(self):
        return {
            "mode": self.mode,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "interval_ms": self.interval_ms,
            "use_gpu": self.use_gpu,
            "format": "multi-page uint16 grayscale uncompressed TIFF",
            "filename_prefix": self.filename_prefix or None,
            "tiff_file": (
                f"{self.filename_prefix}_acquisition_gray16_stack.tif"
                if self.filename_prefix else "acquisition_gray16_stack.tif"
            ),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": (
                "Source stream is 8-bit MJPEG or YUYV422; grayscale values "
                "are scaled to uint16 by value * 257."
            ),
        }

    def _write_metadata(self, metadata: dict):
        with open(self.output_dir / "acquisition_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            metadata = self._make_initial_metadata()
            self._write_metadata(metadata)

            stack_name = (
                f"{self.filename_prefix}_acquisition_gray16_stack.tif"
                if self.filename_prefix else "acquisition_gray16_stack.tif"
            )
            stack_path = self.output_dir / stack_name

            self._running = True
            self._stop_requested = False

            start_t = time.perf_counter()
            last_save_t = 0.0
            last_seq = -1
            saved = 0

            self.progress.emit(0, f"Acquisition started: {stack_path}")

            # BigTIFF avoids the 4 GB classic-TIFF limit during long acquisitions.
            # compression=None guarantees uncompressed TIFF pages.
            with tiff.TiffWriter(str(stack_path), bigtiff=True) as tif:
                while self._running and not self._stop_requested:
                    now = time.perf_counter()
                    elapsed = now - start_t

                    if self.mode == "Time" and elapsed >= self.duration_s:
                        break

                    if self.mode == "Sequence" and saved >= self.frame_count:
                        break

                    if self.interval_ms > 0 and saved > 0:
                        if (now - last_save_t) * 1000.0 < self.interval_ms:
                            self.msleep(1)
                            continue

                    frame, seq = self.camera_source.get_latest_full_bgr()

                    if frame is None:
                        self.msleep(2)
                        continue

                    if seq == last_seq:
                        self.msleep(1)
                        continue

                    gray16 = bgr_to_gray16(frame, use_gpu=self.use_gpu)

                    if gray16 is None:
                        self.msleep(1)
                        continue

                    saved += 1
                    last_seq = seq
                    last_save_t = time.perf_counter()

                    tif.write(
                        gray16,
                        photometric="minisblack",
                        compression=None,
                        metadata={
                            "axes": "YX",
                            "frame_index": int(saved),
                            "source_sequence": int(seq),
                            "elapsed_s": float(elapsed),
                        },
                    )

                    if self.mode == "Sequence":
                        pct = int(saved / max(1, self.frame_count) * 100)
                    elif self.mode == "Time":
                        pct = int(min(100, elapsed / max(self.duration_s, 1e-6) * 100))
                    else:
                        pct = 0

                    self.progress.emit(
                        pct,
                        (
                            f"Recording stack: {saved} frame(s), "
                            f"seq={seq}, GPU={'On' if self.use_gpu else 'Off'}"
                        ),
                    )

            elapsed_total = time.perf_counter() - start_t
            metadata.update(
                {
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "saved_frames": int(saved),
                    "elapsed_s": float(elapsed_total),
                    "average_saved_fps": float(saved / elapsed_total) if elapsed_total > 0 else 0.0,
                }
            )
            self._write_metadata(metadata)

            self._running = False
            self.progress.emit(100, f"Acquisition finished: {saved} frame(s) saved to {stack_path}")
            self.finished.emit(str(self.output_dir))

        except Exception as exc:
            self._running = False
            self.error.emit(str(exc))

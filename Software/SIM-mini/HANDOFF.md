# HANDOFF: miniSIM Scope Jetson Orin NX Deployment

## Purpose

This file is for the next AI/helper session. Read this first before continuing
Jetson Orin NX deployment, camera testing, or further code cleanup.

The project is:

```text
miniSIM_full_libjpeg_turbo/miniSIM_scope
```

It is a PySide6 desktop GUI for miniSIM pattern projection, USB camera preview,
MJPEG/YUYV422 acquisition, SIM/vessel/neuron illumination modes, and 16-bit
grayscale TIFF stack recording.

## Current Status

The project was originally Windows-focused and used FFmpeg DirectShow for USB
camera capture. It has been modified to support Jetson/Linux through FFmpeg
V4L2 while preserving the Windows DirectShow path.

Python syntax validation has passed on the Windows workspace:

```bash
python -m py_compile <all project .py files>
```

The code has not yet been tested on a real Jetson Orin NX device. The most
important next step is real hardware validation with `/dev/video*`.

## Key Files

- `main.py`
  - Main PySide6 application and UI.
  - Now uses unified camera helpers instead of DirectShow-only helpers.
  - The UI shows V4L2 wording on Linux and DirectShow wording on Windows.

- `core/camera.py`
  - Camera platform abstraction.
  - Windows: FFmpeg DirectShow.
  - Linux/Jetson: FFmpeg V4L2.
  - MJPEG path stores latest JPEG bytes and decodes preview with libjpeg-turbo.
  - YUYV422 path converts FFmpeg raw BGR output.
  - `FFmpegCameraThread` is the platform-neutral class.
  - `FFmpegDirectShowCameraThread` remains as a backward-compatible alias.

- `core/constants.py`
  - `DEFAULT_CAMERA_DEVICE_NAME = "/dev/video0"`.
  - `DEFAULT_WINDOWS_CAMERA_DEVICE_NAME = "USB Camera"`.

- `requirements.txt`
  - Python dependencies with version lower bounds.
  - Native Linux packages are intentionally not pip dependencies.

- `JETSON_ORIN_NX_SETUP.md`
  - Full Jetson/Linux setup instructions.
  - Includes apt packages, venv setup, camera checks, and UVC control examples.

- `README.md`
  - Updated to mention Windows DirectShow and Linux/Jetson V4L2 support.

## Python Libraries Used

- `PySide6`: GUI, projection windows, Qt widgets, threads and signals.
- `opencv-python` or JetPack OpenCV: image conversion, color maps, resizing,
  morphology support, and optional CUDA availability checks.
- `PyTurboJPEG`: in-memory MJPEG/JPEG decode and JPEG encode.
- `numpy`: image buffers and numeric processing.
- `scipy`: vessel-mode image processing.
- `scikit-image`: filters, morphology, measurements, blob detection.
- `tifffile`: TIFF stack read/write, including acquisition output.

Native runtime dependencies on Jetson/Linux:

- `ffmpeg`
- `v4l-utils`
- `libturbojpeg0`
- Qt xcb runtime packages for PySide6

## Jetson Setup Summary

From `miniSIM_full_libjpeg_turbo/miniSIM_scope`:

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg v4l-utils \
  libturbojpeg0 \
  libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxkbcommon-x11-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py
```

If PyTurboJPEG cannot find the native library:

```bash
export TURBOJPEG_LIB=/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0
```

If camera permissions fail:

```bash
sudo usermod -aG video $USER
```

Then log out and back in.

## Camera Validation Checklist

Run these on the Jetson before opening the GUI:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/video0
```

Inside the GUI:

1. Confirm Camera Control lists `/dev/video0` or the correct `/dev/video*`.
2. Confirm MJPEG and/or YUYV422 modes populate in Resolution / FPS.
3. Prefer MJPEG first for high-resolution USB cameras.
4. Open camera and check live preview.
5. Test ROI crop.
6. Test Image Acquisition Sequence Mode with a small frame count, e.g. 10.
7. Confirm output TIFF stack and `acquisition_metadata.json` are written.
8. Test SIM, vessel, and neuron modes only after basic camera preview/acquisition works.

## Important Implementation Notes

Linux camera command construction is in `core/camera.py`:

- MJPEG/V4L2 uses:

```text
ffmpeg ... -f v4l2 -input_format mjpeg -video_size WxH -framerate FPS -i /dev/videoX -c:v copy -f image2pipe pipe:1
```

- YUYV422/V4L2 uses:

```text
ffmpeg ... -f v4l2 -input_format yuyv422 -video_size WxH -framerate FPS -i /dev/videoX -pix_fmt bgr24 -vcodec rawvideo -f rawvideo pipe:1
```

Mode discovery prefers:

```bash
v4l2-ctl -d /dev/videoX --list-formats-ext
```

and falls back to FFmpeg format listing.

## Known Risks / Things To Watch

- Jetson OpenCV:
  - `opencv-python` from pip may not have CUDA.
  - The app still works without OpenCV CUDA; GPU options stay disabled.
  - If CUDA OpenCV is required, prefer the JetPack-provided OpenCV package.

- PySide6 on Jetson:
  - Missing xcb libraries can prevent the GUI from launching.
  - Install the Qt xcb packages listed above.

- Camera mode parsing:
  - Different UVC cameras print mode lists differently.
  - If Resolution / FPS is empty or wrong, inspect `v4l2-ctl --list-formats-ext`
    output and adjust parser logic in `query_v4l2_device_modes()`.

- FFmpeg V4L2 input formats:
  - Some cameras report `MJPG` rather than `mjpeg`; FFmpeg usually accepts
    `mjpeg`.
  - Some cameras may expose YUYV as `yuyv422`; this is what the code uses.

- Projection display:
  - Projector logic still expects a secondary screen suitable for 640 x 480
    projection.
  - Use Screen test mode if the secondary screen is larger than 640 x 480.

## Cleanup Already Done

- Removed generated `__pycache__` folders from the project tree.
- Removed unused DirectShow-only imports from `main.py`.
- Removed an old output-folder UI remnant that was created but not shown.
- Kept generic helper functions in `core/image_io.py` even if not currently
  used by `main.py`, because modes and future tools may reuse them.

## Recommended Next AI Actions

1. On Jetson, read this file and `JETSON_ORIN_NX_SETUP.md`.
2. Install native dependencies and Python dependencies.
3. Run camera format commands and paste output if mode detection fails.
4. Launch `python main.py`.
5. If the camera fails to open, capture the FFmpeg command from the GUI error
   message and test it directly in terminal.
6. Keep changes small: first fix camera detection/opening, then acquisition,
   then projection/mode-specific workflows.


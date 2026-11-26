# Jetson Orin NX Linux Setup

This project now supports Linux camera capture through FFmpeg V4L2. On Jetson
Orin NX, USB UVC cameras are normally exposed as `/dev/video0`, `/dev/video1`,
and so on.

## Python libraries used

- `PySide6`: Qt GUI, projector window, dialogs, widgets, signals and threads.
- `opencv-python` / JetPack OpenCV: image conversion, color maps, resizing,
  morphology helpers, and optional CUDA checks.
- `PyTurboJPEG`: in-memory MJPEG/JPEG decode and JPEG encode.
- `numpy`: array math and image buffers.
- `scipy`: vessel processing via `scipy.ndimage`.
- `scikit-image`: filters, morphology, region measurements and blob detection.
- `tifffile`: TIFF stack reading/writing, including acquisition output.

Native Linux tools/libraries:

- `ffmpeg`: camera capture through V4L2 and stream piping.
- `libturbojpeg`: native runtime used by `PyTurboJPEG`.
- `v4l-utils`: `v4l2-ctl` mode/control inspection.
- Qt/X11 libraries: runtime dependencies for `PySide6` on desktop Linux.

## Install system dependencies

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg v4l-utils \
  libturbojpeg0 \
  libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxkbcommon-x11-0
```

If `libturbojpeg` is installed in a custom location, set:

```bash
export TURBOJPEG_LIB=/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0
```

## Create Python environment

```bash
cd miniSIM_full_libjpeg_turbo/miniSIM_scope
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

JetPack already ships an NVIDIA-built OpenCV in many images. If you need CUDA
OpenCV, use the JetPack OpenCV package instead of the pip `opencv-python` wheel.
The application works without OpenCV CUDA; GPU checkboxes stay disabled when
CUDA support is not present.

## Check camera formats

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls
```

The application supports MJPEG and YUYV422 modes. MJPEG is preferred for high
resolution/fps USB cameras because it reduces USB bandwidth and uses
libjpeg-turbo for fast decoding.

## Run

```bash
source .venv/bin/activate
python main.py
```

On Linux, the Camera Control panel automatically lists `/dev/video*` devices
and uses FFmpeg V4L2. If no device is detected, check permissions:

```bash
groups
sudo usermod -aG video $USER
```

Log out and back in after adding the `video` group.

## Useful UVC control examples

```bash
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl -d /dev/video0 --set-ctrl exposure_auto=1
v4l2-ctl -d /dev/video0 --set-ctrl exposure_absolute=200
v4l2-ctl -d /dev/video0 --set-ctrl gain=0
```

Control names vary by camera. Always inspect `--list-ctrls` first.

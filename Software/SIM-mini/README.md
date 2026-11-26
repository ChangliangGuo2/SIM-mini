# miniSIM-Scope Controller

Run:

```bash
pip install -r requirements.txt
python main.py
```

Jetson Orin NX / Linux setup:

```bash
sudo apt update
sudo apt install -y ffmpeg v4l-utils libturbojpeg0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py
```

See `JETSON_ORIN_NX_SETUP.md` for the full Linux dependency list, V4L2 camera
checks, and UVC control examples.


## libjpeg-turbo setup

The Python dependency is installed by `pip install -r requirements.txt`, but the
native libjpeg-turbo runtime must also be present.

- Windows: install the 64-bit libjpeg-turbo package. The default
  `C:\libjpeg-turbo64\bin\turbojpeg.dll` location is detected automatically.
- Linux: install the distribution package that provides `libturbojpeg.so`
  (commonly `libturbojpeg0`).
- macOS: install `jpeg-turbo` so `libturbojpeg.dylib` is available.
- For a custom installation, set `TURBOJPEG_LIB` to the full native-library path.

The JPEG codec keeps the former output contract: contiguous `uint8` BGR arrays,
grayscale JPEG files remain 2-D when loaded with the generic image reader,
full-size decoding is used for processing/saving, and JPEG-native 1/2, 1/4,
or 1/8 reduced decoding is used for preview. JPEG encoding uses quality 95 and
4:2:0 chroma subsampling for color images. No fast-DCT or fast-upsampling
quality flags are enabled.


Project structure:

```text
main.py
core/
  camera.py        # FFmpeg DirectShow/V4L2 MJPEG/YUYV422 acquisition thread
  jpeg_turbo.py    # libjpeg-turbo in-memory JPEG decoder
  config.py
  constants.py
  image_io.py
  projection.py
  save_worker.py
  widgets.py       # image result dialog + histogram/contrast widget
modes/
  sim_mode.py      # Configurable-angle SIM and sparse-SIM acquisition
  vessel_mode.py   # vessel-mask generation and selective illumination
  neuron_mode.py   # neuron stack acquisition, ROI detection, and mask projection
```

Feature notes:

- Camera acquisition supports FFmpeg DirectShow on Windows and FFmpeg V4L2 on Linux/Jetson; choose MJPEG or YUYV422 in Camera Control.
- FFmpeg performs OS camera capture and copies MJPEG frames to the pipe; MJPEG/JPEG bytes are decoded directly to BGR by libjpeg-turbo.

- All JPEG pixel decode/encode paths use libjpeg-turbo: camera MJPEG preview, ROI/YUYV JPEG generation, generic `.jpg/.jpeg` file reading and writing, and neuron-mode JPEG test-image loading. Original camera MJPEG bytes are still saved directly when no crop/re-encode is required.
- Default device name is `USB Camera`.
- The Resolution / FPS list is filtered by the selected video format. Preview resize in the camera thread is disabled; MJPEG preview uses reduced JPEG decode when appropriate.
- Vessel and neuron modes keep their concrete processing pipelines in separate modules.
- Widefield calibration projects large up and right arrows for orientation checks.
- SIM provides selected non-axis-aligned start angles in a dropdown; Sparse SIM uses fixed horizontal/vertical orientations, a default 12-pixel period, one-pixel bright stripe, and 12 phases per orientation.
- SIM acquisition requires fixed manual exposure and gain. Configure the values in
  `Camera Control > Advance`; the app applies and reads them back before opening the
  FFmpeg stream. Windows uses OpenCV DirectShow with an explicit device index, and
  Linux uses `v4l2-ctl`. Failed verification blocks SIM/Sparse-SIM acquisition.
- Conventional SIM and Sparse-SIM both discard four additional frames by default
  when changing from the white-field reference to the first pattern. This is the
  starting value for the 2 FPS YUYV422 path and remains user-adjustable.


## Image Acquisition Panel

The new Image Acquisition panel is placed above Camera Control.

Modes:

- **Time Mode**: saves frames until the selected duration is reached.
- **Sequence Mode**: saves the selected number of frames.
- **Manual Start/Stop**: saves frames continuously until Stop Acquisition is pressed.

Implementation:

- Acquisition code is in `core/camera.py`.
- Frames are saved into one **multi-page 16-bit uncompressed grayscale TIFF stack**.
- The USB camera stream is 8-bit MJPEG or YUYV422; grayscale values are scaled to uint16 by `value * 257`.
- Optional GPU processing uses OpenCV CUDA for BGR-to-grayscale conversion when the installed OpenCV build supports CUDA.
- Each acquisition folder contains `acquisition_gray16_stack.tif` and `acquisition_metadata.json`.


## Status-bar recording feedback

During image acquisition, the status bar shows:

- A progress bar.
- A blinking red `● REC` indicator on the right side.
- Stream FPS and camera status remain visible.

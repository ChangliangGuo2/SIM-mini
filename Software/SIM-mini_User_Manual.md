# SIM-mini User Manual

## 1. Installation and startup

SIM-mini controls pattern projection and USB camera acquisition through a graphical interface. It provides widefield illumination, structured illumination microscopy (SIM) acquisition, sparse SIM acquisition, vessel-mask illumination, and neuron-mask illumination.

The camera capture paths are Windows DirectShow and Linux Video4Linux2 (V4L2). Projection patterns have a logical resolution of 640 × 480 pixels. Normal projection requires a secondary display reported by the operating system at exactly this resolution.

### 1.1 Prepare the application folder

Keep `main.py`, `requirements.txt`, `core`, and `modes` together. Run commands from the directory containing `main.py`. Relative output paths and the pattern cache are resolved from the current working directory.

The source directory may retain the name `miniSIM_scope`; use its actual path when opening a terminal.

### 1.2 Windows installation

Install Python and FFmpeg. Make `ffmpeg.exe` available on `PATH`, or enter its full path in the application's **FFmpeg EXE** field.

Install the 64-bit libjpeg-turbo native runtime for a 64-bit Python environment. The application searches the standard location `C:\libjpeg-turbo64\bin\turbojpeg.dll`. If the library is elsewhere, specify its actual location before starting the application:

```powershell
$env:TURBOJPEG_LIB = 'C:\path\to\turbojpeg.dll'
```

In the directory containing `main.py`, create an environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
.\.venv\Scripts\python.exe main.py
```

The Python JPEG binding is `PyTurboJPEG>=1.8.3,<2.0`. Installing the Python binding does not install the native runtime. A different package exposing a module named `turbojpeg` can cause a binding mismatch.

### 1.3 Linux and Jetson Orin NX installation

On an apt-based system, install the dependencies listed in the project setup instructions:

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg v4l-utils libturbojpeg0 \
  libxcb-cursor0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxkbcommon-x11-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py
```

Package availability depends on the Linux distribution. If native-library discovery fails, set `TURBOJPEG_LIB` to the installed library path. For example, use the following path only if the file exists on the Jetson:

```bash
export TURBOJPEG_LIB=/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0
```

Inspect the camera before opening the application:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls
```

Replace `/dev/video0` with the detected device. If the account lacks camera access, add it to the `video` group and log out and back in:

```bash
sudo usermod -aG video "$USER"
```

CUDA acceleration depends on the installed OpenCV build. An NVIDIA JetPack OpenCV installation and the pip `opencv-python` wheel are different builds; installing the wheel does not establish CUDA support. CPU operation is available.

The project handoff records Linux support in the code but no completed validation on a physical Jetson Orin NX. Establish basic preview and a short recording on the target device before using patterned acquisition.

## 2. Camera connection and projection setup

### 2.1 Connect the camera

1. Open **Camera Control → Base**.
2. Set **FFmpeg EXE** to `ffmpeg` or the full executable path.
3. Click **Reload Devices** and select the camera. Windows shows **DirectShow Device**; Linux shows **V4L2 Device**.
4. Select **Video Format**, then select a supported entry in **Resolution / FPS**. The available entries depend on the selected format and camera.
5. Configure manual exposure and gain as described in Section 3 before opening the camera for SIM acquisition.
6. Click **Open Camera**. Wait for a live image and check the stream status.

MJPEG reduces camera transport bandwidth through lossy JPEG compression. YUYV422 uses uncompressed camera transport. Saving an MJPEG frame as an uncompressed TIFF does not remove compression already introduced by the camera.

The default camera names are `USB Camera` on Windows and `/dev/video0` on Linux. Select the detected device when these defaults do not match the hardware.

### 2.2 Set up the projection display

1. Configure the projector as a secondary desktop display. Set the display geometry reported to the application to 640 × 480 pixels for normal projection.
2. In **Projection Screen**, leave **Screen test mode: allow secondary screen larger than 640x480** unchecked for normal operation.
3. Click **Refresh Screens** and inspect the detected display list. The application selects the first eligible secondary display.
4. Open **Widefield Mode** and click **Calibration: Up + Right Arrows**.
5. Check the observed orientation and field coverage before acquiring data.

The arrow image is an orientation check. The software does not derive a camera-to-projector registration transform from it. Verify that projected masks overlap their intended targets in the physical imaging system.

For screen-only inspection, enable **Screen test mode**. The application opens a 640 × 480 popup on an eligible secondary display, or on the primary display if no secondary target is found. This option changes projection placement; it does not replace camera input for SIM acquisition.

### 2.3 Save the setup

Click **Save Config** after selecting a valid camera mode. Settings are stored in `sim_usb_camera_config.json` beside `main.py`. The application reads this file on startup. Files with computer-specific suffixes are separate files and are not selected automatically by this configuration path.

Saved settings include camera selection and recording settings. They also include exposure, gain, and the camera region of interest. SIM pattern parameters and mask-review parameters are not included in this main configuration save; check them for each session.

## 3. Preview, exposure, and region of interest

### 3.1 Set exposure and gain for SIM

1. Close the camera if it is already open.
2. Open **Camera Control → Advance**.
3. Keep **Require verified manual exposure and gain for SIM acquisition** enabled.
4. Enter **Manual Exposure** and **Manual Gain** using values accepted by the camera driver.
5. On Windows, set **DirectShow Device Index** to the OpenCV index of the selected camera. This numeric index is separate from the camera name selected for FFmpeg.
6. Reopen the camera. Read the **Verification** result.

Windows DirectShow commonly represents exposure as log2(seconds); the interface gives `-1` as an example corresponding to 0.5 seconds. Linux uses the integer exposure units reported by `v4l2-ctl`. Treat these as driver controls rather than a shared millisecond scale.

Exposure and gain are applied before the FFmpeg stream opens. After changing these settings, close and reopen the camera so they are applied and checked. SIM and sparse SIM require a successful verification report for the current camera stream. Disabling the requirement blocks these modes.

Use **Open Native DirectShow Property Panel** on Windows for other driver controls. On Linux, inspect `v4l2-ctl --list-ctrls` before setting device-specific controls.

### 3.2 Adjust display contrast

Use **Histogram / Display Contrast** to inspect image intensity. Keep **Auto display contrast** enabled for automatic preview adjustment, or disable it to set the black and white levels manually. Optional color mapping changes the displayed preview.

Display contrast and color mapping are applied to preview images. Recording and mode-processing paths obtain camera data independently of these display adjustments. Check exposure separately when a bright preview could be caused by automatic display stretching.

### 3.3 Select a camera region of interest

A region of interest (ROI) limits the camera area used by the shared frame-access paths.

1. Open **Camera Control → Advance** and enable the ROI option in **ROI / Fullframe**.
2. Click **Start ROI Pick** and drag a rectangle on the preview, or enter **X**, **Y**, **Width**, and **Height**.
3. Check the resulting crop. The selection is constrained to a 4:3 aspect ratio and bounded by the camera frame.
4. Disable the ROI option to return to the full frame.

The crop affects recorded frames and camera-based mode processing. Finalize it before calculating illumination masks. Changing it afterward changes the coordinate relationship between the camera image and projection.

Camera cropping and neuron ROI marking serve different purposes. Camera cropping selects an image area; neuron ROI marking edits detected target regions within a stack.

## 4. Operating modes

### 4.1 Widefield illumination

In **Widefield Mode**, click **Widefield: Full White (640x480)** to display a white field. Use **Calibration: Up + Right Arrows** to check orientation. Click **Close Illumination** to display black.

For a widefield time series, select the white field and use **Image Acquisition** as described in Section 5. The widefield buttons control illumination and do not start a recording.

### 4.2 Conventional SIM acquisition

SIM acquisition saves one white-field reference followed by nine patterned camera frames. The patterned sequence contains three phases at each of three angles.

1. Confirm a valid projection display and a live camera frame.
2. Check that manual exposure and gain are verified.
3. Set **Save Folder** in **Image Acquisition** if a custom output base directory is required.
4. Open **SIM Mode** and select **Stripe Width** and **Start Angle**.
5. Check **Pattern Angles** and the timing controls.
6. Enable **Save projected pattern TIFFs** if the projected images should also be exported.
7. Click **Start 9-Frame SIM Acquisition** and wait for completion and file saving.

| **Parameter** | **Available values or initial setting** |
|---|---|
| Stripe width | 3 or 6 projector pixels; initial selection 3 |
| Stripe period | Twice the stripe width |
| Start angle | 0°, 5°, 10°, 15°, 20°, 25°, or 45°; initial selection 15° |
| Pattern angles | Start angle, start angle + 120°, and start angle + 240°, wrapped at 360° |
| Phase shifts | 0, 2, and 4 pixels for width 3; 0, 4, and 8 pixels for width 6 |
| Projection Settling Time | Initially 120 ms; range 0–2000 ms |
| Startup Warm-up Frames | Initially 2; range 0–30 |
| Pattern-change Discard Frames | Initially 2; range 0–30 |
| First-pattern Extra Discard Frames | Initially 4; range 0–30 |

All stripe dimensions refer to projector pixels. With the initial 15° start angle, the angle sequence is 15°, 135°, and 255°.

The software first waits for projection settling, then counts new camera frames. Startup warm-up frames are discarded before the white-field reference. Each patterned capture discards the configured transition frames; the first pattern also discards the extra first-pattern frames. With initial settings, the first pattern discards six frames after settling.

These are software initial values, not measured synchronization guarantees for every camera. Inspect the first patterned frame for residual white-field illumination. Increase the relevant delay or discard count when the saved image still shows the preceding projection.

On completion, projection changes to black. The mode exports acquired images and metadata; it does not run SIM reconstruction.

### 4.3 Sparse SIM acquisition

1. Complete the camera and projection checks used for conventional SIM.
2. Open **Sparse SIM Mode**.
3. Set **Sparse Period** and **Bright Stripe Width**.
4. Check the settling and discard controls. Their initial values match conventional SIM.
5. Choose whether to save projected pattern TIFFs.
6. Click the acquisition button. Its frame count changes with the selected period.

| **Parameter** | **Available values or initial setting** |
|---|---|
| Sparse Period | Initially 12 pixels; range 3–64 |
| Bright Stripe Width | Initially 1 pixel; from 1 to period minus 1 |
| Pattern angles | Fixed at 0° and 90° |
| Phase shifts | Every integer shift from 0 to period minus 1 |
| Patterned frame count | Twice the period; initially 24 |

The initial sequence saves one white-field reference plus 24 patterned frames. The same fixed-exposure verification applies to both SIM modes.

### 4.4 Vessel illumination

1. Open the camera and verify projection alignment.
2. Open **Vessel Illumination**. Leave **Test mode (use local image)** disabled for live acquisition.
3. Set **White illumination delay** and **Mask capture delay**. Both initially use 300 ms.
4. Click **Calculate Mask**. The application projects white, waits, and processes a camera image.
5. Inspect the original image, enhanced result, and mask in the result window.
6. Adjust the processing parameters. Use **Apply Parameters & Recalculate** when recomputation is required.
7. Use **Save Mask & Apply Parameters** to save reviewed results.
8. Click **Show Mask** to project the mask. In live mode, the application then captures a mask-illumination image after the selected delay.

Available enhancement methods are **Frangi vesselness**, **Sato tubeness**, and **Meijering curvilinear**. Background removal and denoising precede vessel enhancement. Thresholding and mask cleanup determine the projected area.

Use **Invert image** when vessel polarity requires inversion. Review the mask after changing the scale range or threshold. The projected mask is generated on a 640 × 480 image grid, so verify overlap in the camera preview before using it for selective illumination.

To process an existing image, enable **Test mode (use local image)** and select a local file with **Browse**. This bypasses live camera acquisition for mask calculation. Screen test mode can be enabled separately to inspect projection in a popup.

### 4.5 Neuron illumination

1. Select the intended illumination before recording the neuron stack; **Calculate Mask** does not automatically switch to white-field illumination.
2. Open **Neuron Illumination** with the camera running.
3. Set **Frame count**, **Capture interval**, and **Saved TIFF frame rate**.
4. Leave **Test mode (use local stack/image)** disabled for live acquisition.
5. Click **Calculate Mask**. The application collects a grayscale stack, saves it, and calculates candidate regions.
6. Inspect the result window. Adjust enhancement and detection parameters as needed, then recalculate.
7. Use **Manual ROI Mark** to review candidates. In the stack review window, left-click adds a center and right-click removes a nearby center. Browse or play frames to assess the selected regions.
8. Save the reviewed result with **Save Mask & Apply Parameters**, then click **Show Mask** to project it.

| **Parameter** | **Initial setting and meaning** |
|---|---|
| Frame count | 100 sampled frames |
| Capture interval | 100 ms between timer-driven samples |
| Saved TIFF frame rate | 10 Hz, stored as TIFF timing metadata |
| Detection sensitivity | Sensitive; Balanced and Conservative are also available |
| Max processing frames | 200 frames in the working stack |
| Max processing side px | 640 pixels for the working image |
| Max score/correlation frames | 120 frames for score and correlation processing |

**Saved TIFF frame rate** labels the saved stack; it does not set the USB camera frame rate. Neuron acquisition samples the latest available frame on a timer and does not require a new camera sequence number for each sample. A short interval can therefore repeat a camera frame. Choose the interval with the live camera rate in mind.

The processing pipeline includes motion correction and background subtraction. Candidate detection uses image scores and temporal correlations. Sensitive mode proposes more candidates for manual review; detected regions require inspection before projection.

For local processing, enable **Test mode (use local stack/image)** and select a file. Supported inputs include TIFF stacks and NumPy arrays. Ordinary image files can be loaded as single-frame inputs, but they contain no temporal activity information. The TIFF reader can flatten non-spatial dimensions into the frame axis, so prepare the intended channel and plane before loading a multidimensional dataset.

Large stacks are processed using a sampled and resized working copy. Review detection parameters in that working-image context rather than assuming every processing pixel corresponds to a full-resolution camera pixel.

## 5. Recording and output files

### 5.1 Record a grayscale TIFF stack

The **Image Acquisition** panel records camera frames independently of the mode-specific SIM acquisition buttons.

1. Open the camera and wait for a live frame.
2. Set **Save Folder** or leave it empty for the default location.
3. Enter an optional **File Prefix**. Leave it empty, or enter `None`, to keep the default filename.
4. Select a **Capture Mode** and set its duration or frame count.
5. Set **Interval**. Zero requests each new frame available to the recording worker; a positive value imposes a minimum interval between saved frames.
6. Click **Start Acquisition**. Observe the progress bar and blinking recording indicator.
7. For manual recording, click **Stop Acquisition**. Wait for the finished status so the TIFF and metadata are closed.

| **Capture mode** | **Stop condition** |
|---|---|
| Time Mode | Selected duration is reached |
| Sequence Mode | Selected number of frames is saved |
| Manual Start/Stop | Stop Acquisition is pressed |

Recording reads the latest available frame and avoids saving the same sequence twice. If processing or storage is slower than the camera, intermediate frames can be skipped. Use `saved_frames` and `average_saved_fps` in the final metadata to check the recording result.

### 5.2 Output locations

With **Save Folder** empty, ordinary recording writes to:

```text
acquisitions/
  gray16_YYYYMMDD_HHMMSS/
    acquisition_gray16_stack.tif
    acquisition_metadata.json
```

SIM and mask modes instead use `captures` as their default base directory. If **Save Folder** is set, both recording and mode-specific output use that directory as their base and create their own timestamped subdirectories.

Choose distinct sessions when starting runs within the same second; generated directory names have one-second timestamp resolution.

### 5.3 Ordinary recording format

Ordinary recording produces an uncompressed, multi-page grayscale BigTIFF. The current MJPEG and YUYV422 camera paths supply 8-bit image data. The grayscale recording conversion stores each 8-bit value as `value × 257` in a 16-bit container. This mapping preserves the 8-bit levels and does not create native 16-bit sensor precision.

An optional prefix changes the stack filename to `<prefix>_acquisition_gray16_stack.tif`. The metadata filename remains `acquisition_metadata.json`. Individual TIFF pages include a frame index, source sequence, and elapsed time. Final JSON metadata includes the saved frame count and average saved rate.

### 5.4 SIM and sparse SIM output

Conventional SIM uses a `sim_YYYYMMDD_HHMMSS` directory; sparse SIM uses `sparse_sim_YYYYMMDD_HHMMSS`.

```text
sim_YYYYMMDD_HHMMSS/
  acquisition_metadata.json
  raw_widefield_tiff/
    raw_widefield_seq_<sequence>.tif
  camera_original_tiff/
    frame_01_angle_015_phase_00_shift_0px_seq_<sequence>.tif
    ...
  patterns/                         # Created when pattern export is enabled
    ...
```

The example patterned filename corresponds to the initial conventional SIM settings. Actual angle and shift fields follow the selected sequence.

Camera images are saved as separate uncompressed TIFFs. Color camera frames are converted from the internal BGR channel order to RGB without changing their data type. They are not converted to the ordinary recording panel's 16-bit grayscale format.

The JSON metadata records the camera settings and exposure verification report. It also records pattern order, capture sequence numbers, and timing parameters. Check `status` and the recorded patterned-frame count when assessing completion. Patterned-frame counts exclude the separate white-field reference.

The `sim_patterns` directory in the working directory stores reusable pattern-library data. The optional `patterns` directory inside a capture stores exported projection TIFFs for that run.

### 5.5 Vessel outputs

Vessel results are written under `vessel_YYYYMMDD_HHMMSS`. Saved PNG files include `vessel_original_640x480.png`, `vessel_mask_640x480.png`, and `vessel_compare.png`. Additional images store preprocessing, enhancement response, and the enhanced-plus-original result.

Live **Show Mask** can also save `mask_illumination_camera_frame_seq_<sequence>.jpg` and `mask_illumination_camera_frame_decoded.png`. These images document the camera view after mask projection. Local-image test mode skips that live capture.

The vessel mode retains its output directory after the first result. Further calculations and saves in the same session can replace result files in that directory. Copy a result directory before recalculating if the previous result must be retained.

### 5.6 Neuron outputs

Live neuron runs use `neuron_YYYYMMDD_HHMMSS`; local-input runs use `neuron_test_YYYYMMDD_HHMMSS`.

The live raw stack is `neuron_raw_stack_uncompressed.tif`: an 8-bit grayscale ImageJ TIFF with time, height, and width axes. This is distinct from the 16-bit BigTIFF produced by **Image Acquisition**.

Processed outputs include:

- `neuron_projection.tif`: the processed projection image.
- `neuron_score_map.tif`: the enhanced score map.
- `neuron_labels.tif`: integer region labels.
- `neuron_mask_projector_640x480.png`: the binary projection mask.
- `neuron_compare.png`: the result comparison image.
- `neuron_centers_coordinates.csv`: region identifiers and center coordinates.

Additional files are written when corresponding results are available, including motion shifts and temporal background. Local-input processing can save `neuron_test_input_stack_uncompressed.tif` after conversion to the application's 8-bit grayscale stack representation.

Re-saving a reviewed result updates files in the same result directory. Preserve a copy before changing a result that must remain available.

### 5.7 Finish a session

Stop ordinary recording and wait for its finished status. Wait for mode acquisition and background file saving to complete. Use **Widefield Mode → Close Illumination**, then **Close Camera**. Close the main window and confirm exit.

## 6. Troubleshooting

### The camera is not listed

Check **FFmpeg EXE**, then click **Reload Devices**. On Linux, inspect `v4l2-ctl --list-devices` and account permissions. Close other applications that may be using the camera. Select the detected device rather than relying on its default name.

### The camera is listed but preview does not start

Select a format and resolution entry reported for that camera. Inspect the FFmpeg error shown by the application. Try a lower-bandwidth supported mode if the selected stream fails. On Windows, check the camera driver and whether another application holds the device.

### libjpeg-turbo cannot be loaded

Check that `PyTurboJPEG>=1.8.3,<2.0` is installed in the environment used to launch SIM-mini. Confirm the native library exists and matches the environment architecture. Set `TURBOJPEG_LIB` to its full path before launching. If the error lists missing Python symbols, check for an incompatible `turbojpeg` package in that environment.

### No valid projection screen is found

Inspect **Projection Screen**. Normal operation needs a secondary display reported at exactly 640 × 480. Set the display configuration and click **Refresh Screens**. For interface testing on another display size, enable **Screen test mode**.

### SIM is blocked by exposure verification

Read **Camera Control → Advance → Verification**. Keep the manual-exposure requirement enabled. Close the camera, correct the exposure or gain, and reopen it. On Windows, verify the numeric **DirectShow Device Index** points to the same physical device as the FFmpeg selection. On Linux, inspect the camera's available controls and accepted values.

### A patterned capture contains the previous illumination

Increase **Projection Settling Time** or **Pattern-change Discard Frames**. If only the first patterned image contains white-field carryover, adjust **First-pattern Extra Discard Frames**. Repeat a short run and inspect the saved images alongside their capture metadata.

### SIM aborts while waiting for new frames

The error reports how many new frames were received and required. Check that the live stream is advancing. The wait timeout is at least 5 seconds, or 2 seconds per required frame when that is longer. Check the selected exposure and actual stream rate before changing discard counts.

### A mask does not overlap the intended structures

Recheck the arrow orientation and optical alignment. Confirm the camera ROI matches the area used for mask calculation. Recalculate after changing the crop. A 640 × 480 mask export does not establish physical registration by itself.

### Neuron processing uses too much memory

Reduce **Max processing frames**, **Max processing side px**, or **Max score/correlation frames** in the result parameters. These controls reduce the working-copy size. Loading and retaining the input stack still requires memory, so use a smaller input stack if loading itself fails.

### The saved stack has fewer frames than expected

For ordinary recording, check the final `saved_frames` value and saved rate in `acquisition_metadata.json`. Review the capture mode and interval, then check camera delivery and storage performance. Sequence Mode counts saved frames; Time Mode stops by elapsed duration.

For SIM, inspect the run metadata and count the patterned TIFF files separately from the white-field reference. An aborted run can contain only a partial sequence.

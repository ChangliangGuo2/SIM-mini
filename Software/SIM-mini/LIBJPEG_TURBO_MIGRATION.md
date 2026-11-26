# libjpeg-turbo migration audit

## Replaced JPEG paths

1. `core/camera.py`
   - MJPEG preview/full-frame decoding uses `decode_jpeg_to_bgr()`.
   - ROI-cropped MJPEG frames are re-encoded with `encode_bgr_to_jpeg()`.
   - YUYV422 frames requested as JPEG are encoded with `encode_bgr_to_jpeg()`.
   - FFmpeg remains responsible only for DirectShow capture, YUYV-to-BGR conversion,
     and copying MJPEG packets into the pipe.

2. `core/image_io.py`
   - JPEG files are detected from their SOI signature and decoded with
     `decode_jpeg_unchanged()`.
   - `.jpg`, `.jpeg`, `.jpe`, and `.jfif` output uses
     `encode_image_to_jpeg()` at quality 95.
   - Non-JPEG formats continue to use OpenCV so PNG, BMP, and other supported
     formats retain their previous behavior.

3. `modes/neuron_mode.py`
   - The direct `cv2.imdecode()` path now uses `imread_unicode()`, so JPEG test
     images are decoded through libjpeg-turbo while PNG/BMP continue through
     OpenCV.

## Intentionally not replaced

- `modes/sim_mode.py` loads and saves generated stripe patterns as PNG.
  libjpeg-turbo cannot process PNG, and converting these binary projection
  patterns to lossy JPEG would alter algorithm input.
- TIFF and NumPy stack handling remains unchanged.
- Camera MJPEG bytes saved without ROI are copied verbatim. There is no pixel
  decode/encode operation to accelerate, and re-encoding would reduce quality.

## Preserved behavior

- Processing and camera frames remain contiguous `uint8` BGR arrays.
- Grayscale JPEG loaded through the generic reader remains a 2-D array.
- Preview reduced decoding retains the 1/2, 1/4, and 1/8 JPEG IDCT scale choices.
- JPEG write quality remains 95.
- Color JPEG output explicitly uses 4:2:0 chroma subsampling.
- Decode/encode failures keep the original `None`/`False` handling contracts.

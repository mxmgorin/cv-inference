# Architecture

## Overview

```
        HTTP request (multipart or raw image bytes)
                        │
                        ▼
             src/api/detect.rs         ── extract image bytes, log request
                        │
                        ▼
        src/inference (Detector trait)  ── decode → resize 640×640 → normalize
             └─ yolo.rs (YoloDetector)     → tensor → ONNX Runtime → threshold
                        │                   → NMS → map boxes to original size
                        ▼
            src/model/detection.rs      ── Detection / BoundingBox → JSON
```

The `Detector` trait keeps the inference backend pluggable. `YoloDetector` holds
a single ONNX Runtime session shared behind an `Arc<Mutex<_>>`; each request runs
the blocking inference on Tokio's blocking thread pool so the async runtime is
never blocked.

## Project layout

```
src/
  main.rs              app wiring, router, graceful shutdown
  config.rs            YAML config loading
  error.rs             error type → JSON HTTP responses
  api/detect.rs        POST /detect handler
  inference/mod.rs     Detector trait
  inference/yolo.rs    YOLO ONNX implementation (pre/post-processing, NMS)
  model/detection.rs   Detection / BoundingBox / DetectResponse data model
models/                ONNX model weights (not committed)
docker/Dockerfile      multi-stage container build
scripts/get_model.sh   downloads yolo11n.onnx
config.yaml            server / model / inference settings
docs/                  documentation
```

## Inference pipeline

1. Load the ONNX model at startup.
2. Decode the image (JPEG/PNG).
3. Resize to the model input (640×640).
4. Normalize pixels to `[0, 1]`.
5. Convert to an `NCHW` tensor.
6. Run ONNX Runtime.
7. Apply the confidence threshold.
8. Apply Non-Maximum Suppression (NMS).
9. Return the JSON response.

The model output tensor has shape `[1, 84, 8400]`: for each of the 8400 candidate
anchors, 4 box coordinates (`cx, cy, w, h`) followed by 80 COCO class scores.
Boxes are computed in the 640×640 space, then scaled back to the original image
dimensions before being returned.

## Logging

Each request logs the method, image size, inference time, and detection count:

```
POST /detect
image=1280x720
inference=18ms
objects=5
```

The default log filter is `info,ort=warn`, which hides ONNX Runtime's verbose
startup output. Set `RUST_LOG=info` (or `debug`) to see everything.

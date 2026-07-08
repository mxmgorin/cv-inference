# CV Inference Service (Rust + YOLO + ONNX)

A small, production-style REST API in Rust that accepts an image, runs inference
with a pre-trained **YOLO11** ONNX model via **ONNX Runtime**, and returns the
detected objects as JSON.

The project demonstrates *backend integration* of a computer-vision model —
model loading, image preprocessing, ONNX inference, and detection
post-processing — rather than ML model training.

## Tech stack

- **Rust** (edition 2024)
- **Axum** + **Tokio** — async HTTP server
- **ort** — ONNX Runtime bindings (binaries fetched automatically at build time)
- **image** — decoding & resizing
- **ndarray** — tensor manipulation
- **serde** / **serde_yaml** — JSON responses & YAML config
- **tracing** — structured logging
- **Docker** — containerised deployment

## Architecture overview

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
the blocking inference on Tokio's blocking thread pool.

### Project layout

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
```

## Getting the model

The YOLO11n ONNX weights (~11 MB) are not committed. Download them with:

```bash
./scripts/get_model.sh
# → models/yolo11n.onnx
```

## Build & run (local)

```bash
# 1. get the model
./scripts/get_model.sh

# 2. build (first build also downloads the ONNX Runtime binaries)
cargo build --release

# 3. run — reads config.yaml from the working directory
./target/release/cv-inference
# listening on http://0.0.0.0:8080
```

Override the config path with `CONFIG_PATH=/path/to/config.yaml`, and the log
level with `RUST_LOG=debug`.

## Configuration (`config.yaml`)

```yaml
server:
  port: 8080

model:
  path: models/yolo11n.onnx

inference:
  confidence_threshold: 0.4
  iou_threshold: 0.5
```

## API

### `POST /detect`

Accepts **either**:

- `multipart/form-data` with a `file` field, or
- a raw JPEG/PNG request body.

**Request (multipart):**

```bash
curl -F "file=@cat.jpg" http://localhost:8080/detect
```

**Request (raw body):**

```bash
curl --data-binary @cat.jpg -H "Content-Type: image/jpeg" \
  http://localhost:8080/detect
```

**Response:**

```json
{
  "objects": [
    {
      "class": "person",
      "confidence": 0.97,
      "bbox": {
        "x": 124.0,
        "y": 80.0,
        "width": 200.0,
        "height": 450.0
      }
    }
  ]
}
```

Bounding boxes are in original-image pixel coordinates: `x`/`y` is the top-left
corner, `width`/`height` are the box size.

### `GET /health`

Returns `ok` — useful for container health checks.

## Docker

Build the image (the build downloads the ONNX Runtime binaries):

```bash
docker build -f docker/Dockerfile -t cv-inference .
```

Run the container, mounting the `models/` directory:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/app/models:ro" \
  cv-inference
```

Then:

```bash
curl -F "file=@cat.jpg" http://localhost:8080/detect
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

## Logging

Each request logs the method, image size, inference time, and detection count:

```
POST /detect
image=1280x720
inference=18ms
objects=5
```

## Future improvements

- Swagger / OpenAPI schema
- Multiple detector implementations
- Batch inference
- Prometheus metrics
- Async job queue
- RTSP / video processing

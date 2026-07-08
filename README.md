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
- **ort** — ONNX Runtime bindings (statically linked at build time)
- **image** — decoding & resizing
- **ndarray** — tensor manipulation
- **serde** / **serde_yaml** — JSON responses & YAML config
- **tracing** — structured logging
- **Docker** — containerised deployment

## Quick start

```bash
# 1. download the model (~11 MB) into models/
./scripts/get_model.sh

# 2. build (the first build also downloads & statically links ONNX Runtime)
cargo build --release

# 3. run
./target/release/cv-inference
# listening on http://0.0.0.0:8080
```

In another terminal:

```bash
curl -F "file=@cat.jpg" http://localhost:8080/detect
```

```json
{
  "objects": [
    {
      "class": "person",
      "confidence": 0.97,
      "bbox": { "x": 124.0, "y": 80.0, "width": 200.0, "height": 450.0 }
    }
  ]
}
```

## Documentation

- [Architecture](docs/architecture.md) — design, project layout, inference pipeline, logging
- [API reference](docs/api.md) — endpoints, request/response formats, errors
- [Configuration](docs/configuration.md) — model download, `config.yaml`, environment variables
- [Deployment](docs/deployment.md) — local build & run, Docker
- [Roadmap](docs/roadmap.md) — planned / future improvements

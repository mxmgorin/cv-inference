# Deployment

## Local build & run

```bash
# 1. get the model
./scripts/get_model.sh

# 2. build (the first build also downloads & statically links ONNX Runtime)
cargo build --release

# 3. run — reads config.yaml from the working directory
./target/release/cv-inference
# listening on http://0.0.0.0:8080
```

Run from the project directory so the relative paths in `config.yaml`
(`models/yolo11n.onnx`) resolve. Override the config path with `CONFIG_PATH` and
the log level with `RUST_LOG` (see [configuration.md](configuration.md)).

> ONNX Runtime is statically linked into the binary by the `ort` crate, so there
> is no separate shared library to ship — the release binary is self-contained
> (~28 MB, needs only the C++ standard library at runtime).

## Docker

Build the image (the build downloads the ONNX Runtime static library):

```bash
docker build -f docker/Dockerfile -t cv-inference .
```

Run the container, mounting the `models/` directory so the model file is
available inside it:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/app/models:ro" \
  cv-inference
```

Then send a request:

```bash
curl -F "file=@cat.jpg" http://localhost:8080/detect
```

The `Dockerfile` is a multi-stage build: a `rust:1-bookworm` builder stage
compiles the release binary, and a slim `debian:bookworm-slim` runtime stage
ships only the binary plus `libstdc++6`.

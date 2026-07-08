# ---- Build stage ----
FROM rust:1-bookworm AS builder

WORKDIR /app

# Cache dependencies: copy manifests first, build a dummy target.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# Build the real binary.
COPY src ./src
RUN touch src/main.rs && cargo build --release

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# `ort` statically links ONNX Runtime into the binary, so the only runtime
# dependency beyond the base image is the C++ standard library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/cv-inference /usr/local/bin/cv-inference

COPY config.yaml ./config.yaml

# The models/ directory is mounted at runtime (see README).
EXPOSE 8080
CMD ["cv-inference"]

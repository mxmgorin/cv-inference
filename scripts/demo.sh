#!/usr/bin/env bash
# Regenerate the README demo image: run detection on the sample photo and draw
# the detected bounding boxes. Requires the `draw` feature.
#
# Usage: ./scripts/demo.sh [input-image] [output-image]
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${1:-docs/assets/demo-source.jpg}"
OUTPUT="${2:-docs/assets/demo.jpg}"

./scripts/get_model.sh
cargo build --release --features draw --bin annotate
./target/release/annotate "$INPUT" "$OUTPUT"

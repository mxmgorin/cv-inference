#!/usr/bin/env bash
# Download the default YOLO11n ONNX model into models/.
#
# Usage: ./scripts/get_model.sh
set -euo pipefail

MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
DEST="$DEST_DIR/yolo11n.onnx"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]]; then
  echo "Model already present: $DEST"
  exit 0
fi

echo "Downloading yolo11n.onnx ..."
curl -fSL --retry 3 -o "$DEST" "$MODEL_URL"
echo "Saved to $DEST ($(du -h "$DEST" | cut -f1))"

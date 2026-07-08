# Configuration

## Model weights

The YOLO11n ONNX weights (~11 MB) are **not committed**. Download them with the
helper script:

```bash
./scripts/get_model.sh
# → models/yolo11n.onnx
```

The script pulls `yolo11n.onnx` from the Ultralytics assets release. To use a
different model, drop its `.onnx` file into `models/` and point `model.path` at
it (see below). Any Ultralytics YOLOv8/YOLO11 export with the standard
`[1, 84, 8400]` output works out of the box.

## `config.yaml`

The service reads `config.yaml` from the working directory at startup:

```yaml
server:
  port: 8080

model:
  path: models/yolo11n.onnx

inference:
  confidence_threshold: 0.4   # drop detections below this score
  iou_threshold: 0.5          # NMS overlap threshold
```

`confidence_threshold` and `iou_threshold` take effect on restart — no rebuild
needed.

## Environment variables

| Variable      | Default        | Purpose                                   |
| ------------- | -------------- | ----------------------------------------- |
| `CONFIG_PATH` | `config.yaml`  | Path to the YAML config file              |
| `RUST_LOG`    | `info,ort=warn`| Log filter (e.g. `info`, `debug`, `trace`)|

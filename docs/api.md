# API Reference

Base URL (default): `http://localhost:8080`

## `POST /detect`

Runs object detection on an image and returns the detected objects.

Accepts **either**:

- `multipart/form-data` with a `file` field, or
- a raw JPEG/PNG request body.

### Request — multipart

```bash
curl -F "file=@cat.jpg" http://localhost:8080/detect
```

### Request — raw body

```bash
curl --data-binary @cat.jpg -H "Content-Type: image/jpeg" \
  http://localhost:8080/detect
```

### Response `200 OK`

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

Bounding boxes are in **original-image pixel coordinates**: `x`/`y` is the
top-left corner, `width`/`height` are the box size. The `class` is one of the
80 [COCO](https://cocodataset.org/) categories.

### Errors

Errors are returned as JSON with the appropriate HTTP status:

```json
{ "error": "no `file` field found in multipart form" }
```

| Status | When                                                          |
| ------ | ------------------------------------------------------------- |
| `400`  | Empty body, missing `file` field, or undecodable image        |
| `500`  | Inference failure                                             |

## `GET /health`

Liveness probe. Returns `200 OK` with the body `ok` — useful for container
health checks.

```bash
curl http://localhost:8080/health
# ok
```

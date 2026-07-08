# Roadmap / Future improvements

Ideas for extending the service beyond the current demo scope:

- **Swagger / OpenAPI schema** — machine-readable API description and a docs UI.
- **Multiple detector implementations** — swap models or backends behind the
  existing `Detector` trait (e.g. a remote inference service, or a mock for tests).
- **Batch inference** — accept and process several images per request.
- **Prometheus metrics** — request counts, latency histograms, detections per image.
- **Async job queue** — offload long-running inference and poll for results.
- **RTSP / video processing** — detect on live streams or video files.

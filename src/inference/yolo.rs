//! YOLO detector backed by ONNX Runtime (via the `ort` crate).
//!
//! Works with Ultralytics YOLOv8/YOLO11 ONNX exports, whose single output tensor
//! has shape `[1, 84, 8400]`: 4 box coordinates (cx, cy, w, h) followed by 80
//! COCO class scores, for each of the 8400 candidate anchors.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use image::imageops::FilterType;
use ndarray::{Array, Axis, s};
use ort::{inputs, session::Session, value::TensorRef};

use crate::error::{AppError, Result};
use crate::inference::Detector;
use crate::model::{BoundingBox, Detection};

/// Square input resolution expected by the default YOLO11n export.
const INPUT_SIZE: u32 = 640;

/// COCO class labels, indexed by class id (0..80).
#[rustfmt::skip]
const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

/// A YOLO object detector holding a loaded ONNX Runtime session.
///
/// The session is shared behind an `Arc<Mutex<_>>`: `Session::run` needs `&mut
/// self`, and wrapping it lets the detector be cloned cheaply into the blocking
/// thread pool for each request.
#[derive(Clone)]
pub struct YoloDetector {
    session: Arc<Mutex<Session>>,
    confidence_threshold: f32,
    iou_threshold: f32,
}

impl YoloDetector {
    /// Load the ONNX model from disk and build a detector.
    pub fn new(
        model_path: impl AsRef<Path>,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let session = Session::builder()?.commit_from_file(model_path)?;
        tracing::info!(model = %model_path.display(), "ONNX model loaded");

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            confidence_threshold,
            iou_threshold,
        })
    }

    /// Blocking inference pipeline: decode → resize → normalize → run → post-process.
    fn run_inference(&self, image: &[u8]) -> Result<Vec<Detection>> {
        // 2. Decode the image.
        let original = image::load_from_memory(image)?;
        let (orig_w, orig_h) = (original.width(), original.height());
        tracing::info!("image={orig_w}x{orig_h}");

        // 3. Resize to model input & 4. normalize pixels to [0, 1], layout NCHW.
        let resized = original
            .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
            .to_rgb8();

        // 5. Convert to a tensor of shape [1, 3, H, W].
        let mut input = Array::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for (x, y, pixel) in resized.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            input[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
        }

        // 6. Run ONNX Runtime.
        let started = Instant::now();
        let raw = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| AppError::Internal("inference session mutex poisoned".into()))?;
            let outputs = session.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
            // Output "output0" has shape [1, 84, 8400]; transpose to [8400, 84, 1].
            outputs["output0"].try_extract_array::<f32>()?.t().into_owned()
        };
        tracing::info!("inference={}ms", started.elapsed().as_millis());

        // Scale factors to map 640x640 coordinates back to the original image.
        let scale_x = orig_w as f32 / INPUT_SIZE as f32;
        let scale_y = orig_h as f32 / INPUT_SIZE as f32;

        // 7. Apply the confidence threshold while collecting candidate boxes.
        let predictions = raw.slice(s![.., .., 0]);
        let mut candidates: Vec<Candidate> = Vec::new();
        for row in predictions.axis_iter(Axis(0)) {
            // Best class = argmax over the 80 class scores (indices 4..84).
            let (class_id, &confidence) = row
                .iter()
                .enumerate()
                .skip(4)
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, v)| (idx - 4, v))
                .expect("row always has class scores");

            if confidence < self.confidence_threshold {
                continue;
            }

            let cx = row[0_usize] * scale_x;
            let cy = row[1_usize] * scale_y;
            let w = row[2_usize] * scale_x;
            let h = row[3_usize] * scale_y;

            candidates.push(Candidate {
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
                class_id,
                confidence,
            });
        }

        // 8. Non-Maximum Suppression, then 9. build the response objects.
        let kept = non_maximum_suppression(candidates, self.iou_threshold);
        let detections = kept
            .into_iter()
            .map(|c| Detection {
                class: COCO_CLASSES[c.class_id].to_string(),
                confidence: c.confidence,
                bbox: BoundingBox {
                    x: c.x1.max(0.0),
                    y: c.y1.max(0.0),
                    width: (c.x2 - c.x1).min(orig_w as f32),
                    height: (c.y2 - c.y1).min(orig_h as f32),
                },
            })
            .collect();

        Ok(detections)
    }
}

impl Detector for YoloDetector {
    async fn detect(&self, image: &[u8]) -> Result<Vec<Detection>> {
        // Inference is CPU-bound and blocking; move it off the async runtime.
        let detector = self.clone();
        let image = image.to_vec();
        tokio::task::spawn_blocking(move || detector.run_inference(&image))
            .await
            .map_err(|e| AppError::Internal(format!("inference task failed: {e}")))?
    }
}

/// Internal candidate box in original-image coordinates, used during NMS.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    class_id: usize,
    confidence: f32,
}

impl Candidate {
    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn iou(&self, other: &Candidate) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
}

/// Greedy Non-Maximum Suppression: keep the highest-scoring boxes, dropping any
/// that overlap a kept box (of the same class) by more than `iou_threshold`.
fn non_maximum_suppression(mut candidates: Vec<Candidate>, iou_threshold: f32) -> Vec<Candidate> {
    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut kept: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        let overlaps = kept.iter().any(|k| {
            k.class_id == candidate.class_id && k.iou(&candidate) > iou_threshold
        });
        if !overlaps {
            kept.push(candidate);
        }
    }
    kept
}

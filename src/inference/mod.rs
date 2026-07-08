//! Inference layer: the [`Detector`] abstraction and its YOLO implementation.

pub mod yolo;

pub use yolo::YoloDetector;

use crate::error::Result;
use crate::model::Detection;

/// A pluggable object detector.
///
/// Keeping detection behind a trait lets us swap the backend (a different model,
/// a remote service, a mock in tests) without touching the API layer.
pub trait Detector: Send + Sync {
    /// Run detection on encoded image bytes (JPEG/PNG) and return the objects found.
    async fn detect(&self, image: &[u8]) -> Result<Vec<Detection>>;
}

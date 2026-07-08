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
    //
    // We call `detect` on a concrete type, so the returned future's `Send`-ness
    // is inferred at the call site; no explicit `+ Send` bound is required here.
    #[allow(async_fn_in_trait)]
    async fn detect(&self, image: &[u8]) -> Result<Vec<Detection>>;
}

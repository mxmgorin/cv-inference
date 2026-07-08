//! Data model for detection results (serialised to JSON in the API response).

use serde::Serialize;

/// Axis-aligned bounding box in **original image pixel coordinates**.
///
/// `x`/`y` is the top-left corner; `width`/`height` are the box dimensions.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// A single detected object.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

/// Top-level response body for `POST /detect`.
#[derive(Debug, Clone, Serialize)]
pub struct DetectResponse {
    pub objects: Vec<Detection>,
}

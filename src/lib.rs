//! CV Inference Service — library crate.
//!
//! Exposes the building blocks (config, error type, inference backend, data
//! model) so they can be reused by the server binary and by tools such as the
//! `annotate` CLI.

pub mod api;
pub mod config;
pub mod error;
pub mod inference;
pub mod model;

use std::sync::Arc;

use crate::inference::YoloDetector;

/// Shared application state injected into HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    pub detector: Arc<YoloDetector>,
}

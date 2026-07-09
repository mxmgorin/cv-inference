//! Application configuration loaded from a YAML file (see `config.yaml`).

use std::path::Path;

use serde::Deserialize;

use crate::error::AppError;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
}

impl Config {
    /// Load configuration from the given YAML file path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AppError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|e| {
            AppError::Config(format!(
                "failed to read config file {}: {e}",
                path.display()
            ))
        })?;
        let config: Config = serde_yaml::from_str(&raw)
            .map_err(|e| AppError::Config(format!("failed to parse config: {e}")))?;
        Ok(config)
    }
}

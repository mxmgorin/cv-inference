//! CV Inference Service — HTTP server binary.

use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use cv_inference::{AppState, api, config::Config, error::AppError, inference::YoloDetector};

#[tokio::main]
async fn main() -> Result<(), AppError> {
    init_tracing();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
    let config = Config::load(&config_path)?;
    tracing::info!(?config, "configuration loaded");

    // 1. Load ONNX model at startup.
    let detector = YoloDetector::new(
        &config.model.path,
        config.inference.confidence_threshold,
        config.inference.iou_threshold,
    )?;
    let state = AppState {
        detector: Arc::new(detector),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/detect", post(api::detect::detect))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", config.server.port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Wait for Ctrl+C (or SIGTERM in containers) for a clean shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received");
}

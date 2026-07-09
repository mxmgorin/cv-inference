//! `POST /detect` handler. Accepts either a `multipart/form-data` upload with a
//! `file` field, or a raw JPEG/PNG request body.

use axum::{
    Json,
    body::Bytes,
    extract::{FromRequest, Multipart, Request, State},
    http::header::CONTENT_TYPE,
};

use crate::AppState;
use crate::error::{AppError, Result};
use crate::inference::Detector;
use crate::model::DetectResponse;

pub async fn detect(
    State(state): State<AppState>,
    request: Request,
) -> Result<Json<DetectResponse>> {
    tracing::info!("POST /detect");

    let content_type = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();

    let image_bytes = if content_type.starts_with("multipart/form-data") {
        extract_multipart_file(request).await?
    } else {
        Bytes::from_request(request, &())
            .await
            .map_err(|e| AppError::BadRequest(format!("failed to read request body: {e}")))?
            .to_vec()
    };

    if image_bytes.is_empty() {
        return Err(AppError::BadRequest("empty image payload".into()));
    }

    let objects = state.detector.detect(&image_bytes).await?;
    tracing::info!("objects={}", objects.len());

    Ok(Json(DetectResponse { objects }))
}

/// Pull the `file` field out of a multipart form.
async fn extract_multipart_file(request: Request) -> Result<Vec<u8>> {
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|e| AppError::BadRequest(format!("invalid multipart form: {e}")))?;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("failed to read multipart field: {e}")))?
    {
        if field.name() == Some("file") {
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(format!("failed to read file bytes: {e}")))?;
            return Ok(data.to_vec());
        }
    }

    Err(AppError::BadRequest(
        "no `file` field found in multipart form".into(),
    ))
}

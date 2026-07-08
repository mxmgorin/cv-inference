//! `annotate` — a small demo CLI that runs detection on an image and writes an
//! annotated copy with bounding boxes and labels drawn on it.
//!
//! Build/run with the `draw` feature:
//!
//! ```bash
//! cargo run --release --features draw --bin annotate -- input.jpg output.jpg
//! ```

use std::path::{Path, PathBuf};

use ab_glyph::{FontRef, PxScale};
use image::Rgba;
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;

use cv_inference::config::Config;
use cv_inference::inference::{Detector, YoloDetector};
use cv_inference::model::Detection;

/// A small vivid palette; boxes are coloured by class id.
const PALETTE: [Rgba<u8>; 8] = [
    Rgba([239, 68, 68, 255]),   // red
    Rgba([34, 197, 94, 255]),   // green
    Rgba([59, 130, 246, 255]),  // blue
    Rgba([234, 179, 8, 255]),   // yellow
    Rgba([168, 85, 247, 255]),  // purple
    Rgba([236, 72, 153, 255]),  // pink
    Rgba([249, 115, 22, 255]),  // orange
    Rgba([20, 184, 166, 255]),  // teal
];

const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: annotate <input-image> [output-image]"))?;
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| default_output(&input));

    // Load the model via the same config the server uses.
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
    let config = Config::load(&config_path)?;
    let detector = YoloDetector::new(
        &config.model.path,
        config.inference.confidence_threshold,
        config.inference.iou_threshold,
    )?;

    let bytes = std::fs::read(&input)?;
    let detections = detector.detect(&bytes).await?;
    println!("{} object(s) detected in {}", detections.len(), input.display());

    let mut img = image::load_from_memory(&bytes)?.to_rgba8();
    let font = FontRef::try_from_slice(include_bytes!("../../assets/DejaVuSans-Bold.ttf"))
        .map_err(|e| anyhow::anyhow!("failed to load bundled font: {e}"))?;

    let long_side = img.width().max(img.height()) as f32;
    let thickness = (long_side / 400.0).round().max(2.0) as i32;
    let font_px = (img.height() as f32 * 0.022).max(16.0);
    let scale = PxScale::from(font_px);

    for det in &detections {
        let color = PALETTE[hash_class(&det.class) % PALETTE.len()];
        draw_detection(&mut img, det, color, thickness, scale, &font);
        println!("  {:<14} {:>5.1}%", det.class, det.confidence * 100.0);
    }

    // Drop the alpha channel so the result can be saved as JPEG.
    let rgb = image::DynamicImage::ImageRgba8(img).into_rgb8();
    rgb.save(&output)?;
    println!("annotated image saved to {}", output.display());
    Ok(())
}

fn draw_detection(
    img: &mut image::RgbaImage,
    det: &Detection,
    color: Rgba<u8>,
    thickness: i32,
    scale: PxScale,
    font: &FontRef<'_>,
) {
    let x = det.bbox.x.round() as i32;
    let y = det.bbox.y.round() as i32;
    let w = det.bbox.width.round().max(1.0) as i32;
    let h = det.bbox.height.round().max(1.0) as i32;

    // Box outline (nested rectangles for thickness).
    for t in 0..thickness {
        let rect = Rect::at(x - t, y - t).of_size((w + 2 * t) as u32, (h + 2 * t) as u32);
        draw_hollow_rect_mut(img, rect, color);
    }

    // Label with a filled background for readability.
    let label = format!("{} {:.0}%", det.class, det.confidence * 100.0);
    let (tw, th) = text_size(scale, font, &label);
    let pad = 4_i32;
    let box_w = tw as i32 + pad * 2;
    let box_h = th as i32 + pad * 2;
    let ly = if y - box_h >= 0 { y - box_h } else { y };

    draw_filled_rect_mut(img, Rect::at(x, ly).of_size(box_w as u32, box_h as u32), color);
    draw_text_mut(img, WHITE, x + pad, ly + pad, scale, font, &label);
}

/// Stable per-class colour index (so the same class always gets the same colour).
fn hash_class(class: &str) -> usize {
    class.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize))
}

fn default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{stem}_annotated.jpg"))
}

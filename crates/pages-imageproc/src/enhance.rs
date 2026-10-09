use crate::shadows::remove_shadows;
use image::{GrayImage, Rgb, RgbImage};
use rayon::prelude::*;

/// Convert an RGB image to a Grayscale image using standard luminance weights
pub fn to_grayscale(img: &RgbImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut gray = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x, y);
            // ITU-R BT.601 luminance weights: Y = 0.299 R + 0.587 G + 0.114 B
            let luma = ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8;
            gray.put_pixel(x, y, image::Luma([luma]));
        }
    }
    gray
}

/// High-pass unsharp mask for razor-sharp character and ink edges
pub fn unsharp_mask(img: &RgbImage, amount: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    if w < 5 || h < 5 {
        return img.clone();
    }
    let mut out = img.clone();

    out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
        if y == 0 || y == h - 1 {
            return;
        }
        for (x, _, pixel) in row {
            if x == 0 || x == w - 1 {
                continue;
            }
            let orig = img.get_pixel(x, y);

            for c in 0..3 {
                let sum = (img.get_pixel(x - 1, y - 1)[c] as i32)
                    + (img.get_pixel(x, y - 1)[c] as i32 * 2)
                    + (img.get_pixel(x + 1, y - 1)[c] as i32)
                    + (img.get_pixel(x - 1, y)[c] as i32 * 2)
                    + (orig[c] as i32 * 4)
                    + (img.get_pixel(x + 1, y)[c] as i32 * 2)
                    + (img.get_pixel(x - 1, y + 1)[c] as i32)
                    + (img.get_pixel(x, y + 1)[c] as i32 * 2)
                    + (img.get_pixel(x + 1, y + 1)[c] as i32);
                let blur = sum >> 4;
                let diff = orig[c] as f32 - blur as f32;
                let sharp = (orig[c] as f32 + diff * amount).clamp(0.0, 255.0).round() as u8;
                pixel[c] = sharp;
            }
        }
    });

    out
}

/// CamScanner-style "Magic Pro / Magic Color" signature filter:
/// - Eliminates shadows, creases, and lighting falloff into pure paper white (#FFFFFF)
/// - Deepens handwritten and printed text into rich, crisp black
/// - Vibrantly preserves and boosts colored pen marks (teacher's red marks, blue pens, stamps)
/// - Unsharp masking delivers laser-sharp character clarity
pub fn enhance_magic_color(img: &RgbImage) -> RgbImage {
    let (orig_w, orig_h) = img.dimensions();
    if orig_w < 10 || orig_h < 10 {
        return img.clone();
    }

    // Step 1: Illumination normalization via background field division
    let divided = remove_shadows(img);

    let mut out = RgbImage::new(orig_w, orig_h);

    out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
        for (x, _, pixel) in row {
            let p = divided.get_pixel(x, y);
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;

            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let chroma = max_c - min_c;

            if chroma > 22.0 {
                // Colored ink (e.g. Teacher's red notes, blue signatures, purple stamps)
                // Preserve hue, boost saturation so ink is vivid against white paper
                let mean = (r + g + b) / 3.0;
                let sat_boost = 1.35;
                let r_b = (mean + (r - mean) * sat_boost).clamp(0.0, 255.0);
                let g_b = (mean + (g - mean) * sat_boost).clamp(0.0, 255.0);
                let b_b = (mean + (b - mean) * sat_boost).clamp(0.0, 255.0);

                let factor = if luma > 175.0 { 0.88 } else { 0.95 };
                *pixel = Rgb([
                    (r_b * factor).round() as u8,
                    (g_b * factor).round() as u8,
                    (b_b * factor).round() as u8,
                ]);
            } else if luma >= 212.0 {
                // Paper background: cleanly drive to pure white 255
                if luma >= 225.0 {
                    *pixel = Rgb([255, 255, 255]);
                } else {
                    let t = (luma - 212.0) / 13.0;
                    let w = ((1.0 - t) * luma + t * 255.0).round().clamp(0.0, 255.0) as u8;
                    *pixel = Rgb([w, w, w]);
                }
            } else {
                // Black / pencil / pen ink: apply CamScanner power-law deepening curve
                let t = (luma / 212.0).clamp(0.0, 1.0);
                let ink = (255.0 * t.powf(1.68)).round().clamp(0.0, 255.0) as u8;
                *pixel = Rgb([ink, ink, ink]);
            }
        }
    });

    unsharp_mask(&out, 0.35)
}

/// CamScanner-style "Enhance" filter:
/// High contrast with unsharp masking and moderate paper whitening
pub fn enhance_contrast_sharp(img: &RgbImage) -> RgbImage {
    let divided = remove_shadows(img);
    let mut out = RgbImage::new(img.width(), img.height());

    out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
        for (x, _, pixel) in row {
            let p = divided.get_pixel(x, y);
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                let enhanced = if v > 0.85 {
                    1.0
                } else {
                    (v / 0.85).powf(1.4) * 0.95
                };
                rgb[c] = (enhanced * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            *pixel = Rgb(rgb);
        }
    });

    unsharp_mask(&out, 0.40)
}

/// CamScanner-style "Lighten" filter:
/// Lifts shadows and brightens background softly while preserving light diagrams and pencil tones
pub fn enhance_lighten(img: &RgbImage) -> RgbImage {
    let divided = remove_shadows(img);
    let mut out = RgbImage::new(img.width(), img.height());

    out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
        for (x, _, pixel) in row {
            let p = divided.get_pixel(x, y);
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                let lightened = if v > 0.78 {
                    1.0
                } else {
                    (v / 0.78).powf(1.15) * 0.95
                };
                rgb[c] = (lightened * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            *pixel = Rgb(rgb);
        }
    });

    unsharp_mask(&out, 0.25)
}

/// CamScanner-style "No Shadow" filter:
/// Flattens lighting gradients and removes shadows while keeping natural paper texture and colors
pub fn enhance_no_shadow(img: &RgbImage) -> RgbImage {
    let divided = remove_shadows(img);
    unsharp_mask(&divided, 0.20)
}

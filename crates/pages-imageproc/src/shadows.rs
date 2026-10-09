use image::{Rgb, RgbImage};
use rayon::prelude::*;

/// 1D horizontal min-max / dilation filter
fn dilate_horizontal(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height];
    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let mut max_val = 0u8;
            for k in start..=end {
                let v = src[row_start + k];
                if v > max_val {
                    max_val = v;
                    if max_val == 255 {
                        break;
                    }
                }
            }
            dst[row_start + x] = max_val;
        }
    }
    dst
}

/// 1D vertical dilation filter
fn dilate_vertical(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height];
    for x in 0..width {
        for y in 0..height {
            let start = y.saturating_sub(radius);
            let end = (y + radius).min(height - 1);
            let mut max_val = 0u8;
            for k in start..=end {
                let v = src[k * width + x];
                if v > max_val {
                    max_val = v;
                    if max_val == 255 {
                        break;
                    }
                }
            }
            dst[y * width + x] = max_val;
        }
    }
    dst
}

/// Fast separable 2D box blur
pub fn box_blur(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut temp = vec![0u32; width * height];
    let mut dst = vec![0u8; width * height];

    // Horizontal pass
    for y in 0..height {
        let row_start = y * width;
        let mut sum: u32 = 0;
        let mut count: u32 = 0;

        for x in 0..width.min(radius + 1) {
            sum += src[row_start + x] as u32;
            count += 1;
        }

        for x in 0..width {
            if x + radius < width {
                sum += src[row_start + x + radius] as u32;
                count += 1;
            }
            if x > radius {
                sum -= src[row_start + x - radius - 1] as u32;
                count -= 1;
            }
            temp[row_start + x] = sum / count;
        }
    }

    // Vertical pass
    for x in 0..width {
        let mut sum: u32 = 0;
        let mut count: u32 = 0;

        for y in 0..height.min(radius + 1) {
            sum += temp[y * width + x];
            count += 1;
        }

        for y in 0..height {
            if y + radius < height {
                sum += temp[(y + radius) * width + x];
                count += 1;
            }
            if y > radius {
                sum -= temp[(y - radius - 1) * width + x];
                count -= 1;
            }
            dst[y * width + x] = (sum / count) as u8;
        }
    }

    dst
}

#[inline(always)]
pub fn sample_bg_bilinear(bg: &[u8], sw: usize, sh: usize, u: f32, v: f32) -> f32 {
    let x = (u * (sw - 1) as f32).clamp(0.0, (sw - 1) as f32);
    let y = (v * (sh - 1) as f32).clamp(0.0, (sh - 1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(sw - 1);
    let y1 = (y0 + 1).min(sh - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = bg[y0 * sw + x0] as f32;
    let p10 = bg[y0 * sw + x1] as f32;
    let p01 = bg[y1 * sw + x0] as f32;
    let p11 = bg[y1 * sw + x1] as f32;

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bot = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bot * fy
}

/// Remove phone camera shadows, page creases, and uneven illumination across an RGB image
pub fn remove_shadows(img: &RgbImage) -> RgbImage {
    let (orig_w, orig_h) = img.dimensions();
    if orig_w < 20 || orig_h < 20 {
        return img.clone();
    }

    // Downscale for fast, robust background estimation (removes all text and pencil strokes)
    let max_dim = orig_w.max(orig_h);
    let scale = (450.0 / max_dim as f32).min(1.0);
    let sw = ((orig_w as f32 * scale).round() as u32).max(10);
    let sh = ((orig_h as f32 * scale).round() as u32).max(10);

    let small = image::imageops::resize(img, sw, sh, image::imageops::FilterType::Triangle);
    let sw_usize = sw as usize;
    let sh_usize = sh as usize;

    let mut r_small = vec![0u8; sw_usize * sh_usize];
    let mut g_small = vec![0u8; sw_usize * sh_usize];
    let mut b_small = vec![0u8; sw_usize * sh_usize];

    for y in 0..sh_usize {
        let row = y * sw_usize;
        for x in 0..sw_usize {
            let p = small.get_pixel(x as u32, y as u32);
            r_small[row + x] = p[0];
            g_small[row + x] = p[1];
            b_small[row + x] = p[2];
        }
    }

    // Process channels in parallel with Rayon
    let chans = vec![r_small, g_small, b_small];
    let mut bg_chans: Vec<Vec<u8>> = chans
        .into_par_iter()
        .map(|chan| {
            // Dilate with radius 7 on 450px grid (equivalent to ~50px on full image, obliterating all text)
            let d_h = dilate_horizontal(&chan, sw_usize, sh_usize, 7);
            let d_v = dilate_vertical(&d_h, sw_usize, sh_usize, 7);
            // Smooth with radius 15
            box_blur(&d_v, sw_usize, sh_usize, 15)
        })
        .collect();

    let bg_b = bg_chans.pop().unwrap();
    let bg_g = bg_chans.pop().unwrap();
    let bg_r = bg_chans.pop().unwrap();

    let mut out = RgbImage::new(orig_w, orig_h);

    // Parallelize row processing with Rayon
    out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
        let v = y as f32 / (orig_h - 1).max(1) as f32;
        for (x, _, pixel) in row {
            let u = x as f32 / (orig_w - 1).max(1) as f32;
            let orig_p = img.get_pixel(x, y);

            let r_bg = sample_bg_bilinear(&bg_r, sw_usize, sh_usize, u, v).max(20.0);
            let g_bg = sample_bg_bilinear(&bg_g, sw_usize, sh_usize, u, v).max(20.0);
            let b_bg = sample_bg_bilinear(&bg_b, sw_usize, sh_usize, u, v).max(20.0);

            // Background division
            let r_div = ((orig_p[0] as f32 / r_bg) * 255.0).clamp(0.0, 255.0) as u8;
            let g_div = ((orig_p[1] as f32 / g_bg) * 255.0).clamp(0.0, 255.0) as u8;
            let b_div = ((orig_p[2] as f32 / b_bg) * 255.0).clamp(0.0, 255.0) as u8;

            *pixel = Rgb([r_div, g_div, b_div]);
        }
    });

    out
}

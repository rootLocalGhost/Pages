use crate::binary_image::BinaryImage;
use image::GrayImage;

/// Computes optimal global threshold using Otsu's method (N. Otsu, IEEE TSMC, 1979).
pub fn otsu_threshold(gray: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for pixel in gray.pixels() {
        hist[pixel.0[0] as usize] += 1;
    }

    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 128;
    }

    let mut sum_all: f64 = 0.0;
    for (t, &count) in hist.iter().enumerate() {
        sum_all += (t as f64) * (count as f64);
    }

    let mut weight_bg: u64 = 0;
    let mut sum_bg: f64 = 0.0;
    let mut max_variance = 0.0;
    let mut best_threshold = 128u8;

    for (t, &count) in hist.iter().enumerate() {
        weight_bg += count;
        if weight_bg == 0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0 {
            break;
        }

        sum_bg += (t as f64) * (count as f64);
        let mean_bg = sum_bg / weight_bg as f64;
        let mean_fg = (sum_all - sum_bg) / weight_fg as f64;

        let between_variance =
            (weight_bg as f64) * (weight_fg as f64) * (mean_bg - mean_fg) * (mean_bg - mean_fg);

        if between_variance > max_variance {
            max_variance = between_variance;
            best_threshold = t as u8;
        }
    }

    best_threshold
}

/// Binarize a grayscale image using Otsu's method.
pub fn binarize_otsu(gray: &GrayImage, delta: i32) -> BinaryImage {
    let base_t = otsu_threshold(gray) as i32;
    let threshold = (base_t + delta).clamp(0, 255) as u8;
    BinaryImage::from_gray_image(gray, threshold)
}

/// Sauvola's adaptive document image binarization method (J. Sauvola & M. Pietikäinen, 2000).
///
/// `T(x, y) = mean * (1.0 + k * (std_dev / 128.0 - 1.0))`
/// Uses 2D integral images for O(1) local mean and standard deviation computation.
pub fn binarize_sauvola(
    gray: &GrayImage,
    window_radius: u32,
    k: f64,
    delta: i32,
) -> BinaryImage {
    let (width, height) = gray.dimensions();
    let w = width as usize;
    let h = height as usize;

    // Integral images for sum and sum of squares
    // Dimensions are (w + 1) * (h + 1)
    let stride = w + 1;
    let mut sum_table = vec![0u64; (w + 1) * (h + 1)];
    let mut sq_sum_table = vec![0u64; (w + 1) * (h + 1)];

    for y in 0..h {
        let mut row_sum = 0u64;
        let mut row_sq_sum = 0u64;
        for x in 0..w {
            let val = gray.get_pixel(x as u32, y as u32).0[0] as u64;
            row_sum += val;
            row_sq_sum += val * val;

            let prev_idx = y * stride + (x + 1);
            let curr_idx = (y + 1) * stride + (x + 1);

            sum_table[curr_idx] = sum_table[prev_idx] + row_sum;
            sq_sum_table[curr_idx] = sq_sum_table[prev_idx] + row_sq_sum;
        }
    }

    let mut bin = BinaryImage::new(width, height);
    let r = window_radius as usize;

    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);

        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);

            let area = ((x1 - x0) * (y1 - y0)) as f64;

            let sum = (sum_table[y1 * stride + x1] + sum_table[y0 * stride + x0]) as f64
                - (sum_table[y0 * stride + x1] + sum_table[y1 * stride + x0]) as f64;

            let sq_sum = (sq_sum_table[y1 * stride + x1] + sq_sum_table[y0 * stride + x0]) as f64
                - (sq_sum_table[y0 * stride + x1] + sq_sum_table[y1 * stride + x0]) as f64;

            let mean = sum / area;
            let variance = (sq_sum / area) - (mean * mean);
            let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

            let threshold_f = mean * (1.0 + k * (std_dev / 128.0 - 1.0)) + delta as f64;
            let threshold = threshold_f.clamp(0.0, 255.0) as u8;

            let pixel_val = gray.get_pixel(x as u32, y as u32).0[0];
            if pixel_val <= threshold {
                bin.set_pixel(x as u32, y as u32, true);
            }
        }
    }

    bin
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_otsu_threshold() {
        let mut gray = GrayImage::new(100, 100);
        // Half white, half black
        for y in 0..100 {
            for x in 0..100 {
                let v = if x < 50 { 50 } else { 200 };
                gray.put_pixel(x, y, Luma([v]));
            }
        }

        let t = otsu_threshold(&gray);
        assert!(t >= 50 && t <= 200);
        let bin = binarize_otsu(&gray, 0);
        assert_eq!(bin.count_black_pixels(), 5000);
    }

    #[test]
    fn test_sauvola_binarize() {
        let mut gray = GrayImage::new(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                let v = if x == 20 && y == 20 { 20 } else { 230 };
                gray.put_pixel(x, y, Luma([v]));
            }
        }

        let bin = binarize_sauvola(&gray, 5, 0.2, 0);
        assert!(bin.get_pixel(20, 20));
        assert!(!bin.get_pixel(0, 0));
    }
}

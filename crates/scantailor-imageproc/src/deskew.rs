use crate::binary_image::BinaryImage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewResult {
    /// Skew angle in degrees.
    /// Positive means clockwise tilt (needs counter-clockwise correction).
    pub angle_deg: f64,
    /// Confidence metric of the detection.
    pub confidence: f64,
}

/// Estimates skew angle of a document image using projection profile variance after vertical shearing.
///
/// Ported from Scan Tailor's `SkewFinder` (Joseph Artsimovich).
pub struct SkewFinder {
    pub max_angle_deg: f64,
    pub coarse_step_deg: f64,
    pub fine_step_deg: f64,
}

impl Default for SkewFinder {
    fn default() -> Self {
        Self {
            max_angle_deg: 10.0,
            coarse_step_deg: 0.5,
            fine_step_deg: 0.05,
        }
    }
}

impl SkewFinder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates the variance of adjacent row differences when sheared vertically at `angle_deg`.
    fn evaluate_angle_score(&self, img: &BinaryImage, angle_deg: f64) -> f64 {
        let rad = angle_deg.to_radians();
        let tg = rad.tan();
        let w = img.width as usize;
        let h = img.height as usize;
        let x_center = w as f64 * 0.5;

        let mut row_black_counts = vec![0u32; h];

        // Sample columns with step 2 for performance
        let step_x = 2;
        for x in (0..w).step_by(step_x) {
            let dy = ((x as f64 - x_center) * tg).round() as i32;
            for y in 0..h {
                let src_y = y as i32 - dy;
                if src_y >= 0 && src_y < h as i32 && img.get_pixel(x as u32, src_y as u32) {
                    row_black_counts[y] += 1;
                }
            }
        }

        // Sum squared differences between adjacent lines (matching Scan Tailor's calcScore)
        let mut score = 0.0;
        for y in 1..h {
            let diff = row_black_counts[y] as f64 - row_black_counts[y - 1] as f64;
            score += diff * diff;
        }

        score
    }

    /// Finds the skew angle of the image.
    pub fn find_skew(&self, img: &BinaryImage) -> SkewResult {
        if img.count_black_pixels() < 20 {
            return SkewResult {
                angle_deg: 0.0,
                confidence: 0.0,
            };
        }

        // 1. Coarse search with plateau centering
        let mut best_score = -1.0;
        let mut best_start = -self.max_angle_deg;
        let mut best_end = -self.max_angle_deg;
        let mut sum_scores = 0.0;
        let mut num_scores = 0;

        let mut angle = -self.max_angle_deg;
        while angle <= self.max_angle_deg {
            let score = self.evaluate_angle_score(img, angle);
            sum_scores += score;
            num_scores += 1;

            if score > best_score + 1e-4 {
                best_score = score;
                best_start = angle;
                best_end = angle;
            } else if (score - best_score).abs() <= 1e-4 {
                best_end = angle;
            }
            angle += self.coarse_step_deg;
        }

        let best_coarse_angle = (best_start + best_end) * 0.5;

        // 2. Fine search around the best coarse angle
        let fine_min = (best_coarse_angle - self.coarse_step_deg).max(-self.max_angle_deg);
        let fine_max = (best_coarse_angle + self.coarse_step_deg).min(self.max_angle_deg);

        let mut fine_best_score = -1.0;
        let mut fine_start = fine_min;
        let mut fine_end = fine_min;

        let mut fine_angle = fine_min;
        while fine_angle <= fine_max {
            let score = self.evaluate_angle_score(img, fine_angle);
            if score > fine_best_score + 1e-4 {
                fine_best_score = score;
                fine_start = fine_angle;
                fine_end = fine_angle;
            } else if (score - fine_best_score).abs() <= 1e-4 {
                fine_end = fine_angle;
            }
            fine_angle += self.fine_step_deg;
        }

        let final_angle = (fine_start + fine_end) * 0.5;

        let mean_score = if num_scores > 0 {
            sum_scores / num_scores as f64
        } else {
            1.0
        };

        let confidence = if mean_score > 0.0 {
            best_score / mean_score
        } else {
            0.0
        };

        SkewResult {
            angle_deg: final_angle,
            confidence,
        }
    }
}

/// Rotates/deskews a `BinaryImage` around its center by `angle_deg`.
pub fn rotate_binary_image(img: &BinaryImage, angle_deg: f64) -> BinaryImage {
    if angle_deg.abs() < 1e-4 {
        return img.clone();
    }

    let rad = -angle_deg.to_radians(); // Inverse rotation for backward mapping
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let cx = img.width as f64 * 0.5;
    let cy = img.height as f64 * 0.5;

    let mut out = BinaryImage::new(img.width, img.height);

    for y in 0..img.height {
        let dy = y as f64 - cy;
        for x in 0..img.width {
            let dx = x as f64 - cx;
            let src_x = (dx * cos_a - dy * sin_a + cx).round() as i32;
            let src_y = (dx * sin_a + dy * cos_a + cy).round() as i32;

            if src_x >= 0
                && src_x < img.width as i32
                && src_y >= 0
                && src_y < img.height as i32
                && img.get_pixel(src_x as u32, src_y as u32)
            {
                out.set_pixel(x, y, true);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deskew_horizontal_lines() {
        let mut img = BinaryImage::new(300, 300);
        // Draw 5 horizontal text lines
        for y in [50, 100, 150, 200, 250] {
            for x in 30..270 {
                img.set_pixel(x, y, true);
            }
        }

        let finder = SkewFinder::new();
        let skew = finder.find_skew(&img);

        assert!(skew.angle_deg.abs() < 0.1);
    }

    #[test]
    fn test_deskew_tilted_lines() {
        let mut img = BinaryImage::new(300, 300);
        // Draw lines intentionally tilted by -3.0 degrees
        let tilt_deg = -3.0f64;
        let tilt_rad = tilt_deg.to_radians();
        let tan_tilt = tilt_rad.tan();
        let cx = 150.0;

        for base_y in [70.0, 120.0, 170.0, 220.0] {
            for x in 40..260 {
                let y = (base_y + (x as f64 - cx) * tan_tilt).round() as i32;
                if y >= 0 && y < 300 {
                    img.set_pixel(x, y as u32, true);
                }
            }
        }

        let finder = SkewFinder::new();
        let skew = finder.find_skew(&img);

        // Detected angle should match tilt magnitude
        assert!((skew.angle_deg.abs() - 3.0).abs() < 0.2);

        // Rotating by the detected angle should restore horizontal alignment
        let deskewed = rotate_binary_image(&img, skew.angle_deg);
        let check_skew = finder.find_skew(&deskewed);
        assert!(check_skew.angle_deg.abs() < 0.2);
    }
}

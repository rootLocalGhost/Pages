use crate::binary_image::BinaryImage;
use pages_math::Rect2i;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageLayoutMode {
    SinglePage,
    TwoPages,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PageSplitResult {
    pub mode: PageLayoutMode,
    /// X coordinate of the split line (if TwoPages)
    pub split_x: Option<u32>,
    pub left_page_rect: Rect2i,
    pub right_page_rect: Option<Rect2i>,
}

/// Detects the page gutter/spine split line in a book scan.
pub struct PageSplitDetector {
    /// Search margin from left/right borders (e.g. 0.25 means search in 25%..75% width)
    pub central_band_ratio: f64,
}

impl Default for PageSplitDetector {
    fn default() -> Self {
        Self {
            central_band_ratio: 0.30,
        }
    }
}

impl PageSplitDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates vertical projection profile across X columns to find the spine.
    pub fn detect_split(&self, img: &BinaryImage) -> PageSplitResult {
        let w = img.width;
        let h = img.height;

        let min_x = (w as f64 * (0.5 - self.central_band_ratio * 0.5)).round() as u32;
        let max_x = (w as f64 * (0.5 + self.central_band_ratio * 0.5)).round() as u32;

        let mut col_ink = vec![0u32; w as usize];
        for x in 0..w {
            let mut ink = 0u32;
            for y in 0..h {
                if img.get_pixel(x, y) {
                    ink += 1;
                }
            }
            col_ink[x as usize] = ink;
        }

        // Search for column with minimum ink in the central zone (with 5-column smoothing)
        let mut min_ink_smoothed = u32::MAX;
        let mut best_split_x = w / 2;

        let smooth_r = 3u32;
        for x in min_x..=max_x {
            let x0 = x.saturating_sub(smooth_r) as usize;
            let x1 = (x + smooth_r).min(w - 1) as usize;
            let smoothed: u32 = col_ink[x0..=x1].iter().sum();

            if smoothed < min_ink_smoothed {
                min_ink_smoothed = smoothed;
                best_split_x = x;
            }
        }

        // Decide if this is a two-page layout:
        // Both left and right halves must contain substantial ink.
        let left_ink: u32 = col_ink[0..best_split_x as usize].iter().sum();
        let right_ink: u32 = col_ink[best_split_x as usize..w as usize].iter().sum();

        let total_ink = left_ink + right_ink;
        let is_two_page = total_ink > 200
            && left_ink as f64 > total_ink as f64 * 0.15
            && right_ink as f64 > total_ink as f64 * 0.15;

        if is_two_page {
            PageSplitResult {
                mode: PageLayoutMode::TwoPages,
                split_x: Some(best_split_x),
                left_page_rect: Rect2i::new(0, 0, best_split_x as i32, h as i32),
                right_page_rect: Some(Rect2i::new(
                    best_split_x as i32,
                    0,
                    (w - best_split_x) as i32,
                    h as i32,
                )),
            }
        } else {
            PageSplitResult {
                mode: PageLayoutMode::SinglePage,
                split_x: None,
                left_page_rect: Rect2i::new(0, 0, w as i32, h as i32),
                right_page_rect: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_detector_two_pages() {
        let mut img = BinaryImage::new(200, 100);

        // Left page text (columns 20..80)
        for y in 20..80 {
            for x in 20..80 {
                if (x + y) % 5 == 0 {
                    img.set_pixel(x, y, true);
                }
            }
        }

        // Right page text (columns 120..180)
        for y in 20..80 {
            for x in 120..180 {
                if (x + y) % 5 == 0 {
                    img.set_pixel(x, y, true);
                }
            }
        }

        let detector = PageSplitDetector::new();
        let split = detector.detect_split(&img);

        assert_eq!(split.mode, PageLayoutMode::TwoPages);
        let sx = split.split_x.unwrap();
        // The split should land cleanly in the middle whitespace gutter (80..120)
        assert!((80..=120).contains(&sx));
    }
}

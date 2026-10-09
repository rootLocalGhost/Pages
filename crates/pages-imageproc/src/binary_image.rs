use image::{GrayImage, Luma};

/// An efficient 1-bit per pixel binary raster image.
///
/// Following Scan Tailor's convention:
/// - `1` (true) represents **black** (ink / foreground).
/// - `0` (false) represents **white** (paper / background).
///
/// Rows are packed into 32-bit words (`u32`) with MSB first (bit 31 is the leftmost pixel of a word).
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryImage {
    pub width: u32,
    pub height: u32,
    pub words_per_row: usize,
    pub data: Vec<u32>,
}

impl BinaryImage {
    /// Creates a new binary image filled with white (zeros).
    pub fn new(width: u32, height: u32) -> Self {
        let words_per_row = (width as usize).div_ceil(32);
        let data = vec![0u32; words_per_row * height as usize];
        Self {
            width,
            height,
            words_per_row,
            data,
        }
    }

    /// Creates a new binary image filled with specified value (`true` = black, `false` = white).
    pub fn new_filled(width: u32, height: u32, black: bool) -> Self {
        let words_per_row = (width as usize).div_ceil(32);
        let fill_word = if black { 0xFFFF_FFFF } else { 0x0000_0000 };
        let data = vec![fill_word; words_per_row * height as usize];
        let mut img = Self {
            width,
            height,
            words_per_row,
            data,
        };
        if black {
            img.clear_padding_bits();
        }
        img
    }

    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let word_idx = y as usize * self.words_per_row + (x as usize / 32);
        let bit_idx = 31 - (x % 32);
        (self.data[word_idx] & (1 << bit_idx)) != 0
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, black: bool) {
        if x >= self.width || y >= self.height {
            return;
        }
        let word_idx = y as usize * self.words_per_row + (x as usize / 32);
        let bit_idx = 31 - (x % 32);
        let mask = 1 << bit_idx;
        if black {
            self.data[word_idx] |= mask;
        } else {
            self.data[word_idx] &= !mask;
        }
    }

    #[inline]
    pub fn row(&self, y: u32) -> &[u32] {
        let start = y as usize * self.words_per_row;
        &self.data[start..start + self.words_per_row]
    }

    #[inline]
    pub fn row_mut(&mut self, y: u32) -> &mut [u32] {
        let start = y as usize * self.words_per_row;
        &mut self.data[start..start + self.words_per_row]
    }

    /// Clear unused padding bits at the end of every row so they don't skew counts or bitwise ops.
    pub fn clear_padding_bits(&mut self) {
        let remainder = self.width % 32;
        if remainder != 0 {
            let mask = 0xFFFF_FFFFu32 << (32 - remainder);
            for y in 0..self.height {
                let last_word_idx = y as usize * self.words_per_row + self.words_per_row - 1;
                self.data[last_word_idx] &= mask;
            }
        }
    }

    /// Counts total number of black pixels in the image.
    pub fn count_black_pixels(&self) -> usize {
        self.data.iter().map(|&w| w.count_ones() as usize).sum()
    }

    /// Counts total number of white pixels in the image.
    pub fn count_white_pixels(&self) -> usize {
        (self.width as usize * self.height as usize) - self.count_black_pixels()
    }

    /// Inverts all pixels in the image (black becomes white, white becomes black).
    pub fn invert(&mut self) {
        for word in &mut self.data {
            *word = !*word;
        }
        self.clear_padding_bits();
    }

    /// Returns a new inverted binary image.
    pub fn inverted(&self) -> Self {
        let mut copy = self.clone();
        copy.invert();
        copy
    }

    /// Converts to standard 8-bit grayscale image (`image::GrayImage`),
    /// where white is 255 and black is 0.
    pub fn to_gray_image(&self) -> GrayImage {
        let mut gray = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let val = if self.get_pixel(x, y) { 0u8 } else { 255u8 };
                gray.put_pixel(x, y, Luma([val]));
            }
        }
        gray
    }

    /// Creates a `BinaryImage` from an 8-bit `GrayImage` using a fixed global threshold.
    /// Pixels <= threshold are considered black (foreground).
    pub fn from_gray_image(gray: &GrayImage, threshold: u8) -> Self {
        let (width, height) = gray.dimensions();
        let mut bin = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = gray.get_pixel(x, y).0[0];
                if pixel <= threshold {
                    bin.set_pixel(x, y, true);
                }
            }
        }
        bin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_image_basic() {
        let mut img = BinaryImage::new(40, 10);
        assert_eq!(img.words_per_row, 2);
        assert_eq!(img.count_black_pixels(), 0);

        img.set_pixel(0, 0, true);
        img.set_pixel(31, 0, true);
        img.set_pixel(32, 0, true);
        img.set_pixel(39, 0, true);

        assert!(img.get_pixel(0, 0));
        assert!(img.get_pixel(31, 0));
        assert!(img.get_pixel(32, 0));
        assert!(img.get_pixel(39, 0));
        assert!(!img.get_pixel(1, 0));

        assert_eq!(img.count_black_pixels(), 4);

        img.invert();
        assert!(!img.get_pixel(0, 0));
        assert!(img.get_pixel(1, 0));
        assert_eq!(img.count_black_pixels(), 40 * 10 - 4);
    }
}

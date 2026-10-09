use crate::binary_image::BinaryImage;

/// Structuring element for morphological filters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brick {
    pub half_width: u32,
    pub half_height: u32,
}

impl Brick {
    pub const fn new(half_width: u32, half_height: u32) -> Self {
        Self {
            half_width,
            half_height,
        }
    }

    pub const fn square(radius: u32) -> Self {
        Self::new(radius, radius)
    }
}

/// Binary morphological dilation (expands black/foreground regions).
///
/// In Scan Tailor, black = 1. Dilating black pixels corresponds to bitwise OR.
pub fn dilate(img: &BinaryImage, brick: Brick) -> BinaryImage {
    let mut temp = BinaryImage::new(img.width, img.height);

    // 1. Horizontal dilation
    let rx = brick.half_width as usize;
    if rx == 0 {
        temp.data.copy_from_slice(&img.data);
    } else {
        for y in 0..img.height {
            for x in 0..img.width {
                let x0 = x.saturating_sub(rx as u32);
                let x1 = (x + rx as u32).min(img.width - 1);
                let mut is_black = false;
                for kx in x0..=x1 {
                    if img.get_pixel(kx, y) {
                        is_black = true;
                        break;
                    }
                }
                if is_black {
                    temp.set_pixel(x, y, true);
                }
            }
        }
    }

    // 2. Vertical dilation
    let ry = brick.half_height;
    if ry == 0 {
        return temp;
    }

    let mut result = BinaryImage::new(img.width, img.height);
    for y in 0..img.height {
        let y0 = y.saturating_sub(ry);
        let y1 = (y + ry).min(img.height - 1);
        let row_out = result.row_mut(y);

        for ky in y0..=y1 {
            let row_in = temp.row(ky);
            for (out_w, &in_w) in row_out.iter_mut().zip(row_in.iter()) {
                *out_w |= in_w;
            }
        }
    }

    result.clear_padding_bits();
    result
}

/// Binary morphological erosion (shrinks black/foreground regions).
///
/// In Scan Tailor, black = 1. Eroding black pixels corresponds to bitwise AND.
pub fn erode(img: &BinaryImage, brick: Brick) -> BinaryImage {
    let mut temp = BinaryImage::new(img.width, img.height);

    // 1. Horizontal erosion
    let rx = brick.half_width as usize;
    if rx == 0 {
        temp.data.copy_from_slice(&img.data);
    } else {
        for y in 0..img.height {
            for x in 0..img.width {
                let x0 = x.saturating_sub(rx as u32);
                let x1 = (x + rx as u32).min(img.width - 1);
                let mut all_black = true;
                for kx in x0..=x1 {
                    if !img.get_pixel(kx, y) {
                        all_black = false;
                        break;
                    }
                }
                if all_black {
                    temp.set_pixel(x, y, true);
                }
            }
        }
    }

    // 2. Vertical erosion
    let ry = brick.half_height;
    if ry == 0 {
        return temp;
    }

    let mut result = BinaryImage::new_filled(img.width, img.height, true);
    for y in 0..img.height {
        let y0 = y.saturating_sub(ry);
        let y1 = (y + ry).min(img.height - 1);

        // If brick extends beyond image boundary, outer pixels are considered white (0)
        if y < ry || y + ry >= img.height {
            for x in 0..img.width {
                result.set_pixel(x, y, false);
            }
            continue;
        }

        let row_out = result.row_mut(y);
        for ky in y0..=y1 {
            let row_in = temp.row(ky);
            for (out_w, &in_w) in row_out.iter_mut().zip(row_in.iter()) {
                *out_w &= in_w;
            }
        }
    }

    result.clear_padding_bits();
    result
}

/// Morphological opening (erosion followed by dilation). Removes small noise specs.
pub fn open(img: &BinaryImage, brick: Brick) -> BinaryImage {
    dilate(&erode(img, brick), brick)
}

/// Morphological closing (dilation followed by erosion). Closes small gaps and holes.
pub fn close(img: &BinaryImage, brick: Brick) -> BinaryImage {
    erode(&dilate(img, brick), brick)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilation_erosion() {
        let mut img = BinaryImage::new(30, 30);
        img.set_pixel(15, 15, true);

        let dilated = dilate(&img, Brick::new(1, 1));
        assert_eq!(dilated.count_black_pixels(), 9);

        let eroded = erode(&dilated, Brick::new(1, 1));
        assert_eq!(eroded.count_black_pixels(), 1);
        assert!(eroded.get_pixel(15, 15));
    }
}

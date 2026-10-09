use crate::cylinder_model::CylindricalSurfaceModel;
use image::{GrayImage, Luma};
use pages_imageproc::BinaryImage;

/// Performs backward pixel remapping from warped image coordinates to flattened target coordinates.
pub struct DewarpRemapper;

impl DewarpRemapper {
    /// Dewarp a grayscale image using bilinear interpolation.
    pub fn dewarp_gray(src: &GrayImage, model: &CylindricalSurfaceModel) -> GrayImage {
        let (src_w, src_h) = src.dimensions();
        let dst_w = model.target_width;
        let dst_h = model.target_height;

        let mut dst = GrayImage::new(dst_w, dst_h);

        let inv_w = 1.0 / (dst_w.max(1) - 1) as f64;
        let inv_h = 1.0 / (dst_h.max(1) - 1) as f64;

        for dy in 0..dst_h {
            let v = dy as f64 * inv_h;
            for dx in 0..dst_w {
                let u = dx as f64 * inv_w;
                let src_pt = model.map_unrolled_to_source(u, v);

                let sx = src_pt.x;
                let sy = src_pt.y;

                if sx >= 0.0 && sx < (src_w - 1) as f64 && sy >= 0.0 && sy < (src_h - 1) as f64 {
                    let x0 = sx.floor() as u32;
                    let y0 = sy.floor() as u32;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    let fx = (sx - x0 as f64) as f32;
                    let fy = (sy - y0 as f64) as f32;

                    let p00 = src.get_pixel(x0, y0).0[0] as f32;
                    let p10 = src.get_pixel(x1, y0).0[0] as f32;
                    let p01 = src.get_pixel(x0, y1).0[0] as f32;
                    let p11 = src.get_pixel(x1, y1).0[0] as f32;

                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bottom = p01 * (1.0 - fx) + p11 * fx;
                    let val = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;

                    dst.put_pixel(dx, dy, Luma([val]));
                } else {
                    // Outside boundary is white
                    dst.put_pixel(dx, dy, Luma([255]));
                }
            }
        }

        dst
    }

    /// Dewarp a binary image using nearest-neighbor sampling.
    pub fn dewarp_binary(src: &BinaryImage, model: &CylindricalSurfaceModel) -> BinaryImage {
        let dst_w = model.target_width;
        let dst_h = model.target_height;
        let mut dst = BinaryImage::new(dst_w, dst_h);

        let inv_w = 1.0 / (dst_w.max(1) - 1) as f64;
        let inv_h = 1.0 / (dst_h.max(1) - 1) as f64;

        for dy in 0..dst_h {
            let v = dy as f64 * inv_h;
            for dx in 0..dst_w {
                let u = dx as f64 * inv_w;
                let src_pt = model.map_unrolled_to_source(u, v);

                let sx = src_pt.x.round() as i32;
                let sy = src_pt.y.round() as i32;

                if sx >= 0
                    && sx < src.width as i32
                    && sy >= 0
                    && sy < src.height as i32
                    && src.get_pixel(sx as u32, sy as u32)
                {
                    dst.set_pixel(dx, dy, true);
                }
            }
        }

        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::DewarpCurve;
    use pages_math::Point2d;

    #[test]
    fn test_dewarp_remapper() {
        let mut src = BinaryImage::new(100, 100);
        // Draw a straight horizontal line in middle
        for x in 10..90 {
            src.set_pixel(x, 50, true);
        }

        // Curved model: middle bowed down by 10 pixels
        let top = DewarpCurve::new(vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(50.0, 10.0),
            Point2d::new(100.0, 0.0),
        ]);
        let bottom = DewarpCurve::new(vec![
            Point2d::new(0.0, 100.0),
            Point2d::new(50.0, 110.0),
            Point2d::new(100.0, 100.0),
        ]);

        let model = CylindricalSurfaceModel::new(top, bottom, 100, 100);
        let unrolled = DewarpRemapper::dewarp_binary(&src, &model);

        assert!(unrolled.count_black_pixels() > 0);
    }
}

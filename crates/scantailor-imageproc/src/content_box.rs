use crate::binary_image::BinaryImage;
use crate::connected_components::ConnectedComponentLabeler;
use scantailor_math::Rect2i;

/// Automatically detects the bounding box of printed content on a page,
/// ignoring edge artifacts and dust speckles.
///
/// Ported from Scan Tailor's Select Content stage.
pub fn detect_content_box(
    img: &BinaryImage,
    min_component_pixels: usize,
    margin_padding: i32,
) -> Rect2i {
    let ccl = ConnectedComponentLabeler::label_8_connected(img);

    let mut min_x = img.width as i32;
    let mut max_x = 0i32;
    let mut min_y = img.height as i32;
    let mut max_y = 0i32;
    let mut found_content = false;

    for comp in &ccl.components {
        // Ignore small noise speckles
        if comp.pixel_count < min_component_pixels {
            continue;
        }

        // Also ignore edge-touching border artifacts (like black book edges)
        // that span almost the entire height or width with extreme aspect ratios
        let bb = comp.bounding_box;
        let is_border_artifact = (bb.width > (img.width as i32 * 95 / 100) && bb.height < 15)
            || (bb.height > (img.height as i32 * 95 / 100) && bb.width < 15);

        if is_border_artifact {
            continue;
        }

        found_content = true;
        min_x = min_x.min(bb.x);
        max_x = max_x.max(bb.x + bb.width);
        min_y = min_y.min(bb.y);
        max_y = max_y.max(bb.y + bb.height);
    }

    if !found_content {
        return Rect2i::new(0, 0, img.width as i32, img.height as i32);
    }

    // Apply padding
    let left = (min_x - margin_padding).max(0);
    let top = (min_y - margin_padding).max(0);
    let right = (max_x + margin_padding).min(img.width as i32);
    let bottom = (max_y + margin_padding).min(img.height as i32);

    Rect2i::new(left, top, right - left, bottom - top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_content_box() {
        let mut img = BinaryImage::new(100, 100);

        // Content in rect (20, 30, 40, 30)
        for y in 30..60 {
            for x in 20..60 {
                img.set_pixel(x, y, true);
            }
        }

        // Add 1 noise pixel at (90, 90)
        img.set_pixel(90, 90, true);

        let bbox = detect_content_box(&img, 5, 2);
        assert_eq!(bbox, Rect2i::new(18, 28, 44, 34));
    }
}

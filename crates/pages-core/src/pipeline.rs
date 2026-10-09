use crate::project::{Orientation, OutputMode, PageSettings, SplitMode};
use image::{DynamicImage, GenericImageView, GrayImage, RgbImage};
use pages_imageproc::{
    binarize_otsu, binarize_sauvola, despeckle, detect_content_box, enhance_magic_color,
    remove_shadows, rotate_binary_image, BinaryImage, PageLayoutMode, PageSplitDetector,
    SkewFinder,
};
use pages_math::Rect2i;

#[derive(Debug, Clone)]
pub struct ProcessedPage {
    pub page_number: usize,
    pub binary_image: Option<BinaryImage>,
    pub gray_image: Option<GrayImage>,
    pub rgb_image: Option<RgbImage>,
    pub detected_deskew_angle: f64,
    pub content_box: Rect2i,
}

/// Executes the core Scan Tailor processing pipeline on a document page.
pub struct Pipeline;

impl Pipeline {
    /// Loads and processes an image according to `PageSettings`.
    pub fn process_image(
        img: DynamicImage,
        settings: &PageSettings,
    ) -> anyhow::Result<Vec<ProcessedPage>> {
        // Stage 1: Orientation
        let oriented = match settings.orientation {
            Orientation::Rotate0 => img,
            Orientation::Rotate90 => img.rotate90(),
            Orientation::Rotate180 => img.rotate180(),
            Orientation::Rotate270 => img.rotate270(),
        };

        let rgb = oriented.to_rgb8();
        let gray = oriented.to_luma8();

        // Stage 2: Page Split
        let sub_images: Vec<(GrayImage, RgbImage)> = match settings.split_mode {
            SplitMode::SinglePage => vec![(gray, rgb)],
            SplitMode::TwoPages | SplitMode::Auto => {
                let bin_preview = binarize_otsu(&gray, 0);
                let detector = PageSplitDetector::new();
                let split = detector.detect_split(&bin_preview);

                if let (PageLayoutMode::TwoPages, Some(r2)) = (split.mode, split.right_page_rect) {
                    let r1 = split.left_page_rect;

                    let left_gray = gray
                        .view(r1.x as u32, r1.y as u32, r1.width as u32, r1.height as u32)
                        .to_image();
                    let right_gray = gray
                        .view(r2.x as u32, r2.y as u32, r2.width as u32, r2.height as u32)
                        .to_image();

                    let left_rgb = rgb
                        .view(r1.x as u32, r1.y as u32, r1.width as u32, r1.height as u32)
                        .to_image();
                    let right_rgb = rgb
                        .view(r2.x as u32, r2.y as u32, r2.width as u32, r2.height as u32)
                        .to_image();

                    vec![(left_gray, left_rgb), (right_gray, right_rgb)]
                } else {
                    vec![(gray, rgb)]
                }
            }
        };

        let mut results = Vec::new();

        for (idx, (sub_gray, mut sub_rgb)) in sub_images.into_iter().enumerate() {
            let page_num = settings.page_number * 2 + idx;

            // Optional Photometric Pre-filters: De-Shadow & Despeckle
            if settings.deshadow {
                sub_rgb = remove_shadows(&sub_rgb);
            }
            if settings.despeckle {
                sub_rgb = despeckle(&sub_rgb, 30);
            }

            // Stage 3: Initial Binarization for analysis
            let bin = binarize_sauvola(&sub_gray, 15, 0.25, 0);

            // Stage 4: Deskew
            let skew_angle = match settings.deskew_angle {
                Some(angle) => angle,
                None => {
                    let finder = SkewFinder::new();
                    finder.find_skew(&bin).angle_deg
                }
            };

            let deskewed_bin = rotate_binary_image(&bin, skew_angle);

            // Stage 5: Select Content
            let content_box = match settings.content_box {
                Some(bbox) => bbox,
                None => detect_content_box(&deskewed_bin, 10, 5),
            };

            // Stage 6: Margins & Output Cropping
            let (out_bin, out_gray, out_rgb) = match settings.output_mode {
                OutputMode::BlackAndWhite => (Some(deskewed_bin), None, None),
                OutputMode::Grayscale => (None, Some(sub_gray), None),
                OutputMode::Color => (None, None, Some(sub_rgb)),
                OutputMode::MagicColor => {
                    let magic = enhance_magic_color(&sub_rgb);
                    (None, None, Some(magic))
                }
            };

            results.push(ProcessedPage {
                page_number: page_num,
                binary_image: out_bin,
                gray_image: out_gray,
                rgb_image: out_rgb,
                detected_deskew_angle: skew_angle,
                content_box,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use std::path::PathBuf;

    #[test]
    fn test_pipeline_basic() {
        let mut gray = GrayImage::new(100, 100);
        for y in 30..70 {
            for x in 20..80 {
                gray.put_pixel(x, y, Luma([30]));
            }
        }
        let dyn_img = DynamicImage::ImageLuma8(gray);

        let settings = PageSettings::new(PathBuf::from("test.png"), 1);
        let pages = Pipeline::process_image(dyn_img, &settings).unwrap();

        assert!(!pages.is_empty());
        assert!(pages[0].binary_image.is_some());
    }
}

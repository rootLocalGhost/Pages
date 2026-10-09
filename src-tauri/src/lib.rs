use image::{DynamicImage, ImageReader};
use rayon::prelude::*;
use pages_dewarp::{CylindricalSurfaceModel, DewarpCurve, DewarpRemapper};
use pages_imageproc::{
    binarize_otsu, binarize_sauvola, detect_content_box, rotate_binary_image,
    PageLayoutMode, PageSplitDetector, SkewFinder, ConnectedComponentLabeler,
};
use pages_math::Point2d;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::Emitter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBoxDto {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginsDto {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl Default for MarginsDto {
    fn default() -> Self {
        Self {
            left: 10.0,
            top: 10.0,
            right: 10.0,
            bottom: 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageItem {
    pub id: String,
    pub file_path: String,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub orientation: u32, // 0, 90, 180, 270
    pub split_mode: String, // "auto", "single", "two-pages"
    pub split_x: Option<i32>,
    pub deskew_angle: f64,
    pub auto_deskew_angle: Option<f64>,
    pub content_box: Option<ContentBoxDto>,
    pub margins: MarginsDto,
    pub output_mode: String, // "bw", "grayscale", "color"
    pub binarization_method: String, // "sauvola", "wolf", "otsu"
    pub sauvola_k: f64,
    pub sauvola_window: u32,
    pub despeckle: bool,
    pub dewarp: bool,
    pub dpi: u32,
    pub thumbnail_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitResultDto {
    pub mode: String,
    pub split_x: Option<i32>,
    pub left_rect: Option<ContentBoxDto>,
    pub right_rect: Option<ContentBoxDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePreviewResult {
    pub preview_path: String,
    pub original_path: String,
    pub width: u32,
    pub height: u32,
    pub auto_deskew_angle: Option<f64>,
    pub auto_content_box: Option<ContentBoxDto>,
    pub split_info: Option<SplitResultDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgressEvent {
    pub current: usize,
    pub total: usize,
    pub percent: f32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_cores: usize,
    pub app_version: String,
    pub temp_dir: String,
}

fn get_preview_cache_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("pages_previews");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn apply_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img,
    }
}

// ----------------------------------------------------------------------------
// Tauri Commands
// ----------------------------------------------------------------------------

#[tauri::command]
fn get_system_info() -> Result<SystemInfo, String> {
    Ok(SystemInfo {
        cpu_cores: num_cpus::get(),
        app_version: "0.2.0".to_string(),
        temp_dir: get_preview_cache_dir().to_string_lossy().to_string(),
    })
}

#[tauri::command]
fn import_images(paths: Vec<String>) -> Result<Vec<PageItem>, String> {
    let cache_dir = get_preview_cache_dir();

    let items: Result<Vec<PageItem>, String> = paths
        .into_par_iter()
        .enumerate()
        .map(|(idx, path_str)| {
            let path = Path::new(&path_str);
            let file_name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| format!("page_{:04}", idx + 1));

            let reader = ImageReader::open(&path)
                .map_err(|e| format!("Cannot open {}: {}", path_str, e))?
                .with_guessed_format()
                .map_err(|e| format!("Cannot format {}: {}", path_str, e))?;

            let (width, height) = reader
                .into_dimensions()
                .unwrap_or_else(|_| (1200, 1600));

            // Generate small thumbnail for filmstrip and base preview for instant UI interactions
            let thumb_path = cache_dir.join(format!("thumb_{:x}.jpg", hash_str(&path_str)));
            let base_path = cache_dir.join(format!("base_{:x}.jpg", hash_str(&path_str)));
            if !thumb_path.exists() || !base_path.exists() {
                if let Ok(img) = image::open(&path) {
                    if !thumb_path.exists() {
                        let thumb = img.thumbnail(220, 300);
                        let _ = thumb.save(&thumb_path);
                    }
                    if !base_path.exists() {
                        let base_prev = if img.width() > 1400 || img.height() > 1400 {
                            img.thumbnail(1400, 1400)
                        } else {
                            img
                        };
                        let _ = base_prev.save(&base_path);
                    }
                }
            }

            Ok(PageItem {
                id: format!("page_{}_{}", idx, hash_str(&path_str) % 100000),
                file_path: path_str,
                file_name,
                width,
                height,
                orientation: 0,
                split_mode: "auto".to_string(),
                split_x: None,
                deskew_angle: 0.0,
                auto_deskew_angle: None,
                content_box: None,
                margins: MarginsDto::default(),
                output_mode: "bw".to_string(),
                binarization_method: "sauvola".to_string(),
                sauvola_k: 0.25,
                sauvola_window: 15,
                despeckle: true,
                dewarp: false,
                dpi: 300,
                thumbnail_path: Some(thumb_path.to_string_lossy().to_string()),
            })
        })
        .collect();

    items
}

#[tauri::command]
async fn auto_detect_orientation(image_path: String) -> Result<u32, String> {
    tokio::task::spawn_blocking(move || {
        let reader = ImageReader::open(&image_path)
            .map_err(|e| e.to_string())?
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let (w, h) = reader.into_dimensions().unwrap_or((1200, 1600));
        if w > h {
            Ok(0)
        } else {
            Ok(0)
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn auto_detect_split(image_path: String, orientation: Option<u32>) -> Result<SplitResultDto, String> {
    tokio::task::spawn_blocking(move || {
        let orientation = orientation.unwrap_or(0);
        let img = image::open(&image_path).map_err(|e| e.to_string())?;
        let oriented = apply_orientation(img, orientation);
        let orig_w = oriented.width();
        let orig_h = oriented.height();

        // Downscale for fast split detection
        let small = if orig_w > 1200 || orig_h > 1200 {
            oriented.thumbnail(1200, 1200)
        } else {
            oriented
        };
        let scale = orig_w as f64 / small.width() as f64;

        let gray = small.to_luma8();
        let bin = binarize_otsu(&gray, 0);

        let detector = PageSplitDetector::new();
        let split = detector.detect_split(&bin);

        match split.mode {
            PageLayoutMode::TwoPages => {
                let left_dto = ContentBoxDto {
                    x: (split.left_page_rect.x as f64 * scale) as i32,
                    y: (split.left_page_rect.y as f64 * scale) as i32,
                    width: (split.left_page_rect.width as f64 * scale) as i32,
                    height: (split.left_page_rect.height as f64 * scale) as i32,
                };
                let right_dto = split.right_page_rect.map(|r| ContentBoxDto {
                    x: (r.x as f64 * scale) as i32,
                    y: (r.y as f64 * scale) as i32,
                    width: (r.width as f64 * scale) as i32,
                    height: (r.height as f64 * scale) as i32,
                });

                Ok(SplitResultDto {
                    mode: "two-pages".to_string(),
                    split_x: split.split_x.map(|x| (x as f64 * scale) as i32),
                    left_rect: Some(left_dto),
                    right_rect: right_dto,
                })
            }
            PageLayoutMode::SinglePage => Ok(SplitResultDto {
                mode: "single".to_string(),
                split_x: Some((orig_w / 2) as i32),
                left_rect: Some(ContentBoxDto {
                    x: 0,
                    y: 0,
                    width: orig_w as i32,
                    height: orig_h as i32,
                }),
                right_rect: None,
            }),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

fn load_working_image(cache_dir: &Path, file_path: &str) -> Result<(DynamicImage, f64, u32, u32), String> {
    let base_path = cache_dir.join(format!("base_{:x}.jpg", hash_str(file_path)));
    if base_path.exists() {
        if let Ok(img) = image::open(&base_path) {
            let (orig_w, orig_h) = if let Ok(reader) = ImageReader::open(file_path) {
                if let Ok(fmt) = reader.with_guessed_format() {
                    fmt.into_dimensions().unwrap_or((img.width(), img.height()))
                } else {
                    (img.width(), img.height())
                }
            } else {
                (img.width(), img.height())
            };
            let scale = orig_w as f64 / img.width() as f64;
            return Ok((img, scale, orig_w, orig_h));
        }
    }

    let raw = image::open(file_path).map_err(|e| e.to_string())?;
    let orig_w = raw.width();
    let orig_h = raw.height();
    let small = if orig_w > 1400 || orig_h > 1400 {
        raw.thumbnail(1400, 1400)
    } else {
        raw
    };
    let _ = small.save(&base_path);
    let scale = orig_w as f64 / small.width() as f64;
    Ok((small, scale, orig_w, orig_h))
}

#[tauri::command]
async fn auto_detect_deskew(image_path: String, orientation: Option<u32>) -> Result<f64, String> {
    tokio::task::spawn_blocking(move || {
        let cache_dir = get_preview_cache_dir();
        let orientation = orientation.unwrap_or(0);
        let (img, _scale, _w, _h) = load_working_image(&cache_dir, &image_path)?;
        let oriented = apply_orientation(img, orientation);

        let gray = oriented.to_luma8();
        let bin = binarize_sauvola(&gray, 15, 0.25, 0);

        let finder = SkewFinder::new();
        let result = finder.find_skew(&bin);
        Ok(result.angle_deg)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn auto_detect_content_box(
    image_path: String,
    orientation: Option<u32>,
    deskew_angle: Option<f64>,
) -> Result<ContentBoxDto, String> {
    tokio::task::spawn_blocking(move || {
        let cache_dir = get_preview_cache_dir();
        let orientation = orientation.unwrap_or(0);
        let deskew_angle = deskew_angle.unwrap_or(0.0);
        let (img, scale, _w, _h) = load_working_image(&cache_dir, &image_path)?;
        let oriented = apply_orientation(img, orientation);

        let gray = oriented.to_luma8();
        let bin = binarize_sauvola(&gray, 15, 0.25, 0);

        let rotated_bin = if deskew_angle.abs() > 0.01 {
            rotate_binary_image(&bin, deskew_angle)
        } else {
            bin
        };

        let bbox = detect_content_box(&rotated_bin, 10, 5);
        Ok(ContentBoxDto {
            x: (bbox.x as f64 * scale) as i32,
            y: (bbox.y as f64 * scale) as i32,
            width: (bbox.width as f64 * scale) as i32,
            height: (bbox.height as f64 * scale) as i32,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn batch_auto_detect_content(
    window: tauri::Window,
    pages: Vec<PageItem>,
) -> Result<Vec<PageItem>, String> {
    let cache_dir = get_preview_cache_dir();
    let total = pages.len();
    let completed = Arc::new(AtomicUsize::new(0));

    tokio::task::spawn_blocking(move || {
        let updated: Vec<PageItem> = pages
            .into_par_iter()
            .map(|mut page| {
                if let Ok((img, scale, _w, _h)) = load_working_image(&cache_dir, &page.file_path) {
                    let oriented = apply_orientation(img, page.orientation);
                    let gray = oriented.to_luma8();
                    let bin = binarize_sauvola(&gray, 15, 0.25, 0);

                    let rotated_bin = if page.deskew_angle.abs() > 0.01 {
                        rotate_binary_image(&bin, page.deskew_angle)
                    } else {
                        bin
                    };

                    let bbox = detect_content_box(&rotated_bin, 10, 5);
                    page.content_box = Some(ContentBoxDto {
                        x: (bbox.x as f64 * scale) as i32,
                        y: (bbox.y as f64 * scale) as i32,
                        width: (bbox.width as f64 * scale) as i32,
                        height: (bbox.height as f64 * scale) as i32,
                    });
                }

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                let pct = (done as f32 / total as f32) * 100.0;
                let _ = window.emit(
                    "batch-progress",
                    BatchProgressEvent {
                        current: done,
                        total,
                        percent: pct,
                        message: format!("Detecting content box for page {}/{} individually...", done, total),
                    },
                );

                page
            })
            .collect();

        Ok(updated)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn batch_auto_deskew(
    window: tauri::Window,
    pages: Vec<PageItem>,
) -> Result<Vec<PageItem>, String> {
    let cache_dir = get_preview_cache_dir();
    let total = pages.len();
    let completed = Arc::new(AtomicUsize::new(0));

    tokio::task::spawn_blocking(move || {
        let updated: Vec<PageItem> = pages
            .into_par_iter()
            .map(|mut page| {
                if let Ok((img, _scale, _w, _h)) = load_working_image(&cache_dir, &page.file_path) {
                    let oriented = apply_orientation(img, page.orientation);
                    let gray = oriented.to_luma8();
                    let bin = binarize_sauvola(&gray, 15, 0.25, 0);

                    let finder = SkewFinder::new();
                    let skew_res = finder.find_skew(&bin);
                    page.deskew_angle = skew_res.angle_deg;
                    page.auto_deskew_angle = Some(skew_res.angle_deg);
                }

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                let pct = (done as f32 / total as f32) * 100.0;
                let _ = window.emit(
                    "batch-progress",
                    BatchProgressEvent {
                        current: done,
                        total,
                        percent: pct,
                        message: format!("Straightening page {}/{} individually...", done, total),
                    },
                );

                page
            })
            .collect();

        Ok(updated)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn batch_auto_analyze_all(
    window: tauri::Window,
    pages: Vec<PageItem>,
) -> Result<Vec<PageItem>, String> {
    let cache_dir = get_preview_cache_dir();
    let total = pages.len();
    let completed = Arc::new(AtomicUsize::new(0));

    tokio::task::spawn_blocking(move || {
        let updated: Vec<PageItem> = pages
            .into_par_iter()
            .map(|mut page| {
                if let Ok((img, scale, _w, _h)) = load_working_image(&cache_dir, &page.file_path) {
                    let oriented = apply_orientation(img, page.orientation);
                    let gray = oriented.to_luma8();
                    let bin = binarize_sauvola(&gray, 15, 0.25, 0);

                    // 1. Deskew per page
                    let finder = SkewFinder::new();
                    let skew_res = finder.find_skew(&bin);
                    page.deskew_angle = skew_res.angle_deg;
                    page.auto_deskew_angle = Some(skew_res.angle_deg);

                    // 2. Rotate binary
                    let rotated_bin = if page.deskew_angle.abs() > 0.01 {
                        rotate_binary_image(&bin, page.deskew_angle)
                    } else {
                        bin
                    };

                    // 3. Content box per page
                    let bbox = detect_content_box(&rotated_bin, 10, 5);
                    page.content_box = Some(ContentBoxDto {
                        x: (bbox.x as f64 * scale) as i32,
                        y: (bbox.y as f64 * scale) as i32,
                        width: (bbox.width as f64 * scale) as i32,
                        height: (bbox.height as f64 * scale) as i32,
                    });
                }

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                let pct = (done as f32 / total as f32) * 100.0;
                let _ = window.emit(
                    "batch-progress",
                    BatchProgressEvent {
                        current: done,
                        total,
                        percent: pct,
                        message: format!("Auto-analyzing page {}/{} (Deskew & Content)...", done, total),
                    },
                );

                page
            })
            .collect();

        Ok(updated)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn generate_stage_preview(page: PageItem, stage: String) -> Result<StagePreviewResult, String> {
    tokio::task::spawn_blocking(move || {
        let cache_dir = get_preview_cache_dir();

        let hash_key = format!(
            "{}_{}_{}_{:.2}_{:?}_{}_{}_{:.2}_{}",
            page.id,
            stage,
            page.orientation,
            page.deskew_angle,
            page.content_box,
            page.output_mode,
            page.binarization_method,
            page.sauvola_k,
            page.despeckle
        );
        let preview_path = cache_dir.join(format!("prev_{:x}.jpg", hash_str(&hash_key)));

        // Cache hit: return immediately with zero computation!
        if preview_path.exists() {
            return Ok(StagePreviewResult {
                preview_path: preview_path.to_string_lossy().to_string(),
                original_path: page.file_path,
                width: page.width,
                height: page.height,
                auto_deskew_angle: None,
                auto_content_box: None,
                split_info: None,
            });
        }

        let mut auto_deskew_angle = None;
        let mut auto_content_box = None;
        let mut split_info = None;

        let (img, scale, orig_w, orig_h) = load_working_image(&cache_dir, &page.file_path)?;
        let oriented = apply_orientation(img, page.orientation);
        let preview_img = oriented;

        match stage.as_str() {
            "orientation" => {
                preview_img.save(&preview_path).map_err(|e| e.to_string())?;
            }
            "split" => {
                let gray = preview_img.to_luma8();
                let bin = binarize_otsu(&gray, 0);
                let detector = PageSplitDetector::new();
                let split = detector.detect_split(&bin);

                split_info = Some(SplitResultDto {
                    mode: if split.mode == PageLayoutMode::TwoPages {
                        "two-pages".into()
                    } else {
                        "single".into()
                    },
                    split_x: split.split_x.map(|x| (x as f64 * scale) as i32),
                    left_rect: Some(ContentBoxDto {
                        x: (split.left_page_rect.x as f64 * scale) as i32,
                        y: (split.left_page_rect.y as f64 * scale) as i32,
                        width: (split.left_page_rect.width as f64 * scale) as i32,
                        height: (split.left_page_rect.height as f64 * scale) as i32,
                    }),
                    right_rect: split.right_page_rect.map(|r| ContentBoxDto {
                        x: (r.x as f64 * scale) as i32,
                        y: (r.y as f64 * scale) as i32,
                        width: (r.width as f64 * scale) as i32,
                        height: (r.height as f64 * scale) as i32,
                    }),
                });

                preview_img.save(&preview_path).map_err(|e| e.to_string())?;
            }
            "deskew" => {
                let gray = preview_img.to_luma8();
                let bin = binarize_sauvola(&gray, 15, 0.25, 0);
                let finder = SkewFinder::new();
                let detected = finder.find_skew(&bin).angle_deg;
                auto_deskew_angle = Some(detected);

                let angle_to_apply = if page.deskew_angle.abs() > 0.001 {
                    page.deskew_angle
                } else {
                    detected
                };

                let rotated_bin = rotate_binary_image(&bin, angle_to_apply);
                let out_gray = rotated_bin.to_gray_image();
                out_gray.save(&preview_path).map_err(|e| e.to_string())?;
            }
            "content" => {
                let gray = preview_img.to_luma8();
                let bin = binarize_sauvola(&gray, 15, 0.25, 0);
                let rotated_bin = if page.deskew_angle.abs() > 0.001 {
                    rotate_binary_image(&bin, page.deskew_angle)
                } else {
                    bin
                };

                let detected_box = detect_content_box(&rotated_bin, 10, 5);
                auto_content_box = Some(ContentBoxDto {
                    x: (detected_box.x as f64 * scale) as i32,
                    y: (detected_box.y as f64 * scale) as i32,
                    width: (detected_box.width as f64 * scale) as i32,
                    height: (detected_box.height as f64 * scale) as i32,
                });

                let out_gray = rotated_bin.to_gray_image();
                out_gray.save(&preview_path).map_err(|e| e.to_string())?;
            }
            "margins" => {
                let gray = preview_img.to_luma8();
                let bin = binarize_sauvola(&gray, 15, 0.25, 0);
                let rotated_bin = if page.deskew_angle.abs() > 0.001 {
                    rotate_binary_image(&bin, page.deskew_angle)
                } else {
                    bin
                };

                let out_gray = rotated_bin.to_gray_image();
                out_gray.save(&preview_path).map_err(|e| e.to_string())?;
            }
            "output" => {
                let mut gray = preview_img.to_luma8();

                if page.dewarp {
                    let w = gray.width() as f64;
                    let h = gray.height() as f64;
                    let top_curve = DewarpCurve::new(vec![
                        Point2d::new(0.0, 0.0),
                        Point2d::new(w * 0.5, 12.0),
                        Point2d::new(w, 0.0),
                    ]);
                    let bottom_curve = DewarpCurve::new(vec![
                        Point2d::new(0.0, h),
                        Point2d::new(w * 0.5, h - 12.0),
                        Point2d::new(w, h),
                    ]);
                    let model = CylindricalSurfaceModel::new(top_curve, bottom_curve, gray.width(), gray.height());
                    gray = DewarpRemapper::dewarp_gray(&gray, &model);
                }

                let bin = match page.binarization_method.as_str() {
                    "otsu" => binarize_otsu(&gray, 0),
                    _ => binarize_sauvola(&gray, page.sauvola_window, page.sauvola_k, 0),
                };

                let deskewed_bin = if page.deskew_angle.abs() > 0.001 {
                    rotate_binary_image(&bin, page.deskew_angle)
                } else {
                    bin
                };

                let final_bin = if page.despeckle {
                    ConnectedComponentLabeler::label_8_connected(&deskewed_bin).despeckle(3)
                } else {
                    deskewed_bin
                };

                match page.output_mode.as_str() {
                    "grayscale" => {
                        gray.save(&preview_path).map_err(|e| e.to_string())?;
                    }
                    "color" => {
                        preview_img.save(&preview_path).map_err(|e| e.to_string())?;
                    }
                    _ => {
                        let out_gray = final_bin.to_gray_image();
                        out_gray.save(&preview_path).map_err(|e| e.to_string())?;
                    }
                }
            }
            _ => {
                preview_img.save(&preview_path).map_err(|e| e.to_string())?;
            }
        }

        Ok(StagePreviewResult {
            preview_path: preview_path.to_string_lossy().to_string(),
            original_path: page.file_path,
            width: orig_w,
            height: orig_h,
            auto_deskew_angle,
            auto_content_box,
            split_info,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn run_batch_processing(
    window: tauri::Window,
    pages: Vec<PageItem>,
    output_dir: String,
) -> Result<Vec<String>, String> {
    let out_path = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&out_path).map_err(|e| e.to_string())?;

    let total = pages.len();
    let completed = Arc::new(AtomicUsize::new(0));

    let processed_files: Result<Vec<String>, String> = tokio::task::spawn_blocking(move || {
        let results: Result<Vec<String>, String> = pages
            .par_iter()
            .enumerate()
            .map(|(idx, page)| {
                let img = image::open(&page.file_path).map_err(|e| e.to_string())?;
                let oriented = apply_orientation(img, page.orientation);
                let gray = oriented.to_luma8();

                let bin = match page.binarization_method.as_str() {
                    "otsu" => binarize_otsu(&gray, 0),
                    _ => binarize_sauvola(&gray, page.sauvola_window, page.sauvola_k, 0),
                };

                let deskewed = if page.deskew_angle.abs() > 0.001 {
                    rotate_binary_image(&bin, page.deskew_angle)
                } else {
                    bin
                };

                let final_bin = if page.despeckle {
                    ConnectedComponentLabeler::label_8_connected(&deskewed).despeckle(3)
                } else {
                    deskewed
                };

                let out_file = out_path.join(format!("page_{:04}.png", idx + 1));
                let out_gray = final_bin.to_gray_image();
                out_gray.save(&out_file).map_err(|e| e.to_string())?;

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                let pct = (done as f32 / total as f32) * 100.0;
                let _ = window.emit(
                    "batch-progress",
                    BatchProgressEvent {
                        current: done,
                        total,
                        percent: pct,
                        message: format!("Processed {}/{} pages...", done, total),
                    },
                );

                Ok(out_file.to_string_lossy().to_string())
            })
            .collect();

        results
    })
    .await
    .map_err(|e| e.to_string())?;

    processed_files
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            import_images,
            auto_detect_orientation,
            auto_detect_split,
            auto_detect_deskew,
            auto_detect_content_box,
            batch_auto_detect_content,
            batch_auto_deskew,
            batch_auto_analyze_all,
            generate_stage_preview,
            run_batch_processing,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Pages");
}

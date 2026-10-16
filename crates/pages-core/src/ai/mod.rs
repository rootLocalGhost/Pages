//! Pages AI Neural Restoration & Document Geometry Engine
//! Runs deep learning document alignment, dewarping, and restoration
//! via ONNX Runtime on Hardware Accelerators (DirectML, OpenVINO) or multi-threaded CPU.

use image::{ImageBuffer, Rgb, RgbImage};
use pages_math::{Homography2D, Point2d};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[cfg(feature = "ai")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "ai")]
use std::sync::{Arc, Mutex};

/// Hardware execution device preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AiDeviceChoice {
    #[default]
    Auto,
    DirectML,
    OpenVINO,
    Cpu,
}

impl std::fmt::Display for AiDeviceChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiDeviceChoice::Auto => write!(f, "auto"),
            AiDeviceChoice::DirectML => write!(f, "directml"),
            AiDeviceChoice::OpenVINO => write!(f, "openvino"),
            AiDeviceChoice::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::str::FromStr for AiDeviceChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(AiDeviceChoice::Auto),
            "directml" | "dml" | "gpu" => Ok(AiDeviceChoice::DirectML),
            "openvino" | "ov" => Ok(AiDeviceChoice::OpenVINO),
            "cpu" => Ok(AiDeviceChoice::Cpu),
            other => Err(format!(
                "Unknown device '{other}'. Choices: auto, directml, openvino, cpu"
            )),
        }
    }
}

/// Geometry prediction results extracted from PagesGeomNet (Pages Align).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryPrediction {
    /// 4 corner coordinates in original image pixel space [TL, TR, BR, BL]
    pub corners: [Point2d; 4],
    /// Softmax peak confidence scores for each corner
    pub corner_confidences: [f32; 4],
    /// Average boundary confidence
    pub boundary_score: f32,
    /// Whether the corners form a geometrically valid quad
    pub has_valid_quad: bool,
}

/// Options configuring the end-to-end Pages AI pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPipelineOptions {
    /// Execute Stage 1 geometry detection & projective dewarping
    pub dewarp: bool,
    /// Execute Stage 2 neural restoration & shadow removal (Pages Shine)
    pub clean: bool,
    /// Apply sub-pixel Huber gradient refinement along paper boundary
    pub refine_edges: bool,
    /// Automatically correct residual content tilt
    pub auto_deskew: bool,
    /// Purge colored background spill along margins
    pub clean_borders: bool,
    /// Tile size for sliding-window inference (512 for GPU, 384/256 for low-power PC)
    pub tile_size: usize,
    /// Overlap margin in pixels (64 default, 32 for fast low-power draft)
    pub overlap: usize,
    /// Low-power acceleration mode
    pub fast_mode: bool,
    /// Hardware acceleration target
    pub device: AiDeviceChoice,
}

impl Default for AiPipelineOptions {
    fn default() -> Self {
        Self {
            dewarp: true,
            clean: true,
            refine_edges: true,
            auto_deskew: true,
            clean_borders: true,
            tile_size: 512,
            overlap: 64,
            fast_mode: false,
            device: AiDeviceChoice::Auto,
        }
    }
}

/// Execution telemetry and metrics from the AI processing run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPipelineResult {
    pub corners: Option<[Point2d; 4]>,
    pub detected_skew_deg: f64,
    pub device_used: String,
    pub dewarp_latency_ms: f64,
    pub clean_latency_ms: f64,
    pub total_latency_ms: f64,
}

/// Discovered ONNX model file paths on disk.
#[derive(Debug, Clone)]
pub struct AiModelPaths {
    pub geom_path: PathBuf,
    pub clean_path: PathBuf,
}

/// Search for ONNX model weights in project directories and app bundles.
pub fn discover_models(custom_dir: Option<&Path>) -> anyhow::Result<AiModelPaths> {
    let mut candidate_dirs: Vec<PathBuf> = Vec::new();

    if let Some(d) = custom_dir {
        candidate_dirs.push(d.to_path_buf());
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidate_dirs.push(parent.join("models"));
            candidate_dirs.push(parent.join("resources").join("models"));
            candidate_dirs.push(parent.to_path_buf());
        }
    }

    candidate_dirs.push(PathBuf::from("models"));
    candidate_dirs.push(PathBuf::from("../models"));
    candidate_dirs.push(PathBuf::from("../../models"));
    candidate_dirs.push(PathBuf::from("onnx_models"));
    candidate_dirs.push(PathBuf::from("../onnx_models"));
    candidate_dirs.push(PathBuf::from("../../onnx_models"));
    candidate_dirs.push(PathBuf::from(
        "c:/Users/LEADER/Desktop/Pages-AI/Pages/models",
    ));
    candidate_dirs.push(PathBuf::from(
        "c:/Users/LEADER/Desktop/Pages-AI/onnx_models",
    ));

    for dir in &candidate_dirs {
        let geom_cand1 = dir.join("pages_geom_v1.onnx");
        let clean_cand1 = dir.join("pages_clean_v1.onnx");
        if geom_cand1.exists() && clean_cand1.exists() {
            return Ok(AiModelPaths {
                geom_path: geom_cand1,
                clean_path: clean_cand1,
            });
        }

        // Also check if any .onnx files exist in dir
        let geom_cand2 = dir.join("pages_align_v1.onnx");
        let clean_cand2 = dir.join("pages_shine_v1.onnx");
        if geom_cand2.exists() && clean_cand2.exists() {
            return Ok(AiModelPaths {
                geom_path: geom_cand2,
                clean_path: clean_cand2,
            });
        }
    }

    anyhow::bail!(
        "Could not find Pages AI ONNX models (pages_geom_v1.onnx & pages_clean_v1.onnx). Checked directories: {:?}",
        candidate_dirs
    );
}

/// Primary Neural Inference Engine for Pages.
pub struct PagesAiEngine {
    #[cfg(feature = "ai")]
    geom_session: Option<Arc<Mutex<Session>>>,
    #[cfg(feature = "ai")]
    clean_session: Option<Arc<Mutex<Session>>>,
    device_name: String,
    device_choice: AiDeviceChoice,
}

impl PagesAiEngine {
    /// Initialize an empty/disabled engine.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ai")]
            geom_session: None,
            #[cfg(feature = "ai")]
            clean_session: None,
            device_name: "Disabled".to_string(),
            device_choice: AiDeviceChoice::Auto,
        }
    }

    /// Automatically find and load the models using the preferred hardware device.
    pub fn auto_load(preferred_device: AiDeviceChoice) -> anyhow::Result<Self> {
        let paths = discover_models(None)?;
        Self::load_from_paths(&paths.geom_path, &paths.clean_path, preferred_device)
    }

    /// Load models from explicit file paths.
    pub fn load_from_paths<P1: AsRef<Path>, P2: AsRef<Path>>(
        geom_path: P1,
        clean_path: P2,
        preferred_device: AiDeviceChoice,
    ) -> anyhow::Result<Self> {
        #[cfg(feature = "ai")]
        {
            let geom_p = geom_path.as_ref();
            let clean_p = clean_path.as_ref();

            if !geom_p.exists() {
                anyhow::bail!("Pages Geom model not found at {}", geom_p.display());
            }
            if !clean_p.exists() {
                anyhow::bail!("Pages Clean model not found at {}", clean_p.display());
            }

            let (geom_sess, clean_sess, dev_str) =
                Self::build_sessions(geom_p, clean_p, preferred_device)?;

            Ok(Self {
                geom_session: Some(Arc::new(Mutex::new(geom_sess))),
                clean_session: Some(Arc::new(Mutex::new(clean_sess))),
                device_name: dev_str,
                device_choice: preferred_device,
            })
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = (geom_path, clean_path, preferred_device);
            anyhow::bail!("Pages AI feature is disabled in this build.")
        }
    }

    #[cfg(feature = "ai")]
    fn create_session(
        model_path: &Path,
        num_threads: usize,
        device: AiDeviceChoice,
    ) -> anyhow::Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Session builder error: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Opt level error: {e}"))?
            .with_intra_threads(num_threads)
            .map_err(|e| anyhow::anyhow!("Intra threads error: {e}"))?;

        match device {
            AiDeviceChoice::DirectML => {
                #[cfg(feature = "directml")]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::DirectML::default().into()])
                        .map_err(|e| anyhow::anyhow!("DirectML EP error: {e}"))?;
                }
                #[cfg(not(feature = "directml"))]
                {
                    anyhow::bail!("DirectML feature not enabled in build");
                }
            }
            AiDeviceChoice::OpenVINO => {
                #[cfg(feature = "openvino")]
                {
                    builder = builder
                        .with_execution_providers([ort::ep::OpenVINO::default().into()])
                        .map_err(|e| anyhow::anyhow!("OpenVINO EP error: {e}"))?;
                }
                #[cfg(not(feature = "openvino"))]
                {
                    anyhow::bail!("OpenVINO feature not enabled in build");
                }
            }
            _ => {}
        }

        builder.commit_from_file(model_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load ONNX model from {}: {e}",
                model_path.display()
            )
        })
    }

    #[cfg(feature = "ai")]
    fn build_sessions(
        geom_p: &Path,
        clean_p: &Path,
        preferred_device: AiDeviceChoice,
    ) -> anyhow::Result<(Session, Session, String)> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16);

        let try_init = |device_req: AiDeviceChoice| -> anyhow::Result<(Session, Session, String)> {
            let geom_sess = Self::create_session(geom_p, num_threads, device_req)?;
            let clean_sess = Self::create_session(clean_p, num_threads, device_req)?;

            let label = match device_req {
                AiDeviceChoice::DirectML => "DirectML (DirectX 12 GPU Accelerated)".to_string(),
                AiDeviceChoice::OpenVINO => "OpenVINO (Intel Hardware Accelerated)".to_string(),
                _ => format!("CPU Multi-Core ({num_threads} Threads, AVX2/AVX-512)"),
            };

            Ok((geom_sess, clean_sess, label))
        };

        match preferred_device {
            AiDeviceChoice::DirectML => try_init(AiDeviceChoice::DirectML),
            AiDeviceChoice::OpenVINO => try_init(AiDeviceChoice::OpenVINO),
            AiDeviceChoice::Cpu => try_init(AiDeviceChoice::Cpu),
            AiDeviceChoice::Auto => {
                // Priority 1: DirectML (Works across all Windows GPUs: Intel Arc, AMD, NVIDIA, Qualcomm)
                if let Ok(res) = try_init(AiDeviceChoice::DirectML) {
                    return Ok(res);
                }
                // Priority 2: OpenVINO
                if let Ok(res) = try_init(AiDeviceChoice::OpenVINO) {
                    return Ok(res);
                }
                // Priority 3: CPU multi-threaded
                try_init(AiDeviceChoice::Cpu)
            }
        }
    }

    /// Whether the AI engine is loaded and operational.
    pub fn is_ready(&self) -> bool {
        #[cfg(feature = "ai")]
        {
            self.geom_session.is_some() && self.clean_session.is_some()
        }
        #[cfg(not(feature = "ai"))]
        {
            false
        }
    }

    /// Returns human-readable device name and preference.
    pub fn device_info(&self) -> (&str, AiDeviceChoice) {
        (&self.device_name, self.device_choice)
    }

    /// Detect 4 corners and boundary mask of the document sheet.
    pub fn detect_geometry(&self, img: &RgbImage) -> anyhow::Result<GeometryPrediction> {
        #[cfg(feature = "ai")]
        {
            let sess_arc = match &self.geom_session {
                Some(s) => s.clone(),
                None => anyhow::bail!("Geometry model session is not loaded"),
            };

            let (orig_w, orig_h) = (img.width(), img.height());
            if orig_w == 0 || orig_h == 0 {
                anyhow::bail!("Empty input image");
            }

            // Downsample input to fixed 512x512 thumbnail for geometry network
            let thumb =
                image::imageops::resize(img, 512, 512, image::imageops::FilterType::Triangle);

            // Shape [1, 3, 512, 512] float32 normalized [0.0, 1.0]
            let mut input_tensor = vec![0.0f32; 3 * 512 * 512];
            let ch_stride = 512 * 512;
            for y in 0..512 {
                for x in 0..512 {
                    let px = thumb.get_pixel(x, y);
                    let idx = (y * 512 + x) as usize;
                    input_tensor[idx] = px[0] as f32 / 255.0;
                    input_tensor[ch_stride + idx] = px[1] as f32 / 255.0;
                    input_tensor[2 * ch_stride + idx] = px[2] as f32 / 255.0;
                }
            }

            let input_shape = vec![1i64, 3, 512, 512];
            let tensor =
                ort::value::Tensor::from_array((input_shape, input_tensor.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("Tensor creation failed: {e}"))?;
            let inputs = ort::inputs!["input" => tensor];

            let mut sess = sess_arc
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex lock: {e}"))?;
            let outputs = sess
                .run(inputs)
                .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;

            // Extract corner_heatmaps: [1, 4, 128, 128]
            let (_c_shape, corners_slice) = outputs["corner_heatmaps"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Corner extraction failed: {e}"))?;

            // Sub-pixel parabolic peak extraction for each corner
            let mut corners = [Point2d::default(); 4];
            let mut confs = [0.0f32; 4];
            let hm_stride = 128 * 128;

            for c in 0..4 {
                let mut max_val = -1.0f32;
                let mut best_x = 0usize;
                let mut best_y = 0usize;

                for y in 0..128 {
                    for x in 0..128 {
                        let val = corners_slice[c * hm_stride + y * 128 + x];
                        if val > max_val {
                            max_val = val;
                            best_x = x;
                            best_y = y;
                        }
                    }
                }

                confs[c] = max_val;

                // Parabolic quadratic sub-pixel peak interpolation
                let mut sub_x = best_x as f64;
                let mut sub_y = best_y as f64;

                if best_x > 0 && best_x < 127 && best_y > 0 && best_y < 127 {
                    let v = max_val;
                    let vl = corners_slice[c * hm_stride + best_y * 128 + (best_x - 1)];
                    let vr = corners_slice[c * hm_stride + best_y * 128 + (best_x + 1)];
                    let vt = corners_slice[c * hm_stride + (best_y - 1) * 128 + best_x];
                    let vb = corners_slice[c * hm_stride + (best_y + 1) * 128 + best_x];

                    let denom_x = (2.0 * v - vr - vl).max(1e-5);
                    let denom_y = (2.0 * v - vb - vt).max(1e-5);

                    let dx = ((vr - vl) / (2.0 * denom_x)).clamp(-0.5, 0.5) as f64;
                    let dy = ((vb - vt) / (2.0 * denom_y)).clamp(-0.5, 0.5) as f64;

                    sub_x += dx;
                    sub_y += dy;
                }

                // Map to original image coordinates
                let scale_x = orig_w as f64 / 128.0;
                let scale_y = orig_h as f64 / 128.0;

                corners[c] = Point2d::new(
                    (sub_x * scale_x).clamp(0.0, orig_w as f64 - 1.0),
                    (sub_y * scale_y).clamp(0.0, orig_h as f64 - 1.0),
                );
            }

            // Extract boundary score from boundary_mask if present
            let mut boundary_score = 0.5f32;
            if let Some(boundary_tensor) = outputs.get("boundary_mask") {
                if let Ok((_b_shape, b_slice)) = boundary_tensor.try_extract_tensor::<f32>() {
                    let b_page_stride = 512 * 512;
                    let mut sum = 0.0f32;
                    let sample_step = 8;
                    let mut count = 0;
                    for i in (0..(512 * 512)).step_by(sample_step) {
                        sum += b_slice[b_page_stride + i];
                        count += 1;
                    }
                    if count > 0 {
                        boundary_score = sum / count as f32;
                    }
                }
            }

            // Validate that corners form a convex quad with reasonable area
            let has_valid_quad = is_valid_quad(&corners, orig_w, orig_h);

            Ok(GeometryPrediction {
                corners,
                corner_confidences: confs,
                boundary_score,
                has_valid_quad,
            })
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = img;
            anyhow::bail!("AI feature disabled");
        }
    }

    /// Dewarp a document image given 4 corners using true projective homography.
    pub fn dewarp_page(
        &self,
        img: &RgbImage,
        custom_corners: Option<[Point2d; 4]>,
        refine_edges: bool,
        auto_deskew: bool,
        clean_borders: bool,
    ) -> anyhow::Result<(RgbImage, GeometryPrediction, f64)> {
        let (orig_w, orig_h) = (img.width(), img.height());

        let geom = match custom_corners {
            Some(c) => GeometryPrediction {
                corners: c,
                corner_confidences: [1.0; 4],
                boundary_score: 1.0,
                has_valid_quad: is_valid_quad(&c, orig_w, orig_h),
            },
            None => self.detect_geometry(img)?,
        };

        let mut active_corners = geom.corners;

        // If quad is completely invalid (e.g., pure blank canvas), fallback to full image rect
        if !geom.has_valid_quad {
            active_corners = [
                Point2d::new(0.0, 0.0),
                Point2d::new((orig_w - 1) as f64, 0.0),
                Point2d::new((orig_w - 1) as f64, (orig_h - 1) as f64),
                Point2d::new(0.0, (orig_h - 1) as f64),
            ];
        } else if refine_edges {
            active_corners = refine_quad_edges(img, active_corners);
        }

        let tl = active_corners[0];
        let tr = active_corners[1];
        let br = active_corners[2];
        let bl = active_corners[3];

        let width_top = tr.distance_to(tl);
        let width_bot = br.distance_to(bl);
        let target_w = (width_top.max(width_bot).round() as u32).max(64);

        let height_left = bl.distance_to(tl);
        let height_right = br.distance_to(tr);
        let target_h = (height_left.max(height_right).round() as u32).max(64);

        // Projective homography mapping: destination canvas coordinates -> source camera coordinates
        let dst_pts = [
            Point2d::new(0.0, 0.0),
            Point2d::new((target_w - 1) as f64, 0.0),
            Point2d::new((target_w - 1) as f64, (target_h - 1) as f64),
            Point2d::new(0.0, (target_h - 1) as f64),
        ];

        let h_inv = Homography2D::from_four_point_correspondences(dst_pts, active_corners)
            .ok_or_else(|| anyhow::anyhow!("Singular homography matrix"))?;

        // Parallel backward bilinear sampling across target image rows
        let mut dewarped_buf = vec![0u8; (target_w * target_h * 3) as usize];
        let row_stride = (target_w * 3) as usize;

        dewarped_buf
            .par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..target_w {
                    let pt = h_inv.transform_point(Point2d::new(x as f64, y as f64));
                    let sx = pt.x;
                    let sy = pt.y;

                    let x0 = sx.floor() as i64;
                    let y0 = sy.floor() as i64;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    let fx = (sx - x0 as f64) as f32;
                    let fy = (sy - y0 as f64) as f32;

                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;

                    let cx = |v: i64| (v.clamp(0, orig_w as i64 - 1)) as u32;
                    let cy = |v: i64| (v.clamp(0, orig_h as i64 - 1)) as u32;

                    let p00 = img.get_pixel(cx(x0), cy(y0));
                    let p10 = img.get_pixel(cx(x1), cy(y0));
                    let p01 = img.get_pixel(cx(x0), cy(y1));
                    let p11 = img.get_pixel(cx(x1), cy(y1));

                    let r = (p00[0] as f32 * w00
                        + p10[0] as f32 * w10
                        + p01[0] as f32 * w01
                        + p11[0] as f32 * w11)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let g = (p00[1] as f32 * w00
                        + p10[1] as f32 * w10
                        + p01[1] as f32 * w01
                        + p11[1] as f32 * w11)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let b = (p00[2] as f32 * w00
                        + p10[2] as f32 * w10
                        + p01[2] as f32 * w01
                        + p11[2] as f32 * w11)
                        .round()
                        .clamp(0.0, 255.0) as u8;

                    let offset = (x * 3) as usize;
                    row[offset] = r;
                    row[offset + 1] = g;
                    row[offset + 2] = b;
                }
            });

        let mut dewarped_img =
            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(target_w, target_h, dewarped_buf)
                .ok_or_else(|| anyhow::anyhow!("Failed to construct dewarped RgbImage"))?;

        // Subtle Inset Crop (0.8%) to eliminate border edge bleed
        let inset_x = ((target_w as f64 * 0.008).round() as u32).min(target_w / 8);
        let inset_y = ((target_h as f64 * 0.008).round() as u32).min(target_h / 8);
        if inset_x > 0 && inset_y > 0 && target_w > 2 * inset_x && target_h > 2 * inset_y {
            dewarped_img = image::imageops::crop_imm(
                &dewarped_img,
                inset_x,
                inset_y,
                target_w - 2 * inset_x,
                target_h - 2 * inset_y,
            )
            .to_image();
        }

        // Automatic Fine Content Deskewing
        let mut final_skew_deg = 0.0;
        if auto_deskew {
            let (deskewed, detected_deg) = detect_and_correct_fine_skew(&dewarped_img, 6.0);
            dewarped_img = deskewed;
            final_skew_deg = detected_deg;
        }

        // Clean residual colored border spills
        if clean_borders {
            clean_border_artifacts(&mut dewarped_img, 0.035);
        }

        Ok((dewarped_img, geom, final_skew_deg))
    }

    /// Process high-resolution scan through PagesCleanNet using seamless sliding-window inference.
    pub fn clean_page_tiled(
        &self,
        img: &RgbImage,
        tile_size: usize,
        overlap: usize,
    ) -> anyhow::Result<RgbImage> {
        #[cfg(feature = "ai")]
        {
            let session_arc = match &self.clean_session {
                Some(s) => s.clone(),
                None => anyhow::bail!("AI Clean model session is not loaded"),
            };

            let width = img.width() as usize;
            let height = img.height() as usize;

            if width == 0 || height == 0 {
                return Ok(img.clone());
            }

            let effective_tile = tile_size.clamp(256, 1024);
            let effective_overlap = overlap.min(effective_tile / 2).max(16);
            let step = effective_tile - effective_overlap;

            let mut accum_r = vec![0.0f32; width * height];
            let mut accum_g = vec![0.0f32; width * height];
            let mut accum_b = vec![0.0f32; width * height];
            let mut weights = vec![0.0f32; width * height];

            // Precompute 2D Hann window of size effective_tile x effective_tile
            let mut hann = vec![0.0f32; effective_tile * effective_tile];
            for ty in 0..effective_tile {
                let wy = 0.5
                    * (1.0
                        - ((2.0 * std::f32::consts::PI * ty as f32) / (effective_tile - 1) as f32)
                            .cos());
                for tx in 0..effective_tile {
                    let wx = 0.5
                        * (1.0
                            - ((2.0 * std::f32::consts::PI * tx as f32)
                                / (effective_tile - 1) as f32)
                                .cos());
                    hann[ty * effective_tile + tx] = (wx * wy).max(0.01);
                }
            }

            let ny = if height <= effective_tile {
                1
            } else {
                ((height - effective_overlap + step - 1) / step).max(1)
            };

            let nx = if width <= effective_tile {
                1
            } else {
                ((width - effective_overlap + step - 1) / step).max(1)
            };

            let ch_stride = effective_tile * effective_tile;

            for iy in 0..ny {
                let sy = if height <= effective_tile {
                    0
                } else {
                    (iy * step).min(height - effective_tile)
                };
                let ey = (sy + effective_tile).min(height);
                let tile_h = ey - sy;

                for ix in 0..nx {
                    let sx = if width <= effective_tile {
                        0
                    } else {
                        (ix * step).min(width - effective_tile)
                    };
                    let ex = (sx + effective_tile).min(width);
                    let tile_w = ex - sx;

                    // Fixed-size tensor [1, 3, effective_tile, effective_tile] padded with 1.0 (pure white)
                    let mut input_tensor = vec![1.0f32; 3 * ch_stride];

                    for ty in 0..tile_h {
                        for tx in 0..tile_w {
                            let px = img.get_pixel((sx + tx) as u32, (sy + ty) as u32);
                            let idx = ty * effective_tile + tx;
                            input_tensor[idx] = px[0] as f32 / 255.0;
                            input_tensor[ch_stride + idx] = px[1] as f32 / 255.0;
                            input_tensor[2 * ch_stride + idx] = px[2] as f32 / 255.0;
                        }
                    }

                    let input_shape = vec![1i64, 3, effective_tile as i64, effective_tile as i64];
                    let tensor = ort::value::Tensor::from_array((
                        input_shape,
                        input_tensor.into_boxed_slice(),
                    ))
                    .map_err(|e| anyhow::anyhow!("Clean tensor creation failed: {e}"))?;
                    let inputs = ort::inputs!["input" => tensor];

                    let mut session = session_arc
                        .lock()
                        .map_err(|e| anyhow::anyhow!("Mutex: {e}"))?;
                    let outputs = session
                        .run(inputs)
                        .map_err(|e| anyhow::anyhow!("Clean run: {e}"))?;
                    let (_out_shape, output_slice) = outputs["output"]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| anyhow::anyhow!("Output tensor extraction failed: {e}"))?;

                    for ty in 0..tile_h {
                        for tx in 0..tile_w {
                            let tile_idx = ty * effective_tile + tx;
                            let w = hann[tile_idx];
                            let out_idx = (sy + ty) * width + (sx + tx);

                            let r = output_slice[tile_idx];
                            let g = output_slice[ch_stride + tile_idx];
                            let b = output_slice[2 * ch_stride + tile_idx];

                            accum_r[out_idx] += r * w;
                            accum_g[out_idx] += g * w;
                            accum_b[out_idx] += b * w;
                            weights[out_idx] += w;
                        }
                    }
                }
            }

            let mut out_img = RgbImage::new(width as u32, height as u32);
            for py in 0..height {
                for px in 0..width {
                    let idx = py * width + px;
                    let w = weights[idx].max(1e-6);

                    let r = ((accum_r[idx] / w) * 255.0).clamp(0.0, 255.0) as u8;
                    let g = ((accum_g[idx] / w) * 255.0).clamp(0.0, 255.0) as u8;
                    let b = ((accum_b[idx] / w) * 255.0).clamp(0.0, 255.0) as u8;

                    out_img.put_pixel(px as u32, py as u32, Rgb([r, g, b]));
                }
            }

            Ok(out_img)
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = (img, tile_size, overlap);
            anyhow::bail!("AI feature disabled");
        }
    }

    /// Convenience clean_page with standard settings.
    pub fn clean_page(&self, img: &RgbImage) -> anyhow::Result<RgbImage> {
        self.clean_page_tiled(img, 512, 64)
    }

    /// Complete 2-Stage End-to-End Pages AI Pipeline.
    pub fn run_pipeline(
        &self,
        img: &RgbImage,
        options: &AiPipelineOptions,
    ) -> anyhow::Result<(RgbImage, AiPipelineResult)> {
        let total_start = std::time::Instant::now();
        let mut cur_img = img.clone();
        let mut corners_res = None;
        let mut skew_deg = 0.0;
        let mut dewarp_ms = 0.0;
        let mut clean_ms = 0.0;

        // Stage 1: Geometry & Dewarping
        if options.dewarp {
            let t0 = std::time::Instant::now();
            let (dewarped, geom, detected_skew) = self.dewarp_page(
                &cur_img,
                None,
                options.refine_edges,
                options.auto_deskew,
                options.clean_borders,
            )?;
            dewarp_ms = t0.elapsed().as_secs_f64() * 1000.0;
            cur_img = dewarped;
            corners_res = Some(geom.corners);
            skew_deg = detected_skew;
        }

        // Stage 2: Neural Restoration (Pages Shine)
        if options.clean {
            let t0 = std::time::Instant::now();
            let (tile_sz, ovr) = if options.fast_mode {
                (384, 32)
            } else {
                (options.tile_size, options.overlap)
            };
            let cleaned = self.clean_page_tiled(&cur_img, tile_sz, ovr)?;
            clean_ms = t0.elapsed().as_secs_f64() * 1000.0;
            cur_img = cleaned;
        }

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let result = AiPipelineResult {
            corners: corners_res,
            detected_skew_deg: skew_deg,
            device_used: self.device_name.clone(),
            dewarp_latency_ms: dewarp_ms,
            clean_latency_ms: clean_ms,
            total_latency_ms: total_ms,
        };

        Ok((cur_img, result))
    }
}

// ----------------------------------------------------------------------------
// Geometric & Image Processing Helpers in Pure Rust
// ----------------------------------------------------------------------------

fn is_valid_quad(pts: &[Point2d; 4], w: u32, h: u32) -> bool {
    let [p0, p1, p2, p3] = pts;

    // Shoelace formula for polygon area
    let area = 0.5
        * ((p0.x * p1.y - p1.x * p0.y)
            + (p1.x * p2.y - p2.x * p1.y)
            + (p2.x * p3.y - p3.x * p2.y)
            + (p3.x * p0.y - p0.x * p3.y))
            .abs();

    let img_area = (w as f64) * (h as f64);
    if area < 0.10 * img_area {
        return false;
    }

    // Check cross products for convexity
    let cross_sign = |a: Point2d, b: Point2d, c: Point2d| -> f64 {
        let v1 = b - a;
        let v2 = c - b;
        v1.cross(v2)
    };

    let s0 = cross_sign(*p0, *p1, *p2);
    let s1 = cross_sign(*p1, *p2, *p3);
    let s2 = cross_sign(*p2, *p3, *p0);
    let s3 = cross_sign(*p3, *p0, *p1);

    (s0 > 0.0 && s1 > 0.0 && s2 > 0.0 && s3 > 0.0) || (s0 < 0.0 && s1 < 0.0 && s2 < 0.0 && s3 < 0.0)
}

/// Refine paper quad corners by sampling normal edge gradients and intersecting lines.
pub fn refine_quad_edges(img: &RgbImage, corners: [Point2d; 4]) -> [Point2d; 4] {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let [tl, tr, br, bl] = corners;

    // Helper: samples highest gradient along normal vector
    let sample_edge = |p1: Point2d, p2: Point2d, normal: Point2d| -> Vec<Point2d> {
        let count = 40;
        let mut pts = Vec::with_capacity(count);

        for i in 2..(count - 2) {
            let t = i as f64 / count as f64;
            let center = p1 * (1.0 - t) + p2 * t;

            let mut best_grad = -1.0;
            let mut best_pt = None;

            for step in -25..=25 {
                let test_pt = center + normal * (step as f64);
                let p_in = center + normal * ((step + 3) as f64);
                let p_out = center + normal * ((step - 3) as f64);

                let clamp_pt = |p: Point2d| {
                    let cx = (p.x.round() as i64).clamp(0, w as i64 - 1) as u32;
                    let cy = (p.y.round() as i64).clamp(0, h as i64 - 1) as u32;
                    let px = img.get_pixel(cx, cy);
                    px[0] as f64 * 0.299 + px[1] as f64 * 0.587 + px[2] as f64 * 0.114
                };

                let g_in = clamp_pt(p_in);
                let g_out = clamp_pt(p_out);
                let grad = g_in - g_out;

                if grad > best_grad {
                    best_grad = grad;
                    best_pt = Some(test_pt);
                }
            }

            if let Some(pt) = best_pt {
                if best_grad > 20.0 {
                    pts.push(pt);
                }
            }
        }
        pts
    };

    // Edge normal vectors pointing inwards
    let v_top = tr - tl;
    let n_top = Point2d::new(-v_top.y, v_top.x)
        .normalized()
        .unwrap_or(Point2d::new(0.0, 1.0));
    let n_top = if n_top.y < 0.0 { -n_top } else { n_top };
    let pts_top = sample_edge(tl, tr, n_top);

    let v_right = br - tr;
    let n_right = Point2d::new(-v_right.y, v_right.x)
        .normalized()
        .unwrap_or(Point2d::new(-1.0, 0.0));
    let n_right = if n_right.x > 0.0 { -n_right } else { n_right };
    let pts_right = sample_edge(tr, br, n_right);

    let v_bot = br - bl;
    let n_bot = Point2d::new(-v_bot.y, v_bot.x)
        .normalized()
        .unwrap_or(Point2d::new(0.0, -1.0));
    let n_bot = if n_bot.y > 0.0 { -n_bot } else { n_bot };
    let pts_bot = sample_edge(bl, br, n_bot);

    let v_left = bl - tl;
    let n_left = Point2d::new(-v_left.y, v_left.x)
        .normalized()
        .unwrap_or(Point2d::new(1.0, 0.0));
    let n_left = if n_left.x < 0.0 { -n_left } else { n_left };
    let pts_left = sample_edge(tl, bl, n_left);

    if pts_top.len() < 8 || pts_right.len() < 8 || pts_bot.len() < 8 || pts_left.len() < 8 {
        return corners;
    }

    let l_top = fit_line_pts(&pts_top);
    let l_right = fit_line_pts(&pts_right);
    let l_bot = fit_line_pts(&pts_bot);
    let l_left = fit_line_pts(&pts_left);

    let ref_tl = line_intersect(l_top, l_left);
    let ref_tr = line_intersect(l_top, l_right);
    let ref_br = line_intersect(l_bot, l_right);
    let ref_bl = line_intersect(l_bot, l_left);

    match (ref_tl, ref_tr, ref_br, ref_bl) {
        (Some(ctl), Some(ctr), Some(cbr), Some(cbl)) => {
            let candidate = [ctl, ctr, cbr, cbl];
            for i in 0..4 {
                if candidate[i].distance_to(corners[i]) > 80.0 {
                    return corners;
                }
            }
            if is_valid_quad(&candidate, w as u32, h as u32) {
                candidate
            } else {
                corners
            }
        }
        _ => corners,
    }
}

// Line representation: ax + by + c = 0
#[derive(Clone, Copy)]
struct LineEq {
    a: f64,
    b: f64,
    c: f64,
}

fn fit_line_pts(pts: &[Point2d]) -> LineEq {
    let n = pts.len() as f64;
    let mean_x = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p.y).sum::<f64>() / n;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;

    for p in pts {
        let dx = p.x - mean_x;
        let dy = p.y - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let a = -theta.sin();
    let b = theta.cos();
    let c = -(a * mean_x + b * mean_y);

    LineEq { a, b, c }
}

fn line_intersect(l1: LineEq, l2: LineEq) -> Option<Point2d> {
    let det = l1.a * l2.b - l2.a * l1.b;
    if det.abs() < 1e-6 {
        return None;
    }
    let x = (l1.b * l2.c - l2.b * l1.c) / det;
    let y = (l2.a * l1.c - l1.a * l2.c) / det;
    Some(Point2d::new(x, y))
}

/// Detect and correct fine skew angle using projection profile variance.
pub fn detect_and_correct_fine_skew(img: &RgbImage, max_angle: f64) -> (RgbImage, f64) {
    let (w, h) = (img.width(), img.height());
    if w < 64 || h < 64 {
        return (img.clone(), 0.0);
    }

    // Downscale to max dimension 800 for instant evaluation
    let max_dim = w.max(h);
    let scale = if max_dim > 800 {
        800.0 / max_dim as f64
    } else {
        1.0
    };
    let thumb_w = (w as f64 * scale).round() as u32;
    let thumb_h = (h as f64 * scale).round() as u32;

    let thumb =
        image::imageops::resize(img, thumb_w, thumb_h, image::imageops::FilterType::Nearest);

    let gray = image::imageops::grayscale(&thumb);
    let bin = pages_imageproc::binarize_sauvola(&gray, 15, 0.25, 0);

    let finder = pages_imageproc::SkewFinder {
        max_angle_deg: max_angle,
        coarse_step_deg: 0.25,
        fine_step_deg: 0.05,
    };
    let skew_res = finder.find_skew(&bin);
    let angle_deg = skew_res.angle_deg;

    if angle_deg.abs() < 0.15 {
        return (img.clone(), 0.0);
    }

    // Rotate high-res image by -angle_deg around center using bilinear sampling
    let rad = (-angle_deg).to_radians();
    let cos_t = rad.cos();
    let sin_t = rad.sin();

    let cx = w as f64 * 0.5;
    let cy = h as f64 * 0.5;

    let mut out_buf = vec![0u8; (w * h * 3) as usize];
    let row_stride = (w * 3) as usize;

    out_buf
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let dy = y as f64 - cy;
            for x in 0..w {
                let dx = x as f64 - cx;
                let sx = cx + dx * cos_t - dy * sin_t;
                let sy = cy + dx * sin_t + dy * cos_t;

                let x0 = sx.floor() as i64;
                let y0 = sy.floor() as i64;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = (sx - x0 as f64) as f32;
                let fy = (sy - y0 as f64) as f32;

                let clamp_x = |v: i64| (v.clamp(0, w as i64 - 1)) as u32;
                let clamp_y = |v: i64| (v.clamp(0, h as i64 - 1)) as u32;

                let p00 = img.get_pixel(clamp_x(x0), clamp_y(y0));
                let p10 = img.get_pixel(clamp_x(x1), clamp_y(y0));
                let p01 = img.get_pixel(clamp_x(x0), clamp_y(y1));
                let p11 = img.get_pixel(clamp_x(x1), clamp_y(y1));

                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;

                let r = (p00[0] as f32 * w00
                    + p10[0] as f32 * w10
                    + p01[0] as f32 * w01
                    + p11[0] as f32 * w11)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let g = (p00[1] as f32 * w00
                    + p10[1] as f32 * w10
                    + p01[1] as f32 * w01
                    + p11[1] as f32 * w11)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let b = (p00[2] as f32 * w00
                    + p10[2] as f32 * w10
                    + p01[2] as f32 * w01
                    + p11[2] as f32 * w11)
                    .round()
                    .clamp(0.0, 255.0) as u8;

                let off = (x * 3) as usize;
                row[off] = r;
                row[off + 1] = g;
                row[off + 2] = b;
            }
        });

    let rotated_img =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, out_buf).unwrap_or_else(|| img.clone());

    (rotated_img, angle_deg)
}

/// Cleans residual background artifacts (bedsheets, desks, shadows) along the page outer rim.
pub fn clean_border_artifacts(img: &mut RgbImage, margin_pct: f64) {
    let (w, h) = (img.width(), img.height());
    let mh = ((h as f64 * margin_pct).round() as u32).min(h / 4);
    let mw = ((w as f64 * margin_pct).round() as u32).min(w / 4);

    let is_spill = |p: Rgb<u8>| {
        let r = p[0] as i32;
        let g = p[1] as i32;
        let b = p[2] as i32;
        let max_val = r.max(g).max(b);
        let min_val = r.min(g).min(b);
        let diff = max_val - min_val;
        let mean = (r + g + b) / 3;
        diff > 25 || mean < 170
    };

    // Top & Bottom margin bands
    for y in 0..mh {
        for x in 0..w {
            if is_spill(*img.get_pixel(x, y)) {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
            let by = h - 1 - y;
            if is_spill(*img.get_pixel(x, by)) {
                img.put_pixel(x, by, Rgb([255, 255, 255]));
            }
        }
    }

    // Left & Right margin bands
    for x in 0..mw {
        for y in 0..h {
            if is_spill(*img.get_pixel(x, y)) {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
            let rx = w - 1 - x;
            if is_spill(*img.get_pixel(rx, y)) {
                img.put_pixel(rx, y, Rgb([255, 255, 255]));
            }
        }
    }
}

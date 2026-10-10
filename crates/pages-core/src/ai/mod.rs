//! Pages AI Neural Restoration & Enhancement Engine
//! Runs deep learning document restoration (despeckling, shadow removal, bleed-through elimination)
//! via ONNX Runtime on CPU or Hardware Accelerators (DirectML, OpenVINO).

use image::RgbImage;
#[cfg(feature = "ai")]
use image::Rgb;
use std::path::Path;

#[cfg(feature = "ai")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "ai")]
use std::sync::{Arc, Mutex};

/// Neural inference engine for Pages document processing.
pub struct PagesAiEngine {
    #[cfg(feature = "ai")]
    session: Option<Arc<Mutex<Session>>>,
}

impl PagesAiEngine {
    /// Initialize an empty or disabled AI engine.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ai")]
            session: None,
        }
    }

    /// Load an ONNX model from the specified file path.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        #[cfg(feature = "ai")]
        {
            let path_ref = path.as_ref();
            if !path_ref.exists() {
                anyhow::bail!("Model file not found: {}", path_ref.display());
            }

            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);

            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session builder error: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Opt level error: {e}"))?
                .with_intra_threads(num_threads)
                .map_err(|e| anyhow::anyhow!("Intra threads error: {e}"))?
                .commit_from_file(path_ref)
                .map_err(|e| anyhow::anyhow!("Failed to load ONNX model from file: {e}"))?;

            Ok(Self {
                session: Some(Arc::new(Mutex::new(session))),
            })
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = path;
            anyhow::bail!("AI feature is disabled in this build")
        }
    }

    /// Load an ONNX model directly from embedded bytes.
    pub fn load_from_memory(model_bytes: &[u8]) -> anyhow::Result<Self> {
        #[cfg(feature = "ai")]
        {
            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);

            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session builder error: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Opt level error: {e}"))?
                .with_intra_threads(num_threads)
                .map_err(|e| anyhow::anyhow!("Intra threads error: {e}"))?
                .commit_from_memory(model_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to load ONNX model from memory: {e}"))?;

            Ok(Self {
                session: Some(Arc::new(Mutex::new(session))),
            })
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = model_bytes;
            anyhow::bail!("AI feature is disabled in this build")
        }
    }

    /// Returns true if the neural network weights are active and ready for inference.
    pub fn is_ready(&self) -> bool {
        #[cfg(feature = "ai")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "ai"))]
        {
            false
        }
    }

    /// Process a high-resolution scan through the neural network using tiled sliding-window inference.
    /// Handles arbitrary scan sizes (300/600 DPI) seamlessly without out-of-memory errors.
    pub fn clean_page(&self, img: &RgbImage) -> anyhow::Result<RgbImage> {
        #[cfg(feature = "ai")]
        {
            let session_arc = match &self.session {
                Some(s) => s.clone(),
                None => anyhow::bail!("AI model session is not loaded"),
            };

            let mut session = session_arc.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock AI session mutex: {e}"))?;

            let width = img.width() as usize;
            let height = img.height() as usize;

            if width == 0 || height == 0 {
                return Ok(img.clone());
            }

            const TILE_SIZE: usize = 512;
            const OVERLAP: usize = 64;
            const STEP: usize = TILE_SIZE - OVERLAP;

            // Output accumulator and weight map for seamless feathering
            let mut accum_r = vec![0.0f32; width * height];
            let mut accum_g = vec![0.0f32; width * height];
            let mut accum_b = vec![0.0f32; width * height];
            let mut weights = vec![0.0f32; width * height];

            let mut y = 0;
            while y < height {
                let tile_h = (TILE_SIZE).min(height - y);
                let mut x = 0;

                while x < width {
                    let tile_w = (TILE_SIZE).min(width - x);

                    // Extract tile tensor: Shape [1, 3, tile_h, tile_w] in range [0.0, 1.0]
                    let mut input_tensor = vec![0.0f32; 3 * tile_h * tile_w];
                    let ch_stride = tile_h * tile_w;

                    for ty in 0..tile_h {
                        for tx in 0..tile_w {
                            let px = img.get_pixel((x + tx) as u32, (y + ty) as u32);
                            let idx = ty * tile_w + tx;
                            input_tensor[idx] = px[0] as f32 / 255.0;
                            input_tensor[ch_stride + idx] = px[1] as f32 / 255.0;
                            input_tensor[2 * ch_stride + idx] = px[2] as f32 / 255.0;
                        }
                    }

                    // Create ONNX tensor
                    let input_shape = vec![1i64, 3, tile_h as i64, tile_w as i64];
                    let tensor = ort::value::Tensor::from_array((input_shape, input_tensor.into_boxed_slice()))
                        .map_err(|e| anyhow::anyhow!("Tensor creation failed: {e}"))?;
                    let inputs = ort::inputs!["input" => tensor];

                    // Run inference
                    let outputs = session.run(inputs)
                        .map_err(|e| anyhow::anyhow!("Inference run failed: {e}"))?;
                    let (_out_shape, output_slice) = outputs["output"].try_extract_tensor::<f32>()
                        .map_err(|e| anyhow::anyhow!("Output tensor extraction failed: {e}"))?;

                    // Blend tile with Hann/Cosine window to prevent edge seams
                    for ty in 0..tile_h {
                        let wy = if tile_h > 1 {
                            0.5 * (1.0 - ((2.0 * std::f32::consts::PI * ty as f32) / (tile_h - 1) as f32).cos())
                        } else {
                            1.0
                        };

                        for tx in 0..tile_w {
                            let wx = if tile_w > 1 {
                                0.5 * (1.0 - ((2.0 * std::f32::consts::PI * tx as f32) / (tile_w - 1) as f32).cos())
                            } else {
                                1.0
                            };

                            let w = (wx * wy).max(0.01);
                            let tile_idx = ty * tile_w + tx;
                            let out_idx = (y + ty) * width + (x + tx);

                            let r = output_slice[0 * ch_stride + tile_idx];
                            let g = output_slice[1 * ch_stride + tile_idx];
                            let b = output_slice[2 * ch_stride + tile_idx];

                            accum_r[out_idx] += r * w;
                            accum_g[out_idx] += g * w;
                            accum_b[out_idx] += b * w;
                            weights[out_idx] += w;
                        }
                    }

                    if x + tile_w >= width {
                        break;
                    }
                    x += STEP;
                }

                if y + tile_h >= height {
                    break;
                }
                y += STEP;
            }

            // Normalize weights and reconstruct output image
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
            let _ = img;
            anyhow::bail!("AI feature is disabled in this build")
        }
    }
}

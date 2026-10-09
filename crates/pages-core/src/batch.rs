use crate::pipeline::Pipeline;
use crate::project::Project;
use anyhow::Context;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Batch page processor providing parallel execution over project pages.
pub struct BatchProcessor;

impl BatchProcessor {
    /// Discovers all supported image files in an input directory.
    pub fn discover_images<P: AsRef<Path>>(dir: P) -> anyhow::Result<Vec<PathBuf>> {
        let mut images = Vec::new();
        let entries = std::fs::read_dir(dir)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if matches!(
                        ext_lower.as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
                    ) {
                        images.push(path);
                    }
                }
            }
        }

        images.sort();
        Ok(images)
    }

    /// Processes all pages in a Project in parallel using Rayon and saves outputs.
    pub fn process_project(
        project: &Project,
        on_progress: Option<&(dyn Fn(usize, usize) + Send + Sync)>,
    ) -> anyhow::Result<Vec<PathBuf>> {
        std::fs::create_dir_all(&project.output_directory)?;

        let total = project.pages.len();
        let completed = std::sync::atomic::AtomicUsize::new(0);

        let page_results: Vec<anyhow::Result<Vec<PathBuf>>> = project
            .pages
            .par_iter()
            .map(|page_settings| {
                let img = image::open(&page_settings.source_path)
                    .with_context(|| format!("Failed to open {:?}", page_settings.source_path))?;

                let processed_pages =
                    Pipeline::process_image(img, page_settings).with_context(|| {
                        format!("Failed processing {:?}", page_settings.source_path)
                    })?;

                let mut page_outputs = Vec::new();
                for p in processed_pages {
                    let out_filename = format!("page_{:04}.png", p.page_number);
                    let out_path = project.output_directory.join(out_filename);

                    if let Some(bin) = &p.binary_image {
                        let gray = bin.to_gray_image();
                        gray.save(&out_path)?;
                    } else if let Some(gray) = &p.gray_image {
                        gray.save(&out_path)?;
                    }
                    page_outputs.push(out_path);
                }

                let cur = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if let Some(callback) = on_progress {
                    callback(cur, total);
                }

                Ok(page_outputs)
            })
            .collect();

        let mut all_paths = Vec::new();
        for res in page_results {
            all_paths.extend(res?);
        }
        all_paths.sort();
        Ok(all_paths)
    }
}

use pages_math::Rect2i;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Orientation {
    #[default]
    Rotate0,
    Rotate90,
    Rotate180,
    Rotate270,
}

impl Orientation {
    pub fn angle_degrees(&self) -> f64 {
        match self {
            Orientation::Rotate0 => 0.0,
            Orientation::Rotate90 => 90.0,
            Orientation::Rotate180 => 180.0,
            Orientation::Rotate270 => 270.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SplitMode {
    #[default]
    Auto,
    SinglePage,
    TwoPages,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarginsMm {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl Default for MarginsMm {
    fn default() -> Self {
        Self {
            left: 10.0,
            top: 10.0,
            right: 10.0,
            bottom: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputMode {
    #[default]
    BlackAndWhite,
    Grayscale,
    Color,
}

/// Settings and post-processing metadata for a single page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSettings {
    pub source_path: PathBuf,
    pub page_number: usize,
    pub orientation: Orientation,
    pub split_mode: SplitMode,
    pub deskew_angle: Option<f64>,
    pub content_box: Option<Rect2i>,
    pub margins: MarginsMm,
    pub output_mode: OutputMode,
    pub dpi: u32,
}

impl PageSettings {
    pub fn new(source_path: PathBuf, page_number: usize) -> Self {
        Self {
            source_path,
            page_number,
            orientation: Orientation::default(),
            split_mode: SplitMode::default(),
            deskew_angle: None,
            content_box: None,
            margins: MarginsMm::default(),
            output_mode: OutputMode::default(),
            dpi: 300,
        }
    }
}

/// Project document definition containing collection of pages.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Project {
    pub version: u32,
    pub pages: Vec<PageSettings>,
    pub output_directory: PathBuf,
}

impl Project {
    pub fn new(output_directory: PathBuf) -> Self {
        Self {
            version: 1,
            pages: Vec::new(),
            output_directory,
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let project = serde_json::from_str(&json)?;
        Ok(project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_serialization() {
        let mut proj = Project::new(PathBuf::from("/out"));
        proj.pages
            .push(PageSettings::new(PathBuf::from("p1.png"), 1));

        let json = serde_json::to_string(&proj).unwrap();
        let loaded: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.pages.len(), 1);
        assert_eq!(loaded.pages[0].dpi, 300);
    }
}

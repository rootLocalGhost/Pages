use clap::{Parser, ValueEnum};
use pages_core::{
    BatchProcessor, MarginsMm, Orientation, OutputMode, PageSettings, Project, SplitMode,
};
use pages_imageproc::{detect_content_box, PageSplitDetector, SkewFinder};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum CliLayout {
    Auto,
    Single,
    TwoPages,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum CliOutputMode {
    Bw,
    Grayscale,
    Color,
}

#[derive(Parser, Debug)]
#[command(
    name = "pages-cli",
    version = "1.0.0",
    about = "Pages CLI - High-performance document & book post-processing engine (ScanSmith AI Integration Ready)"
)]
struct Args {
    /// Input directory containing scan images or a single image file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output directory for processed pages
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Target DPI for output pages
    #[arg(long, default_value_t = 300)]
    dpi: u32,

    /// Page layout mode (auto-detect, single page, or two pages)
    #[arg(long, value_enum, default_value_t = CliLayout::Auto)]
    layout: CliLayout,

    /// Output color format (black-and-white, grayscale, or color)
    #[arg(long, value_enum, default_value_t = CliOutputMode::Bw)]
    mode: CliOutputMode,

    /// Fixed orientation rotation in degrees (0, 90, 180, 270)
    #[arg(long)]
    orientation: Option<u32>,

    /// Uniform page margin in mm (default: 10.0)
    #[arg(long, default_value_t = 10.0)]
    margin: f64,

    /// Fixed deskew angle in degrees (if omitted, auto-detection is used)
    #[arg(long)]
    deskew: Option<f64>,

    /// Analyze input files without producing output images (emits JSON analysis)
    #[arg(long)]
    analyze_only: bool,

    /// Save the generated project configuration to a JSON file
    #[arg(long)]
    save_project: Option<PathBuf>,

    /// Load settings from an existing project JSON file instead
    #[arg(short = 'p', long)]
    project: Option<PathBuf>,

    /// Emit machine-readable JSON streaming events to stdout for IPC / integration
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
#[serde(tag = "event")]
enum JsonEvent<'a> {
    #[serde(rename = "start")]
    Start { total: usize },

    #[serde(rename = "progress")]
    Progress {
        done: usize,
        total: usize,
        percent: f32,
    },

    #[serde(rename = "page_completed")]
    PageCompleted {
        page: usize,
        total: usize,
        output_file: &'a Path,
    },

    #[serde(rename = "analysis")]
    Analysis { pages: &'a [PageAnalysisDto] },

    #[serde(rename = "complete")]
    Complete {
        success: bool,
        pages_processed: usize,
        files: &'a [PathBuf],
        elapsed_sec: f64,
    },

    #[serde(rename = "error")]
    Error { success: bool, error: &'a str },
}

#[derive(Serialize, Clone, Debug)]
struct PageAnalysisDto {
    source_path: String,
    width: u32,
    height: u32,
    detected_split_mode: String,
    split_x: Option<i32>,
    deskew_angle_deg: f64,
    content_box: Option<ContentBoxDto>,
}

#[derive(Serialize, Clone, Debug)]
struct ContentBoxDto {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

fn emit_json<T: Serialize>(event: &T) {
    if let Ok(line) = serde_json::to_string(event) {
        println!("{}", line);
    }
}

fn analyze_images(images: &[PathBuf]) -> Vec<PageAnalysisDto> {
    let skew_finder = SkewFinder::new();
    let split_detector = PageSplitDetector::new();

    images
        .iter()
        .filter_map(|img_path| {
            let img = image::open(img_path).ok()?;
            let (w, h) = (img.width(), img.height());
            let gray = img.to_luma8();
            let bin = pages_imageproc::binarize_sauvola(&gray, 30, 0.25, 0);

            let skew = skew_finder.find_skew(&bin);
            let split = split_detector.detect_split(&bin);
            let bbox = detect_content_box(&bin, 50, 10);

            Some(PageAnalysisDto {
                source_path: img_path.to_string_lossy().to_string(),
                width: w,
                height: h,
                detected_split_mode: format!("{:?}", split.mode),
                split_x: split.split_x.map(|x| x as i32),
                deskew_angle_deg: skew.angle_deg,
                content_box: Some(ContentBoxDto {
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height,
                }),
            })
        })
        .collect()
}

fn run_cli() -> anyhow::Result<()> {
    let args = Args::parse();

    let project = if let Some(proj_file) = &args.project {
        let mut proj = Project::load_from_file(proj_file)?;
        if let Some(out) = args.output {
            proj.output_directory = out;
        }
        proj
    } else {
        let input_path = args
            .input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Either --input or --project must be specified"))?;

        let output_dir = args
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from("output"));

        let mut proj = Project::new(output_dir);

        let images = if input_path.is_dir() {
            BatchProcessor::discover_images(input_path)?
        } else {
            vec![input_path.clone()]
        };

        if images.is_empty() {
            if args.json {
                emit_json(&JsonEvent::Error {
                    success: false,
                    error: "No supported images found in input path.",
                });
            } else {
                eprintln!("No supported images found in input path.");
            }
            return Ok(());
        }

        // Handle --analyze-only flag
        if args.analyze_only {
            let analysis = analyze_images(&images);
            if args.json {
                emit_json(&JsonEvent::Analysis { pages: &analysis });
            } else {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            }
            return Ok(());
        }

        let split_mode = match args.layout {
            CliLayout::Auto => SplitMode::Auto,
            CliLayout::Single => SplitMode::SinglePage,
            CliLayout::TwoPages => SplitMode::TwoPages,
        };

        let output_mode = match args.mode {
            CliOutputMode::Bw => OutputMode::BlackAndWhite,
            CliOutputMode::Grayscale => OutputMode::Grayscale,
            CliOutputMode::Color => OutputMode::Color,
        };

        let orientation = match args.orientation {
            Some(90) => Orientation::Rotate90,
            Some(180) => Orientation::Rotate180,
            Some(270) => Orientation::Rotate270,
            _ => Orientation::Rotate0,
        };

        let margins = MarginsMm {
            left: args.margin,
            top: args.margin,
            right: args.margin,
            bottom: args.margin,
        };

        for (idx, img_path) in images.into_iter().enumerate() {
            let mut page = PageSettings::new(img_path, idx + 1);
            page.dpi = args.dpi;
            page.split_mode = split_mode;
            page.output_mode = output_mode;
            page.orientation = orientation;
            page.margins = margins;
            page.deskew_angle = args.deskew;
            proj.pages.push(page);
        }

        proj
    };

    if let Some(save_path) = &args.save_project {
        project.save_to_file(save_path)?;
    }

    let total_pages = project.pages.len();

    if args.json {
        emit_json(&JsonEvent::Start { total: total_pages });
    } else {
        println!("=== Pages Engine (Rust Core) ===");
        println!("Output directory: {}", project.output_directory.display());
        println!("Processing {} pages in parallel...", total_pages);
    }

    let start = std::time::Instant::now();

    let output_files = BatchProcessor::process_project(
        &project,
        Some(&|done, total| {
            if args.json {
                let percent = if total > 0 {
                    (done as f32 / total as f32) * 100.0
                } else {
                    100.0
                };
                emit_json(&JsonEvent::Progress {
                    done,
                    total,
                    percent,
                });
            } else {
                print!("\rProgress: [{}/{}] pages processed...", done, total);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
        }),
    )?;

    let elapsed = start.elapsed();

    if args.json {
        for (i, out_file) in output_files.iter().enumerate() {
            emit_json(&JsonEvent::PageCompleted {
                page: i + 1,
                total: output_files.len(),
                output_file: out_file.as_path(),
            });
        }

        emit_json(&JsonEvent::Complete {
            success: true,
            pages_processed: total_pages,
            files: &output_files,
            elapsed_sec: elapsed.as_secs_f64(),
        });
    } else {
        println!(
            "\nDone! Successfully processed {} pages ({}) in {:.2}s.",
            total_pages,
            output_files.len(),
            elapsed.as_secs_f64()
        );
    }

    Ok(())
}

fn main() {
    if let Err(err) = run_cli() {
        let err_str = format!("{:#}", err);
        if std::env::args().any(|a| a == "--json") {
            emit_json(&JsonEvent::Error {
                success: false,
                error: &err_str,
            });
        } else {
            eprintln!("Error: {}", err_str);
        }
        std::process::exit(1);
    }
}

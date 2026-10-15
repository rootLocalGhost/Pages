use clap::{Parser, ValueEnum};
use pages_core::{
    discover_models, AiDeviceChoice, AiPipelineOptions, BatchProcessor, MarginsMm, Orientation,
    OutputMode, PageSettings, PagesAiEngine, Project, SplitMode,
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
    Magic,
}

#[derive(Parser, Debug)]
#[command(
    name = "pages-cli",
    version,
    about = "Pages CLI - High-performance document engine with native Pages AI neural models (DirectML / OpenVINO / CPU)"
)]
struct Args {
    /// Input directory containing scan images or a single image file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output directory for processed pages
    #[arg(short, long)]
    output: Option<PathBuf>,

    // ------------------------------------------------------------------------
    // Pages AI Neural Engine Controls
    // ------------------------------------------------------------------------
    /// Run the complete 2-stage Pages AI pipeline (Pages Align dewarping + Pages Shine restoration)
    #[arg(long)]
    ai: bool,

    /// Run only Stage 1 AI: neural document corner detection, boundary localization & perspective dewarping
    #[arg(long)]
    ai_dewarp: bool,

    /// Run only Stage 2 AI: neural document illumination shadow removal, bleed-through purge & text enhancement
    #[arg(long)]
    ai_clean: bool,

    /// Hardware acceleration provider (auto, directml, openvino, cpu)
    #[arg(long, default_value = "auto")]
    device: String,

    /// Low-power PC optimization: uses smaller inference tiles and lower overlap for maximum speed on weak CPUs/laptops
    #[arg(long)]
    fast: bool,

    /// Explicit tile size for sliding-window neural inference (default: 512, or 384 in fast mode)
    #[arg(long)]
    tile_size: Option<usize>,

    /// Overlap margin in pixels between sliding tiles (default: 64, or 32 in fast mode)
    #[arg(long)]
    overlap: Option<usize>,

    /// Custom directory containing ONNX models (pages_geom_v1.onnx & pages_clean_v1.onnx)
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Disable Huber edge gradient refinement along paper boundary
    #[arg(long)]
    no_refine: bool,

    /// Disable fine content deskewing
    #[arg(long)]
    no_deskew: bool,

    /// Disable background border spill cleanup
    #[arg(long)]
    no_clean_borders: bool,

    // ------------------------------------------------------------------------
    // Classical Scan Tailor Engine Flags
    // ------------------------------------------------------------------------
    /// Target DPI for output pages
    #[arg(long, default_value_t = 300)]
    dpi: u32,

    /// Page layout mode (auto-detect, single page, or two pages)
    #[arg(long, value_enum, default_value_t = CliLayout::Auto)]
    layout: CliLayout,

    /// Output color format (black-and-white, grayscale, color, or magic color)
    #[arg(long, value_enum, default_value_t = CliOutputMode::Color)]
    mode: CliOutputMode,

    /// Apply 2D Gaussian background illumination normalization (phone camera de-shadowing)
    #[arg(long)]
    deshadow: bool,

    /// Purge isolated toner dust and noise specks
    #[arg(long)]
    despeckle: bool,

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

    /// Emit machine-readable JSON streaming events to stdout (designed for DOCX-AI and Tauri IPC)
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
#[serde(tag = "event")]
enum JsonEvent<'a> {
    #[serde(rename = "start")]
    Start { total: usize },

    #[serde(rename = "ai_init")]
    AiInit {
        device: &'a str,
        device_choice: &'a str,
    },

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
        corners: Option<[[f64; 2]; 4]>,
        deskew_deg: f64,
        latency_ms: f64,
    },

    #[serde(rename = "analysis")]
    Analysis { pages: &'a [PageAnalysisDto] },

    #[serde(rename = "complete")]
    Complete {
        success: bool,
        pages_processed: usize,
        files: &'a [PathBuf],
        elapsed_sec: f64,
        hardware_device: Option<&'a str>,
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

    let input_path = args
        .input
        .as_ref()
        .or_else(|| args.project.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Either --input or --project must be specified (use --help for options)"))?;

    let output_dir = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("output"));
    std::fs::create_dir_all(&output_dir)?;

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

    let is_ai_run = args.ai || args.ai_dewarp || args.ai_clean;

    if is_ai_run {
        let device_choice: AiDeviceChoice = args.device.parse().unwrap_or(AiDeviceChoice::Auto);

        let model_paths = discover_models(args.model_dir.as_deref())?;
        let engine = PagesAiEngine::load_from_paths(
            &model_paths.geom_path,
            &model_paths.clean_path,
            device_choice,
        )?;

        let (dev_name, dev_pref) = engine.device_info();

        if args.json {
            emit_json(&JsonEvent::AiInit {
                device: dev_name,
                device_choice: &format!("{:?}", dev_pref),
            });
            emit_json(&JsonEvent::Start {
                total: images.len(),
            });
        } else {
            println!("================================================================");
            println!(" Pages AI — Neural Restoration & Geometry Engine");
            println!(" Target Compute Device: {}", dev_name);
            println!(" Models: {} | {}", model_paths.geom_path.display(), model_paths.clean_path.display());
            println!("================================================================");
            println!("Processing {} images...", images.len());
        }

        let tile_size = if args.fast {
            384
        } else {
            args.tile_size.unwrap_or(512)
        };

        let overlap = if args.fast {
            32
        } else {
            args.overlap.unwrap_or(64)
        };

        let pipeline_opts = AiPipelineOptions {
            dewarp: args.ai || args.ai_dewarp,
            clean: args.ai || args.ai_clean,
            refine_edges: !args.no_refine,
            auto_deskew: !args.no_deskew,
            clean_borders: !args.no_clean_borders,
            tile_size,
            overlap,
            fast_mode: args.fast,
            device: device_choice,
        };

        let start_time = std::time::Instant::now();
        let mut output_files = Vec::new();

        for (idx, img_path) in images.iter().enumerate() {
            let dyn_img = image::open(img_path)
                .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", img_path.display()))?;

            let rgb_img = dyn_img.to_rgb8();
            let (res_img, report) = engine.run_pipeline(&rgb_img, &pipeline_opts)?;

            let stem = img_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("page");
            let out_file = output_dir.join(format!("{}_ai_restored.png", stem));
            res_img.save(&out_file)?;
            output_files.push(out_file.clone());

            let done = idx + 1;
            let total = images.len();
            let percent = (done as f32 / total as f32) * 100.0;

            let corners_arr = report.corners.map(|c| {
                [
                    [c[0].x, c[0].y],
                    [c[1].x, c[1].y],
                    [c[2].x, c[2].y],
                    [c[3].x, c[3].y],
                ]
            });

            if args.json {
                emit_json(&JsonEvent::Progress {
                    done,
                    total,
                    percent,
                });
                emit_json(&JsonEvent::PageCompleted {
                    page: done,
                    total,
                    output_file: &out_file,
                    corners: corners_arr,
                    deskew_deg: report.detected_skew_deg,
                    latency_ms: report.total_latency_ms,
                });
            } else {
                let corners_str = if let Some(c) = report.corners {
                    format!(
                        "TL:({:.1},{:.1}) TR:({:.1},{:.1}) BR:({:.1},{:.1}) BL:({:.1},{:.1})",
                        c[0].x, c[0].y, c[1].x, c[1].y, c[2].x, c[2].y, c[3].x, c[3].y
                    )
                } else {
                    "N/A".to_string()
                };

                println!(
                    "[{}/{}] {:<20} | Time: {:>7.1}ms (Dewarp: {:.1}ms, Clean: {:.1}ms) | Skew: {:+.2}° | Corners: {}",
                    done,
                    total,
                    img_path.file_name().unwrap_or_default().to_string_lossy(),
                    report.total_latency_ms,
                    report.dewarp_latency_ms,
                    report.clean_latency_ms,
                    report.detected_skew_deg,
                    corners_str
                );
            }
        }

        let elapsed = start_time.elapsed();

        if args.json {
            emit_json(&JsonEvent::Complete {
                success: true,
                pages_processed: images.len(),
                files: &output_files,
                elapsed_sec: elapsed.as_secs_f64(),
                hardware_device: Some(dev_name),
            });
        } else {
            println!(
                "\n[SUCCESS] Processed {} pages in {:.2}s ({:.1} ms/page) using {}.",
                images.len(),
                elapsed.as_secs_f64(),
                (elapsed.as_secs_f64() * 1000.0) / images.len() as f64,
                dev_name
            );
            println!("Output directory: {}", output_dir.display());
        }

        return Ok(());
    }

    // Classical pipeline execution
    let mut proj = Project::new(output_dir);
    let split_mode = match args.layout {
        CliLayout::Auto => SplitMode::Auto,
        CliLayout::Single => SplitMode::SinglePage,
        CliLayout::TwoPages => SplitMode::TwoPages,
    };

    let output_mode = match args.mode {
        CliOutputMode::Bw => OutputMode::BlackAndWhite,
        CliOutputMode::Grayscale => OutputMode::Grayscale,
        CliOutputMode::Color => OutputMode::Color,
        CliOutputMode::Magic => OutputMode::MagicColor,
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

    for (idx, img_path) in images.iter().enumerate() {
        let mut page = PageSettings::new(img_path.clone(), idx + 1);
        page.dpi = args.dpi;
        page.split_mode = split_mode;
        page.output_mode = output_mode;
        page.orientation = orientation;
        page.margins = margins;
        page.deskew_angle = args.deskew;
        page.deshadow = args.deshadow;
        page.despeckle = args.despeckle;
        proj.pages.push(page);
    }

    if let Some(save_path) = &args.save_project {
        proj.save_to_file(save_path)?;
    }

    let total_pages = proj.pages.len();

    if args.json {
        emit_json(&JsonEvent::Start { total: total_pages });
    } else {
        println!("=== Pages Engine (Classical Pipeline) ===");
        println!("Output directory: {}", proj.output_directory.display());
        println!("Processing {} pages in parallel...", total_pages);
    }

    let start = std::time::Instant::now();

    let output_files = BatchProcessor::process_project(
        &proj,
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
                corners: None,
                deskew_deg: 0.0,
                latency_ms: 0.0,
            });
        }

        emit_json(&JsonEvent::Complete {
            success: true,
            pages_processed: total_pages,
            files: &output_files,
            elapsed_sec: elapsed.as_secs_f64(),
            hardware_device: None,
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

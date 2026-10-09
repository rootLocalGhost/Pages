use clap::{Parser, ValueEnum};
use pages_core::{BatchProcessor, PageSettings, Project, SplitMode};
use serde::Serialize;
use std::path::PathBuf;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum CliLayout {
    Auto,
    Single,
    TwoPages,
}

#[derive(Parser, Debug)]
#[command(
    name = "pages-cli",
    version,
    about = "Pages CLI - High-performance document & book post-processing engine"
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

    /// Save the generated project configuration to a JSON file
    #[arg(long)]
    save_project: Option<PathBuf>,

    /// Load settings from an existing project JSON file instead
    #[arg(short = 'p', long)]
    project: Option<PathBuf>,

    /// Emit machine-readable JSON progress and completion events to stdout
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
#[serde(tag = "event")]
enum JsonEvent<'a> {
    #[serde(rename = "progress")]
    Progress {
        done: usize,
        total: usize,
        percent: f32,
    },
    #[serde(rename = "complete")]
    Complete {
        success: bool,
        pages_processed: usize,
        files: &'a [PathBuf],
        elapsed_sec: f64,
    },
    #[serde(rename = "error")]
    Error {
        success: bool,
        error: &'a str,
    },
}

fn emit_json<T: Serialize>(event: &T) {
    if let Ok(line) = serde_json::to_string(event) {
        println!("{}", line);
    }
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

        let split_mode = match args.layout {
            CliLayout::Auto => SplitMode::Auto,
            CliLayout::Single => SplitMode::SinglePage,
            CliLayout::TwoPages => SplitMode::TwoPages,
        };

        for (idx, img_path) in images.into_iter().enumerate() {
            let mut page = PageSettings::new(img_path, idx + 1);
            page.dpi = args.dpi;
            page.split_mode = split_mode;
            proj.pages.push(page);
        }

        proj
    };

    if let Some(save_path) = &args.save_project {
        project.save_to_file(save_path)?;
    }

    if !args.json {
        println!("=== Scan Tailor Universal (Rust Core) ===");
        println!("Output directory: {}", project.output_directory.display());
        println!("Processing {} pages in parallel...", project.pages.len());
    }

    let start = std::time::Instant::now();
    let total_pages = project.pages.len();

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
        // Check if --json is in args
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

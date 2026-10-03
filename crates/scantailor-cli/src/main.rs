use clap::{Parser, ValueEnum};
use scantailor_core::{BatchProcessor, PageSettings, Project, SplitMode};
use std::path::PathBuf;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum CliLayout {
    Auto,
    Single,
    TwoPages,
}

#[derive(Parser, Debug)]
#[command(
    name = "scantailor-cli",
    version,
    about = "Scan Tailor Universal CLI - Interactive post-processing tool for scanned pages (Rust port)"
)]
struct Args {
    /// Input directory containing scan images or a single image file
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for processed pages
    #[arg(short, long)]
    output: PathBuf,

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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Scan Tailor Universal (Rust Core) ===");
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());

    let project = if let Some(proj_file) = &args.project {
        println!("Loading existing project from: {}", proj_file.display());
        Project::load_from_file(proj_file)?
    } else {
        let mut proj = Project::new(args.output.clone());

        let images = if args.input.is_dir() {
            BatchProcessor::discover_images(&args.input)?
        } else {
            vec![args.input.clone()]
        };

        if images.is_empty() {
            println!("No supported images found in input path.");
            return Ok(());
        }

        println!("Discovered {} images.", images.len());

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
        println!("Saving project file to: {}", save_path.display());
        project.save_to_file(save_path)?;
    }

    println!("\nProcessing {} pages in parallel...", project.pages.len());
    let start = std::time::Instant::now();

    BatchProcessor::process_project(&project, Some(&|done, total| {
        print!("\rProgress: [{}/{}] pages processed...", done, total);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }))?;

    let elapsed = start.elapsed();
    println!(
        "\nDone! Successfully processed {} pages in {:.2}s.",
        project.pages.len(),
        elapsed.as_secs_f64()
    );

    Ok(())
}

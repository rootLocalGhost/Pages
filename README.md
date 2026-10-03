# Scan Tailor Universal (Rust Core)

A high-performance, modular Rust port of the **Scan Tailor Universal** interactive post-processing engine for scanned book pages and documents.

This engine is completely decoupled from desktop UI frameworks, featuring a headless batch processing engine and CLI, ready for a modern **Tauri** desktop GUI in the future.

---

## Workspace Architecture

```
.
├── Cargo.toml               # Workspace manifest
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI workflow (Linux, Windows)
│       └── release.yml      # Automated GitHub Releases (Linux, Windows)
└── crates/
    ├── scantailor-math/     # 2D geometry, homography, splines, lines, polygons
    ├── scantailor-imageproc/# Bit-packed BinaryImage, Sauvola & Otsu binarization, CCL, deskew, page split
    ├── scantailor-dewarp/   # Cylindrical distortion models, curve fitting, backward grid remapping
    ├── scantailor-core/     # Post-processing pipeline, Serde project state, Rayon parallel batch runner
    └── scantailor-cli/      # High-performance headless CLI binary
```

---

## Features

- **Geometric & Math Foundations (`scantailor-math`)**:
  - Exact 2D floating-point & integer primitives (`Point2D`, `Line2D`, `Rect2D`, `Polygon2D`).
  - Line projection and partitioning (`ToLineProjector`, `sides_of_line`).
  - Line intersection and rectangle boundary clipping (`line_bounded_by_rect`).
  - 3x3 projective homography with matrix inversion and 4-point DLT estimation (`Homography2D`, `AffineTransform2D`).
  - Catmull-Rom cubic splines with tension control and equidistant polyline sampling (`CubicSpline2D`).
  - Multi-variable linear & quadratic function solvers and gradient estimators.

- **High-Speed Image Processing (`scantailor-imageproc`)**:
  - `BinaryImage`: Memory-efficient bit-packed raster format ($1 = \text{black/ink}$, $0 = \text{white/paper}$) with word-level bitwise manipulation.
  - Adaptive Sauvola binarization with $O(1)$ computation via 2D integral tables.
  - Global Otsu thresholding.
  - Binary morphological operators (`dilate`, `erode`, `open`, `close`) using rectangular structuring elements (`Brick`).
  - Two-pass 8-connected Connected Component Labeling (`ConnectedComponentLabeler`) with union-find and noise `despeckle`.
  - Automatic `SkewFinder`: Sub-tenth-of-a-degree deskew angle estimation via vertical shear projection variance and plateau centering.
  - Automatic `PageSplitDetector`: Book gutter/spine seam detection for splitting 2-page scans into individual pages.
  - `detect_content_box`: Automatic bounding-box detection ignoring edge shadows and scanner borders.

- **Curved Page Dewarping (`scantailor-dewarp`)**:
  - Top and bottom curve parameterization with arc length integration (`DewarpCurve`).
  - 2.5D ruling-based cylindrical distortion model (`CylindricalSurfaceModel`).
  - Backward coordinate grid mapping with bilinear interpolation (`DewarpRemapper`).

- **Batch Pipeline & Project State (`scantailor-core`)**:
  - Multi-stage pipeline: `Orientation` ➔ `PageSplit` ➔ `Deskew` ➔ `SelectContent` ➔ `Margins` ➔ `Output`.
  - Full multithreaded processing across CPU cores using `rayon`.
  - Serializable/deserializable project format (`Project`, `PageSettings`) in JSON.

---

## How to Use the CLI

### 1. Build from Source

Ensure you have Rust (1.80+) installed:

```bash
# Clone the repository
git clone https://github.com/trufanov-nok/scantailor-universal.git
cd scantailor-universal

# Run tests
cargo test --workspace

# Build the optimized release binary
cargo build --release
```

The compiled binary will be located at:
- **Windows**: `target\release\scantailor-cli.exe`
- **Linux**: `target/release/scantailor-cli`

---

### 2. Command Line Options

```
Usage: scantailor-cli [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
  -i, --input <INPUT>                Input directory containing scan images or a single image file
  -o, --output <OUTPUT>              Output directory for processed pages
      --dpi <DPI>                    Target DPI for output pages [default: 300]
      --layout <LAYOUT>              Page layout mode [default: auto] [possible values: auto, single, two-pages]
      --save-project <SAVE_PROJECT>  Save the generated project configuration to a JSON file
  -p, --project <PROJECT>            Load settings from an existing project JSON file instead
  -h, --help                         Print help
  -V, --version                      Print version
```

---

### 3. Usage Examples

#### Process a Folder of Book Scans
Automatically detects whether pages are dual-page scans, splits them into individual pages, deskews text lines, selects content, and writes binarized output images:

```bash
# Windows
.\target\release\scantailor-cli.exe -i "C:\Scans\RawBook" -o "C:\Scans\CleanedBook"

# Linux
./target/release/scantailor-cli -i /path/to/raw_scans -o /path/to/output
```

#### Enforce Single Page or Two Pages Split Mode
If all images are known to be two-page book spreads:
```bash
./target/release/scantailor-cli -i ./scans -o ./output --layout two-pages
```

If images are already single pages and should not be split:
```bash
./target/release/scantailor-cli -i ./scans -o ./output --layout single
```

#### Custom DPI
Set target output resolution (default is 300 DPI):
```bash
./target/release/scantailor-cli -i ./scans -o ./output --dpi 600
```

#### Save & Resume with Project Files
Export the detected settings into a reusable JSON project file:
```bash
./target/release/scantailor-cli -i ./scans -o ./output --save-project my_book.json
```

Re-run or batch process using the saved project configuration:
```bash
./target/release/scantailor-cli -p my_book.json -o ./output
```

---

## Automated GitHub Releases

This repository includes a GitHub Actions release workflow configured in [`.github/workflows/release.yml`](.github/workflows/release.yml).

### Supported Platforms
- **Linux x86_64** (`x86_64-unknown-linux-gnu`) ➔ `scantailor-cli-linux-x86_64.tar.gz`
- **Windows x86_64** (`x86_64-pc-windows-msvc`) ➔ `scantailor-cli-windows-x86_64.zip`

### How to Trigger a Release

#### Method 1: Push a Version Tag (Recommended)
Tag your commit with a version starting with `v` and push it:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will automatically:
1. Build optimized binaries on both Linux and Windows runners.
2. Package the binaries with `README.md` and `LICENSE`.
3. Create a new GitHub Release with release notes and attach the compiled archives.

#### Method 2: Manual Trigger (`workflow_dispatch`)
1. Go to your repository on GitHub.
2. Click on the **Actions** tab.
3. Select the **Release** workflow from the left sidebar.
4. Click **Run workflow** on the `master` (or `main`) branch.

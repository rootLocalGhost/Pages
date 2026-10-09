# Pages

A high-performance, modular document restoration and post-processing engine and desktop studio written in pure Rust and TypeScript.

Built from first mathematical and computer vision principles, **Pages** transforms raw digital scans, camera photos, and manuscripts into publication-ready, pristine, deskewed, cropped, and binarized documents.

---

## Workspace Architecture

```
.
├── Cargo.toml               # Workspace manifest (MIT Licensed)
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI workflow (Linux, Windows)
│       └── release.yml      # Automated GitHub Releases (Linux, Windows)
├── crates/
│   ├── pages-math/          # 2D geometry, projective homography, splines, lines, polygons
│   ├── pages-imageproc/     # Bit-packed BinaryImage, Sauvola & Otsu binarization, CCL, deskew, page split
│   ├── pages-dewarp/        # Cylindrical distortion models, curve fitting, backward grid remapping
│   ├── pages-core/          # Post-processing pipeline, Serde project state, Rayon parallel batch runner
│   └── pages-cli/           # High-performance headless CLI binary
├── src-tauri/               # Tauri v2 desktop shell
└── src/                     # SolidJS Neo-Brutalist UI workstation
```

---

## Core Capabilities

- **Geometric & Mathematical Foundations (`pages-math`)**:
  - Exact 2D floating-point & integer primitives (`Point2D`, `Line2D`, `Rect2D`, `Polygon2D`).
  - Line projection and partitioning (`ToLineProjector`, `sides_of_line`).
  - Boundary clipping algorithms (`line_bounded_by_rect`).
  - Projective homography matrix inversion and 4-point DLT estimation (`Homography2D`, `AffineTransform2D`).
  - Catmull-Rom cubic splines with tension control and equidistant polyline sampling (`CubicSpline2D`).
  - Multi-variable linear & quadratic function solvers and gradient estimators.

- **High-Speed Image Processing (`pages-imageproc`)**:
  - `BinaryImage`: Memory-efficient bit-packed raster format ($1 = \text{black/ink}$, $0 = \text{white/paper}$) with 64-bit word manipulation.
  - Adaptive Sauvola binarization with $O(1)$ constant time execution via 2D integral tables.
  - Global Otsu thresholding.
  - Binary morphological operators (`dilate`, `erode`, `open`, `close`) using rectangular structuring elements (`Brick`).
  - Two-pass 8-connected Connected Component Labeling (`ConnectedComponentLabeler`) with union-find and noise despeckling.
  - Automatic `SkewFinder`: Sub-tenth-of-a-degree deskew angle estimation via vertical shear projection variance and plateau centering.
  - Automatic `PageSplitDetector`: Book gutter/spine seam detection for splitting 2-page spreads into individual pages.
  - `detect_content_box`: Automatic content bounding box detection isolating text blocks from scanner margins and borders.

- **Curved Page Dewarping (`pages-dewarp`)**:
  - Top and bottom curve parameterization with arc length integration (`DewarpCurve`).
  - 2.5D ruling-based cylindrical distortion model (`CylindricalSurfaceModel`).
  - Backward coordinate grid mapping with bilinear interpolation (`DewarpRemapper`).

- **Batch Pipeline & Project State (`pages-core`)**:
  - 6-stage non-destructive pipeline: `Orientation` ➔ `PageSplit` ➔ `Deskew` ➔ `SelectContent` ➔ `Margins` ➔ `Output`.
  - Full multithreaded processing across CPU cores using `rayon`.
  - Serializable/deserializable project format (`Project`, `PageSettings`) in JSON.

- **Desktop Studio (`pages-gui` / Tauri v2 & SolidJS)**:
  - Neo-Brutalist responsive tactile workstation with Light & Dark themes.
  - CamScanner-grade instant page switching and canvas zooming/panning.
  - Interactive Before/After split curtain comparison slider.
  - Scrollable control sidebar with live stage previews and batch processors.

---

## Building from Source

Ensure you have Rust (1.80+) and Bun installed:

```bash
# Clone the repository
git clone https://github.com/your-username/pages.git
cd pages

# Run all Rust workspace tests
cargo test --workspace

# Build the headless CLI
cargo build --release -p pages-cli

# Run the desktop app in development
bun install
bun run tauri dev

# Build the desktop release installer
bun run tauri build
```

---

## Command Line Usage (`pages-cli`)

```
Usage: pages-cli [OPTIONS] --input <INPUT> --output <OUTPUT>

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

### CLI Examples

#### Batch Process a Folder of Scans
```bash
# Windows
.\target\release\pages-cli.exe -i "C:\Scans\RawBook" -o "C:\Scans\CleanedBook"

# Linux / macOS
./target/release/pages-cli -i /path/to/raw_scans -o /path/to/output
```

#### Enforce Two-Page Book Spread Split
```bash
./target/release/pages-cli -i ./scans -o ./output --layout two-pages
```

#### Save & Resume Project State
```bash
# Save analyzed settings
./target/release/pages-cli -i ./scans -o ./output --save-project book.json

# Resume or run using existing project configuration
./target/release/pages-cli -p book.json -o ./output
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

# Pages AI: Local Neural Engine Architecture & Training Blueprint

> **Hardware Target:** Intel Arc A770 (16 GB VRAM) via WSL2 (PyTorch 2.5+ upstream `torch.xpu`)  
> **Host Platform:** Pages App (Rust Backend + SolidJS / Tauri v2 Desktop GUI + Headless CLI)  
> **Target Runtime:** ONNX Runtime (`ort` crate) with OpenVINO / DirectML EP & AVX-512/AVX2 CPU Fallback  
> **Status:** Architecture Blueprint & Implementation Specification

---

## 1. Executive Overview

This document specifies the technical design, dataset acquisition, neural network architectures, training protocols, and native Rust embedding for the custom AI engine in **Pages**. 

Rather than relying on heavy general-purpose models or slow cloud APIs, **Pages AI** utilizes a modular, two-stage neural pipeline designed from scratch to run locally with ultra-low latency on consumer hardware:
1. **`PagesGeomNet` (Geometry & Dewarping)**: Detects document corners, boundaries, and 3D cylindrical folding deformation.
2. **`PagesCleanNet` (Restoration & Despeckle)**: Eliminates non-uniform illumination shadows, suppresses ink bleed-through, removes toner speckles, and restores crisp character strokes.

```
                         RAW CAMERA / SCANNER INPUT
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │        Stage 1: PagesGeomNet            │
                │     • 4-Corner Heatmap Regression       │
                │     • Boundary Polygon Segmentation     │
                │     • Spine / Ridge Crease Detection    │
                └────────────────────┬────────────────────┘
                                     │
                     Backward Bilinear Coordinate Remap
                           (Accelerated in Rust)
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │        Stage 2: PagesCleanNet           │
                │     • 2D Illumination Shadow Purge      │
                │     • High-Frequency Edge Preservation  │
                │     • Connected Toner Despeckling       │
                │     • Contrast Normalization            │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                      RESTORED HIGH-RESOLUTION PAGE
```

---

## 2. Hardware & Training Environment

### 2.1 Intel Arc A770 Hardware Profile
- **GPU Architecture:** Intel Alchemist (Xe-HPG)
- **VRAM:** 16 GB GDDR6 (256-bit bus, 560 GB/s bandwidth)
- **Matrix Engines:** 512 Intel XMX (Xe Matrix eXtensions) engines
- **Floating Point Capabilities:** Native `bfloat16`, `float16`, and `int8` hardware acceleration

### 2.2 Training Runtime: WSL2 Ubuntu + Upstream PyTorch `xpu`
We bypass deprecated extension packages (such as legacy IPEX) in favor of PyTorch 2.5+'s official upstream `torch.xpu` backend.

```bash
# WSL2 Ubuntu 24.04 Setup
sudo apt-get update && sudo apt-get install -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    libze-dev hwloc clinfo

# Verify Intel Arc A770 in WSL2
clinfo | grep "Device Name"
# Output should show: Intel(R) Arc(TM) A770 Graphics

# Install Upstream PyTorch with XPU Support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

### 2.3 Verification Script (`verify_a770.py`)
```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"XPU Available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Device Name: {torch.xpu.get_device_name(0)}")
    print(f"Memory Available: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run test matrix multiplication using bfloat16 on XMX
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    y = torch.matmul(x, x)
    print(f"Tensor Compute Test: SUCCESS ({y.shape})")
```

---

## 3. Neural Architecture Specifications

### 3.1 Model 1: `PagesGeomNet` (Geometry, Boundary & Dewarping)

`PagesGeomNet` is an ultra-lightweight, dual-branch feature pyramid network engineered to produce spatial geometric priors from a resized $512 \times 512$ thumbnail of the document.

```
Input (B, 3, 512, 512)
        │
┌───────┴───────┐
│ Mobile-Stem   │ Conv 3x3 (Stride 2) -> Depthwise Separable Conv
└───────┬───────┘
        ▼
┌───────────────┐
│ Stage 1-4     │ Inverted Residual Blocks (Channels: 32 -> 64 -> 128 -> 256)
└───────┬───────┘
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Branch A: Corner Heatmaps     │ │ Branch B: Boundary & Spine    │
│ Sub-pixel Keypoint Conv       │ │ Bilinear Upsample + Skip Convs│
│ Outputs: 4 Channels (Corners) │ │ Outputs: 2 Channels (Mask &   │
│ Peak Softmax Coordinate Head  │ │          Book Spine Seam)     │
└───────────────────────────────┘ └───────────────────────────────┘
```

- **Parameters:** ~1.85 Million
- **Model Footprint (ONNX FP16):** ~3.7 MB
- **Inference Latency:** ~8 ms (Arc A770), ~35 ms (Modern CPU)
- **Loss Formulation:**
  $$\mathcal{L}_{\text{geom}} = \lambda_1 \mathcal{L}_{\text{wing}}(\hat{K}, K) + \lambda_2 \mathcal{L}_{\text{bce}}(\hat{M}, M) + \lambda_3 \mathcal{L}_{\text{dice}}(\hat{M}, M)$$
  Where $\mathcal{L}_{\text{wing}}$ provides robust gradients for small corner keypoint deviations without exploding on outliers.

---

### 3.2 Model 2: `PagesCleanNet` (Restoration, Shadows & Despeckle)

`PagesCleanNet` processes document image patches ($512 \times 512$ sliding window with $64\text{px}$ overlap) to remove uneven lighting, scanning shadows, bleed-through, and dust artifacts while locking character stroke fidelity.

It adopts a **Nonlinear Activation Free (NAF) Block** design, which eliminates expensive non-linear activations (GELU/ReLU) in favor of element-wise Simple Gates and Multi-Dconv Spatial Attention:

```
[Input Patch: X] 
       │
       ├─── LayerNorm ───> Depthwise Conv 3x3 ───> Simple Gate (Halve Channels)
       │                                                   │
       │                                            Spatial Attention
       │                                                   │
       │                                            Linear Projection
       │                                                   │
       └─── (+) <──────────────────────────────────────────┘
```

#### Architecture Parameters:
- **Encoder Stages:** 4 Stages (Channels: `[32, 64, 128, 256]`, Block count: `[2, 2, 4, 8]`)
- **Decoder Stages:** 4 Stages with Skip Connections
- **Parameters:** ~4.2 Million
- **Model Footprint (ONNX FP16):** ~8.4 MB (INT8 Quantized: ~4.3 MB)
- **Inference Latency:** ~35–45 ms (Arc A770), ~140 ms (Multi-threaded AVX-512 CPU)
- **Loss Formulation:**
  $$\mathcal{L}_{\text{clean}} = \mathcal{L}_{\text{Charbonnier}}(I_{\text{out}}, I_{\text{gt}}) + 0.2 \cdot \mathcal{L}_{\text{SSIM}}(I_{\text{out}}, I_{\text{gt}}) + 0.1 \cdot \mathcal{L}_{\text{Edge}}(|\nabla I_{\text{out}}|, |\nabla I_{\text{gt}}|)$$

---

## 4. Public Dataset Sourcing Strategy

Because manual dataset creation is out of scope, the training pipeline leverages established, open research datasets combined with an automated on-the-fly degradation synthesis engine.

### 4.1 Sourced Datasets

| Dataset | Provider / Source | Primary Utility | Size |
| :--- | :--- | :--- | :--- |
| **Doc3D** | UMass Amherst (`cvit.iiit.ac.in`) | 3D document warping, 4-corner keypoints, depth maps, unwarping UV grids. | 100,000+ renders |
| **SmartDoc** | ICDAR Benchmark (`smartdoc.univ-lr.fr`) | Real smartphone document captures with ground truth quadrilateral corners under motion blur and tilt. | 25,000+ frames |
| **SD-Doc (Jung et al.)** | Document Shadow Benchmark | Real-world shadowed documents with perfectly registered shadow-free digital reference scans. | 4,200 image pairs |
| **DIBCO & H-DIBCO** | ICDAR Competitions (2009–2019) | Heavily degraded historical manuscripts, bleed-through, faded toner, toner speckles, and ground truth binarization. | 3,500 samples |
| **Gutenberg Digital Corpus** | Project Gutenberg Open Archive | High-resolution pristine PDF book pages used as ground-truth clean roots for synthetic shadow injection. | 50,000+ pages |

### 4.2 On-The-Fly Degradation Pipeline (PyTorch `Dataset`)
During training of `PagesCleanNet`, pristine ground truth pages ($I_{\text{gt}}$) undergo automated dynamic degradations to generate paired inputs ($I_{\text{in}}$):
1. **2D Gaussian Illumination Shadows:** Random focal points, non-uniform ambient light drops, and book gutter darkness.
2. **Ink Bleed-Through Injection:** Blending the reverse side of another random text page with low alpha ($0.05 \le \alpha \le 0.25$) and spatial blur.
3. **Scanner Glass Dust / Speckle Injection:** Poisson-distributed micro-salt noise ($1\text{px} \times 1\text{px}$ to $4\text{px} \times 4\text{px}$) resembling dirty flatbed scanner beds.
4. **Mild Blur & Sensor Noise:** Variable Gaussian blur ($\sigma \in [0.2, 1.2]$) + Gaussian sensor shot noise.

---

## 5. Training Protocol on Intel Arc A770

### 5.1 Training Hyperparameters

```python
TRAINING_CONFIG = {
    "device": "xpu",
    "precision": torch.bfloat16,
    "batch_size": 16,
    "patch_size": 512,
    "optimizer": "AdamW",
    "lr_initial": 2e-4,
    "lr_min": 1e-6,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingWarmRestarts",
    "warmup_epochs": 5,
    "max_epochs": 80,
    "gradient_accumulation_steps": 2,  # Effective batch size = 32
}
```

### 5.2 PyTorch XPU Training Loop Outline
```python
import torch
from torch.amp import autocast

scaler = torch.xpu.amp.GradScaler(enabled=True)

for epoch in range(TRAINING_CONFIG["max_epochs"]):
    model.train()
    for batch_idx, (dirty_inputs, clean_targets) in enumerate(dataloader):
        dirty_inputs = dirty_inputs.to("xpu", non_blocking=True)
        clean_targets = clean_targets.to("xpu", non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="xpu", dtype=torch.bfloat16):
            outputs = model(dirty_inputs)
            loss = criterion(outputs, clean_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
```

---

## 6. ONNX Export & Rust Integration Architecture

### 6.1 Native ONNX Export Pipeline
Trained weights are exported to standardized ONNX graphs with dynamic batch and spatial dimensions:

```python
def export_pages_models(geom_model, clean_model):
    # 1. Export PagesGeomNet (Fixed 512x512 input for fast landmark detection)
    dummy_geom = torch.randn(1, 3, 512, 512, device="cpu")
    torch.onnx.export(
        geom_model.to("cpu"),
        dummy_geom,
        "pages_geom_v1.onnx",
        input_names=["input_image"],
        output_names=["corner_heatmaps", "boundary_mask"],
        opset_version=18,
    )

    # 2. Export PagesCleanNet (Dynamic tile resolution)
    dummy_clean = torch.randn(1, 3, 512, 512, device="cpu")
    torch.onnx.export(
        clean_model.to("cpu"),
        dummy_clean,
        "pages_clean_v1.onnx",
        input_names=["input_patch"],
        output_names=["restored_patch"],
        dynamic_axes={
            "input_patch": {0: "batch", 2: "height", 3: "width"},
            "restored_patch": {0: "batch", 2: "height", 3: "width"}
        },
        opset_version=18,
    )
```

### 6.2 Pure Rust Engine Integration (`ort` crate)

In `crates/pages-core/src/ai/mod.rs`:

```rust
use ort::{
    execution_providers::{DirectMLExecutionProvider, OpenVINOExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};

pub struct PagesAiEngine {
    geom_session: Session,
    clean_session: Session,
    device_name: String,
}

impl PagesAiEngine {
    pub fn new(model_dir: &std::path::Path) -> anyhow::Result<Self> {
        // 1. Try Intel OpenVINO Execution Provider (Peak Arc A770 performance)
        let openvino_ep = OpenVINOExecutionProvider::default()
            .with_device_type("GPU.0"); // Targets Intel Arc A770

        // 2. Try DirectML EP (Windows DirectX 12 hardware)
        let directml_ep = DirectMLExecutionProvider::default();

        // 3. Fallback to CPU multi-threading
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?;

        let geom_path = model_dir.join("pages_geom_v1.onnx");
        let clean_path = model_dir.join("pages_clean_v1.onnx");

        // Initialize sessions with hardware acceleration priority
        let geom_session = builder
            .clone()
            .with_execution_providers([openvino_ep.build(), directml_ep.build()])
            .commit_from_file(geom_path)?;

        let clean_session = builder
            .with_execution_providers([openvino_ep.build(), directml_ep.build()])
            .commit_from_file(clean_path)?;

        Ok(Self {
            geom_session,
            clean_session,
            device_name: "Intel Arc A770 (OpenVINO / DirectML Accelerated)".to_string(),
        })
    }
}
```

### 6.3 Seamless Sliding Window Inference in Rust
To process ultra-high resolution documents (e.g. $2400 \times 3200$ pixels @ 300 DPI) without out-of-memory errors or edge seam boundaries:
1. Divide the high-resolution image into overlapping $512 \times 512$ tiles with a $64\text{px}$ overlap margin.
2. Apply a 2D Hann / Raised-Cosine spatial weighting window to each tile.
3. Pass tiles through `clean_session` in parallel batches.
4. Accumulate overlapping reconstructed patches in an output floating-point buffer and normalize by accumulated weights.
5. **Result:** Perfectly seamless, artifact-free restoration across arbitrary image dimensions.

---

## 7. UI & UX Integration in Pages Studio

The neural features integrate directly into the existing Pages Neo-Brutalist 6-stage studio:

1. **Stage 3 (Deskew & Dewarping)**:
   - Toggle: `[⚡ AI Neural Dewarp & Bounds]`
   - Visual Feedback: Overlays predicted 4 corners in neon cyan and displays the 3D surface grid curvature in real time.
2. **Stage 6 (Output & Binarization)**:
   - Output Mode Option: `[⭐ Magic AI Color]` & `[📄 AI Crisp B&W]`
   - Controls:
     - `AI De-Shadow Intensity` (Slider 0–100%)
     - `Toner Despeckle Purge` (Toggle)
     - `Ink Bleed Suppression` (Slider 0–100%)
3. **System Telemetry Badge**:
   - Header badge in Pages Desktop: `⚡ Intel Arc A770 (OpenVINO AI Engine Active)` or `⚡ CPU Vector Engine (AVX-512)`.

---

## 8. Step-by-Step Execution Plan

```
Phase 1: Environment & Dataset Acquisition
  ├── Verify WSL2 PyTorch 2.5+ XPU setup on Intel Arc A770
  ├── Clone download script for Doc3D, SmartDoc, DIBCO, and SD-Doc
  └── Build synthetic degradation data-loader (shadows, specks, bleed)

Phase 2: Model Architecture Development
  ├── Implement PagesGeomNet in PyTorch
  ├── Implement PagesCleanNet in PyTorch
  └── Verify forward/backward pass memory utilization on Arc A770

Phase 3: Training & Validation
  ├── Train PagesGeomNet (80 Epochs, evaluate Keypoint RMSE & IoU)
  ├── Train PagesCleanNet (80 Epochs, evaluate PSNR & SSIM)
  └── Checkpoint top-performing weights

Phase 4: ONNX Export & Native Rust Embedding
  ├── Export models to ONNX (FP16 & INT8 calibration)
  ├── Integrate ort crate with OpenVINO / DirectML in crates/pages-core
  └── Implement seamless overlapping tiled inference in pure Rust

Phase 5: Studio UI Polish & Release
  ├── Add AI Dewarp handles and Magic AI controls to Pages SolidJS Studio
  └── Update CI/CD release workflow to package ONNX weight assets
```

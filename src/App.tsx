import { createSignal, onMount, For, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Stage,
  PageItem,
  StagePreviewResult,
  BatchProgressEvent,
  SystemInfo,
  ToastMessage,
  ContentBoxDto,
} from "./types";
import "./App.css";

const STAGES: { id: Stage; num: number; name: string; desc: string }[] = [
  { id: "orientation", num: 1, name: "Orientation", desc: "Rotate & flip scans" },
  { id: "split", num: 2, name: "Split Pages", desc: "Separate dual spreads" },
  { id: "deskew", num: 3, name: "Deskew", desc: "Sub-degree rotation correction" },
  { id: "content", num: 4, name: "Content Box", desc: "Detect & crop text body" },
  { id: "margins", num: 5, name: "Margins", desc: "Uniform layout & gutter margins" },
  { id: "output", num: 6, name: "Output", desc: "Sauvola binarize & dewarp" },
];

export default function App() {
  // Application State
  const [pages, setPages] = createSignal<PageItem[]>([]);
  const [selectedPageId, setSelectedPageId] = createSignal<string | null>(null);
  const [activeStage, setActiveStage] = createSignal<Stage>("orientation");
  const [previewResult, setPreviewResult] = createSignal<StagePreviewResult | null>(null);
  const [systemInfo, setSystemInfo] = createSignal<SystemInfo | null>(null);
  const [isLoading, setIsLoading] = createSignal(false);

  // Viewport / Canvas Transform State
  const [zoom, setZoom] = createSignal(1.0);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = createSignal(false);
  const [startPan, setStartPan] = createSignal({ x: 0, y: 0 });

  // Batch Processing State
  const [isBatchProcessing, setIsBatchProcessing] = createSignal(false);
  const [batchProgress, setBatchProgress] = createSignal<BatchProgressEvent>({
    current: 0,
    total: 0,
    percent: 0,
    message: "",
  });
  const [showBatchModal, setShowBatchModal] = createSignal(false);

  // Toast System State
  const [toasts, setToasts] = createSignal<ToastMessage[]>([]);

  const addToast = (text: string, type: "info" | "success" | "error" = "info") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, text, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  };

  const activePage = () => pages().find((p) => p.id === selectedPageId()) || null;

  // Initialize
  onMount(async () => {
    try {
      const info = await invoke<SystemInfo>("get_system_info");
      setSystemInfo(info);
    } catch {}

    // Listen for batch progress events from Tauri backend
    try {
      const { listen } = await import("@tauri-apps/api/event");
      await listen<BatchProgressEvent>("batch-progress", (event) => {
        setBatchProgress(event.payload);
      });
    } catch {}
  });

  // Load preview for currently selected page and stage
  const refreshPreview = async () => {
    const page = activePage();
    if (!page) {
      setPreviewResult(null);
      return;
    }
    setIsLoading(true);
    try {
      const res = await invoke<StagePreviewResult>("generate_stage_preview", {
        page,
        stage: activeStage(),
      });
      setPreviewResult(res);
    } catch (err: any) {
      addToast(`Preview error: ${err}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  // Import images dialog
  const handleImportImages = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{ name: "Scan Images", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] }],
      });

      if (!selected) return;

      const filePaths = Array.isArray(selected) ? selected : [selected];
      if (filePaths.length === 0) return;

      addToast(`Importing ${filePaths.length} scan pages...`, "info");
      const imported = await invoke<PageItem[]>("import_images", { paths: filePaths });

      setPages((prev) => [...prev, ...imported]);
      if (!selectedPageId() && imported.length > 0) {
        setSelectedPageId(imported[0].id);
      }
      addToast(`Successfully loaded ${imported.length} pages`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Import failed: ${err}`, "error");
    }
  };

  // Stage Switch
  const setStage = async (stage: Stage) => {
    setActiveStage(stage);
    await refreshPreview();
  };

  // Select page
  const selectPage = async (id: string) => {
    setSelectedPageId(id);
    await refreshPreview();
  };

  // Page Rotations
  const rotateCurrentPage = async (delta: number) => {
    const page = activePage();
    if (!page) return;
    const newOrient = (page.orientation + delta + 360) % 360;

    setPages((prev) =>
      prev.map((p) => (p.id === page.id ? { ...p, orientation: newOrient } : p))
    );
    await refreshPreview();
  };

  // Auto-Detects
  const runAutoDeskew = async () => {
    const page = activePage();
    if (!page) return;
    try {
      const detected = await invoke<number>("auto_detect_deskew", {
        imagePath: page.file_path,
        orientation: page.orientation,
      });
      setPages((prev) =>
        prev.map((p) =>
          p.id === page.id ? { ...p, deskew_angle: detected, auto_deskew_angle: detected } : p
        )
      );
      addToast(`Detected skew angle: ${detected.toFixed(2)}°`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Deskew detection failed: ${err}`, "error");
    }
  };

  const runAutoContentBox = async () => {
    const page = activePage();
    if (!page) return;
    try {
      const box = await invoke<ContentBoxDto>("auto_detect_content_box", {
        imagePath: page.file_path,
        orientation: page.orientation,
        deskewAngle: page.deskew_angle,
      });
      setPages((prev) =>
        prev.map((p) => (p.id === page.id ? { ...p, content_box: box } : p))
      );
      addToast(`Detected content box: ${box.width}x${box.height}`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Content detection failed: ${err}`, "error");
    }
  };

  // Batch Apply to Scope
  const applyStageToAllPages = async () => {
    const curr = activePage();
    if (!curr) return;
    const stage = activeStage();

    setPages((prev) =>
      prev.map((p) => {
        switch (stage) {
          case "orientation":
            return { ...p, orientation: curr.orientation };
          case "split":
            return { ...p, split_mode: curr.split_mode, split_x: curr.split_x };
          case "deskew":
            return { ...p, deskew_angle: curr.deskew_angle };
          case "margins":
            return { ...p, margins: { ...curr.margins } };
          case "output":
            return {
              ...p,
              output_mode: curr.output_mode,
              binarization_method: curr.binarization_method,
              sauvola_k: curr.sauvola_k,
              sauvola_window: curr.sauvola_window,
              despeckle: curr.despeckle,
              dewarp: curr.dewarp,
              dpi: curr.dpi,
            };
          default:
            return p;
        }
      })
    );
    addToast(`Applied ${stage} settings to all ${pages().length} pages!`, "success");
  };

  // Run Batch Processing
  const startBatchProcessing = async () => {
    if (pages().length === 0) {
      addToast("Please import scan pages first", "error");
      return;
    }

    try {
      const outputDir = await open({
        directory: true,
        multiple: false,
        title: "Select Output Folder for Processed Scans",
      });

      if (!outputDir) return;

      setShowBatchModal(true);
      setIsBatchProcessing(true);
      setBatchProgress({
        current: 0,
        total: pages().length,
        percent: 0,
        message: "Starting multithreaded processing with Rayon...",
      });

      const results = await invoke<string[]>("run_batch_processing", {
        pages: pages(),
        outputDir: outputDir as string,
      });

      addToast(`Batch complete! Saved ${results.length} pages to folder`, "success");
    } catch (err: any) {
      addToast(`Batch processing error: ${err}`, "error");
    } finally {
      setIsBatchProcessing(false);
    }
  };

  // Canvas Mouse Interactions
  const handleMouseDown = (e: MouseEvent) => {
    if (e.button === 0 || e.button === 1) {
      setIsPanning(true);
      setStartPan({ x: e.clientX - pan().x, y: e.clientY - pan().y });
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isPanning()) {
      setPan({ x: e.clientX - startPan().x, y: e.clientY - startPan().y });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.88;
    setZoom((z) => Math.min(Math.max(z * factor, 0.15), 6.0));
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div class="app-shell" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
      {/* 1. Header Bar */}
      <header class="header-bar">
        <div class="brand-section">
          <img src="/logo.svg" alt="ScanTailor-Next Logo" class="brand-logo" />
          <div class="brand-title">
            <span>ScanTailor-Next</span>
            <span class="brand-badge">Pure Rust</span>
          </div>
        </div>

        <div class="header-actions">
          <button class="header-btn" onClick={handleImportImages} title="Import image scans">
            <span>➕</span>
            <span>Import Scans</span>
          </button>
          <button
            class="header-btn primary"
            onClick={startBatchProcessing}
            disabled={pages().length === 0}
            title="Batch process all pages and export"
          >
            <span>⚡</span>
            <span>Batch Process ({pages().length})</span>
          </button>
        </div>
      </header>

      {/* 2. Interactive Stage Stepper Bar */}
      <nav class="stepper-bar">
        <div class="stages-list">
          <For each={STAGES}>
            {(st) => (
              <button
                class={`stage-tab ${activeStage() === st.id ? "active" : ""}`}
                onClick={() => setStage(st.id)}
                title={st.desc}
              >
                <span class="stage-num">{st.num}</span>
                <span>{st.name}</span>
              </button>
            )}
          </For>
        </div>

        <div class="stepper-actions">
          <button class="header-btn" onClick={refreshPreview} disabled={!activePage()}>
            <span>🔄</span>
            <span>Refresh Preview</span>
          </button>
        </div>
      </nav>

      {/* 3. Main 3-Column Workspace */}
      <main class="workspace-container">
        {/* Left Column: Filmstrip Thumbnails */}
        <aside class="sidebar-filmstrip">
          <div class="filmstrip-header">
            <span class="filmstrip-title">Pages</span>
            <span class="filmstrip-count">{pages().length}</span>
          </div>

          <div class="filmstrip-list">
            <Show
              when={pages().length > 0}
              fallback={
                <div style={{ padding: "20px", "text-align": "center", color: "var(--text-dim)" }}>
                  No scans loaded. Click "Import Scans" above to begin.
                </div>
              }
            >
              <For each={pages()}>
                {(page, idx) => (
                  <div
                    class={`filmstrip-item ${selectedPageId() === page.id ? "active" : ""}`}
                    onClick={() => selectPage(page.id)}
                  >
                    <div class="thumb-preview-box">
                      <Show
                        when={page.thumbnail_path}
                        fallback={<span style={{ color: "var(--text-dim)", "font-size": "10px" }}>P{idx() + 1}</span>}
                      >
                        <img
                          src={convertFileSrc(page.thumbnail_path!)}
                          alt={page.file_name}
                          class="thumb-img"
                        />
                      </Show>
                    </div>

                    <div class="thumb-info">
                      <span class="thumb-name">{idx() + 1}. {page.file_name}</span>
                      <span class="thumb-meta">{page.width}x{page.height} • {page.orientation}°</span>
                      <span class="thumb-badge">{page.output_mode.toUpperCase()}</span>
                    </div>
                  </div>
                )}
              </For>
            </Show>
          </div>
        </aside>

        {/* Center Column: Interactive Canvas Viewport */}
        <section class="viewport-center">
          <div
            class="canvas-container"
            onMouseDown={handleMouseDown}
            onWheel={handleWheel}
          >
            <Show
              when={previewResult()}
              fallback={
                <div class="canvas-placeholder">
                  <div class="placeholder-icon">📄</div>
                  <div class="placeholder-title">No Scan Selected</div>
                  <div class="placeholder-desc">
                    Import document or book scan pages from the top toolbar to begin processing.
                  </div>
                  <button class="header-btn primary" onClick={handleImportImages}>
                    Import Scans
                  </button>
                </div>
              }
            >
              <img
                src={convertFileSrc(previewResult()!.preview_path)}
                alt="Stage preview"
                class="stage-image-preview"
                style={{
                  transform: `translate(${pan().x}px, ${pan().y}px) scale(${zoom()})`,
                }}
              />
            </Show>
          </div>

          {/* Floating Canvas HUD */}
          <div class="canvas-hud">
            <button class="hud-btn" onClick={() => setZoom((z) => Math.max(z * 0.85, 0.15))} title="Zoom Out">
              🔍-
            </button>
            <span class="hud-zoom-val">{Math.round(zoom() * 100)}%</span>
            <button class="hud-btn" onClick={() => setZoom((z) => Math.min(z * 1.15, 6.0))} title="Zoom In">
              🔍+
            </button>
            <div class="hud-divider"></div>
            <button class="hud-btn" onClick={resetView} title="Reset to 100%">
              1:1
            </button>
            <button class="hud-btn" onClick={() => { setZoom(0.65); setPan({ x: 0, y: 0 }); }} title="Fit Screen">
              ⛶ Fit
            </button>
          </div>
        </section>

        {/* Right Column: Contextual Stage Inspector Panel */}
        <aside class="sidebar-inspector">
          <div class="inspector-header">
            <div class="inspector-stage-title">
              <span>Stage Inspector</span>
            </div>
            <span class="inspector-stage-tag">{activeStage()}</span>
          </div>

          <div class="inspector-body">
            <Show when={activePage()} fallback={<div style={{ color: "var(--text-dim)" }}>Select a page to configure settings.</div>}>
              {/* STAGE 1: ORIENTATION */}
              <Show when={activeStage() === "orientation"}>
                <div class="control-card">
                  <div class="card-title">Orientation Controls</div>
                  <div class="control-row">
                    <span class="control-label">Current Rotation</span>
                    <span class="control-value">{activePage()!.orientation}°</span>
                  </div>
                  <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "8px", "margin-top": "6px" }}>
                    <button class="action-btn" onClick={() => rotateCurrentPage(-90)}>
                      <span>↺</span> Rotate 90° CCW
                    </button>
                    <button class="action-btn" onClick={() => rotateCurrentPage(90)}>
                      <span>↻</span> Rotate 90° CW
                    </button>
                  </div>
                  <button class="action-btn secondary" onClick={() => rotateCurrentPage(180)} style={{ "margin-top": "4px" }}>
                    <span>🔄</span> Turn 180°
                  </button>
                </div>
              </Show>

              {/* STAGE 2: PAGE SPLIT */}
              <Show when={activeStage() === "split"}>
                <div class="control-card">
                  <div class="card-title">Layout & Page Splitting</div>
                  <div class="segment-group">
                    <button
                      class={`segment-btn ${activePage()!.split_mode === "single" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, split_mode: "single" } : p));
                        refreshPreview();
                      }}
                    >
                      Single Page
                    </button>
                    <button
                      class={`segment-btn ${activePage()!.split_mode === "two-pages" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, split_mode: "two-pages" } : p));
                        refreshPreview();
                      }}
                    >
                      Two Pages (Spread)
                    </button>
                    <button
                      class={`segment-btn ${activePage()!.split_mode === "auto" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, split_mode: "auto" } : p));
                        refreshPreview();
                      }}
                    >
                      Auto Detect
                    </button>
                  </div>
                </div>
              </Show>

              {/* STAGE 3: DESKEW */}
              <Show when={activeStage() === "deskew"}>
                <div class="control-card">
                  <div class="card-title">
                    <span>Angle Adjustment</span>
                    <span class="control-value">{activePage()!.deskew_angle.toFixed(2)}°</span>
                  </div>
                  <div class="slider-group">
                    <input
                      type="range"
                      min="-45"
                      max="45"
                      step="0.05"
                      class="slider-input"
                      value={activePage()!.deskew_angle}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value);
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, deskew_angle: val } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </div>
                  <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "8px", "margin-top": "6px" }}>
                    <button class="action-btn" onClick={runAutoDeskew}>
                      <span>⚡</span> Auto Deskew
                    </button>
                    <button
                      class="action-btn secondary"
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, deskew_angle: 0.0 } : p));
                        refreshPreview();
                      }}
                    >
                      Reset 0.0°
                    </button>
                  </div>
                </div>
              </Show>

              {/* STAGE 4: CONTENT BOX */}
              <Show when={activeStage() === "content"}>
                <div class="control-card">
                  <div class="card-title">Content Detection</div>
                  <button class="action-btn" onClick={runAutoContentBox}>
                    <span>🎯</span> Auto-Detect Content Box
                  </button>
                  <Show when={activePage()!.content_box}>
                    <div class="control-row">
                      <span class="control-label">Box Size</span>
                      <span class="control-value">{activePage()!.content_box!.width} x {activePage()!.content_box!.height} px</span>
                    </div>
                  </Show>
                </div>
              </Show>

              {/* STAGE 5: MARGINS */}
              <Show when={activeStage() === "margins"}>
                <div class="control-card">
                  <div class="card-title">Uniform Margins (mm)</div>
                  <div class="control-row">
                    <span class="control-label">Left</span>
                    <input
                      type="number"
                      style={{ width: "60px", background: "var(--bg-input)", color: "#fff", border: "1px solid var(--border-default)", padding: "3px 6px", "border-radius": "4px" }}
                      value={activePage()!.margins.left}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value) || 0;
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, left: val } } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </div>
                  <div class="control-row">
                    <span class="control-label">Right</span>
                    <input
                      type="number"
                      style={{ width: "60px", background: "var(--bg-input)", color: "#fff", border: "1px solid var(--border-default)", padding: "3px 6px", "border-radius": "4px" }}
                      value={activePage()!.margins.right}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value) || 0;
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, right: val } } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </div>
                  <div class="control-row">
                    <span class="control-label">Top</span>
                    <input
                      type="number"
                      style={{ width: "60px", background: "var(--bg-input)", color: "#fff", border: "1px solid var(--border-default)", padding: "3px 6px", "border-radius": "4px" }}
                      value={activePage()!.margins.top}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value) || 0;
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, top: val } } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </div>
                  <div class="control-row">
                    <span class="control-label">Bottom</span>
                    <input
                      type="number"
                      style={{ width: "60px", background: "var(--bg-input)", color: "#fff", border: "1px solid var(--border-default)", padding: "3px 6px", "border-radius": "4px" }}
                      value={activePage()!.margins.bottom}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value) || 0;
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, bottom: val } } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </div>
                </div>
              </Show>

              {/* STAGE 6: OUTPUT & BINARIZATION */}
              <Show when={activeStage() === "output"}>
                <div class="control-card">
                  <div class="card-title">Color & Output Mode</div>
                  <div class="segment-group">
                    <button
                      class={`segment-btn ${activePage()!.output_mode === "bw" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, output_mode: "bw" } : p));
                        refreshPreview();
                      }}
                    >
                      B&W
                    </button>
                    <button
                      class={`segment-btn ${activePage()!.output_mode === "grayscale" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, output_mode: "grayscale" } : p));
                        refreshPreview();
                      }}
                    >
                      Grayscale
                    </button>
                    <button
                      class={`segment-btn ${activePage()!.output_mode === "color" ? "active" : ""}`}
                      onClick={() => {
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, output_mode: "color" } : p));
                        refreshPreview();
                      }}
                    >
                      Color
                    </button>
                  </div>

                  <Show when={activePage()!.output_mode === "bw"}>
                    <div class="control-row" style={{ "margin-top": "6px" }}>
                      <span class="control-label">Sauvola Factor k</span>
                      <span class="control-value">{activePage()!.sauvola_k.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.45"
                      step="0.01"
                      class="slider-input"
                      value={activePage()!.sauvola_k}
                      onInput={(e) => {
                        const val = parseFloat(e.currentTarget.value);
                        setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, sauvola_k: val } : p));
                      }}
                      onChange={refreshPreview}
                    />
                  </Show>

                  <div class="toggle-row" style={{ "margin-top": "6px" }}>
                    <span class="control-label">Purge Toner Specks (Despeckle)</span>
                    <label class="toggle-switch">
                      <input
                        type="checkbox"
                        checked={activePage()!.despeckle}
                        onChange={(e) => {
                          const val = e.currentTarget.checked;
                          setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, despeckle: val } : p));
                          refreshPreview();
                        }}
                      />
                      <span class="toggle-slider"></span>
                    </label>
                  </div>

                  <div class="toggle-row">
                    <span class="control-label">2.5D Cylindrical Dewarp</span>
                    <label class="toggle-switch">
                      <input
                        type="checkbox"
                        checked={activePage()!.dewarp}
                        onChange={(e) => {
                          const val = e.currentTarget.checked;
                          setPages((prev) => prev.map((p) => p.id === activePage()!.id ? { ...p, dewarp: val } : p));
                          refreshPreview();
                        }}
                      />
                      <span class="toggle-slider"></span>
                    </label>
                  </div>
                </div>
              </Show>

              {/* Universal Scope Card */}
              <div class="scope-selector-card">
                <div class="scope-title">Apply Settings Scope</div>
                <div class="scope-btn-grid">
                  <button class="scope-btn" onClick={applyStageToAllPages}>
                    All Pages
                  </button>
                  <button class="scope-btn" onClick={() => addToast("Current page is active", "info")}>
                    This Page
                  </button>
                </div>
              </div>
            </Show>
          </div>
        </aside>
      </main>

      {/* 4. Bottom Status Bar */}
      <footer class="status-bar">
        <div class="status-left">
          <div style={{ display: "flex", "align-items": "center", gap: "6px" }}>
            <span class="status-dot"></span>
            <span>Rayon Thread Pool ({systemInfo()?.cpu_cores || 8} Cores)</span>
          </div>
          <span>•</span>
          <span>{pages().length} Pages in Project</span>
        </div>

        <div class="status-right">
          <span>{activePage() ? `${activePage()!.width}x${activePage()!.height} px` : "No Page Selected"}</span>
          <span>•</span>
          <span>ScanTailor-Next v0.2.0</span>
        </div>
      </footer>

      {/* 5. Batch Progress Modal */}
      <Show when={showBatchModal()}>
        <div class="modal-backdrop">
          <div class="modal-dialog">
            <div class="modal-header">
              <span class="modal-title">Batch Processing Pages</span>
              <Show when={!isBatchProcessing()}>
                <button class="modal-close-btn" onClick={() => setShowBatchModal(false)}>✕</button>
              </Show>
            </div>
            <div class="modal-body">
              <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px" }}>
                <span>{batchProgress().message}</span>
                <span style={{ "font-family": "var(--font-mono)", color: "var(--accent-glow)" }}>
                  {Math.round(batchProgress().percent)}%
                </span>
              </div>
              <div class="progress-track">
                <div class="progress-fill" style={{ width: `${batchProgress().percent}%` }}></div>
              </div>
              <Show when={!isBatchProcessing()}>
                <button class="header-btn primary" onClick={() => setShowBatchModal(false)}>
                  Close
                </button>
              </Show>
            </div>
          </div>
        </div>
      </Show>

      {/* 6. Toast Notification Container */}
      <div class="toast-container">
        <For each={toasts()}>
          {(t) => (
            <div class={`toast ${t.type}`}>
              <span>{t.type === "success" ? "✔" : t.type === "error" ? "✖" : "ℹ"}</span>
              <span>{t.text}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

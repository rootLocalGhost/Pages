import { createSignal, onMount, For, Show, createMemo } from "solid-js";
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

const STAGES: { id: Stage; num: number; name: string; desc: string; icon: string }[] = [
  { id: "orientation", num: 1, name: "Orientation", desc: "Rotate scans", icon: "↺" },
  { id: "split", num: 2, name: "Split Pages", desc: "Dual spread cut", icon: "✂" },
  { id: "deskew", num: 3, name: "Deskew", desc: "Fine straighten", icon: "📐" },
  { id: "content", num: 4, name: "Content Box", desc: "Detect & crop", icon: "⛶" },
  { id: "margins", num: 5, name: "Margins", desc: "Uniform layout", icon: "📏" },
  { id: "output", num: 6, name: "Output", desc: "Sauvola binarize", icon: "⚡" },
];

export default function App() {
  // Theme State
  const [theme, setTheme] = createSignal<"light" | "dark">(
    (localStorage.getItem("scantailor-theme") as "light" | "dark") || "light"
  );

  const toggleTheme = () => {
    const next = theme() === "light" ? "dark" : "light";
    setTheme(next);
    localStorage.setItem("scantailor-theme", next);
  };

  // Application State
  const [pages, setPages] = createSignal<PageItem[]>([]);
  const [selectedPageId, setSelectedPageId] = createSignal<string | null>(null);
  const [activeStage, setActiveStage] = createSignal<Stage>("orientation");
  const [previewResult, setPreviewResult] = createSignal<StagePreviewResult | null>(null);
  const [systemInfo, setSystemInfo] = createSignal<SystemInfo | null>(null);
  const [isLoading, setIsLoading] = createSignal(false);
  const [searchFilter, setSearchFilter] = createSignal("");
  const [filmstripCollapsed, setFilmstripCollapsed] = createSignal(false);

  // Viewport & Canvas Pan/Zoom State
  const [zoom, setZoom] = createSignal(0.85);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = createSignal(false);
  const [startPan, setStartPan] = createSignal({ x: 0, y: 0 });

  // Interactive Canvas Overlays
  const [showDeskewGrid, setShowDeskewGrid] = createSignal(true);
  const [curtainPos, setCurtainPos] = createSignal(0.5);
  const [isDraggingCurtain, setIsDraggingCurtain] = createSignal(false);
  const [isDraggingCutLine, setIsDraggingCutLine] = createSignal(false);

  // Content Box Interactive State
  const [isDraggingCropBox, setIsDraggingCropBox] = createSignal(false);
  const [cropDragHandle, setCropDragHandle] = createSignal<string | null>(null);
  const [cropDragStart, setCropDragStart] = createSignal({ mouseX: 0, mouseY: 0, box: { x: 0, y: 0, width: 0, height: 0 } });

  // Batch Processing State
  const [isBatchProcessing, setIsBatchProcessing] = createSignal(false);
  const [batchProgress, setBatchProgress] = createSignal<BatchProgressEvent>({
    current: 0,
    total: 0,
    percent: 0,
    message: "",
  });
  const [showBatchModal, setShowBatchModal] = createSignal(false);

  // Toast System
  const [toasts, setToasts] = createSignal<ToastMessage[]>([]);

  const addToast = (text: string, type: "info" | "success" | "error" = "info") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, text, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  const activePage = createMemo(() => pages().find((p) => p.id === selectedPageId()) || null);

  const filteredPages = createMemo(() => {
    const q = searchFilter().trim().toLowerCase();
    if (!q) return pages();
    return pages().filter((p) => p.file_name.toLowerCase().includes(q));
  });

  // App Mount Initialization
  onMount(async () => {
    try {
      const info = await invoke<SystemInfo>("get_system_info");
      setSystemInfo(info);
    } catch {}

    try {
      const { listen } = await import("@tauri-apps/api/event");
      await listen<BatchProgressEvent>("batch-progress", (event) => {
        setBatchProgress(event.payload);
      });
    } catch {}

    // Global Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key >= "1" && e.key <= "6") {
        const stageIdx = parseInt(e.key) - 1;
        if (STAGES[stageIdx]) {
          switchStage(STAGES[stageIdx].id);
        }
      } else if (e.key === "r" || e.key === "R") {
        rotatePage(90);
      } else if (e.key === "t" || e.key === "T") {
        toggleTheme();
      } else if (e.key === "0") {
        resetView();
      }
    });
  });

  // Refresh Preview for Active Page & Stage
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

      // If on Content stage and content_box is not yet set, use detected content box
      if (activeStage() === "content" && !page.content_box && res.auto_content_box) {
        setPages((prev) =>
          prev.map((p) => (p.id === page.id ? { ...p, content_box: res.auto_content_box } : p))
        );
      }
    } catch (err: any) {
      addToast(`Preview error: ${err}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  // Import Scan Images
  const handleImportImages = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{ name: "Scan Images", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] }],
      });

      if (!selected) return;
      const filePaths = Array.isArray(selected) ? selected : [selected];
      if (filePaths.length === 0) return;

      addToast(`Importing ${filePaths.length} scan files...`, "info");
      const imported = await invoke<PageItem[]>("import_images", { paths: filePaths });

      setPages((prev) => [...prev, ...imported]);
      if (!selectedPageId() && imported.length > 0) {
        setSelectedPageId(imported[0].id);
      }
      addToast(`Loaded ${imported.length} scan pages successfully!`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Import failed: ${err}`, "error");
    }
  };

  // Stage Switching
  const switchStage = async (stage: Stage) => {
    setActiveStage(stage);
    // If switching to Content and box is missing, calculate default
    const curr = activePage();
    if (stage === "content" && curr && !curr.content_box && previewResult()) {
      const w = previewResult()!.width || 2000;
      const h = previewResult()!.height || 3000;
      const defaultBox: ContentBoxDto = {
        x: Math.round(w * 0.08),
        y: Math.round(h * 0.08),
        width: Math.round(w * 0.84),
        height: Math.round(h * 0.84),
      };
      setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, content_box: defaultBox } : p)));
    }
    await refreshPreview();
  };

  // Select Page
  const selectPage = async (id: string) => {
    setSelectedPageId(id);
    await refreshPreview();
  };

  // Rotation
  const rotatePage = async (delta: number) => {
    const curr = activePage();
    if (!curr) return;
    const newRot = (curr.orientation + delta + 360) % 360;
    setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, orientation: newRot } : p)));
    await refreshPreview();
  };

  // Auto Orientation
  const handleAutoOrientation = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const rot = await invoke<number>("auto_detect_orientation", { imagePath: curr.file_path });
      setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, orientation: rot } : p)));
      addToast(`Orientation detected: ${rot}°`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Auto-orientation error: ${err}`, "error");
    }
  };

  // Auto Split
  const handleAutoSplit = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const split = await invoke<any>("auto_detect_split", {
        imagePath: curr.file_path,
        orientation: curr.orientation,
      });
      const mode = split.mode === "two-pages" ? "two-pages" : "single";
      setPages((prev) =>
        prev.map((p) =>
          p.id === curr.id
            ? { ...p, split_mode: mode, split_x: split.split_x ? Math.round(split.split_x) : null }
            : p
        )
      );
      addToast(`Detected split mode: ${mode}`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Auto-split error: ${err}`, "error");
    }
  };

  // Auto Deskew
  const handleAutoDeskew = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const angle = await invoke<number>("auto_detect_deskew", {
        imagePath: curr.file_path,
        orientation: curr.orientation,
      });
      setPages((prev) =>
        prev.map((p) => (p.id === curr.id ? { ...p, deskew_angle: angle, auto_deskew_angle: angle } : p))
      );
      addToast(`Detected deskew angle: ${angle.toFixed(2)}°`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Deskew calculation failed: ${err}`, "error");
    }
  };

  // Auto Content Box
  const handleAutoContentBox = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const box = await invoke<ContentBoxDto>("auto_detect_content_box", {
        imagePath: curr.file_path,
        orientation: curr.orientation,
        deskewAngle: curr.deskew_angle,
      });
      setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, content_box: box } : p)));
      addToast(`Content box detected: ${box.width}×${box.height} px`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Content box detection error: ${err}`, "error");
    }
  };

  // Content Box Presets
  const applyContentPreset = (mode: "full" | "margin85" | "book75") => {
    const curr = activePage();
    const prev = previewResult();
    if (!curr || !prev) return;
    const w = prev.width;
    const h = prev.height;

    let newBox: ContentBoxDto;
    if (mode === "full") {
      newBox = { x: 0, y: 0, width: w, height: h };
    } else if (mode === "margin85") {
      newBox = {
        x: Math.round(w * 0.075),
        y: Math.round(h * 0.075),
        width: Math.round(w * 0.85),
        height: Math.round(h * 0.85),
      };
    } else {
      newBox = {
        x: Math.round(w * 0.125),
        y: Math.round(h * 0.125),
        width: Math.round(w * 0.75),
        height: Math.round(h * 0.75),
      };
    }

    setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, content_box: newBox } : p)));
    addToast(`Applied content preset: ${mode}`, "info");
  };

  // Update Split Position
  const updateSplitX = async (val: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, split_x: val } : p)));
    await refreshPreview();
  };

  // Update Deskew Angle
  const updateDeskewAngle = async (angle: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, deskew_angle: angle } : p)));
    await refreshPreview();
  };

  // Update Sauvola Parameters
  const updateSauvola = async (k: number, window: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) =>
      prev.map((p) => (p.id === curr.id ? { ...p, sauvola_k: k, sauvola_window: window } : p))
    );
    await refreshPreview();
  };

  // Update Margins
  const updateMargins = async (field: keyof PageItem["margins"], val: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) =>
      prev.map((p) =>
        p.id === curr.id ? { ...p, margins: { ...p.margins, [field]: val } } : p
      )
    );
    await refreshPreview();
  };

  // Apply Stage to All Pages
  const applyStageToAll = async () => {
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
          case "content":
            return { ...p, content_box: curr.content_box ? { ...curr.content_box } : null };
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
    addToast(`Stage "${stage}" settings applied to all ${pages().length} pages!`, "success");
  };

  // Run Batch Processing
  const startBatchProcessing = async () => {
    if (pages().length === 0) {
      addToast("Please import scan pages before running batch", "error");
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
        message: "Rayon multithreaded engine initialized...",
      });

      const results = await invoke<string[]>("run_batch_processing", {
        pages: pages(),
        outputDir: outputDir as string,
      });

      addToast(`Batch complete! Processed ${results.length} scan pages successfully.`, "success");
    } catch (err: any) {
      addToast(`Batch processing error: ${err}`, "error");
    } finally {
      setIsBatchProcessing(false);
    }
  };

  // Canvas Viewport Pan / Zoom Handlers
  const handleCanvasMouseDown = (e: MouseEvent) => {
    if ((e.target as HTMLElement).classList.contains("canvas-viewport-root")) {
      setIsPanning(true);
      setStartPan({ x: e.clientX - pan().x, y: e.clientY - pan().y });
    }
  };

  const handleCanvasMouseMove = (e: MouseEvent) => {
    if (isPanning()) {
      setPan({ x: e.clientX - startPan().x, y: e.clientY - startPan().y });
    }

    // Split Line Drag
    if (isDraggingCutLine()) {
      const curr = activePage();
      const prev = previewResult();
      if (curr && prev && prev.width > 0) {
        const container = document.querySelector(".image-canvas-wrapper") as HTMLElement;
        if (container) {
          const rect = container.getBoundingClientRect();
          const relativeX = (e.clientX - rect.left) / rect.width;
          const clampedX = Math.max(0.05, Math.min(0.95, relativeX));
          const pixelX = Math.round(clampedX * prev.width);
          setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, split_x: pixelX } : p)));
        }
      }
    }

    // Curtain Slider Drag
    if (isDraggingCurtain()) {
      const container = document.querySelector(".image-canvas-wrapper") as HTMLElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const rel = (e.clientX - rect.left) / rect.width;
        setCurtainPos(Math.max(0.02, Math.min(0.98, rel)));
      }
    }

    // Content Box Drag or Resize
    if ((isDraggingCropBox() || cropDragHandle()) && previewResult()) {
      const container = document.querySelector(".image-canvas-wrapper") as HTMLElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const scaleX = previewResult()!.width / rect.width;
        const scaleY = previewResult()!.height / rect.height;
        const deltaX = (e.clientX - cropDragStart().mouseX) * scaleX;
        const deltaY = (e.clientY - cropDragStart().mouseY) * scaleY;
        const orig = cropDragStart().box;
        const curr = activePage();

        if (curr) {
          let nextBox = { ...orig };

          if (isDraggingCropBox()) {
            nextBox.x = Math.max(0, Math.min(previewResult()!.width - orig.width, Math.round(orig.x + deltaX)));
            nextBox.y = Math.max(0, Math.min(previewResult()!.height - orig.height, Math.round(orig.y + deltaY)));
          } else if (cropDragHandle()) {
            const h = cropDragHandle();
            if (h === "br") {
              nextBox.width = Math.max(50, Math.round(orig.width + deltaX));
              nextBox.height = Math.max(50, Math.round(orig.height + deltaY));
            } else if (h === "tl") {
              const newX = Math.min(orig.x + orig.width - 50, Math.round(orig.x + deltaX));
              const newY = Math.min(orig.y + orig.height - 50, Math.round(orig.y + deltaY));
              nextBox.width = orig.width + (orig.x - newX);
              nextBox.height = orig.height + (orig.y - newY);
              nextBox.x = newX;
              nextBox.y = newY;
            } else if (h === "r") {
              nextBox.width = Math.max(50, Math.round(orig.width + deltaX));
            } else if (h === "b") {
              nextBox.height = Math.max(50, Math.round(orig.height + deltaY));
            }
          }

          setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, content_box: nextBox } : p)));
        }
      }
    }
  };

  const handleCanvasMouseUp = () => {
    if (isPanning()) setIsPanning(false);
    if (isDraggingCutLine()) {
      setIsDraggingCutLine(false);
      refreshPreview();
    }
    if (isDraggingCurtain()) setIsDraggingCurtain(false);
    if (isDraggingCropBox()) setIsDraggingCropBox(false);
    if (cropDragHandle()) setCropDragHandle(null);
  };

  const handleCanvasWheel = (e: WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    setZoom((z) => Math.max(0.15, Math.min(5.0, z * factor)));
  };

  const resetView = () => {
    setZoom(0.85);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div
      class="neo-app"
      data-theme={theme()}
      onMouseMove={handleCanvasMouseMove}
      onMouseUp={handleCanvasMouseUp}
    >
      {/* Top Header Navbar */}
      <header class="neo-header">
        <div class="brand-row">
          <div class="brand-icon">⚡</div>
          <div class="brand-text-wrap">
            <span class="brand-name">ScanTailor Next</span>
            <span class="brand-tagline">Neo-Brutalist Digitization Studio</span>
          </div>
        </div>

        <div class="header-center-telemetry">
          <Show when={systemInfo()}>
            <div class="neo-pill-badge lime">
              <span class="status-dot green"></span>
              <span>Rayon Multithreading ({systemInfo()!.cpu_cores} Cores)</span>
            </div>
          </Show>

          <div class="neo-pill-badge blue">
            <span>Pages Loaded:</span>
            <span class="font-mono">{pages().length}</span>
          </div>
        </div>

        <div class="header-actions-row">
          <button class="neo-btn icon-only sm" onClick={toggleTheme} title="Toggle Light/Dark Theme">
            {theme() === "light" ? "🌙" : "☀️"}
          </button>

          <button class="neo-btn lime" onClick={handleImportImages}>
            📁 Import Scans
          </button>

          <button class="neo-btn purple" onClick={startBatchProcessing} disabled={pages().length === 0}>
            🚀 Render All ({pages().length})
          </button>
        </div>
      </header>

      {/* Pipeline Stepper Banner */}
      <nav class="neo-stepper-bar">
        <div class="stepper-tabs">
          <For each={STAGES}>
            {(stage) => (
              <button
                class={`stage-tab-btn ${activeStage() === stage.id ? "active" : ""}`}
                onClick={() => switchStage(stage.id)}
              >
                <div class="stage-index-badge">{stage.num}</div>
                <span>{stage.icon} {stage.name}</span>
              </button>
            )}
          </For>
        </div>

        <div>
          <button class="neo-btn sm" onClick={applyStageToAll} title="Apply stage settings to all pages">
            ⚡ Apply to All Pages
          </button>
        </div>
      </nav>

      {/* Main 3-Column Studio Layout */}
      <main class={`neo-workspace-grid ${filmstripCollapsed() ? "filmstrip-collapsed" : ""}`}>
        {/* Left Column: Filmstrip Sidebar */}
        <aside class="filmstrip-panel">
          <div class="filmstrip-header-row">
            <div class="filmstrip-heading">
              <span>Filmstrip</span>
              <span class="count-chip">{pages().length}</span>
            </div>
            <button
              class="neo-btn icon-only sm"
              onClick={() => setFilmstripCollapsed(!filmstripCollapsed())}
              title="Toggle Sidebar"
            >
              {filmstripCollapsed() ? "»" : "«"}
            </button>
          </div>

          <div class="search-box-wrap">
            <input
              type="text"
              class="neo-input"
              placeholder="Search page filename..."
              value={searchFilter()}
              onInput={(e) => setSearchFilter(e.currentTarget.value)}
            />
          </div>

          <div class="filmstrip-scroll">
            <Show
              when={pages().length > 0}
              fallback={
                <div style={{ "text-align": "center", "padding": "40px 10px", "color": "var(--ink-muted)" }}>
                  <div style={{ "font-size": "32px", "margin-bottom": "8px" }}>📄</div>
                  <div class="font-heading" style={{ "font-weight": "800", "margin-bottom": "6px" }}>No Scans Yet</div>
                  <p style={{ "font-size": "11px", "margin-bottom": "14px" }}>Import book or manuscript files to start</p>
                  <button class="neo-btn lime sm" onClick={handleImportImages}>
                    + Import Scans
                  </button>
                </div>
              }
            >
              <For each={filteredPages()}>
                {(page, idx) => (
                  <div
                    class={`neo-page-card ${page.id === selectedPageId() ? "active" : ""}`}
                    onClick={() => selectPage(page.id)}
                  >
                    <div class="page-thumb-box">
                      <span class="page-number-tag">#{idx() + 1}</span>
                      <Show
                        when={page.thumbnail_path}
                        fallback={<span style={{ "font-size": "20px" }}>📄</span>}
                      >
                        <img
                          src={convertFileSrc(page.thumbnail_path!)}
                          alt={page.file_name}
                          class="page-thumb-img"
                        />
                      </Show>
                    </div>

                    <div class="page-details">
                      <span class="page-title-text" title={page.file_name}>
                        {page.file_name}
                      </span>
                      <div class="page-metrics-text">
                        <span>{page.width} × {page.height} px</span>
                        <span>{page.dpi} DPI • {page.orientation}°</span>
                      </div>
                      <div class="page-badge-row">
                        <span class="sticker-badge blue">{page.output_mode}</span>
                        <Show when={page.deskew_angle !== 0}>
                          <span class="sticker-badge coral">{page.deskew_angle.toFixed(1)}°</span>
                        </Show>
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </Show>
          </div>
        </aside>

        {/* Center Column: Canvas Viewport & Overlays */}
        <section class="neo-canvas-column">
          <div
            class="canvas-viewport-root"
            onMouseDown={handleCanvasMouseDown}
            onWheel={handleCanvasWheel}
          >
            <Show
              when={previewResult()}
              fallback={
                <div style={{ "text-align": "center", "background": "var(--bg-surface)", "padding": "40px", "border": "var(--border-width) solid var(--border-color)", "border-radius": "var(--radius-lg)", "box-shadow": "var(--shadow-hard-lg)", "max-width": "460px" }}>
                  <div style={{ "font-size": "48px", "margin-bottom": "12px" }}>📖</div>
                  <h2 class="font-heading" style={{ "font-size": "18px", "font-weight": "800", "margin-bottom": "8px" }}>
                    ScanTailor Studio
                  </h2>
                  <p style={{ "font-size": "12px", "color": "var(--ink-muted)", "margin-bottom": "20px" }}>
                    Import documents or book scans from the top bar to inspect, deskew, and binarize.
                  </p>
                  <button class="neo-btn lime" onClick={handleImportImages}>
                    📁 Import Scan Files
                  </button>
                </div>
              }
            >
              <div
                class="image-canvas-wrapper"
                style={{
                  transform: `translate(${pan().x}px, ${pan().y}px) scale(${zoom()})`,
                }}
              >
                <img
                  src={convertFileSrc(previewResult()!.preview_path)}
                  alt="Stage preview"
                  class="canvas-image"
                />

                {/* Stage 2 Overlay: Split Cutter Line */}
                <Show when={activeStage() === "split" && activePage()?.split_mode === "two-pages"}>
                  <div class="split-cutter-container">
                    <span class="zone-indicator-badge left">[ Page 1: Left ]</span>
                    <span class="zone-indicator-badge right">[ Page 2: Right ]</span>
                    <div
                      class="split-guide-bar"
                      style={{
                        left: `${
                          activePage()?.split_x && previewResult()!.width > 0
                            ? (activePage()!.split_x! / previewResult()!.width) * 100
                            : 50
                        }%`,
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setIsDraggingCutLine(true);
                      }}
                    >
                      <div class="split-tag-sticker">✂ CUT HERE</div>
                    </div>
                  </div>
                </Show>

                {/* Stage 3 Overlay: Deskew Grid Alignment */}
                <Show when={activeStage() === "deskew" && showDeskewGrid()}>
                  <div
                    class="deskew-grid-lines"
                    style={{
                      transform: `rotate(${activePage()?.deskew_angle || 0}deg)`,
                    }}
                  ></div>
                </Show>

                {/* Stage 4 Overlay: Draggable & Resizable Content Box */}
                <Show when={activeStage() === "content" && activePage()?.content_box && previewResult()!.width > 0}>
                  <div
                    class="content-box-mask"
                    style={{
                      left: `${(activePage()!.content_box!.x / previewResult()!.width) * 100}%`,
                      top: `${(activePage()!.content_box!.y / previewResult()!.height) * 100}%`,
                      width: `${(activePage()!.content_box!.width / previewResult()!.width) * 100}%`,
                      height: `${(activePage()!.content_box!.height / previewResult()!.height) * 100}%`,
                    }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      setIsDraggingCropBox(true);
                      setCropDragStart({
                        mouseX: e.clientX,
                        mouseY: e.clientY,
                        box: { ...activePage()!.content_box! },
                      });
                    }}
                  >
                    <div class="content-box-badge">
                      {activePage()!.content_box!.width} × {activePage()!.content_box!.height} px
                    </div>
                    {/* Handles */}
                    <div
                      class="neo-handle tl"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("tl");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                    <div
                      class="neo-handle tr"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("tr");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                    <div
                      class="neo-handle bl"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("bl");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                    <div
                      class="neo-handle br"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("br");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                    <div
                      class="neo-handle r"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("r");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                    <div
                      class="neo-handle b"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setCropDragHandle("b");
                        setCropDragStart({ mouseX: e.clientX, mouseY: e.clientY, box: { ...activePage()!.content_box! } });
                      }}
                    ></div>
                  </div>
                </Show>

                {/* Stage 5 Overlay: Margin Guides */}
                <Show when={activeStage() === "margins"}>
                  <div class="margin-rule-v" style={{ left: `${activePage()?.margins.left || 10}%` }}></div>
                  <div class="margin-rule-v" style={{ right: `${activePage()?.margins.right || 10}%` }}></div>
                  <div class="margin-rule-h" style={{ top: `${activePage()?.margins.top || 10}%` }}></div>
                  <div class="margin-rule-h" style={{ bottom: `${activePage()?.margins.bottom || 10}%` }}></div>
                </Show>

                {/* Stage 6 Overlay: Output Split Curtain Comparison */}
                <Show when={activeStage() === "output"}>
                  <div
                    class="curtain-bar-wrap"
                    style={{ left: `${curtainPos() * 100}%` }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      setIsDraggingCurtain(true);
                    }}
                  >
                    <div class="curtain-disc">⇄</div>
                  </div>
                </Show>
              </div>
            </Show>
          </div>

          {/* Floating Canvas HUD */}
          <div class="neo-canvas-hud">
            <button class="neo-btn icon-only sm" onClick={() => setZoom((z) => Math.max(z * 0.85, 0.15))} title="Zoom Out">
              −
            </button>
            <span class="hud-pill-readout">{Math.round(zoom() * 100)}%</span>
            <button class="neo-btn icon-only sm" onClick={() => setZoom((z) => Math.min(z * 1.15, 5.0))} title="Zoom In">
              +
            </button>
            <button class="neo-btn sm" onClick={resetView} title="Reset Scale">
              1:1
            </button>
            <button class="neo-btn sm" onClick={() => { setZoom(0.7); setPan({ x: 0, y: 0 }); }} title="Fit Screen">
              ⛶ Fit
            </button>
          </div>
        </section>

        {/* Right Column: Stage Inspector Panel */}
        <aside class="neo-inspector">
          <Show
            when={activePage()}
            fallback={
              <div class="neo-card">
                <span style={{ "color": "var(--ink-muted)", "font-weight": "600" }}>
                  Select a scan page to inspect and calibrate parameters.
                </span>
              </div>
            }
          >
            {/* Stage Banner */}
            <div class="inspector-stage-banner">
              <div class="stage-banner-icon">
                {STAGES.find((s) => s.id === activeStage())?.icon}
              </div>
              <div>
                <div class="stage-banner-title">
                  {STAGES.find((s) => s.id === activeStage())?.name}
                </div>
                <div class="stage-banner-desc">
                  {STAGES.find((s) => s.id === activeStage())?.desc}
                </div>
              </div>
            </div>

            {/* Stage 1: Orientation */}
            <Show when={activeStage() === "orientation"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Rotate Page</span>
                  <span class="font-mono" style={{ "font-weight": "800" }}>{activePage()?.orientation}°</span>
                </div>
                <div class="neo-segmented-group">
                  <button class="neo-segment-btn" onClick={() => rotatePage(-90)}>
                    ↺ -90°
                  </button>
                  <button class="neo-segment-btn" onClick={() => rotatePage(180)}>
                    ⇅ 180°
                  </button>
                  <button class="neo-segment-btn" onClick={() => rotatePage(90)}>
                    ↻ +90°
                  </button>
                </div>
                <button class="neo-btn purple" onClick={handleAutoOrientation}>
                  ⚡ Auto Detect Orientation
                </button>
              </div>
            </Show>

            {/* Stage 2: Split Pages */}
            <Show when={activeStage() === "split"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Spread Split Mode</span>
                </div>
                <div class="neo-segmented-group">
                  <button
                    class={`neo-segment-btn ${activePage()?.split_mode === "single" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, split_mode: "single" } : p)));
                      refreshPreview();
                    }}
                  >
                    Single Page
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.split_mode === "two-pages" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, split_mode: "two-pages" } : p)));
                      refreshPreview();
                    }}
                  >
                    Two-Page Spread
                  </button>
                </div>

                <Show when={activePage()?.split_mode === "two-pages"}>
                  <div class="neo-slider-wrap">
                    <div class="slider-info-row">
                      <span class="slider-caption">Cutter Position (X)</span>
                      <span class="slider-val-tag">
                        {activePage()?.split_x || Math.round((previewResult()?.width || 2000) / 2)} px
                      </span>
                    </div>
                    <input
                      type="range"
                      min={100}
                      max={previewResult()?.width || 3000}
                      value={activePage()?.split_x || Math.round((previewResult()?.width || 2000) / 2)}
                      class="neo-range-slider"
                      onInput={(e) => updateSplitX(parseInt(e.currentTarget.value))}
                    />
                  </div>
                </Show>

                <button class="neo-btn orange" onClick={handleAutoSplit}>
                  ✂ Auto Detect Split
                </button>
              </div>
            </Show>

            {/* Stage 3: Deskew */}
            <Show when={activeStage() === "deskew"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Deskew Angle Calibration</span>
                  <span class="slider-val-tag">
                    {activePage()?.deskew_angle.toFixed(2)}°
                  </span>
                </div>
                <input
                  type="range"
                  min="-20.0"
                  max="20.0"
                  step="0.05"
                  value={activePage()?.deskew_angle || 0}
                  class="neo-range-slider"
                  onInput={(e) => updateDeskewAngle(parseFloat(e.currentTarget.value))}
                />
                <div class="neo-segmented-group">
                  <button class="neo-segment-btn" onClick={() => updateDeskewAngle((activePage()?.deskew_angle || 0) - 0.2)}>
                    -0.2°
                  </button>
                  <button class="neo-segment-btn" onClick={() => updateDeskewAngle(0)}>
                    Reset 0°
                  </button>
                  <button class="neo-segment-btn" onClick={() => updateDeskewAngle((activePage()?.deskew_angle || 0) + 0.2)}>
                    +0.2°
                  </button>
                </div>
                <div class="neo-toggle-box" onClick={() => setShowDeskewGrid(!showDeskewGrid())}>
                  <div>
                    <div class="toggle-title-bold">Alignment Grid</div>
                    <div class="toggle-sub-hint">Visual horizontal & vertical guide lines</div>
                  </div>
                  <div class={`neo-switch ${showDeskewGrid() ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>
                <button class="neo-btn purple" onClick={handleAutoDeskew}>
                  📐 Auto Straighten Deskew
                </button>
              </div>
            </Show>

            {/* Stage 4: Content Box */}
            <Show when={activeStage() === "content"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Content Bounding Box</span>
                  <Show when={activePage()?.content_box}>
                    <span class="slider-val-tag">
                      {activePage()!.content_box!.width}×{activePage()!.content_box!.height}
                    </span>
                  </Show>
                </div>

                <div class="neo-segmented-group">
                  <button class="neo-segment-btn" onClick={() => applyContentPreset("full")}>
                    100% Full
                  </button>
                  <button class="neo-segment-btn" onClick={() => applyContentPreset("margin85")}>
                    85% Margin
                  </button>
                  <button class="neo-segment-btn" onClick={() => applyContentPreset("book75")}>
                    75% Book
                  </button>
                </div>

                <Show when={activePage()?.content_box}>
                  <div class="neo-grid-2x2">
                    <div class="input-field-block">
                      <span class="input-label-text">X Offset</span>
                      <div class="neo-unit-input">
                        <input
                          type="number"
                          value={activePage()!.content_box!.x}
                          onInput={(e) => {
                            const val = parseInt(e.currentTarget.value) || 0;
                            setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, content_box: { ...p.content_box!, x: val } } : p));
                          }}
                        />
                        <span class="unit-label">px</span>
                      </div>
                    </div>
                    <div class="input-field-block">
                      <span class="input-label-text">Y Offset</span>
                      <div class="neo-unit-input">
                        <input
                          type="number"
                          value={activePage()!.content_box!.y}
                          onInput={(e) => {
                            const val = parseInt(e.currentTarget.value) || 0;
                            setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, content_box: { ...p.content_box!, y: val } } : p));
                          }}
                        />
                        <span class="unit-label">px</span>
                      </div>
                    </div>
                    <div class="input-field-block">
                      <span class="input-label-text">Width</span>
                      <div class="neo-unit-input">
                        <input
                          type="number"
                          value={activePage()!.content_box!.width}
                          onInput={(e) => {
                            const val = parseInt(e.currentTarget.value) || 100;
                            setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, content_box: { ...p.content_box!, width: val } } : p));
                          }}
                        />
                        <span class="unit-label">px</span>
                      </div>
                    </div>
                    <div class="input-field-block">
                      <span class="input-label-text">Height</span>
                      <div class="neo-unit-input">
                        <input
                          type="number"
                          value={activePage()!.content_box!.height}
                          onInput={(e) => {
                            const val = parseInt(e.currentTarget.value) || 100;
                            setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, content_box: { ...p.content_box!, height: val } } : p));
                          }}
                        />
                        <span class="unit-label">px</span>
                      </div>
                    </div>
                  </div>
                </Show>

                <button class="neo-btn purple" onClick={handleAutoContentBox}>
                  ⛶ Auto Detect Content (AI)
                </button>
              </div>
            </Show>

            {/* Stage 5: Margins */}
            <Show when={activeStage() === "margins"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Margins (mm)</span>
                </div>
                <div class="neo-grid-2x2">
                  <div class="input-field-block">
                    <span class="input-label-text">Top</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.top || 10}
                        onInput={(e) => updateMargins("top", parseFloat(e.currentTarget.value) || 0)}
                      />
                      <span class="unit-label">mm</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Bottom</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.bottom || 10}
                        onInput={(e) => updateMargins("bottom", parseFloat(e.currentTarget.value) || 0)}
                      />
                      <span class="unit-label">mm</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Left</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.left || 10}
                        onInput={(e) => updateMargins("left", parseFloat(e.currentTarget.value) || 0)}
                      />
                      <span class="unit-label">mm</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Right</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.right || 10}
                        onInput={(e) => updateMargins("right", parseFloat(e.currentTarget.value) || 0)}
                      />
                      <span class="unit-label">mm</span>
                    </div>
                  </div>
                </div>
              </div>
            </Show>

            {/* Stage 6: Output & Sauvola */}
            <Show when={activeStage() === "output"}>
              <div class="neo-card">
                <div class="neo-card-header">
                  <span class="neo-card-title">Binarization Method</span>
                </div>
                <div class="neo-segmented-group">
                  <button
                    class={`neo-segment-btn ${activePage()?.binarization_method === "sauvola" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, binarization_method: "sauvola" } : p));
                      refreshPreview();
                    }}
                  >
                    Sauvola
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.binarization_method === "otsu" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, binarization_method: "otsu" } : p));
                      refreshPreview();
                    }}
                  >
                    Otsu
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.output_mode === "grayscale" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, output_mode: "grayscale" } : p));
                      refreshPreview();
                    }}
                  >
                    Grayscale
                  </button>
                </div>

                <div class="neo-slider-wrap">
                  <div class="slider-info-row">
                    <span class="slider-caption">Sauvola Coefficient (k)</span>
                    <span class="slider-val-tag">{activePage()?.sauvola_k.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.50"
                    step="0.01"
                    value={activePage()?.sauvola_k || 0.20}
                    class="neo-range-slider"
                    onInput={(e) => updateSauvola(parseFloat(e.currentTarget.value), activePage()?.sauvola_window || 35)}
                  />
                </div>

                <div class="neo-slider-wrap">
                  <div class="slider-info-row">
                    <span class="slider-caption">Window Radius</span>
                    <span class="slider-val-tag">{activePage()?.sauvola_window} px</span>
                  </div>
                  <input
                    type="range"
                    min="11"
                    max="75"
                    step="2"
                    value={activePage()?.sauvola_window || 35}
                    class="neo-range-slider"
                    onInput={(e) => updateSauvola(activePage()?.sauvola_k || 0.20, parseInt(e.currentTarget.value))}
                  />
                </div>

                <div
                  class="neo-toggle-box"
                  onClick={() => {
                    const next = !activePage()?.despeckle;
                    setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, despeckle: next } : p));
                    refreshPreview();
                  }}
                >
                  <div>
                    <div class="toggle-title-bold">Despeckling Filter</div>
                    <div class="toggle-sub-hint">Filter dust and scanning noise</div>
                  </div>
                  <div class={`neo-switch ${activePage()?.despeckle ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                <div
                  class="neo-toggle-box"
                  onClick={() => {
                    const next = !activePage()?.dewarp;
                    setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, dewarp: next } : p));
                    refreshPreview();
                  }}
                >
                  <div>
                    <div class="toggle-title-bold">3D Cylindrical Dewarp</div>
                    <div class="toggle-sub-hint">Flatten book spine curvature</div>
                  </div>
                  <div class={`neo-switch ${activePage()?.dewarp ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>
              </div>
            </Show>

            {/* Batch Action Footer Card */}
            <div class="neo-batch-card">
              <span class="batch-card-title">Batch Processing</span>
              <button class="neo-btn sm" onClick={applyStageToAll}>
                ⚡ Apply Stage to All Pages
              </button>
              <button class="neo-btn lime" onClick={startBatchProcessing}>
                🚀 Render All ({pages().length} Pages)
              </button>
            </div>
          </Show>
        </aside>
      </main>

      {/* Batch Progress Modal */}
      <Show when={showBatchModal()}>
        <div class="neo-modal-backdrop">
          <div class="neo-modal-box">
            <div style={{ "display": "flex", "align-items": "center", "gap": "12px" }}>
              <div style={{ "font-size": "28px" }}>⚙️</div>
              <div>
                <h3 class="font-heading" style={{ "font-size": "16px", "font-weight": "800" }}>Batch Processing</h3>
                <span style={{ "font-size": "11px", "color": "var(--ink-muted)" }}>
                  Multi-threaded Rust Rayon Engine
                </span>
              </div>
            </div>

            <div class="neo-progress-track">
              <div
                class="neo-progress-bar"
                style={{ width: `${batchProgress().percent}%` }}
              ></div>
            </div>

            <div style={{ "display": "flex", "justify-content": "space-between", "font-family": "var(--font-mono)", "font-size": "12px", "font-weight": "700" }}>
              <span>{batchProgress().message}</span>
              <span>
                {batchProgress().current} / {batchProgress().total} ({Math.round(batchProgress().percent)}%)
              </span>
            </div>

            <div style={{ "display": "flex", "justify-content": "flex-end", "margin-top": "6px" }}>
              <button
                class="neo-btn sm"
                disabled={isBatchProcessing()}
                onClick={() => setShowBatchModal(false)}
              >
                {isBatchProcessing() ? "Processing..." : "Close"}
              </button>
            </div>
          </div>
        </div>
      </Show>

      {/* Toast Notifications */}
      <div class="neo-toast-box">
        <For each={toasts()}>
          {(t) => (
            <div class={`neo-toast-item ${t.type}`}>
              <span>{t.text}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

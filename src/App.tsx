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
  { id: "orientation", num: 1, name: "Orientation", desc: "Rotate & flip scans", icon: "↺" },
  { id: "split", num: 2, name: "Split Pages", desc: "Dual spread cutting", icon: "✂" },
  { id: "deskew", num: 3, name: "Deskew", desc: "Fine straighten angle", icon: "📐" },
  { id: "content", num: 4, name: "Content Box", desc: "Per-page detect & crop", icon: "⛶" },
  { id: "margins", num: 5, name: "Margins", desc: "Uniform layout borders", icon: "📏" },
  { id: "output", num: 6, name: "Output", desc: "Sauvola binarize & filters", icon: "⚡" },
];

export default function App() {
  // Theme State
  const [theme, setTheme] = createSignal<"light" | "dark">(
    (localStorage.getItem("pages-theme") as "light" | "dark") || "light"
  );

  const toggleTheme = () => {
    const next = theme() === "light" ? "dark" : "light";
    setTheme(next);
    localStorage.setItem("pages-theme", next);
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

  // Instant Display Cache (CamScanner-grade 0ms display on selection)
  const [instantPreviewSrc, setInstantPreviewSrc] = createSignal<string | null>(null);

  // Viewport & Canvas Pan/Zoom State
  const [zoom, setZoom] = createSignal(0.85);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = createSignal(false);
  const [startPan, setStartPan] = createSignal({ x: 0, y: 0 });

  // Interactive Canvas Overlays
  const [showDeskewGrid, setShowDeskewGrid] = createSignal(true);
  const [showCurtainComparison, setShowCurtainComparison] = createSignal(true);
  const [curtainPos, setCurtainPos] = createSignal(0.5);
  const [isDraggingCurtain, setIsDraggingCurtain] = createSignal(false);
  const [isDraggingCutLine, setIsDraggingCutLine] = createSignal(false);

  // Neural AI Engine State
  const [selectedAiDevice, setSelectedAiDevice] = createSignal("auto");
  const [isFastMode, setIsFastMode] = createSignal(false);
  const [activeAiCornerIndex, setActiveAiCornerIndex] = createSignal<number | null>(null);

  // Content Box Interactive State
  const [isDraggingCropBox, setIsDraggingCropBox] = createSignal(false);
  const [cropDragHandle, setCropDragHandle] = createSignal<string | null>(null);
  const [cropDragStart, setCropDragStart] = createSignal({ mouseX: 0, mouseY: 0, box: { x: 0, y: 0, width: 0, height: 0 } });

  // Batch Processing & Analysis State
  const [isBatchProcessing, setIsBatchProcessing] = createSignal(false);
  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysisStatus, setAnalysisStatus] = createSignal<BatchProgressEvent>({
    current: 0,
    total: 0,
    percent: 0,
    message: "",
  });
  const [showBatchModal, setShowBatchModal] = createSignal(false);

  // Freemium Licensing & Daily Quota State (LemonSqueezy preview & Developer code)
  const getTodayDateKey = () => new Date().toISOString().slice(0, 10);

  const getPagesUsage = (): number => {
    try {
      const raw = localStorage.getItem("pages_daily_quota");
      if (!raw) return 0;
      const data = JSON.parse(raw);
      if (data.date !== getTodayDateKey()) return 0;
      return typeof data.count === "number" ? data.count : 0;
    } catch {
      return 0;
    }
  };

  const [isProUnlocked, setIsProUnlocked] = createSignal(
    localStorage.getItem("pages_pro_unlocked") === "true"
  );
  const [dailyUsageCount, setDailyUsageCount] = createSignal(getPagesUsage());
  const [showLicenseModal, setShowLicenseModal] = createSignal(false);
  const [licenseModalReason, setLicenseModalReason] = createSignal<{
    type: "import" | "quota" | "manual";
    title: string;
    message: string;
    pendingPaths?: string[];
  }>({
    type: "manual",
    title: "Pages Pro & Quotas",
    message: "Manage license status and daily quotas.",
  });
  const [secretDevKey, setSecretDevKey] = createSignal("");
  const [devKeyError, setDevKeyError] = createSignal("");

  const incrementPagesUsage = (amount: number) => {
    const today = getTodayDateKey();
    const curr = getPagesUsage();
    const updated = curr + amount;
    localStorage.setItem("pages_daily_quota", JSON.stringify({ date: today, count: updated }));
    setDailyUsageCount(updated);
  };

  const unlockDeveloperPro = (key: string): boolean => {
    const validKeys = [
      "PAGES-DEV-2026",
      "DEV-GHOST-2026",
      "DEV-UNLIMITED",
      "DOCX-DEV-2026",
      "LEMON-DEV-ACCESS",
      "GHOST-DEV-KEY",
    ];
    if (validKeys.includes(key.trim().toUpperCase())) {
      localStorage.setItem("pages_pro_unlocked", "true");
      setIsProUnlocked(true);
      setDevKeyError("");
      addToast("Developer Pro License Activated! Unlimited access granted.", "success");
      return true;
    } else {
      setDevKeyError("Invalid developer secret code. Contact team or enter valid key.");
      return false;
    }
  };

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
        setAnalysisStatus(event.payload);
      });
    } catch {}

    // Global Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key >= "1" && e.key <= "6") {
        const stageIdx = parseInt(e.key) - 1;
        if (STAGES[stageIdx]) {
          setActiveStage(STAGES[stageIdx].id);
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

  // Debounced & Thread-safe Preview Refresh
  let previewDebounceTimer: any = null;
  let activeRequestId = 0;

  const refreshPreview = async () => {
    const page = activePage();
    if (!page) {
      setPreviewResult(null);
      setInstantPreviewSrc(null);
      return;
    }
    const reqId = ++activeRequestId;
    setIsLoading(true);
    try {
      const res = await invoke<StagePreviewResult>("generate_stage_preview", {
        page,
        stage: activeStage(),
      });
      // Only commit if this is the newest active request
      if (reqId === activeRequestId) {
        setPreviewResult(res);
        setInstantPreviewSrc(null); // High-res preview loaded!

        // If on Content stage and content_box is not yet set, use detected content box
        if (activeStage() === "content" && !page.content_box && res.auto_content_box) {
          setPages((prev) =>
            prev.map((p) => (p.id === page.id ? { ...p, content_box: res.auto_content_box } : p))
          );
        }
      }
    } catch (err: any) {
      if (reqId === activeRequestId) {
        addToast(`Preview error: ${err}`, "error");
      }
    } finally {
      if (reqId === activeRequestId) {
        setIsLoading(false);
      }
    }
  };

  const debouncedRefreshPreview = (delay = 140) => {
    if (previewDebounceTimer) clearTimeout(previewDebounceTimer);
    previewDebounceTimer = setTimeout(() => {
      refreshPreview();
    }, delay);
  };

  // Import Scan Images (Single primary import flow)
  const handleImportImages = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{ name: "Scan Images", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] }],
      });

      if (!selected) return;
      const filePaths = Array.isArray(selected) ? selected : [selected];
      if (filePaths.length === 0) return;

      if (!isProUnlocked() && filePaths.length > 5) {
        setLicenseModalReason({
          type: "import",
          title: "Import Limit (Free Tier: Max 5 Images)",
          message: `Free tier allows importing up to 5 images at once. You selected ${filePaths.length} scan files. Upgrade to Pages Pro or enter a Developer Secret Code for unlimited batch imports.`,
          pendingPaths: filePaths,
        });
        setShowLicenseModal(true);
        return;
      }

      addToast(`Importing ${filePaths.length} scan files...`, "info");
      const imported = await invoke<PageItem[]>("import_images", { paths: filePaths });

      setPages((prev) => [...prev, ...imported]);
      if (!selectedPageId() && imported.length > 0) {
        selectPage(imported[0].id);
      }
      addToast(`Loaded ${imported.length} scan pages successfully!`, "success");
    } catch (err: any) {
      addToast(`Import failed: ${err}`, "error");
    }
  };

  const importFirstFivePending = async () => {
    const paths = licenseModalReason().pendingPaths || [];
    const firstFive = paths.slice(0, 5);
    setShowLicenseModal(false);
    if (firstFive.length === 0) return;
    try {
      addToast(`Importing first ${firstFive.length} scan files (Free Tier)...`, "info");
      const imported = await invoke<PageItem[]>("import_images", { paths: firstFive });
      setPages((prev) => [...prev, ...imported]);
      if (!selectedPageId() && imported.length > 0) {
        selectPage(imported[0].id);
      }
      addToast(`Loaded ${imported.length} scan pages successfully!`, "success");
    } catch (err: any) {
      addToast(`Import failed: ${err}`, "error");
    }
  };

  // Select Page (Instant 0ms display via cached thumbnail)
  const selectPage = async (id: string) => {
    setSelectedPageId(id);
    const p = pages().find((item) => item.id === id);
    if (p?.thumbnail_path) {
      setInstantPreviewSrc(convertFileSrc(p.thumbnail_path));
    }
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

  // Auto Orientation (Current Page)
  const handleAutoOrientation = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const rot = await invoke<number>("auto_detect_orientation", { imagePath: curr.file_path });
      setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, orientation: rot } : p)));
      addToast(`Orientation detected: ${rot}°`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Orientation detection failed: ${err}`, "error");
    }
  };

  // Auto Split (Current Page)
  const handleAutoSplit = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const res = await invoke<any>("auto_detect_split", {
        imagePath: curr.file_path,
        orientation: curr.orientation,
      });
      setPages((prev) =>
        prev.map((p) =>
          p.id === curr.id
            ? {
                ...p,
                split_mode: res.mode === "two-pages" ? "two-pages" : "single",
                split_x: res.split_x || p.split_x,
              }
            : p
        )
      );
      addToast(`Detected split mode: ${res.mode}`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Split detection failed: ${err}`, "error");
    }
  };

  // Auto Deskew (Current Page)
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
      addToast(`Straightened page: ${angle.toFixed(2)}°`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Deskew calculation failed: ${err}`, "error");
    }
  };

  // Batch Auto-Deskew (Individual Calculation per Image)
  const handleBatchAutoDeskew = async () => {
    if (pages().length === 0) return;
    setIsAnalyzing(true);
    setAnalysisStatus({
      current: 0,
      total: pages().length,
      percent: 0,
      message: "Straightening all pages individually...",
    });
    try {
      const updated = await invoke<PageItem[]>("batch_auto_deskew", { pages: pages() });
      setPages(updated);
      addToast(`Straightened ${updated.length} pages individually!`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Batch deskew failed: ${err}`, "error");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Auto Content Box (Current Page)
  const handleAutoContentBox = async () => {
    const curr = activePage();
    if (!curr) return;
    try {
      const bbox = await invoke<ContentBoxDto>("auto_detect_content_box", {
        imagePath: curr.file_path,
        orientation: curr.orientation,
        deskewAngle: curr.deskew_angle,
      });
      setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, content_box: bbox } : p)));
      addToast(`Detected content box: ${bbox.width}×${bbox.height}px`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Content box detection failed: ${err}`, "error");
    }
  };

  // Batch Auto Content Box (Individual Detection per Image)
  const handleBatchAutoContentBox = async () => {
    if (pages().length === 0) return;
    setIsAnalyzing(true);
    setAnalysisStatus({
      current: 0,
      total: pages().length,
      percent: 0,
      message: "Detecting content boxes individually per page...",
    });
    try {
      const updated = await invoke<PageItem[]>("batch_auto_detect_content", { pages: pages() });
      setPages(updated);
      addToast(`Detected unique content boxes for all ${updated.length} pages!`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Batch content detection failed: ${err}`, "error");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Master Smart Auto-Analyze ALL Pages (Full Individual Pipeline)
  const handleBatchAutoAnalyzeAll = async () => {
    if (pages().length === 0) return;
    setIsAnalyzing(true);
    setAnalysisStatus({
      current: 0,
      total: pages().length,
      percent: 0,
      message: "Running smart multithreaded analysis per page...",
    });
    try {
      const updated = await invoke<PageItem[]>("batch_auto_analyze_all", { pages: pages() });
      setPages(updated);
      addToast(`Analyzed all ${updated.length} pages (deskew & content boxes)!`, "success");
      await refreshPreview();
    } catch (err: any) {
      addToast(`Batch analysis failed: ${err}`, "error");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Update Split Position
  const updateSplitX = (val: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, split_x: val } : p)));
    debouncedRefreshPreview(80);
  };

  // Update Deskew Angle
  const updateDeskewAngle = (deg: number) => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) => prev.map((p) => (p.id === curr.id ? { ...p, deskew_angle: deg } : p)));
    debouncedRefreshPreview(80);
  };

  // Content Box Presets
  const applyContentPreset = (preset: "full" | "margin85" | "book75") => {
    const curr = activePage();
    const prev = previewResult();
    if (!curr || !prev) return;

    let w = prev.width;
    let h = prev.height;
    let box: ContentBoxDto;

    if (preset === "full") {
      box = { x: 0, y: 0, width: w, height: h };
    } else if (preset === "margin85") {
      box = {
        x: Math.round(w * 0.075),
        y: Math.round(h * 0.075),
        width: Math.round(w * 0.85),
        height: Math.round(h * 0.85),
      };
    } else {
      box = {
        x: Math.round(w * 0.125),
        y: Math.round(h * 0.125),
        width: Math.round(w * 0.75),
        height: Math.round(h * 0.75),
      };
    }

    setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, content_box: box } : p)));
    refreshPreview();
  };

  // Apply Margins to All
  const applyMarginsToAll = () => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) => prev.map((p) => ({ ...p, margins: { ...curr.margins } })));
    addToast(`Margins applied to all ${pages().length} pages!`, "success");
  };

  // Apply Output Settings to All
  const applyOutputSettingsToAll = () => {
    const curr = activePage();
    if (!curr) return;
    setPages((prev) =>
      prev.map((p) => ({
        ...p,
        output_mode: curr.output_mode,
        binarization_method: curr.binarization_method,
        sauvola_k: curr.sauvola_k,
        sauvola_window: curr.sauvola_window,
        despeckle: curr.despeckle,
        dewarp: curr.dewarp,
        ai_dewarp_active: curr.ai_dewarp_active,
        dpi: curr.dpi,
      }))
    );
    addToast(`Output settings applied to all ${pages().length} pages!`, "success");
  };

  // Neural Hardware Execution Provider Switcher
  const handleDeviceChange = async (device: string) => {
    setSelectedAiDevice(device);
    try {
      await invoke("set_ai_device", { device });
      const info = await invoke<SystemInfo>("get_system_info");
      setSystemInfo(info);
      addToast(`AI compute device switched to: ${info.ai_hardware}`, "success");
    } catch (err: any) {
      addToast(`Failed to switch device: ${err}`, "error");
    }
  };

  // 1-Click Magic AI Restore (Full 2-Stage Neural Dewarp + CleanNet)
  const handleMagicAiRestore = async () => {
    const page = activePage();
    if (!page) {
      addToast("Please select a scan page first", "info");
      return;
    }
    setIsLoading(true);
    addToast("Running Magic AI: 2-stage neural dewarp + pure white restoration...", "info");
    try {
      const res = await invoke<StagePreviewResult>("ai_restore_page", {
        page,
        fast: isFastMode(),
      });
      setPreviewResult(res);
      setPages((all) =>
        all.map((p) =>
          p.id === page.id
            ? {
                ...p,
                output_mode: "magic_ai",
                ai_dewarp_active: true,
                ai_corners: res.ai_corners || p.ai_corners,
              }
            : p
        )
      );
      setActiveStage("output");
      setShowCurtainComparison(true);
      addToast("Magic AI Restoration complete! Check the Before/After comparison.", "success");
    } catch (err: any) {
      addToast(`Magic AI Restore failed: ${err}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  // AI Neural Perspective Dewarp & Corner Detection (PagesGeomNet)
  const handleToggleAiDewarp = async () => {
    const page = activePage();
    if (!page) return;
    const next = !page.ai_dewarp_active;
    setIsLoading(true);
    try {
      let corners = page.ai_corners;
      if (next && !corners) {
        addToast("Detecting document boundary quad via PagesGeomNet...", "info");
        corners = await invoke<[number, number, number, number, number, number, number, number]>("ai_detect_corners", { page });
      }
      setPages((all) =>
        all.map((p) =>
          p.id === page.id
            ? { ...p, ai_dewarp_active: next, ai_corners: corners }
            : p
        )
      );
      refreshPreview();
      addToast(next ? "AI Neural Perspective Dewarp enabled" : "AI Neural Dewarp disabled", "info");
    } catch (err: any) {
      addToast(`AI Dewarp error: ${err}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRedetectCorners = async () => {
    const page = activePage();
    if (!page) return;
    setIsLoading(true);
    addToast("Re-analyzing scan geometry with PagesGeomNet...", "info");
    try {
      const corners = await invoke<[number, number, number, number, number, number, number, number]>("ai_detect_corners", { page });
      setPages((all) =>
        all.map((p) =>
          p.id === page.id
            ? { ...p, ai_dewarp_active: true, ai_corners: corners }
            : p
        )
      );
      refreshPreview();
      addToast("Document boundary corners re-detected!", "success");
    } catch (err: any) {
      addToast(`Detection failed: ${err}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetCorners = () => {
    const page = activePage();
    if (!page) return;
    const w = previewResult()?.width || page.width;
    const h = previewResult()?.height || page.height;
    const padX = Math.round(w * 0.03);
    const padY = Math.round(h * 0.03);
    const resetCorners: [number, number, number, number, number, number, number, number] = [
      padX, padY,
      w - padX, padY,
      w - padX, h - padY,
      padX, h - padY,
    ];
    setPages((all) =>
      all.map((p) =>
        p.id === page.id
          ? { ...p, ai_corners: resetCorners }
          : p
      )
    );
    debouncedRefreshPreview(60);
    addToast("Corners reset to image frame margin", "info");
  };

  // Run Batch Processing & Export
  const startBatchProcessing = async () => {
    if (pages().length === 0) {
      addToast("Please import scan pages before running batch", "error");
      return;
    }

    if (!isProUnlocked()) {
      const used = getPagesUsage();
      const needed = pages().length;
      if (used >= 15) {
        setLicenseModalReason({
          type: "quota",
          title: "Daily Free Limit Reached (15/15 Pages)",
          message: `You have processed ${used}/15 pages today. Free tier allows up to 15 page processings per day. LemonSqueezy payment integration coming soon!`,
        });
        setShowLicenseModal(true);
        return;
      }
      if (used + needed > 15) {
        const remaining = 15 - used;
        setLicenseModalReason({
          type: "quota",
          title: "Batch Exceeds Daily Free Quota",
          message: `You have ${remaining} free page processing(s) remaining today, but current project has ${needed} pages. Upgrade or use developer code to render all.`,
        });
        setShowLicenseModal(true);
        return;
      }
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
      setAnalysisStatus({
        current: 0,
        total: pages().length,
        percent: 0,
        message: "Rayon multithreaded export engine running...",
      });

      const results = await invoke<string[]>("run_batch_processing", {
        pages: pages(),
        outputDir: outputDir as string,
      });

      incrementPagesUsage(results.length);
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

    // Curtain Slider Drag (Real Before / After Reveals)
    if (isDraggingCurtain()) {
      const container = document.querySelector(".curtain-container") as HTMLElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const rel = (e.clientX - rect.left) / rect.width;
        setCurtainPos(Math.max(0.01, Math.min(0.99, rel)));
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

    // AI Quad Corner Anchor Drag (PagesGeomNet bounds)
    if (activeAiCornerIndex() !== null && previewResult()) {
      const container = document.querySelector(".image-canvas-wrapper") as HTMLElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const scaleX = previewResult()!.width / rect.width;
        const scaleY = previewResult()!.height / rect.height;
        const imgX = (e.clientX - rect.left) * scaleX;
        const imgY = (e.clientY - rect.top) * scaleY;
        const clampedX = Math.max(0, Math.min(previewResult()!.width, Math.round(imgX)));
        const clampedY = Math.max(0, Math.min(previewResult()!.height, Math.round(imgY)));

        const curr = activePage();
        if (curr && curr.ai_corners) {
          const nextCorners: [number, number, number, number, number, number, number, number] = [...curr.ai_corners];
          const idx = activeAiCornerIndex()!;
          nextCorners[idx * 2] = clampedX;
          nextCorners[idx * 2 + 1] = clampedY;
          setPages((all) => all.map((p) => (p.id === curr.id ? { ...p, ai_corners: nextCorners } : p)));
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
    if (activeAiCornerIndex() !== null) {
      setActiveAiCornerIndex(null);
      debouncedRefreshPreview(60);
    }
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
            <span class="brand-name">Pages</span>
            <span class="brand-tagline">Document Restoration Studio</span>
          </div>
        </div>

        <div class="header-center-telemetry">
          <Show when={systemInfo()?.ai_ready}>
            <div class="neo-pill-badge ai-telemetry-pill" title="Neural Hardware Execution Provider">
              <span class="status-dot cyan pulse"></span>
              <span class="ai-chip-label">AI:</span>
              <select
                class="ai-hardware-select"
                value={selectedAiDevice()}
                onChange={(e) => handleDeviceChange(e.currentTarget.value)}
              >
                <option value="auto">⚡ Auto ({systemInfo()!.ai_hardware})</option>
                <option value="directml">🎮 DirectML (Intel Arc / GPU)</option>
                <option value="openvino">🔷 OpenVINO (Xe / CPU)</option>
                <option value="cpu">💻 CPU Multithreaded</option>
              </select>
            </div>
          </Show>

          <Show when={systemInfo()}>
            <div class="neo-pill-badge lime">
              <span class="status-dot green"></span>
              <span>Rayon ({systemInfo()!.cpu_cores} Cores)</span>
            </div>
          </Show>

          <div class="neo-pill-badge blue">
            <span>Pages:</span>
            <span class="font-mono">{pages().length}</span>
          </div>

          <Show
            when={isProUnlocked()}
            fallback={
              <div
                class="neo-pill-badge orange cursor-pointer"
                onClick={() => {
                  setLicenseModalReason({
                    type: "manual",
                    title: "Pages Free Quota",
                    message: `Free Tier Active: ${dailyUsageCount()}/15 daily pages processed. Max 5 images per import.`,
                  });
                  setShowLicenseModal(true);
                }}
                title="Click to view license status / upgrade"
              >
                <span>Free: {dailyUsageCount()}/15 Today</span>
              </div>
            }
          >
            <div
              class="neo-pill-badge lime cursor-pointer"
              onClick={() => {
                setLicenseModalReason({
                  type: "manual",
                  title: "Pages Pro Active",
                  message: "Developer Pro License Active. Unlimited imports and multithreaded exports enabled.",
                });
                setShowLicenseModal(true);
              }}
              title="Pro Developer License Active"
            >
              <span class="status-dot green"></span>
              <span>Pro Unlocked</span>
            </div>
          </Show>
        </div>

        <div class="header-actions-row">
          <button class="neo-btn icon-only sm" onClick={toggleTheme} title="Toggle Light/Dark Theme">
            {theme() === "light" ? "🌙" : "☀️"}
          </button>

          {/* PRIMARY IMPORT BUTTON */}
          <button class="neo-btn lime" onClick={handleImportImages}>
            📁 Import Scans
          </button>

          {/* 1-CLICK MAGIC AI RESTORE BUTTON */}
          <button
            class="neo-btn magic-ai-glow"
            onClick={handleMagicAiRestore}
            disabled={!activePage() || isLoading()}
            title="1-Click Magic AI: Neural Perspective Dewarp + Pure-White Paper Restoration"
          >
            ⚡ Magic AI Restore
          </button>

          <button class="neo-btn purple" onClick={startBatchProcessing} disabled={pages().length === 0}>
            🚀 Render All ({pages().length})
          </button>
        </div>
      </header>

      {/* Animated Top Processing Banner */}
      <Show when={isAnalyzing() || isBatchProcessing()}>
        <div class="neo-processing-banner">
          <div class="processing-info">
            <span class="processing-spinner">⚡</span>
            <span>{analysisStatus().message || "Processing operation in parallel..."}</span>
          </div>

          <div class="processing-progress-wrap">
            <div class="neo-progress-track banner">
              <div
                class="neo-progress-fill striped"
                style={{ width: `${Math.min(100, Math.max(5, analysisStatus().percent))}%` }}
              ></div>
            </div>
            <span class="processing-percent-chip">{Math.round(analysisStatus().percent)}%</span>
          </div>
        </div>
      </Show>

      {/* Main 3-Column Studio Layout */}
      <main class={`neo-workspace-grid ${filmstripCollapsed() ? "filmstrip-collapsed" : ""}`}>
        {/* Left Column: Filmstrip Sidebar */}
        <aside class="filmstrip-panel">
          <div class="filmstrip-header-row">
            <div class="filmstrip-heading">
              <span>Scans List</span>
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
              placeholder="Search filename..."
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
                  <div class="font-heading" style={{ "font-weight": "800", "margin-bottom": "4px" }}>No Scans Loaded</div>
                  <p style={{ "font-size": "11px" }}>Use Import Scans above to load files</p>
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
          {/* Floating Canvas Loading Indicator */}
          <Show when={isLoading()}>
            <div class="canvas-loading-pill">
              <span class="processing-spinner">⚡</span>
              <span>Rendering Preview...</span>
            </div>
          </Show>

          <div
            class="canvas-viewport-root"
            onMouseDown={handleCanvasMouseDown}
            onWheel={handleCanvasWheel}
          >
            <Show
              when={activePage()}
              fallback={
                <div style={{ "text-align": "center", "background": "var(--bg-surface)", "padding": "40px", "border": "var(--border-width) solid var(--border-color)", "border-radius": "var(--radius-lg)", "box-shadow": "var(--shadow-hard-lg)", "max-width": "460px" }}>
                  <div style={{ "font-size": "48px", "margin-bottom": "12px" }}>📖</div>
                  <h2 class="font-heading" style={{ "font-size": "20px", "font-weight": "800", "margin-bottom": "8px" }}>
                    Pages Studio
                  </h2>
                  <p style={{ "font-size": "12px", "color": "var(--ink-muted)", "margin-bottom": "20px" }}>
                    Import documents, manuscripts, or book scans to auto-straighten, crop, and binarize.
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
                {/* Instant Low-Res / Thumbnail Placeholder (0ms feedback on page switch) */}
                <Show when={instantPreviewSrc() && (!previewResult() || isLoading())}>
                  <img
                    src={instantPreviewSrc()!}
                    alt="Fast preview"
                    class="canvas-image"
                    style={{ "filter": "brightness(0.95)" }}
                  />
                </Show>

                {/* Main Preview with Functional Before/After Curtain Split in Output Stage */}
                <Show when={previewResult()}>
                  <Show
                    when={activeStage() === "output" && showCurtainComparison() && activePage()}
                    fallback={
                      <img
                        src={convertFileSrc(previewResult()!.preview_path)}
                        alt="Stage preview"
                        class="canvas-image"
                      />
                    }
                  >
                    {/* Functional Before / After Curtain Comparison */}
                    <div class="curtain-container">
                      {/* Underlay: Original Raw Scan (Before) */}
                      <img
                        src={convertFileSrc(activePage()!.file_path)}
                        alt="Original scan (Before)"
                        class="curtain-before-img"
                      />
                      <div class="curtain-label before">BEFORE (RAW)</div>

                      {/* Overlay: Processed Sauvola Preview (After), Clipped by Curtain Position */}
                      <div
                        class="curtain-after-wrap"
                        style={{
                          "clip-path": `inset(0 0 0 ${curtainPos() * 100}%)`,
                        }}
                      >
                        <img
                          src={convertFileSrc(previewResult()!.preview_path)}
                          alt="Processed preview (After)"
                          class="curtain-after-img"
                        />
                        <div class="curtain-label after">AFTER (PROCESSED)</div>
                      </div>

                      {/* Draggable Divider Bar */}
                      <div
                        class="curtain-bar-wrap"
                        style={{ left: `${curtainPos() * 100}%` }}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          setIsDraggingCurtain(true);
                        }}
                      >
                        <div class="curtain-disc" title="Drag to compare before & after">⇄</div>
                      </div>
                    </div>
                  </Show>
                </Show>

                {/* Stage 2 Overlay: Split Cutter Line */}
                <Show when={activeStage() === "split" && activePage()?.split_mode === "two-pages"}>
                  <div class="split-cutter-container">
                    <span class="zone-indicator-badge left">[ Page 1: Left ]</span>
                    <span class="zone-indicator-badge right">[ Page 2: Right ]</span>
                    <div
                      class="split-guide-bar"
                      style={{
                        left: `${
                          activePage()?.split_x && (previewResult()?.width || activePage()!.width) > 0
                            ? (activePage()!.split_x! / (previewResult()?.width || activePage()!.width)) * 100
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

                {/* Stage 3 Overlay: AI Quad Neural Bounds (PagesGeomNet) */}
                <Show when={activeStage() === "deskew" && activePage()?.ai_dewarp_active && activePage()?.ai_corners && (previewResult()?.width || activePage()!.width) > 0}>
                  <div class="ai-corners-overlay">
                    <svg
                      viewBox={`0 0 ${previewResult()?.width || activePage()!.width} ${previewResult()?.height || activePage()!.height}`}
                      class="ai-corners-svg"
                    >
                      <polygon
                        points={`${activePage()!.ai_corners![0]},${activePage()!.ai_corners![1]} ${activePage()!.ai_corners![2]},${activePage()!.ai_corners![3]} ${activePage()!.ai_corners![4]},${activePage()!.ai_corners![5]} ${activePage()!.ai_corners![6]},${activePage()!.ai_corners![7]}`}
                        class="ai-quad-polygon"
                      />
                    </svg>

                    {/* 4 Interactive Corner Anchors */}
                    <div
                      class="ai-corner-handle tl"
                      style={{
                        left: `${(activePage()!.ai_corners![0] / (previewResult()?.width || activePage()!.width)) * 100}%`,
                        top: `${(activePage()!.ai_corners![1] / (previewResult()?.height || activePage()!.height)) * 100}%`,
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveAiCornerIndex(0);
                      }}
                      title="Top-Left Quad Corner (Drag to adjust)"
                    >
                      <span>TL</span>
                    </div>

                    <div
                      class="ai-corner-handle tr"
                      style={{
                        left: `${(activePage()!.ai_corners![2] / (previewResult()?.width || activePage()!.width)) * 100}%`,
                        top: `${(activePage()!.ai_corners![3] / (previewResult()?.height || activePage()!.height)) * 100}%`,
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveAiCornerIndex(1);
                      }}
                      title="Top-Right Quad Corner (Drag to adjust)"
                    >
                      <span>TR</span>
                    </div>

                    <div
                      class="ai-corner-handle br"
                      style={{
                        left: `${(activePage()!.ai_corners![4] / (previewResult()?.width || activePage()!.width)) * 100}%`,
                        top: `${(activePage()!.ai_corners![5] / (previewResult()?.height || activePage()!.height)) * 100}%`,
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveAiCornerIndex(2);
                      }}
                      title="Bottom-Right Quad Corner (Drag to adjust)"
                    >
                      <span>BR</span>
                    </div>

                    <div
                      class="ai-corner-handle bl"
                      style={{
                        left: `${(activePage()!.ai_corners![6] / (previewResult()?.width || activePage()!.width)) * 100}%`,
                        top: `${(activePage()!.ai_corners![7] / (previewResult()?.height || activePage()!.height)) * 100}%`,
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveAiCornerIndex(3);
                      }}
                      title="Bottom-Left Quad Corner (Drag to adjust)"
                    >
                      <span>BL</span>
                    </div>
                  </div>
                </Show>

                {/* Stage 4 Overlay: Draggable & Resizable Content Box */}
                <Show when={activeStage() === "content" && activePage()?.content_box && (previewResult()?.width || activePage()!.width) > 0}>
                  <div
                    class="content-box-mask"
                    style={{
                      left: `${(activePage()!.content_box!.x / (previewResult()?.width || activePage()!.width)) * 100}%`,
                      top: `${(activePage()!.content_box!.y / (previewResult()?.height || activePage()!.height)) * 100}%`,
                      width: `${(activePage()!.content_box!.width / (previewResult()?.width || activePage()!.width)) * 100}%`,
                      height: `${(activePage()!.content_box!.height / (previewResult()?.height || activePage()!.height)) * 100}%`,
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

        {/* Right Column: Unified Stage Inspector Panel (All Controls Scrollable & Permanently Expanded) */}
        <aside class="neo-inspector">
          {/* Master Actions Bar */}
          <div class="inspector-master-actions">
            <button
              class="neo-btn lime"
              onClick={handleBatchAutoAnalyzeAll}
              disabled={pages().length === 0 || isAnalyzing()}
              title="Runs deskew and content box detection individually for each page"
            >
              ✨ Smart Auto-Analyze All Pages
            </button>
            <button
              class="neo-btn purple sm"
              onClick={startBatchProcessing}
              disabled={pages().length === 0}
            >
              🚀 Export All Pages to Disk
            </button>
          </div>

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
            {/* Control Card 1: Orientation */}
            <div
              class={`neo-stage-card ${activeStage() === "orientation" ? "active" : ""}`}
              onClick={() => setActiveStage("orientation")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">1</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">↺ Orientation & Rotation</span>
                    <span class="stage-card-desc">Rotate & flip scan orientation</span>
                  </div>
                </div>
                <span class="font-mono" style={{ "font-weight": "800", "font-size": "12px" }}>
                  {activePage()?.orientation}°
                </span>
              </div>

              <div class="stage-card-body">
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

                <div class="stage-action-btn-row">
                  <button class="neo-btn sm" onClick={handleAutoOrientation}>
                    ⚡ Auto-Detect Orientation (Current)
                  </button>
                </div>
              </div>
            </div>

            {/* Control Card 2: Page Spread & Split */}
            <div
              class={`neo-stage-card ${activeStage() === "split" ? "active" : ""}`}
              onClick={() => setActiveStage("split")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">2</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">✂ Page Spread & Split</span>
                    <span class="stage-card-desc">Single page or book dual-spread cut</span>
                  </div>
                </div>
                <span class="sticker-badge blue">{activePage()?.split_mode}</span>
              </div>

              <div class="stage-card-body">
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
                        {activePage()?.split_x || Math.round((previewResult()?.width || activePage()!.width) / 2)} px
                      </span>
                    </div>
                    <input
                      type="range"
                      min={100}
                      max={previewResult()?.width || activePage()!.width || 3000}
                      value={activePage()?.split_x || Math.round((previewResult()?.width || activePage()!.width) / 2)}
                      class="neo-range-slider"
                      onInput={(e) => updateSplitX(parseInt(e.currentTarget.value))}
                    />
                  </div>
                </Show>

                <div class="stage-action-btn-row">
                  <button class="neo-btn sm orange" onClick={handleAutoSplit}>
                    ✂ Auto-Detect Split Line (Current)
                  </button>
                </div>
              </div>
            </div>

            {/* Control Card 3: Deskew & Alignment */}
            <div
              class={`neo-stage-card ${activeStage() === "deskew" ? "active" : ""}`}
              onClick={() => setActiveStage("deskew")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">3</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">📐 Deskew & Alignment</span>
                    <span class="stage-card-desc">Straighten text lines and scan tilt</span>
                  </div>
                </div>
                <span class="slider-val-tag">
                  {activePage()?.deskew_angle.toFixed(2)}°
                </span>
              </div>

              <div class="stage-card-body">
                {/* AI Neural Dewarp (PagesGeomNet) */}
                <div
                  class={`neo-toggle-box ${activePage()?.ai_dewarp_active ? "ai-active" : ""}`}
                  onClick={handleToggleAiDewarp}
                >
                  <div>
                    <div class="toggle-title-bold" style={{ display: "flex", "align-items": "center", gap: "6px" }}>
                      <span>⚡ AI Neural Dewarp & Bounds</span>
                      <span class="ai-tag-pill">PagesGeomNet</span>
                    </div>
                    <div class="toggle-sub-hint">Sub-pixel corner detection & true 2D projective homography</div>
                  </div>
                  <div class={`neo-switch ${activePage()?.ai_dewarp_active ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                <Show when={activePage()?.ai_dewarp_active}>
                  <div class="ai-dewarp-meta-card">
                    <div class="ai-corner-summary">
                      <span class="ai-corner-label">Detected Paper Quad:</span>
                      <span class="font-mono" style={{ "font-size": "11px", color: "var(--neo-cyan, #00f5ff)" }}>
                        {activePage()?.ai_corners ? "4 Corners Extracted & Locked" : "Pending Detection"}
                      </span>
                    </div>
                    <div class="batch-btn-row" style={{ "margin-top": "8px" }}>
                      <button class="neo-btn sm cyan" onClick={handleRedetectCorners} title="Re-run PagesGeomNet corner inference">
                        🎯 Re-Detect Corners
                      </button>
                      <button class="neo-btn sm" onClick={handleResetCorners} title="Reset corners to image frame bounds">
                        ↺ Frame Quad
                      </button>
                    </div>
                  </div>
                </Show>

                <div class="neo-card-header" style={{ "margin-top": "10px" }}>
                  <span class="neo-card-title">Fine Angle Rotation</span>
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
                    <div class="toggle-title-bold">Alignment Grid Overlay</div>
                    <div class="toggle-sub-hint">Visual horizontal & vertical alignment lines</div>
                  </div>
                  <div class={`neo-switch ${showDeskewGrid() ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                <div class="batch-btn-row">
                  <button class="neo-btn sm" onClick={handleAutoDeskew}>
                    📐 Auto Straighten (Current)
                  </button>
                  <button class="neo-btn sm purple" onClick={handleBatchAutoDeskew} title="Calculates skew angle for each page individually">
                    ✨ Straighten ALL (Per-Page)
                  </button>
                </div>
              </div>
            </div>

            {/* Control Card 4: Content Bounding Box */}
            <div
              class={`neo-stage-card ${activeStage() === "content" ? "active" : ""}`}
              onClick={() => setActiveStage("content")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">4</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">⛶ Content Bounding Box</span>
                    <span class="stage-card-desc">Interactive crop frame & margin detector</span>
                  </div>
                </div>
                <Show when={activePage()?.content_box}>
                  <span class="slider-val-tag">
                    {activePage()!.content_box!.width}×{activePage()!.content_box!.height}
                  </span>
                </Show>
              </div>

              <div class="stage-card-body">
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
                            const val = parseInt(e.currentTarget.value) || 50;
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
                            const val = parseInt(e.currentTarget.value) || 50;
                            setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, content_box: { ...p.content_box!, height: val } } : p));
                          }}
                        />
                        <span class="unit-label">px</span>
                      </div>
                    </div>
                  </div>
                </Show>

                <div class="batch-btn-row">
                  <button class="neo-btn sm" onClick={handleAutoContentBox}>
                    ⚡ Detect Box (Current)
                  </button>
                  <button class="neo-btn sm lime" onClick={handleBatchAutoContentBox} title="Detects unique bounding box for each page individually">
                    ✨ Detect ALL (Per-Page)
                  </button>
                </div>
              </div>
            </div>

            {/* Control Card 5: Page Margins */}
            <div
              class={`neo-stage-card ${activeStage() === "margins" ? "active" : ""}`}
              onClick={() => setActiveStage("margins")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">5</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">📏 Page Margins</span>
                    <span class="stage-card-desc">Uniform layout padding & borders</span>
                  </div>
                </div>
              </div>

              <div class="stage-card-body">
                <div class="neo-grid-2x2">
                  <div class="input-field-block">
                    <span class="input-label-text">Top Margin</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.top || 10}
                        onInput={(e) => {
                          const val = parseFloat(e.currentTarget.value) || 0;
                          setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, top: val } } : p));
                        }}
                      />
                      <span class="unit-label">%</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Bottom</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.bottom || 10}
                        onInput={(e) => {
                          const val = parseFloat(e.currentTarget.value) || 0;
                          setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, bottom: val } } : p));
                        }}
                      />
                      <span class="unit-label">%</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Left Margin</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.left || 10}
                        onInput={(e) => {
                          const val = parseFloat(e.currentTarget.value) || 0;
                          setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, left: val } } : p));
                        }}
                      />
                      <span class="unit-label">%</span>
                    </div>
                  </div>
                  <div class="input-field-block">
                    <span class="input-label-text">Right</span>
                    <div class="neo-unit-input">
                      <input
                        type="number"
                        value={activePage()?.margins.right || 10}
                        onInput={(e) => {
                          const val = parseFloat(e.currentTarget.value) || 0;
                          setPages((all) => all.map((p) => p.id === activePage()!.id ? { ...p, margins: { ...p.margins, right: val } } : p));
                        }}
                      />
                      <span class="unit-label">%</span>
                    </div>
                  </div>
                </div>

                <div class="stage-action-btn-row">
                  <button class="neo-btn sm" onClick={applyMarginsToAll}>
                    📏 Apply Margins to All Pages
                  </button>
                </div>
              </div>
            </div>

            {/* Control Card 6: Output & Binarization */}
            <div
              class={`neo-stage-card ${activeStage() === "output" ? "active" : ""}`}
              onClick={() => setActiveStage("output")}
            >
              <div class="stage-card-header">
                <div class="stage-card-header-left">
                  <div class="stage-card-badge">6</div>
                  <div class="stage-card-title-wrap">
                    <span class="stage-card-title">⚡ Output & Filters</span>
                    <span class="stage-card-desc">Sauvola binarization, despeckle & dewarp</span>
                  </div>
                </div>
                <span class="sticker-badge lime">{activePage()?.output_mode}</span>
              </div>

              <div class="stage-card-body">
                <div class="neo-card-header">
                  <span class="neo-card-title">Color Mode</span>
                </div>
                <div class="neo-segmented-group">
                  <button
                    class={`neo-segment-btn ${activePage()?.output_mode === "bw" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, output_mode: "bw" } : p)));
                      refreshPreview();
                    }}
                  >
                    B/W Clean
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.output_mode === "grayscale" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, output_mode: "grayscale" } : p)));
                      refreshPreview();
                    }}
                  >
                    Grayscale
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.output_mode === "color" ? "active" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, output_mode: "color" } : p)));
                      refreshPreview();
                    }}
                  >
                    Color
                  </button>
                  <button
                    class={`neo-segment-btn ${activePage()?.output_mode === "magic_ai" ? "active magic-ai-tab" : ""}`}
                    onClick={() => {
                      setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, output_mode: "magic_ai" } : p)));
                      refreshPreview();
                    }}
                  >
                    ⚡ Magic AI
                  </button>
                </div>

                <Show when={activePage()?.output_mode === "magic_ai"}>
                  <div class="ai-feature-card">
                    <div class="ai-feature-title">
                      <div style={{ display: "flex", "align-items": "center", gap: "6px" }}>
                        <span class="ai-badge-title">Pages Shine (CleanNet)</span>
                        <span class="sticker-badge cyan">Neural Stage 2</span>
                      </div>
                    </div>
                    <p class="ai-feature-desc">
                      100% pure white paper background restoration, native text darkening, shadow scrub, and ink sharpening.
                    </p>

                    <div
                      class={`neo-toggle-box ${isFastMode() ? "ai-active" : ""}`}
                      onClick={() => {
                        setIsFastMode(!isFastMode());
                        refreshPreview();
                      }}
                      style={{ "margin-top": "8px" }}
                    >
                      <div>
                        <div class="toggle-title-bold">⚡ Low-Power Fast Mode</div>
                        <div class="toggle-sub-hint">384px fast-tiled inference (optimized for laptops & low-power PCs)</div>
                      </div>
                      <div class={`neo-switch ${isFastMode() ? "active" : ""}`}>
                        <div class="neo-switch-knob"></div>
                      </div>
                    </div>
                  </div>
                </Show>

                <Show when={activePage()?.output_mode === "bw"}>
                  <div class="neo-card-header">
                    <span class="neo-card-title">Binarization Method</span>
                  </div>
                  <div class="neo-segmented-group">
                    <button
                      class={`neo-segment-btn ${activePage()?.binarization_method === "sauvola" ? "active" : ""}`}
                      onClick={() => {
                        setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, binarization_method: "sauvola" } : p)));
                        refreshPreview();
                      }}
                    >
                      Sauvola (Adaptive)
                    </button>
                    <button
                      class={`neo-segment-btn ${activePage()?.binarization_method === "otsu" ? "active" : ""}`}
                      onClick={() => {
                        setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, binarization_method: "otsu" } : p)));
                        refreshPreview();
                      }}
                    >
                      Otsu (Global)
                    </button>
                  </div>

                  <Show when={activePage()?.binarization_method === "sauvola"}>
                    <div class="neo-slider-wrap">
                      <div class="slider-info-row">
                        <span class="slider-caption">Sauvola Sensitivity (K)</span>
                        <span class="slider-val-tag">{activePage()?.sauvola_k.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.05"
                        max="0.50"
                        step="0.01"
                        value={activePage()?.sauvola_k || 0.25}
                        class="neo-range-slider"
                        onInput={(e) => {
                          const val = parseFloat(e.currentTarget.value);
                          setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, sauvola_k: val } : p)));
                          debouncedRefreshPreview(120);
                        }}
                      />
                    </div>

                    <div class="neo-slider-wrap">
                      <div class="slider-info-row">
                        <span class="slider-caption">Window Radius</span>
                        <span class="slider-val-tag">{activePage()?.sauvola_window} px</span>
                      </div>
                      <input
                        type="range"
                        min="5"
                        max="50"
                        step="2"
                        value={activePage()?.sauvola_window || 15}
                        class="neo-range-slider"
                        onInput={(e) => {
                          const val = parseInt(e.currentTarget.value);
                          setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, sauvola_window: val } : p)));
                          debouncedRefreshPreview(120);
                        }}
                      />
                    </div>
                  </Show>
                </Show>

                <div
                  class="neo-toggle-box"
                  onClick={() => {
                    const next = !activePage()?.despeckle;
                    setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, despeckle: next } : p)));
                    refreshPreview();
                  }}
                >
                  <div>
                    <div class="toggle-title-bold">Despeckle Noise Filter</div>
                    <div class="toggle-sub-hint">Eliminate scan artifacts and dust dots</div>
                  </div>
                  <div class={`neo-switch ${activePage()?.despeckle ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                <div
                  class="neo-toggle-box"
                  onClick={() => {
                    const next = !activePage()?.dewarp;
                    setPages((all) => all.map((p) => (p.id === activePage()!.id ? { ...p, dewarp: next } : p)));
                    refreshPreview();
                  }}
                >
                  <div>
                    <div class="toggle-title-bold">Dewarp Curvature</div>
                    <div class="toggle-sub-hint">Flatten book binding curvature</div>
                  </div>
                  <div class={`neo-switch ${activePage()?.dewarp ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                {/* Before/After Split Comparison Toggle */}
                <div
                  class="neo-toggle-box"
                  onClick={() => setShowCurtainComparison(!showCurtainComparison())}
                >
                  <div>
                    <div class="toggle-title-bold">Before / After Slider</div>
                    <div class="toggle-sub-hint">Interactive split curtain comparison</div>
                  </div>
                  <div class={`neo-switch ${showCurtainComparison() ? "active" : ""}`}>
                    <div class="neo-switch-knob"></div>
                  </div>
                </div>

                <div class="batch-btn-row">
                  <button class="neo-btn sm" onClick={refreshPreview}>
                    ⚡ Render Output (Current)
                  </button>
                  <button class="neo-btn sm" onClick={applyOutputSettingsToAll}>
                    ✨ Apply Settings to All
                  </button>
                </div>
              </div>
            </div>
          </Show>
        </aside>
      </main>

      {/* Batch Processing Modal */}
      <Show when={showBatchModal()}>
        <div class="neo-modal-backdrop">
          <div class="neo-modal-box">
            <div class="neo-card-header">
              <span class="font-heading" style={{ "font-size": "16px", "font-weight": "800" }}>
                🚀 Multithreaded Processing Engine
              </span>
              <span class="sticker-badge purple">Rayon Active</span>
            </div>

            <p style={{ "font-size": "12px", "color": "var(--ink-muted)" }}>
              Processing pages across all available CPU cores. Sauvola binarization, deskewing, and despeckle filters are being rendered.
            </p>

            <div class="neo-progress-track">
              <div
                class="neo-progress-bar"
                style={{ width: `${Math.min(100, Math.max(5, analysisStatus().percent))}%` }}
              ></div>
            </div>

            <div style={{ "display": "flex", "justify-content": "space-between", "font-family": "var(--font-mono)", "font-size": "11px" }}>
              <span>{analysisStatus().message || "Processing scans..."}</span>
              <span>{analysisStatus().current} / {analysisStatus().total}</span>
            </div>

            <Show when={!isBatchProcessing()}>
              <button class="neo-btn lime" onClick={() => setShowBatchModal(false)}>
                ✓ Done (Close Window)
              </button>
            </Show>
          </div>
        </div>
      </Show>

      {/* License & LemonSqueezy Paywall Modal */}
      <Show when={showLicenseModal()}>
        <div class="neo-modal-backdrop" onClick={() => setShowLicenseModal(false)}>
          <div class="neo-modal-box neo-paywall-modal" onClick={(e) => e.stopPropagation()}>
            <div class="neo-card-header">
              <span class="font-heading" style={{ "font-size": "16px", "font-weight": "800" }}>
                {licenseModalReason().title}
              </span>
              <span class={`sticker-badge ${isProUnlocked() ? "lime" : "coral"}`}>
                {isProUnlocked() ? "PRO UNLOCKED" : "FREE TIER"}
              </span>
            </div>

            <p style={{ "font-size": "13px", "color": "var(--ink)", "line-height": "1.5" }}>
              {licenseModalReason().message}
            </p>

            <div class="neo-lemon-card">
              <span class="neo-lemon-icon">🍋</span>
              <div class="neo-lemon-text">
                <span class="neo-lemon-title">LemonSqueezy Checkout Integration Coming</span>
                <span class="neo-lemon-subtitle">
                  Paid subscription / lifetime licenses via LemonSqueezy are rolling out soon. Free tier: 5 images per import & 15 daily page processings.
                </span>
              </div>
            </div>

            <div class="neo-quota-pills-row">
              <div class="neo-pill-badge blue">
                <span>Daily Quota:</span>
                <span class="font-mono">{dailyUsageCount()} / 15 Pages</span>
              </div>
              <div class="neo-pill-badge purple">
                <span>Import Limit:</span>
                <span class="font-mono">5 Images / batch</span>
              </div>
            </div>

            {/* If import modal, offer partial import */}
            <Show when={licenseModalReason().type === "import" && (licenseModalReason().pendingPaths?.length || 0) > 0}>
              <div style={{ "display": "flex", "gap": "10px", "margin-top": "4px" }}>
                <button class="neo-btn blue" onClick={importFirstFivePending}>
                  📥 Import First 5 Images Only
                </button>
              </div>
            </Show>

            {/* Secret Developer Code Section */}
            <div class="neo-dev-unlock-section">
              <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center" }}>
                <span style={{ "font-family": "var(--font-heading)", "font-size": "12px", "font-weight": "800" }}>
                  🔑 Developer / Tester Override Code:
                </span>
                <Show when={isProUnlocked()}>
                  <span style={{ "font-size": "11px", "color": "var(--neo-lime)", "font-weight": "700" }}>
                    ✓ Pro Unlocked
                  </span>
                </Show>
              </div>

              <div class="neo-dev-input-row">
                <input
                  type="text"
                  class="neo-input"
                  placeholder="Enter secret developer code..."
                  value={secretDevKey()}
                  onInput={(e) => setSecretDevKey(e.currentTarget.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      if (unlockDeveloperPro(secretDevKey())) {
                        if (licenseModalReason().type === "import" && licenseModalReason().pendingPaths) {
                          const p = licenseModalReason().pendingPaths!;
                          setShowLicenseModal(false);
                          invoke<PageItem[]>("import_images", { paths: p }).then((imported) => {
                            setPages((prev) => [...prev, ...imported]);
                            if (!selectedPageId() && imported.length > 0) selectPage(imported[0].id);
                            addToast(`Loaded ${imported.length} scan pages successfully!`, "success");
                          });
                        }
                      }
                    }
                  }}
                />
                <button
                  class="neo-btn lime sm"
                  onClick={() => {
                    if (unlockDeveloperPro(secretDevKey())) {
                      if (licenseModalReason().type === "import" && licenseModalReason().pendingPaths) {
                        const p = licenseModalReason().pendingPaths!;
                        setShowLicenseModal(false);
                        invoke<PageItem[]>("import_images", { paths: p }).then((imported) => {
                          setPages((prev) => [...prev, ...imported]);
                          if (!selectedPageId() && imported.length > 0) selectPage(imported[0].id);
                          addToast(`Loaded ${imported.length} scan pages successfully!`, "success");
                        });
                      }
                    }
                  }}
                >
                  Unlock
                </button>
              </div>

              <Show when={devKeyError()}>
                <span style={{ "color": "var(--neo-coral)", "font-size": "11px", "font-weight": "700" }}>
                  {devKeyError()}
                </span>
              </Show>

              <Show when={isProUnlocked()}>
                <button
                  class="neo-btn coral sm"
                  style={{ "align-self": "flex-start", "margin-top": "4px" }}
                  onClick={() => {
                    localStorage.removeItem("pages_pro_unlocked");
                    setIsProUnlocked(false);
                    addToast("Pro mode disabled (Testing Free Tier limits)", "info");
                  }}
                >
                  🔒 Lock Again (Test Free Tier)
                </button>
              </Show>
            </div>

            <div style={{ "display": "flex", "justify-content": "flex-end", "gap": "10px", "margin-top": "6px" }}>
              <button class="neo-btn sm" onClick={() => setShowLicenseModal(false)}>
                Close
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
              <span>{t.type === "success" ? "✓" : t.type === "error" ? "⚠" : "ℹ"}</span>
              <span>{t.text}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

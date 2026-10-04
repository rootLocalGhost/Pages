export type Stage =
  | "orientation"
  | "split"
  | "deskew"
  | "content"
  | "margins"
  | "output";

export interface ContentBoxDto {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MarginsDto {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface SplitResultDto {
  mode: string;
  split_x?: number | null;
  left_rect?: ContentBoxDto | null;
  right_rect?: ContentBoxDto | null;
}

export interface PageItem {
  id: string;
  file_path: string;
  file_name: string;
  width: number;
  height: number;
  orientation: number; // 0, 90, 180, 270
  split_mode: "auto" | "single" | "two-pages";
  split_x?: number | null;
  deskew_angle: number;
  auto_deskew_angle?: number | null;
  content_box?: ContentBoxDto | null;
  margins: MarginsDto;
  output_mode: "bw" | "grayscale" | "color";
  binarization_method: "sauvola" | "wolf" | "otsu";
  sauvola_k: number;
  sauvola_window: number;
  despeckle: boolean;
  dewarp: boolean;
  dpi: number;
  thumbnail_path?: string | null;
}

export interface StagePreviewResult {
  preview_path: string;
  original_path: string;
  width: number;
  height: number;
  auto_deskew_angle?: number | null;
  auto_content_box?: ContentBoxDto | null;
  split_info?: SplitResultDto | null;
}

export interface BatchProgressEvent {
  current: number;
  total: number;
  percent: number;
  message: string;
}

export interface SystemInfo {
  cpu_cores: number;
  app_version: string;
  temp_dir: string;
}

export interface ToastMessage {
  id: number;
  type: "info" | "success" | "error" | "warning";
  text: string;
}

import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [solidPlugin()],
  clearScreen: false,
  server: {
    port: 1430,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1431,
        }
      : undefined,
    watch: {
      ignored: [
        "**/src-tauri/**",
        "**/target/**",
        "**/crates/**",
        "**/output/**",
        "**/scans/**",
        "**/ScanSmith-AI/**",
      ],
    },
  },
  build: {
    target: "esnext",
    minify: "esbuild",
  },
});

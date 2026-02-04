import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    base: "/",
    plugins: [react()],

    clearScreen: false,

    server: {
      port: parseInt(env.VITE_DEV_PORT || "5173", 10),
      strictPort: true,
      fs: { allow: [".."] },

      proxy: {
        "/api": {
          target: env.VITE_COMFY_URL || "http://localhost:8188",
          changeOrigin: true,
          secure: false,
          ws: true,

          configure: (proxy) => {
            proxy.on("proxyReq", (proxyReq) => {
              // ✅ Make ComfyUI happy: Origin must match Host
              proxyReq.setHeader("origin", "http://localhost:8188");
            });
          },
        },
      },
    },


    envPrefix: ["VITE_", "TAURI_"],

    resolve: {
      alias: { "@": resolve(__dirname, "src") },
    },

    build: {
      target: process.env.TAURI_PLATFORM == "windows" ? "chrome105" : "safari13",
      minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
      sourcemap: !!process.env.TAURI_DEBUG,
    },

    assetsInclude: ["**/*.pdf"],
  };
});

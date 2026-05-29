import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
// Production is served under a GitHub Pages project path; dev proxies /api to FastAPI.
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/sp500-quant-dashboard/" : "/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8001",
    },
  },
}));

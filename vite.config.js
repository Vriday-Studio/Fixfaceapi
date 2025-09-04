import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react({ jsxRuntime: "automatic" })],
  resolve: { dedupe: ["@tensorflow/tfjs","@tensorflow/tfjs-core","@tensorflow/tfjs-backend-wasm","face-api.js"] },
  optimizeDeps: { include: ["@tensorflow/tfjs","@tensorflow/tfjs-backend-wasm","face-api.js"] }
});
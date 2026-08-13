import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves a project site from /<repo>/, so assets must be requested
// with that prefix. `BASE_PATH` is set by the deploy workflow; local dev and a
// user/organisation site both want plain "/".
const base = process.env.BASE_PATH ?? "/";

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: "dist",
    // The wasm bundle is ~2.3 MB; the default 500 kB warning is just noise here.
    chunkSizeWarningLimit: 4096,
  },
  // wasm-pack writes an ES module that fetches its own .wasm next to itself.
  // Vite needs to treat that .wasm as an asset rather than try to bundle it.
  assetsInclude: ["**/*.wasm"],
  server: { fs: { allow: [".."] } },
});

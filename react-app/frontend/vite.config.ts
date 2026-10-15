import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // During `npm run dev`, forward API calls to the FastAPI backend so the
    // frontend can just call fetch('/api/...') with no base URL, in both
    // dev and the production build (where FastAPI serves these same static
    // files and '/api/...' is already same-origin).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Build straight into the FastAPI project's static/ folder.
    outDir: '../backend/static',
    emptyOutDir: true,
  },
})

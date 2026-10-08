import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Where the dev server forwards API calls. `npm run dev` at the repo root sets
// LEAFCARE_API_URL when the API runs on a non-default port.
const api = process.env.LEAFCARE_API_URL || 'http://127.0.0.1:4000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': api,
      '/uploads': api,
    },
  },
});

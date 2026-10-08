import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApp } from './app.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT) || 4000;
// Localhost only by default (Docker sets 0.0.0.0). Set LEAFCARE_HOST=0.0.0.0 to
// open the app from other devices on your network, e.g. a phone camera.
// (Not plain HOST: some shells, e.g. tcsh, export HOST=<hostname>.)
const host = (process.env.LEAFCARE_HOST || '').trim() || '127.0.0.1';

const app = createApp({
  // 127.0.0.1 rather than localhost: newer Node versions may resolve localhost to
  // IPv6 (::1) first, while the ML service listens on IPv4.
  mlUrl: process.env.ML_SERVICE_URL || 'http://127.0.0.1:5001',
  dataDir: process.env.DATA_DIR || path.join(root, 'data'),
  clientDist: process.env.CLIENT_DIST || path.join(root, '..', 'client', 'dist'),
});

// LEAFCARE_PUBLIC_URL lets Docker print the address users should open.
const shown = process.env.LEAFCARE_PUBLIC_URL || `http://${host === '127.0.0.1' ? 'localhost' : host}:${port}`;
app
  .listen(port, host, () => console.log(`API server listening on ${shown}`))
  .on('error', (err) => {
    if (err.code !== 'EADDRINUSE') throw err;
    console.error(`Port ${port} is already in use — is LeafCare already running? Stop that program or set PORT to another port.`);
    process.exit(1);
  });

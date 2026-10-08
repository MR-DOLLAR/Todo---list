import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApp } from './app.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT) || 4000;

const app = createApp({
  // 127.0.0.1 rather than localhost: newer Node versions may resolve localhost to
  // IPv6 (::1) first, while the ML service listens on IPv4.
  mlUrl: process.env.ML_SERVICE_URL || 'http://127.0.0.1:5001',
  dataDir: process.env.DATA_DIR || path.join(root, 'data'),
  clientDist: process.env.CLIENT_DIST || path.join(root, '..', 'client', 'dist'),
});

app.listen(port, () => console.log(`API server listening on http://localhost:${port}`));

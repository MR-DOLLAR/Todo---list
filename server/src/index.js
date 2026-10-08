import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApp } from './app.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT) || 4000;

const app = createApp({
  mlUrl: process.env.ML_SERVICE_URL || 'http://localhost:5000',
  dataDir: process.env.DATA_DIR || path.join(root, 'data'),
  clientDist: process.env.CLIENT_DIST || path.join(root, '..', 'client', 'dist'),
});

app.listen(port, () => console.log(`API server listening on http://localhost:${port}`));

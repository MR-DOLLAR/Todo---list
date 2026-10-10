import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';
import { HistoryStore } from './store.js';

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);
const NOT_BUILT_PAGE = `<!doctype html><meta charset="utf-8"><title>LeafCare API</title>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;padding:0 16px;line-height:1.5">
<h1>LeafCare API is running</h1>
<p>The web interface has not been built, so there is nothing to show on this port yet. Either:</p>
<ul>
<li><b>Development:</b> open <a href="http://localhost:5173">http://localhost:5173</a> (started by <code>npm run dev</code> in <code>client/</code>, or <code>npm run dev</code> from the repo root), or</li>
<li><b>Single server:</b> run <code>npm run build</code> in <code>client/</code>, then reload this page.</li>
</ul>
<p>API health: <a href="/api/health">/api/health</a></p></body>`;
const EXT = { 'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/bmp': '.bmp' };

export function createApp({ mlUrl, dataDir, clientDist, logger = true }) {
  const uploadsDir = path.join(dataDir, 'uploads');
  fs.mkdirSync(uploadsDir, { recursive: true });
  const store = new HistoryStore(path.join(dataDir, 'history.json'));

  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: (_req, file, cb) =>
      ALLOWED_TYPES.has(file.mimetype)
        ? cb(null, true)
        : cb(Object.assign(new Error('Only JPEG, PNG, WebP or BMP images are allowed'), { status: 400 })),
  });

  async function ml(pathname, init) {
    let res;
    try {
      res = await fetch(`${mlUrl}${pathname}`, init);
    } catch {
      throw Object.assign(
        new Error(`ML service is unavailable at ${mlUrl}. Start it with "python app.py" in ml-service/.`),
        { status: 503 },
      );
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(body.error || 'ML service error'), { status: res.status });
    return body;
  }

  const app = express();
  app.use(cors());
  app.use(express.json());
  if (logger) app.use(morgan('dev'));
  app.use('/uploads', express.static(uploadsDir, { maxAge: '7d' }));

  const api = express.Router();

  api.get('/health', async (_req, res) => {
    try {
      res.json({ status: 'ok', ml: await ml('/health') });
    } catch {
      res.status(503).json({ status: 'degraded', ml: null });
    }
  });

  api.post('/diagnose', upload.single('image'), async (req, res, next) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

      const form = new FormData();
      form.append('image', new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);
      if (req.body.crop) form.append('crop', req.body.crop);
      if (req.body.preference) form.append('preference', req.body.preference);
      const result = await ml('/predict', { method: 'POST', body: form });

      const id = crypto.randomUUID();
      const filename = `${id}${EXT[req.file.mimetype]}`;
      await fsp.writeFile(path.join(uploadsDir, filename), req.file.buffer);

      const record = await store.add({
        id,
        createdAt: new Date().toISOString(),
        imageUrl: `/uploads/${filename}`,
        crop: req.body.crop || null,
        preference: req.body.preference || 'integrated',
        notes: (req.body.notes || '').slice(0, 500),
        result,
      });
      res.status(201).json(record);
    } catch (err) {
      next(err);
    }
  });

  api.get('/history', async (req, res) => {
    const limit = Math.min(Number(req.query.limit) || 50, 200);
    const items = (await store.list()).slice(0, limit).map(({ result, ...r }) => ({
      ...r,
      prediction: result.prediction,
      status: result.status || 'confident',
      severity: result.severity,
      urgency: result.treatment_plan.urgency,
    }));
    res.json(items);
  });

  api.get('/history/:id', async (req, res) => {
    const record = await store.get(req.params.id);
    if (!record) return res.status(404).json({ error: 'Not found' });
    res.json(record);
  });

  api.delete('/history/:id', async (req, res) => {
    const removed = await store.remove(req.params.id);
    if (!removed) return res.status(404).json({ error: 'Not found' });
    await fsp.rm(path.join(uploadsDir, path.basename(removed.imageUrl)), { force: true });
    res.status(204).end();
  });

  api.get('/stats', async (_req, res) => {
    const records = await store.list();
    const byDisease = {};
    const bySeverity = { none: 0, mild: 0, moderate: 0, severe: 0 };
    for (const { result } of records) {
      const key = `${result.prediction.crop} – ${result.prediction.disease}`;
      byDisease[key] = (byDisease[key] || 0) + 1;
      bySeverity[result.severity.level] = (bySeverity[result.severity.level] || 0) + 1;
    }
    res.json({
      total: records.length,
      healthy: records.filter((r) => r.result.prediction.healthy).length,
      diseased: records.filter((r) => !r.result.prediction.healthy).length,
      bySeverity,
      topDiseases: Object.entries(byDisease)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, count]) => ({ name, count })),
    });
  });

  api.get('/diseases', async (req, res, next) => {
    try {
      const qs = new URLSearchParams();
      if (req.query.crop) qs.set('crop', req.query.crop);
      if (req.query.q) qs.set('q', req.query.q);
      res.json(await ml(`/diseases?${qs}`));
    } catch (err) {
      next(err);
    }
  });

  api.get('/diseases/:id', async (req, res, next) => {
    try {
      res.json(await ml(`/diseases/${encodeURIComponent(req.params.id)}`));
    } catch (err) {
      next(err);
    }
  });

  api.get('/crops', async (_req, res, next) => {
    try {
      res.json(await ml('/crops'));
    } catch (err) {
      next(err);
    }
  });

  app.use('/api', api);
  app.use('/api', (_req, res) => res.status(404).json({ error: 'Not found' }));

  // Serve the built React app. Checked per request so a build made after the
  // server started is picked up without a restart.
  if (clientDist) {
    app.use(express.static(clientDist));
    app.get(/^(?!\/api|\/uploads).*/, (_req, res) => {
      const index = path.join(clientDist, 'index.html');
      if (fs.existsSync(index)) return res.sendFile(index);
      res.status(404).type('html').send(NOT_BUILT_PAGE);
    });
  }

  app.use((err, _req, res, _next) => {
    const status = err.status || (err instanceof multer.MulterError ? 400 : 500);
    if (status >= 500 && status !== 503) console.error(err);
    res.status(status).json({ error: err.message || 'Internal server error' });
  });

  return app;
}

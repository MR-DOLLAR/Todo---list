import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createApp } from '../src/app.js';

const fakeResult = {
  mode: 'heuristic',
  prediction: { label: 'Tomato___Early_blight', confidence: 0.91, crop: 'Tomato', disease: 'Early Blight', healthy: false },
  severity: { level: 'moderate', affected_area_pct: 20 },
  treatment_plan: { urgency: 'medium', steps: [] },
};

let mlServer, apiServer, base, dataDir;

before(async () => {
  mlServer = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/predict') {
      req.resume();
      req.on('end', () => res.end(JSON.stringify(fakeResult)));
    } else if (req.url === '/health') res.end(JSON.stringify({ status: 'ok', mode: 'heuristic' }));
    else if (req.url.startsWith('/diseases')) res.end(JSON.stringify([{ id: 'x' }]));
    else res.writeHead(404).end('{}');
  });
  await new Promise((r) => mlServer.listen(0, r));
  dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'leafcare-'));
  const app = createApp({ mlUrl: `http://localhost:${mlServer.address().port}`, dataDir, logger: false });
  apiServer = app.listen(0);
  await new Promise((r) => apiServer.on('listening', r));
  base = `http://localhost:${apiServer.address().port}/api`;
});

after(() => {
  apiServer.close();
  mlServer.close();
  fs.rmSync(dataDir, { recursive: true, force: true });
});

function imageForm(type = 'image/png') {
  const form = new FormData();
  form.append('image', new Blob([Buffer.from('fake')], { type }), 'leaf.png');
  form.append('crop', 'Tomato');
  return form;
}

test('health reports ML status', async () => {
  const body = await (await fetch(`${base}/health`)).json();
  assert.equal(body.ml.mode, 'heuristic');
});

test('diagnose → history → stats → delete', async () => {
  const res = await fetch(`${base}/diagnose`, { method: 'POST', body: imageForm() });
  assert.equal(res.status, 201);
  const record = await res.json();
  assert.equal(record.result.prediction.disease, 'Early Blight');
  assert.ok(fs.existsSync(path.join(dataDir, record.imageUrl)));

  const history = await (await fetch(`${base}/history`)).json();
  assert.equal(history[0].id, record.id);
  assert.equal(history[0].urgency, 'medium');

  const stats = await (await fetch(`${base}/stats`)).json();
  assert.equal(stats.diseased, 1);
  assert.equal(stats.bySeverity.moderate, 1);

  assert.equal((await fetch(`${base}/history/${record.id}`, { method: 'DELETE' })).status, 204);
  assert.equal((await fetch(`${base}/history/${record.id}`)).status, 404);
  assert.ok(!fs.existsSync(path.join(dataDir, record.imageUrl)));
});

test('rejects non-image uploads and missing files', async () => {
  assert.equal((await fetch(`${base}/diagnose`, { method: 'POST', body: imageForm('text/plain') })).status, 400);
  assert.equal((await fetch(`${base}/diagnose`, { method: 'POST', body: new FormData() })).status, 400);
});

test('proxies disease library', async () => {
  assert.deepEqual(await (await fetch(`${base}/diseases?crop=Tomato`)).json(), [{ id: 'x' }]);
});

test('unknown API routes return JSON 404', async () => {
  const res = await fetch(`${base}/nope`);
  assert.equal(res.status, 404);
  assert.deepEqual(await res.json(), { error: 'Not found' });
});

test('explains an unbuilt client, then serves the build without a restart', async () => {
  const dist = fs.mkdtempSync(path.join(os.tmpdir(), 'leafcare-dist-'));
  const server = createApp({ mlUrl: 'http://127.0.0.1:9', dataDir, clientDist: dist, logger: false }).listen(0);
  await new Promise((r) => server.on('listening', r));
  const url = `http://localhost:${server.address().port}`;
  try {
    const before = await fetch(`${url}/history`);
    assert.equal(before.status, 404);
    assert.match(await before.text(), /web interface has not been built/);

    fs.writeFileSync(path.join(dist, 'index.html'), '<div id="root"></div>');
    const after = await fetch(`${url}/history`);
    assert.equal(after.status, 200);
    assert.match(await after.text(), /id="root"/);

    const down = await fetch(`${url}/api/crops`);
    assert.equal(down.status, 503);
    assert.match((await down.json()).error, /python app\.py/);
  } finally {
    server.close();
    fs.rmSync(dist, { recursive: true, force: true });
  }
});

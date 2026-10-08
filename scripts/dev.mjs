// Start the whole app in one terminal.
//   npm run dev   → ML service + API server + Vite dev server   (open http://localhost:5173)
//   npm start     → builds the client, then ML service + API server (open http://localhost:4000)
// Ctrl+C stops everything.
import net from 'node:net';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { bold, dirs, exists, fail, green, isWin, red, runNpm, spawn, venvPython, yellow } from './common.mjs';

const prod = process.argv.includes('--prod');
const ML_PORT = Number(process.env.ML_PORT) || 5001;
const API_PORT = Number(process.env.PORT) || 4000;
const WEB_PORT = 5173;
const url = prod ? `http://localhost:${API_PORT}` : `http://localhost:${WEB_PORT}`;
const viteBin = path.join(dirs.client, 'node_modules', 'vite', 'bin', 'vite.js');

if (!exists(venvPython) || !exists(path.join(dirs.server, 'node_modules')) || !exists(viteBin)) {
  fail(`Dependencies are not installed yet. Run ${bold('npm run setup')} first.`);
}
const deps = spawnSync(venvPython, ['-c', 'import flask, flask_cors, numpy, PIL'], { cwd: dirs.ml, encoding: 'utf8' });
if (deps.status !== 0) fail(`The Python packages are not fully installed. Run ${bold('npm run setup')} again.`);

// A port is taken if something accepts connections on it (any local address)
// or if we cannot bind it ourselves.
function canConnect(port, host) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host });
    const done = (result) => {
      sock.destroy();
      resolve(result);
    };
    sock.setTimeout(500, () => done(false));
    sock.once('connect', () => done(true));
    sock.once('error', () => done(false));
  });
}

function canBind(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once('error', () => resolve(false));
    srv.once('listening', () => srv.close(() => resolve(true)));
    srv.listen(port, '127.0.0.1');
  });
}

async function portFree(port) {
  if ((await canConnect(port, '127.0.0.1')) || (await canConnect(port, '::1'))) return false;
  return canBind(port);
}

const command = prod ? 'npm start' : 'npm run dev';
const envExample = (name, value) =>
  isWin ? `set ${name}=${value}&& ${command}   (PowerShell: $env:${name}=${value}; ${command})` : `${name}=${value} ${command}`;
const ports = [[ML_PORT, 'ML service', 'ML_PORT'], [API_PORT, 'API server', 'PORT']];
if (!prod) ports.push([WEB_PORT, 'web dev server', null]);
for (const [port, name, envVar] of ports) {
  if (!(await portFree(port))) {
    fail(`Port ${port} (needed by the ${name}) is already in use — is LeafCare already running in another terminal?\n  ` +
      (envVar ? `Stop that program, or choose another port, e.g.  ${envExample(envVar, port + 10)}` : 'Stop that program first.'));
  }
}

if (prod) {
  console.log(bold('▸ Building the web client…'));
  if (!runNpm(['run', 'build'], dirs.client).ok) fail('Client build failed (see the message above).');
}

const children = [];
let stopping = false;

function start(tag, colorFn, cmd, args, cwd, env = {}) {
  const child = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  const prefix = colorFn(`[${tag}]`.padEnd(6));
  const pipe = (stream, out) => {
    let buf = '';
    stream.on('data', (chunk) => {
      buf += chunk;
      const lines = buf.split(/\r?\n/);
      buf = lines.pop();
      for (const line of lines) out.write(`${prefix} ${line}\n`);
    });
  };
  pipe(child.stdout, process.stdout);
  pipe(child.stderr, process.stderr);
  child.on('error', (err) => {
    console.error(`${prefix} ${red(`failed to start: ${err.message}`)}`);
    stop(1);
  });
  child.on('exit', (code, signal) => {
    // On Ctrl+C the terminal signals the children too; give our own SIGINT
    // handler a moment to run so a normal shutdown isn't reported as a crash.
    setTimeout(() => {
      if (stopping) return;
      console.error(`\n${prefix} ${red(`exited (${signal || `code ${code}`}) — stopping the other services.`)}`);
      stop(1);
    }, 300);
  });
  children.push(child);
}

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) if (child.exitCode === null) child.kill();
  setTimeout(() => process.exit(code), 500);
}
process.on('SIGINT', () => stop(0));
process.on('SIGTERM', () => stop(0));

const cyan = (s) => (process.stdout.isTTY ? `\x1b[36m${s}\x1b[0m` : s);
const magenta = (s) => (process.stdout.isTTY ? `\x1b[35m${s}\x1b[0m` : s);

// The ML service is only ever called by the API, so it stays on localhost.
start('ml', green, venvPython, ['app.py'], dirs.ml, { PORT: String(ML_PORT), LEAFCARE_HOST: '127.0.0.1', PYTHONUNBUFFERED: '1' });
start('api', cyan, process.execPath, ['src/index.js'], dirs.server, {
  PORT: String(API_PORT),
  ML_SERVICE_URL: `http://127.0.0.1:${ML_PORT}`,
});
if (!prod) {
  start('web', magenta, process.execPath, [viteBin], dirs.client, { LEAFCARE_API_URL: `http://127.0.0.1:${API_PORT}` });
}

// Wait until the API reports the ML service healthy (and Vite is up in dev mode).
async function ready() {
  const ok = async (u, check) => {
    try {
      const res = await fetch(u);
      return check ? check(await res.json()) : res.ok;
    } catch {
      return false;
    }
  };
  // Wait for the API itself first (polling through Vite before the API listens
  // makes Vite print proxy-error traces), then confirm through the URL the user
  // will open, so a misrouted proxy can't be reported as ready.
  for (let i = 0; i < 120 && !stopping; i++) {
    const api = await ok(`http://127.0.0.1:${API_PORT}/api/health`, (b) => b.status === 'ok');
    if (api && (await ok(`${url}/api/health`, (b) => b.status === 'ok'))) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

if (await ready()) {
  const health = await fetch(`${url}/api/health`).then((r) => r.json()).catch(() => ({}));
  const mode = health.ml?.mode === 'model' ? `trained model, ${health.ml.classes} classes` : yellow('demo (heuristic) mode');
  console.log(`\n  ${green('✔ LeafCare is running')} (${mode})\n\n    Open ${bold(url)} in your browser.  Press Ctrl+C to stop.\n`);
} else if (!stopping) {
  console.log(yellow(`\n  ⚠ Services did not report ready within 60 s — check the log lines above. Then try ${url}\n`));
}

// Start the whole app in one terminal.
//   npm run dev   → ML service + API server + Vite dev server   (open http://localhost:5173)
//   npm start     → builds the client, then ML service + API server (open http://localhost:4000)
// Ctrl+C stops everything.
import net from 'node:net';
import path from 'node:path';
import { bold, dirs, exists, fail, green, red, runNpm, spawn, venvPython, yellow } from './common.mjs';

const prod = process.argv.includes('--prod');
const ML_PORT = Number(process.env.ML_PORT) || 5001;
const API_PORT = Number(process.env.PORT) || 4000;
const WEB_PORT = 5173;
const url = prod ? `http://localhost:${API_PORT}` : `http://localhost:${WEB_PORT}`;
const viteBin = path.join(dirs.client, 'node_modules', 'vite', 'bin', 'vite.js');

if (!exists(venvPython) || !exists(path.join(dirs.server, 'node_modules')) || !exists(viteBin)) {
  fail(`Dependencies are not installed yet. Run ${bold('npm run setup')} first.`);
}

function portFree(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once('error', () => resolve(false));
    srv.once('listening', () => srv.close(() => resolve(true)));
    srv.listen(port, '127.0.0.1');
  });
}

const ports = [[ML_PORT, 'ML service', 'ML_PORT'], [API_PORT, 'API server', 'PORT']];
if (!prod) ports.push([WEB_PORT, 'web dev server', null]);
for (const [port, name, envVar] of ports) {
  if (!(await portFree(port))) {
    fail(`Port ${port} (needed by the ${name}) is already in use — is LeafCare already running in another terminal?` +
      (envVar ? ` Stop that program, or choose another port, e.g. ${envVar}=${port + 10} npm run dev` : ' Stop that program first.'));
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

start('ml', green, venvPython, ['app.py'], dirs.ml, { PORT: String(ML_PORT), PYTHONUNBUFFERED: '1' });
start('api', cyan, process.execPath, ['src/index.js'], dirs.server, {
  PORT: String(API_PORT),
  ML_SERVICE_URL: `http://127.0.0.1:${ML_PORT}`,
});
if (!prod) start('web', magenta, process.execPath, [viteBin, '--strictPort'], dirs.client);

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
  for (let i = 0; i < 120 && !stopping; i++) {
    const api = await ok(`http://127.0.0.1:${API_PORT}/api/health`, (b) => b.status === 'ok');
    const web = prod || (await ok(`http://localhost:${WEB_PORT}/`));
    if (api && web) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

if (await ready()) {
  const health = await fetch(`http://127.0.0.1:${API_PORT}/api/health`).then((r) => r.json()).catch(() => ({}));
  const mode = health.ml?.mode === 'model' ? `trained model, ${health.ml.classes} classes` : yellow('demo (heuristic) mode');
  console.log(`\n  ${green('✔ LeafCare is running')} (${mode})\n\n    Open ${bold(url)} in your browser.  Press Ctrl+C to stop.\n`);
} else if (!stopping) {
  console.log(yellow(`\n  ⚠ Services did not report ready within 60 s — check the log lines above. Then try ${url}\n`));
}

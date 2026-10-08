// Shared helpers for the cross-platform setup / dev / test scripts.
// Uses only Node built-ins so it runs before any `npm install`.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const isWin = process.platform === 'win32';
export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const dirs = {
  ml: path.join(root, 'ml-service'),
  server: path.join(root, 'server'),
  client: path.join(root, 'client'),
};
export const venvPython = isWin
  ? path.join(dirs.ml, '.venv', 'Scripts', 'python.exe')
  : path.join(dirs.ml, '.venv', 'bin', 'python');

// Python versions with onnxruntime wheels (3.15 has none yet).
export const PY_MIN = [3, 10];
export const PY_MAX = [3, 14];

const color = (code) => (s) => (process.stdout.isTTY ? `\x1b[${code}m${s}\x1b[0m` : s);
export const green = color(32);
export const red = color(31);
export const yellow = color(33);
export const bold = color(1);

export function fail(message) {
  console.error(`\n${red('✖')} ${message}\n`);
  process.exit(1);
}

export function pythonVersion(cmd, args = []) {
  const r = spawnSync(cmd, [...args, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], {
    encoding: 'utf8',
    timeout: 20000,
  });
  if (r.status !== 0 || !r.stdout) return null;
  const [major, minor] = r.stdout.trim().split('.').map(Number);
  return Number.isInteger(major) && Number.isInteger(minor) ? [major, minor] : null;
}

export function supported([major, minor]) {
  const v = major * 100 + minor;
  return v >= PY_MIN[0] * 100 + PY_MIN[1] && v <= PY_MAX[0] * 100 + PY_MAX[1];
}

// Run npm without a shell where possible: `npm run x` sets npm_execpath to
// npm-cli.js, which we can run with this same Node binary on every OS.
export function npmCommand(args) {
  const cli = process.env.npm_execpath;
  if (cli && cli.endsWith('.js')) return { cmd: process.execPath, args: [cli, ...args], shell: false };
  // Windows needs a shell to run npm.cmd.
  return { cmd: 'npm', args, shell: isWin };
}

export function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (r.error) return { ok: false, error: r.error };
  return { ok: r.status === 0, status: r.status };
}

export function runNpm(args, cwd) {
  const { cmd, args: full, shell } = npmCommand(args);
  return run(cmd, full, { cwd, shell });
}

export function exists(p) {
  return fs.existsSync(p);
}

export { spawn };

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

// Returns [major, minor] with .machine, .bits and .macos (major version, or 0)
// attached, or null if the interpreter can't be run.
export function pythonVersion(cmd, args = []) {
  const probe = 'import sys, platform, struct; print("%d.%d %s %d %s" % (sys.version_info[0], sys.version_info[1], ' +
    'platform.machine() or "-", struct.calcsize("P") * 8, platform.mac_ver()[0] or "0"))';
  const r = spawnSync(cmd, [...args, '-c', probe], { encoding: 'utf8', timeout: 20000 });
  if (r.status !== 0 || !r.stdout) return null;
  const [ver, machine = '-', bits = '64', mac = '0'] = r.stdout.trim().split(/\s+/);
  const [major, minor] = ver.split('.').map(Number);
  if (!Number.isInteger(major) || !Number.isInteger(minor)) return null;
  return Object.assign([major, minor], {
    machine: machine.toLowerCase(),
    bits: Number(bits),
    macos: Number(mac.split('.')[0]) || 0,
  });
}

// Why onnxruntime (1.18+) can't be installed for this interpreter, or null if it
// can. Based on the wheels published on PyPI.
export function unsupportedReason(version) {
  const [major, minor] = version;
  const v = major * 100 + minor;
  const label = `Python ${major}.${minor}`;
  if (v < PY_MIN[0] * 100 + PY_MIN[1] || v > PY_MAX[0] * 100 + PY_MAX[1]) {
    return `${label} is not supported (need ${PY_MIN.join('.')}–${PY_MAX.join('.')})`;
  }
  if (version.bits === 32 && (!isWin || v >= 313)) return `32-bit ${label} has no onnxruntime build; use 64-bit Python`;
  if (isWin && version.machine === 'arm64' && v < 311) return `${label} on Windows ARM has no onnxruntime build (use 3.11+)`;
  if (process.platform === 'darwin') {
    const intel = ['x86_64', 'i386'].includes(version.machine);
    if (intel && v >= 314) return `${label} on an Intel Mac has no onnxruntime build (use 3.10–3.13)`;
    if (version.macos && v >= 314 && version.macos < 14) return `${label} needs macOS 14 or newer here (use 3.12)`;
    if (version.macos && v === 313 && version.macos < 13) return `${label} needs macOS 13 or newer here (use 3.12)`;
  }
  return null;
}

// Visual C++ runtime download for this Windows machine's CPU.
export function vcRedistUrl(machine = process.arch) {
  return /arm64/i.test(machine)
    ? 'https://aka.ms/vs/17/release/vc_redist.arm64.exe'
    : 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
}

export function supported(version) {
  return unsupportedReason(version) === null;
}

// Run npm without a shell where possible: `npm run x` sets npm_execpath to
// npm-cli.js, which we can run with this same Node binary on every OS.
export function npmCommand(args) {
  const cli = process.env.npm_execpath;
  if (cli && cli.endsWith('.js')) return { cmd: process.execPath, args: [cli, ...args], shell: false };
  // Windows needs a shell to run npm.cmd; pass one command string (passing an
  // args array with shell:true is deprecated in Node 24).
  if (isWin) return { cmd: ['npm', ...args].join(' '), args: [], shell: true };
  return { cmd: 'npm', args, shell: false };
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

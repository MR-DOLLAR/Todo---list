// One-time setup: Python virtual environment + packages for ml-service,
// npm packages for server/ and client/.   Usage: npm run setup
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {
  PY_MAX, PY_MIN, bold, dirs, exists, fail, green, isWin, pythonVersion, run, runNpm, unsupportedReason, vcRedistUrl,
  venvPython, yellow,
} from './common.mjs';

const fmt = ([a, b]) => `${a}.${b}`;
const range = `${fmt(PY_MIN)}–${fmt(PY_MAX)}`;
const venvDir = path.join(dirs.ml, '.venv');
const removeVenv = () => fs.rmSync(venvDir, { recursive: true, force: true });
const installHint = isWin
  ? 'Install 64-bit Python 3.13 from https://www.python.org/downloads/ (or: winget install Python.Python.3.13)'
  : process.platform === 'darwin'
    ? 'Install Python 3.13: brew install python@3.13 (or from https://www.python.org/downloads/)'
    : 'Install Python 3.12/3.13 with your package manager, e.g. sudo apt install python3 python3-venv';

function step(title) {
  console.log(`\n${bold('▸')} ${bold(title)}`);
}

function findPython() {
  // Prefer the best-supported versions first.
  const preferred = ['3.13', '3.12', '3.11', '3.14', '3.10'];
  const candidates = isWin
    ? [...preferred.map((v) => ['py', [`-${v}`]]), ['py', ['-3']], ['python', []]]
    : [...preferred.map((v) => [`python${v}`, []]), ['python3', []], ['python', []]];
  const seen = [];
  for (const [cmd, args] of candidates) {
    const version = pythonVersion(cmd, args);
    if (!version) continue;
    const reason = unsupportedReason(version);
    if (!reason) return { cmd, args, version, seen };
    seen.push(`${[cmd, ...args].join(' ')}: ${reason}`);
  }
  return { seen };
}

step(`Node.js ${process.versions.node}`);
const nodeMajor = Number(process.versions.node.split('.')[0]);
if (nodeMajor < 20) fail(`Node.js 20 or newer is required (22 LTS recommended). Download it from https://nodejs.org`);
console.log(green('  ok'));

step('Python virtual environment (ml-service/.venv)');
let venvVersion = exists(venvPython) ? pythonVersion(venvPython) : null;
if (venvVersion) {
  const reason = unsupportedReason(venvVersion);
  if (reason) {
    fail(`The existing ml-service/.venv can't be used: ${reason}.\n  ${installHint}, then delete the folder ` +
      'ml-service/.venv and run "npm run setup" again.');
  }
  // A venv left behind by an interrupted setup may lack pip; rebuild it.
  if (spawnSync(venvPython, ['-m', 'pip', '--version'], { encoding: 'utf8' }).status !== 0) {
    console.log(yellow('  Existing .venv is incomplete (no pip) — recreating it.'));
    removeVenv();
    venvVersion = null;
  }
} else if (exists(venvDir)) {
  console.log(yellow('  Existing .venv is broken — recreating it.'));
  removeVenv();
}
if (!venvVersion) {
  const py = findPython();
  if (!py.cmd) {
    const found = py.seen.length ? `\n  Found: ${py.seen.join('; ')}.` : ' No Python was found.';
    fail(`Python ${range} (64-bit) is required; 3.12 or 3.13 recommended.${found}\n  ${installHint}, ` +
      'then open a new terminal and run "npm run setup" again.');
  }
  console.log(`  Using ${[py.cmd, ...py.args].join(' ')} (Python ${fmt(py.version)})`);
  const r = spawnSync(py.cmd, [...py.args, '-m', 'venv', '.venv'], { cwd: dirs.ml, encoding: 'utf8' });
  if (r.status !== 0) {
    const out = `${r.stdout || ''}${r.stderr || ''}`;
    console.error(out);
    removeVenv(); // don't leave a half-made venv behind for the next run
    if (/ensurepip|-venv/.test(out)) {
      fail(`Python ${fmt(py.version)} is missing its venv module. On Ubuntu/Debian run:\n` +
        `    sudo apt install python${fmt(py.version)}-venv\n  then run "npm run setup" again.`);
    }
    fail('Could not create the Python virtual environment (see the message above).');
  }
  venvVersion = pythonVersion(venvPython);
  if (!venvVersion) {
    removeVenv();
    fail('The virtual environment was created but its Python does not run (see above). Run "npm run setup" again.');
  }
}
console.log(green(`  ok (Python ${fmt(venvVersion)})`));

step('Python packages (ml-service/requirements-dev.txt)');
const pip = run(venvPython, ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-dev.txt'], {
  cwd: dirs.ml,
});
if (!pip.ok) {
  fail('pip install failed (see the message above). Common causes:\n' +
    '  • No internet connection, or a proxy/firewall blocking pypi.org.\n' +
    '  • "No matching distribution found for onnxruntime": this Python version or CPU architecture has no\n' +
    `    onnxruntime build. ${installHint}, delete ml-service/.venv and run "npm run setup" again.`);
}

const ort = spawnSync(venvPython, ['-c', 'import onnxruntime'], { cwd: dirs.ml, encoding: 'utf8' });
if (ort.status !== 0) {
  console.log(yellow('  ⚠ onnxruntime is installed but cannot load, so the app will run in demo (heuristic) mode.'));
  if (isWin) {
    console.log(yellow('    Install the Microsoft Visual C++ Redistributable and run setup again:'));
    const arm = /arm64/i.test(venvVersion.machine);
    console.log(yellow(`    ${vcRedistUrl(venvVersion.machine)}   (or: winget install Microsoft.VCRedist.2015+.${arm ? 'arm64' : 'x64'})`));
  } else {
    console.log((ort.stderr || '').trim().split('\n').slice(-3).join('\n'));
  }
} else {
  console.log(green('  ok'));
}

for (const name of ['server', 'client']) {
  step(`npm packages (${name}/)`);
  const r = runNpm(['install', '--no-fund'], dirs[name]);
  if (!r.ok) fail(`npm install failed in ${name}/ (see the message above).`);
  console.log(green('  ok'));
}

console.log(`\n${green('✔ Setup complete.')} Start the app with:\n`);
console.log(`    ${bold('npm run dev')}     development mode → http://localhost:5173`);
console.log(`    ${bold('npm start')}       single server   → http://localhost:4000\n`);
if (isWin) console.log(`  (PowerShell blocking npm? Use ${bold('npm.cmd run dev')} or a Command Prompt.)\n`);

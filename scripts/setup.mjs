// One-time setup: Python virtual environment + packages for ml-service,
// npm packages for server/ and client/.   Usage: npm run setup
import { spawnSync } from 'node:child_process';
import {
  PY_MAX, PY_MIN, bold, dirs, exists, fail, green, isWin, pythonVersion, run, runNpm, supported, venvPython, yellow,
} from './common.mjs';

const fmt = ([a, b]) => `${a}.${b}`;
const range = `${fmt(PY_MIN)}–${fmt(PY_MAX)}`;

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
    seen.push(`${[cmd, ...args].join(' ')} → ${fmt(version)}`);
    if (supported(version)) return { cmd, args, version, seen };
  }
  return { seen };
}

step(`Node.js ${process.versions.node}`);
const nodeMajor = Number(process.versions.node.split('.')[0]);
if (nodeMajor < 20) fail(`Node.js 20 or newer is required (22 LTS recommended). Download it from https://nodejs.org`);
console.log(green('  ok'));

step('Python virtual environment (ml-service/.venv)');
let venvVersion = exists(venvPython) ? pythonVersion(venvPython) : null;
if (venvVersion && !supported(venvVersion)) {
  console.log(yellow(`  Existing .venv uses Python ${fmt(venvVersion)}, which is not supported (${range}).`));
  fail(`Delete the folder ml-service/.venv and run "npm run setup" again with Python ${range} installed.`);
}
if (!venvVersion) {
  const py = findPython();
  if (!py.cmd) {
    const found = py.seen.length ? `Found: ${py.seen.join(', ')}.` : 'No Python found.';
    fail(`Python ${range} is required (3.12 or 3.13 recommended). ${found}\n` +
      (isWin
        ? '  Install it from https://www.python.org/downloads/ or: winget install Python.Python.3.13'
        : '  macOS: brew install python@3.13   ·   Ubuntu/Debian: sudo apt install python3 python3-venv'));
  }
  console.log(`  Using ${[py.cmd, ...py.args].join(' ')} (Python ${fmt(py.version)})`);
  const r = spawnSync(py.cmd, [...py.args, '-m', 'venv', '.venv'], { cwd: dirs.ml, encoding: 'utf8' });
  if (r.status !== 0) {
    const out = `${r.stdout || ''}${r.stderr || ''}`;
    console.error(out);
    if (/ensurepip|python3.*-venv/.test(out)) {
      fail('Your Python is missing the venv module. On Ubuntu/Debian run: sudo apt install python3-venv ' +
        '(then delete ml-service/.venv if it exists and run "npm run setup" again).');
    }
    fail('Could not create the Python virtual environment (see the message above).');
  }
  venvVersion = pythonVersion(venvPython);
  if (!venvVersion) fail('The virtual environment was created but its Python does not run. Delete ml-service/.venv and retry.');
}
console.log(green(`  ok (Python ${fmt(venvVersion)})`));

step('Python packages (ml-service/requirements-dev.txt)');
const pip = run(venvPython, ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-dev.txt'], {
  cwd: dirs.ml,
});
if (!pip.ok) fail('pip install failed (see the message above). Check your internet connection and Python version.');

const ort = spawnSync(venvPython, ['-c', 'import onnxruntime'], { cwd: dirs.ml, encoding: 'utf8' });
if (ort.status !== 0) {
  console.log(yellow('  ⚠ onnxruntime is installed but cannot load, so the app will run in demo (heuristic) mode.'));
  if (isWin) {
    console.log(yellow('    Install the Microsoft Visual C++ Redistributable and run setup again:'));
    console.log(yellow('    https://aka.ms/vs/17/release/vc_redist.x64.exe   (or: winget install Microsoft.VCRedist.2015+.x64)'));
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

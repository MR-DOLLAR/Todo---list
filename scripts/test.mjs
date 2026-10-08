// Run all automated tests: Python (ml-service) and Node (server).   Usage: npm test
import { bold, dirs, exists, fail, green, run, runNpm, venvPython } from './common.mjs';

if (!exists(venvPython)) fail(`Run ${bold('npm run setup')} first.`);

console.log(bold('\n▸ ml-service (pytest)'));
const py = run(venvPython, ['-m', 'pytest', '-q'], { cwd: dirs.ml });
console.log(bold('\n▸ server (node --test)'));
const node = runNpm(['test'], dirs.server);

if (!py.ok || !node.ok) fail('Some tests failed.');
console.log(green('\n✔ All tests passed.\n'));

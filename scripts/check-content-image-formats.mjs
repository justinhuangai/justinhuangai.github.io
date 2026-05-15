import { readFileSync } from 'node:fs';
import { relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { globSync } from 'node:fs';

const root = fileURLToPath(new URL('..', import.meta.url));
const contentFiles = globSync('src/content/**/*.{md,mdx}', { cwd: root });
const disallowedRasterPattern = /\.(?:png|jpe?g|gif|avif)(?:[?#][^\s"')\]]*)?(?=["')\]\s]|$)/gi;

const failures = [];

for (const file of contentFiles) {
  const absolutePath = new URL(`../${file}`, import.meta.url);
  const source = readFileSync(absolutePath, 'utf8');
  const lines = source.split(/\r?\n/);

  lines.forEach((line, index) => {
    const matches = line.match(disallowedRasterPattern);
    if (!matches) return;

    failures.push({
      file: relative(root, file),
      line: index + 1,
      matches: [...new Set(matches.map((match) => match.replace(/[?#].*$/, '')))],
    });
  });
}

if (failures.length > 0) {
  console.error('Content image format check failed: use .webp for raster images in src/content.');
  for (const failure of failures) {
    console.error(`- ${failure.file}:${failure.line} -> ${failure.matches.join(', ')}`);
  }
  process.exit(1);
}

console.log('Content image format check passed: raster images in src/content use WebP.');

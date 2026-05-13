import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import ts from 'typescript';
import { createIndex, close } from 'pagefind';

import { defaultLocale as DEFAULT_LOCALE, locales as LOCALES } from '../config/locale-meta.mjs';

const repoRoot = process.cwd();
const sourceFile = path.join(repoRoot, 'src/data/directory-links.ts');
const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'directory-pagefind-'));
const tempModule = path.join(tempDir, 'directory-links.mjs');
const TS_NO_CHECK_HEADER = '// @ts-nocheck\n';

function getOutputPath() {
  const outputFlagIndex = process.argv.indexOf('--output');
  if (outputFlagIndex !== -1 && process.argv[outputFlagIndex + 1]) {
    return path.resolve(repoRoot, process.argv[outputFlagIndex + 1]);
  }

  return path.join(repoRoot, 'dist/pagefind');
}

function resolveLocalizedValue(locale, value) {
  if (typeof value === 'string') {
    return value;
  }

  return value?.[locale] ?? value?.[DEFAULT_LOCALE] ?? value?.['zh-CN'] ?? value?.['zh-TW'] ?? '';
}

function slugify(input) {
  return input
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function createDirectoryAnchorId(input, index) {
  const slug = slugify(input);
  return slug ? `directory-group-${index + 1}-${slug}` : `directory-group-${index + 1}`;
}

function createDirectoryCardId(groupInput, linkInput, groupIndex, linkIndex) {
  const slug = slugify(`${groupInput}-${linkInput}`);
  return slug
    ? `directory-card-${groupIndex + 1}-${linkIndex + 1}-${slug}`
    : `directory-card-${groupIndex + 1}-${linkIndex + 1}`;
}

function normalizePagefindLanguage(locale) {
  if (locale.startsWith('zh-')) {
    return 'zh';
  }

  if (locale === 'pt-BR') {
    return 'pt';
  }

  return locale.split('-')[0] || DEFAULT_LOCALE;
}

function buildDirectoryPath(locale, anchorId) {
  const prefix = locale === DEFAULT_LOCALE ? '' : `/${locale}`;
  return `${prefix}/directory#${anchorId}`;
}

async function annotateJavaScriptFiles(rootDir) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });

  for (const entry of entries) {
    const entryPath = path.join(rootDir, entry.name);

    if (entry.isDirectory()) {
      await annotateJavaScriptFiles(entryPath);
      continue;
    }

    if (!entry.isFile() || path.extname(entry.name) !== '.js') {
      continue;
    }

    const source = await fs.readFile(entryPath, 'utf8');

    if (source.startsWith(TS_NO_CHECK_HEADER)) {
      continue;
    }

    await fs.writeFile(entryPath, `${TS_NO_CHECK_HEADER}${source}`, 'utf8');
  }
}

try {
  const source = await fs.readFile(sourceFile, 'utf8');
  const transpiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: sourceFile,
  });

  await fs.writeFile(tempModule, transpiled.outputText, 'utf8');
  const { directoryGroupsSource } = await import(`${pathToFileURL(tempModule).href}?t=${Date.now()}`);

  const outputPath = getOutputPath();
  await fs.rm(outputPath, { force: true, recursive: true });

  const { index, errors } = await createIndex();

  if (errors?.length) {
    throw new Error(`Pagefind index initialization failed: ${errors.join(', ')}`);
  }

  for (const locale of LOCALES) {
    for (const [groupIndex, group] of directoryGroupsSource.entries()) {
      const groupDisplayName = resolveLocalizedValue(locale, group.spec.displayName);
      const anchorId = createDirectoryAnchorId(groupDisplayName, groupIndex);

      for (const [linkIndex, link] of group.links.entries()) {
        const linkDisplayName = resolveLocalizedValue(locale, link.spec.displayName);
        const linkDescription = resolveLocalizedValue(locale, link.spec.description);

        await index.addCustomRecord({
          url: buildDirectoryPath(locale, anchorId),
          language: normalizePagefindLanguage(locale),
          content: [linkDisplayName, linkDescription, groupDisplayName, link.spec.url]
            .filter(Boolean)
            .join('\n'),
          meta: {
            title: linkDisplayName,
            description: linkDescription,
            group: groupDisplayName,
            locale,
            cardId: createDirectoryCardId(groupDisplayName, linkDisplayName, groupIndex, linkIndex),
            anchorId,
            href: link.spec.url,
          },
          filters: {
            locale: [locale],
            group: [anchorId],
          },
        });
      }
    }
  }

  await index.writeFiles({ outputPath });
  await annotateJavaScriptFiles(outputPath);
} finally {
  await close();
  await fs.rm(tempDir, { force: true, recursive: true });
}

# Justin Huang Site

**English** | [简体中文](./README_ZH-CN.md) | [繁體中文](./README_ZH-TW.md) | [한국어](./README_KO-KR.md)

This branch contains the Astro source that powers [justinhuangai.github.io](https://justinhuangai.github.io).

The site is a personal writing space about AI, technology, writing, and life. It is built on top of the Aither design system, but this repository is the live site instance, not the generic theme template.

## What Makes This Project Different

- Human and agent entry points coexist on the same content model.
- Machine-readable endpoints are first-class: `/protocol.json`, `/agent/home.json`, `/skill.md`, `/llms.txt`, `/llms-full.txt`, `/api/posts.json`, and per-post `.md`.
- The runtime is multilingual, with shared locale metadata for routing, UI, and validation.
- The reading experience is intentionally design-driven, with a large theme system layered on top of a static Astro architecture.

## Current Stack

- Astro 6
- Tailwind CSS v4
- React 19 islands where interaction is needed
- GitHub Pages deployment from `gh-pages`
- Locales: `en-US`, `zh-CN`, `zh-TW`, `ko-KR`

## Branch Model

- `main`: small public repository landing page
- `gh-pages`: source of truth for the live site and deployment workflow
- temporary branches such as `source`: experiments only

Operational details live in [MAINTAINING.md](./MAINTAINING.md).

## Quick Start

```bash
pnpm install
pnpm dev
pnpm validate
```

## Validation

`pnpm validate` now runs the full publish safety chain:

- `pnpm check:content` checks post coverage across the configured locales and flags suspicious math markup in posts
- `pnpm check` runs Astro diagnostics
- `pnpm typecheck` runs TypeScript without emit
- `pnpm build` produces the static site
- `pnpm check:agent` smoke-tests the built agent protocol endpoints

## Key Files

- `config/locale-meta.mjs`: single source of truth for supported locales
- `src/config/site.ts`: site identity, nav, analytics, comments, UI defaults
- `src/content/`: localized posts and static content
- `src/lib/agent-protocol.ts`: machine-readable discovery and policy documents
- `.github/workflows/deploy-github-pages.yml`: CI/CD for GitHub Pages
- `MAINTAINING.md`: deployment and upgrade workflow

## Deployment

Pushes to `origin/gh-pages` trigger the GitHub Pages workflow in `.github/workflows/deploy-github-pages.yml`.

Before pushing, run:

```bash
pnpm validate
```

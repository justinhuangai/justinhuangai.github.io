---
title: 👋 Hello World
date: "2026-01-01T16:07:13+08:00"
category: Tutorial
description: Welcome to Astro-Theme-Aither — an AI-native Astro theme that believes text itself is beautiful.
tags: [hello, astro]
pinned: false
---

Welcome to Astro-Theme-Aither.

This is an AI-native blog theme built on one belief: text itself is beautiful. A unified sans-serif system font stack, Apple HIG typography parameters, and a layout that stays out of your way. Everything here serves a single goal — making your words look and feel beautiful.

## Why Another Blog Theme

The web is full of blog themes, so a fair question is: why build another one? The answer comes down to priorities. Most themes optimize for visual impact — large hero images, complex layouts, animated transitions. These look stunning in a demo but get in the way when someone actually sits down to read a 2,000-word article.

Astro-Theme-Aither starts from a different premise. The content is the product. The theme's job is to present that content with the care it deserves: Apple HIG body text parameters (17px / 1.47 / -0.022em), generous whitespace, and a vertical rhythm that makes long-form reading comfortable rather than exhausting.

This philosophy extends to the technical decisions too. The theme uses Astro's islands architecture — only interactive components (theme switcher, language switcher, locale detection, mobile nav) load JavaScript. Everything else is static HTML and CSS. No layout shifts, no loading spinners. The page loads, and you read.

## Get Started

Getting up and running takes just a few minutes:

1. **Clone the repository** — use the GitHub template button or clone directly with `git clone`
2. **Install dependencies** — run `pnpm install` to pull in all packages
3. **Configure your site** — edit `src/config/site.ts` to set your site title, description, and nav links
4. **Set up services** — copy `.env.example` to `.env` and fill in your API keys (GA, Crisp, Giscus)
5. **Replace sample content** — swap the posts in `src/content/posts/` with your own Markdown files
6. **Start developing** — run `pnpm dev` to launch the local dev server with hot reloading
7. **Deploy** — push to `gh-pages` and let the included GitHub Pages workflow publish the site

### Project Structure

```
src/
├── components/     # Reusable Astro & React components
├── config/         # Site configuration (site.ts)
├── content/        # Your Markdown posts (organized by locale)
├── i18n/           # Translations and locale utilities
├── layouts/        # Page layouts (Layout.astro)
├── lib/            # Shared utilities (posts, formatter, markdown-endpoint)
├── pages/          # Route pages (per locale)
└── styles/         # Global CSS with Tailwind v4 @theme tokens
```

Each directory has a clear responsibility. Components are small and composable. Layouts handle the document shell. Pages define routes. Content holds your writing organized by locale.

### Writing Your First Post

Create a new `.md` file in `src/content/posts/en-US/` with the following frontmatter:

```markdown showLanguage
---
title: Your Post Title
date: "2026-01-15T16:27:43+08:00"
category: General
description: A brief summary for SEO and social previews
tags: [topic, another]
pinned: false
---

Your content starts here.
```

The `title`, `date`, and `category` fields are required. Use ISO 8601 for `date`, including seconds and a timezone offset, for example `2026-01-15T16:27:43+08:00`. The `description` field is strongly recommended because it populates the meta description tag and Open Graph previews. Tags are optional. Set `pinned: false` to pin a post to the top of the list.

For multilingual content, create the same file in each locale directory (`zh-CN/`, `zh-TW/`, `ko-KR/`) with translated content.

## What You Get

Out of the box, you have a production-ready blogging platform with every feature you need and none of the bloat you don't.

### Content Features

- **RSS feed** — automatically generated at `/rss.xml`
- **Sitemap** — auto-generated via `@astrojs/sitemap`
- **SEO meta tags** — Open Graph, Twitter cards, and canonical URLs on every page
- **JSON-LD** — Article structured data for AI and search engines
- **Dark mode** — Light / Dark / System toggle with circular reveal animation via View Transitions API
- **i18n** — multi-language support with automatic browser language detection
- **Post pinning** — pin important posts to the top of the list
- **Pagination** — file-based SSG pagination with page number navigation

### AI-Native Features

- **llms.txt** — AI agent content index at `/llms.txt`
- **llms-full.txt** — full-text content for AI consumption at `/llms-full.txt`
- **Markdown endpoints** — append `.md` to any post URL for clean Markdown output
- **robots.txt** — explicitly welcomes AI crawlers (GPTBot, ClaudeBot, PerplexityBot)

### Developer Features

- **TypeScript throughout** — strict mode, fully typed components and utilities
- **Content Collections** — type-safe Markdown with frontmatter validation at build time
- **Tailwind CSS v4** — `@theme` design tokens for easy customization
- **Validation workflow** — content coverage checks plus agent protocol smoke tests through `pnpm validate`
- **Deploy** — GitHub Pages workflow included
- **Google Analytics** — optional, via environment variable
- **Crisp Chat** — optional live chat, via environment variable
- **Giscus Comments** — optional GitHub Discussions powered comments

### Performance

Because the theme outputs static HTML with minimal JavaScript islands, performance is excellent by default. You should expect Lighthouse scores of 100 across the board — Performance, Accessibility, Best Practices, and SEO.

## Customization

- **Colors** — edit CSS custom properties in `src/styles/global.css`
- **Fonts** — swap font-family values in the Tailwind theme configuration
- **Navigation** — update nav links in `src/config/site.ts`
- **Services** — set environment variables in `.env` for GA, Crisp, and Giscus
- **Languages** — add new locales in `src/i18n/` and create corresponding page routes

For deeper changes, the component architecture is deliberately simple. Each component does one thing, reads its props, and renders HTML.

## A Note on Design Philosophy

The visual simplicity of this theme is intentional, but it is not the same as engineering simplicity. Under the hood, the theme handles a surprising number of concerns: Apple HIG typography parameters, accessible color contrast ratios in both light and dark modes, View Transitions API animations, automatic browser language detection, proper semantic HTML structure, AI-friendly content endpoints, and careful attention to the reading experience on screens ranging from phones to ultrawide monitors.

Good design is invisible. When you read an article on this theme and simply enjoy the writing without noticing the theme at all — that is the design working exactly as intended.

Happy writing.

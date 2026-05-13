---
title: 👋 你好，世界
date: "2026-01-01T16:07:13+08:00"
category: Tutorial
description: 欢迎来到 Astro-Theme-Aither——一个相信文字本身就很美的 AI 原生 Astro 主题。
tags: [hello, astro]
pinned: false
---

欢迎来到 Astro-Theme-Aither。

这是一个基于一个信念构建的 AI 原生博客主题：文字本身就很美。统一的无衬线系统字体栈、Apple HIG 排版参数，以及不喧宾夺主的布局。这里的一切都服务于一个目标——让你的文字看起来优美、读起来舒适。

## 为什么再造一个博客主题

互联网上有无数博客主题，那为什么还要再做一个？答案在于优先级。大多数主题为视觉冲击力而优化——大图、复杂布局、华丽动画。这些在演示中很好看，但当读者真正坐下来阅读一篇两千字的文章时，它们只会碍事。

Astro-Theme-Aither 从不同的前提出发：内容就是产品。主题的职责是以它应得的认真态度呈现内容：Apple HIG 正文参数（17px / 1.47 / -0.022em）、充足的留白、让长文阅读变得舒适而不是疲惫的垂直节奏。

技术决策也延续了这一理念。主题使用 Astro 的岛屿架构——只有交互组件（主题切换、语言切换、语言检测、移动端导航）加载 JavaScript。其他一切都是静态 HTML 和 CSS。没有布局偏移，没有加载动画。页面加载完毕，你开始阅读。

## 开始使用

搭建只需几步：

1. **克隆仓库** — 使用 GitHub 模板按钮或直接 `git clone`
2. **安装依赖** — 运行 `pnpm install`
3. **配置站点** — 编辑 `src/config/site.ts` 设置标题、描述和导航
4. **配置服务** — 复制 `.env.example` 为 `.env`，填入 API 密钥（GA、Crisp、Giscus）
5. **替换内容** — 用你自己的文章替换 `src/content/posts/` 中的示例
6. **本地开发** — 运行 `pnpm dev` 启动热更新开发服务器
7. **部署** — 推送到 `gh-pages`，内置 GitHub Pages 工作流会自动发布站点

### 项目结构

```
src/
├── components/     # Astro 和 React 组件
├── config/         # 站点配置（site.ts）
├── content/        # Markdown 文章（按语言组织）
├── i18n/           # 翻译和语言工具
├── layouts/        # 页面布局（Layout.astro）
├── lib/            # 共享工具（posts, formatter, markdown-endpoint）
├── pages/          # 路由页面（按语言）
└── styles/         # 全局 CSS + Tailwind v4 @theme 令牌
```

### 写第一篇文章

在 `src/content/posts/zh-CN/` 下创建 `.md` 文件：

```markdown showLanguage
---
title: 我的第一篇文章
date: "2026-01-15T16:27:43+08:00"
category: General
description: SEO 和社交预览的简短摘要
tags: [topic, another]
pinned: false
---

正文从这里开始。
```

`title`、`date`、`category` 是必填项。`date` 建议使用带时区、精确到秒的 ISO 8601 格式，例如 `2026-01-15T16:27:43+08:00`。`description` 强烈建议填写。设置 `pinned: false` 可将文章置顶。

多语言内容只需在对应语言目录（`en-US/`、`zh-TW/`、`ko-KR/`）创建同名文件。

## 开箱即用

### 内容功能

- **RSS 订阅** — 自动生成 `/rss.xml`
- **站点地图** — 通过 `@astrojs/sitemap` 自动生成
- **SEO 标签** — 每页自动生成 Open Graph、Twitter Cards、规范 URL
- **JSON-LD** — Article 结构化数据，服务 AI 和搜索引擎
- **深色模式** — 浅色 / 深色 / 系统切换，View Transitions API 圆形展开动画
- **i18n** — 多语言支持，自动浏览器语言检测
- **文章置顶** — 将重要文章固定在列表顶部
- **分页** — 基于文件的 SSG 分页，带页码导航

### AI 原生功能

- **llms.txt** — AI agent 内容索引，`/llms.txt`
- **llms-full.txt** — AI 全文消费，`/llms-full.txt`
- **Markdown 端点** — 任何文章 URL 后加 `.md` 获取纯 Markdown
- **robots.txt** — 明确欢迎 AI 爬虫（GPTBot、ClaudeBot、PerplexityBot）

### 开发者功能

- **TypeScript** — 严格模式，全类型化
- **Content Collections** — 构建时 frontmatter 类型验证
- **Tailwind CSS v4** — `@theme` 设计令牌
- **校验链路** — 通过 `pnpm validate` 运行内容覆盖检查和 agent 协议 smoke test
- **部署** — 内置 GitHub Pages 工作流
- **Google Analytics** — 可选，环境变量配置
- **Crisp Chat** — 可选在线客服，环境变量配置
- **Giscus 评论** — 可选 GitHub Discussions 评论

### 性能

静态 HTML + 最小 JavaScript 岛屿 = Lighthouse 四项满分。

## 自定义

- **颜色** — 编辑 `src/styles/global.css` 中的 CSS 变量
- **字体** — 修改 Tailwind 主题配置中的字体族
- **导航** — 更新 `src/config/site.ts` 中的导航数组
- **服务** — 在 `.env` 中设置 GA、Crisp、Giscus 环境变量
- **语言** — 在 `src/i18n/` 添加新语言，创建对应路由

## 设计理念

主题的视觉简洁是刻意的，但这不等于工程简单。底层处理了大量关注点：Apple HIG 排版参数、明暗两种模式的无障碍色彩对比、View Transitions API 动画、自动浏览器语言检测、语义化 HTML 结构、AI 友好的内容端点，以及从手机到超宽屏的阅读体验优化。

好的设计是隐形的。当你在这个主题上阅读一篇文章，只是单纯享受文字而完全没注意到主题的存在——这就是设计在按预期工作。

祝写作愉快。

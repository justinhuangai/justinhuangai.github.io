# Justin Huang Site

[English](./README.md) | **简体中文** | [繁體中文](./README_ZH-TW.md) | [한국어](./README_KO-KR.md)

这个分支包含 [justinhuangai.github.io](https://justinhuangai.github.io) 的 Astro 源码。

站点本身是一个关于 AI、技术、写作和生活的个人写作空间。视觉和交互建立在 Aither 设计系统之上，但这个仓库是实际线上站点实例，不是通用主题模板。

## 这个项目的特别之处

- 同一份内容同时服务人类读者和 AI agent。
- 机器可读端点是一等能力：`/protocol.json`、`/agent/home.json`、`/skill.md`、`/llms.txt`、`/llms-full.txt`、`/api/posts.json`、每篇文章的 `.md`。
- 运行时是多语言站点，路由、UI 和校验共享同一份 locale 配置。
- 阅读体验是设计驱动的，主题系统很丰富，但底层仍然保持 Astro 静态架构的稳定性。

## 当前技术栈

- Astro 6
- Tailwind CSS v4
- React 19 islands
- GitHub Pages，从 `gh-pages` 分支部署
- 当前 locale：`en-US`、`zh-CN`、`zh-TW`、`ko-KR`

## 分支约定

- `main`：对外展示的仓库主页说明
- `gh-pages`：线上站点源码与部署真源
- `source` 等临时分支：实验用途，不进入正式发布流

详细维护流程见 [MAINTAINING.md](./MAINTAINING.md)。

## 快速开始

```bash
pnpm install
pnpm dev
pnpm validate
```

## 校验链路

现在 `pnpm validate` 会串起完整发布校验：

- `pnpm check:content`：检查各 locale 的文章覆盖是否一致，并拦截文章里可疑的数学公式标记写法
- `pnpm check`：运行 Astro 诊断
- `pnpm typecheck`：运行 TypeScript 无输出校验
- `pnpm build`：构建静态站点
- `pnpm check:agent`：对构建产物中的 agent 协议端点做 smoke test

## 关键文件

- `config/locale-meta.mjs`：支持语言的单一真源
- `src/config/site.ts`：站点身份、导航、分析、评论和 UI 默认值
- `src/content/`：多语言内容
- `src/lib/agent-protocol.ts`：机器可读发现、协议和策略文档
- `.github/workflows/deploy-github-pages.yml`：GitHub Pages 发布流程
- `MAINTAINING.md`：部署与升级说明

## 部署

推送到 `origin/gh-pages` 会触发 `.github/workflows/deploy-github-pages.yml` 中的 GitHub Pages 工作流。

推送前请先执行：

```bash
pnpm validate
```

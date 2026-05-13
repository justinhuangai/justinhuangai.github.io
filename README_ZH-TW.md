# Justin Huang Site

[English](./README.md) | [简体中文](./README_ZH-CN.md) | **繁體中文** | [한국어](./README_KO-KR.md)

這個分支包含 [justinhuangai.github.io](https://justinhuangai.github.io) 的 Astro 原始碼。

站點本身是一個關於 AI、技術、寫作與生活的個人寫作空間。視覺與互動建立在 Aither 設計系統之上，但這個倉庫是實際上線站點實例，不是通用主題模板。

## 這個專案的特點

- 同一份內容同時服務人類讀者與 AI agent。
- 機器可讀端點是一等能力：`/protocol.json`、`/agent/home.json`、`/skill.md`、`/llms.txt`、`/llms-full.txt`、`/api/posts.json`、每篇文章的 `.md`。
- 執行期是多語站點，路由、UI 與驗證共享同一份 locale 設定。
- 閱讀體驗由設計驅動，主題系統豐富，但底層仍保持 Astro 靜態架構的穩定性。

## 目前技術棧

- Astro 6
- Tailwind CSS v4
- React 19 islands
- GitHub Pages，從 `gh-pages` 分支部署
- 目前 locale：`en-US`、`zh-CN`、`zh-TW`、`ko-KR`

## 分支約定

- `main`：對外展示的倉庫首頁說明
- `gh-pages`：上線站點原始碼與部署真源
- `source` 等臨時分支：實驗用途，不進入正式發佈流程

詳細維護流程見 [MAINTAINING.md](./MAINTAINING.md)。

## 快速開始

```bash
pnpm install
pnpm dev
pnpm validate
```

## 驗證鏈路

現在 `pnpm validate` 會串起完整發佈檢查：

- `pnpm check:content`：檢查各 locale 的文章覆蓋是否一致，並攔截文章裡可疑的數學公式標記寫法
- `pnpm check`：執行 Astro 診斷
- `pnpm typecheck`：執行 TypeScript 無輸出檢查
- `pnpm build`：建構靜態站點
- `pnpm check:agent`：對建構產物中的 agent 協議端點做 smoke test

## 關鍵檔案

- `config/locale-meta.mjs`：支援語言的單一真源
- `src/config/site.ts`：站點身份、導覽、分析、評論與 UI 預設值
- `src/content/`：多語內容
- `src/lib/agent-protocol.ts`：機器可讀發現、協議與策略文件
- `.github/workflows/deploy-github-pages.yml`：GitHub Pages 發佈流程
- `MAINTAINING.md`：部署與升級說明

## 部署

推送到 `origin/gh-pages` 會觸發 `.github/workflows/deploy-github-pages.yml` 裡的 GitHub Pages 工作流。

推送前請先執行：

```bash
pnpm validate
```

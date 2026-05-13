---
title: 👋 你好，世界
date: "2026-01-01T16:07:13+08:00"
category: Tutorial
description: 歡迎來到 Astro-Theme-Aither——一個相信文字本身就很美的 AI 原生 Astro 主題。
tags: [hello, astro]
pinned: false
---

歡迎來到 Astro-Theme-Aither。

這是一個基於一個信念構建的 AI 原生部落格主題：文字本身就很美。統一的無襯線系統字型棧、Apple HIG 排版參數，以及不喧賓奪主的版面。這裡的一切都服務於一個目標——讓你的文字看起來優美、讀起來舒適。

## 為什麼再造一個部落格主題

網路上有無數部落格主題，那為什麼還要再做一個？答案在於優先順序。大多數主題為視覺衝擊力而最佳化——大圖、複雜版面、華麗動畫。這些在展示中很好看，但當讀者真正坐下來閱讀一篇兩千字的文章時，它們只會礙事。

Astro-Theme-Aither 從不同的前提出發：內容就是產品。主題的職責是以它應得的認真態度呈現內容：Apple HIG 正文參數（17px / 1.47 / -0.022em）、充足的留白、讓長文閱讀變得舒適而不是疲憊的垂直節奏。

技術決策也延續了這一理念。主題使用 Astro 的島嶼架構——只有互動元件（主題切換、語言切換、語言偵測、行動端導覽）載入 JavaScript。其他一切都是靜態 HTML 和 CSS。沒有版面偏移，沒有載入動畫。頁面載入完畢，你開始閱讀。

## 開始使用

搭建只需幾步：

1. **複製倉庫** — 使用 GitHub 範本按鈕或直接 `git clone`
2. **安裝依賴** — 執行 `pnpm install`
3. **設定網站** — 編輯 `src/config/site.ts` 設定標題、描述和導覽
4. **設定服務** — 複製 `.env.example` 為 `.env`，填入 API 金鑰（GA、Crisp、Giscus）
5. **替換內容** — 用你自己的文章替換 `src/content/posts/` 中的範例
6. **本地開發** — 執行 `pnpm dev` 啟動熱更新開發伺服器
7. **部署** — 推送到 `gh-pages`，內建 GitHub Pages 工作流會自動發佈網站

### 專案結構

```
src/
├── components/     # Astro 和 React 元件
├── config/         # 網站設定（site.ts）
├── content/        # Markdown 文章（按語言組織）
├── i18n/           # 翻譯和語言工具
├── layouts/        # 頁面版面（Layout.astro）
├── lib/            # 共用工具（posts, formatter, markdown-endpoint）
├── pages/          # 路由頁面（按語言）
└── styles/         # 全域 CSS + Tailwind v4 @theme 令牌
```

### 寫第一篇文章

在 `src/content/posts/zh-TW/` 下建立 `.md` 檔案：

```markdown showLanguage
---
title: 我的第一篇文章
date: "2026-01-15T16:27:43+08:00"
category: General
description: SEO 和社群預覽的簡短摘要
tags: [topic, another]
pinned: false
---

正文從這裡開始。
```

`title`、`date`、`category` 是必填項。`date` 建議使用帶時區、精確到秒的 ISO 8601 格式，例如 `2026-01-15T16:27:43+08:00`。`description` 強烈建議填寫。設定 `pinned: false` 可將文章置頂。

多語系內容只需在對應語言目錄（`en-US/`、`zh-CN/`、`ko-KR/`）建立同名檔案。

## 開箱即用

### 內容功能

- **RSS 訂閱** — 自動產生 `/rss.xml`
- **網站地圖** — 透過 `@astrojs/sitemap` 自動產生
- **SEO 標籤** — 每頁自動產生 Open Graph、Twitter Cards、規範 URL
- **JSON-LD** — Article 結構化資料，服務 AI 和搜尋引擎
- **深色模式** — 淺色 / 深色 / 系統切換，View Transitions API 圓形展開動畫
- **i18n** — 多語言支援，自動瀏覽器語言偵測
- **文章置頂** — 將重要文章固定在列表頂部
- **分頁** — 基於檔案的 SSG 分頁，帶頁碼導覽

### AI 原生功能

- **llms.txt** — AI agent 內容索引，`/llms.txt`
- **llms-full.txt** — AI 全文消費，`/llms-full.txt`
- **Markdown 端點** — 任何文章 URL 後加 `.md` 取得純 Markdown
- **robots.txt** — 明確歡迎 AI 爬蟲（GPTBot、ClaudeBot、PerplexityBot）

### 開發者功能

- **TypeScript** — 嚴格模式，全型別化
- **Content Collections** — 建構時 frontmatter 型別驗證
- **Tailwind CSS v4** — `@theme` 設計令牌
- **驗證鏈路** — 透過 `pnpm validate` 執行內容覆蓋檢查與 agent 協議 smoke test
- **部署** — 內建 GitHub Pages 工作流
- **Google Analytics** — 可選，環境變數設定
- **Crisp Chat** — 可選線上客服，環境變數設定
- **Giscus 評論** — 可選 GitHub Discussions 評論

### 效能

靜態 HTML + 最小 JavaScript 島嶼 = Lighthouse 四項滿分。

## 自訂

- **顏色** — 編輯 `src/styles/global.css` 中的 CSS 變數
- **字型** — 修改 Tailwind 主題設定中的字型族
- **導覽** — 更新 `src/config/site.ts` 中的導覽陣列
- **服務** — 在 `.env` 中設定 GA、Crisp、Giscus 環境變數
- **語言** — 在 `src/i18n/` 新增語言，建立對應路由

## 設計理念

主題的視覺簡潔是刻意的，但這不等於工程簡單。底層處理了大量關注點：Apple HIG 排版參數、明暗兩種模式的無障礙色彩對比、View Transitions API 動畫、自動瀏覽器語言偵測、語意化 HTML 結構、AI 友好的內容端點，以及從手機到超寬螢幕的閱讀體驗最佳化。

好的設計是隱形的。當你在這個主題上閱讀一篇文章，只是單純享受文字而完全沒注意到主題的存在——這就是設計在按預期工作。

祝寫作愉快。

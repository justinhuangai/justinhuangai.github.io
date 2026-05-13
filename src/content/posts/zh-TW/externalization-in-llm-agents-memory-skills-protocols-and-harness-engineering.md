---
title: "《Externalization in LLM Agents》：LLM Agent 的認知外部化"
description: "這篇綜述把 memory、skills、protocols 和 harness engineering 放進同一個視角：Agent 的進步越來越像是在模型外部重寫任務，而不只是讓模型權重更強。"
date: "2026-05-13T16:00:00+08:00"
category: "Paper Reading"
tags: [paper-reading, agents, externalization, harness-engineering, memory, skills, protocols, LLM]
pinned: false
---

很多關於 Agent 的討論，最後都會回到模型排行榜。

哪個模型推理更強，哪個上下文視窗更長，哪個在程式 benchmark 裡得分更高，這些問題當然重要。但 [《Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering》](/papers/2604.08224v1.pdf) 指向了另一個更底層的重心。

它問的不是「模型還能學會什麼」，而是：

**哪些原本壓在模型身上的認知負擔，其實應該被搬到模型外部？**

這就是論文使用 `externalization` 這個詞的原因。它不是單純替 LLM 加幾個外掛，也不是把工具清單塞進 prompt。它描述的是任務表示方式的改變。人類使用紙、地圖、日曆和電腦也是這樣：外部物件沒有讓大腦本身突然變大，卻改變了問題的形狀。回憶變成辨識，心算變成符號操作，臨場發揮變成遵循流程。

LLM Agent 也正在經歷類似的轉變。

![論文 Figure 1：LLM Agent 設計中的外部化主線](/images/posts/externalization-in-llm-agents/figure-1-externalization-overview.png)

*Figure 1, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), format-converted for web display, content unchanged.*

## 這篇綜述釐清了什麼

這不是一篇新模型論文，也不是 benchmark 論文。它的價值在於概念整理：替正在發生的工程遷移命名。

早期我們習慣把能力理解成權重裡的東西：更大模型、更好預訓練、更強對齊。接著，能力越來越多地被放進上下文裡組織：prompt、few-shot 範例、RAG、推理軌跡、工具說明。到了實際 Agent 系統，能力又進一步進入更厚的 runtime：記憶庫、檔案系統、技能庫、工具協議、沙箱、審批、日誌、評估器、子 Agent 編排。

論文把這條路徑概括成：

**weights → context → harness**

![作者理解圖：LLM Agent 的認知外部化地圖](/images/posts/externalization-in-llm-agents/cognitive-externalization-map.webp)

*作者使用 AI 生成的概念圖，不是論文原圖。它總結了本文對認知外部化的理解：模型能力從 weights，經由 context，逐步進入 memory、skills、protocols 和 governance 等 harness-level infrastructure。*

這個概括很有用，因為它符合真實系統的手感。模型仍然重要，但它不再獨自承擔整個任務。它被放進一個會記住、會檢索、會約束、會審計、會失敗恢復的環境裡。

換句話說，可靠性不只是模型屬性，也是任務如何被外部化的結果。

## Memory：把時間搬到模型外面

最容易理解的外部化是 memory。

如果 Agent 只依賴上下文視窗，每次執行都只是帶著一段臨時工作記憶。更長的上下文有幫助，但不能消除選擇問題：什麼該放進來，什麼該忘掉，什麼已經過時，什麼只是噪音。更關鍵的是，上下文是暫時性的。沒有外部狀態，任務結束後很多經驗就消失了。

外部 memory 把問題從「模型能不能憑空回憶」改寫成「模型能不能辨識並使用被檢索出的狀態」。這和 cognitive artifacts 的觀點一致：購物清單不是擴大生物記憶，而是改變記憶任務。

對 Agent 來說，memory 可以保存使用者偏好、專案約定、歷史決策、失敗軌跡、領域事實和重複出現的限制。真正困難的地方不只是儲存，而是篩選：哪些狀態值得寫入，什麼時候檢索，檢索多少，如何壓縮，如何避免舊狀態污染新判斷。

所以 memory 不是「更長上下文」的替代品，而是時間維度上的基礎設施。

## Skills：把過程搬到模型外面

第二種外部化是 skills。

這裡的 skill 不是「模型會呼叫某個工具」，而是一段可重用的過程性知識。它可能包含步驟、判斷啟發式、停止條件、升級規則、失敗恢復方式和安全限制。一個成熟的 skill 告訴 Agent：這類任務通常應該怎麼做。

這和工具調用不是同一層抽象。工具提供動作，protocol 定義動作如何被發現與呼叫，skill 則封裝如何把動作組織成一件可重複完成的事。

![論文 Figure 5：技能作為外部化的過程性知識](/images/posts/externalization-in-llm-agents/figure-5-skills-lifecycle.png)

*Figure 5, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

軟體工程 Agent 讓這件事特別具體。

模型可能知道如何改程式、跑測試、讀錯誤日誌、寫 PR 摘要。但穩定完成任務，還需要很多本地流程：先讀哪些檔案，如何保護使用者改動，什麼時候用 `rg`，什麼時候跑完整驗證，如何處理既有 staged changes，哪些命令不能隨便執行。

如果這些都留給模型每次現場推導，行為就會漂移。當它們被外部化成 skill，任務就從「臨場發明工作流」變成「選擇並執行已驗證的流程」。

這也是 skills 容易被低估的地方。它不炫，但它直接降低行為方差。

## Protocols：把互動秩序搬到模型外面

第三種外部化是 protocols。

沒有 protocol 的 Agent 互動，本質上是自由文本協商。模型說自己要呼叫工具，工具回傳一段文字，另一個 Agent 再猜那段文字代表什麼。這在 demo 裡可行，在生產系統裡會很脆。

Protocol 把含糊互動變成機器可讀的契約。工具如何發現，參數如何宣告，權限如何表達，錯誤如何回傳，Agent 之間如何委託，使用者審批如何進入流程，這些都不應該只靠 prompt 約定。

它的價值也不只是互操作性。Protocol 還提供治理入口。只要互動是結構化的，系統就能驗證、審計、回放、監控和限權。自由文本很靈活，但很難治理。

## Harness：不是包裝器，而是認知環境

論文最重要的收束，是把 memory、skills 和 protocols 放進 harness engineering 裡。

Harness 不是包在模型外面的一層薄 wrapper。它是 Agent 實際運作的認知環境：控制迴圈、上下文預算、權限模型、沙箱、人工審批、日誌、評估、失敗恢復、子 Agent 編排，都在這裡發生。

![論文 Figure 3：Harnessed LLM Agent 的外部化架構](/images/posts/externalization-in-llm-agents/figure-3-harnessed-agent-architecture.png)

*Figure 3, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

重點是，harness 不是和 memory、skills、protocols 並列的另一個模組。它是承載並協調這些外部化結構的 runtime。Memory 提供狀態，skills 提供過程，protocols 提供互動結構，harness 決定它們何時被載入、如何彼此影響、要受哪些約束。

這也解釋了為什麼成熟 Agent 越來越像小型操作環境。它們有資源、權限、生命週期、日誌、策略和恢復路徑，而不只是「一個模型加一串工具」。

## 能力邊界正在移動

這篇論文最有用的地方，是它改變了「能力在哪裡」這個問題。

如果能力只在權重裡，改進 Agent 主要就是換模型、微調或重新訓練。如果能力也在上下文裡，prompt 和檢索就成為工程主戰場。如果能力進一步進入 harness，那麼系統設計本身就是能力的一部分。

這不是在貶低模型。相反，越強的模型，越值得被放進更好的外部結構裡。LLM 擅長綜合、判斷和泛化，但它並不天然擅長持久記憶、重複流程、權限治理、長期狀態和跨系統協調。

外部化不是作弊。它是工程上承認邊界，然後重寫任務。

好的 Agent 系統不會逼模型每次都從零開始想辦法。它會把可沉澱的狀態變成 memory，把可重用的流程變成 skills，把可治理的交換變成 protocols，再用 harness 把它們組織成可運行的環境。

## 外部化也有代價

這個框架很漂亮，但外部化不是免費擴展。

Memory 會帶來過時狀態、隱私邊界和檢索污染。Skills 可能過期、過度擬合，或在錯誤組合時變得不安全。Protocols 可能碎片化成互不相容的標準，也可能把系統鎖進僵硬介面。Harness 則會增加複雜度：審批、日誌、沙箱、策略和子流程越多，就越需要工程紀律。

還有評估問題。如果 Agent 能力分布在模型與外部基礎設施之間，我們到底在評測什麼？同一個模型放進不同 harness，表現可能完全不同。「模型能力」和「Agent 能力」已經不是同一件事。

這正是論文值得讀的地方。它不是說外部化解決一切，而是把問題放在正確的位置：可靠的 agency 來自模型與環境的共同設計。

## 實用判斷

這篇綜述可以壓縮成一句實用判斷：

**Agent engineering is increasingly harness engineering.**

不是因為模型不再重要，而是因為模型已經進入一個更大的認知系統。Memory 讓它跨越時間，skills 讓它穩定流程，protocols 讓它遵守互動秩序，harness 讓整件事可觀察、有權限、可恢復。

所以下次看一個 Agent 系統，我們不只問它用了哪個模型，也要問：

- 它外部化了哪些狀態？
- 它帶有哪些可重用技能？
- 它的工具與 Agent 互動是否 protocolized？
- 它的 harness 如何處理權限、失敗、日誌與人工審批？

這些問題，通常比模型名稱更接近系統的真實能力。

---
title: "《Language Models are Few-Shot Learners》：GPT-3 與上下文學習"
date: "2026-02-11T16:22:54+08:00"
category: "Paper Reading"
description: 更大的模型，更善於從上下文中誘發能力，附真實 Python 程式碼
tags: [paper-reading, gpt-3, AI, LLM, python]
pinned: false
---

2020 年 5 月 28 日，OpenAI 在 arXiv（一個學術論文預印本網站，論文不用等期刊審稿就能直接發布）上傳了一篇 75 頁的論文：[《Language Models are Few-Shot Learners》](/papers/2005.14165v4.pdf)（語言模型是少樣本學習者）。

作者有 31 人，全部來自 OpenAI。第一作者 Tom B. Brown，其餘包括 Jared Kaplan（縮放定律的核心研究者）、Alec Radford（GPT-1 和 GPT-2 的主要設計者）、Ilya Sutskever（OpenAI 聯合創始人兼首席科學家）、Dario Amodei（OpenAI 研究副總裁）等。

這份作者名單後來分化出了幾家最重要的 AI 公司：Dario Amodei 和 Jared Kaplan 離開 OpenAI 創立了 Anthropic，Ilya Sutskever 後來也聯合創立了 Safe Superintelligence Inc.（SSI）。

論文的核心主張很直接：把語言模型做大，大到 1750 億參數，它就能在不更新任何權重的情況下，僅靠幾個示例就完成各種任務：有時甚至逼近經過專門微調的模型。

這不是任務級微調，而是在固定參數下，透過上下文適配任務的能力：論文稱之為**上下文學習（in-context learning）**。

## 0. 先認幾個詞

如果你對 GPT-3 這類模型的工作方式還沒有概念，先記住下面幾個詞就夠了：

- `語言模型`：給它一段前文，它的基本工作就是預測下一個詞。
- `參數量`：模型裡可學習的數字總數。你可以粗略把它理解成模型的「腦容量」。
- `prompt / 提示詞`：你餵給模型看的任務說明、示例和輸入。
- `上下文視窗`：模型一次能看到多少文本的容量。例子太多，塞不進去，就沒辦法一起看。
- `few-shot / one-shot / zero-shot`：分別指給多個例子、給一個例子、完全不給例子。
- `in-context learning / 上下文學習`：不改模型參數，只靠 prompt 裡的說明和例子，就讓模型臨時學會怎麼做任務。

## 1. 要解決什麼問題

[BERT](/zh-TW/posts/bert/) 確立的「預訓練 + 微調」範式在 2020 年已經是主流做法。效果很好，但論文指出了三個根本問題。

第一，每個新任務仍然需要一個標註資料集。標註資料的獲取成本高，且很多實際任務根本沒有對應的標註集。

第二，微調後的模型在測試基準上的表現，不一定反映真實泛化能力。模型可能只是學到了訓練資料中的虛假相關性（spurious correlations）：在基準集裡得分很高，但換個分布就崩了。

第三，人類不是這樣學習的。人類看一兩個例子，聽一句自然語言指令，就能完成新任務。而當時的 NLP 系統，每個新任務都需要成千上萬條標註資料來微調。

論文的出發點是：如果模型足夠大，它在預訓練階段累積的知識是否足以讓它直接「讀懂」任務描述和少量示例，然後給出答案？

## 2. 核心想法：不更新參數，只給提示

GPT-3 的評估方式和之前所有大模型都不一樣。它定義了三種設置，全部不涉及梯度更新：

**少樣本（Few-Shot）**：給模型一段任務描述，加上 10 到 100 個示例（具體數量取決於上下文窗口能裝多少），然後讓它完成新的輸入。不更新權重，不做反向傳播。

**單樣本（One-Shot）**：只給一個示例。這最接近人類學習新任務的方式：有人給你演示一次，你就上手。

**零樣本（Zero-Shot）**：連示例都不給，只有一句自然語言指令。這是最難的設置，但也是最實用的：如果模型真的「理解」了任務本身，它不應該需要任何例子。

```python showLanguage
from dataclasses import dataclass
from typing import Union


@dataclass
class ZeroShot:
    instruction: str
    prompt: str


@dataclass
class OneShot:
    instruction: str
    example: tuple[str, str]
    prompt: str


@dataclass
class FewShot:
    instruction: str
    examples: list[tuple[str, str]]
    prompt: str


EvalSetting = Union[ZeroShot, OneShot, FewShot]


def build_prompt(setting: EvalSetting) -> str:
    if isinstance(setting, ZeroShot):
        return f"{setting.instruction}\n{setting.prompt}"

    if isinstance(setting, OneShot):
        example_input, example_output = setting.example
        return f"{setting.instruction}\n{example_input} {example_output}\n{setting.prompt}"

    lines = [setting.instruction]
    lines.extend(f"{example_input} {example_output}" for example_input, example_output in setting.examples)
    lines.append(setting.prompt)
    return "\n".join(lines)
```

論文把這種能力叫做**上下文學習**：模型在預訓練時，從海量文本中隱式地學到了各種任務的模式；推理時，示例被拼接進上下文，模型在前向傳播的過程中「識別」出當前任務是什麼，然後完成它。論文用「元學習」來描述這個過程：預訓練是外循環，上下文學習是內循環。

這和微調的區別是根本性的。微調修改模型參數來適應任務，上下文學習不修改任何東西：同一個模型，同一組權重，只靠輸入文本的不同，就能切換任務。

## 3. 模型架構與規模

GPT-3 的架構本身沒有新發明。它和 GPT-2 一樣，就是 [Transformer](/zh-TW/posts/attention-is-all-you-need/) 的解碼器部分，一層層堆起來。改動只有一處：在 Transformer 層中交替使用稠密注意力和局部帶狀稀疏注意力（來自 Sparse Transformer）。

真正不同的是規模。論文訓練了 8 個不同大小的模型，參數量跨越三個數量級：

| 模型 | 參數量 | 層數 | 隱藏維度 | 注意力頭數 |
|------|--------|------|----------|-----------|
| GPT-3 Small | 1.25 億 | 12 | 768 | 12 |
| GPT-3 Medium | 3.5 億 | 24 | 1024 | 16 |
| GPT-3 Large | 7.6 億 | 24 | 1536 | 16 |
| GPT-3 XL | 13 億 | 24 | 2048 | 24 |
| GPT-3 2.7B | 27 億 | 32 | 2560 | 32 |
| GPT-3 6.7B | 67 億 | 32 | 4096 | 32 |
| GPT-3 13B | 130 億 | 40 | 5140 | 40 |
| **GPT-3 175B** | **1750 億** | **96** | **12288** | **96** |

1750 億參數，96 層，96 個注意力頭，隱藏維度 12288。上下文窗口 2048 個詞元。這個規模在當時是前所未見的：比 GPT-2（15 億參數）大了 100 多倍。

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT3Config:
    n_params: int
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_ctx: int


def gpt3_175b() -> GPT3Config:
    return GPT3Config(
        n_params=175_000_000_000,
        n_layers=96,
        d_model=12_288,
        n_heads=96,
        d_head=128,
        d_ff=49_152,
        n_ctx=2_048,
    )
```

論文訓練這些模型的目的很明確：驗證縮放定律（scaling laws）。之前 Kaplan 等人的研究（就是這篇論文的共同作者之一）已經表明，語言模型的損失和參數量之間存在平滑的冪律關係。GPT-3 把這個假設推到了 1750 億參數的規模，看看上下文學習能力是否也遵循同樣的規律。

答案是肯定的：模型越大，少樣本學習的提升越陡。零樣本性能隨模型規模穩步上升，少樣本性能的上升速度更快。這意味著大模型不只是「更準」，它們在利用上下文資訊的效率上也更高。

## 4. 訓練資料

GPT-3 在大約 3000 億個詞元上訓練，資料來自五個來源：

| 資料集 | 詞元數 | 訓練佔比 |
|--------|--------|----------|
| Common Crawl（過濾後） | 4100 億 | 約 60% |
| WebText2 | 190 億 | 約 22% |
| Books1 | 120 億 | 約 8% |
| Books2 | 550 億 | 約 8% |
| 英文 Wikipedia | 30 億 | 約 3% |

注意一個關鍵細節：資料集的取樣比例和它們的大小不成正比。品質更高的資料集（WebText2、Books、Wikipedia）被過取樣了：WebText2 在訓練中被看了 2.9 遍，Wikipedia 被看了 3.4 遍，而 Common Crawl 連一遍都沒看完（0.44 遍）。論文有意用少量過擬合的代價，換取更高品質的訓練訊號。

Common Crawl 的原始資料有 45TB，經過三步處理：（1）基於與高品質參考語料的相似度做過濾；（2）文件級模糊去重；（3）混入已知的高品質資料集來增加多樣性。過濾後剩下 570GB，約 4100 億詞元。

所有模型在 V100 GPU 上訓練，使用微軟提供的高頻寬叢集。

## 5. 實驗結果

論文在二十多個資料集上做了評估，覆蓋 9 大類任務。以下是幾個關鍵結果。

**語言建模**：在 Penn Tree Bank 上，GPT-3 少樣本困惑度（perplexity，衡量模型對文本的「意外程度」，越低越好）達到 20.50，刷新了當時的紀錄。在 LAMBADA（需要根據長距離上下文預測最後一個詞）上，零樣本準確率 76.2%，少樣本 86.4%，大幅超過之前的最好結果。

**翻譯**：GPT-3 從未被專門訓練過翻譯，但在法語→英語翻譯上，少樣本 BLEU 分數達到 32.6，超過了無監督神經機器翻譯的最好結果。不過英語→法語方向（25.2 BLEU）和微調模型的差距仍然很大。一個有趣的發現：GPT-3 翻譯成英語的能力明顯強於從英語翻譯出去，這和訓練資料以英語為主有直接關係。

**閉卷問答**：在 TriviaQA 上，少樣本準確率（exact match）71.2%，超過了同一閉卷設置下經過微調的模型。模型不看任何參考文件，純靠參數裡儲存的知識回答問題。

**SuperGLUE**：在這個綜合基準上，GPT-3 的少樣本表現已經接近一些經過微調的強基線，但仍落後於當時最強的專門微調系統。

**合成任務**：論文還設計了一些專門測試上下文學習能力的新任務。比如給模型幾個「造新詞」的例子（定義一個不存在的詞，然後用它造句），GPT-3 能正確地學會並使用這個新詞。再比如三位數加法，少樣本準確率接近 100%（兩位數加法也幾乎完美），但四五位數時急劇下降。

```python showLanguage
from typing import Callable, Protocol


class AutoregressiveModel(Protocol):
    def forward(self, tokens: list[int]) -> list[list[float]]:
        ...


def in_context_learning(
    model: AutoregressiveModel,
    examples: list[tuple[str, str]],
    query: str,
    tokenize: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    sample_from: Callable[[list[float]], int],
    eos_token: int,
) -> str:
    prompt_lines = [f"{example_input} {example_output}" for example_input, example_output in examples]
    prompt_lines.append(query)
    prompt = "\n".join(prompt_lines)

    context = tokenize(prompt)
    output_tokens: list[int] = []

    while True:
        logits = model.forward(context)
        next_token = sample_from(logits[-1])
        if next_token == eos_token:
            break
        output_tokens.append(next_token)
        context.append(next_token)

    return decode(output_tokens)
```

## 6. 資料污染問題

論文在第四章花了大量篇幅討論一個棘手的問題：訓練資料和測試資料的重疊。

GPT-3 的訓練資料包含大量網路文本，而很多測試基準的內容也在網路上公開存在。這意味著模型可能在訓練時就「看過」了測試題。論文團隊嘗試在訓練前移除這些重疊，但由於一個處理流程中的 bug，部分重疊沒有被完全清除。而重新訓練一遍的成本太高，不現實。

他們的做法是：為每個基準建構一個「乾淨子集」（移除所有和訓練資料有 13-gram 重疊的樣本），然後對比模型在完整集和乾淨子集上的表現。結論是：大多數基準上，污染對結果的影響很小。但 PIQA 和 Winograd 兩個資料集存在可疑的表現下降，論文對這些結果加了星號標註。

這種誠實在當時相當罕見。多數論文對資料污染問題避而不談。GPT-3 不僅主動調查，還開發了系統化的檢測工具。這本身就是對後續研究的一個貢獻。

## 7. 局限性

論文在第五章對自身局限性的討論相當坦率。

**文本連貫性**：GPT-3 在文件級別仍然會出現語義重複、自相矛盾、甚至生成無意義句子的情況。生成品質雖然比 GPT-2 好了很多，但長文本的連貫性仍然不夠。

**常識物理**：GPT-3 對「把起司放進冰箱，它會融化嗎？」這類常識物理問題表現不佳。它能處理語言層面的推理，但對物理世界的理解仍然是膚淺的。

**單向性的代價**：作為自回歸模型，GPT-3 只能從左往右看。論文承認，在需要雙向上下文的任務上（比如判斷兩個句子裡同一個詞的含義是否相同），GPT-3 的少樣本表現不如經過微調的雙向模型。這說明在 GPT-3 的自回歸設定下，這類任務並不是它的強項；單向建模目標本身會帶來結構性偏好。

**取樣效率**：GPT-3 在預訓練階段看了約 3000 億個詞元，遠超人類一生接觸的文本量。論文明確指出，即使少樣本學習在推理時很高效，預訓練的資料需求仍然巨大。

**推理成本**：1750 億參數的模型，推理成本高且不方便部署。論文提到蒸餾（distillation，用大模型的輸出來訓練小模型）是一個可能的方向，但在千億參數量級上還沒有嘗試過。

## 8. 社會影響

論文用了整整一個章節（第六章）討論社會影響，涵蓋三個方面。

**濫用風險**：GPT-3 生成的新聞文章，人類評估者的識別準確率接近隨機猜測（約 52%）。模型越強，生成的虛假文本越難辨別。論文團隊表示已經在監控論壇和聊天群，追蹤惡意使用的趨勢。

**偏見**：論文用大量實驗測試了 GPT-3 在性別、種族和宗教方面的偏見。例如，在職業-性別關聯測試中，GPT-3 更傾向於將「nurse」和女性關聯、將「banker」和男性關聯。在宗教-情感關聯中，「Islam」更多地與暴力相關詞共現。論文承認這些偏見來自訓練資料，但沒有給出解決方案。

**能源消耗**：訓練 GPT-3 需要大量算力，論文引用了估算資料但沒有公布具體的能耗數字。不過論文指出，一旦訓練完成，模型可以被多次使用到不同任務上，比為每個任務單獨訓練模型更節能。

## 9. 幾點觀察

讀完這篇論文，有幾個感受。

第一，GPT-3 展示了一件事：規模能把上下文學習推到可用閾值。1750 億參數的模型不只是「更大的 GPT-2」，它在上下文學習上的表現比小模型強出了一個量級。模型在沒有任何參數更新的情況下，僅靠上下文中的幾個示例就能完成新任務。這種能力不是顯式手工設計出來的，而是在規模擴大過程中逐步增強，到 GPT-3 這個量級才第一次變得足夠清晰、足夠實用。BERT 證明了預訓練的價值，GPT-3 證明了規模的價值。

第二，論文的寫作方式值得注意。31 個作者，75 頁篇幅，用了大量實驗來回答一個簡單的問題：更大的模型是否更善於利用少量示例？他們沒有迴避局限性：文本連貫性、常識推理、資料污染、偏見：全部正面討論。這種嚴謹程度，在後來的大模型論文中反而越來越少見了。

第三，這篇論文的作者列表就是一部 AI 行業分裂史。Dario Amodei 和 Jared Kaplan 後來創立了 Anthropic（Claude 的開發商），Ilya Sutskever 後來離開 OpenAI 創立了 SSI。這些人在 2020 年還在同一個團隊裡合作寫論文，兩年後就走向了不同的方向。論文裡關於社會影響和安全風險的討論，也許正是後來分歧的伏筆。

第四，從技術演進的角度看，GPT-3 是從「預訓練 + 微調」到「預訓練 + 提示」的轉折點。BERT 的思路是：先學通用知識，再在每個任務上微調參數。GPT-3 說：如果模型夠大，微調這一步可以省掉：直接用自然語言告訴模型你要做什麼。這個思路後來演化成了 ChatGPT、Claude 等產品的核心互動範式：使用者用自然語言提問，模型直接回答。

從 Seq2Seq 的編碼-解碼，到 [Bahdanau 注意力](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)的「該看哪裡」，到 [Transformer](/zh-TW/posts/attention-is-all-you-need/) 的「所有位置同時看」，到 [BERT](/zh-TW/posts/bert/) 的「先學再調」，再到 GPT-3 的「大到不用調」：每一步都在減少人工干預，增加模型自主完成任務的能力。

GPT-3 不是終點。但它第一次讓人們認真思考一個問題：如果繼續把模型做大，還會湧現出什麼？

這個問題的答案，就是後來發生的一切。

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學：為什麼更大的模型可預測地更好
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

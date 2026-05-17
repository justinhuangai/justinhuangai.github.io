---
title: "《Training Compute-Optimal Large Language Models》：Chinchilla 改變了什麼"
date: "2026-03-11T16:58:04+08:00"
category: "Paper Reading"
description: Chinchilla 論文：為什麼大多數大模型其實訓練不足，以及如何聰明地分配算力預算，附真實 Python 核心程式碼
tags: [paper-reading, chinchilla, scaling-laws, AI, LLM, python]
pinned: false
---

Chinchilla 要糾正的不是「規模是否有效」，而是固定算力下規模該怎麼分。2022 年很多大型模型把錢主要花在參數上，卻沒有給同等增長的資料；結果不是模型太小，而是模型沒有被充分訓練。

[《Training Compute-Optimal Large Language Models》](/papers/2203.15556v1.pdf) 的關鍵結論不是反對大型模型，而是指出參數和資料必須共同吃滿算力預算。更強不等於只把模型做大，更強來自參數、資料和訓練步數之間的正確比例。

## 0. 先認幾個詞

這篇論文會頻繁談到「該把預算花在哪裡」，所以先把下面幾個詞認熟，會更容易抓住重點：

- `算力預算`：你願意為這次訓練總共花多少計算資源。
- `參數量`：模型有多大。
- `token / 詞元`：模型訓練時實際讀進去的最小文本單位，可以粗略理解成「模型看到的字或詞片段」。
- `loss / 損失`：模型整體錯得有多厲害，通常越低越好。
- `scaling law / 縮放定律`：當參數量、資料量、算力變化時，模型表現如何跟著變化。
- `欠訓練 / undertrained`：不是模型太小，而是訓練資料和訓練步數沒跟上，模型的潛力沒有被充分用出來。

## 1. 要解決什麼問題

到 2022 年初，AI 社群已經內化了 [Kaplan 等人（2020）](/zh-TW/posts/scaling-laws-for-neural-language-models/)的一個明確教訓：更大的模型可預測地更好。縮放定律論文已經表明性能遵循冪律，而且在固定算力預算下，你應該把模型做到盡可能大。

業界照做了。到 2022 年春天，GPT-3 有 1750 億參數，訓練了 3000 億詞元。DeepMind 自己的 Gopher 有 2800 億參數，訓練了 3000 億詞元。隨後 Google 又發布了 5400 億參數的 PaLM。趨勢很明顯：往上堆參數。

但一個問題藏在眼皮底下。Kaplan 等人的結論是，當你擴大算力時，大部分預算應該花在模型大小上（N ∝ C^0.73），而相對少地花在訓練資料上（D ∝ C^0.27）。這意味著：把模型做得巨大，用適量的資料訓練就好。

Hoffmann 的團隊問了一個簡單的問題：這真的對嗎？

## 2. 三種獨立方法，同一個答案

這篇論文格外有說服力的地方在於方法論。團隊沒有依賴單一實驗，而是從三個完全獨立的角度切入同一個問題，三者最終收斂到同一個答案。

**方法一：固定算力，改變分配。** 他們訓練了超過 400 個模型，參數量從 7000 萬到 160 億以上，每個模型在模型大小和訓練資料之間有不同的分配比例，但總算力相同。對每個算力水平，他們找出哪個模型大小能讓損失最低。

**方法二：IsoFLOP 曲線。** 他們訓練了 9 種不同大小的模型（從 7000 萬到 100 億參數），使用不同量的資料，特別設計為每組實驗使用大致相同的總算力。然後擬合曲線，找出每個算力水平對應的最優模型大小。

**方法三：擬合參數化損失函數。** 他們把以下方程式擬合到所有訓練結果：

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

其中 E 是不可約損失（自然語言的熵：任何模型都無法做得更好），A/N^α 反映模型大小的瓶頸，B/D^β 反映資料量的瓶頸。從擬合出的參數，他們推導出最優的 N 和 D 作為算力的函數。

三種方法一致同意：

$$
N_{\mathrm{opt}} \propto C^a, \quad D_{\mathrm{opt}} \propto C^b, \quad a \approx 0.50, \quad b \approx 0.50
$$

```python showLanguage
def optimal_scaling(compute: float) -> tuple[float, float]:
    a = 0.50
    b = 0.50
    n_opt = compute ** a
    d_opt = compute ** b
    return n_opt, d_opt
```

指數 a ≈ b ≈ 0.5 的真正含義是：隨著算力增長，模型大小和訓練資料應當按近似相同的比例一起擴張。算力增長 10 倍時，兩者都大約增到 3.2 倍；算力翻倍時，兩者都大約增到 1.4 倍。換句話說，模型大小每翻一倍，訓練資料也應該翻一倍。這直接推翻了 Kaplan 等人的結論：他們認為算力主要應該花在模型大小上。

## 3. 為什麼 Kaplan 會錯

這不是誰「做錯了」，而是實驗設定不同，最終導致了不同的最優分配結論。兩個團隊都做了嚴謹的工作。

Kaplan 等人使用了固定的學習率調度，沒有根據訓練長度進行調整。當你讓模型訓練更多步而不調整學習率調度時，性能會下降：不是因為模型本身更差，而是因為優化過程不理想。這讓長時間訓練看起來沒那麼有效，從而偏向了更大的模型配更少的訓練步數。

Hoffmann 的團隊為每次訓練調整了學習率調度，確保每種配置都有公平的機會。這樣做之後，在更多資料上訓練更久，其實比 Kaplan 的數字所暗示的有價值得多。

```python showLanguage
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    n_params: float
    n_tokens: float
    schedule: Literal["fixed", "cosine_with_warmup"]
    warmup_steps: int
    total_steps: int
```

## 4. 參數化損失函數

論文的方法三值得細看，因為它給出了一個完整的性能數學模型：

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

其中擬合出的常數是：

- E = 1.69 — 不可約損失（自然語言的熵）
- A = 406.4, α = 0.34 — 模型大小項
- B = 410.7, β = 0.28 — 資料量項

這個方程式的結構值得細品。損失有三個組成部分：一個你永遠無法突破的下限（E），參數太少的懲罰（A/N^α），以及資料太少的懲罰（B/D^β）。模型大小的懲罰和資料量的懲罰是相加的：它們爭奪你的注意力和算力預算。

```python showLanguage
def estimated_loss(n_params: float, n_tokens: float) -> float:
    e = 1.69
    a = 406.4
    alpha = 0.34
    b = 410.7
    beta = 0.28
    return e + a / (n_params ** alpha) + b / (n_tokens ** beta)


def optimal_params_and_tokens(compute_flops: float) -> tuple[float, float]:
    alpha = 0.34
    beta = 0.28
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    g = 2.0

    base = compute_flops / 6.0
    n_opt = g * (base ** a)
    d_opt = (1.0 / g) * (base ** b)
    return n_opt, d_opt
```

## 5. 關鍵表格

論文的 Table 1 列出了當時幾個大模型的實際參數量和訓練詞元數，Table 3 則給出了不同模型大小下 compute-optimal 的詞元估計。把兩張表對照著看，就像一份對整個業界的審計報告：

| 模型 | 參數量 | 實際訓練詞元 | Chinchilla 最優詞元 |
|------|--------|-------------|---------------------|
| GPT-3 | 1750 億 | 3000 億 | 3.7 兆 |
| Gopher | 2800 億 | 3000 億 | 5.9 兆 |
| Jurassic-1 | 1780 億 | 3000 億 | 3.7 兆 |
| MT-NLG | 5300 億 | 2700 億 | 11.0 兆 |

每個模型都只用了大約 3000 億詞元。但按照 Chinchilla 的分析，GPT-3 應該用 3.7 兆詞元訓練：是實際的 12 倍以上。Gopher 應該接近 6 兆。最大的 MT-NLG（5300 億參數）應該用 11 兆詞元訓練：是實際訓練資料的 40 倍。

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelComparison:
    name: str
    params_billions: float
    tokens_used_billions: float
    optimal_tokens_billions: float


def industry_models() -> list[ModelComparison]:
    return [
        ModelComparison("GPT-3", 175.0, 300.0, 3_700.0),
        ModelComparison("Gopher", 280.0, 300.0, 5_900.0),
        ModelComparison("Jurassic-1", 178.0, 300.0, 3_700.0),
        ModelComparison("MT-NLG", 530.0, 270.0, 11_000.0),
    ]
```

規律非常明顯。整個業界不約而同地使用了差不多的訓練資料量：大約 3000 億詞元：不管模型有多大。彷彿所有人都認定 3000 億詞元「夠了」，然後把所有多出來的算力都堆到更大的模型上。Chinchilla 說這恰恰做反了。

## 6. 實證：Chinchilla 對決 Gopher

為了驗證理論，團隊訓練了 Chinchilla：一個 700 億參數的模型，用 1.4 兆詞元訓練。Chinchilla 使用的算力預算和 Gopher（2800 億參數，3000 億詞元）相同：同樣的總訓練成本，只是分配方式不同。

結果很決定性。Chinchilla 在幾乎所有基準測試上都優於 Gopher，儘管小了 4 倍：

- **MMLU**（大規模多任務語言理解）：Chinchilla 67.6% vs. Gopher 60.0% vs. GPT-3 43.9%
- **閱讀理解**（RACE-h）：Chinchilla 73.3% vs. Gopher 71.6%
- **常識推理**（HellaSwag）：Chinchilla 80.8% vs. Gopher 79.2%
- **BIG-bench**：Chinchilla 在大多數任務上優於 Gopher

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params_billions: float
    tokens_billions: float
    mmlu_accuracy: float


def chinchilla_vs_gopher() -> tuple[float, float]:
    gopher = ModelConfig("Gopher", 280.0, 300.0, 60.0)
    chinchilla = ModelConfig("Chinchilla", 70.0, 1_400.0, 67.6)

    gopher_flops = 6.0 * gopher.params_billions * 1e9 * gopher.tokens_billions * 1e9
    chinchilla_flops = 6.0 * chinchilla.params_billions * 1e9 * chinchilla.tokens_billions * 1e9
    return gopher_flops, chinchilla_flops
```

一個小 4 倍的模型在幾乎所有基準上打敗更大的模型：用同樣的算力：這是非常有力的實證。算力沒有浪費，只是從參數重新導向了資料。

## 7. 實際影響

Chinchilla 論文對業界產生了立即而具體的影響。

**更小的模型部署成本更低。** 訓練成本是一次性的，但推理成本：實際執行模型來生成文字的成本：與模型大小成正比，每一次使用者發送請求都要付出。一個 700 億參數的模型，部署成本是 2800 億參數模型的四分之一。如果更小的模型性能還更好，那是雙贏：更好的品質加上更低的成本。

**資料成了瓶頸。** Chinchilla 之前，限制因素是算力：你能搞到多少 GPU？Chinchilla 之後，限制因素變成了資料：你去哪裡找幾兆高品質的詞元？這引發了全業界對訓練資料的爭奪：大規模網路爬取、資料集篩選，最終催生了合成資料運動。

**LLaMA 時刻。** Meta 的 LLaMA（2023 年 2 月）可以說是 Chinchilla 縮放定律最直接的應用。LLaMA-13B 用 1 兆詞元訓練，在大多數基準上超過了 GPT-3（1750 億參數）。LLaMA-65B 用 1.4 兆詞元訓練，與 Chinchilla 和 PaLM-540B 處於同一水平。Meta 明確引用了 Chinchilla 論文，刻意訓練更小的模型、使用遠超過去慣例的資料量。

```python showLanguage
def inference_cost_comparison() -> tuple[float, float]:
    gopher_cost_per_token = 280.0
    chinchilla_cost_per_token = 70.0

    queries_per_day = 1_000_000.0
    tokens_per_query = 500.0

    daily_cost_gopher = queries_per_day * tokens_per_query * gopher_cost_per_token
    daily_cost_chinchilla = queries_per_day * tokens_per_query * chinchilla_cost_per_token
    return daily_cost_gopher, daily_cost_chinchilla
```

## 8. 這篇論文改變了什麼問題

Chinchilla 的釘子句是：**它不是反對大型模型，而是要求參數和資料一起吃滿算力預算。**

這篇論文的力量在於，它沒有喊停 scaling，而是修正了 scaling 的配比。Kaplan 給了產業做大的信心；Chinchilla 追問同樣一筆算力下，大多少、訓多少才是最優。答案不是繼續堆參數，而是讓參數量和訓練 token 近似同步增長。

這個判斷把「模型大小」從單一崇拜裡拉了出來。一個 2800 億參數但只看 3000 億 token 的模型，可能不如一個 700 億參數但看 1.4 兆 token 的模型。能力不是參數量本身，而是算力預算被參數和資料共同吸收後的結果。

下次看大型模型發布，不要先問參數多大。先問它訓練了多少 token、算力預算如何分配、是否欠訓練。真正的規模化不是把模型吹大，而是讓每一份算力都落到能降低 loss 的地方。

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力

---
title: "《Scaling Laws for Neural Language Models》：規模的數學"
date: "2026-03-01T16:45:39+08:00"
category: "Paper Reading"
description: 規模的數學：為什麼更大的模型可預測地更好，附真實 Python 核心程式碼
tags: [paper-reading, scaling-laws, AI, LLM, python]
pinned: false
---

訓練大型模型首先是一個預算分配問題。參數、資料和算力都要錢；問題不是「越大越好」這種口號，而是下一塊錢該買參數、買資料，還是買更長訓練。

[《Scaling Laws for Neural Language Models》](/papers/2001.08361v1.pdf) 把大型模型訓練從經驗主義推進到預算函數。它沒有發明新架構，卻給了產業一個更硬的判斷工具：在花錢之前，先估計回報曲線。

## 0. 先認幾個詞

如果你平時不常看這種帶公式的論文，先把這幾個詞認熟，後面會順很多：

- `參數量`：模型裡一共有多少可學習參數，也就是模型的大小。
- `資料量`：訓練時一共餵給模型多少文本。
- `算力 / compute`：訓練時總共做了多少計算。你可以先把它當成「電費帳單」。
- `loss / 損失`：模型犯錯有多嚴重，通常越低越好。
- `冪律 / power law`：某個量按固定指數變化的關係；畫在對數座標上，經常會接近一條直線。
- `對數座標`：像 `1、10、100、1000` 這樣按倍數增長的刻度，不是 `1、2、3、4` 那樣均勻加一。

## 1. 要解決什麼問題

到 2020 年初，深度學習社群已經知道更大的模型通常表現更好。但「通常」不是科學。人們回答不了最基本的實際問題：如果我把算力預算翻倍，性能會提升多少？這筆預算該花在更大的模型、更多的資料、還是更長的訓練上？有沒有一個公式？

這篇論文回答了這些問題。不是靠直覺，不是靠經驗法則：靠方程式。

## 2. 冪律：核心發現

論文的核心發現是，語言模型的性能遵循**冪律（power laws）**。在論文測量到的區間內，只要性能主要受某一個因素限制、而不是被另外兩個因素卡住，測試損失（衡量模型預測下一個詞的能力：越低越好）和參數量、資料量、算力在雙對數座標上都近似呈直線關係。

三個方程式總結了整篇論文：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$

$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$

$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

不用被符號嚇到，拆開來看：

- **L** 是測試損失：一個數字就能概括模型表現如何。越低越好
- **N** 是參數數量（模型大小）。參數越多，模型能儲存的模式就越多
- **D** 是訓練用的資料詞元數量。資料越多，可以學到的模式就越多
- **C** 是訓練用的總算力，以 PetaFLOP-days 為單位（1 PetaFLOP-day = 10^15 次浮點運算跑一整天）
- **N_c, D_c, C_c** 是常數（曲線上的參考點）
- **α**（alpha）是指數：它告訴你雙對數圖上直線的斜率。指數越大，代表隨著規模擴大，性能提升越快

關鍵洞察：這些是冪律，不是對數曲線。對數曲線很快就會趨平：輸入翻倍，輸出幾乎不動。冪律則大方得多：至少在論文測量到的區間內，性能沒有出現明顯「撞牆」，而是持續沿著冪律改善。論文也提醒了：這個趨勢不可能無限延伸到零損失，最終一定會變平：但在觀測範圍內，趨勢乾淨俐落。

```python showLanguage
def power_law_loss(x: float, x_c: float, alpha: float) -> float:
    return (x_c / x) ** alpha


def scaling_law_examples() -> dict[str, float]:
    alpha_n = 0.076
    alpha_d = 0.095
    alpha_c = 0.050

    return {
        "10x_params": 10.0 ** alpha_n,
        "10x_data": 10.0 ** alpha_d,
        "10x_compute": 10.0 ** alpha_c,
    }
```

指數本身就說明了問題。資料集大小（α = 0.095）每多一個量級帶來的提升最大。模型大小（α = 0.076）次之。算力（α = 0.050）最小：因為如果不合理分配算力到模型大小和訓練時長上，單純堆算力是浪費的。真正的槓桿在於擴大正確的東西。

## 3. 在論文測試過的範圍內，架構形狀的重要性低於總規模

這是論文最讓人意外的地方。

團隊測試了不同深度（層數）、寬度（隱藏維度）、注意力頭數和前饋維度的 Transformer。在論文測試過的 Transformer 形狀範圍內，只要非嵌入參數量接近，深度和寬度的具體分配對損失影響很小。

一個只有 2 層但隱藏維度巨大的 Transformer？和一個 40 層但隱藏維度很小的 Transformer 損失差不多：前提是非嵌入參數預算接近。

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureExperiment:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int


def non_embedding_params(config: ArchitectureExperiment) -> int:
    n = config.n_layers
    d = config.d_model
    d_ff = config.d_ff
    return n * (4 * d * d + 2 * d * d_ff + 4 * d)
```

這有一個深遠的含義：你不需要花好幾個禮拜去搜尋「最優」架構。選一個合理的 Transformer 形狀，然後把精力集中在把它做大就好。論文之所以把 embedding 參數從 N 中排除，是因為他們發現 embedding 參數對性能的貢獻遠小於非 embedding 參數：模型的「思考」能力在 Transformer 層裡，不在詞表裡。

## 4. 模型何時過擬合：資料瓶頸

更大不一定更好：如果你的資料集太小的話。論文這裡真正漂亮的地方，是給出了一個統一的二維公式，把模型規模和資料規模如何共同決定性能寫進了一個式子：

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
$$

這個公式說的是：損失不是單獨由模型大小或資料大小決定的：而是由兩者共同決定。當 N 大到第一項消失時，剩下的項表明損失被資料卡住了。當 D 大到第二項消失時，剩下的是模型規模的瓶頸。公式在兩種情況之間平滑過渡，過擬合就是兩項競爭的自然結果。

從這個關係出發，論文給出了一個粗略的經驗門檻：過擬合開始明顯影響性能的臨界點：

$$
D \gtrsim 5 \times 10^3 \times N^{0.74}
$$

白話翻譯：模型越大，需要的資料就越多：但增長是次線性的。大 10 倍的模型只需要約 10^0.74 ≈ 5.5 倍的資料。更大的模型樣本效率更高：它們能從每個訓練詞元中提取更多資訊。

```python showLanguage
def loss_nd(n_params: float, n_tokens: float) -> float:
    n_c = 8.8e13
    d_c = 5.4e13
    alpha_n = 0.076
    alpha_d = 0.095
    ratio = alpha_n / alpha_d
    return ((n_c / n_params) ** ratio + d_c / n_tokens) ** alpha_d


def min_dataset_tokens(n_params: float) -> float:
    return 5_000.0 * n_params ** 0.74
```

按這條關係粗略推算，175B 級別模型要把過擬合壓在論文討論的閾值附近，資料量應接近兆級 token。反過來看，GPT-3 的 3000 億詞元其實並不充裕。這也說明「模型多大、資料餵多少」並不是拍腦袋，而是有可分析的權衡：後來業界重新審視這個配比（最典型的就是 Chinchilla 論文，Hoffmann 等人，2022 年），正是因為意識到了許多大模型的資料其實不夠。

## 5. 計算最優訓練：真正的重點

如果你有固定的算力預算，該怎麼花？這是論文中最有實際意義的問題，答案卻違反直覺。

論文發現最優分配遵循：

$$
N_{\mathrm{opt}} \propto C^{0.73}
$$

$$
B_{\mathrm{opt}} \propto C^{0.24}
$$

$$
S_{\mathrm{opt}} \propto C^{0.03}
$$

翻譯一下：如果算力預算增加 10 倍，你應該把模型做大約 5.4 倍，批次大小增加約 1.7 倍，訓練時長幾乎不變（約多 1.07 倍的步數）。

違反直覺的地方在於：**你應該訓練非常大的模型，然後在遠未收斂之前就停下來。** 大多數人的本能是把一個小模型徹底訓練到收斂。縮放定律說的恰好相反：在相同的算力預算下，一個部分訓練的大模型優於一個完全訓練的小模型。

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeAllocation:
    n_params: float
    batch_size: float
    training_steps: float


def optimal_allocation(compute: float) -> ComputeAllocation:
    return ComputeAllocation(
        n_params=compute ** 0.73,
        batch_size=compute ** 0.24,
        training_steps=compute ** 0.03,
    )


def is_compute_efficient(n_params: float, compute: float) -> bool:
    optimal_n = compute ** 0.73
    return abs(n_params / optimal_n - 1.0) < 0.5
```

這個結果塑造了整個產業。五個月後發表的 GPT-3 直接遵循了這個邏輯：訓練一個在當時規模空前的 1750 億參數模型，而不是把一個小模型徹底訓練完。後來的「Chinchilla」論文（Hoffmann et al., 2022）更新了這些指數，並指出大多數大型模型相對於最優資料分配其實是訓練不足的：但核心洞察，即存在一個可計算的最優權衡，源頭在這裡。

## 6. 臨界批次大小：何時該增加平行度

論文還發現批次大小存在一個「甜蜜點」，而且它取決於當前的損失值：

$$
B_{\mathrm{crit}} \propto L^{-4.8}
$$

隨著訓練推進、損失降低，臨界批次大小會增長。訓練初期損失高，小批次就夠用：每個批次提供的梯度信號夠強。後期模型已經學完了簡單的模式，需要更大的批次來平均掉噪聲才能繼續進步。

低於臨界批次大小時，批次翻倍大致能讓訓練時間減半（完美的平行化）。高於臨界批次大小時，批次翻倍幾乎沒有幫助：只是在浪費算力。

```python showLanguage
def critical_batch_size(loss: float, b_star: float, l_star: float) -> float:
    return b_star * (l_star / loss) ** 4.8
```

這是很實用的工程智慧。很多團隊全程使用固定的批次大小。縮放定律告訴你應該隨著訓練推進逐步增大批次：開始用小批次，模型變好後再放大。

## 7. 這篇論文改變了什麼問題

Scaling Laws 的釘子句是：**訓練大型模型不是玄學試錯，而是預算函數。**

這篇論文最重要的貢獻不是某個指數，而是把「更大更好」從經驗判斷變成可估算的曲線。參數、資料、算力不再只是工程變數，而是可以放進同一張帳本裡比較邊際回報的投資項。

它沒有告訴產業永遠只做更大模型。它告訴產業，在給定假設和觀測範圍內，花錢之前應該先算：擴大參數、增加資料、延長訓練，哪一個最可能帶來更低 loss。GPT-3 的規模決策之所以看起來突然變得合理，背後就是這種預算信心。

下次看模型規模，不要問「是不是軍備競賽」。先問預算函數是什麼，瓶頸變數是誰，邊際收益還剩多少。真正成熟的規模化，不是燒更多錢，而是知道錢燒到哪裡會變成能力。

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

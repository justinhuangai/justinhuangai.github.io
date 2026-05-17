---
title: "《Sequence to Sequence Learning with Neural Networks》：編碼器-解碼器範式的起點"
date: "2026-01-24T16:41:08+08:00"
category: "Paper Reading"
description: 編碼器-解碼器範式的確立，附真實 Python 程式碼
tags: [paper-reading, seq2seq, AI, LLM, python]
pinned: false
---

Seq2Seq 的底層矛盾很樸素：輸入和輸出長度都不固定，傳統翻譯流水線能處理，卻很難端到端最佳化。[《Sequence to Sequence Learning with Neural Networks》](/papers/1409.3215v3.pdf) 把問題改寫成兩個介面：一個網路讀完整段輸入，另一個網路逐步生成輸出。

這篇論文的價值不在於把機器翻譯一次解決，而在於證明端到端序列映射可行。它的弱點也同樣清楚：所有資訊都要擠過一個固定長度向量。後來的注意力機制和 Transformer，都是從這個瓶頸往外長出來的。

## 0. 先認幾個詞

如果你沒有機器學習背景，可以先按這篇論文的工作流程，記住下面幾個詞：

- `Seq2Seq / 序列到序列`：把一段輸入序列直接變成另一段輸出序列，比如把英文句子變成法文句子。
- `Encoder / 編碼器`：負責把輸入從頭到尾讀完。
- `Decoder / 解碼器`：負責把輸出一個詞一個詞寫出來。
- `RNN / 循環神經網路`：一種只能按順序處理文本的舊架構。
- `LSTM`：RNN 的改良版，更擅長在長句子裡記住前面的資訊。
- `向量 / vector`：你可以先把它理解成「用一串數字壓縮出來的一份摘要」。

## 1. 要解決什麼問題

2014 年，深度神經網路已經在圖像識別等任務上取得突破，但像機器翻譯這種「直接把一段可變長序列映射到另一段可變長序列」的任務，神經網路還不擅長。

一句英語可能是 5 個詞，翻譯成法語變成 7 個詞。輸入和輸出的長度不同，而且沒有簡單的一一對應關係。

傳統的解決方案是把大量人工設計的規則和統計特徵拼在一起，形成一個複雜的翻譯流水線（統計機器翻譯，SMT）。它能用，但每個元件都要單獨調參，而且很難整體最佳化。

論文提出了一個更簡潔的思路：能不能用一個端到端的神經網路，直接從源語言序列映射到目標語言序列？

## 2. 核心架構：編碼器-解碼器

論文的方法可以用一句話概括：**一個 LSTM 讀，另一個 LSTM 寫。**

LSTM（Long Short-Term Memory，長短期記憶網路）是一種特殊的 RNN，專門設計來處理長距離依賴問題。普通 RNN 在序列很長時容易「遺忘」前面的內容，LSTM 通過引入門控機制（決定哪些資訊保留、哪些丟棄）來緩解這個問題。

具體流程：

1. **編碼器**（一個 4 層深度 LSTM）從頭到尾讀完源句子，把整個句子壓縮成一組固定長度的最終狀態，交給解碼器作為起點
2. **解碼器**（另一個 4 層深度 LSTM）以這個向量為起點，一個詞一個詞地生成目標語言的翻譯，直到輸出結束符號 \<EOS\>

論文給出的機率公式：

$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_t p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

翻譯成人話：給定源句子 x，生成目標句子 y 的機率，等於每一步生成下一個詞的機率連乘起來。每一步的預測都依賴兩樣東西：編碼器壓縮出來的向量 v，以及之前已經生成的所有詞。

```python showLanguage
import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=4, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=4, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def encode(
        self,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(source_tokens)
        outputs, state = self.encoder(embedded)
        return outputs, state

    def decode(
        self,
        encoder_state: tuple[torch.Tensor, torch.Tensor],
        max_steps: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> list[int]:
        prev_token = torch.tensor([[bos_token_id]], dtype=torch.long, device=encoder_state[0].device)
        state = encoder_state
        generated: list[int] = []

        for _ in range(max_steps):
            embedded = self.embedding(prev_token)
            output, state = self.decoder(embedded, state)
            logits = self.output_proj(output[:, -1, :])
            next_token_id = int(logits.argmax(dim=-1).item())
            if next_token_id == eos_token_id:
                break
            generated.append(next_token_id)
            prev_token = torch.tensor([[next_token_id]], dtype=torch.long, device=logits.device)

        return generated
```

架構本身不複雜。論文的貢獻不在於發明了某個新元件，而在於證明了這個簡單的框架真的能用，而且效果好到能和精心調校的傳統系統競爭。

## 3. 三個關鍵設計決策

論文在實驗中發現了三個對性能影響很大的設計選擇：

**第一，用兩個獨立的 LSTM。** 編碼器和解碼器不共享參數。這樣做稍微增加了參數量，但讓模型能更好地分別處理源語言和目標語言的特性。論文提到這也讓同時訓練多個語言對成為可能。

**第二，用深層 LSTM。** 論文用了 4 層 LSTM，每多一層，困惑度降低近 10%。淺層 LSTM（1-2 層）的效果明顯更差。深度給了模型更大的表示空間。

**第三，把源句子倒過來讀。** 這是論文最出人意料的發現。把源句子 "a, b, c" 反轉成 "c, b, a" 再餵給編碼器，BLEU 分數從 25.9 跳到 30.6，提升了將近 5 分。

為什麼反轉有效？論文的解釋是：正常順序下，源句子第一個詞離目標句子第一個詞很遠（中間隔了整個源句子）。反轉之後，源句子的前幾個詞和目標句子的前幾個詞在時間上靠得很近，給梯度（模型用來調整參數的訊號）創造了更多的「短距離依賴」，讓最佳化變得更容易。

```python showLanguage
import torch


def reverse_source(source_tokens: list[int]) -> list[int]:
    return list(reversed(source_tokens))


source_sentence = [11, 23, 37, 42]
reversed_source = reverse_source(source_sentence)
source_tensor = torch.tensor([reversed_source], dtype=torch.long)
```

這個 trick 簡單到幾乎不像是正經的研究貢獻，但它確實有效，而且揭示了一個深層問題：RNN 對序列裡元素之間的距離很敏感，距離越近越容易學。這個問題後來被注意力機制從根本上解決了。

## 4. 實驗結果

論文在 WMT '14 英法翻譯任務上做了實驗。

關鍵數字：
- **單個反轉 LSTM**，beam size 12：30.59 BLEU
- **5 個反轉 LSTM 的集成**，beam size 2：34.50 BLEU
- **5 個反轉 LSTM 的集成**，beam size 12：**34.81 BLEU**
- **傳統短語翻譯系統**（Moses baseline）：33.30 BLEU

在論文報告的實驗設置下，5 個 LSTM 的集成以 34.81 分超過了傳統短語翻譯系統的 33.30 分。考慮到 LSTM 的詞表只有 8 萬詞（遇到詞表外的詞只能輸出 UNK），而傳統系統的詞表幾乎不受限，這個結果很有說服力。

論文還用 LSTM 對傳統系統的 1000-best 候選列表做重排序，BLEU 分數進一步提升到 36.5，接近當時的最佳公開結果（37.0）。

另一個值得注意的發現：相比當時其他神經方法，LSTM 在長句子上的性能退化沒那麼嚴重。這和當時其他研究者報告的長句子性能急劇下降形成了對比，論文將此歸功於反轉源句子的策略。

## 5. 模型的「理解力」

論文還做了一個有趣的視覺化實驗。把不同句子輸入編碼器，取出最終的隱藏狀態向量，用 PCA 降維到二維平面上畫出來。

結果顯示：
- 意思相近的句子在向量空間裡聚在一起
- 主動語態和被動語態的句子（"I gave her a card" vs "I was given a card by her"）落在相近的位置
- 詞序不同但意思相同的句子也能被正確聚類

這至少說明，編碼器學到的表示不只是簡單的詞袋統計（把詞混在一起不管順序），而是包含了相當多的句法和語義資訊。

## 6. 訓練細節

**模型規格**：4 層 LSTM，每層 1000 個單元，詞嵌入維度 1000，總參數量 3.84 億。其中 6400 萬是純循環連接參數。

**硬體**：8 塊 GPU。每層 LSTM 分配一塊 GPU，剩餘 4 塊 GPU 用來並行化 softmax（因為詞表有 8 萬個詞，softmax 計算量很大）。訓練約 10 天。

**最佳化器**：SGD，不帶動量，初始學習率 0.7。訓練 5 個 epoch 之後每半個 epoch 將學習率減半，總共訓練 7.5 個 epoch。

**梯度裁剪**：當梯度的 L2 範數超過閾值 5 時，按比例縮小。這是為了防止梯度爆炸（梯度值突然變得極大，導致參數更新失控）。

**批次最佳化**：把長度相近的句子放在同一個批次裡，避免短句子為長句子「陪跑」浪費計算資源，帶來了 2 倍的訓練加速。

## 7. 這篇論文改變了什麼問題

Seq2Seq 的釘子句是：**端到端映射可行，但固定向量會成為瓶頸。**

這篇論文把機器翻譯從一條人工拼裝的流水線，改成一個可以整體訓練的映射問題。它證明神經網路可以先讀完整段輸入，再逐步寫出輸出；輸入和輸出不需要等長，也不需要手工對齊。

但它同時把瓶頸暴露得很徹底。所有源句資訊都要壓進同一個向量，句子越長，壓縮損失越明顯。反轉源句子的技巧能緩解距離問題，卻不能改變資訊必須過窄門的事實。

所以下次看 Seq2Seq，不要只記住「編碼器-解碼器」。更該問：這個系統把資訊壓在哪裡？那個壓縮點，會不會變成下一代架構必須繞開的瓶頸？

---

**論文共讀系列**

- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學：為什麼更大的模型可預測地更好
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

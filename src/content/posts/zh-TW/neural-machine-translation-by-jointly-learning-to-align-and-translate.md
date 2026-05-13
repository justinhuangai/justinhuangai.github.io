---
title: "論文共讀：《Neural Machine Translation by Jointly Learning to Align and Translate》（通過聯合學習對齊與翻譯實現神經機器翻譯）"
date: "2026-01-11T16:26:19+08:00"
category: "Paper Reading"
description: 注意力機制的起源，附真實 Python 程式碼
tags: [paper-reading, attention, AI, LLM, python]
pinned: false
---

2014 年 9 月 1 日，三個人在 arXiv（一個學術論文預印本網站，論文不用等期刊審稿就能直接發布）上傳了一篇論文：[《Neural Machine Translation by Jointly Learning to Align and Translate》](/papers/1409.0473v7.pdf)（通過聯合學習對齊與翻譯實現神經機器翻譯）。

這三個人是 Dzmitry Bahdanau、KyungHyun Cho 和 Yoshua Bengio，來自蒙特婁大學。Yoshua Bengio 是深度學習「三巨頭」之一，另外兩位是 Geoffrey Hinton 和 Yann LeCun；三人共同獲得了 2018 年圖靈獎。Bahdanau 當時還是博士生。

這篇論文的核心貢獻可以概括成一件事：讓翻譯模型在生成每個詞的時候，學會回頭看源句子的不同部分。聽起來理所當然，但在當時的神經機器翻譯研究裡，這是一個非常新的想法。它有一個名字，叫「注意力機制」。

三年後，Google 的八個人把這個想法推到了極致，寫出了[《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部）。所以如果你想理解 Transformer，這篇論文是最重要的前史之一。

## 0. 先認幾個詞

如果你完全沒有機器學習背景，先順著這篇論文真正想修補的地方，記住下面幾個詞：

- `編碼器-解碼器 / Encoder-Decoder`：一部分先把源句子讀完，另一部分再把目標句子一個詞一個詞寫出來。
- `RNN / 循環神經網路`：當時主流的序列模型。它必須按順序處理文本，不能一下子同時看整句。
- `hidden state / 隱藏狀態`：模型讀到某個位置時，手上那份臨時筆記。
- `alignment / 對齊`：源句子裡的哪一部分，對應目標句子當前要生成的這個詞。
- `attention`：生成每個詞時，不再只盯著一個總壓縮結果，而是主動回頭看源句子裡更 relevant 的位置。

## 1. 問題出在哪

2014 年的神經機器翻譯有一個標準架構：編碼器-解碼器（Encoder-Decoder）。編碼器是一個循環神經網路（RNN），把源句子從頭到尾讀一遍，把整個句子壓縮成一個固定長度的向量（可以理解為一串固定數量的數字）。解碼器是另一個 RNN，從這個向量出發，一個詞一個詞地生成翻譯。

問題很明顯：不管源句子是 5 個詞還是 50 個詞，編碼器都要把它壓進同一個長度的向量裡。短句子還行，長句子就丟資訊。就像你讓一個人讀完一整頁書，然後只能用一句話複述，句子越長，遺漏越多。

論文用實驗證明了這一點：當句子長度超過 30 個詞，傳統編碼器-解碼器的翻譯品質急劇下降。

這就是「固定長度瓶頸」。

## 2. 核心想法：別壓縮，讓解碼器自己去找

論文的解決方案很直覺：既然把整個句子壓成一個向量會丟資訊，那就不壓了。編碼器保留每個位置的註解向量（annotation，由雙向 RNN 的前向和後向隱藏狀態拼接而成，可以理解為每個詞處理完之後產生的中間結果），解碼器在生成每個目標詞時，自己決定該重點看源句子的哪些部分。

這就是注意力機制的核心：**不再強迫所有資訊擠過一個瓶頸，而是讓模型學會在需要的時候回頭找需要的資訊。**

具體來說，分三步：

**第一步，打分。** 解碼器在生成第 i 個目標詞之前，會用自己當前的狀態 s_{i-1} 和編碼器每個位置的隱藏狀態 h_j 做比較，算出一個「對齊分數」e_{ij}。分數越高，說明生成當前目標詞時，源句子的第 j 個位置越重要。

論文用的打分函數是：

$$
e_{ij} = a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j)
$$

這叫「加性注意力」（additive attention）。把解碼器狀態和編碼器狀態各自做一次線性變換（乘以矩陣），加起來，過一個 tanh（一種把數值壓縮到 -1 到 1 之間的函數），再和一個向量 v_a 做點積，得到一個標量分數。

**第二步，歸一化。** 用 softmax 把所有位置的分數轉成機率，加起來等於 1：

$$
\alpha_{ij} = \operatorname{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

**第三步，加權求和。** 用這些機率對編碼器的隱藏狀態做加權求和，得到一個「上下文向量」c_i：

$$
c_i = \sum_j \alpha_{ij} h_j
$$

這個上下文向量就是解碼器在生成第 i 個詞時，從源句子裡提取到的關鍵資訊。每生成一個詞，上下文向量都不一樣，因為模型關注的源句子位置不一樣。

用 Python（基於 PyTorch）寫出來：

```python showLanguage
import torch
from torch import nn


def bahdanau_attention(
    decoder_state: torch.Tensor,
    encoder_outputs: torch.Tensor,
    w_a: nn.Linear,
    u_a: nn.Linear,
    v_a: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_features = w_a(decoder_state).unsqueeze(1)
    encoder_features = u_a(encoder_outputs)
    scores = v_a(torch.tanh(decoder_features + encoder_features)).squeeze(-1)
    weights = torch.softmax(scores, dim=-1)
    context = torch.sum(weights.unsqueeze(-1) * encoder_outputs, dim=1)
    return context, weights
```

和後來 Transformer 用的「點積注意力」（Q 和 K 直接做點積）不同，這篇論文用的是「加性注意力」（先各自做線性變換，再加起來）。兩種方法各有特點，但點積注意力更適合用高效矩陣乘法實現；再加上 Transformer 去掉了 RNN 的順序依賴，注意力才真正成為可大規模並行的核心算子。

## 3. 編碼器：雙向 RNN

單向 RNN 從左往右讀句子，到了最後一個詞才輸出一個總結向量。問題是：每個位置的隱藏狀態主要只帶著左側上下文，看不到右邊。

論文用了雙向 RNN（BiRNN）來解決這個問題。一個 RNN 從左往右讀，另一個從右往左讀，然後把兩個方向的隱藏狀態拼起來。這樣每個位置的隱藏狀態就同時包含了左邊和右邊的上下文。

```python showLanguage
import torch
from torch import nn


class BidirectionalRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(inputs)
        return outputs
```

論文裡每個方向各有 1000 個隱藏單元，拼起來就是 2000 維。這比單向 RNN 多了一倍參數，但換來的是每個位置都能看到完整的上下文。

## 4. 解碼器：每一步都重新對齊

把編碼器和注意力機制組裝起來，解碼器的工作流程就清楚了：

1. 編碼器用雙向 RNN 讀完源句子，保留每個位置的隱藏狀態（註解向量）
2. 解碼器開始生成翻譯，每生成一個詞之前：
   - 用當前狀態和所有註解向量算注意力權重
   - 加權求和得到上下文向量
   - 結合上下文向量、上一個生成的詞和當前狀態，預測下一個詞

```python showLanguage
import torch
from torch import nn


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=embedding_dim + 2 * hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.w_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def decode_step(
        self,
        prev_word: torch.Tensor,
        prev_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, _ = bahdanau_attention(
            prev_state.squeeze(0),
            encoder_outputs,
            self.w_a,
            self.u_a,
            self.v_a,
        )
        rnn_input = torch.cat([prev_word, context.unsqueeze(1)], dim=-1)
        output, new_state = self.rnn(rnn_input, prev_state)
        logits = self.output_proj(output[:, -1, :])
        return logits, new_state
```

關鍵在於：每生成一個目標詞，解碼器都會重新計算注意力分布。翻譯第一個詞時可能重點關注源句子的開頭，翻譯最後一個詞時可能重點關注源句子的結尾。這種動態對齊能力，是之前的固定向量架構做不到的。

## 5. 實驗結果

論文在英法翻譯任務上做了實驗（使用 WMT '14 資料集），用 BLEU 分數（衡量機器翻譯和人工翻譯接近程度的標準評分，滿分 100）衡量效果。

關鍵對比：
- **RNNencdec-50**（傳統編碼器-解碼器，訓練時最長 50 詞）：26.71 BLEU
- **RNNsearch-50**（帶注意力的模型，訓練時最長 50 詞）：**34.16 BLEU**
- **Moses**（當時最強的傳統短語翻譯系統）：33.30 BLEU

提升了 7.45 分。在論文報告的實驗設置裡，帶注意力的神經模型已經達到甚至超過了當時強勢的傳統短語翻譯系統。

更關鍵的發現在論文的 Figure 2：隨著句子長度增加，傳統編碼器-解碼器的 BLEU 分數急速下跌，而帶注意力的模型幾乎不受影響。這直接驗證了論文的核心假設：固定長度向量是瓶頸，注意力機制可以繞過它。

論文還展示了注意力權重的視覺化。在英法翻譯裡，注意力權重幾乎形成了一條對角線，說明模型自動學會了「英語第 1 個詞對應法語第 1 個詞，英語第 2 個詞對應法語第 2 個詞」的對齊關係。遇到語序不同的情況（比如法語的形容詞放在名詞後面），注意力權重也會對應地偏移。模型不需要任何人工對齊標註，就學會了這些。

## 6. 我的思考

讀完這篇論文，有幾個感受。

第一，這篇論文解決的問題極其明確：編碼器把整個句子壓成一個向量，長句子丟資訊。解決方案也極其直覺：別壓了，讓解碼器自己去找。好的研究往往就是這樣，問題清晰，解法自然。

第二，注意力機制在這篇論文裡還是 RNN 的配角。編碼器仍然是循環的（雙向 RNN），解碼器也是循環的，注意力只是在兩者之間架了一座橋。三年後 Vaswani 等人問了一個更激進的問題：既然注意力這麼好用，能不能把 RNN 整個扔掉，只留注意力？答案就是 Transformer。

第三，用真實 Python 重寫這篇論文的注意力機制時，你會發現它的計算流程比 Transformer 的 Scaled Dot-Product Attention 複雜不少。加性注意力需要額外的權重矩陣 W_a、U_a、v_a，而點積注意力只需要 Q 和 K 直接相乘再縮放。從「加法」到「乘法」，看似一小步，實際上大幅簡化了計算，也更適合用矩陣運算高效實現。

第四，Bahdanau 當時是博士生，Bengio 是他的導師。一個博士生的論文，定義了此後十年 AI 研究的核心元件。注意力機制從這裡開始，經過 Transformer 的放大，最終成為 GPT、BERT、LLaMA 的基石。

這篇論文沒有發明什麼複雜的數學。它只是問了一個簡單的問題：解碼器為什麼不能回頭看？

然後它讓解碼器回頭看了。

這一看，看出了整個時代。

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學：為什麼更大的模型可預測地更好
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

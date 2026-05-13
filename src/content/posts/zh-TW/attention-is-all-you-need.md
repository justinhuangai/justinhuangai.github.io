---
title: "《Attention Is All You Need》：Transformer 的設計原點"
date: "2026-01-06T16:18:46+08:00"
category: "Paper Reading"
description: 分享我對 Transformer 論文的理解，附真實 Python 程式碼
tags: [paper-reading, transformer, AI, LLM, python]
pinned: false
---

2017 年 6 月 12 日，八個人在 arXiv（一個學術論文預印本網站，論文不用等期刊審稿就能直接發布）上傳了一篇論文，標題只有五個詞：[《Attention Is All You Need》](/papers/1706.03762v7.pdf)（注意力就是你所需要的全部）。

這八個人是 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser 和 Illia Polosukhin，當時大多在 Google Brain 和 Google Research 工作。

論文發出之後，這個八人組幾乎全部散開。Noam Shazeer 離開 Google 創立了 Character.AI，後來又被 Google 高價請回；Aidan Gomez 從多倫多大學博士還沒畢業就創立了 Cohere，做企業級大模型；Llion Jones 去了日本，創立了 Sakana AI；Illia Polosukhin 走了一條誰都沒想到的路，創立了 NEAR Protocol，做區塊鏈；Ashish Vaswani 和 Niki Parmar 搭檔創立了 Adept AI，後來又一起創立了 Essential AI；Jakob Uszkoreit 創立了 Inceptive，用 AI 設計 RNA 藥物；Łukasz Kaiser 則加入了 OpenAI，參與了 GPT 系列的研發。

八位作者，七家公司，橫跨 AI、區塊鏈、生物技術。

近九年後的今天，ChatGPT、Claude、DeepSeek、Qwen，這些 AI 產品的底層架構思路，大多都能追溯到這 15 頁紙。

這篇文章是我讀完論文後的理解，附帶真實 Python 程式碼示例。不是翻譯，不是摘要。沒有技術背景也能讀下去。

## 0. 先認幾個詞

如果你沒有機器學習背景，先順著這篇論文真正想取代的舊方案，記住下面幾個詞就夠了：

- `RNN / 循環神經網路`：一種更早的序列模型。它處理句子時必須一個詞一個詞往後讀，像人用手指著文章逐行看。
- `attention`：從很多資訊裡，挑出當前最該看的那幾部分。你可以先把它理解成「有選擇地回頭看重點」。
- `Query / Key / Value`：注意力機制裡的三個角色。Query 像「我現在想找什麼」，Key 像「每段資訊貼著什麼標籤」，Value 則是「真正被取回來的內容」。
- `Transformer`：以 attention 為核心搭起來的一整套架構。它不靠循環一步步往前推，而是讓每個位置都能直接看其他位置。
- `並行`：這裡不是說模型更聰明，而是說它能同時處理很多位置，不必像 RNN 那樣排隊。

## 1. 一句話說清楚

在 Transformer 之前，AI 處理語言的方式像是一個人用手指著書，一個字一個字地往下讀。讀到第 100 個字的時候，第 1 個字說了什麼，已經記不太清了。句子越長，遺忘越嚴重。這就是循環神經網路（RNN，一種早期的 AI 架構）的根本瓶頸。

論文的作者們問了一個問題：**為什麼一定要按順序讀？**

和必須逐步處理的 RNN 不同，Transformer 可以並行處理整段輸入，直接建模任意兩個位置之間的關係。不用排隊，不用等前一個詞處理完才能看下一個。

論文管這個核心能力叫「注意力」。標題要表達的不是「模型裡真的只剩注意力」，而是：在序列建模裡，注意力第一次被推到了主角的位置，不再需要循環和卷積（一種通過滑動窗口提取局部特徵的方法）作為骨架。

## 2. 注意力到底在做什麼

想像你走進一個嘈雜的酒吧，二十個人同時在說話。你的大腦不會平均分配注意力給每個人。有人喊了你的名字，你的耳朵瞬間鎖定那個方向，其他聲音自動變成背景噪音。

Transformer 對每個詞做同樣的事。論文裡定義了三個角色：

- **Query（查詢）**：這個詞在找什麼資訊。相當於你的耳朵在搜尋「誰在叫我」
- **Key（鍵）**：這個詞能提供什麼資訊。相當於酒吧裡每個人的聲音特徵
- **Value（值）**：這個詞實際攜帶的內容。相當於那個人說的具體話

每個詞的 Query 會和其他詞的 Key 做匹配。匹配度高的，就從對方的 Value 裡獲取更多資訊。匹配度低的，直接忽略。

論文給出的公式叫 Scaled Dot-Product Attention：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

看到公式別慌。一步一步拆：

- **QK^T**：Q 和 K 做點積。什麼是點積？把兩組數字對應位置相乘，再加起來。比如 [1, 2] 和 [3, 4] 的點積是 1×3 + 2×4 = 11。數字越大，說明兩個詞越相關。這一步算的就是每對詞之間的「匹配分數」
- **/ √d_k**：除以一個數來縮放。d_k 是向量的長度（向量可以理解為「一串用來描述某個東西的數字」，比如用 64 個數字描述一個詞的含義）。為什麼要除？因為數字串越長，點積結果越大。不縮放時，維度越大點積的方差越大，softmax 容易進入飽和區（幾乎所有機率集中在一個詞上），梯度（模型用來調整自身參數的訊號）變得很小，訓練會不穩定
- **softmax**：把一組分數轉換成機率，所有機率加起來等於 1。比如三個詞的分數是 [10, 2, 1]，softmax 之後大概變成 [0.99, 0.007, 0.003]。分數最高的那個詞幾乎拿走了全部注意力，其他的被壓到接近零
- **× V**：用這些機率去加權每個詞的實際內容。機率高的詞貢獻大，機率低的詞貢獻小。最終輸出是一個融合了關鍵資訊的新向量

用 Python（基於 PyTorch）寫出來：

```python showLanguage
import math
import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    d_k = key.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ value
```

就這麼幾行程式碼。很多後來改變行業的能力，底層都建立在這幾行運算之上。

## 3. 多頭注意力：同時從多個角度看

單個注意力頭通常只能偏向某一類關係模式。但語言這東西，一句話裡藏著好幾層意思。

拿「我昨天在深圳吃了潮汕牛肉火鍋」來說：
- 「我」和「吃了」之間是誰做了什麼的關係
- 「昨天」和「吃了」之間是時間關係
- 「深圳」和「潮汕牛肉火鍋」之間是地點與食物的關係

讓一個頭同時兼顧這麼多層次，很難。論文的做法是用多頭機制：派出 8 個頭並行運算，讓模型有機會從不同子空間同時觀察一句話，最後把各自的發現拼起來。

論文原文的公式：

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

拆開看：
- **head_1, ..., head_h**：8 個頭各自獨立做一次注意力運算，得到 8 份結果
- **Concat**：把 8 份結果首尾相連，拼成一個長向量
- **W^O**：一次線性變換（可以理解為「乘以一個矩陣」），把拼接後的長向量壓回原來的維度。相當於一個主管聽完 8 個調查員的匯報，輸出一份綜合結論

```python showLanguage
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = torch.softmax(scores, dim=-1)
        heads = weights @ v

        batch_size, _, target_len, _ = heads.shape
        merged = heads.transpose(1, 2).contiguous()
        merged = merged.view(batch_size, target_len, self.num_heads * self.d_head)
        return self.out_proj(merged)
```

論文裡的參數：模型用 512 個數字描述一個詞（d_model = 512），8 個頭，每個頭分到 64 個數字（512 ÷ 8 = 64）。8 個頭的總計算量和 1 個 512 維的頭差不多，但表達能力強得多。用同樣的代價，換來多視角的理解力。這筆帳算得漂亮。

## 4. 位置編碼：告訴模型詞的順序

Transformer 並行處理整個句子，速度是快了，但代價是它丟掉了詞的先後順序。如果沒有額外的位置資訊，注意力機制本身並不知道「貓吃魚」和「魚吃貓」有什麼區別。這顯然不行。

怎麼補救？給每個位置生成一個獨一無二的「地址編碼」，加到詞的向量上。模型看到的不再是「貓」和「魚」，而是「第 1 個位置的貓」和「第 3 個位置的魚」。

論文用正弦和餘弦函數來生成這個編碼：

$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
\operatorname{PE}(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

公式看著唬人，核心思路很直觀：
- **pos**：詞在句子裡的位置（第 1 個、第 2 個、第 3 個……）
- **i**：向量的第幾個維度。偶數位置用 sin，奇數位置用 cos
- **10000^(2i/d_model)**：一個隨維度變化的縮放因子。低維度變化快，高維度變化慢。就像時鐘：秒針一分鐘轉一圈，時針十二小時才轉一圈。不同「指針」覆蓋不同的時間尺度，組合在一起就能精確定位任意時刻

最終效果：每個位置得到一串獨一無二的數字指紋，模型靠這個指紋區分詞的先後順序。

```python showLanguage
import math
import torch


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    encoding = torch.zeros(seq_len, d_model)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding
```

為什麼偏偏選正弦餘弦？因為它有一個優雅的數學性質：兩個詞相隔固定距離，無論它們出現在句首還是句尾，位置編碼之間的關係是一樣的。模型不用死記「位置 3 和位置 8」的關係，只需要學會「相隔 5 個位置」意味著什麼。論文團隊也試過讓模型自己學位置編碼，效果差不多，但正弦版本有一個額外的優勢：它能處理訓練時沒見過的更長句子。

## 5. 編碼器與解碼器

Transformer 的完整架構分兩半。

**編碼器**（6 層堆疊）負責讀懂輸入。每層包含兩個子層：一個多頭自注意力，一個前饋網路。每個子層都有兩個保護機制：

- **殘差連接**：把子層的輸入直接加到輸出上，即 x + Sublayer(x)。為什麼？想像你給一張照片加濾鏡。如果濾鏡效果不好，殘差連接保證你還能看到原圖。在深層網路裡，資訊每經過一層都會被變換，傳到第六層可能已經面目全非。殘差連接讓原始訊號可以「抄近道」直達深層，防止資訊在傳遞中丟失
- **層歸一化**（LayerNorm）：把數值調整到統一範圍，防止有的數字大到爆炸、有的小到消失。類似於考試成績標準化，不管原始卷面分差異多大，標準化後都在一個可比較的區間

**解碼器**（6 層堆疊）負責生成輸出。結構和編碼器類似，但多了兩個關鍵設計：

第一，**交叉注意力**：解碼器生成每個詞時，會回頭「看」編碼器的輸出。翻譯場景下，就是一邊寫英文一邊回頭看中文原文。

第二，**遮罩**（masking）：生成第 3 個詞時，只允許看到前 2 個詞，第 4 個及之後的位置被遮蔽（注意力分數設為負無窮，經過 softmax 後變成零）。道理很簡單：你寫作文的時候，下一個字還沒寫出來，不能偷看。

```python showLanguage
from typing import Optional

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder(self.embedding(src_token_ids))
        hidden = self.decoder(self.embedding(tgt_token_ids), memory, tgt_mask=tgt_mask)
        return self.output_proj(hidden)
```

還有一個容易被忽略的元件：前饋網路。公式是 FFN(x) = max(0, xW1 + b1)W2 + b2。翻譯成白話：先把每個詞的 512 維向量擴大到 2048 維（乘以一個矩陣再加一個偏置），用 ReLU 過濾一遍（所有負數變成零，正數保留），再壓回 512 維。ReLU 這一步是關鍵：它引入了「非線性」，讓模型能學到直線畫不出來的複雜模式。如果全是線性變換，多層疊加在數學上仍可合併為單層，非線性是模型表示複雜模式的前提。

## 6. 訓練細節

架構設計完了，怎麼訓練它？論文在這裡也有不少講究。

**硬體**：8 塊 NVIDIA P100 GPU。基礎版模型訓練 12 小時（10 萬步），大號模型訓練 3.5 天（30 萬步）。放在今天看，這個成本低得驚人。

**最佳化器**：用的是 Adam（一種讓模型自動調整參數的演算法），但學習率的設計很巧妙。學習率決定了模型每一步「邁多大步子」。步子太大容易跨過最佳解，太小又走得慢。論文的策略是前 4000 步逐漸提速（warmup，熱身），避免一開始更新過猛；4000 步之後按計畫逐漸減速，讓訓練後期更穩定。先升後降，前半段大膽探索，後半段精細打磨。

**正則化**：兩招。第一招是 Dropout，訓練時隨機關掉 10% 的神經元（可以理解為網路中的計算節點），迫使模型不依賴任何單一路徑，學到更穩健的特徵。第二招是 label smoothing（標籤平滑，ε = 0.1）：訓練時不告訴模型「正確答案的機率是 100%」，而是說「90% 是正確答案，剩下 10% 分給其他選項」。這會讓模型在一個指標上變差（困惑度，衡量模型有多「拿不準」），但翻譯品質反而更好。直覺上說，一個承認自己不是 100% 確定的模型，比一個過度自信的模型更可靠。

**結果**：論文用 BLEU 分數（機器翻譯的標準評分，衡量機器翻譯和人工翻譯有多接近，滿分 100）來衡量效果。英德翻譯 28.4 分，英法翻譯 41.8 分，都刷新了當時的紀錄。訓練成本比之前的方法低了一到兩個數量級。更快，更強，更便宜。

## 7. 我的思考

讀完這篇論文，有幾個感受。

第一，這篇論文的核心洞察極其簡潔：扔掉順序處理的包袱，讓注意力機制直接建模任意兩個位置之間的關係。Self-Attention、殘差連接、Layer Normalization，沒有一個是新發明。真正的突破不在於發明新工具，而在於作者們敢賭「這些簡單的積木拼在一起就夠了」，然後用實驗證明了自己是對的。

第二，用真實 Python 程式碼寫出來的過程讓我更深地理解了每一個設計決策。當你自己寫出 Scaled Dot-Product Attention，你會切實感受到那個 √d_k 的縮放有多重要。當你實現 masking，你會理解自回歸生成的約束從何而來。論文讀十遍，不如自己寫一遍。

第三，真正讓我震撼的，不是它後來衍生出了多少模型，而是它當年就把問題改寫了：從「怎麼按順序記住一句話」，變成「怎麼讓每個位置直接找到它最該看的資訊」。GPT、BERT、T5、LLaMA，全是這個問題改寫之後的產物。

一個足夠好的架構，能走多遠，取決於有多少人願意在它上面繼續建設。

這篇論文給出了那個架構。

《Attention Is All You Need》（注意力就是你所需要的全部）。

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學：為什麼更大的模型可預測地更好
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

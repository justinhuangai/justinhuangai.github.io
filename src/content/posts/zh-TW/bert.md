---
title: "《BERT》：語言理解預訓練範式的確立"
date: "2026-01-31T16:52:21+08:00"
category: "Paper Reading"
description: 預訓練範式的確立，附真實 Python 程式碼
tags: [paper-reading, bert, AI, LLM, python]
pinned: false
---

BERT 要解決的不是「再造一個更深的 Transformer」，而是 NLP 任務長期被拆碎的問題。問答、分類、序列標註各有一套模型和資料介面，模型學到的語言知識很難被統一複用。

[《BERT》](/papers/1810.04805v2.pdf) 的關鍵不只是雙向 Transformer，而是把語言理解任務統一成預訓練表示的複用問題。先學一套通用表示，再用很薄的任務層去適配不同問題；這個轉向比模型結構本身更重要。

## 0. 先認幾個詞

如果你對大模型的訓練流程還不熟，可以先記住這篇論文最關鍵的幾個詞：

- `Transformer`：BERT 用的基礎架構。你可以先把它理解成一台能同時結合左右上下文來處理句子的機器。
- `預訓練`：先在海量通用文本上學語言本身，而不是一開始就做具體任務。
- `微調`：把預訓練得到的能力，再稍微調整到某個具體任務上。
- `雙向`：預測一個位置時，不只看左邊，也看右邊。
- `MLM / 遮蔽語言模型`：故意遮住一部分詞，讓模型根據上下文把它們猜出來。
- `NSP / 下一句預測`：讓模型判斷兩個句子是不是前後相連。

## 1. 要解決什麼問題

2018 年，NLP 領域有一個尷尬的現狀：每個任務都需要從頭設計專門的模型架構。做問答要一套模型，做情感分析要另一套，做命名實體識別又要一套。每個任務的標註資料都不多，訓練出來的模型也很難遷移到其他任務。

當時已經有人嘗試過預訓練的思路。ELMo 用雙向 LSTM 學習上下文表示，但它只是把預訓練的特徵「拼」到下游模型上，架構本身還是任務專用的。OpenAI GPT 用 Transformer 做預訓練再微調，但它只能從左往右看（單向），每個詞只能關注它前面的詞，看不到後面的。

論文認為，單向語言模型在需要深度雙向上下文的語言理解任務上存在明顯限制。比如：

> "他拿起了 _____ ，開始演奏。"

如果只看左邊（"他拿起了"），填空的答案可能是任何東西。但看到右邊（"開始演奏"），你立刻知道是某種樂器。對很多語言理解任務來說，雙向上下文天然更有利。

## 2. 核心想法：遮住一些詞，讓模型猜

BERT 的解法很直覺：既然雙向語言模型沒法用傳統方式訓練（因為每個詞會間接「看到自己」），那就換個訓練目標。

**遮蔽語言模型（Masked Language Model，MLM）**：隨機遮住輸入中 15% 的詞：具體做法是把它們替換成一個特殊標記 \[MASK\]：然後讓模型根據上下文猜出被遮住的詞。這個想法來自心理學中的完形填空（Cloze task，1953 年 Taylor 提出），就像上面那道填空題一樣。

遮住之後，模型必須同時利用左邊和右邊的上下文來預測，雙向理解就自然產生了。

但直接把所有被選中的詞替換成 \[MASK\] 標記會引入一個問題：微調時輸入裡不會出現 \[MASK\]，預訓練和微調之間產生了不匹配。論文的解決方案：被選中的 15% 的詞裡，80% 替換成 \[MASK\]，10% 替換成隨機詞，10% 保持不變。這樣模型不能只靠「看到 \[MASK\] 就知道要預測」，而是必須對每個位置都保持理解能力。

```python showLanguage
import random
from typing import Optional, Sequence


def mask_tokens(
    tokens: Sequence[str],
    mask_prob: float = 0.15,
    vocab: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[int], list[str]]:
    if vocab is None:
        vocab = tokens

    masked = list(tokens)
    positions: list[int] = []
    labels: list[str] = []

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            positions.append(i)
            labels.append(token)

            r = random.random()
            if r < 0.8:
                masked[i] = "[MASK]"
            elif r < 0.9:
                masked[i] = random.choice(vocab)

    return masked, positions, labels
```

## 3. 第二個預訓練任務：下一句預測

很多 NLP 任務（比如問答、自然語言推理）需要理解兩個句子之間的關係，但語言模型本身不直接建模這種關係。

論文加了第二個預訓練任務：**下一句預測（Next Sentence Prediction，NSP）**。給模型兩個句子 A 和 B，50% 的情況下 B 是 A 的真實下一句，50% 的情況下 B 是從語料庫裡隨機抽的。模型要判斷 B 是不是 A 的下一句。

這個任務的設計很簡單，但論文的消融實驗（ablation study，逐一去掉某個元件看效果變化）顯示，去掉 NSP 會明顯降低問答和自然語言推理任務的表現；不過後來也有工作（如 RoBERTa）對 NSP 的必要性提出了不同結論。

```python showLanguage
from dataclasses import dataclass


@dataclass
class PretrainingExample:
    tokens: list[str]
    segment_ids: list[int]
    masked_positions: list[int]
    masked_labels: list[str]
    is_next: bool
```

## 4. 模型架構

BERT 的架構其實沒有什麼新發明。它就是 [Transformer](/zh-TW/posts/attention-is-all-you-need/) 的編碼器部分，一層層堆起來。

論文給出了兩個規格：

- **BERT_BASE**：12 層，隱藏維度 768，12 個注意力頭，參數量 1.1 億
- **BERT_LARGE**：24 層，隱藏維度 1024，16 個注意力頭，參數量 3.4 億

BERT_BASE 的參數量和 OpenAI GPT 差不多，方便直接對比。兩者最關鍵的區別只有一個：GPT 用的是單向注意力（每個詞只能看左邊），BERT 用的是雙向注意力（每個詞能看到所有位置）。

輸入的表示由三部分相加構成：

- **詞嵌入（Token Embedding）**：WordPiece 分詞，詞表 30,000
- **段嵌入（Segment Embedding）**：標記這個詞屬於句子 A 還是句子 B
- **位置嵌入（Position Embedding）**：告訴模型詞的位置（BERT 用的是學習得到的位置編碼，不是正弦餘弦）

每個輸入序列的開頭都加一個特殊標記 \[CLS\]，它在最後一層的隱藏狀態被用來做句子級別的分類（比如 NSP、情感分析）。兩個句子之間用 \[SEP\] 分隔。

```python showLanguage
import torch
from torch import nn


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_positions: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        position_ids = position_ids.expand_as(token_ids)
        embeddings = (
            self.token_embedding(token_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(position_ids)
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        max_positions: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_positions, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embeddings(token_ids, segment_ids)
        return self.encoder(hidden)
```

## 5. 微調：一個模型適配所有任務

BERT 最優雅的地方在於微調的簡單性。預訓練完成後，不管什麼下游任務，做法幾乎一樣：在 BERT 上面加一層任務相關的輸出層，然後用少量標註資料微調所有參數。

- **文本分類**（情感分析、自然語言推理）：取 \[CLS\] 位置的輸出向量，接一個線性分類器
- **問答**（給一段文章，找出答案的起止位置）：對每個詞的輸出向量做兩次線性變換，分別預測答案的開始和結束位置
- **序列標註**（命名實體識別）：對每個詞的輸出向量接一個分類器，逐詞預測標籤

預訓練可能需要幾天，但微調通常只要幾十分鐘到幾小時（單塊 TPU 上大部分任務不超過 1 小時）。這個效率差異是「預訓練 + 微調」範式的核心吸引力。

## 6. 實驗結果

論文在 11 個 NLP 任務上做了實驗，全部刷新了當時的紀錄。

**GLUE 基準**（通用語言理解評估，包含 8 個子任務）：
- BERT_LARGE 平均分 80.5%，比之前最好的 OpenAI GPT 高出 7.7 個百分點
- 在最大的子任務 MNLI 上提升了 4.6%

**SQuAD v1.1**（閱讀理解問答，Test F1）：
- BERT_LARGE 單模型 + TriviaQA 資料：F1 91.8，超過人類表現（91.2）
- BERT_LARGE 集成模型 + TriviaQA 資料：F1 93.2

**SQuAD v2.0**（包含無法回答的問題）：
- F1 達到 83.1，比之前最好的系統高出 5.1 個百分點

**SWAG**（常識推理）：
- 準確率 86.3%，比 OpenAI GPT 高出 8.3 個百分點

論文還做了模型大小的消融實驗，發現一個重要結論：更大的模型在所有任務上都更好，即使在標註資料很少（只有 3,600 條）的任務上也是如此。這和當時的直覺（小資料集容易過擬合大模型）不太一樣，說明預訓練提供的知識可以有效緩解小資料集的過擬合（模型把訓練資料「死記硬背」，對新資料表現差）風險。

## 7. 訓練細節

**預訓練資料**：BooksCorpus（8 億詞）+ 英文 Wikipedia（25 億詞），只使用文本段落，去掉了列表、表格和標題。論文強調必須用文件級語料而不是打亂的句子級語料，這樣才能提取長距離的上下文關係。

**分詞**：WordPiece，詞表大小 30,000。WordPiece 會把不常見的詞拆成更小的子詞單元，比如 "playing" 可能被拆成 "play" + "##ing"。

**最佳化器**：Adam，學習率 1e-4，前 10,000 步線性熱身，然後線性衰減。批次大小 256 個序列，最大序列長度 512。

**硬體**：BERT_BASE 在 4 塊 Cloud TPU（16 塊 TPU 晶片）上訓練 4 天。BERT_LARGE 在 16 塊 Cloud TPU（64 塊 TPU 晶片）上訓練 4 天。

**Dropout**：所有層的 dropout 率為 0.1。激活函數用的是 GELU（Gaussian Error Linear Unit），而不是 Transformer 原版的 ReLU。

## 8. 這篇論文改變了什麼問題

BERT 的釘子句是：**語言理解可以先學成一種可複用表示，再分發給具體任務。**

它的關鍵不只是 Transformer 編碼器，也不只是「雙向」。真正的動作是用 MLM 和 NSP 把無標註文本變成預訓練信號，再讓分類、問答、序列標註這些任務共享同一套輸入格式和表示底座。

這改變了 NLP 的工程單位。過去每個任務都像一個單獨專案；BERT 之後，許多任務變成同一份預訓練表示上的薄適配層。模型不再從每個任務的標註資料裡重新學語言，而是先擁有一套通用語言底座，再學任務邊界。

下次看一個理解模型，不要只問它是不是雙向。更該問：它學到的表示能否被複用？任務差異是被塞進模型主體，還是被壓縮到一個輕量介面裡？

---

**論文共讀系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學：為什麼更大的模型可預測地更好
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 如何最有效地分配算力

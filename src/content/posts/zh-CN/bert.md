---
title: "论文共读：《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》（BERT：用于语言理解的深度双向 Transformer 预训练）"
date: "2026-01-31T16:52:21+08:00"
category: "Paper Reading"
description: 预训练范式的确立，附真实 Python 代码
tags: [paper-reading, bert, AI, LLM, python]
pinned: false
---

2018 年 10 月 11 日，Google AI Language 团队在 arXiv（一个学术论文预印本网站，论文不用等期刊审稿就能直接发布）上传了一篇论文：[《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/papers/1810.04805v2.pdf)（BERT：用于语言理解的深度双向 Transformer 预训练）。

作者是 Jacob Devlin、Ming-Wei Chang、Kenton Lee 和 Kristina Toutanova，四人均来自 Google。Devlin 此前在微软研究院工作，加入 Google 后主导了 BERT 的设计和实现。

BERT 的全名是 Bidirectional Encoder Representations from Transformers：来自 Transformer 的双向编码器表示。它做了一件在当时看来非常大胆的事：先在海量无标注文本上做通用预训练，然后只需要加一层输出层、在具体任务上做少量微调，就能拿到最优结果。

这个「先预训练、再微调」的范式，后来成为了整个 NLP 领域的标准做法。GPT 系列也用了类似的思路，但走的是另一条路：单向生成。BERT 选择了双向理解。两条路后来各自发展出了庞大的模型家族。

## 0. 先认几个词

如果你对大模型训练流程还不熟，可以先记住这篇论文最关键的几个词：

- `Transformer`：BERT 用的基础架构。你可以先把它理解成一台能同时结合左右上下文来处理句子的机器。
- `预训练`：先在海量通用文本上学语言本身，而不是一上来就做具体任务。
- `微调`：把预训练得到的能力，再稍微调整到某个具体任务上。
- `双向`：预测一个位置时，不只看左边，也看右边。
- `MLM / 遮蔽语言模型`：故意遮住一部分词，让模型根据上下文把它们猜出来。
- `NSP / 下一句预测`：让模型判断两个句子是不是前后相连。

## 1. 要解决什么问题

2018 年，NLP 领域有一个尴尬的现状：每个任务都需要从头设计专门的模型架构。做问答要一套模型，做情感分析要另一套，做命名实体识别又要一套。每个任务的标注数据都不多，训练出来的模型也很难迁移到其他任务。

当时已经有人尝试过预训练的思路。ELMo 用双向 LSTM 学习上下文表示，但它只是把预训练的特征「拼」到下游模型上，架构本身还是任务专用的。OpenAI GPT 用 Transformer 做预训练再微调，但它只能从左往右看（单向），每个词只能关注它前面的词，看不到后面的。

论文认为，单向语言模型在需要深度双向上下文的语言理解任务上存在明显限制。比如：

> "他拿起了 _____ ，开始演奏。"

如果只看左边（"他拿起了"），填空的答案可能是任何东西。但看到右边（"开始演奏"），你立刻知道是某种乐器。对很多语言理解任务来说，双向上下文天然更有利。

## 2. 核心想法：遮住一些词，让模型猜

BERT 的解法很直觉：既然双向语言模型没法用传统方式训练（因为每个词会间接「看到自己」），那就换个训练目标。

**遮蔽语言模型（Masked Language Model，MLM）**：随机遮住输入中 15% 的词：具体做法是把它们替换成一个特殊标记 \[MASK\]：然后让模型根据上下文猜出被遮住的词。这个想法来自心理学中的完形填空（Cloze task，1953 年 Taylor 提出），就像上面那道填空题一样。

遮住之后，模型必须同时利用左边和右边的上下文来预测，双向理解就自然产生了。

但直接把所有被选中的词替换成 \[MASK\] 标记会引入一个问题：微调时输入里不会出现 \[MASK\]，预训练和微调之间产生了不匹配。论文的解决方案：被选中的 15% 的词里，80% 替换成 \[MASK\]，10% 替换成随机词，10% 保持不变。这样模型不能只靠「看到 \[MASK\] 就知道要预测」，而是必须对每个位置都保持理解能力。

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

## 3. 第二个预训练任务：下一句预测

很多 NLP 任务（比如问答、自然语言推理）需要理解两个句子之间的关系，但语言模型本身不直接建模这种关系。

论文加了第二个预训练任务：**下一句预测（Next Sentence Prediction，NSP）**。给模型两个句子 A 和 B，50% 的情况下 B 是 A 的真实下一句，50% 的情况下 B 是从语料库里随机抽的。模型要判断 B 是不是 A 的下一句。

这个任务的设计很简单，但论文的消融实验（ablation study，逐一去掉某个组件看效果变化）显示，去掉 NSP 会明显降低问答和自然语言推理任务的表现；不过后来也有工作（如 RoBERTa）对 NSP 的必要性提出了不同结论。

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

## 4. 模型架构

BERT 的架构其实没有什么新发明。它就是 [Transformer](/zh-CN/posts/attention-is-all-you-need/) 的编码器部分，一层层堆起来。

论文给出了两个规格：

- **BERT_BASE**：12 层，隐藏维度 768，12 个注意力头，参数量 1.1 亿
- **BERT_LARGE**：24 层，隐藏维度 1024，16 个注意力头，参数量 3.4 亿

BERT_BASE 的参数量和 OpenAI GPT 差不多，方便直接对比。两者最关键的区别只有一个：GPT 用的是单向注意力（每个词只能看左边），BERT 用的是双向注意力（每个词能看到所有位置）。

输入的表示由三部分相加构成：

- **词嵌入（Token Embedding）**：WordPiece 分词，词表 30,000
- **段嵌入（Segment Embedding）**：标记这个词属于句子 A 还是句子 B
- **位置嵌入（Position Embedding）**：告诉模型词的位置（BERT 用的是学习得到的位置编码，不是正弦余弦）

每个输入序列的开头都加一个特殊标记 \[CLS\]，它在最后一层的隐藏状态被用来做句子级别的分类（比如 NSP、情感分析）。两个句子之间用 \[SEP\] 分隔。

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

## 5. 微调：一个模型适配所有任务

BERT 最优雅的地方在于微调的简单性。预训练完成后，不管什么下游任务，做法几乎一样：在 BERT 上面加一层任务相关的输出层，然后用少量标注数据微调所有参数。

- **文本分类**（情感分析、自然语言推理）：取 \[CLS\] 位置的输出向量，接一个线性分类器
- **问答**（给一段文章，找出答案的起止位置）：对每个词的输出向量做两次线性变换，分别预测答案的开始和结束位置
- **序列标注**（命名实体识别）：对每个词的输出向量接一个分类器，逐词预测标签

预训练可能需要几天，但微调通常只要几十分钟到几小时（单块 TPU 上大部分任务不超过 1 小时）。这个效率差异是「预训练 + 微调」范式的核心吸引力。

## 6. 实验结果

论文在 11 个 NLP 任务上做了实验，全部刷新了当时的记录。

**GLUE 基准**（通用语言理解评估，包含 8 个子任务）：
- BERT_LARGE 平均分 80.5%，比之前最好的 OpenAI GPT 高出 7.7 个百分点
- 在最大的子任务 MNLI 上提升了 4.6%

**SQuAD v1.1**（阅读理解问答，Test F1）：
- BERT_LARGE 单模型 + TriviaQA 数据：F1 91.8，超过人类表现（91.2）
- BERT_LARGE 集成模型 + TriviaQA 数据：F1 93.2

**SQuAD v2.0**（包含无法回答的问题）：
- F1 达到 83.1，比之前最好的系统高出 5.1 个百分点

**SWAG**（常识推理）：
- 准确率 86.3%，比 OpenAI GPT 高出 8.3 个百分点

论文还做了模型大小的消融实验，发现一个重要结论：更大的模型在所有任务上都更好，即使在标注数据很少（只有 3,600 条）的任务上也是如此。这和当时的直觉（小数据集容易过拟合大模型）不太一样，说明预训练提供的知识可以有效缓解小数据集的过拟合（模型把训练数据「死记硬背」，对新数据表现差）风险。

## 7. 训练细节

**预训练数据**：BooksCorpus（8 亿词）+ 英文 Wikipedia（25 亿词），只使用文本段落，去掉了列表、表格和标题。论文强调必须用文档级语料而不是打乱的句子级语料，这样才能提取长距离的上下文关系。

**分词**：WordPiece，词表大小 30,000。WordPiece 会把不常见的词拆成更小的子词单元，比如 "playing" 可能被拆成 "play" + "##ing"。

**优化器**：Adam，学习率 1e-4，前 10,000 步线性热身，然后线性衰减。批次大小 256 个序列，最大序列长度 512。

**硬件**：BERT_BASE 在 4 块 Cloud TPU（16 块 TPU 芯片）上训练 4 天。BERT_LARGE 在 16 块 Cloud TPU（64 块 TPU 芯片）上训练 4 天。

**Dropout**：所有层的 dropout 率为 0.1。激活函数用的是 GELU（Gaussian Error Linear Unit），而不是 Transformer 原版的 ReLU。

## 8. 我的思考

读完这篇论文，有几个感受。

第一，BERT 的真正贡献不是模型架构（它就是 Transformer 编码器），而是训练方法。遮蔽语言模型这个想法看起来简单，但它巧妙地解决了一个根本矛盾：怎么在不让模型「作弊」的前提下，同时利用双向上下文。80/10/10 的遮蔽策略更是精心设计的，解决了预训练和微调之间的不匹配问题。

第二，BERT 和 GPT 的分野在这篇论文里就很清楚了。GPT 的自回归目标更天然适合生成；BERT 的双向编码更适合判别式语言理解任务。后来 GPT 走向了更大的规模和更强的生成能力，BERT 则衍生出了 RoBERTa、ALBERT、DeBERTa 等一系列理解型模型。两条路线至今仍在各自的领域里发挥作用。

第三，「预训练 + 微调」这个范式的影响远超 NLP。计算机视觉后来也全面转向了类似的思路（ViT、MAE），甚至多模态模型（CLIP、GPT-4V）也是在大规模预训练的基础上做微调或提示。BERT 不是第一个做预训练的，但它是第一个用如此简洁的方式，把预训练从一种有用技巧，推进成了 NLP 的主流工作范式。

第四，用真实 Python 重写 BERT 的输入处理时，你会感受到它的设计有多工整。\[CLS\] + 句子A + \[SEP\] + 句子B + \[SEP\]，配上三种嵌入相加，整个流程可以用一套统一的代码处理分类、问答、序列标注等完全不同的任务。这种「一个模型适配所有任务」的简洁性，是它真正的力量所在。

这篇论文的标题里有一个词很关键：Pre-training。在 BERT 之前，每个 NLP 任务都在从零开始学。BERT 证明了一件事：语言的通用知识可以先学好，然后迁移到几乎任何任务上。

这个想法改变了整个领域的工作方式。

---

**论文共读系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-CN/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神经网络进行序列到序列学习） — 编码器-解码器范式的确立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《Attention Is All You Need》](/zh-CN/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成为主角，Transformer 的诞生
- [《Scaling Laws for Neural Language Models》](/zh-CN/posts/scaling-laws-for-neural-language-models/)（神经语言模型的缩放定律） — 规模的数学：为什么更大的模型可预测地更好
- [《Language Models are Few-Shot Learners》](/zh-CN/posts/language-models-are-few-shot-learners/)（语言模型是少样本学习者） — 更大的模型，更善于从上下文中诱发能力
- [《Training Compute-Optimal Large Language Models》](/zh-CN/posts/training-compute-optimal-large-language-models/)（训练计算最优的大语言模型） — 怎样花算力最划算

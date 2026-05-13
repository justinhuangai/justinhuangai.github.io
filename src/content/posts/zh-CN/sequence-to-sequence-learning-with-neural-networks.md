---
title: "论文共读：《Sequence to Sequence Learning with Neural Networks》（使用神经网络进行序列到序列学习）"
date: "2026-01-24T16:41:08+08:00"
category: "Paper Reading"
description: 编码器-解码器范式的确立，附真实 Python 代码
tags: [paper-reading, seq2seq, AI, LLM, python]
pinned: false
---

2014 年 9 月 10 日，三个 Google 研究员在 arXiv（一个学术论文预印本网站，论文不用等期刊审稿就能直接发布）上传了一篇论文：[《Sequence to Sequence Learning with Neural Networks》](/papers/1409.3215v3.pdf)（使用神经网络进行序列到序列学习）。

作者是 Ilya Sutskever、Oriol Vinyals 和 Quoc V. Le，均来自 Google。Sutskever 是 AlexNet 的作者之一，与 Alex Krizhevsky、Geoffrey Hinton 合作完成了那篇引爆深度学习的计算机视觉论文，后来成为 OpenAI 联合创始人之一；Vinyals 后来在 DeepMind 主导了 AlphaStar（星际争霸 AI）；Quoc V. Le 则在 Google 推动了 AutoML 等研究。

这篇论文做了一件看似简单的事：用一个神经网络读完一句话，压成一个向量，再用另一个神经网络从这个向量里生成翻译。输入和输出可以长度不同、语言不同、结构不同。这个框架有一个名字，叫「序列到序列」（Sequence to Sequence，简称 Seq2Seq）。

它确立了编码器-解码器（Encoder-Decoder）这个范式。后来 [Bahdanau 在此基础上加入了注意力机制](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)，再后来 [Vaswani 等人用 Transformer 把整个架构重写](/zh-CN/posts/attention-is-all-you-need/)。但起点，是这篇论文。

## 0. 先认几个词

如果你没有机器学习背景，可以先按这篇论文的工作流，记住下面几个词：

- `Seq2Seq / 序列到序列`：把一段输入序列直接变成另一段输出序列，比如把英文句子变成法文句子。
- `Encoder / 编码器`：负责把输入从头到尾读完。
- `Decoder / 解码器`：负责把输出一个词一个词写出来。
- `RNN / 循环神经网络`：一种只能按顺序处理文本的旧架构。
- `LSTM`：RNN 的改良版，更擅长在长句子里记住前面的信息。
- `向量 / vector`：你可以先把它理解成“用一串数字压缩出来的一份摘要”。

## 1. 要解决什么问题

2014 年，深度神经网络已经在图像识别等任务上取得突破，但像机器翻译这种「直接把一段可变长序列映射到另一段可变长序列」的任务，神经网络还不擅长。

一句英语可能是 5 个词，翻译成法语变成 7 个词。输入和输出的长度不同，而且没有简单的一一对应关系。

传统的解决方案是把大量人工设计的规则和统计特征拼在一起，形成一个复杂的翻译流水线（统计机器翻译，SMT）。它能用，但每个组件都要单独调参，而且很难整体优化。

论文提出了一个更简洁的思路：能不能用一个端到端的神经网络，直接从源语言序列映射到目标语言序列？

## 2. 核心架构：编码器-解码器

论文的方法可以用一句话概括：**一个 LSTM 读，另一个 LSTM 写。**

LSTM（Long Short-Term Memory，长短期记忆网络）是一种特殊的 RNN，专门设计来处理长距离依赖问题。普通 RNN 在序列很长时容易「遗忘」前面的内容，LSTM 通过引入门控机制（决定哪些信息保留、哪些丢弃）来缓解这个问题。

具体流程：

1. **编码器**（一个 4 层深度 LSTM）从头到尾读完源句子，把整个句子压缩成一组固定长度的最终状态，交给解码器作为起点
2. **解码器**（另一个 4 层深度 LSTM）以这个向量为起点，一个词一个词地生成目标语言的翻译，直到输出结束符号 \<EOS\>

论文给出的概率公式：

$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_t p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

翻译成人话：给定源句子 x，生成目标句子 y 的概率，等于每一步生成下一个词的概率连乘起来。每一步的预测都依赖两样东西：编码器压缩出来的向量 v，以及之前已经生成的所有词。

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

架构本身不复杂。论文的贡献不在于发明了某个新组件，而在于证明了这个简单的框架真的能用，而且效果好到能和精心调校的传统系统竞争。

## 3. 三个关键设计决策

论文在实验中发现了三个对性能影响很大的设计选择：

**第一，用两个独立的 LSTM。** 编码器和解码器不共享参数。这样做稍微增加了参数量，但让模型能更好地分别处理源语言和目标语言的特性。论文提到这也让同时训练多个语言对成为可能。

**第二，用深层 LSTM。** 论文用了 4 层 LSTM，每多一层，困惑度降低近 10%。浅层 LSTM（1-2 层）的效果明显更差。深度给了模型更大的表示空间。

**第三，把源句子倒过来读。** 这是论文最出人意料的发现。把源句子 "a, b, c" 反转成 "c, b, a" 再喂给编码器，BLEU 分数从 25.9 跳到 30.6，提升了将近 5 分。

为什么反转有效？论文的解释是：正常顺序下，源句子第一个词离目标句子第一个词很远（中间隔了整个源句子）。反转之后，源句子的前几个词和目标句子的前几个词在时间上靠得很近，给梯度（模型用来调整参数的信号）创造了更多的「短距离依赖」，让优化变得更容易。

```python showLanguage
import torch


def reverse_source(source_tokens: list[int]) -> list[int]:
    return list(reversed(source_tokens))


source_sentence = [11, 23, 37, 42]
reversed_source = reverse_source(source_sentence)
source_tensor = torch.tensor([reversed_source], dtype=torch.long)
```

这个 trick 简单到几乎不像是正经的研究贡献，但它确实有效，而且揭示了一个深层问题：RNN 对序列里元素之间的距离很敏感，距离越近越容易学。这个问题后来被注意力机制从根本上解决了。

## 4. 实验结果

论文在 WMT '14 英法翻译任务上做了实验。

关键数字：
- **单个反转 LSTM**，beam size 12：30.59 BLEU
- **5 个反转 LSTM 的集成**，beam size 2：34.50 BLEU
- **5 个反转 LSTM 的集成**，beam size 12：**34.81 BLEU**
- **传统短语翻译系统**（Moses baseline）：33.30 BLEU

在论文报告的实验设置下，5 个 LSTM 的集成以 34.81 分超过了传统短语翻译系统的 33.30 分。考虑到 LSTM 的词表只有 8 万词（遇到词表外的词只能输出 UNK），而传统系统的词表几乎不受限，这个结果很有说服力。

论文还用 LSTM 对传统系统的 1000-best 候选列表做重排序，BLEU 分数进一步提升到 36.5，接近当时的最佳公开结果（37.0）。

另一个值得注意的发现：相比当时其他神经方法，LSTM 在长句子上的性能退化没那么严重。这和当时其他研究者报告的长句子性能急剧下降形成了对比，论文将此归功于反转源句子的策略。

## 5. 模型的「理解力」

论文还做了一个有趣的可视化实验。把不同句子输入编码器，取出最终的隐藏状态向量，用 PCA 降维到二维平面上画出来。

结果显示：
- 意思相近的句子在向量空间里聚在一起
- 主动语态和被动语态的句子（"I gave her a card" vs "I was given a card by her"）落在相近的位置
- 词序不同但意思相同的句子也能被正确聚类

这至少说明，编码器学到的表示不只是简单的词袋统计（把词混在一起不管顺序），而是包含了相当多的句法和语义信息。

## 6. 训练细节

**模型规格**：4 层 LSTM，每层 1000 个单元，词嵌入维度 1000，总参数量 3.84 亿。其中 6400 万是纯循环连接参数。

**硬件**：8 块 GPU。每层 LSTM 分配一块 GPU，剩余 4 块 GPU 用来并行化 softmax（因为词表有 8 万个词，softmax 计算量很大）。训练约 10 天。

**优化器**：SGD，不带动量，初始学习率 0.7。训练 5 个 epoch 之后每半个 epoch 将学习率减半，总共训练 7.5 个 epoch。

**梯度裁剪**：当梯度的 L2 范数超过阈值 5 时，按比例缩小。这是为了防止梯度爆炸（梯度值突然变得极大，导致参数更新失控）。

**批次优化**：把长度相近的句子放在同一个批次里，避免短句子为长句子「陪跑」浪费计算资源，带来了 2 倍的训练加速。

## 7. 我的思考

读完这篇论文，有几个感受。

第一，这篇论文的野心很大但方法很简单。用一个 LSTM 读，另一个 LSTM 写，中间就靠一个向量传递信息。没有注意力，没有复杂的对齐机制，甚至没有任何关于语言结构的先验假设。然后它真的跑通了，而且效果强到足以和当时精心调校的传统系统竞争。这说明一个道理：在数据和算力充足的条件下，简单的端到端方法可以出乎意料地强。

第二，反转源句子这个发现很有启发性。它不是一个优雅的解法，更像是一个 hack。但它揭示了 RNN 的根本局限：对序列里元素之间的距离敏感。Bahdanau 的注意力机制让模型可以「跳着看」，不再受距离限制。Transformer 更进一步，完全抛弃了顺序处理，让任意两个位置之间的距离始终是 1。从反转到注意力到 Transformer，是同一个问题的三代解法。

第三，这篇论文和 Bahdanau 的论文几乎同时发表（都是 2014 年 9 月）。Sutskever 确立了编码器-解码器范式，Bahdanau 指出了固定长度向量的瓶颈并用注意力机制解决了它。两篇论文像是同一枚硬币的正反面：一面是框架，一面是修补框架最大缺陷的方法。

第四，用真实 Python 重写的过程中，你会感受到这个架构有多简洁。编码器就是循环读完，解码器就是循环写出来。但也正因为简洁，它的天花板很明显：所有信息必须挤过一个固定长度的向量。这个瓶颈在你写代码的时候会变得格外直观。

一个向量能装下多少信息？这是这篇论文暗含的问题。

对更长、更复杂的句子来说，不够。

于是后来有了注意力，有了 Transformer。

---

**论文共读系列**

- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《Attention Is All You Need》](/zh-CN/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成为主角，Transformer 的诞生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-CN/posts/bert/)（BERT：用于语言理解的深度双向 Transformer 预训练） — 预训练范式的确立
- [《Scaling Laws for Neural Language Models》](/zh-CN/posts/scaling-laws-for-neural-language-models/)（神经语言模型的缩放定律） — 规模的数学：为什么更大的模型可预测地更好
- [《Language Models are Few-Shot Learners》](/zh-CN/posts/language-models-are-few-shot-learners/)（语言模型是少样本学习者） — 更大的模型，更善于从上下文中诱发能力
- [《Training Compute-Optimal Large Language Models》](/zh-CN/posts/training-compute-optimal-large-language-models/)（训练计算最优的大语言模型） — 怎样花算力最划算

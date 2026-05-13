---
title: "《Attention Is All You Need》：Transformer 的设计原点"
date: "2026-01-06T16:18:46+08:00"
category: "Paper Reading"
description: 分享我对 Transformer 论文的理解，附真实 Python 代码
tags: [paper-reading, transformer, AI, LLM, python]
pinned: false
---

2017 年 6 月 12 日，八个人在 arXiv（一个学术论文预印本网站，论文不用等期刊审稿就能直接发布）上传了一篇论文，标题只有五个词：[《Attention Is All You Need》](/papers/1706.03762v7.pdf)（注意力就是你所需要的全部）。

这八个人是 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser 和 Illia Polosukhin，当时大多在 Google Brain 和 Google Research 工作。

论文发出之后，这个八人组几乎全部散开。Noam Shazeer 离开 Google 创立了 Character.AI，后来又被 Google 高价请回；Aidan Gomez 从多伦多大学博士还没毕业就创立了 Cohere，做企业级大模型；Llion Jones 去了日本，创立了 Sakana AI；Illia Polosukhin 走了一条谁都没想到的路，创立了 NEAR Protocol，做区块链；Ashish Vaswani 和 Niki Parmar 搭档创立了 Adept AI，后来又一起创立了 Essential AI；Jakob Uszkoreit 创立了 Inceptive，用 AI 设计 RNA 药物；Łukasz Kaiser 则加入了 OpenAI，参与了 GPT 系列的研发。

八位作者，七家公司，横跨 AI、区块链、生物技术。

近九年后的今天，ChatGPT、Claude、DeepSeek、Qwen，这些 AI 产品的底层架构思路，大多都能追溯到这 15 页纸。

这篇文章是我读完论文后的理解，附带真实 Python 代码示例。不是翻译，不是摘要。没有技术背景也能读下去。

## 0. 先认几个词

如果你没有机器学习背景，先按这篇论文真正想替换掉的旧方案，记住下面几个词就够了：

- `RNN / 循环神经网络`：一种更早的序列模型。它处理句子时必须一个词一个词往后读，像人用手指着文章逐行看。
- `attention`：从很多信息里，挑出当前最该看的那几部分。你可以先把它理解成“有选择地回头看重点”。
- `Query / Key / Value`：注意力机制里的三个角色。Query 像“我现在想找什么”，Key 像“每段信息贴着什么标签”，Value 则是“真正被取出来的内容”。
- `Transformer`：以 attention 为核心搭起来的一整套架构。它不靠循环一步步往前推，而是让每个位置都能直接看其他位置。
- `并行`：这里不是说模型更聪明，而是说它能同时处理很多位置，不必像 RNN 那样排队。

## 1. 一句话说清楚

在 Transformer 之前，AI 处理语言的方式像是一个人用手指着书，一个字一个字地往下读。读到第 100 个字的时候，第 1 个字说了什么，已经记不太清了。句子越长，遗忘越严重。这就是循环神经网络（RNN，一种早期的 AI 架构）的根本瓶颈。

论文的作者们问了一个问题：**为什么一定要按顺序读？**

和必须逐步处理的 RNN 不同，Transformer 可以并行处理整段输入，直接建模任意两个位置之间的关系。不用排队，不用等前一个词处理完才能看下一个。

论文管这个核心能力叫「注意力」。注意力机制最早由 [Bahdanau 等人在 2014 年提出](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)，当时是作为 RNN 的辅助组件。这篇论文的标题要表达的不是「模型里真的只剩注意力」，而是：在序列建模里，注意力第一次被推到了主角的位置，不再需要循环和卷积（一种通过滑动窗口提取局部特征的方法）作为骨架。

## 2. 注意力到底在做什么

想象你走进一个嘈杂的酒吧，二十个人同时在说话。你的大脑不会平均分配注意力给每个人。有人喊了你的名字，你的耳朵瞬间锁定那个方向，其他声音自动变成背景噪音。

Transformer 对每个词做同样的事。论文里定义了三个角色：

- **Query（查询）**：这个词在找什么信息。相当于你的耳朵在搜索「谁在叫我」
- **Key（键）**：这个词能提供什么信息。相当于酒吧里每个人的声音特征
- **Value（值）**：这个词实际携带的内容。相当于那个人说的具体话

每个词的 Query 会和其他词的 Key 做匹配。匹配度高的，就从对方的 Value 里获取更多信息。匹配度低的，直接忽略。

论文给出的公式叫 Scaled Dot-Product Attention：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

看到公式别慌。一步一步拆：

- **QK^T**：Q 和 K 做点积。什么是点积？把两组数字对应位置相乘，再加起来。比如 [1, 2] 和 [3, 4] 的点积是 1×3 + 2×4 = 11。数字越大，说明两个词越相关。这一步算的就是每对词之间的「匹配分数」
- **/ √d_k**：除以一个数来缩放。d_k 是向量的长度（向量可以理解为「一串用来描述某个东西的数字」，比如用 64 个数字描述一个词的含义）。为什么要除？因为数字串越长，点积结果越大。不缩放时，维度越大点积的方差越大，softmax 容易进入饱和区（几乎所有概率集中在一个词上），梯度（模型用来调整自身参数的信号）变得很小，训练会不稳定
- **softmax**：把一组分数转换成概率，所有概率加起来等于 1。比如三个词的分数是 [10, 2, 1]，softmax 之后大概变成 [0.99, 0.007, 0.003]。分数最高的那个词几乎拿走了全部注意力，其他的被压到接近零
- **× V**：用这些概率去加权每个词的实际内容。概率高的词贡献大，概率低的词贡献小。最终输出是一个融合了关键信息的新向量

用 Python（基于 PyTorch）写出来：

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

就这么几行代码。很多后来改变行业的能力，底层都建立在这几行运算之上。

## 3. 多头注意力：同时从多个角度看

单个注意力头通常只能偏向某一类关系模式。但语言这东西，一句话里藏着好几层意思。

拿「我昨天在深圳吃了潮汕牛肉火锅」来说：
- 「我」和「吃了」之间是谁做了什么的关系
- 「昨天」和「吃了」之间是时间关系
- 「深圳」和「潮汕牛肉火锅」之间是地点与食物的关系

让一个头同时兼顾这么多层次，很难。论文的做法是用多头机制：派出 8 个头并行运算，让模型有机会从不同子空间同时观察一句话，最后把各自的发现拼起来。

论文原文的公式：

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

拆开看：
- **head_1, ..., head_h**：8 个头各自独立做一次注意力运算，得到 8 份结果
- **Concat**：把 8 份结果首尾相连，拼成一个长向量
- **W^O**：一次线性变换（可以理解为「乘以一个矩阵」），把拼接后的长向量压回原来的维度。相当于一个主管听完 8 个调查员的汇报，输出一份综合结论

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

论文里的参数：模型用 512 个数字描述一个词（d_model = 512），8 个头，每个头分到 64 个数字（512 ÷ 8 = 64）。8 个头的总计算量和 1 个 512 维的头差不多，但表达能力强得多。用同样的代价，换来多视角的理解力。这笔账算得漂亮。

## 4. 位置编码：告诉模型词的顺序

Transformer 并行处理整个句子，速度是快了，但代价是它丢掉了词的先后顺序。如果没有额外的位置信息，注意力机制本身并不知道「猫吃鱼」和「鱼吃猫」有什么区别。这显然不行。

怎么补救？给每个位置生成一个独一无二的「地址编码」，加到词的向量上。模型看到的不再是「猫」和「鱼」，而是「第 1 个位置的猫」和「第 3 个位置的鱼」。

论文用正弦和余弦函数来生成这个编码：

$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
\operatorname{PE}(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

公式看着唬人，核心思路很直观：
- **pos**：词在句子里的位置（第 1 个、第 2 个、第 3 个……）
- **i**：向量的第几个维度。偶数位置用 sin，奇数位置用 cos
- **10000^(2i/d_model)**：一个随维度变化的缩放因子。低维度变化快，高维度变化慢。就像时钟：秒针一分钟转一圈，时针十二小时才转一圈。不同「指针」覆盖不同的时间尺度，组合在一起就能精确定位任意时刻

最终效果：每个位置得到一串独一无二的数字指纹，模型靠这个指纹区分词的先后顺序。

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

为什么偏偏选正弦余弦？因为它有一个优雅的数学性质：两个词相隔固定距离，无论它们出现在句首还是句尾，位置编码之间的关系是一样的。模型不用死记「位置 3 和位置 8」的关系，只需要学会「相隔 5 个位置」意味着什么。论文团队也试过让模型自己学位置编码，效果差不多，但正弦版本有一个额外的优势：它能处理训练时没见过的更长句子。

## 5. 编码器与解码器

Transformer 的完整架构分两半。

**编码器**（6 层堆叠）负责读懂输入。每层包含两个子层：一个多头自注意力，一个前馈网络。每个子层都有两个保护机制：

- **残差连接**：把子层的输入直接加到输出上，即 x + Sublayer(x)。为什么？想象你给一张照片加滤镜。如果滤镜效果不好，残差连接保证你还能看到原图。在深层网络里，信息每经过一层都会被变换，传到第六层可能已经面目全非。残差连接让原始信号可以「抄近道」直达深层，防止信息在传递中丢失
- **层归一化**（LayerNorm）：把数值调整到统一范围，防止有的数字大到爆炸、有的小到消失。类似于考试成绩标准化，不管原始卷面分差异多大，标准化后都在一个可比较的区间

**解码器**（6 层堆叠）负责生成输出。结构和编码器类似，但多了两个关键设计：

第一，**交叉注意力**：解码器生成每个词时，会回头「看」编码器的输出。翻译场景下，就是一边写英文一边回头看中文原文。

第二，**遮罩**（masking）：生成第 3 个词时，只允许看到前 2 个词，第 4 个及之后的位置被屏蔽（注意力分数设为负无穷，经过 softmax 后变成零）。道理很简单：你写作文的时候，下一个字还没写出来，不能偷看。

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

还有一个容易被忽略的组件：前馈网络。公式是 FFN(x) = max(0, xW1 + b1)W2 + b2。翻译成人话：先把每个词的 512 维向量扩大到 2048 维（乘以一个矩阵再加一个偏置），用 ReLU 过滤一遍（所有负数变成零，正数保留），再压回 512 维。ReLU 这一步是关键：它引入了「非线性」，让模型能学到直线画不出来的复杂模式。如果全是线性变换，多层叠加在数学上仍可合并为单层，非线性是模型表示复杂模式的前提。

## 6. 训练细节

架构设计完了，怎么训练它？论文在这里也有不少讲究。

**硬件**：8 块 NVIDIA P100 GPU。基础版模型训练 12 小时（10 万步），大号模型训练 3.5 天（30 万步）。放在今天看，这个成本低得惊人。

**优化器**：用的是 Adam（一种让模型自动调整参数的算法），但学习率的设计很巧妙。学习率决定了模型每一步「迈多大步子」。步子太大容易跨过最优解，太小又走得慢。论文的策略是前 4000 步逐渐提速（warmup，热身），避免一开始更新过猛；4000 步之后按计划逐渐减速，让训练后期更稳定。先升后降，前半段大胆探索，后半段精细打磨。

**正则化**：两招。第一招是 Dropout，训练时随机关掉 10% 的神经元（可以理解为网络中的计算节点），迫使模型不依赖任何单一路径，学到更鲁棒的特征。第二招是 label smoothing（标签平滑，ε = 0.1）：训练时不告诉模型「正确答案的概率是 100%」，而是说「90% 是正确答案，剩下 10% 分给其他选项」。这会让模型在一个指标上变差（困惑度，衡量模型有多「拿不准」），但翻译质量反而更好。直觉上说，一个承认自己不是 100% 确定的模型，比一个过度自信的模型更可靠。

**结果**：论文用 BLEU 分数（机器翻译的标准评分，衡量机器翻译和人工翻译有多接近，满分 100）来衡量效果。英德翻译 28.4 分，英法翻译 41.8 分，都刷新了当时的记录。训练成本比之前的方法低了一到两个数量级。更快，更强，更便宜。

## 7. 我的思考

读完这篇论文，有几个感受。

第一，这篇论文的核心洞察极其简洁：扔掉顺序处理的包袱，让注意力机制直接建模任意两个位置之间的关系。Self-Attention、残差连接、Layer Normalization，没有一个是新发明。真正的突破不在于发明新工具，而在于作者们敢赌「这些简单的积木拼在一起就够了」，然后用实验证明了自己是对的。

第二，用真实 Python 代码写出来的过程让我更深地理解了每一个设计决策。当你自己写出 Scaled Dot-Product Attention，你会切实感受到那个 √d_k 的缩放有多重要。当你实现 masking，你会理解自回归生成的约束从何而来。论文读十遍，不如自己写一遍。

第三，真正让我震撼的，不是它后来衍生出了多少模型，而是它当年就把问题改写了：从「怎么按顺序记住一句话」，变成「怎么让每个位置直接找到它最该看的信息」。GPT、BERT、T5、LLaMA，全是这个问题改写之后的产物。

一个足够好的架构，能走多远，取决于有多少人愿意在它上面继续建设。

这篇论文给出了那个架构。

《Attention Is All You Need》（注意力就是你所需要的全部）。

---

**论文共读系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-CN/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神经网络进行序列到序列学习） — 编码器-解码器范式的确立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-CN/posts/bert/)（BERT：用于语言理解的深度双向 Transformer 预训练） — 预训练范式的确立
- [《Scaling Laws for Neural Language Models》](/zh-CN/posts/scaling-laws-for-neural-language-models/)（神经语言模型的缩放定律） — 规模的数学：为什么更大的模型可预测地更好
- [《Language Models are Few-Shot Learners》](/zh-CN/posts/language-models-are-few-shot-learners/)（语言模型是少样本学习者） — 更大的模型，更善于从上下文中诱发能力
- [《Training Compute-Optimal Large Language Models》](/zh-CN/posts/training-compute-optimal-large-language-models/)（训练计算最优的大语言模型） — 怎样花算力最划算

---
title: "《Scaling Laws for Neural Language Models》：规模的数学"
date: "2026-03-01T16:45:39+08:00"
category: "Paper Reading"
description: 规模的数学：为什么更大的模型可预测地更强，附真实 Python 核心代码
tags: [paper-reading, scaling-laws, AI, LLM, python]
pinned: false
---

训练大模型首先是一个预算分配问题。参数、数据和算力都要钱；问题不是“越大越好”这种口号，而是下一块钱该买参数、买数据，还是买更长训练。

[《Scaling Laws for Neural Language Models》](/papers/2001.08361v1.pdf) 把大模型训练从经验主义推进到预算函数。它没有发明新架构，却给了行业一个更硬的判断工具：在花钱之前，先估计回报曲线。

## 0. 先认几个词

如果你平时不常看这种带公式的论文，先把这几个词认熟，后面会顺很多：

- `参数量`：模型里一共有多少可学习参数，也就是模型的大小。
- `数据量`：训练时一共喂给模型多少文本。
- `算力 / compute`：训练时总共做了多少计算。你可以先把它当成“电费账单”。
- `loss / 损失`：模型犯错有多严重，通常越低越好。
- `幂律 / power law`：某个量按固定指数变化的关系；画在对数坐标上，经常会接近一条直线。
- `对数坐标`：像 `1、10、100、1000` 这样按倍数增长的刻度，不是 `1、2、3、4` 那样均匀加一。

## 1. 要解决什么问题

到 2020 年初，AI 圈已经知道一件事：模型越大，效果越好。但「越大越好」不是科学，是感觉。

大家回答不了几个最基本的问题：算力预算翻一倍，效果能好多少？这笔钱是该花在更大的模型上、更多的数据上，还是更久的训练上？有没有一个公式，可以在花钱之前就算出来？

这篇论文给出了公式。不是靠拍脑袋，也不是靠经验法则：靠方程。

## 2. 幂律：核心发现

论文的核心发现，一句话就能说清：**AI 模型的表现好坏，和它的「个头」之间存在一个简洁的数学关系。**

具体来说，论文测量了三样东西对模型表现的影响：模型有多大（参数量）、喂了多少数据、烧了多少算力。在论文观测到的范围内，只要模型的瓶颈主要在其中一项上，表现和这一项之间的关系都能画成一条漂亮的直线：前提是你把坐标轴的刻度取成对数（也就是 1, 10, 100, 1000 这样等距排列，而不是 1, 2, 3, 4）。

这种「对数坐标下的直线关系」，数学上叫**幂律（power law）**。

三个方程概括了整篇论文：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$

$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$

$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

别被符号吓到。拆开就是几个简单的角色：

- **L** 是「测试损失」：你可以理解为模型的考试成绩，只不过分数越低代表越好（想象成失误率：失误越少，能力越强）
- **N** 是参数量，也就是模型的「个头」。参数越多，模型能记住的规律就越多
- **D** 是训练数据量，也就是模型的「课本厚度」。课本越厚，能学到的东西就越多
- **C** 是训练消耗的总算力，也就是「电费账单」。单位是 PetaFLOP-days：每秒做一千万亿次运算，连续跑一整天
- **N_c、D_c、C_c** 是常数，当参考点用的
- **α**（alpha）是幂律的指数，决定了「个头翻倍时，成绩能进步多少」。指数越大，同样的投入换来的进步越大

为什么幂律这么重要？因为它意味着回报不会快速见顶。

打个比方：如果 AI 的进步像背单词：前 1000 个很容易，后面越来越慢，背到 5000 个就几乎不动了：那就是对数增长，回报递减极快。但幂律不一样，它像修路：你修到 10 公里的时候觉得效果不错，修到 100 公里效果更好，修到 1000 公里效果又上了一个台阶。每上一个量级都有实实在在的回报。

当然，论文也提醒了：路不可能修到无限远。这个趋势最终一定会变平。但在论文观测到的范围内，这条线走得干净利落，没有撞墙的迹象。

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

三个指数本身就在讲故事。数据量（α = 0.095）扩大一个量级带来的进步最大。模型大小（α = 0.076）次之。算力（α = 0.050）最低：因为如果你只是堆算力，却不合理分配到模型大小和训练时长上，就是在烧钱。真正的杠杆在于：扩对东西。

## 3. 在论文测过的范围内，怎么搭不重要，搭多大才重要

这是论文最出人意料的发现。

Transformer 有很多可以调的「形状」参数：堆多少层（深度）、每层有多宽（隐藏维度）、用多少个注意力头、前馈网络有多大。直觉上你可能觉得，这些比例关系至关重要，调好了事半功倍。

但论文发现：在它测过的 Transformer 形状范围内，这些比例关系几乎不影响最终表现。真正起决定作用的是一个数字：非嵌入参数的总量。

什么是「非嵌入参数」？简单说，模型的参数分两种：一种是「词典」（嵌入层，负责把文字转成数字），另一种是「大脑」（Transformer 层，负责理解和推理）。论文发现，真正决定模型能力的是「大脑」的大小，不是「词典」的大小。

一个只有 2 层但每层特别宽的 Transformer？和一个 40 层但每层很窄的 Transformer？只要它们的「大脑」总参数量接近，考试成绩就差不多。

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

这个发现的实际意义很直接：你不用花几周去搜索「最优架构」。选一个合理的 Transformer 形状，然后把精力放在做大就行。

## 4. 模型什么时候会「死记硬背」：数据瓶颈

模型不是越大越好：如果你的课本太薄的话。

想象一个记忆力超强的学生，你只给他一本 100 页的教材，他很快就能把这本书倒背如流。但这不叫「学会了」，这叫「背下来了」。考试一换题型就傻眼。这就是过拟合：模型把训练数据死记硬背了，却没有学到真正的规律。

论文这里真正漂亮的地方，是给出了一个统一公式，把「模型多大」和「数据多少」如何共同决定表现写进了一个式子：

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
$$

这个公式说的是：模型的考试成绩不是由「个头」或「课本厚度」单独决定的，而是两者一起。如果模型够大但数据不够，性能就卡在数据上；如果数据够多但模型太小，性能就卡在模型上。过拟合，就是「个头大、课本薄」这对矛盾的自然结果。

从这个关系出发，论文还给了一个粗略的经验门槛：「课本至少要多厚，才不会让这个学生背书」：

$$
D \gtrsim 5 \times 10^3 \times N^{0.74}
$$

用大白话说：模型大 10 倍，课本只需要厚大约 5.5 倍就够了。更大的模型学习效率更高：同样看一页书，它能悟到更多。

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

按这个公式粗略一算，GPT-3 那个级别（1750 亿参数）的模型要想不「背书」，课本厚度应该接近万亿词元。但 GPT-3 实际只喂了 3000 亿词元：远没到安全线。回头看，GPT-3 的数据其实是偏少的。这也是为什么后来业界重新审视了「模型多大、数据喂多少」这个配比：最典型的就是 Chinchilla 论文（Hoffmann 等人，2022 年），直接指出：之前那些大模型，数据普遍喂少了。

## 5. 算力最优分配：钱该怎么花

如果你有一笔固定的算力预算：比如说够你租 1000 块 GPU 跑一个月：应该怎么花？这是论文里最有实际价值的问题，答案相当反直觉。

论文发现最优分配遵循：

$$
N_{\mathrm{opt}} \propto C^{0.73}
$$

$$
B_{\mathrm{opt}} \propto C^{0.24}
$$

$$
S_{\mathrm{opt}} \propto C^{0.03}
$$

翻译成人话：如果你的预算涨了 10 倍，你应该把模型做大约 5.4 倍，每次喂的数据量增加约 1.7 倍，训练时间几乎不延长（大约只多 7%）。

钱主要花在哪？花在把模型做大上。

反直觉的部分来了：**你应该造一个尽可能大的模型，然后不用训到头就可以停。** 大多数人的直觉是「我选个中等大小的模型，然后慢慢训，训到极致」。缩放定律说的恰恰相反：同样一笔钱，一个没训完的大模型，比一个训透了的小模型，表现更好。

就像装修：同样 50 万预算，与其在一个 60 平的小户型里堆满顶配材料，不如买一个 120 平的大户型做个简装。空间大了，住起来怎么都比小房子舒服。

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

这个结论深刻地改变了整个行业。五个月后发布的 GPT-3 直接遵循了这个思路：造一个当时规模空前的 1750 亿参数模型，而不是把小模型训到极致。后来的 Chinchilla 论文（Hoffmann 等人，2022 年）更新了具体的指数，认为大多数大模型的数据其实喂少了：但「最优权衡是可以算出来的」这个核心洞察，源头在这里。

## 6. 临界批大小：什么时候该加机器

训练 AI 模型的时候，你可以选择每次给模型看多少数据再更新一次：这叫「批大小」（batch size）。批大小越大，你可以同时用更多 GPU 并行处理，训练速度就越快。但并不是加机器就一定有用。

论文发现，批大小存在一个「甜蜜点」：

$$
B_{\mathrm{crit}} \propto L^{-4.8}
$$

训练刚开始的时候，模型还很「菜」，每批数据都能给它很大启发，小批量就够了。但训练到后期，简单的规律都学完了，每批数据带来的新信息越来越少，这时候就需要更大的批量来「凑够信号」。

甜蜜点以下，加机器很划算：机器翻倍，训练时间几乎减半。甜蜜点以上，加机器就是烧钱：多出来的机器几乎不加速。

```python showLanguage
def critical_batch_size(loss: float, b_star: float, l_star: float) -> float:
    return b_star * (l_star / loss) ** 4.8
```

很多团队全程用固定的批大小训练。缩放定律告诉你：应该随着训练推进逐步加大：开始用小批量，模型变强后再加机器。

## 7. 这篇论文改变了什么问题

Scaling Laws 的钉子句是：**训练大模型不是玄学试错，而是预算函数。**

这篇论文最重要的贡献不是某个指数，而是把“更大更好”从经验判断变成可估算的曲线。参数、数据、算力不再只是工程变量，而是可以放进同一张账本里比较边际回报的投资项。

它没有告诉行业永远只做更大模型。它告诉行业，在给定假设和观测范围内，花钱之前应该先算：扩大参数、增加数据、延长训练，哪一个最可能带来更低 loss。GPT-3 的规模决策之所以看起来突然变得合理，背后就是这种预算信心。

下次看模型规模，不要问“是不是军备竞赛”。先问预算函数是什么，瓶颈变量是谁，边际收益还剩多少。真正成熟的规模化，不是烧更多钱，而是知道钱烧到哪里会变成能力。

---

**论文共读系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-CN/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神经网络进行序列到序列学习） — 编码器-解码器范式的确立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《Attention Is All You Need》](/zh-CN/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成为主角，Transformer 的诞生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-CN/posts/bert/)（BERT：用于语言理解的深度双向 Transformer 预训练） — 预训练范式的确立
- [《Language Models are Few-Shot Learners》](/zh-CN/posts/language-models-are-few-shot-learners/)（语言模型是少样本学习者） — 更大的模型，更善于从上下文中诱发能力
- [《Training Compute-Optimal Large Language Models》](/zh-CN/posts/training-compute-optimal-large-language-models/)（训练计算最优的大语言模型） — 怎样花算力最划算

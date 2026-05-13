---
title: "《Training Compute-Optimal Large Language Models》：Chinchilla 改变了什么"
date: "2026-03-11T16:58:04+08:00"
category: "Paper Reading"
description: Chinchilla 论文：为什么 2022 年的大模型全都「喂少了」，以及算力预算到底该怎么分配，附真实 Python 核心代码
tags: [paper-reading, chinchilla, scaling-laws, AI, LLM, python]
pinned: false
---

2022 年 3 月 29 日，DeepMind 的一个团队在 arXiv（学术论文预印本网站，论文不用等期刊审稿就能直接发布）上传了一篇论文：[《Training Compute-Optimal Large Language Models》](/papers/2203.15556v1.pdf)（训练计算最优的大语言模型）。

第一作者是 Jordan Hoffmann，联合作者包括 Sebastian Borgeaud、Arthur Mensch、Elena Buchatskaya、Trevor Cai、Eliza Rutherford 等一大串名字：当时全部在 DeepMind。值得一提的是，Arthur Mensch 后来联合创办了 Mistral AI，欧洲最耀眼的 AI 公司之一。

这篇论文圈内通常叫它「Chinchilla 论文」，名字来自团队为验证结论而训练的一个 700 亿参数模型。那只龙猫（chinchilla）比论文标题更出圈：在 AI 圈子里，「Chinchilla scaling」成了这篇论文核心观点的代名词。

而这个核心观点，简单、大胆，并且让整个行业都有点不舒服：**2022 年那些最大的语言模型，很多并不是「模型不够大」，而是在各自的算力预算下被显著欠训练了。**

## 0. 先认几个词

这篇论文会频繁谈到“该把预算花在哪”，所以先把下面几个词认熟，会更容易抓住重点：

- `算力预算`：你愿意为这次训练总共花多少计算资源。
- `参数量`：模型有多大。
- `token / 词元`：模型训练时实际读进去的最小文本单位，可以粗略理解成“模型看到的字或词片段”。
- `loss / 损失`：模型总体错得有多厉害，通常越低越好。
- `scaling law / 缩放定律`：当参数量、数据量、算力变化时，模型表现如何跟着变化。
- `欠训练 / undertrained`：不是模型太小，而是训练数据和训练步数没跟上，模型的潜力没有被充分用出来。

## 1. 要解决什么问题

先交代一下背景。

2020 年，OpenAI 发了一篇 [缩放定律论文](/zh-CN/posts/scaling-laws-for-neural-language-models/)（Kaplan 等人），核心结论是：模型越大效果越好，而且好多少是可以用公式算出来的。到 2022 年春天，整个行业把这句话奉为圣旨。

GPT-3，1750 亿参数，训了 3000 亿词元（词元就是模型看的「字」，大约每个英文单词切成 1-2 个词元，中文大约一个字一个词元）。DeepMind 自家的 Gopher，2800 亿参数，也训了 3000 亿词元。随后 Google 又发布了 5400 亿参数的 PaLM。趋势很明确：参数拉满。

但有一个问题就摆在那儿，大家却视而不见。

Kaplan 等人的结论是：有更多钱（算力）的时候，大部分预算应该花在造更大的模型上（用数学写就是 N ∝ C^0.73），训练数据只需要跟着慢慢涨就行（D ∝ C^0.27）。翻译成白话就是：模型尽量大，数据差不多就行。

Hoffmann 的团队提了一个很简单的问题：这个结论靠谱吗？

## 2. 三条路走到同一个答案

这篇论文最有说服力的地方在于方法论。他们没有只做一组实验就下结论：而是从三个完全独立的角度切入，三条路走到了同一个终点。

**方法一：总预算不变，换着分。** 打个比方：你有 50 万装修预算，可以选择买贵的家电配便宜家具，也可以反过来。他们就是这么做的：训了超过 400 个模型，参数量从 7000 万到 160 多亿不等，每个模型分到的「模型大小 vs 训练数据」比例不同，但总算力一样。对于每个预算档位，他们找出效果最好的那个分配方案。

**方法二：画等高线。** 他们训了 9 种不同大小的模型（从 7000 万到 100 亿参数），每种喂不同量的数据，专门设计成每一组的总算力大致相等。就像在地图上画等高线一样，沿着每条「等算力线」找最优点。

**方法三：直接写方程，用数据拟合。** 他们假设模型表现可以用下面这个方程描述：

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

这方程其实在说一件很直觉的事：模型考得不好，要么是脑子不够大（N 太小），要么是书看少了（D 太小），要么两个都有。E 是一个谁都突破不了的下限，学术上叫「熵」（entropy）。熵是信息论里的一个概念，衡量的是「一件事有多不确定」。抛一枚均匀硬币，正反各 50%，不确定性最大，熵最高。如果硬币灌了铅，99% 正面朝上，结果几乎没悬念，熵就很低。语言也有熵。「太阳从___升起」，这句话的熵很低，几乎所有人都会填「东方」。但「我今天吃了___」，后面可以是火锅、三明治、亏，答案五花八门，熵就高。自然语言整体的熵，就是 AI 模型表现的理论天花板：不管模型多强、数据多多，它不可能比这条线更好，因为语言本身就有这么多不确定性。把所有训练数据代入这个方程拟合，就能反推出：给定一笔算力预算，模型该多大、数据该多少。

三条路的答案一致：

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

a ≈ b ≈ 0.5 的真正含义是：随着算力增长，模型大小和训练数据应当按近似相同的比例一起扩张。算力翻 10 倍时，两者都大约增到 3.2 倍；算力翻 2 倍时，两者都大约增到 1.4 倍。换句话说，模型大小每翻一倍，训练数据也应该翻一倍。这跟 Kaplan 等人的结论直接矛盾：Kaplan 说算力主要应该花在模型大小上。

打个比方：Kaplan 的建议像是「买了一套 200 平的豪宅，但只摆了几件家具」。Chinchilla 说：「不对，你应该买 100 平的房子，然后好好装修：家具配齐、软装到位。同样的总花费，住着舒服得多。」

## 3. Kaplan 为什么算偏了

这不是谁「做错了」，而是实验设定不同，最终导致了不同的最优分配结论。两个团队都做了认真的工作。

差别出在一个训练细节上：学习率调度。

学习率是什么？简单说，AI 模型在训练过程中会不断调整自己的参数。学习率就是「每次调整幅度有多大」。一般来说，训练初期步子大一点（快速学），后期步子小一点（精细调）。这个「先大后小」的节奏安排，就叫学习率调度。

Kaplan 等人用了固定的学习率调度：不管你打算训多久，节奏都一样。但合理的做法是：训得越久，后期的调整步子应该越小越细。他们没做这个适配，导致训练时间一拉长，后半段的学习效率就掉下来了。这就让「训更久」看起来不划算，间接得出了「别在训练时长上花钱，把钱花在模型大小上」的结论。

Hoffmann 的团队给每次训练都单独调整了学习率调度，让每种配置都能发挥最佳水平。一旦做到这点，训更多数据的回报远比 Kaplan 的数字暗示的要大。

这件事的教训很深刻：缩放定律是经验法则，不是物理常数。实验条件变了，结论就可能变。

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

## 4. 那个描述一切的方程

方法三值得单独展开看一看，因为它给出了一个完整的数学模型来描述模型表现：

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

拟合出来的具体数字：

- E = 1.69 ： 地板值，也就是自然语言的熵。「我今天吃了___」后面填什么，连人都没法百分百答对，这种不确定性就是熵。不管模型多强、数据多多，表现不可能好过这个数
- A = 406.4，α = 0.34 ： 「脑子不够用」的罚分
- B = 410.7，β = 0.28 ： 「书看少了」的罚分

这个方程的结构很好理解。模型表现不好，原因无非三个：第一，语言本身就有不确定性，神仙来了也预测不了（E）；第二，模型太小，记不住那么多规律（A/N^α）；第三，训练数据太少，见过的世面不够（B/D^β）。后两个罚分是加法关系：哪块短板更严重，哪块就拖后腿更厉害。

就好比考试成绩不好有两个原因：一是脑子不够用（模型太小），二是书看少了（数据不够）。这两块是独立的短板，得分别补。

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

## 5. 那张让全行业尴尬的表

论文的 Table 1 列了当时几个大模型的实际参数量和训练词元数，Table 3 则给出了不同模型大小下 compute-optimal 的词元估计。把两张表对照着看，就像一份对整个行业的审计报告：

| 模型 | 参数量 | 实际训练词元 | Chinchilla 最优词元 |
|------|--------|-------------|-------------------|
| GPT-3 | 1750 亿 | 3000 亿 | 3.7 万亿 |
| Gopher | 2800 亿 | 3000 亿 | 5.9 万亿 |
| Jurassic-1 | 1780 亿 | 3000 亿 | 3.7 万亿 |
| MT-NLG | 5300 亿 | 2700 亿 | 11.0 万亿 |

发现规律了没有？所有模型训练数据都在 3000 亿词元左右。好像整个行业不约而同地认定「3000 亿够了」，然后把多余的算力全砸在了参数量上。

按 Chinchilla 的分析，GPT-3 应该训 3.7 万亿词元：是实际数据量的 12 倍多。Gopher 应该看将近 6 万亿。最夸张的是 MT-NLG，5300 亿参数的巨无霸，应该训 11 万亿词元：实际只喂了 2700 亿，差了 40 倍。

这些模型不是太小，是被饿着了。

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

## 6. 实锤：Chinchilla vs. Gopher

光说不练假把式。为了验证理论，团队训了 Chinchilla：一个 700 亿参数的模型，喂了 1.4 万亿词元。关键在于：Chinchilla 和 Gopher（2800 亿参数，3000 亿词元）用了完全一样的算力预算。同样的钱，只是分法不同。

结果一目了然。Chinchilla 虽然只有 Gopher 四分之一的参数量，在几乎所有基准测试上都赢了：

- **MMLU**（大规模多任务语言理解）：Chinchilla 67.6% vs. Gopher 60.0% vs. GPT-3 43.9%
- **阅读理解**（RACE-h）：Chinchilla 73.3% vs. Gopher 71.6%
- **常识推理**（HellaSwag）：Chinchilla 80.8% vs. Gopher 79.2%
- **BIG-bench**：Chinchilla 在大多数任务上优于 Gopher

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

一个参数量只有对手四分之一的模型，同样的花费，在几乎每项指标上都赢了：这不是理论推导，这是真刀真枪的对比。算力没有浪费，只是换了个花法：从堆参数，变成了喂数据。

打个比方：同样 100 万的教育预算，Gopher 的策略是「请一个超级天才，但只给他一本薄教材」；Chinchilla 的策略是「请一个聪明的学生，给他一整个图书馆」。后者考得更好。

## 7. 实际影响

Chinchilla 论文立刻改变了行业的做事方式。

**小模型更省钱。** 训练成本是一次性的，但用户每次提问，模型都要跑一遍：这个「推理成本」跟模型大小直接挂钩，而且是每一次请求都要付的。你可以理解为：训练费是买房，推理费是物业费。房子（模型）越大，物业费越贵，而且得一直交。一个 700 亿参数的模型，每次服务用户的成本只有 2800 亿模型的四分之一。如果小模型效果还更好，那就是双赢：质量更高，账单更低。

**数据成了瓶颈。** Chinchilla 之前，大家抢的是 GPU：你能搞到多少卡？Chinchilla 之后，大家抢的变成了数据：你从哪儿找几万亿高质量词元？这直接引发了全行业的数据争夺战：大规模网页抓取、数据集精选工程，以及后来的合成数据运动（让 AI 生成训练数据来训 AI）。

**LLaMA 时刻。** Meta 的 LLaMA（2023 年 2 月）可以说是 Chinchilla 缩放定律最直接的应用。LLaMA-13B（130 亿参数）在 1 万亿词元上训练，在大多数基准上超过了 GPT-3（1750 亿参数）：一个比你小十几倍的模型考得比你好，就因为人家书读得多。LLaMA-65B 在 1.4 万亿词元上训练，跟 Chinchilla 和 PaLM-540B 不相上下。Meta 在论文中明确引用了 Chinchilla，刻意选择更小的模型配更多的数据。

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

## 8. 几点观察

第一，这篇论文是一次优雅的纠偏。它没有否定 Kaplan 等人的工作，而是在同一个框架下发现了一个方法论上的差异（固定学习率调度），修正之后得出了不同的结论。损失函数 L̂(N, D) = E + A/N^α + B/D^β 是 Kaplan 公式的精化，不是推翻。科学最好的样子就是这样：有人认真做了一件事，另一个人更认真地做了一遍，领域就往前走了一步。

第二，论文最值得注意的不是数学：是理论和实践之间的鸿沟。整个行业的人都看到 3000 亿词元成了默认值，但没有人认真质疑过，直到这个团队把账算清楚。那些模型不是太小，是被饿着了。解决办法不是造更大的模型，而是喂更多的数据。

第三，等比例缩放的结论（a ≈ b ≈ 0.5）美在简洁。模型大小和数据量之间没有不对称性。有更多的算力？两边等量加。「下一块钱算力该花在哪？」Chinchilla 给出的答案不是继续单押参数量，而是让模型大小和训练数据近似等比例地一起增长。

第四，实际遗产是巨大的。Chinchilla 之前，通往更强 AI 的路是「做更大」。Chinchilla 之后，路变成了「训更好」。这一个转变让那些买不起最大参数量但能搞到大量数据的团队也有了机会。LLaMA、Mistral，以及整个开源大模型生态，都直接受益于这个洞察。

Kaplan 的论文说：更大的模型可预测地更强。Chinchilla 的论文说：没错，但你们大的方式不对。别囤参数了，多喂数据。

一篇论文给了行业放心做大的信心，另一篇教会了行业怎么把大做对。

---

**论文共读系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-CN/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神经网络进行序列到序列学习） — 编码器-解码器范式的确立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《Attention Is All You Need》](/zh-CN/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成为主角，Transformer 的诞生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-CN/posts/bert/)（BERT：用于语言理解的深度双向 Transformer 预训练） — 预训练范式的确立
- [《Scaling Laws for Neural Language Models》](/zh-CN/posts/scaling-laws-for-neural-language-models/)（神经语言模型的缩放定律） — 规模的数学
- [《Language Models are Few-Shot Learners》](/zh-CN/posts/language-models-are-few-shot-learners/)（语言模型是少样本学习者） — 更大的模型，更善于从上下文中诱发能力

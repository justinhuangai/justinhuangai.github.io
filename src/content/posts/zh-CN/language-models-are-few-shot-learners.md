---
title: "《Language Models are Few-Shot Learners》：GPT-3 与上下文学习"
date: "2026-02-11T16:22:54+08:00"
category: "Paper Reading"
description: 更大的模型，更善于从上下文中诱发能力，附真实 Python 代码
tags: [paper-reading, gpt-3, AI, LLM, python]
pinned: false
---

2020 年 5 月 28 日，OpenAI 在 arXiv（一个学术论文预印本网站，论文不用等期刊审稿就能直接发布）上传了一篇 75 页的论文：[《Language Models are Few-Shot Learners》](/papers/2005.14165v4.pdf)（语言模型是少样本学习者）。

作者有 31 人，全部来自 OpenAI。第一作者 Tom B. Brown，其余包括 Jared Kaplan（缩放定律的核心研究者）、Alec Radford（GPT-1 和 GPT-2 的主要设计者）、Ilya Sutskever（OpenAI 联合创始人兼首席科学家）、Dario Amodei（OpenAI 研究副总裁）等。

这份作者名单后来分化出了几家最重要的 AI 公司：Dario Amodei 和 Jared Kaplan 离开 OpenAI 创立了 Anthropic，Ilya Sutskever 后来也联合创立了 Safe Superintelligence Inc.（SSI）。

论文的核心主张很直接：把语言模型做大，大到 1750 亿参数，它就能在不更新任何权重的情况下，仅靠几个示例就完成各种任务：有时甚至逼近经过专门微调的模型。

这不是任务级微调，而是在固定参数下，通过上下文适配任务的能力：论文称之为**上下文学习（in-context learning）**。

## 0. 先认几个词

如果你对 GPT-3 这类模型的工作方式还没有概念，先记住下面几个词就够了：

- `语言模型`：给它一段前文，它的基本工作就是预测下一个词。
- `参数量`：模型里可学习的数字总数。你可以粗略把它理解成模型的“脑容量”。
- `prompt / 提示词`：你喂给模型看的任务说明、示例和输入。
- `上下文窗口`：模型一次能看到多少文本的容量。例子太多，塞不进去，就没法一起看。
- `few-shot / one-shot / zero-shot`：分别指给多个例子、给一个例子、完全不给例子。
- `in-context learning / 上下文学习`：不改模型参数，只靠 prompt 里的说明和例子，就让模型临时学会怎么做任务。

## 1. 要解决什么问题

[BERT](/zh-CN/posts/bert/) 确立的「预训练 + 微调」范式在 2020 年已经是主流做法。效果很好，但论文指出了三个根本问题。

第一，每个新任务仍然需要一个标注数据集。标注数据的获取成本高，且很多实际任务根本没有对应的标注集。

第二，微调后的模型在测试基准上的表现，不一定反映真实泛化能力。模型可能只是学到了训练数据中的虚假相关性（spurious correlations）：在基准集里得分很高，但换个分布就崩了。

第三，人类不是这样学习的。人类看一两个例子，听一句自然语言指令，就能完成新任务。而当时的 NLP 系统，每个新任务都需要成千上万条标注数据来微调。

论文的出发点是：如果模型足够大，它在预训练阶段积累的知识是否足以让它直接「读懂」任务描述和少量示例，然后给出答案？

## 2. 核心想法：不更新参数，只给提示

GPT-3 的评估方式和之前所有大模型都不一样。它定义了三种设置，全部不涉及梯度更新：

**少样本（Few-Shot）**：给模型一段任务描述，加上 10 到 100 个示例（具体数量取决于上下文窗口能装多少），然后让它完成新的输入。不更新权重，不做反向传播。

**单样本（One-Shot）**：只给一个示例。这最接近人类学习新任务的方式：有人给你演示一次，你就上手。

**零样本（Zero-Shot）**：连示例都不给，只有一句自然语言指令。这是最难的设置，但也是最实用的：如果模型真的「理解」了任务本身，它不应该需要任何例子。

```python showLanguage
from dataclasses import dataclass
from typing import Union


@dataclass
class ZeroShot:
    instruction: str
    prompt: str


@dataclass
class OneShot:
    instruction: str
    example: tuple[str, str]
    prompt: str


@dataclass
class FewShot:
    instruction: str
    examples: list[tuple[str, str]]
    prompt: str


EvalSetting = Union[ZeroShot, OneShot, FewShot]


def build_prompt(setting: EvalSetting) -> str:
    if isinstance(setting, ZeroShot):
        return f"{setting.instruction}\n{setting.prompt}"

    if isinstance(setting, OneShot):
        example_input, example_output = setting.example
        return f"{setting.instruction}\n{example_input} {example_output}\n{setting.prompt}"

    lines = [setting.instruction]
    lines.extend(f"{example_input} {example_output}" for example_input, example_output in setting.examples)
    lines.append(setting.prompt)
    return "\n".join(lines)
```

论文把这种能力叫做**上下文学习**：模型在预训练时，从海量文本中隐式地学到了各种任务的模式；推理时，示例被拼接进上下文，模型在前向传播的过程中「识别」出当前任务是什么，然后完成它。论文用「元学习」来描述这个过程：预训练是外循环，上下文学习是内循环。

这和微调的区别是根本性的。微调修改模型参数来适应任务，上下文学习不修改任何东西：同一个模型，同一组权重，只靠输入文本的不同，就能切换任务。

## 3. 模型架构与规模

GPT-3 的架构本身没有新发明。它和 GPT-2 一样，就是 [Transformer](/zh-CN/posts/attention-is-all-you-need/) 的解码器部分，一层层堆起来。改动只有一处：在 Transformer 层中交替使用稠密注意力和局部带状稀疏注意力（来自 Sparse Transformer）。

真正不同的是规模。论文训练了 8 个不同大小的模型，参数量跨越三个数量级：

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 |
|------|--------|------|----------|-----------|
| GPT-3 Small | 1.25 亿 | 12 | 768 | 12 |
| GPT-3 Medium | 3.5 亿 | 24 | 1024 | 16 |
| GPT-3 Large | 7.6 亿 | 24 | 1536 | 16 |
| GPT-3 XL | 13 亿 | 24 | 2048 | 24 |
| GPT-3 2.7B | 27 亿 | 32 | 2560 | 32 |
| GPT-3 6.7B | 67 亿 | 32 | 4096 | 32 |
| GPT-3 13B | 130 亿 | 40 | 5140 | 40 |
| **GPT-3 175B** | **1750 亿** | **96** | **12288** | **96** |

1750 亿参数，96 层，96 个注意力头，隐藏维度 12288。上下文窗口 2048 个词元。这个规模在当时是前所未见的：比 GPT-2（15 亿参数）大了 100 多倍。

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT3Config:
    n_params: int
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_ctx: int


def gpt3_175b() -> GPT3Config:
    return GPT3Config(
        n_params=175_000_000_000,
        n_layers=96,
        d_model=12_288,
        n_heads=96,
        d_head=128,
        d_ff=49_152,
        n_ctx=2_048,
    )
```

论文训练这些模型的目的很明确：验证缩放定律（scaling laws）。之前 Kaplan 等人的研究（就是这篇论文的共同作者之一）已经表明，语言模型的损失和参数量之间存在平滑的幂律关系。GPT-3 把这个假设推到了 1750 亿参数的规模，看看上下文学习能力是否也遵循同样的规律。

答案是肯定的：模型越大，少样本学习的提升越陡。零样本性能随模型规模稳步上升，少样本性能的上升速度更快。这意味着大模型不只是「更准」，它们在利用上下文信息的效率上也更高。

## 4. 训练数据

GPT-3 在大约 3000 亿个词元上训练，数据来自五个来源：

| 数据集 | 词元数 | 训练占比 |
|--------|--------|----------|
| Common Crawl（过滤后） | 4100 亿 | 约 60% |
| WebText2 | 190 亿 | 约 22% |
| Books1 | 120 亿 | 约 8% |
| Books2 | 550 亿 | 约 8% |
| 英文 Wikipedia | 30 亿 | 约 3% |

注意一个关键细节：数据集的采样比例和它们的大小不成正比。质量更高的数据集（WebText2、Books、Wikipedia）被过采样了：WebText2 在训练中被看了 2.9 遍，Wikipedia 被看了 3.4 遍，而 Common Crawl 连一遍都没看完（0.44 遍）。论文有意用少量过拟合的代价，换取更高质量的训练信号。

Common Crawl 的原始数据有 45TB，经过三步处理：（1）基于与高质量参考语料的相似度做过滤；（2）文档级模糊去重；（3）混入已知的高质量数据集来增加多样性。过滤后剩下 570GB，约 4100 亿词元。

所有模型在 V100 GPU 上训练，使用微软提供的高带宽集群。

## 5. 实验结果

论文在二十多个数据集上做了评估，覆盖 9 大类任务。以下是几个关键结果。

**语言建模**：在 Penn Tree Bank 上，GPT-3 少样本困惑度（perplexity，衡量模型对文本的「意外程度」，越低越好）达到 20.50，刷新了当时的记录。在 LAMBADA（需要根据长距离上下文预测最后一个词）上，零样本准确率 76.2%，少样本 86.4%，大幅超过之前的最好结果。

**翻译**：GPT-3 从未被专门训练过翻译，但在法语→英语翻译上，少样本 BLEU 分数达到 32.6，超过了无监督神经机器翻译的最好结果。不过英语→法语方向（25.2 BLEU）和微调模型的差距仍然很大。一个有趣的发现：GPT-3 翻译成英语的能力明显强于从英语翻译出去，这和训练数据以英语为主有直接关系。

**闭卷问答**：在 TriviaQA 上，少样本准确率（exact match）71.2%，超过了同一闭卷设置下经过微调的模型。模型不看任何参考文档，纯靠参数里存储的知识回答问题。

**SuperGLUE**：在这个综合基准上，GPT-3 的少样本表现已经接近一些经过微调的强基线，但仍落后于当时最强的专门微调系统。

**合成任务**：论文还设计了一些专门测试上下文学习能力的新任务。比如给模型几个「造新词」的例子（定义一个不存在的词，然后用它造句），GPT-3 能正确地学会并使用这个新词。再比如三位数加法，少样本准确率接近 100%（两位数加法也几乎完美），但四五位数时急剧下降。

```python showLanguage
from typing import Callable, Protocol


class AutoregressiveModel(Protocol):
    def forward(self, tokens: list[int]) -> list[list[float]]:
        ...


def in_context_learning(
    model: AutoregressiveModel,
    examples: list[tuple[str, str]],
    query: str,
    tokenize: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    sample_from: Callable[[list[float]], int],
    eos_token: int,
) -> str:
    prompt_lines = [f"{example_input} {example_output}" for example_input, example_output in examples]
    prompt_lines.append(query)
    prompt = "\n".join(prompt_lines)

    context = tokenize(prompt)
    output_tokens: list[int] = []

    while True:
        logits = model.forward(context)
        next_token = sample_from(logits[-1])
        if next_token == eos_token:
            break
        output_tokens.append(next_token)
        context.append(next_token)

    return decode(output_tokens)
```

## 6. 数据污染问题

论文在第四章花了大量篇幅讨论一个棘手的问题：训练数据和测试数据的重叠。

GPT-3 的训练数据包含大量互联网文本，而很多测试基准的内容也在互联网上公开存在。这意味着模型可能在训练时就「看过」了测试题。论文团队尝试在训练前移除这些重叠，但由于一个处理流程中的 bug，部分重叠没有被完全清除。而重新训练一遍的成本太高，不现实。

他们的做法是：为每个基准构建一个「干净子集」（移除所有和训练数据有 13-gram 重叠的样本），然后对比模型在完整集和干净子集上的表现。结论是：大多数基准上，污染对结果的影响很小。但 PIQA 和 Winograd 两个数据集存在可疑的表现下降，论文对这些结果加了星号标注。

这种诚实在当时相当罕见。多数论文对数据污染问题避而不谈。GPT-3 不仅主动调查，还开发了系统化的检测工具。这本身就是对后续研究的一个贡献。

## 7. 局限性

论文在第五章对自身局限性的讨论相当坦率。

**文本连贯性**：GPT-3 在文档级别仍然会出现语义重复、自相矛盾、甚至生成无意义句子的情况。生成质量虽然比 GPT-2 好了很多，但长文本的连贯性仍然不够。

**常识物理**：GPT-3 对「把奶酪放进冰箱，它会融化吗？」这类常识物理问题表现不佳。它能处理语言层面的推理，但对物理世界的理解仍然是肤浅的。

**单向性的代价**：作为自回归模型，GPT-3 只能从左往右看。论文承认，在需要双向上下文的任务上（比如判断两个句子里同一个词的含义是否相同），GPT-3 的少样本表现不如经过微调的双向模型。这说明在 GPT-3 的自回归设定下，这类任务并不是它的强项；单向建模目标本身会带来结构性偏好。

**采样效率**：GPT-3 在预训练阶段看了约 3000 亿个词元，远超人类一生接触的文本量。论文明确指出，即使少样本学习在推理时很高效，预训练的数据需求仍然巨大。

**推理成本**：1750 亿参数的模型，推理成本高且不方便部署。论文提到蒸馏（distillation，用大模型的输出来训练小模型）是一个可能的方向，但在千亿参数量级上还没有尝试过。

## 8. 社会影响

论文用了整整一个章节（第六章）讨论社会影响，涵盖三个方面。

**滥用风险**：GPT-3 生成的新闻文章，人类评估者的识别准确率接近随机猜测（约 52%）。模型越强，生成的虚假文本越难辨别。论文团队表示已经在监控论坛和聊天群，追踪恶意使用的趋势。

**偏见**：论文用大量实验测试了 GPT-3 在性别、种族和宗教方面的偏见。例如，在职业-性别关联测试中，GPT-3 更倾向于将「nurse」和女性关联、将「banker」和男性关联。在宗教-情感关联中，「Islam」更多地与暴力相关词共现。论文承认这些偏见来自训练数据，但没有给出解决方案。

**能源消耗**：训练 GPT-3 需要大量算力，论文引用了估算数据但没有公布具体的能耗数字。不过论文指出，一旦训练完成，模型可以被多次使用到不同任务上，比为每个任务单独训练模型更节能。

## 9. 几点观察

读完这篇论文，有几个感受。

第一，GPT-3 展示了一件事：规模能把上下文学习推到可用阈值。1750 亿参数的模型不只是「更大的 GPT-2」，它在上下文学习上的表现比小模型强出了一个量级。模型在没有任何参数更新的情况下，仅靠上下文中的几个示例就能完成新任务。这种能力不是显式手工设计出来的，而是在规模扩大过程中逐步增强，到 GPT-3 这个量级才第一次变得足够清晰、足够实用。BERT 证明了预训练的价值，GPT-3 证明了规模的价值。

第二，论文的写作方式值得注意。31 个作者，75 页篇幅，用了大量实验来回答一个简单的问题：更大的模型是否更善于利用少量示例？他们没有回避局限性：文本连贯性、常识推理、数据污染、偏见：全部正面讨论。这种严谨程度，在后来的大模型论文中反而越来越少见了。

第三，这篇论文的作者列表就是一部 AI 行业分裂史。Dario Amodei 和 Jared Kaplan 后来创立了 Anthropic（Claude 的开发商），Ilya Sutskever 后来离开 OpenAI 创立了 SSI。这些人在 2020 年还在同一个团队里合作写论文，两年后就走向了不同的方向。论文里关于社会影响和安全风险的讨论，也许正是后来分歧的伏笔。

第四，从技术演进的角度看，GPT-3 是从「预训练 + 微调」到「预训练 + 提示」的转折点。BERT 的思路是：先学通用知识，再在每个任务上微调参数。GPT-3 说：如果模型够大，微调这一步可以省掉：直接用自然语言告诉模型你要做什么。这个思路后来演化成了 ChatGPT、Claude 等产品的核心交互范式：用户用自然语言提问，模型直接回答。

从 Seq2Seq 的编码-解码，到 [Bahdanau 注意力](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)的「该看哪里」，到 [Transformer](/zh-CN/posts/attention-is-all-you-need/) 的「所有位置同时看」，到 [BERT](/zh-CN/posts/bert/) 的「先学再调」，再到 GPT-3 的「大到不用调」：每一步都在减少人工干预，增加模型自主完成任务的能力。

GPT-3 不是终点。但它第一次让人们认真思考一个问题：如果继续把模型做大，还会涌现出什么？

这个问题的答案，就是后来发生的一切。

---

**论文共读系列**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-CN/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神经网络进行序列到序列学习） — 编码器-解码器范式的确立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-CN/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通过联合学习对齐与翻译实现神经机器翻译） — 注意力机制的起源
- [《Attention Is All You Need》](/zh-CN/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成为主角，Transformer 的诞生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-CN/posts/bert/)（BERT：用于语言理解的深度双向 Transformer 预训练） — 预训练范式的确立
- [《Scaling Laws for Neural Language Models》](/zh-CN/posts/scaling-laws-for-neural-language-models/)（神经语言模型的缩放定律） — 规模的数学：为什么更大的模型可预测地更好
- [《Training Compute-Optimal Large Language Models》](/zh-CN/posts/training-compute-optimal-large-language-models/)（训练计算最优的大语言模型） — 怎样花算力最划算

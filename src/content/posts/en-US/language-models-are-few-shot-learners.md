---
title: "Language Models are Few-Shot Learners: GPT-3 and In-Context Learning"
date: "2026-02-11T16:22:54+08:00"
category: "Paper Reading"
description: Larger models, better at eliciting abilities from context, with real Python code examples
tags: [paper-reading, gpt-3, AI, LLM, python]
pinned: false
---

GPT-3's question was not simply whether models could keep getting larger. It asked whether task adaptation had to update parameters at all. In the BERT paradigm, every new task wanted fine-tuning; that meant more data, training pipelines, and deployed model variants.

[*Language Models are Few-Shot Learners*](/papers/2005.14165v4.pdf) moved task adaptation from parameter updates into context. The weights stay fixed. Instructions and a few examples enter the prompt. The task is organized at inference time.

## 1. The Problem

The "pre-train + fine-tune" paradigm established by [BERT](/posts/bert/) was already mainstream by 2020. It worked well, but the paper identified three fundamental issues.

First, every new task still requires a labeled dataset. Labeled data is expensive to collect, and many real-world tasks have no corresponding labeled set at all.

Second, a fine-tuned model's performance on test benchmarks does not necessarily reflect genuine generalization. The model may have simply learned spurious correlations in the training data — scoring high on the benchmark but collapsing under distribution shift.

Third, humans do not learn this way. A human can see one or two examples, hear a natural language instruction, and handle a new task. The NLP systems of that era required thousands of labeled samples to fine-tune for each new task.

The paper's starting point: if a model is large enough, can the knowledge it accumulates during pre-training allow it to directly "read" a task description and a few examples, then produce the answer?

## 2. The Core Idea: No Parameter Updates, Just Prompts

GPT-3's evaluation methodology differed from every large model before it. It defined three settings, none of which involve gradient updates:

**Few-Shot**: give the model a task description plus 10 to 100 examples (the exact number depends on how many fit in the context window), then have it complete a new input. No weight updates, no backpropagation.

**One-Shot**: give just one example. This most closely mirrors how humans learn a new task — someone demonstrates once, and you take it from there.

**Zero-Shot**: no examples at all, just a natural language instruction. This is the hardest setting, but also the most practical — if the model truly "understands" the task itself, it should not need any examples.

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

The paper calls this capability **in-context learning**: during pre-training, the model implicitly learns patterns for a wide variety of tasks from massive amounts of text; at inference time, examples are concatenated into the context, and the model "recognizes" the current task during the forward pass and completes it. The paper describes this process using the language of "meta-learning" — pre-training is the outer loop, in-context learning is the inner loop.

The distinction from fine-tuning is fundamental. Fine-tuning modifies model parameters to fit a task. In-context learning modifies nothing — the same model, the same weights, switching tasks purely by varying the input text.

## 3. Model Architecture and Scale

GPT-3's architecture is not a new invention. Like GPT-2, it is just the decoder portion of the [Transformer](/posts/attention-is-all-you-need/), stacked layer by layer. The only modification: alternating between dense attention and local banded sparse attention (from Sparse Transformer) within the Transformer layers.

What is genuinely different is the scale. The paper trained 8 models of varying sizes, spanning three orders of magnitude in parameter count:

| Model | Parameters | Layers | Hidden Size | Attention Heads |
|-------|-----------|--------|-------------|-----------------|
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 |
| GPT-3 Large | 760M | 24 | 1536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 |
| GPT-3 13B | 13B | 40 | 5140 | 40 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** |

175 billion parameters, 96 layers, 96 attention heads, hidden dimension of 12288. Context window of 2048 tokens. This scale was unprecedented at the time — over 100 times larger than GPT-2's 1.5 billion parameters.

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

The purpose of training these models was explicit: to validate scaling laws. Earlier work by Kaplan et al. (one of this paper's co-authors) had already shown a smooth power-law relationship between language model loss and parameter count. GPT-3 pushed that hypothesis to 175 billion parameters to see whether in-context learning ability follows the same pattern.

The answer is yes: the larger the model, the steeper the improvement in few-shot learning. Zero-shot performance rises steadily with scale, and few-shot performance rises even faster. This means larger models are not just "more accurate" — they are also more efficient at leveraging contextual information.

## 4. Training Data

GPT-3 was trained on approximately 300 billion tokens, drawn from five sources:

| Dataset | Tokens | Training Mix |
|---------|--------|--------------|
| Common Crawl (filtered) | 410B | ~60% |
| WebText2 | 19B | ~22% |
| Books1 | 12B | ~8% |
| Books2 | 55B | ~8% |
| English Wikipedia | 3B | ~3% |

Note a key detail: the sampling proportions are not proportional to dataset size. Higher-quality datasets (WebText2, Books, Wikipedia) were oversampled — WebText2 was seen 2.9 times during training, Wikipedia 3.4 times, while Common Crawl was not even seen once in full (0.44 epochs). The paper deliberately traded a small amount of overfitting for higher-quality training signal.

The raw Common Crawl data was 45TB. It went through three processing steps: (1) filtering based on similarity to high-quality reference corpora; (2) document-level fuzzy deduplication; (3) mixing in known high-quality datasets for diversity. After filtering, 570GB remained — roughly 410 billion tokens.

All models were trained on V100 GPUs using a high-bandwidth cluster provided by Microsoft.

## 5. Experimental Results

The paper evaluated across more than twenty datasets, covering 9 major task categories. Here are several key results.

**Language Modeling**: on Penn Tree Bank, GPT-3 few-shot perplexity (a measure of how "surprised" the model is by text — lower is better) reached 20.50, setting a new record. On LAMBADA (which requires predicting the final word based on long-range context), zero-shot accuracy was 76.2%, few-shot 86.4%, substantially surpassing the previous best.

**Translation**: GPT-3 was never specifically trained for translation, yet on French-to-English, few-shot BLEU score reached 32.6, exceeding the best unsupervised neural machine translation result. However, English-to-French (25.2 BLEU) still lagged significantly behind fine-tuned models. The asymmetry is revealing: GPT-3 is noticeably better at translating into English than out of it, directly reflecting the English-heavy composition of its training data.

**Closed-Book QA**: on TriviaQA, few-shot accuracy (exact match) was 71.2%, surpassing fine-tuned models under the same closed-book setting. The model references no documents — it answers purely from knowledge stored in its parameters.

**SuperGLUE**: on this comprehensive benchmark, GPT-3's few-shot performance approached some strong fine-tuned baselines, but still trailed the strongest dedicated fine-tuned systems of the time.

**Synthetic Tasks**: the paper also designed novel tasks specifically to test in-context learning. For example, giving the model a few examples of "made-up words" (defining a nonexistent word and then using it in a sentence), GPT-3 could correctly learn and use the new word. Three-digit addition was nearly 100% accurate in few-shot (two-digit was also near-perfect), but accuracy dropped sharply at four and five digits.

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

## 6. Data Contamination

The paper devotes substantial space in Section 4 to a thorny issue: overlap between training data and test data.

GPT-3's training data includes vast amounts of internet text, and many test benchmarks are publicly available on the internet. This means the model may have "seen" the test questions during training. The team attempted to remove these overlaps before training, but due to a bug in the processing pipeline, some overlaps were not fully cleaned. Retraining from scratch was too expensive to be practical.

Their approach: for each benchmark, construct a "clean subset" (removing all samples with 13-gram overlaps against the training data), then compare model performance on the full set versus the clean subset. The conclusion: for most benchmarks, contamination had minimal impact on results. However, PIQA and Winograd showed suspicious performance drops, and the paper flagged those results with asterisks.

This level of honesty was quite rare at the time. Most papers avoid discussing data contamination entirely. GPT-3 not only proactively investigated the issue but also developed systematic detection tools. That itself is a contribution to subsequent research.

## 7. Limitations

The paper's discussion of its own limitations in Section 5 is remarkably candid.

**Text Coherence**: GPT-3 still exhibits semantic repetition, self-contradiction, and even nonsensical sentences at the document level. Generation quality is much better than GPT-2, but long-form coherence remains insufficient.

**Commonsense Physics**: GPT-3 performs poorly on commonsense physics questions like "If you put cheese in a refrigerator, will it melt?" It can handle linguistic reasoning, but its understanding of the physical world remains superficial.

**The Cost of Unidirectionality**: as an autoregressive model, GPT-3 can only look left-to-right. The paper acknowledges that on tasks requiring bidirectional context (such as determining whether the same word in two sentences carries the same meaning), GPT-3's few-shot performance falls short of fine-tuned bidirectional models. This indicates that such tasks are not GPT-3's strength under its autoregressive setup; the unidirectional modeling objective introduces a structural bias.

**Sample Efficiency**: GPT-3 saw approximately 300 billion tokens during pre-training, far exceeding the amount of text a human encounters in a lifetime. The paper explicitly notes that even though few-shot learning is efficient at inference time, the data requirements for pre-training remain enormous.

**Inference Cost**: a 175-billion-parameter model is expensive to run and difficult to deploy. The paper mentions distillation (using a large model's outputs to train a smaller model) as a possible direction, but notes it has not yet been attempted at the hundred-billion-parameter scale.

## 8. Societal Impact

The paper dedicates an entire section (Section 6) to societal impact, covering three areas.

**Misuse Risks**: human evaluators could identify GPT-3-generated news articles at only about chance level (~52% accuracy). The stronger the model, the harder its fabricated text is to detect. The team reported that they were monitoring forums and chat groups to track trends in malicious use.

**Bias**: the paper ran extensive experiments testing GPT-3's biases across gender, race, and religion. For example, in occupation-gender association tests, GPT-3 was more likely to associate "nurse" with female and "banker" with male. In religion-sentiment associations, "Islam" co-occurred more frequently with violence-related words. The paper acknowledges these biases originate from the training data but offers no solution.

**Energy Consumption**: training GPT-3 requires massive compute, and the paper cites estimates but does not disclose specific energy figures. However, it points out that once trained, the model can be applied to many different tasks, making it more energy-efficient than training a separate model for each task.

## 9. What This Paper Changed

GPT-3's sharpest lesson is: **task adaptation can move from parameter updates into context.**

The paper did not merely say that larger models are stronger. It changed the interface between people and models. Previously, making a model handle a new task usually meant collecting data, fine-tuning parameters, and deploying another version. GPT-3 showed another route: keep the weights fixed, put instructions, examples, and input into context, and let the model adapt at inference time.

This did not end fine-tuning. It moved part of the adaptation cost from training into context design. A prompt stopped being just input text; it became a lightweight task program. The context window stopped being only a capacity number; it became a temporary workspace.

The next time you look at a large model, do not ask only how many parameters it has. Ask where task adaptation lives: in the weights, in the context, or in external tools and workflows. That question is closer to the product than model size is.

---

**Paper Reading Series**

- [*Sequence to Sequence Learning with Neural Networks*](/posts/sequence-to-sequence-learning-with-neural-networks/) — Establishing the encoder-decoder paradigm
- [*Neural Machine Translation by Jointly Learning to Align and Translate*](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) — The origin of attention
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — Attention takes center stage: the birth of the Transformer
- [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](/posts/bert/) — Establishing the pre-training paradigm
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — The mathematics of scale
- [*Training Compute-Optimal Large Language Models*](/posts/training-compute-optimal-large-language-models/) — How to spend your compute budget wisely

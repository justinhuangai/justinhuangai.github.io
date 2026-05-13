---
title: "Paper Reading: Training Compute-Optimal Large Language Models"
date: "2026-03-11T16:58:04+08:00"
category: "Paper Reading"
description: The Chinchilla paper — why most large models were undertrained, and how to spend your compute budget wisely, with real Python code examples
tags: [paper-reading, chinchilla, scaling-laws, AI, LLM, python]
pinned: false
---

On March 29, 2022, a team of researchers from DeepMind uploaded a paper to arXiv (a preprint server where researchers can publish papers without waiting for journal peer review): [*Training Compute-Optimal Large Language Models*](/papers/2203.15556v1.pdf).

The first author is Jordan Hoffmann, with co-authors including Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, and many others — all at DeepMind at the time. Arthur Mensch would later co-found Mistral AI, one of Europe's most prominent AI companies.

The paper is often called the "Chinchilla paper," after the 70-billion-parameter model the team trained to validate their findings. That name stuck — not the paper's title, but the animal. In AI circles, "Chinchilla scaling" became shorthand for the paper's central claim.

And that claim was simple, bold, and uncomfortable for most of the industry: **many of the biggest language models of 2022 were not "too small" — they were significantly undertrained given their compute budgets.**

## 1. The Question

By early 2022, the AI community had internalized a clear lesson from [Kaplan et al. (2020)](/posts/scaling-laws-for-neural-language-models/): bigger models are predictably better. The scaling laws paper had shown that performance follows power laws, and that for a given compute budget, you should make the model as large as possible.

The industry took that advice to heart. By spring 2022, GPT-3 had 175 billion parameters trained on 300 billion tokens. DeepMind's own Gopher had 280 billion parameters trained on 300 billion tokens. Google would soon release PaLM with 540 billion parameters. The trend was clear: crank up the parameter count.

But there was a problem hiding in plain sight. Kaplan et al. had concluded that when you scale compute, most of the budget should go to model size (N ∝ C^0.73) and relatively little to training data (D ∝ C^0.27). This meant: make the model huge, train it on a moderate amount of data.

Hoffmann's team asked a simple question: is that actually right?

## 2. Three Independent Approaches, One Answer

What makes this paper unusually convincing is its methodology. The team did not rely on a single experiment. They approached the same question from three completely independent angles, and all three converged on the same answer.

**Approach 1: Fix the compute, vary the split.** They trained over 400 models ranging from 70 million to over 16 billion parameters, each with a different allocation between model size and training data, but with the same total compute. For each compute level, they found which model size minimized loss.

**Approach 2: IsoFLOP profiles.** They trained models of 9 different sizes (from 70M to 10B parameters) on varying amounts of data, specifically designed so each group of runs used approximately the same total compute. Then they fit curves to find the optimal model size for each compute level.

**Approach 3: Fit a parametric loss function.** They fit the following equation to all their training runs:

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

Where E is the irreducible loss (the entropy of natural language — no model can do better), A/N^α captures the model-size bottleneck, and B/D^β captures the data bottleneck. From the fitted parameters, they derived optimal N and D as functions of compute.

All three approaches agreed:

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

The exponents a ≈ b ≈ 0.5 mean that as compute grows, model size and training data should scale at approximately the same rate. When compute grows 10x, both should increase by roughly 3.2x; when compute doubles, both increase by roughly 1.4x. In other words, for every doubling of model size, the number of training tokens should also double. This directly contradicts Kaplan et al., who said compute should be spent primarily on model size.

## 3. Why Kaplan Got It Wrong

This is not a matter of one team getting it wrong. Both teams did rigorous work. The difference lies in experimental setup, which ultimately led to different optimal-allocation conclusions.

Kaplan et al. used a fixed learning rate schedule that did not adjust for training duration. When you train a model for more steps without adjusting the learning rate schedule, performance suffers — not because the model is inherently worse, but because the optimization is suboptimal. This made long training runs look less effective than they actually are, biasing the results toward larger models trained for fewer steps.

Hoffmann's team adjusted the learning rate schedule for each training run, ensuring each configuration got a fair shot. When you do this, training longer on more data turns out to be much more valuable than Kaplan's numbers suggested.

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

## 4. The Parametric Loss Function

The paper's Approach 3 deserves a closer look because it gives a complete mathematical model of performance:

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

Where the fitted constants are:

- E = 1.69 — the irreducible loss (entropy of natural language)
- A = 406.4, α = 0.34 — the model-size term
- B = 410.7, β = 0.28 — the data term

The structure of this equation is worth studying. Loss has three components: a floor you can never get below (E), a penalty for having too few parameters (A/N^α), and a penalty for having too little data (B/D^β). The model-size penalty and data penalty are additive — they compete for your attention and your compute budget.

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

## 5. The Damning Table

The paper's Table 1 lists the actual parameter counts and training tokens for several well-known models. Table 3 gives the compute-optimal token estimates for different model sizes. Put the two tables side by side, and you get something that reads like an audit of the entire industry:

| Model | Parameters | Tokens Used | Chinchilla-Optimal Tokens |
|-------|-----------|-------------|---------------------------|
| GPT-3 | 175B | 300B | 3.7T |
| Gopher | 280B | 300B | 5.9T |
| Jurassic-1 | 178B | 300B | 3.7T |
| MT-NLG | 530B | 270B | 11.0T |

Every single model was trained on roughly 300 billion tokens. But according to Chinchilla's analysis, GPT-3 should have been trained on 3.7 trillion tokens — more than 12 times what it actually saw. Gopher should have seen nearly 6 trillion. MT-NLG, the largest of the bunch at 530 billion parameters, should have been trained on 11 trillion tokens — 40 times its actual training data.

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

The pattern is striking. The entire industry had settled on roughly the same amount of training data — around 300 billion tokens — regardless of model size. It was as if everyone had decided that 300B tokens was "enough" and poured all additional compute into making models bigger. Chinchilla says this was exactly backwards.

## 6. The Proof: Chinchilla vs. Gopher

To validate their theory, the team trained Chinchilla: a 70-billion-parameter model on 1.4 trillion tokens. Chinchilla used the same compute budget as Gopher (280 billion parameters, 300 billion tokens) — the same total training cost, just allocated differently.

The result was decisive. Chinchilla outperformed Gopher on nearly every benchmark, despite being 4 times smaller:

- **MMLU** (Massive Multitask Language Understanding): Chinchilla 67.6% vs. Gopher 60.0% vs. GPT-3 43.9%
- **Reading comprehension** (RACE-h): Chinchilla 73.3% vs. Gopher 71.6%
- **Common sense** (HellaSwag): Chinchilla 80.8% vs. Gopher 79.2%
- **BIG-bench**: Chinchilla outperformed Gopher on the majority of tasks

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

A model that is 4 times smaller beating the larger model on nearly every benchmark — using the same compute — is a powerful demonstration. The compute was not wasted; it was simply redirected from parameters to data.

## 7. The Practical Consequences

The Chinchilla paper had immediate, concrete consequences for the industry.

**Smaller models are cheaper to use.** Training cost is a one-time expense, but inference cost — the cost of actually running the model to generate text — scales with model size, every single time a user sends a query. A 70B model is 4x cheaper to serve than a 280B model. If the smaller model performs better, the win is double: better quality at lower cost.

**Data became the bottleneck.** Before Chinchilla, the limiting factor was compute: how many GPUs can you get? After Chinchilla, the limiting factor shifted to data: where do you find trillions of high-quality tokens? This sparked an industry-wide scramble for training data — web scraping at massive scale, dataset curation efforts, and eventually the synthetic data movement.

**The LLaMA moment.** Meta's LLaMA (February 2023) was arguably the most direct application of Chinchilla scaling. LLaMA-13B, trained on 1 trillion tokens, outperformed GPT-3 (175B) on most benchmarks. LLaMA-65B, trained on 1.4 trillion tokens, was competitive with Chinchilla and PaLM-540B. Meta explicitly cited the Chinchilla paper and deliberately trained smaller models on far more data than earlier conventions would have suggested.

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

## 8. My Takeaways

First, this paper is a correction — and a graceful one. It takes Kaplan et al.'s framework, identifies a methodological flaw (fixed learning rate schedules), fixes it, and arrives at a different answer. It does not dismiss the earlier work; it builds on it. The parametric loss function L̂(N, D) = E + A/N^α + B/D^β is a refinement of Kaplan's formulation, not a replacement. Science at its best is exactly this: someone does careful work, someone else does more careful work, and the field moves forward.

Second, the paper's most surprising finding is not the math — it is the gap between theory and practice. Everyone in the industry could see that 300 billion tokens was becoming a default. Nobody questioned it seriously until this team ran the numbers. The models were not small; they were starved. The solution was not to build bigger — it was to feed more.

Third, the equal-scaling result (a ≈ b ≈ 0.5) is beautiful in its simplicity. There is no asymmetry between model size and data. If you have more compute, scale both equally. No complicated allocation strategy needed. "Where should I spend my next dollar of compute?" Chinchilla's answer is not to keep betting on parameter count alone, but to let model size and training data grow at approximately the same rate.

Fourth, the practical legacy is enormous. Before Chinchilla, the path to better AI was "make it bigger." After Chinchilla, the path became "train it better." This one shift made powerful models accessible to organizations that could not afford the largest parameter counts but could curate large datasets. LLaMA, Mistral, and the entire open-source LLM ecosystem owe a direct debt to this insight.

The Kaplan paper said: bigger models are predictably better. The Chinchilla paper said: yes, but you have been making them big in the wrong way. Stop hoarding parameters. Start feeding data.

One paper gave the industry permission to scale. The other taught it how.

---

**Paper Reading Series**

- [*Sequence to Sequence Learning with Neural Networks*](/posts/sequence-to-sequence-learning-with-neural-networks/) — Establishing the encoder-decoder paradigm
- [*Neural Machine Translation by Jointly Learning to Align and Translate*](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) — The origin of attention
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — Attention takes center stage: the birth of the Transformer
- [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](/posts/bert/) — Establishing the pre-training paradigm
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — The mathematics of scale
- [*Language Models are Few-Shot Learners*](/posts/language-models-are-few-shot-learners/) — Larger models, better at eliciting abilities from context

---
title: "Technical Report Reading: Attention Residuals"
date: "2026-03-19T16:49:27+08:00"
category: "Technical Report Reading"
description: "A reading of Kimi Team's Attention Residuals technical report: why residual connections should become attention-like too, and how Full AttnRes / Block AttnRes turn that idea into a trainable, deployable system"
tags: [technical-report-reading, residual-connections, transformer, AI, LLM, python]
pinned: false
---

On March 16, 2026, Kimi Team uploaded a technical report to arXiv: [*Attention Residuals*](/papers/2603.15031v1.pdf).

You can tell what the authors really care about just from the shape of the report. It is not simply "here is a new module." It walks through `motivation -> AttnRes -> Block AttnRes -> infrastructure -> experiments -> discussion`, and in doing so, it retells a deeper question: what is a residual connection actually doing?

## 0. A Few Terms First

If you do not have a machine learning background, it helps to build intuition in the same order this report does:

- `Transformer`: the basic architecture behind most large language models today. You can think of it as a machine that processes information layer by layer.
- `hidden state`: the model's internal intermediate representation at a given layer. Roughly speaking, it is the model's temporary working note at that point.
- `residual connection`: a path between layers that preserves the old input, then adds the layer's new computation on top of it.
- `residual`: closer to the new increment added by the current layer, the "extra part" introduced inside that residual connection.
- `attention`: a mechanism for selecting which pieces of information matter most right now. A useful first intuition is "looking at the important parts selectively."
- `PreNorm`: normalizing values before entering a layer, then doing the actual computation afterward. You can think of it as adjusting the volume before continuing the mix.

## 1. The One-Sentence Version

This technical report raises a question:

**If the Transformer has already replaced recurrence with attention along the sequence dimension, why is information aggregation along the depth dimension still stuck with fixed addition?**

Modern LLMs almost all use a common layer pattern: first PreNorm, then a residual path. In plain language: normalize the scale, compute something new, then add that new result back to the original input. We usually think of this as a tool for stable optimization, something that helps very deep networks avoid falling apart during training. But the report reminds us that residual connections play another equally important role that has largely gone underexamined:

**They define how information is aggregated across depth.**

If the formula below is not your favorite thing in the world, do not get stuck on it. The plain-English translation right after it is the part that matters.

The standard residual rule is simple:

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1})
$$

You can split it into two parts:

- $h_{l-1}$: the old content, meaning the representation already produced by the previous layer
- $f_{l-1}(h_{l-1})$: the new increment computed by the current layer, which is closer to what "residual" means as a word

And the act of adding those two parts back together is what is more precisely called the residual connection.

If you expand the recurrence, you get:

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$

In plain English, that means: the input seen by layer $l$ is basically "the embedding plus the uniform sum of all previous layer outputs." Every earlier layer gets weight 1. There is no selection, no suppression, no way to say "for this step I should care more about layer 3 than layer 17."

The core idea of AttnRes is just one sentence:

**Replace fixed residual addition with a softmax attention operation over depth.**

## 2. What Is Wrong with the Old Residual Rule

The most important thing about this report is not that it proposes a new formula. It is that it turns something everyone had gotten used to back into a problem.

Standard residual connections have long been treated as optimization infrastructure. As long as gradients can pass through, the mechanism is considered to have done its job. But from the perspective of information flow, that path is surprisingly crude.

Imagine you are working on a document that keeps being revised. At each round, instead of selecting the most relevant parts of older versions and merging them thoughtfully, you just append the full text of every previous draft to the end. By revision 20, the important insights from revision 3 are still technically there, but they are buried inside an ever-thickening pile.

That is the PreNorm problem the report highlights. It builds on observations from SiameseNorm and argues that under PreNorm, the magnitude of the `hidden state` grows approximately like $O(L)$ with depth. Here, hidden state is just the model's internal running note at each layer. The result is:

- later layers see an increasingly bloated historical sum
- early-layer information does not disappear, but it gets diluted
- later layers are forced to emit larger and larger magnitudes if they want to be heard

The report calls this `PreNorm dilution`. It is an excellent name. The problem is not that gradients vanish, nor that training explodes. It is that each layer's relative contribution gets progressively washed out.

There is a line of reasoning under the surface here that I really like: along the sequence dimension, we stopped being satisfied with "treat every past token the same" a long time ago. That is why attention exists. So why are we still willing to accept "sum every previous layer with equal weight" along the depth dimension?

## 3. What AttnRes Actually Does

The form of AttnRes is clean. Layer $l$ no longer mechanically receives the sum of all previous outputs. Instead, it performs a weighted selection over those historical representations:

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i
$$

The weights $\alpha_{i \to l}$ come from a softmax. If you are not used to that term, the easiest working definition is: softmax turns a set of scores into weights that add up to 1, which lets the model say clearly "look more here, less there":

$$
\alpha_{i \to l} = \operatorname{softmax}\left(w_l^T \operatorname{RMSNorm}(k_i)\right)
$$

If you have never worked with attention before, here is the cheapest way to think about it:

- `query`: what the current layer is looking for
- `key`: what kind of index label each historical layer carries
- `value`: the content that actually gets retrieved and aggregated

Three details in the design matter a lot.

First, **the query is not computed from the current hidden state. It is a learned pseudo-query vector $w_l$ for each layer.**  
This is slightly counterintuitive. Normally, when we see attention, we assume the query must come from the current input. Here the authors intentionally make it layer-specific rather than token-specific. The upside is that multiple queries inside the same block can be computed in batches ahead of time, which opens the door for later infrastructure optimizations.

Second, **the keys and values come directly from previous layer outputs.**  
That means the input dependence does not vanish. It just lives in the layer representations themselves rather than in a dynamic query. Different samples produce different previous-layer outputs, so the depth attention remains input-dependent in the end.

Third, **the keys are normalized with RMSNorm first.**  
This is a small but important choice. Without normalization, layers with larger magnitudes would automatically dominate the dot products. Then the attention weights would reflect "which layer is louder" more than "which layer is more relevant."

In Python (using PyTorch), one clean implementation looks like this:

```python showLanguage
import torch
from torch import nn


def attention_residual(
    sources: list[torch.Tensor],
    pseudo_query: torch.Tensor,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    keys = torch.stack([norm(source) for source in sources], dim=0)
    values = torch.stack(sources, dim=0)

    logits = keys @ pseudo_query
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values).sum(dim=0)
```

At first glance, this looks like "put attention on top of residuals." But I think a more accurate description is:

**It turns the residual connection from a fixed accumulator into a selective depth retriever.**

## 4. The Report Gives an Idea, and It Gives Engineering

**In one sentence: the report proposes Full AttnRes and pushes that idea into a trainable, deployable, and clearly costed engineering solution.**

Full AttnRes lets every layer attend to all previous layers. Theoretically, it is easy to understand, and even the raw arithmetic cost is not terrifying, because network depth $L$ is usually much smaller than sequence length $T$. So the authors argue that $O(L^2 d)$ arithmetic alone is not the scariest part.

The real problems show up in large-scale training:

- activation recomputation turns intermediate layer outputs from discardable values into objects you must preserve
- pipeline parallelism means those cross-layer representations may need to travel across stages
- once every layer must see every previous layer, communication and caching pressure rise quickly

That is why they introduce **Block AttnRes**.

The idea is to divide the $L$ layers into $N$ blocks. Inside a block, you first use ordinary summation to accumulate a block representation. Across blocks, you then apply attention. So:

- Full AttnRes attends to every historical layer
- Block AttnRes attends to summaries of historical blocks, plus the partial sum inside the current block

In other words, it trades fine-grained cross-layer attention for summary-level cross-block attention to gain scalability.

And the authors do not stop at saying "we grouped layers, so memory gets better." They actually work through the systems side of the bill:

- during training they use **cross-stage caching** to avoid repeatedly shipping historical blocks through the pipeline
- during inference they use **two-phase computation**
- phase one computes inter-block attention in parallel
- phase two computes intra-block lookback sequentially, then merges results with online softmax

The appendix and `table/memory_access.tex` contain the hardest numbers in the whole report. Under the report's representative setting:

- standard residual: per-layer residual mechanism I/O is `3d`
- naive Full AttnRes: `130d`
- optimized Full AttnRes: `24d`
- Block AttnRes: `5.5d`
- mHC: `34d`

That comparison says a lot. Block AttnRes is not "as cheap as a standard residual." But it has already moved from "obviously impractical" to "interesting enough to try in a real system." And the measured overhead is modest:

- training wall-clock overhead is below 4%
- inference latency overhead is below 2%

That is why this reads like a real systems-minded technical report to me. A lot of papers have new ideas and fuzzy accounting. This one cares about the accounting.

## 5. What Matters Most in the Experiments

**More than any individual score in the main experiments, what matters is that AttnRes shows the same directional signal in scaling behavior, training dynamics, and downstream capability.**

### 5.1 Scaling Law Results: Not a One-Off Win

The authors first run scaling-law experiments across five model sizes, comparing Baseline, Full AttnRes, and Block AttnRes.

The fitted curves are:

- Baseline: $1.891 \times C^{-0.057}$
- Block AttnRes: $1.870 \times C^{-0.058}$
- Full AttnRes: $1.865 \times C^{-0.057}$

The most important thing here is not which slope differs by how much. It is this:

**AttnRes stays consistently lower across the compute range.**

The report gives a clean headline claim: at `5.6 PFLOP/s-days`, the loss of Block AttnRes is equivalent to what the baseline would need about `1.25x` more compute to reach.

So this does not look like "we happened to tune one model size well." It looks like a reasonably stable scaling benefit.

### 5.2 The Main Model Is Not a Toy

The main experiment is not a small toy benchmark. It uses a large Kimi Linear configuration:

- `48B total / 3B activated parameters`
- 27 Transformer blocks, which means 54 layers
- 8-of-256 routed experts plus 1 shared expert
- pretraining on `1.4T tokens`

That matters, because it shows the authors are not just drawing pretty curves on small models. They actually inserted this residual redesign into a large training recipe.

### 5.3 In the Training Dynamics, Output Magnitudes Stop Running Away

In the baseline, output magnitudes keep rising with depth. The values in the plot are dramatic: the early blocks sit around `0.04`, `0.06`, `0.10`, while later blocks climb to `10.47` and `12.15`. That is PreNorm dilution made visible.

Block AttnRes looks completely different. The magnitudes show a kind of periodic reset at block boundaries, fluctuating roughly between `0.21` and `1.91`, without the same runaway upward drift.

This matters because it suggests AttnRes is not merely "a few more benchmark points at the end." It is changing how representations accumulate across depth during training itself.

### 5.4 Downstream Tasks: The Biggest Gains Are in Reasoning and Code

After pretraining, AttnRes is no worse than the baseline on all listed evaluations, and several gains stand out:

- MMLU: `73.5 -> 74.6`
- GPQA-Diamond: `36.9 -> 44.4`
- Math: `53.5 -> 57.1`
- HumanEval: `59.1 -> 62.2`
- C-Eval: `79.6 -> 82.5`

The most interesting part is that gains are larger on tasks like GPQA, Math, and HumanEval, where multi-step reasoning or program synthesis matter more. The report's explanation is that if later layers can retrieve earlier-layer representations more selectively, compositional tasks benefit more. I think that explanation makes sense.

Complex reasoning is often not limited by missing information. It is limited by important information getting buried deep inside the network.

## 6. What the Ablations Tell Us

**The key takeaway from the ablations is not that "denser connections are stronger," but that input-dependent selective aggregation along depth is what is doing the work.**

The ablation section is strong because it does not only show that the method helps. It also tries to show why.

Some of the most interesting takeaways:

- **DenseFormer reaches 1.767, almost identical to the baseline at 1.766.**  
  So merely being able to access all previous layers is not enough. What matters is whether the weighting is input-dependent.

- **mHC gets to 1.747, which is already a clear improvement.**  
  That suggests dynamic mixing along the depth dimension is genuinely useful.

- **Full AttnRes reaches 1.737.**  
  Lower than the baseline, DenseFormer, and mHC, which suggests explicit softmax depth attention is the stronger route.

- **SWA, which only looks at a recent window, gets 1.764.**  
  That is valuable because it shows the gain is not just "look at the most recent few layers." The gain comes from selectively reaching further back when needed.

- **Changing the block size among 2, 4, and 8 keeps loss around 1.746.**  
  That is why the authors settle on roughly 8 blocks in the end. It is not arbitrary. It is a good engineering-effectiveness sweet spot.

- **An input-dependent query version reaches 1.731, even better than Full AttnRes.**  
  This is especially interesting. It means the pseudo-query design in the report is not the performance ceiling. It is a compromise chosen to make infrastructure optimizations easier. In other words, the authors are not unaware of stronger variants. They are deliberately choosing a more scalable one.

That is one reason I like this report. When you read the main text, the ablations, and the systems section together, you can see the real trade-off clearly: the goal is not blindly minimizing loss at any cost. The goal is something strong enough, while still trainable in practice.

## 7. How to Read This Report

First, the most important thing here is not that the report invents a new module. It is that it elevates residual connections from "optimization stability tool" back into "information routing mechanism."

Once you adopt that lens, many old questions get reframed. Residuals stop looking like mere gradient highways. They become depth aggregation rules. And then new questions appear almost automatically:

- can each layer selectively access earlier layers?
- are there attention-sink-like effects along depth?
- were older residual variants already doing something like depth-wise linear attention?

That is exactly where the discussion section becomes interesting. The authors reinterpret a bunch of residual variants through the lens of a `depth mixing matrix`, and go one step further:

**Many existing methods are, in essence, doing linear attention along the depth dimension; AttnRes is doing softmax attention along depth.**

That is a bold framing, but also an illuminating one. It is basically saying: the Transformer once moved the sequence dimension from recurrence toward softmax attention; AttnRes is trying to push the depth dimension one step further too.

Second, the report feels like an example of "ask the problem correctly first, then make the system workable." It does not obsess over making every local piece maximally fancy. For example, the query is intentionally layer-specific instead of token-dependent. That may not be the absolute strongest choice in terms of raw performance, but it creates room for batching, two-phase computation, and pipeline caching. A deployable technical report is often not about the flashiest local design. It is about what survives under global constraints.

Third, this line from the report says it well:

**Why is depth-wise aggregation still fixed while everything else has become adaptive?**

It puts the issue nicely.

## 8. Where the Report Stops

First, this is still a **technical report / arXiv preprint**, not a peer-reviewed conference paper. The safest attitude is not "it has proven the future." It is "it has proposed a powerful lens and backed it with an implementation that looks engineering-feasible."

Second, the large-scale results are tied to the Kimi Linear line of architecture: MoE, hybrid KDA/MLA attention, and a Moonlight / DeepSeek-V3-style training recipe. That does not weaken the result, but it does mean we should not automatically extrapolate it to every dense decoder-only Transformer.

Third, the report itself admits that Full AttnRes is stronger, while Block AttnRes is the practical answer under today's hardware constraints. If memory, bandwidth, and interconnect improve further, or if more efficient variants of depth attention appear, today's block design probably will not be the endpoint.

## 9. Final Impression

If you reduce the last decade of large-model architecture progress to a very rough storyline:

- Seq2Seq asked: how do we compress one sequence into another?
- Bahdanau asked: why can't decoding look back at different positions in the input?
- Transformer asked: why must sequence modeling depend on recurrence?
- Chinchilla asked: why should extra compute mainly go into parameter count?

Then *Attention Residuals* asks:

**Why is information aggregation across depth still living in the era of "sum every historical layer equally"?**

That question alone is already valuable.

I do not know whether AttnRes will become a default configuration in a few years the way PreNorm did. But I am quite sure this technical report turns residual connections back into something worth thinking about, designing, and optimizing.

People used to say attention rewrote sequence modeling.

This report is trying to rewrite residuals.

In spring 2026, the Kimi team's work already makes one thing clear: when Scaling Laws begin to show signs of nearing a bottleneck, structural innovation in LLMs will continue to emerge.

---

**Further Reading**

- [*Sequence to Sequence Learning with Neural Networks*](/posts/sequence-to-sequence-learning-with-neural-networks/) — the establishment of the encoder-decoder paradigm
- [*Neural Machine Translation by Jointly Learning to Align and Translate*](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) — the origin of the attention mechanism
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — when attention became the lead role and the Transformer was born
- [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](/posts/bert/) — the establishment of the pretraining paradigm
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — the mathematics of scale
- [*Language Models are Few-Shot Learners*](/posts/language-models-are-few-shot-learners/) — bigger models become better at pulling abilities out of context
- [*Training Compute-Optimal Large Language Models*](/posts/training-compute-optimal-large-language-models/) — how to spend compute wisely

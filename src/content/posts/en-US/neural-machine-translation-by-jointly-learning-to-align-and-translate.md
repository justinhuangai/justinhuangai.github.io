---
title: "Paper Reading: Neural Machine Translation by Jointly Learning to Align and Translate"
date: "2026-01-11T16:26:19+08:00"
category: "Paper Reading"
description: The origin of attention mechanism, with real Python code examples
tags: [paper-reading, attention, AI, LLM, python]
pinned: false
---

On September 1, 2014, three researchers uploaded a paper to arXiv (a preprint server where researchers can publish papers without waiting for journal peer review): [*Neural Machine Translation by Jointly Learning to Align and Translate*](/papers/1409.0473v7.pdf).

The three were Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio, from the University of Montreal. Yoshua Bengio is one of the "three godfathers" of deep learning, alongside Geoffrey Hinton and Yann LeCun; the three shared the 2018 Turing Award. Bahdanau was still a PhD student at the time.

The core contribution of this paper can be summarized in one thing: teaching a translation model to look back at different parts of the source sentence when generating each word. It sounds obvious in hindsight, but in the neural machine translation research of the time, this was a genuinely novel idea. It has a name: the "attention mechanism."

Three years later, eight people at Google pushed this idea to its logical extreme and wrote [*Attention Is All You Need*](/posts/attention-is-all-you-need/). So if you want to understand the Transformer, this paper is one of its most important predecessors.

## 1. The Problem

The standard architecture for neural machine translation in 2014 was the encoder-decoder. The encoder, a Recurrent Neural Network (RNN), reads the source sentence from start to finish and compresses the entire sentence into a single fixed-length vector (think of it as a list of numbers with a fixed count). The decoder, another RNN, starts from this vector and generates the translation one word at a time.

The problem is obvious: whether the source sentence is 5 words or 50 words, the encoder must squeeze it into a vector of the same length. Short sentences are fine, but long sentences lose information. It is like asking someone to read an entire page and then summarize it in a single sentence -- the longer the page, the more gets lost.

The paper demonstrated this experimentally: when sentence length exceeded 30 words, the translation quality of the conventional encoder-decoder dropped sharply.

This is the "fixed-length bottleneck."

## 2. The Core Idea: Stop Compressing, Let the Decoder Look for Itself

The paper's solution is intuitive: if compressing the entire sentence into a single vector loses information, then stop compressing. The encoder retains the annotation vector at every position (formed by concatenating the forward and backward hidden states of a bidirectional RNN -- think of it as the intermediate result produced after processing each word), and the decoder, when generating each target word, decides for itself which parts of the source sentence to focus on.

This is the heart of the attention mechanism: **instead of forcing all information through a single bottleneck, let the model learn to look back and find what it needs, when it needs it.**

Specifically, it works in three steps:

**Step 1: Scoring.** Before generating the i-th target word, the decoder compares its current state s_{i-1} with each encoder position's hidden state h_j, producing an "alignment score" e_{ij}. The higher the score, the more important position j in the source sentence is for generating the current target word.

The scoring function used in the paper:

$$
e_{ij} = a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j)
$$

This is called "additive attention." The decoder state and encoder state each undergo a linear transformation (multiply by a matrix), the results are added together, passed through tanh (a function that squashes values to between -1 and 1), and then dot-producted with a vector v_a to produce a scalar score.

**Step 2: Normalization.** Softmax converts all position scores into probabilities that sum to 1:

$$
\alpha_{ij} = \operatorname{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

**Step 3: Weighted sum.** These probabilities are used to compute a weighted sum of the encoder's hidden states, producing a "context vector" c_i:

$$
c_i = \sum_j \alpha_{ij} h_j
$$

This context vector is the key information the decoder extracts from the source sentence when generating the i-th word. The context vector is different for each generated word, because the model focuses on different source positions each time.

In Python (using PyTorch):

```python showLanguage
import torch
from torch import nn


def bahdanau_attention(
    decoder_state: torch.Tensor,
    encoder_outputs: torch.Tensor,
    w_a: nn.Linear,
    u_a: nn.Linear,
    v_a: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_features = w_a(decoder_state).unsqueeze(1)
    encoder_features = u_a(encoder_outputs)
    scores = v_a(torch.tanh(decoder_features + encoder_features)).squeeze(-1)
    weights = torch.softmax(scores, dim=-1)
    context = torch.sum(weights.unsqueeze(-1) * encoder_outputs, dim=1)
    return context, weights
```

Unlike the "dot-product attention" used later in the Transformer (where Q and K are directly dot-producted), this paper uses "additive attention" (each is linearly transformed first, then added together). The two approaches have different characteristics, but dot-product attention is better suited for efficient matrix multiplication; combined with the Transformer's removal of RNN's sequential dependency, attention finally became a core operator that could be massively parallelized.

## 3. The Encoder: Bidirectional RNN

A unidirectional RNN reads the sentence left to right, outputting a summary vector only after the last word. The problem: each position's hidden state mainly carries left-side context and cannot see what is to the right.

The paper solves this with a bidirectional RNN (BiRNN). One RNN reads left to right, another reads right to left, and then the hidden states from both directions are concatenated. This way, each position's hidden state contains context from both the left and the right.

```python showLanguage
import torch
from torch import nn


class BidirectionalRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(inputs)
        return outputs
```

In the paper, each direction has 1000 hidden units, concatenated to 2000 dimensions. This doubles the parameters compared to a unidirectional RNN, but in return every position can see the full context.

## 4. The Decoder: Realigning at Every Step

Putting the encoder and attention mechanism together, the decoder's workflow becomes clear:

1. The encoder reads the source sentence with a bidirectional RNN, retaining the hidden state (annotation vector) at every position
2. The decoder begins generating the translation, and before generating each word:
   - Computes attention weights using the current state and all annotation vectors
   - Produces a context vector via weighted sum
   - Combines the context vector, the previously generated word, and the current state to predict the next word

```python showLanguage
import torch
from torch import nn


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=embedding_dim + 2 * hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.w_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def decode_step(
        self,
        prev_word: torch.Tensor,
        prev_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, _ = bahdanau_attention(
            prev_state.squeeze(0),
            encoder_outputs,
            self.w_a,
            self.u_a,
            self.v_a,
        )
        rnn_input = torch.cat([prev_word, context.unsqueeze(1)], dim=-1)
        output, new_state = self.rnn(rnn_input, prev_state)
        logits = self.output_proj(output[:, -1, :])
        return logits, new_state
```

The key point: every time the decoder generates a target word, it recomputes the attention distribution. When translating the first word, it might focus on the beginning of the source sentence; when translating the last word, it might focus on the end. This dynamic alignment capability is something the previous fixed-vector architecture simply could not do.

## 5. Experimental Results

The paper ran experiments on the English-to-French translation task (using the WMT '14 dataset), measuring performance with BLEU scores (a standard metric for machine translation, measuring how close the machine output is to human translation, with a maximum of 100).

Key comparisons:
- **RNNencdec-50** (conventional encoder-decoder, trained on sentences up to 50 words): 26.71 BLEU
- **RNNsearch-50** (model with attention, trained on sentences up to 50 words): **34.16 BLEU**
- **Moses** (the strongest conventional phrase-based translation system at the time): 33.30 BLEU

An improvement of 7.45 points. In the experimental setup reported by the paper, the attention-based neural model had matched or even surpassed the dominant conventional phrase-based translation system.

The more critical finding is in the paper's Figure 2: as sentence length increased, the conventional encoder-decoder's BLEU score dropped steeply, while the attention-based model was barely affected. This directly validated the paper's core hypothesis: the fixed-length vector is the bottleneck, and the attention mechanism can bypass it.

The paper also visualized the attention weights. In English-to-French translation, the attention weights nearly formed a diagonal line, showing that the model had automatically learned that "English word 1 corresponds to French word 1, English word 2 corresponds to French word 2." When word order differed (for instance, French adjectives placed after nouns), the attention weights shifted accordingly. The model learned all of this without any manual alignment annotations.

## 6. My Takeaways

After reading this paper, a few things stand out.

First, the problem this paper solves is extremely clear: the encoder compresses the entire sentence into a single vector, and long sentences lose information. The solution is equally intuitive: stop compressing, and let the decoder look for itself. Good research is often like this -- the problem is clear, and the solution follows naturally.

Second, attention in this paper is still a supporting role to the RNN. The encoder is still recurrent (bidirectional RNN), the decoder is still recurrent, and attention merely bridges the two. Three years later, Vaswani et al. asked a far more radical question: if attention works so well, can we throw away the RNN entirely and keep only attention? The answer was the Transformer.

Third, when rewriting this paper's attention mechanism in real Python, you will notice that its computation is considerably more complex than the Transformer's Scaled Dot-Product Attention. Additive attention requires extra weight matrices W_a, U_a, v_a, while dot-product attention only needs Q and K to be directly multiplied and scaled. Going from "addition" to "multiplication" seems like a small step, but in practice it dramatically simplified the computation and made it far more suitable for efficient matrix operations.

Fourth, Bahdanau was a PhD student at the time, and Bengio was his advisor. A PhD student's paper ended up defining the core component of AI research for the next decade. The attention mechanism started here, was amplified by the Transformer, and ultimately became the foundation of GPT, BERT, and LLaMA.

This paper did not invent any complicated mathematics. It simply asked a straightforward question: why can't the decoder look back?

Then it let the decoder look back.

And that look changed an entire era.

---

**Paper Reading Series**

- [*Sequence to Sequence Learning with Neural Networks*](/posts/sequence-to-sequence-learning-with-neural-networks/) — Establishing the encoder-decoder paradigm
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — Attention takes center stage: the birth of the Transformer
- [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](/posts/bert/) — Establishing the pre-training paradigm
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — The mathematics of scale
- [*Language Models are Few-Shot Learners*](/posts/language-models-are-few-shot-learners/) — Larger models, better at eliciting abilities from context
- [*Training Compute-Optimal Large Language Models*](/posts/training-compute-optimal-large-language-models/) — How to spend your compute budget wisely

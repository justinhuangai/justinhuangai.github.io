---
title: "Paper Reading: Sequence to Sequence Learning with Neural Networks"
date: "2026-01-24T16:41:08+08:00"
category: "Paper Reading"
description: Establishing the encoder-decoder paradigm, with real Python code examples
tags: [paper-reading, seq2seq, AI, LLM, python]
pinned: false
---

On September 10, 2014, three Google researchers uploaded a paper to arXiv (a preprint server where researchers can publish papers without waiting for journal peer review): [*Sequence to Sequence Learning with Neural Networks*](/papers/1409.3215v3.pdf).

The authors are Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, all from Google. Sutskever was one of the authors of AlexNet, collaborating with Alex Krizhevsky and Geoffrey Hinton on the paper that ignited the deep learning revolution; he later became a co-founder of OpenAI. Vinyals went on to lead AlphaStar (DeepMind's StarCraft AI) at DeepMind. Quoc V. Le drove AutoML and other research at Google.

This paper did something deceptively simple: use one neural network to read a sentence and compress it into a vector, then use another neural network to generate a translation from that vector. The input and output can differ in length, language, and structure. This framework has a name: "Sequence to Sequence" (Seq2Seq).

It established the encoder-decoder paradigm. Later, [Bahdanau added attention on top of it](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/), and then [Vaswani et al. rewrote the entire architecture with the Transformer](/posts/attention-is-all-you-need/). But the starting point was this paper.

## 1. The Problem

By 2014, deep neural networks had already achieved breakthroughs in tasks like image recognition, but for tasks like machine translation -- directly mapping a variable-length sequence to another variable-length sequence -- neural networks were still struggling.

An English sentence might be 5 words, and its French translation 7 words. The input and output differ in length, with no simple one-to-one correspondence.

The conventional solution was to piece together a large number of hand-designed rules and statistical features into a complex translation pipeline (Statistical Machine Translation, SMT). It worked, but each component had to be tuned separately, and end-to-end optimization was difficult.

The paper proposed a simpler idea: can a single end-to-end neural network map directly from a source language sequence to a target language sequence?

## 2. Core Architecture: Encoder-Decoder

The paper's approach can be summarized in one sentence: **one LSTM reads, another LSTM writes.**

LSTM (Long Short-Term Memory) is a special type of RNN designed to handle long-range dependencies. Standard RNNs tend to "forget" earlier content as sequences get longer. LSTMs mitigate this through gating mechanisms that decide which information to keep and which to discard.

The specific workflow:

1. The **encoder** (a 4-layer deep LSTM) reads the source sentence from start to finish, compressing the entire sentence into a set of fixed-length final states, which are handed to the decoder as its starting point
2. The **decoder** (another 4-layer deep LSTM) starts from this state and generates the target language translation one word at a time, until it outputs the end-of-sentence symbol \<EOS\>

The probability formula from the paper:

$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_t p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

In plain language: given a source sentence x, the probability of generating target sentence y equals the product of the probability of generating each next word at every step. Each step's prediction depends on two things: the vector v compressed by the encoder, and all previously generated words.

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

The architecture itself is not complicated. The paper's contribution was not in inventing a new component, but in proving that this simple framework actually worked -- and worked well enough to compete with carefully tuned traditional systems.

## 3. Three Key Design Decisions

The paper identified three design choices with major impact on performance:

**First, use two separate LSTMs.** The encoder and decoder do not share parameters. This slightly increases the parameter count, but allows the model to better handle the distinct characteristics of source and target languages. The paper noted this also makes it possible to train on multiple language pairs simultaneously.

**Second, use deep LSTMs.** The paper used 4-layer LSTMs, with each additional layer reducing perplexity by nearly 10%. Shallow LSTMs (1-2 layers) performed significantly worse. Depth gave the model a larger representational space.

**Third, reverse the source sentence.** This was the paper's most surprising finding. Reversing the source sentence "a, b, c" to "c, b, a" before feeding it to the encoder bumped the BLEU score from 25.9 to 30.6 -- an improvement of nearly 5 points.

Why does reversal help? The paper's explanation: in normal order, the first word of the source sentence is far from the first word of the target sentence (the entire source sentence sits in between). After reversal, the first few words of the source and target sentences are temporally close, creating more "short-range dependencies" for the gradient (the signal the model uses to adjust its parameters), making optimization easier.

```python showLanguage
import torch


def reverse_source(source_tokens: list[int]) -> list[int]:
    return list(reversed(source_tokens))


source_sentence = [11, 23, 37, 42]
reversed_source = reverse_source(source_sentence)
source_tensor = torch.tensor([reversed_source], dtype=torch.long)
```

This trick is so simple it barely seems like a legitimate research contribution, but it genuinely worked, and it revealed a deeper issue: RNNs are sensitive to the distance between elements in a sequence -- the closer, the easier to learn. This problem was later solved fundamentally by the attention mechanism.

## 4. Experimental Results

The paper ran experiments on the WMT '14 English-to-French translation task.

Key numbers:
- **Single reversed LSTM**, beam size 12: 30.59 BLEU
- **Ensemble of 5 reversed LSTMs**, beam size 2: 34.50 BLEU
- **Ensemble of 5 reversed LSTMs**, beam size 12: **34.81 BLEU**
- **Conventional phrase-based translation system** (Moses baseline): 33.30 BLEU

In the experimental setup reported by the paper, the ensemble of 5 LSTMs surpassed the conventional phrase-based system with 34.81 versus 33.30. Considering the LSTM had a vocabulary of only 80,000 words (outputting UNK for any out-of-vocabulary word) while the conventional system's vocabulary was virtually unlimited, this result is quite compelling.

The paper also used the LSTM to re-rank the conventional system's 1000-best candidate list, pushing the BLEU score further to 36.5, approaching the best published result at the time (37.0).

Another noteworthy finding: compared to other neural methods at the time, the LSTM showed less severe performance degradation on long sentences. This contrasted with the steep long-sentence performance drops reported by other researchers, and the paper attributed this to the source reversal strategy.

## 5. What the Model "Understands"

The paper also ran an interesting visualization experiment. Different sentences were fed into the encoder, the final hidden state vectors were extracted, and PCA was used to project them onto a 2D plane.

The results showed:
- Sentences with similar meaning clustered together in the vector space
- Active and passive voice sentences ("I gave her a card" vs "I was given a card by her") landed in nearby positions
- Sentences with different word order but the same meaning were also correctly clustered

This at least suggests that the encoder's learned representations go beyond simple bag-of-words statistics (mixing words together regardless of order) and contain a substantial amount of syntactic and semantic information.

## 6. Training Details

**Model specifications**: 4-layer LSTM, 1000 units per layer, word embedding dimension of 1000, total parameter count of 384 million. Of these, 64 million are pure recurrent connection parameters.

**Hardware**: 8 GPUs. One GPU per LSTM layer, with the remaining 4 GPUs used to parallelize the softmax (the vocabulary of 80,000 words makes softmax computation expensive). Training took roughly 10 days.

**Optimizer**: SGD without momentum, initial learning rate of 0.7. After 5 epochs, the learning rate was halved every half epoch, for a total of 7.5 epochs.

**Gradient clipping**: when the L2 norm of the gradient exceeded a threshold of 5, it was scaled down proportionally. This prevents gradient explosion (gradient values suddenly becoming extremely large, causing parameter updates to go haywire).

**Batch optimization**: sentences of similar length were grouped into the same batch, preventing short sentences from wasting compute cycles while "waiting" for long sentences. This yielded a 2x training speedup.

## 7. My Takeaways

After reading this paper, a few things stand out.

First, this paper had great ambition but a simple method. One LSTM reads, another LSTM writes, and all information passes through a single vector in between. No attention, no complex alignment mechanism, not even any prior assumptions about language structure. And then it actually worked, with results strong enough to compete with carefully tuned traditional systems. The lesson: given sufficient data and compute, simple end-to-end methods can be surprisingly powerful.

Second, the source reversal finding is quite instructive. It is not an elegant solution -- more of a hack. But it revealed a fundamental limitation of RNNs: sensitivity to the distance between elements in a sequence. Bahdanau's attention mechanism let the model "skip around," no longer constrained by distance. The Transformer went further, abandoning sequential processing entirely, making the distance between any two positions always 1. From reversal to attention to Transformer -- three generations of solutions to the same problem.

Third, this paper and Bahdanau's paper were published almost simultaneously (both in September 2014). Sutskever established the encoder-decoder paradigm; Bahdanau identified the fixed-length vector bottleneck and solved it with the attention mechanism. The two papers are like two sides of the same coin: one is the framework, the other is the fix for the framework's biggest flaw.

Fourth, rewriting this in real Python, you can feel how minimal the architecture is. The encoder just loops through the input; the decoder just loops out the output. But precisely because of this simplicity, its ceiling is obvious: all information must squeeze through a fixed-length vector. This bottleneck becomes especially visceral when you are writing the code yourself.

How much information can a single vector hold? That is the implicit question of this paper.

For longer, more complex sentences -- not enough.

And so, later came attention, and later came the Transformer.

---

**Paper Reading Series**

- [*Neural Machine Translation by Jointly Learning to Align and Translate*](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) — The origin of attention
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — Attention takes center stage: the birth of the Transformer
- [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](/posts/bert/) — Establishing the pre-training paradigm
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — The mathematics of scale
- [*Language Models are Few-Shot Learners*](/posts/language-models-are-few-shot-learners/) — Larger models, better at eliciting abilities from context
- [*Training Compute-Optimal Large Language Models*](/posts/training-compute-optimal-large-language-models/) — How to spend your compute budget wisely

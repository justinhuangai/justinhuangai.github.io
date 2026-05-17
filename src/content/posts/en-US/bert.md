---
title: "BERT: The Pre-Training Blueprint for Language Understanding"
date: "2026-01-31T16:52:21+08:00"
category: "Paper Reading"
description: Establishing the pre-training paradigm, with real Python code examples
tags: [paper-reading, bert, AI, LLM, python]
pinned: false
---

BERT was not trying to build a deeper Transformer for its own sake. It was attacking the fragmentation of NLP tasks. Question answering, classification, and sequence labeling each had their own model shape and data interface, so language knowledge was hard to reuse cleanly.

[*BERT*](/papers/1810.04805v2.pdf) matters not just because it used a bidirectional Transformer, but because it turned language understanding into a problem of reusing pretrained representations. Learn a general representation first, then attach a thin task layer. That shift mattered more than the architecture alone.

## 1. The Problem

In 2018, NLP had an awkward status quo: every task required its own specially designed model architecture. Question answering needed one model, sentiment analysis needed another, named entity recognition yet another. Labeled data for each task was scarce, and models trained on one task were hard to transfer to others.

There had already been attempts at pre-training. ELMo used bidirectional LSTMs to learn contextual representations, but it merely "bolted" pre-trained features onto task-specific architectures — the architecture itself was still task-specific. OpenAI GPT used the Transformer for pre-training and fine-tuning, but it could only look left-to-right (unidirectional) — each token could only attend to tokens before it, never after.

The paper argued that unidirectional language models have significant limitations on language understanding tasks that require deep bidirectional context. For example:

> "He picked up the _____ and started playing."

Looking only at the left context ("He picked up the"), the answer could be anything. But seeing the right context ("and started playing"), you immediately know it is some kind of musical instrument. For many language understanding tasks, bidirectional context is naturally more advantageous.

## 2. The Core Idea: Mask Some Words, Make the Model Guess

BERT's solution is intuitive: since a bidirectional language model cannot be trained the traditional way (each word would indirectly "see itself"), change the training objective.

**Masked Language Model (MLM)**: randomly mask 15% of the input tokens — specifically, replace them with a special \[MASK\] token — then have the model predict the masked words from context. This idea comes from the Cloze task in psychology (proposed by Taylor in 1953), just like the fill-in-the-blank exercise above.

After masking, the model must use both left and right context to make predictions, and bidirectional understanding emerges naturally.

But replacing all selected tokens with \[MASK\] introduces a problem: \[MASK\] never appears during fine-tuning, creating a mismatch between pre-training and fine-tuning. The paper's solution: of the selected 15% of tokens, 80% are replaced with \[MASK\], 10% are replaced with a random token, and 10% are left unchanged. This way, the model cannot simply rely on "I see \[MASK\] so I need to predict" — it must maintain understanding at every position.

```python showLanguage
import random
from typing import Optional, Sequence


def mask_tokens(
    tokens: Sequence[str],
    mask_prob: float = 0.15,
    vocab: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[int], list[str]]:
    if vocab is None:
        vocab = tokens

    masked = list(tokens)
    positions: list[int] = []
    labels: list[str] = []

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            positions.append(i)
            labels.append(token)

            r = random.random()
            if r < 0.8:
                masked[i] = "[MASK]"
            elif r < 0.9:
                masked[i] = random.choice(vocab)

    return masked, positions, labels
```

## 3. The Second Pre-training Task: Next Sentence Prediction

Many NLP tasks (such as question answering and natural language inference) require understanding the relationship between two sentences, but language models do not directly model such relationships.

The paper added a second pre-training task: **Next Sentence Prediction (NSP)**. The model is given two sentences A and B — 50% of the time B is the actual next sentence after A, and 50% of the time B is randomly drawn from the corpus. The model must judge whether B actually follows A.

The task design is simple, but the paper's ablation study (removing one component at a time to observe the effect) showed that removing NSP noticeably hurt performance on question answering and natural language inference tasks; however, later work (such as RoBERTa) reached different conclusions about the necessity of NSP.

```python showLanguage
from dataclasses import dataclass


@dataclass
class PretrainingExample:
    tokens: list[str]
    segment_ids: list[int]
    masked_positions: list[int]
    masked_labels: list[str]
    is_next: bool
```

## 4. Model Architecture

BERT's architecture is not a new invention. It is simply the encoder portion of the [Transformer](/posts/attention-is-all-you-need/), stacked layer by layer.

The paper specifies two sizes:

- **BERT_BASE**: 12 layers, hidden size 768, 12 attention heads, 110M parameters
- **BERT_LARGE**: 24 layers, hidden size 1024, 16 attention heads, 340M parameters

BERT_BASE has roughly the same parameter count as OpenAI GPT, enabling direct comparison. The most critical difference between the two is just one thing: GPT uses unidirectional attention (each token can only see tokens to its left), while BERT uses bidirectional attention (each token can see all positions).

The input representation is the sum of three components:

- **Token Embedding**: WordPiece tokenization, 30,000 vocabulary
- **Segment Embedding**: marks whether a token belongs to sentence A or sentence B
- **Position Embedding**: tells the model the position of each token (BERT uses learned position embeddings, not sinusoidal)

Every input sequence begins with a special \[CLS\] token, whose final-layer hidden state is used for sentence-level classification (e.g., NSP, sentiment analysis). Two sentences are separated by \[SEP\].

```python showLanguage
import torch
from torch import nn


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_positions: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        position_ids = position_ids.expand_as(token_ids)
        embeddings = (
            self.token_embedding(token_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(position_ids)
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        max_positions: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_positions, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embeddings(token_ids, segment_ids)
        return self.encoder(hidden)
```

## 5. Fine-tuning: One Model for Every Task

The most elegant aspect of BERT is the simplicity of fine-tuning. Once pre-training is complete, the procedure is nearly identical regardless of the downstream task: add one task-specific output layer on top of BERT, then fine-tune all parameters with a small amount of labeled data.

- **Text classification** (sentiment analysis, natural language inference): take the output vector at the \[CLS\] position and feed it to a linear classifier
- **Question answering** (given a passage, find the answer span): apply two linear transformations to each token's output vector, predicting the start and end positions of the answer
- **Sequence labeling** (named entity recognition): attach a classifier to each token's output vector, predicting labels token by token

Pre-training may take days, but fine-tuning typically takes only minutes to hours (most tasks under 1 hour on a single TPU). This efficiency gap is the core appeal of the "pre-train + fine-tune" paradigm.

## 6. Experimental Results

The paper ran experiments on 11 NLP tasks, setting new records on all of them.

**GLUE benchmark** (General Language Understanding Evaluation, 8 sub-tasks):
- BERT_LARGE averaged 80.5%, a 7.7 percentage point improvement over the previous best (OpenAI GPT)
- On the largest sub-task MNLI, a 4.6% improvement

**SQuAD v1.1** (reading comprehension QA, Test F1):
- BERT_LARGE single model + TriviaQA data: F1 91.8, surpassing human performance (91.2)
- BERT_LARGE ensemble + TriviaQA data: F1 93.2

**SQuAD v2.0** (includes unanswerable questions):
- F1 of 83.1, a 5.1 point improvement over the previous best system

**SWAG** (commonsense reasoning):
- Accuracy of 86.3%, an 8.3 point improvement over OpenAI GPT

The paper also ran ablation experiments on model size and found an important conclusion: larger models performed better on all tasks, even on tasks with very little labeled data (as few as 3,600 examples). This ran counter to the prevailing intuition that small datasets would cause large models to overfit (memorize training data and perform poorly on new data), suggesting that pre-trained knowledge effectively mitigates this risk.

## 7. Training Details

**Pre-training data**: BooksCorpus (800M words) + English Wikipedia (2,500M words), using only text passages and discarding lists, tables, and headers. The paper emphasized the importance of using document-level corpora rather than shuffled sentence-level corpora, in order to capture long-range contextual relationships.

**Tokenization**: WordPiece with a vocabulary of 30,000. WordPiece splits uncommon words into smaller subword units — for example, "playing" might be split into "play" + "##ing".

**Optimizer**: Adam, learning rate 1e-4, with linear warmup over the first 10,000 steps followed by linear decay. Batch size of 256 sequences, maximum sequence length of 512.

**Hardware**: BERT_BASE was trained on 4 Cloud TPUs (16 TPU chips) for 4 days. BERT_LARGE was trained on 16 Cloud TPUs (64 TPU chips) for 4 days.

**Dropout**: 0.1 across all layers. The activation function is GELU (Gaussian Error Linear Unit), rather than the original Transformer's ReLU.

## 8. What This Paper Changed

BERT's sharpest lesson is: **language understanding can be learned first as a reusable representation, then handed to specific tasks.**

The key is not only the Transformer encoder, and not only bidirectionality. The move was to turn unlabeled text into pretraining signal through MLM and NSP, then let classification, question answering, and sequence labeling share the same input format and representation base.

That changed the engineering unit of NLP. Before BERT, each task often looked like its own project. After BERT, many tasks became thin adaptation layers on top of the same pretrained representation. The model no longer relearned language from each task dataset; it learned a general language base first, then learned task boundaries.

The next time you look at an understanding model, do not only ask whether it is bidirectional. Ask whether its representation can be reused, and whether task differences live inside the model body or in a light interface on top.

---

**Paper Reading Series**

- [*Sequence to Sequence Learning with Neural Networks*](/posts/sequence-to-sequence-learning-with-neural-networks/) — Establishing the encoder-decoder paradigm
- [*Neural Machine Translation by Jointly Learning to Align and Translate*](/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) — The origin of attention
- [*Attention Is All You Need*](/posts/attention-is-all-you-need/) — Attention takes center stage: the birth of the Transformer
- [*Scaling Laws for Neural Language Models*](/posts/scaling-laws-for-neural-language-models/) — The mathematics of scale
- [*Language Models are Few-Shot Learners*](/posts/language-models-are-few-shot-learners/) — Larger models, better at eliciting abilities from context
- [*Training Compute-Optimal Large Language Models*](/posts/training-compute-optimal-large-language-models/) — How to spend your compute budget wisely

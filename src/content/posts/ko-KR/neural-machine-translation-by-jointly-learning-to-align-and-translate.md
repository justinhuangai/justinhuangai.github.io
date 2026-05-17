---
title: "《Neural Machine Translation by Jointly Learning to Align and Translate》: Transformer 이전의 어텐션"
date: "2026-01-11T16:26:19+08:00"
category: "Paper Reading"
description: 어텐션 메커니즘의 기원, 실제 Python 코드 예시 포함
tags: [paper-reading, attention, AI, LLM, python]
pinned: false
---

Seq2Seq는 end-to-end 번역이 가능하다는 것을 보였지만, 하나의 단단한 병목을 남겼습니다. 원문 전체가 하나의 고정 벡터에 들어가야 했습니다. [《Neural Machine Translation by Jointly Learning to Align and Translate》](/papers/1409.0473v7.pdf)의 첫 번째 의미는 "문장 전체를 기억하라"를 "매 단계 관련 정보를 다시 찾아라"로 바꾼 데 있습니다.

여기서 attention은 아직 Transformer의 주인공이 아닙니다. RNN 옆에 붙은 검색 경로입니다. 진짜 변화는 공식이 아니라 과제의 형태입니다. Decoder는 더 이상 하나의 요약 벡터만 믿지 않습니다. 단어를 생성할 때마다 입력과 다시 정렬할 수 있습니다.

## 0. 먼저 몇 가지 용어부터

머신러닝 배경이 전혀 없어도, 이 논문이 고치려 한 병목만 먼저 잡으면 읽기가 훨씬 쉬워진다:

- `encoder-decoder`: 한쪽이 원문을 끝까지 읽고, 다른 한쪽이 번역문을 한 단어씩 써 내려가는 구조다.
- `RNN / 순환 신경망`: 당시 주류였던 시퀀스 모델이다. 텍스트를 반드시 순서대로 처리해야 한다.
- `hidden state / 은닉 상태`: 모델이 어떤 위치까지 읽은 뒤 손에 쥐고 있는 임시 메모라고 생각하면 된다.
- `alignment / 정렬`: 원문 어느 부분이 지금 생성하려는 번역 단어와 대응되는지를 뜻한다.
- `attention`: 하나의 압축 벡터만 바라보는 대신, 필요할 때 원문의 더 relevant 한 위치를 다시 보게 만드는 방식이다.

## 1. 문제

2014년 신경 기계 번역의 표준 아키텍처는 encoder-decoder였다. Encoder인 RNN(Recurrent Neural Network)이 원문을 처음부터 끝까지 읽고 전체 문장을 하나의 고정 길이 벡터(정해진 개수의 숫자 목록이라고 생각하면 된다)로 압축한다. Decoder인 또 다른 RNN이 이 벡터로부터 시작하여 한 번에 한 단어씩 번역을 생성한다.

문제는 명확하다: 원문이 5단어든 50단어든, encoder는 동일한 길이의 벡터에 모든 것을 욱여넣어야 한다. 짧은 문장은 괜찮지만, 긴 문장은 정보가 손실된다. 누군가에게 한 페이지를 통째로 읽고 한 문장으로 요약하라고 하는 것과 같다 -- 페이지가 길어질수록 빠지는 내용이 많아진다.

논문은 이를 실험으로 입증했다: 문장 길이가 30단어를 넘으면 기존 encoder-decoder의 번역 품질이 급격히 하락했다.

이것이 바로 "고정 길이 병목(fixed-length bottleneck)"이다.

## 2. 핵심 아이디어: 압축을 멈추고, decoder가 스스로 찾게 하라

논문의 해결책은 직관적이다: 전체 문장을 하나의 벡터로 압축하면 정보가 손실되니, 압축을 멈추면 된다. Encoder가 모든 위치의 annotation 벡터를 유지하고(bidirectional RNN의 순방향과 역방향 hidden state를 연결한 것 -- 각 단어를 처리한 후 생성되는 중간 결과물이라고 생각하면 된다), decoder가 각 목표 단어를 생성할 때 스스로 원문의 어느 부분에 집중할지 결정한다.

이것이 attention 메커니즘의 핵심이다: **모든 정보를 하나의 병목으로 강제로 통과시키는 대신, 모델이 필요할 때 필요한 것을 스스로 되돌아가서 찾도록 학습시키는 것이다.**

구체적으로 세 단계로 이루어진다:

**1단계: 점수 계산.** i번째 목표 단어를 생성하기 전에, decoder는 현재 상태 s_{i-1}을 encoder 각 위치의 hidden state h_j와 비교하여 "정렬 점수" e_{ij}를 산출한다. 점수가 높을수록, 현재 목표 단어를 생성하는 데 원문의 j번째 위치가 더 중요하다는 뜻이다.

논문에서 사용한 점수 함수:

$$
e_{ij} = a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j)
$$

이것을 "additive attention"이라 부른다. Decoder 상태와 encoder 상태에 각각 선형 변환(행렬 곱)을 적용하고, 그 결과를 더한 뒤, tanh(값을 -1에서 1 사이로 압축하는 함수)를 통과시키고, 벡터 v_a와 내적하여 스칼라 점수를 만든다.

**2단계: 정규화.** Softmax가 모든 위치의 점수를 합이 1이 되는 확률로 변환한다:

$$
\alpha_{ij} = \operatorname{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$

**3단계: 가중합.** 이 확률을 사용하여 encoder의 hidden state에 대한 가중합을 계산하고, "context 벡터" c_i를 만든다:

$$
c_i = \sum_j \alpha_{ij} h_j
$$

이 context 벡터가 decoder가 i번째 단어를 생성할 때 원문에서 추출한 핵심 정보다. 모델이 매번 원문의 서로 다른 위치에 집중하기 때문에, context 벡터는 생성되는 단어마다 다르다.

Python(PyTorch 기반)으로 구현하면:

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

나중에 Transformer에서 사용한 "dot-product attention"(Q와 K를 직접 내적하는 방식)과 달리, 이 논문은 "additive attention"(각각을 먼저 선형 변환한 뒤 더하는 방식)을 사용한다. 두 접근 방식은 서로 다른 특성을 가지지만, dot-product attention이 효율적인 행렬 곱 연산에 더 적합하다. Transformer가 RNN의 순차적 의존성을 제거한 것과 결합되어, attention은 마침내 대규모 병렬화가 가능한 핵심 연산자가 되었다.

## 3. Encoder: Bidirectional RNN

단방향 RNN은 문장을 왼쪽에서 오른쪽으로 읽으며, 마지막 단어 이후에만 요약 벡터를 출력한다. 문제는: 각 위치의 hidden state가 주로 왼쪽 문맥만 담고 있어 오른쪽을 볼 수 없다는 점이다.

논문은 bidirectional RNN(BiRNN)으로 이를 해결한다. 하나의 RNN이 왼쪽에서 오른쪽으로, 다른 하나가 오른쪽에서 왼쪽으로 읽고, 양 방향의 hidden state를 연결한다. 이렇게 하면 각 위치의 hidden state가 왼쪽과 오른쪽 양쪽의 문맥을 모두 포함하게 된다.

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

논문에서 각 방향은 1000개의 hidden unit을 가지며, 연결하면 2000차원이 된다. 단방향 RNN에 비해 파라미터가 두 배가 되지만, 그 대가로 모든 위치에서 전체 문맥을 볼 수 있다.

## 4. Decoder: 매 단계마다 재정렬

Encoder와 attention 메커니즘을 합치면, decoder의 작업 흐름이 명확해진다:

1. Encoder가 bidirectional RNN으로 원문을 읽고, 모든 위치의 hidden state(annotation 벡터)를 유지한다
2. Decoder가 번역을 생성하기 시작하며, 각 단어를 생성하기 전에:
   - 현재 상태와 모든 annotation 벡터를 사용하여 attention 가중치를 계산한다
   - 가중합으로 context 벡터를 만든다
   - Context 벡터, 이전에 생성한 단어, 현재 상태를 결합하여 다음 단어를 예측한다

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

핵심 포인트: decoder가 목표 단어를 생성할 때마다 attention 분포를 다시 계산한다. 첫 번째 단어를 번역할 때는 원문의 시작 부분에 집중하고, 마지막 단어를 번역할 때는 끝 부분에 집중할 수 있다. 이런 동적 정렬 능력은 이전의 고정 벡터 아키텍처로는 절대 할 수 없는 것이다.

## 5. 실험 결과

논문은 영어-프랑스어 번역 과제(WMT '14 데이터셋 사용)에서 실험을 수행하고, BLEU 점수(기계 번역의 표준 평가 지표로, 기계 출력이 사람 번역에 얼마나 가까운지를 측정하며 최대 100점)로 성능을 측정했다.

주요 비교:
- **RNNencdec-50** (기존 encoder-decoder, 50단어 이하 문장으로 학습): 26.71 BLEU
- **RNNsearch-50** (attention 적용 모델, 50단어 이하 문장으로 학습): **34.16 BLEU**
- **Moses** (당시 가장 강력한 기존 구문 기반 번역 시스템): 33.30 BLEU

7.45점 향상. 논문의 실험 설정에서, attention 기반 신경 모델은 당시 지배적이었던 기존 구문 기반 번역 시스템과 대등하거나 심지어 능가하는 성능을 보였다.

더 중요한 발견은 논문의 Figure 2에 있다: 문장 길이가 증가하면 기존 encoder-decoder의 BLEU 점수가 급격히 하락한 반면, attention 기반 모델은 거의 영향을 받지 않았다. 이는 논문의 핵심 가설을 직접 검증한다: 고정 길이 벡터가 병목이며, attention 메커니즘이 이를 우회할 수 있다는 것이다.

논문은 또한 attention 가중치를 시각화했다. 영어-프랑스어 번역에서 attention 가중치는 거의 대각선을 이루며, 모델이 "영어 단어 1은 프랑스어 단어 1에 대응, 영어 단어 2는 프랑스어 단어 2에 대응"하는 것을 자동으로 학습했음을 보여준다. 어순이 다를 때(예를 들어 프랑스어에서 형용사가 명사 뒤에 오는 경우) attention 가중치가 그에 맞게 이동했다. 모델은 이 모든 것을 수동 정렬 주석 없이 학습했다.

## 6. 이 논문이 바꾼 질문

Attention의 첫 번째 의미는 이것입니다. **문장 전체를 기억하라를 매 단계 관련 정보를 다시 찾아라로 바꾼다.**

이 논문은 Transformer를 발명한 것도, RNN을 버린 것도 아닙니다. 인터페이스를 바꿨습니다. Encoder는 더 이상 하나의 요약만 넘기지 않습니다. 입력의 각 위치에 대한 표현을 보관합니다. Decoder는 단어를 생성할 때마다 원문의 어디를 볼지 다시 결정합니다.

이 변화가 긴 문장의 운명을 바꿨습니다. 모델은 더 이상 하나의 고정 벡터에 모든 것을 기억하도록 강요받지 않습니다. 번역은 반복되는 검색과 생성이 됩니다. Alignment는 외부 주석이 아니라 학습 목표 안에서 생겨납니다.

다음에 attention을 볼 때는 공식이 얼마나 예쁜지부터 묻지 않는 편이 좋습니다. 어떤 과제를 기억 문제에서 검색 문제로 바꿨는지 물어야 합니다. 그것이 나중에 Transformer로 확대될 수 있었던 이유입니다.

---

**논문 읽기 시리즈**

- [《Sequence to Sequence Learning with Neural Networks》](/ko-KR/posts/sequence-to-sequence-learning-with-neural-networks/) (신경망을 이용한 시퀀스-투-시퀀스 학습) — Encoder-decoder 패러다임의 확립
- [《Attention Is All You Need》](/ko-KR/posts/attention-is-all-you-need/) (어텐션만 있으면 충분하다) — Attention이 주역이 되다: Transformer의 탄생
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전 학습 패러다임의 확립
- [《Scaling Laws for Neural Language Models》](/ko-KR/posts/scaling-laws-for-neural-language-models/) (신경 언어 모델을 위한 스케일링 법칙) — 스케일의 수학: 왜 더 큰 모델이 예측 가능하게 더 좋은가
- [《Language Models are Few-Shot Learners》](/ko-KR/posts/language-models-are-few-shot-learners/) (언어 모델은 퓨샷 학습자다) — 더 큰 모델, 문맥에서 능력을 더 잘 이끌어내다
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — 컴퓨팅 예산을 현명하게 쓰는 법

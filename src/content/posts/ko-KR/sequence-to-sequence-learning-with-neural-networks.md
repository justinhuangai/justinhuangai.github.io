---
title: "《Sequence to Sequence Learning with Neural Networks》: 인코더-디코더 패러다임의 출발점"
date: "2026-01-24T16:41:08+08:00"
category: "Paper Reading"
description: 인코더-디코더 패러다임의 확립, 실제 Python 코드 예시 포함
tags: [paper-reading, seq2seq, AI, LLM, python]
pinned: false
---

Seq2Seq의 출발점은 단순한 제약입니다. 입력과 출력의 길이가 모두 고정되어 있지 않습니다. 전통적인 번역 파이프라인은 이를 처리할 수 있었지만, 끝에서 끝까지 함께 최적화하기는 어려웠습니다. [《Sequence to Sequence Learning with Neural Networks》](/papers/1409.3215v3.pdf)는 이 문제를 두 개의 학습 가능한 인터페이스로 바꿨습니다. 하나의 네트워크가 입력 전체를 읽고, 다른 네트워크가 출력을 한 단계씩 생성합니다.

이 논문의 가치는 기계 번역을 한 번에 해결했다는 데 있지 않습니다. end-to-end sequence mapping이 가능하다는 것을 보였다는 데 있습니다. 동시에 약점도 분명했습니다. 모든 정보가 하나의 고정 길이 벡터를 지나야 했습니다. Attention과 Transformer는 이 병목에서 자라났습니다.

## 0. 먼저 몇 가지 용어부터

머신러닝 배경이 없다면, 이 논문의 작업 흐름에 맞춰 아래 용어부터 익혀 두면 된다:

- `Seq2Seq / 시퀀스-투-시퀀스`: 하나의 입력 시퀀스를 다른 출력 시퀀스로 직접 바꾸는 방식이다. 예를 들어 영어 문장을 프랑스어 문장으로 바꾸는 식이다.
- `encoder`: 입력을 처음부터 끝까지 읽는 부분이다.
- `decoder`: 출력을 한 단어씩 써 내려가는 부분이다.
- `RNN / 순환 신경망`: 텍스트를 순서대로만 처리할 수 있는 오래된 구조다.
- `LSTM`: RNN의 개선형으로, 긴 문장에서 앞쪽 정보를 더 오래 기억하도록 만든 구조다.
- `vector / 벡터`: 일단은 "숫자들로 압축해 둔 요약본" 정도로 이해하면 충분하다.

## 1. 문제

2014년까지 심층 신경망은 이미지 인식 같은 과제에서 이미 돌파구를 마련했지만, 기계 번역처럼 가변 길이의 시퀀스를 또 다른 가변 길이의 시퀀스로 직접 매핑하는 과제에서는 여전히 고전하고 있었다.

영어 문장이 5단어일 때 프랑스어 번역은 7단어가 될 수 있다. 입력과 출력의 길이가 다르고, 단순한 일대일 대응이 존재하지 않는다.

기존의 해결책은 대량의 수작업 규칙과 통계적 특징을 조합해 복잡한 번역 파이프라인(Statistical Machine Translation, SMT)을 구성하는 것이었다. 작동은 했지만 각 구성 요소를 따로 튜닝해야 했고, end-to-end 최적화가 어려웠다.

이 논문은 더 단순한 아이디어를 제안했다: 하나의 end-to-end 신경망으로 소스 언어 시퀀스에서 타깃 언어 시퀀스로 직접 매핑할 수 있지 않을까?

## 2. 핵심 아키텍처: Encoder-Decoder

이 논문의 접근법은 한 문장으로 요약된다: **하나의 LSTM이 읽고, 또 하나의 LSTM이 쓴다.**

LSTM(Long Short-Term Memory)은 장기 의존성을 처리하기 위해 설계된 특수한 RNN이다. 표준 RNN은 시퀀스가 길어질수록 앞부분의 내용을 "잊어버리는" 경향이 있는데, LSTM은 게이팅 메커니즘을 통해 어떤 정보를 유지하고 어떤 정보를 버릴지 결정함으로써 이를 완화한다.

구체적인 워크플로우:

1. **Encoder**(4층 deep LSTM)가 소스 문장을 처음부터 끝까지 읽고, 문장 전체를 고정 길이의 최종 상태 세트로 압축하여 decoder의 시작점으로 넘긴다
2. **Decoder**(또 다른 4층 deep LSTM)가 이 상태에서 출발하여 타깃 언어 번역을 한 단어씩 생성하고, 문장 끝 기호 \<EOS\>를 출력하면 멈춘다

논문에 나오는 확률 공식:

$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_t p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

쉽게 말하면: 소스 문장 x가 주어졌을 때, 타깃 문장 y를 생성할 확률은 매 단계에서 다음 단어를 생성할 확률의 곱이다. 각 단계의 예측은 두 가지에 의존한다: encoder가 압축한 벡터 v, 그리고 이전까지 생성된 모든 단어.

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

아키텍처 자체는 복잡하지 않다. 이 논문의 기여는 새로운 구성 요소를 발명한 것이 아니라, 이 단순한 프레임워크가 실제로 작동한다는 것을 -- 그것도 정교하게 튜닝된 전통적 시스템과 경쟁할 만큼 잘 작동한다는 것을 증명한 데 있다.

## 3. 세 가지 핵심 설계 결정

논문은 성능에 큰 영향을 미치는 세 가지 설계 선택을 밝혔다:

**첫째, 두 개의 별도 LSTM을 사용한다.** Encoder와 decoder가 파라미터를 공유하지 않는다. 이로 인해 파라미터 수가 약간 늘어나지만, 모델이 소스 언어와 타깃 언어의 서로 다른 특성을 더 잘 처리할 수 있게 된다. 논문은 또한 이를 통해 여러 언어 쌍을 동시에 학습시키는 것도 가능하다고 언급했다.

**둘째, deep LSTM을 사용한다.** 논문은 4층 LSTM을 사용했으며, 층을 하나 추가할 때마다 perplexity가 거의 10%씩 감소했다. 얕은 LSTM(1~2층)은 성능이 현저히 떨어졌다. 깊이는 모델에 더 큰 표현 공간을 제공한다.

**셋째, 소스 문장을 뒤집는다.** 이것이 논문에서 가장 반직관적인 발견이었다. 소스 문장 "a, b, c"를 "c, b, a"로 뒤집어 encoder에 입력하자 BLEU 점수가 25.9에서 30.6으로 -- 약 5포인트나 올랐다.

왜 뒤집기가 도움이 될까? 논문의 설명: 정순서에서는 소스 문장의 첫 번째 단어가 타깃 문장의 첫 번째 단어와 멀리 떨어져 있다(소스 문장 전체가 그 사이에 놓인다). 뒤집으면 소스와 타깃 문장의 앞부분 단어들이 시간적으로 가까워져 gradient(모델이 파라미터를 조정하는 데 사용하는 신호)에 더 많은 "단거리 의존성"이 만들어지고, 이것이 최적화를 쉽게 만든다.

```python showLanguage
import torch


def reverse_source(source_tokens: list[int]) -> list[int]:
    return list(reversed(source_tokens))


source_sentence = [11, 23, 37, 42]
reversed_source = reverse_source(source_sentence)
source_tensor = torch.tensor([reversed_source], dtype=torch.long)
```

이 트릭은 너무 단순해서 정당한 연구 기여처럼 보이지 않을 정도지만, 실제로 효과가 있었고, 더 깊은 문제를 드러냈다: RNN은 시퀀스 내 요소 간 거리에 민감하다 -- 가까울수록 배우기 쉽다. 이 문제는 이후 attention 메커니즘에 의해 근본적으로 해결되었다.

## 4. 실험 결과

논문은 WMT '14 영어-프랑스어 번역 과제에서 실험을 수행했다.

주요 수치:
- **단일 reversed LSTM**, beam size 12: 30.59 BLEU
- **5개 reversed LSTM 앙상블**, beam size 2: 34.50 BLEU
- **5개 reversed LSTM 앙상블**, beam size 12: **34.81 BLEU**
- **기존 구(句) 기반 번역 시스템** (Moses baseline): 33.30 BLEU

논문이 보고한 실험 설정에서, 5개 LSTM 앙상블이 기존 구 기반 시스템을 34.81 대 33.30으로 앞질렀다. LSTM의 어휘가 80,000단어에 불과하고(어휘 밖의 단어는 UNK로 출력) 기존 시스템의 어휘는 사실상 무제한이었다는 점을 고려하면, 이 결과는 상당히 인상적이다.

논문은 또한 LSTM을 사용해 기존 시스템의 1000-best 후보 목록을 재순위화하여 BLEU 점수를 36.5까지 끌어올렸으며, 이는 당시 발표된 최고 결과(37.0)에 근접한 수치였다.

또 하나 주목할 만한 발견: 당시의 다른 신경망 기법과 비교했을 때, LSTM은 긴 문장에서의 성능 저하가 덜 심했다. 이는 다른 연구자들이 보고한 가파른 긴 문장 성능 하락과 대조적이었으며, 논문은 이를 소스 뒤집기 전략 덕분이라고 설명했다.

## 5. 모델이 "이해하는" 것

논문은 구조를 드러내는 시각화 실험도 수행했다. 다양한 문장을 encoder에 입력하고 최종 hidden state 벡터를 추출한 뒤, PCA로 2차원 평면에 투영했다.

결과:
- 의미가 유사한 문장들이 벡터 공간에서 가까이 모였다
- 능동태와 수동태 문장("I gave her a card" vs "I was given a card by her")이 인접한 위치에 나타났다
- 어순이 다르지만 의미가 같은 문장들도 올바르게 군집화되었다

이는 최소한 encoder가 학습한 표현이 단순한 bag-of-words 통계(단어들을 순서에 상관없이 뒤섞는 것)를 넘어서 상당한 양의 구문 및 의미 정보를 담고 있음을 시사한다.

## 6. 학습 상세

**모델 사양**: 4층 LSTM, 층당 1000 유닛, 단어 임베딩 차원 1000, 총 파라미터 수 3억 8400만 개. 이 중 6400만 개는 순수 순환 연결 파라미터이다.

**하드웨어**: GPU 8개. LSTM 층당 GPU 1개, 나머지 4개는 softmax 병렬화에 사용(어휘 80,000단어로 인해 softmax 연산이 무거움). 학습에 약 10일이 소요되었다.

**Optimizer**: 모멘텀 없는 SGD, 초기 학습률 0.7. 5 에포크 이후 반 에포크마다 학습률을 절반으로 줄이며, 총 7.5 에포크 학습.

**Gradient clipping**: gradient의 L2 norm이 임계값 5를 초과하면 비례적으로 축소. 이는 gradient explosion(gradient 값이 갑자기 극단적으로 커져 파라미터 업데이트가 엉망이 되는 현상)을 방지한다.

**배치 최적화**: 비슷한 길이의 문장을 같은 배치에 묶어, 짧은 문장이 긴 문장을 "기다리며" 연산 자원을 낭비하는 것을 방지. 이로 인해 학습 속도가 2배 향상되었다.

## 7. 이 논문이 바꾼 질문

Seq2Seq의 핵심 문장은 이것입니다. **end-to-end mapping은 가능하지만, 고정 벡터는 병목이 된다.**

이 논문은 기계 번역을 사람이 조립한 파이프라인에서 하나의 시스템으로 학습할 수 있는 mapping 문제로 바꿨습니다. 신경망이 입력 전체를 읽고, 출력을 한 단계씩 생성할 수 있음을 보였습니다. 입력과 출력의 길이가 같을 필요도, 사람이 정렬을 붙일 필요도 없었습니다.

동시에 병목도 분명히 드러났습니다. 원문 정보 전체가 하나의 벡터를 지나야 했습니다. 문장이 길수록 압축 손실은 커집니다. source sentence를 뒤집는 요령은 거리 문제를 완화했지만, 정보가 좁은 문을 지나야 한다는 사실은 바꾸지 못했습니다.

다음에 Seq2Seq를 볼 때는 "encoder-decoder"에서 멈추지 않는 편이 좋습니다. 이 시스템은 정보를 어디에 압축하는가? 그 압축 지점이 다음 아키텍처가 우회해야 할 병목이 되는가?

---

**논문 읽기 시리즈**

- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/ko-KR/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) (정렬과 번역을 공동으로 학습하는 신경 기계 번역) — Attention의 기원
- [《Attention Is All You Need》](/ko-KR/posts/attention-is-all-you-need/) (어텐션만 있으면 충분하다) — Attention이 주인공이 되다: Transformer의 탄생
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전학습 패러다임의 확립
- [《Scaling Laws for Neural Language Models》](/ko-KR/posts/scaling-laws-for-neural-language-models/) (신경 언어 모델을 위한 스케일링 법칙) — 스케일의 수학: 왜 더 큰 모델이 예측 가능하게 더 좋은가
- [《Language Models are Few-Shot Learners》](/ko-KR/posts/language-models-are-few-shot-learners/) (언어 모델은 퓨샷 학습자다) — 더 큰 모델, 맥락에서 능력을 이끌어내는 데 더 뛰어나다
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — 컴퓨팅 예산을 현명하게 쓰는 법

---
title: "논문 읽기: 《Attention Is All You Need》 (어텐션만 있으면 충분하다)"
date: "2026-01-06T16:18:46+08:00"
category: "Paper Reading"
description: Transformer 논문에 대한 이해, 실제 Python 코드 예시 포함
tags: [paper-reading, transformer, AI, LLM, python]
pinned: false
---

2017년 6월 12일, 여덟 명이 arXiv(연구자들이 학술지 심사를 기다리지 않고 논문을 발표할 수 있는 프리프린트 서버)에 논문 한 편을 올렸다. 제목은 단 다섯 단어: [《Attention Is All You Need》](/papers/1706.03762v7.pdf) (어텐션만 있으면 충분하다).

여덟 명은 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, 그리고 Illia Polosukhin이었으며, 대부분 당시 Google Brain과 Google Research에서 일하고 있었다.

논문이 발표된 후 이들은 뿔뿔이 흩어졌다. Noam Shazeer는 Google을 떠나 Character.AI를 창업했다가, 나중에 프리미엄을 받고 다시 Google에 인수되었다. Aidan Gomez는 토론토 대학교에서 박사 과정을 마치기도 전에 Cohere를 설립하여 기업용 대규모 언어 모델을 구축했다. Llion Jones는 일본으로 건너가 Sakana AI를 창립했다. Illia Polosukhin은 아무도 예상하지 못한 길을 걸었다 -- 블록체인 프로젝트 NEAR Protocol을 시작한 것이다. Ashish Vaswani와 Niki Parmar는 함께 Adept AI를 공동 창립한 뒤, 나중에 Essential AI를 설립했다. Jakob Uszkoreit는 AI를 활용해 RNA 기반 의약품을 설계하는 Inceptive를 창립했다. Łukasz Kaiser는 OpenAI에 합류해 GPT 시리즈 개발에 기여했다.

여덟 명의 저자, 일곱 개의 회사, AI, 블록체인, 바이오테크에 걸친 행보.

약 9년이 지난 지금, ChatGPT, Claude, DeepSeek, Qwen -- 이 AI 제품들의 기본 아키텍처는 거의 모두 그 15페이지짜리 논문으로 거슬러 올라갈 수 있다.

이 글은 내가 논문을 읽고 이해한 내용을 정리한 것이며, 실제 Python 코드 예시 포함했다. 번역도, 요약도 아니다. 기술적 배경 지식 없이도 따라갈 수 있다.

## 0. 먼저 몇 가지 용어부터

머신러닝 배경이 전혀 없어도, 이 논문이 실제로 무엇을 바꾸려 했는지만 따라가면 된다. 아래 용어만 먼저 잡아두자:

- `RNN / 순환 신경망`: 더 오래된 시퀀스 모델이다. 문장을 처리할 때 한 단어씩 순서대로 읽어야 한다.
- `attention`: 많은 정보 중에서 지금 가장 봐야 할 부분을 골라내는 메커니즘이다. 일단은 "중요한 부분을 골라 다시 보는 방식" 정도로 이해하면 충분하다.
- `Query / Key / Value`: attention 안의 세 역할이다. Query는 "지금 무엇을 찾고 있는가", Key는 "각 정보가 어떤 표식을 달고 있는가", Value는 "실제로 가져오는 내용"에 가깝다.
- `Transformer`: attention을 중심으로 세운 전체 아키텍처다. 순환에 기대지 않고, 각 위치가 다른 위치를 직접 볼 수 있다.
- `병렬 처리`: 더 똑똑하다는 뜻이 아니라, RNN처럼 줄 서서 처리하지 않고 많은 위치를 동시에 처리할 수 있다는 뜻이다.

## 1. 한 문장 요약

Transformer 이전에는 AI가 언어를 처리하는 방식이 마치 손가락으로 한 단어씩 짚어가며 책을 읽는 것과 같았다. 100번째 단어에 도달하면 1번째 단어가 무슨 내용이었는지 이미 흐려져 있다. 문장이 길수록 망각은 심해진다. 이것이 Recurrent Neural Networks(RNN, 이전 세대의 AI 아키텍처)의 근본적인 병목이었다.

저자들은 간단한 질문을 던졌다: **왜 꼭 순서대로 읽어야 하는가?**

RNN이 토큰을 단계별로 처리해야 하는 것과 달리, Transformer는 입력 전체를 병렬로 처리하며 임의의 두 위치 사이의 관계를 직접 모델링한다. 줄 설 필요도 없고, 이전 단어가 끝날 때까지 기다릴 필요도 없다.

논문은 이 핵심 능력을 "attention"이라 불렀다. 제목이 말하는 것은 "모델에 문자 그대로 attention 외에 아무것도 없다"가 아니다. 시퀀스 모델링에서 attention이 처음으로 주연으로 승격되어, 더 이상 순환(recurrence)이나 합성곱(sliding window로 지역적 특징을 추출하는 방법)을 뼈대로 필요로 하지 않게 되었다는 뜻이다.

## 2. Attention이 실제로 하는 일

시끄러운 바에 들어갔다고 상상해 보자. 스무 명이 동시에 이야기하고 있다. 우리의 뇌는 모든 목소리에 균등하게 주의를 기울이지 않는다. 누군가 이름을 부르면, 귀가 즉시 그 방향으로 고정된다. 다른 모든 소리는 배경 소음으로 사라진다.

Transformer는 모든 단어에 대해 같은 일을 한다. 논문은 세 가지 역할을 정의한다:

- **Query**: 이 단어가 찾고 있는 것. 마치 "방금 내 이름을 부른 사람이 누구지?"라고 찾는 귀와 같다
- **Key**: 이 단어가 제공할 수 있는 것. 바에 있는 각 사람의 음성적 특징과 같다
- **Value**: 이 단어가 담고 있는 실제 내용. 그 사람이 실제로 하고 있는 말과 같다

각 단어의 Query는 다른 모든 단어의 Key와 대조된다. 매칭 점수가 높으면 해당 단어의 Value에서 더 많은 정보를 가져온다. 매칭 점수가 낮으면 사실상 무시된다.

논문이 제시하는 공식은 Scaled Dot-Product Attention이라 불린다:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

공식을 보고 당황하지 않아도 된다. 하나씩 풀어보자:

- **QK^T**: Q와 K의 내적. 내적이란? 두 숫자 리스트를 원소별로 곱한 다음 합산하는 것이다. 예를 들어 [1, 2]와 [3, 4]의 내적은 1x3 + 2x4 = 11이다. 결과가 클수록 두 단어의 관련성이 높다. 이 단계에서는 모든 단어 쌍에 대해 "매칭 점수"를 계산한다
- **/ sqrt(d_k)**: 스케일링 인수로 나눈다. d_k는 벡터의 길이다(벡터는 "무언가를 설명하는 숫자 리스트"로 생각하면 된다 -- 예를 들어 한 단어의 의미를 설명하는 64개의 숫자). 왜 나눌까? 숫자 리스트가 길어질수록 내적 값이 커지는 경향이 있기 때문이다. 스케일링 없이는 차원이 높을수록 내적의 분산이 커져, softmax가 포화 상태에 빠지고(거의 모든 확률이 한 단어에 집중), gradient(모델이 자체 파라미터를 조정하는 데 사용하는 신호)가 줄어들어 학습이 불안정해진다
- **softmax**: 점수 집합을 합이 1인 확률로 변환한다. 예를 들어 세 단어의 점수가 [10, 2, 1]이면, softmax는 이를 대략 [0.99, 0.007, 0.003]으로 변환한다. 가장 높은 점수의 단어가 거의 모든 attention을 차지하고, 나머지는 0에 가깝게 밀린다
- **x V**: 이 확률을 사용해 각 단어의 실제 내용을 가중 결합한다. 높은 확률의 단어는 더 많이 기여하고, 낮은 확률의 단어는 적게 기여한다. 최종 출력은 핵심 정보를 융합한 새로운 벡터다

Python(PyTorch 기반)으로 구현하면:

```python showLanguage
import math
import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    d_k = key.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ value
```

불과 몇 줄의 코드다. 이후 산업을 뒤바꾼 수많은 역량이 이 몇 가지 연산 위에 구축되었다.

## 3. Multi-Head Attention: 여러 각도에서 동시에 바라보기

하나의 attention head는 한 가지 유형의 관계 패턴에 고정되기 쉽다. 하지만 언어는 하나의 문장에 여러 층위의 의미를 담고 있다.

"어제 고양이가 매트 위에 앉았다"를 예로 들어보자:
- "고양이"와 "앉았다"는 주어-동사 관계 (누가 무엇을 했는지)
- "어제"와 "앉았다"는 시간 관계 (언제 일어났는지)
- "위에"와 "매트"는 공간 관계 (어디서 일어났는지)

하나의 head에게 이 모든 층위를 동시에 처리하라고 하는 것은 무리한 요구다. 논문의 해법은 multi-head 메커니즘이다: 8개의 head를 파견해 병렬로 실행하여, 모델이 서로 다른 부분공간에서 문장을 동시에 관찰할 기회를 주고, 마지막에 결과를 연결(concatenate)한다.

논문의 공식:

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

풀어보면:
- **head_1, ..., head_h**: 8개의 head가 각각 독립적으로 하나의 attention 계산을 수행하여 8개의 별도 결과를 생성한다
- **Concat**: 8개의 결과를 모두 끝에서 끝으로 이어 붙여 하나의 긴 벡터를 만든다
- **W^O**: 이어 붙인 긴 벡터를 원래 차원으로 다시 투영하는 선형 변환("행렬을 곱한다"고 생각하면 된다). 마치 관리자가 8명의 조사관으로부터 보고를 듣고 하나의 종합 결론을 내리는 것과 같다

```python showLanguage
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = torch.softmax(scores, dim=-1)
        heads = weights @ v

        batch_size, _, target_len, _ = heads.shape
        merged = heads.transpose(1, 2).contiguous()
        merged = merged.view(batch_size, target_len, self.num_heads * self.d_head)
        return self.out_proj(merged)
```

논문의 파라미터: 모델은 각 단어를 설명하는 데 512개의 숫자를 사용하고(d_model = 512), 8개의 head가 각각 64개의 숫자를 받는다(512 / 8 = 64). 8개 head의 총 계산량은 단일 512차원 head와 대략 같지만, 표현력은 훨씬 크다. 같은 비용으로 다중 관점의 이해를 얻는다. 매우 좋은 거래다.

## 4. Positional Encoding: 모델에게 단어 순서 알려주기

Transformer는 전체 문장을 병렬로 처리하므로 빠르지만, 그 대가로 단어 순서를 잃는다. 추가적인 위치 정보가 없으면, attention 메커니즘만으로는 "고양이가 물고기를 먹었다"와 "물고기가 고양이를 먹었다"의 차이를 구분할 수 없다. 이래서는 안 된다.

해결책: 각 위치에 대해 고유한 "주소 코드"를 생성하여 단어의 벡터에 더한다. 모델은 더 이상 단순히 "고양이"와 "물고기"를 보는 것이 아니라 "위치 1의 고양이"와 "위치 3의 물고기"를 본다.

논문은 사인 함수와 코사인 함수를 사용하여 이 인코딩을 생성한다:

$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
\operatorname{PE}(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

공식이 겁나 보이지만, 핵심 아이디어는 직관적이다:
- **pos**: 문장에서의 단어 위치 (1번째, 2번째, 3번째, ...)
- **i**: 벡터의 몇 번째 차원인지. 짝수 위치는 sin, 홀수 위치는 cos을 사용한다
- **10000^(2i/d_model)**: 차원에 따라 변하는 스케일링 인수. 저차원은 빠르게 진동하고, 고차원은 느리게 진동한다. 시계에 비유하면: 초침은 1분에 한 바퀴를 돌고, 시침은 12시간이 걸린다. 서로 다른 "바늘"이 서로 다른 시간 스케일을 커버하며, 함께 사용하면 어떤 순간이든 정확히 짚어낼 수 있다

최종 결과: 각 위치가 고유한 숫자 지문을 받고, 모델은 이 지문을 사용해 단어 순서를 구분한다.

```python showLanguage
import math
import torch


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    encoding = torch.zeros(seq_len, d_model)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding
```

왜 하필 사인과 코사인일까? 이 함수들에는 우아한 수학적 성질이 있기 때문이다: 고정된 거리만큼 떨어진 두 위치의 인코딩 관계는, 그 위치가 문장의 처음에 있든 끝에 있든 동일하다. 모델은 "위치 3과 위치 8의 관계"를 외울 필요가 없다 -- "5개 위치 떨어짐"이 무엇을 의미하는지만 학습하면 된다. 논문 팀은 모델이 positional encoding을 스스로 학습하게 하는 방법도 시도했는데, 결과는 비슷했다. 하지만 사인파 버전에는 장점이 하나 더 있다: 학습 중에 본 적 없는 길이의 문장도 처리할 수 있다는 것이다.

## 5. Encoder와 Decoder

Transformer의 전체 아키텍처는 두 부분으로 나뉜다.

**Encoder**(6개의 층을 쌓은 구조)는 입력을 이해하는 역할을 한다. 각 층에는 두 개의 하위 층이 있다: multi-head self-attention 하나, feed-forward 네트워크 하나. 각 하위 층에는 두 가지 보호 메커니즘이 있다:

- **잔차 연결(Residual connection)**: 하위 층의 입력을 출력에 직접 더한다. 즉, x + Sublayer(x)이다. 왜? 사진에 필터를 적용한다고 생각해 보자. 필터 결과가 나쁘더라도, 잔차 연결이 있으면 원본 이미지를 여전히 볼 수 있다. 깊은 네트워크에서는 정보가 매 층에서 변환되어 여섯 번째 층에 이르면 알아볼 수 없게 될 수 있다. 잔차 연결은 원래 신호가 "지름길"을 통해 깊은 층까지 직접 도달하게 하여, 정보가 전달 과정에서 소실되는 것을 방지한다
- **Layer normalization** (LayerNorm): 값을 균일한 범위로 재조정하여, 일부 숫자가 무한대로 폭발하고 다른 숫자가 0으로 사라지는 것을 방지한다. 시험 점수를 표준화하는 것과 비슷하다 -- 원래 점수가 아무리 다르더라도, 표준화하면 비교 가능한 척도가 된다

**Decoder**(6개의 층을 쌓은 구조)는 출력을 생성하는 역할을 한다. 구조는 encoder와 유사하지만, 두 가지 중요한 추가 요소가 있다:

첫째, **cross-attention**: decoder가 각 단어를 생성할 때 encoder의 출력을 되돌아본다. 번역 시나리오에서, 이는 영어를 쓰면서 중국어 원문을 다시 확인하는 것과 같다.

둘째, **masking**: 3번째 단어를 생성할 때, 모델은 처음 2개 단어만 볼 수 있다. 4번째 위치 이후는 차단된다(attention 점수를 음의 무한대로 설정하면 softmax 이후 0이 된다). 논리는 간단하다: 글을 쓸 때 다음 단어는 아직 쓰지 않았으므로 미리 볼 수 없다.

```python showLanguage
from typing import Optional

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder(self.embedding(src_token_ids))
        hidden = self.decoder(self.embedding(tgt_token_ids), memory, tgt_mask=tgt_mask)
        return self.output_proj(hidden)
```

놓치기 쉬운 구성 요소가 하나 더 있다: feed-forward 네트워크. 공식은 FFN(x) = max(0, xW1 + b1)W2 + b2이다. 쉽게 말하면: 각 단어의 512차원 벡터를 2048차원으로 확장하고(행렬을 곱하고 바이어스를 더한 다음), ReLU를 통과시키고(모든 음수는 0이 되고 양수는 그대로 유지), 다시 512차원으로 압축한다. ReLU 단계가 핵심이다: "비선형성"을 도입하여, 직선으로는 절대 포착할 수 없는 복잡한 패턴을 모델이 학습할 수 있게 해준다. 모든 연산이 선형이라면, 여러 층을 쌓아도 수학적으로는 단일 층과 동일하다. 비선형성은 복잡성을 모델링하기 위한 전제 조건이다.

## 6. 학습 세부 사항

아키텍처를 설계했으면, 어떻게 학습시킬까? 논문은 이 부분에도 심혈을 기울였다.

**하드웨어**: NVIDIA P100 GPU 8개. 기본 모델은 12시간(100,000 스텝), 대형 모델은 3.5일(300,000 스텝) 학습했다. 오늘날 기준으로 보면 놀라울 정도로 낮은 비용이다.

**Optimizer**: Adam(모델 파라미터를 자동으로 조정하는 알고리즘)을 사용하되, 영리한 learning rate 스케줄을 적용했다. Learning rate는 모델이 매번 업데이트할 때 얼마나 큰 보폭을 내딛는지를 결정한다. 보폭이 너무 크면 최적점을 지나칠 위험이 있고, 너무 작으면 시간이 낭비된다. 논문의 전략: 처음 4,000 스텝 동안 점진적으로 올리고(warmup), 초반에 과도하게 공격적인 업데이트를 피한다. 4,000 스텝 이후에는 스케줄에 따라 점차 감소시켜 학습 후반을 안정화한다. 올랐다가 내리기 -- 전반부에는 대담한 탐색, 후반부에는 세밀한 조정.

**정규화**: 두 가지 기법을 사용한다. 첫째는 Dropout: 학습 중에 뉴런(네트워크 내의 계산 노드)의 10%를 무작위로 비활성화하여, 모델이 특정 경로에 의존하지 않고 더 견고한 특징을 학습하도록 강제한다. 둘째는 label smoothing(epsilon = 0.1): 모델에게 "정답의 확률이 100%"라고 알려주는 대신, "정답에 90%, 나머지 선택지에 10%를 분배"한다고 알려준다. 이렇게 하면 실제로 perplexity(모델이 얼마나 "불확실한지" 측정하는 지표) 자체는 나빠지지만, 번역 품질은 향상된다. 직관적으로, 자신이 100% 확신하지 않는다고 인정하는 모델이 과신하는 모델보다 더 신뢰할 수 있다.

**결과**: 논문은 BLEU 점수(기계 번역의 표준 지표로, 기계 출력이 인간 번역에 얼마나 가까운지 측정하며 최대 100점)로 성능을 평가한다. 영어-독일어: 28.4점. 영어-프랑스어: 41.8점. 둘 다 당시 최고 기록을 갱신했다. 학습 비용은 이전 접근 방식 대비 1~2자릿수 낮았다. 더 빠르고, 더 강하고, 더 저렴하다.

## 7. 나의 소감

이 논문을 읽고 나서 몇 가지가 인상적이었다.

첫째, 이 논문의 핵심 통찰은 놀라울 정도로 간결하다: 순차 처리의 짐을 버리고, attention 메커니즘이 임의의 두 위치 사이의 관계를 직접 모델링하게 하는 것이다. Self-Attention, 잔차 연결, Layer Normalization -- 이 중 어느 것도 새로운 발명이 아니었다. 진정한 돌파구는 새로운 도구를 발명한 것이 아니라, "이 간단한 빌딩 블록들을 조합하면 충분하다"는 것에 저자들이 기꺼이 승부를 건 것 -- 그리고 실험으로 스스로를 증명한 것이다.

둘째, 이것을 실제 Python 코드로 옮겨 적으면서 모든 설계 결정에 대한 이해가 깊어졌다. Scaled Dot-Product Attention을 직접 작성하면, 그 sqrt(d_k) 스케일링이 왜 중요한지 피부로 느끼게 된다. masking을 구현하면, 자기회귀 생성 제약이 정확히 어디서 오는지 이해하게 된다. 논문을 열 번 읽는 것보다 직접 한 번 구현하는 것이 낫다.

셋째, 나를 진정으로 감탄하게 한 것은 이후 얼마나 많은 모델을 탄생시켰는가가 아니라, 2017년에 문제를 재정의한 것이었다: "어떻게 문장을 순서대로 기억할 것인가"에서 "어떻게 모든 위치가 가장 필요한 정보를 직접 찾게 할 것인가"로. GPT, BERT, T5, LLaMA -- 이 모두가 그 재정의의 산물이다.

충분히 좋은 아키텍처가 얼마나 멀리 갈 수 있는지는, 얼마나 많은 사람이 그 위에 계속 쌓아 올릴 의향이 있는지에 달려 있다.

이 논문이 우리에게 그 아키텍처를 주었다.

《Attention Is All You Need》 (어텐션만 있으면 충분하다).

---

**논문 읽기 시리즈**

- [《Sequence to Sequence Learning with Neural Networks》](/ko-KR/posts/sequence-to-sequence-learning-with-neural-networks/) (신경망을 이용한 시퀀스-투-시퀀스 학습) — Encoder-decoder 패러다임의 확립
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/ko-KR/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) (정렬과 번역을 공동으로 학습하는 신경 기계 번역) — Attention의 기원
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전 학습 패러다임의 확립
- [《Scaling Laws for Neural Language Models》](/ko-KR/posts/scaling-laws-for-neural-language-models/) (신경 언어 모델을 위한 스케일링 법칙) — 스케일의 수학: 왜 더 큰 모델이 예측 가능하게 더 좋은가
- [《Language Models are Few-Shot Learners》](/ko-KR/posts/language-models-are-few-shot-learners/) (언어 모델은 퓨샷 학습자다) — 더 큰 모델, 맥락에서 능력을 이끌어내는 데 더 뛰어남
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — 컴퓨팅 예산을 현명하게 쓰는 법

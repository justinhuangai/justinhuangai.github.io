---
title: "《Attention Residuals》: 잔차 연결도 어텐션처럼 만들기"
date: "2026-03-19T16:49:27+08:00"
category: "Technical Report Reading"
description: "Kimi Team의 Attention Residuals 기술 리포트 읽기: 왜 residual connection도 attention처럼 바뀌어야 하는지, 그리고 Full AttnRes / Block AttnRes가 그 아이디어를 어떻게 학습 가능하고 배포 가능한 시스템으로 만드는지"
tags: [technical-report-reading, residual-connections, transformer, AI, LLM, python]
pinned: false
---

2026년 3월 16일, Kimi Team은 arXiv에 기술 리포트 한 편을 올렸다: [《Attention Residuals》](/papers/2603.15031v1.pdf) (어텐션 잔차).

이 리포트에서 저자들이 진짜로 힘을 준 부분은 구조만 봐도 드러난다. 단순히 "새 모듈 하나를 제안했다"가 아니다. `motivation -> AttnRes -> Block AttnRes -> infrastructure -> experiments -> discussion`의 순서로, residual connection이 실제로 무엇을 하고 있는지를 처음부터 다시 설명한다.

## 0. 먼저 몇 가지 용어부터

머신러닝 배경이 전혀 없어도, 이 리포트가 실제로 파고드는 문제만 따라가면 충분하다. 아래 순서대로 감을 잡으면 된다:

- `Transformer`: 오늘날 대부분의 대형 언어 모델이 사용하는 기본 아키텍처다. 층층이 정보를 처리하는 기계라고 생각하면 된다.
- `hidden state`: 특정 층에서 모델이 들고 있는 내부 중간 표현이다. 대략 "모델이 지금 머릿속에 적어 둔 임시 메모"에 가깝다.
- `residual connection / 잔차 연결`: 이전 내용을 보존한 채, 현재 층에서 새로 계산한 것을 그 위에 더하는 경로다.
- `residual / 잔차`: 그 잔차 연결 안에서 새로 더해지는 증가분에 더 가깝다.
- `attention`: 많은 정보 중에서 지금 가장 중요하게 봐야 할 부분을 골라내는 메커니즘이다.
- `PreNorm`: 층에 들어가기 전에 값의 스케일을 먼저 정리하고, 그다음 계산을 수행하는 방식이다.

## 1. 한 문장으로 요약하면

이 기술 리포트는 한 가지 질문을 제기한다:

**Transformer가 이미 시퀀스 차원에서는 recurrence를 attention으로 대체했는데, 왜 깊이 차원에서의 정보 집계는 아직도 고정된 덧셈에 머물러 있는가?**

현대 LLM은 거의 모두 비슷한 층 구조를 쓴다. 먼저 PreNorm을 하고, 그다음 residual 경로를 지난다. 쉽게 말하면 값의 스케일을 정리한 뒤 새로운 계산 결과를 원래 입력에 더해 준다. 우리는 보통 이 구조를 "학습을 안정화하는 장치" 정도로 기억한다. 깊은 네트워크가 훈련 도중 무너지지 않게 해 주는 장치 말이다. 그런데 저자들은 residual connection이 그보다 더 중요한 역할도 하고 있다고 말한다:

**깊이를 따라 정보가 어떻게 모이느냐를 residual connection이 결정한다는 것이다.**

아래 수식이 낯설다면 거기서 멈출 필요는 없다. 바로 뒤의 풀어서 쓴 설명이 핵심이다.

표준 residual 규칙은 아주 단순하다:

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1})
$$

이를 두 부분으로 나누어 보면:

- $h_{l-1}$: 이전 층이 이미 만들어 둔 기존 내용
- $f_{l-1}(h_{l-1})$: 현재 층이 새로 계산한 증가분, 즉 "잔차"라는 말에 더 가까운 부분

그리고 이 둘을 다시 합치는 전체 동작이 더 정확하게는 residual connection이다.

이 점화식을 펼치면 다음과 같다:

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$

사람 말로 옮기면, $l$번째 층이 보는 입력은 사실상 "embedding + 이전 모든 층 출력의 균등 합"이다. 모든 층의 가중치가 1이다. 선택도 없고, 억제도 없고, "이번 단계에서는 17층보다 3층을 더 참고하자" 같은 메커니즘도 없다.

AttnRes의 핵심 아이디어는 한 문장으로 끝난다:

**residual을 고정 덧셈이 아니라, 깊이 방향의 softmax attention으로 바꾸자는 것이다.**

## 2. 기존 residual은 뭐가 문제인가

이 기술 리포트에서 가장 중요한 것은 새 수식을 제안했다는 사실이 아니다. 사람들이 너무 익숙해져서 더 이상 문제라고 여기지 않던 것을 다시 문제로 만들었다는 점이다.

표준 residual은 오랫동안 "최적화 안정화 장치"로만 취급되어 왔다. 그래디언트만 잘 지나가면 역할을 다한 셈이었다. 하지만 정보 흐름 관점에서 보면, 이 경로는 놀랄 만큼 거칠다.

계속 수정되는 문서를 상상해 보자. 매번 이전 버전에서 정말 relevant 한 부분만 골라 정리하는 대신, 모든 이전 초안을 통째로 문서 뒤에 붙여 버린다고 해 보자. 20번째 수정본쯤 되면 3번째 버전의 중요한 통찰은 여전히 "존재"하긴 하지만, 두꺼워진 더미 속에 묻혀 버린다.

저자들이 지적하는 PreNorm 문제는 바로 이것이다. 리포트는 SiameseNorm의 관찰을 인용하면서, PreNorm 아래에서는 `hidden state`의 크기가 깊이에 따라 대략 $O(L)$로 커진다고 강조한다. hidden state는 결국 각 층이 들고 있는 내부 메모다. 그 결과:

- 뒤쪽 층으로 갈수록 점점 더 부푼 "역사적 총합"을 보게 되고
- 초기 층의 정보는 사라지지는 않지만 계속 희석되며
- 뒤쪽 층이 존재감을 가지려면 더 큰 크기의 출력을 내야 한다

리포트는 이 현상을 `PreNorm dilution`이라고 부른다. 정말 정확한 이름이다. 문제는 그래디언트가 끊기는 것도 아니고, 학습이 폭발하는 것도 아니다. 각 층의 상대적 기여가 점점 묽어지는 것이다.

나는 이 리포트의 밑바닥에 흐르는 한 가지 사고가 특히 마음에 들었다. 시퀀스 차원에서는 우리는 이미 "과거 토큰을 모두 똑같이 취급하는 방식"에 만족하지 않았고, 그래서 attention이 등장했다. 그렇다면 왜 깊이 차원에서는 여전히 "모든 이전 층을 같은 가중치로 더하는 방식"을 받아들이고 있는가?

## 3. AttnRes는 실제로 무엇을 하는가

AttnRes의 형태는 꽤 깔끔하다. $l$번째 층은 더 이상 이전 모든 출력의 합을 기계적으로 받지 않는다. 대신 과거 표현들에 대해 가중 선택을 수행한다:

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i
$$

가중치 $\alpha_{i \to l}$는 softmax에서 나온다. softmax가 익숙하지 않다면 이렇게 생각해도 된다. 여러 점수를 받아서 합이 1인 가중치로 바꾸는 과정이다. 그래서 모델은 "여기를 더 보고, 저기는 덜 보자"를 분명하게 표현할 수 있다:

$$
\alpha_{i \to l} = \operatorname{softmax}\left(w_l^T \operatorname{RMSNorm}(k_i)\right)
$$

attention을 처음 접한다면 가장 쉬운 직관은 이렇다:

- `query`: 지금 현재 층이 찾고 있는 것
- `key`: 각 과거 층이 달고 있는 일종의 색인 표식
- `value`: 실제로 가져와서 집계하는 내용

여기서 중요한 설계 포인트는 세 가지다.

첫째, **query는 현재 hidden state에서 즉석 계산하지 않는다. 층마다 학습되는 pseudo-query 벡터 $w_l$를 둔다.**  
이건 약간 반직관적이다. 우리는 보통 attention을 보면 query가 현재 입력에서 나와야 한다고 생각한다. 그런데 저자들은 query를 토큰별 동적 벡터가 아니라 층별 파라미터로 잡았다. 그 장점은 같은 block 안의 여러 query를 미리 배치로 계산할 수 있다는 점이고, 바로 그 때문에 뒤에서 설명할 인프라 최적화가 가능해진다.

둘째, **key와 value는 이전 층의 출력에서 직접 온다.**  
즉 입력 의존성이 사라지는 것은 아니다. query가 아니라 각 층의 표현 그 자체에 입력 의존성이 남아 있다. 서로 다른 샘플은 서로 다른 이전 층 출력을 만들기 때문에, 깊이 attention은 결국 여전히 input-dependent 하다.

셋째, **key 앞에 RMSNorm을 붙인다.**  
이건 작은 설계 같지만 꽤 중요하다. 정규화를 하지 않으면 크기가 큰 층이 점곱에서 자동으로 유리해진다. 그러면 attention 가중치는 "누가 더 relevant 한가"보다 "누가 더 크게 말하는가"를 반영하게 된다.

Python(PyTorch 기반)으로 쓰면 대략 이렇게 된다:

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

겉으로 보면 이것은 "attention을 residual 위에 올린 것"처럼 보인다. 하지만 나는 더 정확한 표현이 따로 있다고 본다:

**residual connection을 고정된 누산기에서, 선택적으로 깊이를 검색하는 장치로 바꾼 것이다.**

## 4. 이 리포트는 아이디어도 주고, 엔지니어링도 준다

**한 문장 결론: 이 리포트는 Full AttnRes를 제안했고, 그 아이디어를 실제로 학습 가능하고 배포 가능하며 비용까지 계산된 엔지니어링 해법으로 밀어 붙였다.**

Full AttnRes는 모든 층이 자신보다 앞선 모든 층을 보게 한다. 이론적으로는 이해하기 쉽고, 순수 계산량만 보면 아주 무섭지도 않다. 네트워크 깊이 $L$은 보통 시퀀스 길이 $T$보다 훨씬 작기 때문에, 저자들도 $O(L^2 d)$라는 계산 자체가 가장 큰 공포는 아니라고 말한다.

문제는 대규모 학습에서 터진다:

- activation recomputation 때문에 원래 버려도 되던 중간 출력이 다시 꼭 저장해야 하는 대상이 되고
- pipeline parallelism 때문에 그 층 간 표현들이 stage 사이를 오가야 하며
- 모든 층이 모든 이전 층을 봐야 하면 통신과 캐시 압력이 급격히 커진다

그래서 저자들은 **Block AttnRes**를 제안한다.

방법은 $L$개의 층을 $N$개의 block으로 나누는 것이다. block 내부에서는 먼저 일반적인 합으로 block representation을 만들고, block 사이에서는 attention을 적용한다. 즉:

- Full AttnRes는 모든 과거 층을 본다
- Block AttnRes는 과거 block들의 요약과 현재 block 안의 부분합을 본다

요약하면, 더 세밀한 층 단위 attention을 요약 수준의 block attention으로 바꿔 확장성을 얻는 셈이다.

그리고 저자들은 "block으로 나눴으니 메모리가 줄었다" 수준에서 멈추지 않는다. 시스템 쪽 장부도 실제로 계산한다:

- 학습 단계에서는 **cross-stage caching**을 써서 pipeline 안에서 과거 block을 반복 전송하지 않게 하고
- 추론 단계에서는 **two-phase computation**을 사용하며
- 1단계에서는 inter-block attention을 병렬 계산하고
- 2단계에서는 intra-block lookback을 순차 계산한 다음 online softmax merge로 결과를 합친다

부록과 `table/memory_access.tex`에 이 리포트에서 가장 하드코어한 숫자들이 나온다. 리포트의 대표 설정에서는:

- 표준 residual: 층당 residual 메커니즘 I/O가 `3d`
- naive Full AttnRes: `130d`
- 최적화된 Full AttnRes: `24d`
- Block AttnRes: `5.5d`
- mHC: `34d`

이 숫자들은 많은 것을 말해 준다. Block AttnRes가 "표준 residual만큼 싸다"는 뜻은 아니다. 하지만 이미 "현실적으로 불가능하다"에서 "현실 시스템에서 시도해 볼 만하다"로 내려왔다. 그리고 실제 측정된 오버헤드도 작다:

- 학습 wall-clock overhead는 4% 미만
- 추론 latency overhead는 2% 미만

그래서 나는 이것이 진짜 시스템 지향 기술 리포트처럼 읽힌다고 느꼈다. 아이디어는 새롭지만 비용 계산은 흐릿한 논문이 많다. 이 리포트는 비용 계산까지 신경 쓴다.

## 5. 실험에서 진짜 봐야 할 것

**주 실험의 개별 점수보다 더 중요한 것은, AttnRes가 스케일링 추세, 학습 동역학, 다운스트림 성능 전반에서 같은 방향의 신호를 보여 준다는 점이다.**

### 5.1 스케일링 법칙 결과: 한 번 운 좋게 이긴 것이 아니다

저자들은 먼저 다섯 개의 모델 규모에서 scaling-law 실험을 수행하며 Baseline, Full AttnRes, Block AttnRes를 비교한다.

피팅된 곡선은 다음과 같다:

- Baseline: $1.891 \times C^{-0.057}$
- Block AttnRes: $1.870 \times C^{-0.058}$
- Full AttnRes: $1.865 \times C^{-0.057}$

여기서 가장 중요한 것은 기울기 차이가 얼마나 되느냐가 아니다. 핵심은 이것이다:

**AttnRes는 compute 범위 전반에서 일관되게 더 낮다.**

리포트는 전달하기 쉬운 요약 문장도 준다. `5.6 PFLOP/s-days` 예산 지점에서 Block AttnRes의 loss는 baseline이 약 `1.25x` 더 많은 compute를 써야 겨우 도달할 수준이라는 것이다.

즉 이것은 "특정 모델 크기에서 우연히 튜닝이 잘 맞은 결과"로 보이지 않는다. 꽤 안정적인 규모 이득으로 읽힌다.

### 5.2 메인 모델 실험은 장난감이 아니다

주 실험은 작은 toy benchmark가 아니다. Kimi Linear 기반의 큰 설정을 사용한다:

- `48B total / 3B activated parameters`
- 27개의 Transformer blocks, 즉 54개 층
- 8-of-256 routed experts + 1개의 shared expert
- `1.4T tokens` 사전학습

이것이 중요한 이유는, 저자들이 단지 작은 모델에서 예쁜 곡선을 그린 것이 아니라 이 residual 재설계를 실제 대규모 학습 레시피 안에 넣어 보았다는 뜻이기 때문이다.

### 5.3 학습 동역학에서 출력 크기가 더 이상 폭주하지 않는다

Baseline에서는 output magnitude가 깊이에 따라 계속 상승한다. 그림 속 값은 극적이다. 앞 block의 `0.04`, `0.06`, `0.10`이 뒤 block에서 `10.47`, `12.15`까지 올라간다. 이것이야말로 PreNorm dilution이 눈에 보이는 형태다.

반면 Block AttnRes는 전혀 다른 모양을 보인다. block 경계에서 일종의 주기적 리셋이 생기고, 크기는 대략 `0.21`에서 `1.91` 사이를 오갈 뿐이다. 같은 식의 폭주 상승이 없다.

이게 중요한 이유는 AttnRes가 마지막 benchmark에서 "몇 점 더 얻었다"는 수준이 아니라, 학습 중 표현이 깊이를 따라 쌓이는 방식 자체를 바꾸고 있다는 신호이기 때문이다.

### 5.4 다운스트림 태스크: 특히 추론과 코드에서 이득이 크다

사전학습 이후 AttnRes는 리포트에 나온 모든 평가에서 baseline보다 나쁘지 않으며, 특히 다음 지표들이 눈에 띈다:

- MMLU: `73.5 -> 74.6`
- GPQA-Diamond: `36.9 -> 44.4`
- Math: `53.5 -> 57.1`
- HumanEval: `59.1 -> 62.2`
- C-Eval: `79.6 -> 82.5`

특히 GPQA, Math, HumanEval처럼 다단계 추론이나 코드 생성이 중요한 태스크에서 상승폭이 더 크다. 저자들의 설명은 이렇다. 뒤쪽 층이 앞쪽 층의 표현을 더 선택적으로 회수할 수 있다면 compositional tasks가 더 큰 이득을 본다. 나는 이 설명이 설득력 있다고 본다.

복잡한 추론에서 진짜 문제는 정보가 아예 없는 것이 아니라, 중요한 정보가 네트워크 깊은 곳에 묻혀 버리는 것이다.

## 6. Ablation은 무엇을 말해 주는가

**ablation의 핵심 결론은 "더 촘촘하게 연결하면 더 세다"가 아니라, 깊이 방향에서 입력 의존적인 선택적 집계를 한다는 사실 자체가 효과를 만든다는 점이다.**

이 리포트의 ablation이 좋은 이유는 단지 "효과가 있다"를 보여 주는 데서 멈추지 않고, 왜 효과가 있는지까지 건드리기 때문이다.

가장 흥미로운 몇 가지 결과를 보면:

- **DenseFormer는 1.767로, baseline 1.766과 거의 같다.**  
  단지 모든 이전 층에 접근할 수 있다는 사실만으로는 부족하다는 뜻이다. 중요한 것은 가중치가 input-dependent 하냐는 점이다.

- **mHC는 1.747까지 내려간다.**  
  깊이 차원에서의 동적 혼합이 실제로 유효하다는 뜻이다.

- **Full AttnRes는 1.737까지 내려간다.**  
  baseline, DenseFormer, mHC보다 더 낮다. 명시적인 softmax depth attention이 더 강한 방향이라는 뜻이다.

- **SWA는 최근 창만 보는 방식인데 1.764에 그친다.**  
  이건 중요하다. AttnRes의 이득이 단순히 "최근 몇 층을 더 본다"에 있지 않고, 필요할 때 더 먼 층까지 선택적으로 볼 수 있다는 데 있음을 보여 준다.

- **block size를 2, 4, 8로 바꿔도 loss는 모두 1.746 안팎이다.**  
  그래서 저자들이 최종적으로 약 8개 block을 택한 것이다. 감이 아니라 성능과 엔지니어링 사이의 sweet spot이다.

- **input-dependent query 버전은 1.731로 Full AttnRes보다도 더 좋다.**  
  이건 특히 흥미롭다. 현재 리포트의 pseudo-query 설계가 성능 상한이 아니라는 뜻이기 때문이다. 이것은 인프라 최적화를 위해 고른 절충안이다. 다시 말해, 저자들은 더 강한 버전을 몰라서 안 쓴 것이 아니라, 확장 가능한 버전을 의식적으로 택했다.

그래서 이 리포트가 재미있다. 본문, ablation, 시스템 설계를 함께 보면 진짜 선택 기준이 드러난다. 무작정 최저 loss만 쫓는 것이 아니라, 충분히 강하면서 실제로 훈련 가능한 설계를 찾고 있는 것이다.

## 7. 이 리포트를 어떻게 볼 것인가

첫째, 이 리포트의 가장 중요한 점은 새 모듈을 발명했다는 사실이 아니다. residual connection을 다시 "최적화 안정화 도구"가 아니라 "정보 라우팅 메커니즘"으로 끌어올렸다는 점이다.

이 관점을 받아들이는 순간 많은 것이 새로 보인다. residual은 단지 gradient highway가 아니다. depth aggregation rule이기도 하다. 그러면 자연스럽게 이런 질문들이 따라온다:

- 각 층은 정말로 앞선 층들에 선택적으로 접근할 수 있는가?
- 깊이 차원에도 attention sink 같은 현상이 있는가?
- 기존 residual 변형들은 사실상 depth-wise linear attention에 가까운 것 아닌가?

바로 그 지점에서 discussion 섹션이 흥미로워진다. 저자들은 여러 residual 변형을 `depth mixing matrix`라는 관점에서 다시 묶어 내고, 한 걸음 더 나아가 이렇게 말한다:

**기존 방법들 상당수는 본질적으로 깊이 차원에서 linear attention을 하고 있고, AttnRes는 깊이 차원의 softmax attention을 한다.**

대담한 주장이다. 하지만 꽤 영감을 주는 주장이다. Transformer가 한때 시퀀스 차원을 recurrence에서 softmax attention으로 밀어냈다면, AttnRes는 깊이 차원도 한 걸음 더 밀어 보려 한다는 뜻이기 때문이다.

둘째, 이 리포트의 기질은 "문제를 먼저 제대로 묻고, 그다음 시스템이 돌아가게 만든다"에 가깝다. 각 부품을 무조건 가장 화려하게 만들려 하지 않는다. 예를 들어 query를 token-dependent 동적 벡터가 아니라 layer-specific 파라미터로 둔 것은, 순수 성능만 보면 최강일 필요는 없다. 하지만 그 덕분에 batching, two-phase computation, pipeline cache가 가능해진다. 실제로 굴러가는 기술 리포트는 가장 과감한 국소 설계가 아니라, 전체 제약 아래에서 살아남는 선택으로 만들어지는 경우가 많다.

셋째, 이 리포트의 이 문장은 꽤 잘 짚는다:

**Why is depth-wise aggregation still fixed while everything else has become adaptive?**

정말 잘 짚은 질문이다.

## 8. 이 리포트의 경계

첫째, 이것은 여전히 **technical report / arXiv preprint**이지, 동료 심사를 거친 학회 논문이 아니다. 가장 안전한 태도는 "미래를 증명했다"가 아니라, "강력한 관점을 제시했고, 공학적으로 구현 가능한 해법을 함께 내놓았다"이다.

둘째, 대규모 결과는 Kimi Linear 계열 아키텍처에 묶여 있다. MoE, KDA/MLA 혼합 attention, Moonlight / DeepSeek-V3 스타일 학습 레시피가 함께 들어간다. 결과 자체의 가치는 줄지 않지만, 모든 dense decoder-only Transformer에 자동으로 일반화할 수는 없다는 뜻이기도 하다.

셋째, 리포트 스스로도 Full AttnRes가 더 강하다고 인정한다. Block AttnRes는 오늘날 하드웨어 제약 아래에서의 현실적 답이다. 앞으로 메모리, 대역폭, interconnect가 더 좋아지거나, 더 효율적인 depth attention 변형이 나온다면 지금의 block 설계가 최종형일 가능성은 낮다.

## 9. 마지막 인상

지난 10년간 대형 모델 아키텍처의 흐름을 아주 거칠게 요약하면:

- Seq2Seq는 "한 시퀀스를 어떻게 다른 시퀀스로 압축할 것인가?"를 물었고
- Bahdanau는 "왜 디코딩할 때 입력의 다른 위치를 다시 볼 수 없지?"를 물었고
- Transformer는 "왜 시퀀스 모델링이 꼭 recurrence에 의존해야 하지?"를 물었고
- Chinchilla는 "왜 늘어난 compute는 주로 파라미터 수에만 써야 하지?"를 물었다

그렇다면 *Attention Residuals*는 이렇게 묻는다:

**왜 깊이 방향의 정보 집계는 아직도 "모든 과거 층을 똑같이 더하는 시대"에 머물러 있는가?**

이 질문만으로도 이미 충분히 가치가 있다.

몇 년 뒤 AttnRes가 PreNorm처럼 기본 설정이 될지는 나도 모른다. 하지만 이 기술 리포트가 residual connection을 다시 생각하고, 설계하고, 최적화할 가치가 있는 대상으로 되돌려 놓았다는 점만은 꽤 확신한다.

사람들은 attention이 시퀀스 모델링을 다시 썼다고 말했다.

이 리포트는 residual을 다시 쓰려 하고 있다.

2026년 봄, Kimi 팀의 작업은 이미 한 가지를 보여준다: Scaling Laws가 병목에 가까워지는 징후를 드러내기 시작할 때, LLM의 구조 혁신은 계속해서 등장할 것이다.

---

**더 읽어보기**

- [《Sequence to Sequence Learning with Neural Networks》](/ko-KR/posts/sequence-to-sequence-learning-with-neural-networks/) (신경망을 이용한 시퀀스-투-시퀀스 학습) — encoder-decoder 패러다임의 출발점
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/ko-KR/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) (정렬과 번역을 공동으로 학습하는 신경 기계 번역) — attention 메커니즘의 기원
- [《Attention Is All You Need》](/ko-KR/posts/attention-is-all-you-need/) (어텐션만 있으면 충분하다) — attention이 주연이 되고 Transformer가 탄생한 순간
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전학습 패러다임의 확립
- [《Scaling Laws for Neural Language Models》](/ko-KR/posts/scaling-laws-for-neural-language-models/) (신경 언어 모델을 위한 스케일링 법칙) — 규모에 관한 수학
- [《Language Models are Few-Shot Learners》](/ko-KR/posts/language-models-are-few-shot-learners/) (언어 모델은 퓨샷 학습자다) — 더 큰 모델은 컨텍스트에서 더 많은 능력을 끌어낸다
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — compute를 가장 잘 쓰는 방법

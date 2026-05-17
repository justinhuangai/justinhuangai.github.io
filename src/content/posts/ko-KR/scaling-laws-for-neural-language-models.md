---
title: "《Scaling Laws for Neural Language Models》: 규모의 수학"
date: "2026-03-01T16:45:39+08:00"
category: "Paper Reading"
description: 규모의 수학 — 더 큰 모델이 예측 가능하게 더 나은 이유, 실제 Python 코드 예시 포함
tags: [paper-reading, scaling-laws, AI, LLM, python]
pinned: false
---

대형 모델 학습은 먼저 예산 배분 문제입니다. 파라미터, 데이터, compute는 모두 비용입니다. 질문은 "클수록 좋다"가 아닙니다. 다음 1달러를 파라미터에 쓸지, 데이터에 쓸지, 더 긴 학습에 쓸지입니다.

[《Scaling Laws for Neural Language Models》](/papers/2001.08361v1.pdf)는 대형 모델 학습을 경험주의에서 예산 함수로 옮겼습니다. 새 아키텍처를 발명한 논문은 아닙니다. 대신 돈을 쓰기 전에 수익 곡선을 추정하는 더 단단한 판단 도구를 줬습니다.

## 0. 먼저 몇 가지 용어부터

수식이 나오는 논문이 익숙하지 않다면, 아래 용어부터 잡아두면 뒤가 훨씬 편해진다:

- `파라미터 수`: 모델 안에 들어 있는 학습 가능한 숫자의 총량, 즉 모델 크기다.
- `데이터 양`: 학습할 때 모델에게 얼마나 많은 텍스트를 먹였는지를 뜻한다.
- `컴퓨팅 / compute`: 학습 전체에 투입한 계산량이다. 일단은 "전기요금 청구서"처럼 생각해도 된다.
- `loss / 손실`: 모델이 얼마나 틀리고 있는지를 요약한 값이다. 보통 낮을수록 좋다.
- `power law / 멱법칙`: 어떤 양이 일정한 지수 관계로 변하는 패턴이다. 로그 좌표에 그리면 직선에 가깝게 보이는 경우가 많다.
- `로그-로그 좌표`: `1, 10, 100, 1000`처럼 배수로 커지는 눈금을 쓰는 좌표계다.

## 1. 질문

2020년 초, 딥러닝 커뮤니티는 이미 더 큰 모델이 더 나은 성능을 "내는 경향이 있다"는 것을 알고 있었다. 하지만 "경향이 있다"는 과학이 아니다. 기본적인 실용적 질문에 답할 수 없었다: 컴퓨팅 예산을 두 배로 늘리면 성능이 얼마나 향상될까? 그 예산을 더 큰 모델, 더 많은 데이터, 더 긴 학습 중 어디에 써야 할까? 공식이 있을까?

이 논문은 그 질문들에 답했다. 직감도 아니고, 경험 법칙도 아니고 — 방정식으로.

## 2. Power Laws: 핵심 발견

이 논문의 핵심 발견은 언어 모델의 성능이 **power law(멱법칙)**를 따른다는 것이다. 논문이 측정한 범위 내에서, 성능이 주로 하나의 요인에 의해 제한되고 나머지 두 요인에 의해 병목되지 않을 때, 테스트 손실(모델이 다음 단어를 얼마나 잘 예측하는지를 나타내는 지표 — 낮을수록 좋다)을 파라미터 수, 데이터셋 크기, 컴퓨팅에 대해 log-log 그래프에 그리면 대략 직선 관계를 보인다.

세 개의 방정식이 논문 전체를 요약한다:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$

$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$

$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

표기법에 겁먹을 필요 없다. 하나씩 풀어보자:

- **L**은 테스트 손실이다 — 모델 성능을 하나의 숫자로 요약한 것이다. 낮을수록 좋다
- **N**은 파라미터 수(모델 크기)이다. 파라미터가 많을수록 모델이 더 많은 패턴을 저장할 수 있다
- **D**는 모델이 학습한 데이터 토큰 수이다. 데이터가 많을수록 학습할 패턴이 많다
- **C**는 학습에 사용된 총 컴퓨팅이다. PetaFLOP-day 단위로 측정된다(1 PetaFLOP-day = 10^15 부동소수점 연산을 하루 종일 수행한 것)
- **N_c, D_c, C_c**는 상수(곡선 위의 기준점)이다
- **α**(알파)는 지수이다 — log-log 그래프에서 직선의 기울기를 알려준다. 지수가 클수록 규모를 키울 때 성능이 더 빠르게 향상된다

핵심 통찰: 이것들은 멱법칙이지, 대수(logarithmic) 곡선이 아니다. 대수 곡선은 빠르게 평탄해진다 — 입력을 두 배로 늘려도 출력은 거의 움직이지 않는다. 멱법칙은 훨씬 관대하다: 적어도 논문이 측정한 범위 내에서, 성능은 뚜렷한 벽에 부딪히는 징후 없이 멱법칙 추세를 따라 꾸준히 개선되었다. 논문도 이 추세가 영원히 0까지 내려갈 수는 없으며 결국은 평탄해질 것이라고 명시했지만, 관측된 범위 내에서는 추세가 깔끔하게 유지되었다.

```python showLanguage
def power_law_loss(x: float, x_c: float, alpha: float) -> float:
    return (x_c / x) ** alpha


def scaling_law_examples() -> dict[str, float]:
    alpha_n = 0.076
    alpha_d = 0.095
    alpha_c = 0.050

    return {
        "10x_params": 10.0 ** alpha_n,
        "10x_data": 10.0 ** alpha_d,
        "10x_compute": 10.0 ** alpha_c,
    }
```

지수들이 이야기를 들려준다. 데이터셋 크기(α = 0.095)가 스케일링 배수당 가장 큰 향상을 가져온다. 모델 크기(α = 0.076)가 그 다음이다. 컴퓨팅(α = 0.050)은 가장 적은 향상을 보인다 — 모델 크기와 학습 시간 사이에 적절히 배분하지 않고 컴퓨팅만 늘리는 것은 낭비이기 때문이다. 진짜 레버리지는 올바른 것을 스케일링하는 데서 나온다.

## 3. 논문이 테스트한 범위 내에서, 아키텍처 형태보다 전체 규모가 더 중요하다

여기서 논문은 핵심 지점을 분명히 한다.

팀은 서로 다른 깊이(레이어 수), 너비(히든 차원), 어텐션 헤드 수, 피드포워드 차원을 가진 Transformer들을 테스트했다. 논문이 테스트한 Transformer 형태 범위 내에서, 비임베딩 파라미터 수가 비슷한 한, 깊이와 너비의 구체적인 배분은 손실에 미치는 영향이 작았다.

레이어 2개에 거대한 히든 차원을 가진 Transformer? 비슷한 비임베딩 파라미터 예산 하에서 레이어 40개에 작은 히든 차원을 가진 것과 대략 같은 손실을 보인다.

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureExperiment:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int


def non_embedding_params(config: ArchitectureExperiment) -> int:
    n = config.n_layers
    d = config.d_model
    d_ff = config.d_ff
    return n * (4 * d * d + 2 * d * d_ff + 4 * d)
```

이것은 심오한 함의를 갖는다: "최적의" 아키텍처를 찾는 데 몇 주를 소비할 필요가 없다. 합리적인 Transformer 구조를 하나 골라서, 그것을 키우는 데 에너지를 집중하면 된다. 논문은 임베딩 파라미터가 비임베딩 파라미터보다 성능에 기여하는 바가 훨씬 적다는 것을 발견했기 때문에 N에서 임베딩 파라미터를 명시적으로 제외했다 — 모델의 "사고" 능력은 어휘 테이블이 아니라 Transformer 레이어에 있다.

## 4. 모델이 과적합할 때: 데이터 병목

더 크다고 항상 더 나은 건 아니다 — 데이터셋이 너무 작으면 그렇다. 논문에서 진짜 아름다운 부분은, 모델 규모와 데이터 규모가 어떻게 함께 성능을 결정하는지를 하나의 통합된 이변수 공식으로 포착한 것이다:

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
$$

이 공식이 말하는 것은: 손실은 모델 크기나 데이터 크기 하나만의 함수가 아니라, 둘 다의 함수라는 것이다. N이 충분히 커서 첫 번째 항이 사라지면, 남은 항은 손실이 데이터에 의해 병목되었음을 보여준다. D가 충분히 커서 두 번째 항이 사라지면, 남은 것은 모델 규모 병목이다. 공식은 두 체제 사이를 부드럽게 보간하며, 과적합은 두 항이 경쟁하는 자연스러운 결과로 포착된다.

이 관계에서 논문은 과적합이 본격적으로 영향을 미치기 시작하는 대략적인 경험적 임계점을 도출했다:

$$
D \gtrsim 5 \times 10^3 \times N^{0.74}
$$

쉽게 말하면: 모델을 크게 만들수록 필요한 데이터의 양이 늘어난다 — 하지만 선형 이하로. 10배 큰 모델은 10^0.74 ≈ 5.5배의 데이터만 더 필요하다. 더 큰 모델은 더 샘플 효율적이다: 학습 데이터의 각 토큰에서 더 많은 정보를 추출한다.

```python showLanguage
def loss_nd(n_params: float, n_tokens: float) -> float:
    n_c = 8.8e13
    d_c = 5.4e13
    alpha_n = 0.076
    alpha_d = 0.095
    ratio = alpha_n / alpha_d
    return ((n_c / n_params) ** ratio + d_c / n_tokens) ** alpha_d


def min_dataset_tokens(n_params: float) -> float:
    return 5_000.0 * n_params ** 0.74
```

이 관계를 대략적으로 추산하면, 175B 규모 모델이 과적합을 논문이 논의한 임계값 근처로 억제하려면 거의 1조 토큰에 가까운 데이터가 필요하다. 반대로 보면, GPT-3의 3,000억 토큰은 사실 넉넉하지 않았다. 이것은 「모델을 얼마나 크게 만들고, 데이터를 얼마나 먹일 것인가」가 감이 아니라 분석 가능한 트레이드오프라는 것을 보여준다 — 이후 업계가 이 비율을 재검토한 것(가장 대표적으로 Chinchilla 논문, Hoffmann et al., 2022)도 많은 대형 모델의 데이터가 실제로 부족했다는 인식 때문이었다.

## 5. 컴퓨팅 효율적 학습: 진짜 핵심

고정된 컴퓨팅 예산이 있다면, 어떻게 써야 할까? 이것이 논문에서 가장 실용적으로 중요한 질문이고, 답은 직관에 반한다.

논문은 최적 배분이 다음을 따른다는 것을 발견했다:

$$
N_{\mathrm{opt}} \propto C^{0.73}
$$

$$
B_{\mathrm{opt}} \propto C^{0.24}
$$

$$
S_{\mathrm{opt}} \propto C^{0.03}
$$

해석: 컴퓨팅 예산이 10배 늘면, 모델을 ~5.4배 키우고, 배치 크기를 ~1.7배 늘리고, 학습 시간은 거의 늘리지 않아야 한다(~1.07배 더 많은 스텝).

직관에 반하는 부분: **아주 큰 모델을 학습시키고 수렴 훨씬 전에 멈춰야 한다.** 대부분의 사람들의 직감은 작은 모델을 완전히 학습시키는 것이다. 스케일링 법칙은 그 반대를 말한다 — 같은 컴퓨팅 예산에서, 부분적으로 학습된 큰 모델이 완전히 학습된 작은 모델을 이긴다.

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeAllocation:
    n_params: float
    batch_size: float
    training_steps: float


def optimal_allocation(compute: float) -> ComputeAllocation:
    return ComputeAllocation(
        n_params=compute ** 0.73,
        batch_size=compute ** 0.24,
        training_steps=compute ** 0.03,
    )


def is_compute_efficient(n_params: float, compute: float) -> bool:
    optimal_n = compute ** 0.73
    return abs(n_params / optimal_n - 1.0) < 0.5
```

이 결과는 업계 전체를 형성했다. 이 논문 5개월 후에 나온 GPT-3는 이 논리를 직접적으로 따랐다: 작은 모델을 완전히 학습시키는 대신, 당시 기준으로 거대한 1,750억 파라미터 모델을 학습시켰다. 이후의 "Chinchilla" 논문(Hoffmann et al., 2022)은 이 지수들을 업데이트하고 대부분의 대형 모델이 최적 데이터 배분 대비 실제로 과소 학습되었다고 주장했지만 — 계산 가능한 최적 트레이드오프가 존재한다는 핵심 통찰은 여기서 시작되었다.

## 6. 임계 배치 크기: 병렬화 시점 알기

논문은 배치 크기에 "최적 지점"이 있다는 것도 발견했으며, 그것은 현재 손실에 의존한다:

$$
B_{\mathrm{crit}} \propto L^{-4.8}
$$

학습이 진행되고 손실이 줄어들수록, 임계 배치 크기는 커진다. 학습 초기에는 손실이 높아서 작은 배치도 충분하다 — 각 배치가 충분히 강한 그래디언트 신호를 제공한다. 나중에는 모델이 이미 쉬운 패턴을 학습했기 때문에, 노이즈를 평균화하고 진전을 이루려면 더 큰 배치가 필요하다.

임계 배치 크기 이하에서는, 배치를 두 배로 늘리면 학습 시간이 대략 절반으로 줄어든다(완벽한 병렬화). 그 이상에서는, 배치를 두 배로 늘려도 거의 도움이 되지 않는다 — 컴퓨팅을 낭비할 뿐이다.

```python showLanguage
def critical_batch_size(loss: float, b_star: float, l_star: float) -> float:
    return b_star * (l_star / loss) ** 4.8
```

이것은 실용적인 엔지니어링 지혜이다. 많은 팀이 학습 내내 고정된 배치 크기를 사용한다. 스케일링 법칙은 학습이 진행됨에 따라 배치 크기를 늘려야 한다고 말한다 — 작게 시작해서, 모델이 나아짐에 따라 키워라.

## 7. 이 논문이 바꾼 질문

Scaling Laws의 핵심 문장은 이것입니다. **대형 모델 학습은 감각적 시행착오가 아니라 예산 함수다.**

이 논문의 가장 중요한 기여는 특정 지수가 아닙니다. "클수록 좋다"를 경험적 감각에서 추정 가능한 곡선으로 바꾼 것입니다. 파라미터, 데이터, compute는 더 이상 흩어진 엔지니어링 변수가 아니라 같은 장부에서 marginal return을 비교할 수 있는 투자 항목이 됐습니다.

이 논문은 업계에 영원히 더 큰 모델만 만들라고 말하지 않았습니다. 주어진 가정과 관측 범위 안에서 돈을 쓰기 전에 계산하라고 말했습니다. 파라미터를 늘릴지, 데이터를 늘릴지, 학습을 더 할지, 무엇이 loss를 더 낮출 가능성이 큰가? GPT-3의 규모 결정이 덜 무모해 보였던 배경에는 이런 예산 신뢰가 있었습니다.

다음에 모델 scale을 볼 때는 이것이 군비 경쟁인지부터 묻지 않는 편이 좋습니다. 예산 함수가 무엇인지, 병목 변수가 무엇인지, marginal return이 얼마나 남았는지를 물어야 합니다. 성숙한 scaling은 돈을 더 태우는 것이 아니라 돈이 어디에서 능력으로 바뀌는지 아는 것입니다.

---

**논문 읽기 시리즈**

- [《Sequence to Sequence Learning with Neural Networks》](/ko-KR/posts/sequence-to-sequence-learning-with-neural-networks/) (신경망을 이용한 시퀀스-투-시퀀스 학습) — 인코더-디코더 패러다임의 확립
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/ko-KR/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) (정렬과 번역을 공동으로 학습하는 신경 기계 번역) — 어텐션의 기원
- [《Attention Is All You Need》](/ko-KR/posts/attention-is-all-you-need/) (어텐션만 있으면 충분하다) — 어텐션이 주역이 되다: Transformer의 탄생
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전 학습 패러다임의 확립
- [《Language Models are Few-Shot Learners》](/ko-KR/posts/language-models-are-few-shot-learners/) (언어 모델은 퓨샷 학습자다) — 더 큰 모델, 컨텍스트에서 더 잘 능력을 이끌어내다
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — 컴퓨팅 예산을 현명하게 쓰는 법

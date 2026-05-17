---
title: "《Language Models are Few-Shot Learners》: GPT-3와 인컨텍스트 학습"
date: "2026-02-11T16:22:54+08:00"
category: "Paper Reading"
description: 더 큰 모델, 컨텍스트에서 더 잘 능력을 이끌어내다, 실제 Python 코드 예시 포함
tags: [paper-reading, gpt-3, AI, LLM, python]
pinned: false
---

GPT-3의 질문은 단순히 모델을 더 키울 수 있는가가 아니었습니다. 과제 적응이 반드시 파라미터 업데이트를 필요로 하는가였습니다. BERT 패러다임에서는 새 과제마다 fine-tuning이 필요했고, 데이터, 학습 파이프라인, 배포 버전이 함께 늘어났습니다.

[《Language Models are Few-Shot Learners》](/papers/2005.14165v4.pdf)의 핵심은 task adaptation을 파라미터 업데이트에서 context로 옮긴 것입니다. 가중치는 고정됩니다. 지시문과 몇 개의 예제가 prompt에 들어갑니다. 과제는 추론 시점에 임시로 조직됩니다.

## 0. 먼저 몇 가지 용어부터

GPT-3 같은 모델이 어떻게 작동하는지 아직 감이 없다면, 아래 용어만 먼저 잡아두면 충분하다:

- `언어 모델`: 앞에 주어진 문맥을 보고 다음 단어를 예측하는 모델이다.
- `파라미터 수`: 모델 안에서 학습되는 숫자의 총량이다. 대략 모델의 "두뇌 용량"이라고 생각해도 된다.
- `prompt / 프롬프트`: 모델에게 보여 주는 작업 설명, 예시, 입력 전체를 말한다.
- `context window / 컨텍스트 윈도우`: 모델이 한 번에 읽을 수 있는 텍스트의 최대 길이다.
- `few-shot / one-shot / zero-shot`: 예시를 여러 개 주는 경우, 하나만 주는 경우, 아예 주지 않는 경우를 각각 뜻한다.
- `in-context learning`: 파라미터를 바꾸지 않고, 프롬프트 안의 설명과 예시만으로 모델이 임시로 작업 방식을 익히는 현상이다.

## 1. 문제 정의

[BERT](/ko-KR/posts/bert/)가 확립한 "사전 학습 + fine-tuning" 패러다임은 2020년에 이미 주류였다. 잘 작동했지만, 논문은 세 가지 근본적인 문제를 지적했다.

첫째, 모든 새로운 태스크에는 여전히 레이블이 붙은 데이터셋이 필요하다. 레이블 데이터는 수집 비용이 비싸고, 많은 실제 태스크에는 대응하는 레이블 데이터셋이 아예 존재하지 않는다.

둘째, fine-tuning된 모델이 테스트 벤치마크에서 보이는 성능이 반드시 진정한 일반화를 반영하는 것은 아니다. 모델이 단순히 학습 데이터의 허위 상관관계(spurious correlation)를 학습했을 수 있다 — 벤치마크에서는 높은 점수를 얻지만 분포 변화(distribution shift) 앞에서는 무너지는 것이다.

셋째, 인간은 이런 식으로 학습하지 않는다. 인간은 한두 가지 예시를 보고, 자연어 지시를 듣고, 새로운 태스크를 처리할 수 있다. 그 시절의 NLP 시스템은 새로운 태스크마다 수천 개의 레이블 샘플로 fine-tuning해야 했다.

논문의 출발점: 모델이 충분히 크다면, 사전 학습 동안 축적한 지식으로 태스크 설명과 몇 가지 예시를 직접 "읽고" 답을 생성할 수 있을까?

## 2. 핵심 아이디어: 파라미터 업데이트 없이, 프롬프트만으로

GPT-3의 평가 방법론은 이전의 모든 대형 모델과 달랐다. 그래디언트 업데이트가 전혀 없는 세 가지 설정을 정의했다:

**Few-Shot**: 모델에 태스크 설명과 10~100개의 예시(정확한 수는 컨텍스트 윈도우에 맞는 양에 따라 다름)를 제공한 뒤, 새로운 입력을 완성하게 한다. 가중치 업데이트도 없고, 역전파도 없다.

**One-Shot**: 예시를 딱 하나만 제공한다. 인간이 새로운 태스크를 학습하는 방식과 가장 비슷하다 — 누군가 한 번 시범을 보이면, 그다음부터는 스스로 해내는 것이다.

**Zero-Shot**: 예시가 전혀 없고, 자연어 지시만 있다. 가장 어려운 설정이지만, 가장 실용적이기도 하다 — 모델이 태스크 자체를 진정으로 "이해"한다면 예시가 필요 없어야 한다.

```python showLanguage
from dataclasses import dataclass
from typing import Union


@dataclass
class ZeroShot:
    instruction: str
    prompt: str


@dataclass
class OneShot:
    instruction: str
    example: tuple[str, str]
    prompt: str


@dataclass
class FewShot:
    instruction: str
    examples: list[tuple[str, str]]
    prompt: str


EvalSetting = Union[ZeroShot, OneShot, FewShot]


def build_prompt(setting: EvalSetting) -> str:
    if isinstance(setting, ZeroShot):
        return f"{setting.instruction}\n{setting.prompt}"

    if isinstance(setting, OneShot):
        example_input, example_output = setting.example
        return f"{setting.instruction}\n{example_input} {example_output}\n{setting.prompt}"

    lines = [setting.instruction]
    lines.extend(f"{example_input} {example_output}" for example_input, example_output in setting.examples)
    lines.append(setting.prompt)
    return "\n".join(lines)
```

논문은 이 능력을 **in-context learning**이라고 부른다: 사전 학습 과정에서 모델이 방대한 텍스트로부터 다양한 태스크 패턴을 암묵적으로 학습하고, 추론 시점에 예시들이 컨텍스트에 이어 붙여지면 모델이 순방향 패스 중에 현재 태스크를 "인식"하고 수행한다. 논문은 이 과정을 "메타 학습"의 언어로 설명한다 — 사전 학습이 외부 루프이고, in-context learning이 내부 루프이다.

Fine-tuning과의 차이는 근본적이다. Fine-tuning은 태스크에 맞추기 위해 모델 파라미터를 수정한다. In-context learning은 아무것도 수정하지 않는다 — 같은 모델, 같은 가중치, 순전히 입력 텍스트만 바꿔서 태스크를 전환한다.

## 3. 모델 아키텍처와 규모

GPT-3의 아키텍처는 새로운 발명이 아니다. GPT-2와 마찬가지로, [Transformer](/ko-KR/posts/attention-is-all-you-need/)의 디코더 부분만을 층층이 쌓은 것이다. 유일한 변경점은 Transformer 레이어 내에서 밀집 어텐션과 로컬 밴드 희소 어텐션(Sparse Transformer에서 차용)을 번갈아 사용한 것이다.

진정한 차이는 규모에 있다. 논문은 파라미터 수가 세 자릿수에 걸쳐 분포된 8개 모델을 학습시켰다:

| 모델 | 파라미터 | 레이어 | 히든 크기 | 어텐션 헤드 |
|-------|-----------|--------|-------------|-----------------|
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 |
| GPT-3 Large | 760M | 24 | 1536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 |
| GPT-3 13B | 13B | 40 | 5140 | 40 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** |

1,750억 개의 파라미터, 96개의 레이어, 96개의 어텐션 헤드, 히든 차원 12288. 컨텍스트 윈도우 2048 토큰. 이 규모는 당시 전례가 없었다 — GPT-2의 15억 파라미터보다 100배 이상 크다.

```python showLanguage
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT3Config:
    n_params: int
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_ctx: int


def gpt3_175b() -> GPT3Config:
    return GPT3Config(
        n_params=175_000_000_000,
        n_layers=96,
        d_model=12_288,
        n_heads=96,
        d_head=128,
        d_ff=49_152,
        n_ctx=2_048,
    )
```

이 모델들을 학습시킨 목적은 명확했다: scaling laws를 검증하는 것이다. Kaplan 등(이 논문의 공저자 중 한 명)의 이전 연구에서 언어 모델의 손실과 파라미터 수 사이에 매끄러운 멱법칙(power-law) 관계가 있음을 이미 보여줬다. GPT-3는 그 가설을 1,750억 파라미터까지 밀어붙여 in-context learning 능력도 같은 패턴을 따르는지 확인했다.

답은 "그렇다"였다: 모델이 클수록 few-shot 학습의 개선 폭이 더 가팔랐다. Zero-shot 성능은 규모에 따라 꾸준히 상승하고, few-shot 성능은 더 빠르게 상승한다. 이는 더 큰 모델이 단순히 "더 정확한" 것이 아니라, 컨텍스트 정보를 활용하는 데도 더 효율적이라는 뜻이다.

## 4. 학습 데이터

GPT-3는 5개 출처에서 수집한 약 3,000억 개의 토큰으로 학습되었다:

| 데이터셋 | 토큰 수 | 학습 비중 |
|---------|--------|--------------|
| Common Crawl (필터링 후) | 410B | ~60% |
| WebText2 | 19B | ~22% |
| Books1 | 12B | ~8% |
| Books2 | 55B | ~8% |
| English Wikipedia | 3B | ~3% |

주목할 점이 있다: 샘플링 비율은 데이터셋 크기에 비례하지 않는다. 고품질 데이터셋(WebText2, Books, Wikipedia)이 오버샘플링되었다 — WebText2는 학습 중 2.9회, Wikipedia는 3.4회 반복 사용된 반면, Common Crawl은 전체를 한 번도 다 보지 못했다(0.44 에폭). 논문은 의도적으로 약간의 오버피팅을 감수하고 더 높은 품질의 학습 신호를 택했다.

원시 Common Crawl 데이터는 45TB였다. 세 단계의 처리를 거쳤다: (1) 고품질 참조 말뭉치와의 유사도를 기반으로 필터링; (2) 문서 수준의 퍼지 중복 제거; (3) 다양성을 위해 알려진 고품질 데이터셋을 혼합. 필터링 후 570GB가 남았다 — 대략 4,100억 토큰이다.

모든 모델은 Microsoft가 제공한 고대역폭 클러스터의 V100 GPU에서 학습되었다.

## 5. 실험 결과

논문은 20개 이상의 데이터셋에 걸쳐 9가지 주요 태스크 범주를 평가했다. 몇 가지 핵심 결과를 정리하면 다음과 같다.

**언어 모델링**: Penn Tree Bank에서 GPT-3 few-shot 퍼플렉서티(모델이 텍스트에 얼마나 "놀라는지"를 측정한 것 — 낮을수록 좋다)는 20.50으로 새로운 기록을 세웠다. LAMBADA(장거리 컨텍스트를 기반으로 마지막 단어를 예측하는 태스크)에서 zero-shot 정확도는 76.2%, few-shot은 86.4%로 이전 최고 기록을 크게 앞섰다.

**번역**: GPT-3는 번역을 위해 특별히 학습된 적이 없지만, 프랑스어-영어 few-shot BLEU 점수가 32.6으로 최고의 비지도 신경 기계 번역 결과를 넘어섰다. 반면 영어-프랑스어(25.2 BLEU)는 fine-tuning된 모델에 비해 여전히 상당히 뒤처졌다. 비대칭이 드러난다: GPT-3는 영어로 번역하는 것이 영어에서 다른 언어로 번역하는 것보다 눈에 띄게 잘했는데, 이는 학습 데이터의 영어 편중을 직접적으로 반영한다.

**Closed-Book QA**: TriviaQA에서 few-shot 정확도(exact match)는 71.2%로, 같은 closed-book 설정 하에서 fine-tuning된 모델을 넘어섰다. 모델은 어떤 문서도 참조하지 않고 — 파라미터에 저장된 지식만으로 답한다.

**SuperGLUE**: 이 종합 벤치마크에서 GPT-3의 few-shot 성능은 일부 강력한 fine-tuning 베이스라인에 근접했지만, 당시 가장 강력한 전용 fine-tuning 시스템에는 여전히 미치지 못했다.

**합성 태스크**: 논문은 in-context learning을 테스트하기 위해 새로운 태스크도 설계했다. 예를 들어, "만들어낸 단어"의 예시 몇 개를 제공하면(존재하지 않는 단어를 정의하고 문장에서 사용하는 것), GPT-3는 새 단어를 올바르게 학습하고 사용할 수 있었다. 세 자릿수 덧셈은 few-shot에서 거의 100% 정확했고(두 자릿수도 거의 완벽했다), 네 자릿수와 다섯 자릿수에서는 정확도가 급격히 떨어졌다.

```python showLanguage
from typing import Callable, Protocol


class AutoregressiveModel(Protocol):
    def forward(self, tokens: list[int]) -> list[list[float]]:
        ...


def in_context_learning(
    model: AutoregressiveModel,
    examples: list[tuple[str, str]],
    query: str,
    tokenize: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    sample_from: Callable[[list[float]], int],
    eos_token: int,
) -> str:
    prompt_lines = [f"{example_input} {example_output}" for example_input, example_output in examples]
    prompt_lines.append(query)
    prompt = "\n".join(prompt_lines)

    context = tokenize(prompt)
    output_tokens: list[int] = []

    while True:
        logits = model.forward(context)
        next_token = sample_from(logits[-1])
        if next_token == eos_token:
            break
        output_tokens.append(next_token)
        context.append(next_token)

    return decode(output_tokens)
```

## 6. 데이터 오염

논문은 섹션 4에서 까다로운 문제에 상당한 지면을 할애한다: 학습 데이터와 테스트 데이터의 중복.

GPT-3의 학습 데이터에는 방대한 양의 인터넷 텍스트가 포함되어 있고, 많은 테스트 벤치마크가 인터넷에 공개되어 있다. 이는 모델이 학습 중에 테스트 문제를 "본" 적이 있을 수 있다는 뜻이다. 팀은 학습 전에 이러한 중복을 제거하려 했지만, 처리 파이프라인의 버그로 인해 일부 중복이 완전히 정리되지 않았다. 처음부터 다시 학습시키는 것은 비용상 현실적이지 않았다.

그들의 접근 방식: 각 벤치마크에 대해 "클린 서브셋"(학습 데이터와의 13-gram 중복이 있는 모든 샘플을 제거)을 구성한 뒤, 전체 세트와 클린 서브셋에서의 모델 성능을 비교했다. 결론: 대부분의 벤치마크에서 오염이 결과에 미친 영향은 미미했다. 다만 PIQA와 Winograd는 의심스러운 성능 하락을 보였고, 논문은 해당 결과에 별표를 표시했다.

이 정도의 정직함은 당시로서는 상당히 드물었다. 대부분의 논문은 데이터 오염에 대한 논의를 아예 회피한다. GPT-3는 이 문제를 능동적으로 조사했을 뿐만 아니라 체계적인 탐지 도구까지 개발했다. 그 자체가 후속 연구에 대한 기여이다.

## 7. 한계

논문 섹션 5의 한계에 대한 논의는 상당히 솔직하다.

**텍스트 일관성**: GPT-3는 문서 수준에서 여전히 의미 반복, 자기 모순, 심지어 말이 안 되는 문장을 보인다. 생성 품질은 GPT-2보다 훨씬 나아졌지만, 긴 글에서의 일관성은 여전히 부족하다.

**상식 물리**: GPT-3는 "치즈를 냉장고에 넣으면 녹을까?"와 같은 상식 물리 질문에서 성능이 떨어진다. 언어적 추론은 가능하지만, 물리 세계에 대한 이해는 여전히 피상적이다.

**단방향성의 비용**: 자기회귀 모델인 GPT-3는 왼쪽에서 오른쪽으로만 볼 수 있다. 논문은 양방향 컨텍스트가 필요한 태스크(예: 두 문장에서 같은 단어가 같은 의미를 갖는지 판별하는 것)에서 GPT-3의 few-shot 성능이 fine-tuning된 양방향 모델에 미치지 못한다고 인정한다. 이는 이러한 태스크가 자기회귀 설정에서 GPT-3의 강점이 아님을 보여준다; 단방향 모델링 목표가 구조적 편향을 도입한다.

**샘플 효율성**: GPT-3는 사전 학습 중 약 3,000억 개의 토큰을 봤는데, 이는 인간이 평생 접하는 텍스트 양을 훨씬 넘는다. 논문은 few-shot 학습이 추론 시점에서는 효율적이지만, 사전 학습을 위한 데이터 요구량은 여전히 막대하다고 명시적으로 언급한다.

**추론 비용**: 1,750억 파라미터 모델은 실행 비용이 높고 배포가 어렵다. 논문은 디스틸레이션(대형 모델의 출력을 사용해 소형 모델을 학습시키는 것)을 가능한 방향으로 언급하지만, 천억 파라미터 규모에서는 아직 시도되지 않았다고 밝힌다.

## 8. 사회적 영향

논문은 섹션 6 전체를 사회적 영향에 할애하며, 세 가지 영역을 다룬다.

**오용 위험**: 인간 평가자는 GPT-3가 생성한 뉴스 기사를 우연 수준(~52% 정확도)으로밖에 식별하지 못했다. 모델이 강력할수록 생성된 텍스트를 탐지하기가 더 어렵다. 팀은 악의적 사용의 동향을 추적하기 위해 포럼과 채팅 그룹을 모니터링하고 있다고 보고했다.

**편향**: 논문은 GPT-3의 성별, 인종, 종교에 걸친 편향을 테스트하는 광범위한 실험을 수행했다. 예를 들어, 직업-성별 연관 테스트에서 GPT-3는 "간호사"를 여성과, "은행가"를 남성과 연관시키는 경향이 더 강했다. 종교-감정 연관에서는 "이슬람"이 폭력 관련 단어와 더 자주 동시 출현했다. 논문은 이러한 편향이 학습 데이터에서 비롯됨을 인정하지만 해결책은 제시하지 않는다.

**에너지 소비**: GPT-3 학습에는 막대한 컴퓨팅이 필요하며, 논문은 추정치를 인용하되 구체적인 에너지 수치는 공개하지 않는다. 다만 한 번 학습된 모델은 여러 태스크에 적용할 수 있어, 각 태스크마다 별도의 모델을 학습시키는 것보다 에너지 효율이 높다고 지적한다.

## 9. 이 논문이 바꾼 질문

GPT-3의 핵심 문장은 이것입니다. **task adaptation은 파라미터 업데이트에서 context로 이동할 수 있다.**

이 논문은 단순히 큰 모델이 더 강하다고 말한 것이 아닙니다. 사람과 모델 사이의 인터페이스를 바꿨습니다. 이전에는 모델이 새 과제를 하게 하려면 데이터를 모으고, 파라미터를 fine-tune하고, 새 버전을 배포해야 했습니다. GPT-3는 다른 경로를 보였습니다. 가중치는 고정하고, 지시문, 예제, 입력을 context에 넣어 추론 시점에 적응하게 합니다.

이것이 fine-tuning의 끝은 아닙니다. 적응 비용의 일부를 학습 단계에서 context 설계 단계로 옮긴 것입니다. Prompt는 더 이상 단순한 입력 텍스트가 아닙니다. 가벼운 task program이 됩니다. Context window도 단순한 용량 숫자가 아니라 임시 작업 공간이 됩니다.

다음에 대형 모델을 볼 때는 파라미터 수만 묻지 않는 편이 좋습니다. task adaptation이 어디에 있는지 물어야 합니다. 가중치 안인가, context 안인가, 외부 도구와 workflow 안인가? 이 질문이 모델 크기보다 제품 형태에 더 가깝습니다.

---

**논문 읽기 시리즈**

- [《Sequence to Sequence Learning with Neural Networks》](/ko-KR/posts/sequence-to-sequence-learning-with-neural-networks/) (신경망을 이용한 시퀀스-투-시퀀스 학습) — 인코더-디코더 패러다임의 확립
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/ko-KR/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/) (정렬과 번역을 공동으로 학습하는 신경 기계 번역) — 어텐션의 기원
- [《Attention Is All You Need》](/ko-KR/posts/attention-is-all-you-need/) (어텐션만 있으면 충분하다) — 어텐션이 주역이 되다: Transformer의 탄생
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/ko-KR/posts/bert/) (BERT: 언어 이해를 위한 깊은 양방향 트랜스포머 사전학습) — 사전 학습 패러다임의 확립
- [《Scaling Laws for Neural Language Models》](/ko-KR/posts/scaling-laws-for-neural-language-models/) (신경 언어 모델을 위한 스케일링 법칙) — 스케일의 수학: 왜 더 큰 모델이 예측 가능하게 더 좋은가
- [《Training Compute-Optimal Large Language Models》](/ko-KR/posts/training-compute-optimal-large-language-models/) (연산량 최적의 대규모 언어 모델 학습) — 컴퓨팅 예산을 현명하게 쓰는 법

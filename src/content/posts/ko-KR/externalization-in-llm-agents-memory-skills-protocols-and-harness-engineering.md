---
title: "《Externalization in LLM Agents》: LLM Agent의 인지적 외부화"
description: "Externalization in LLM Agents를 cognitive artifacts 관점에서 읽는다. Agent의 진전은 점점 더 memory, skills, protocols, harness를 모델 밖의 인프라로 옮기는 일에 가까워지고 있다."
date: "2026-05-13T16:00:00+08:00"
category: "Paper Reading"
tags: [paper-reading, agents, externalization, harness-engineering, memory, skills, protocols, LLM]
pinned: false
---

Agent 이야기는 자주 모델 순위로 돌아간다.

어떤 모델이 더 잘 추론하는지, 어떤 모델의 컨텍스트 창이 더 긴지, 어떤 모델이 코딩 benchmark에서 이겼는지. 모두 중요한 질문이다. 하지만 [“Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering”](/papers/2604.08224v1.pdf)은 조금 다른 중심을 가리킨다.

이 논문이 묻는 질문은 “모델이 무엇을 더 배울 수 있는가”가 아니다.

**모델이 더 이상 혼자 짊어지지 않아도 되는 인지적 부담은 무엇인가?**

그래서 핵심 단어가 `externalization`이다. 단순히 LLM 주변에 컴포넌트를 몇 개 붙인다는 뜻이 아니다. 작업의 표현 방식을 바꾼다는 뜻에 가깝다. 인간이 종이, 지도, 달력, 컴퓨터를 쓰는 방식도 그렇다. 외부 도구는 뇌 자체를 키우지 않는다. 대신 문제의 모양을 바꾼다. 회상은 인식으로 바뀌고, 암산은 기호 조작으로 바뀌며, 즉흥적 판단은 절차 수행으로 바뀐다.

LLM Agent도 비슷한 이동을 겪고 있다.

![논문 Figure 1: LLM Agent 설계 원리로서의 외부화](/images/posts/externalization-in-llm-agents/figure-1-externalization-overview.png)

*Figure 1, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), format-converted for web display, content unchanged.*

## 이 survey가 정리해 주는 것

이 논문은 새 모델 논문도 아니고 benchmark 논문도 아니다. 가치는 개념 정리에 있다. 이미 벌어지고 있는 engineering migration에 이름을 붙인다.

초기에는 능력을 주로 weights 안에 있는 것으로 봤다. 더 큰 모델, 더 좋은 pretraining, 더 강한 alignment가 중심이었다. 이후 능력은 점점 context 안에서 조직되기 시작했다. prompt, few-shot 예시, RAG, reasoning trace, tool description이 중요해졌다. 지금의 실전 Agent 시스템은 더 두꺼운 runtime에 의존한다. memory store, file system, skill library, tool protocol, sandbox, approval gate, log, evaluator, sub-agent orchestration이 모두 포함된다.

논문은 이 흐름을 이렇게 압축한다.

**weights → context → harness**

![LLM Agent의 인지적 외부화에 대한 저자의 개념 지도](/images/posts/externalization-in-llm-agents/cognitive-externalization-map.webp)

*저자가 AI로 생성한 개념도이며 논문 원본 그림이 아니다. 이 글의 해석을 요약하면 모델 능력은 weights에서 context를 거쳐 memory, skills, protocols, governance 같은 harness-level infrastructure로 이동한다.*

이 프레이밍이 좋은 이유는 실제 시스템의 감각과 맞기 때문이다. 모델은 여전히 중요하다. 하지만 모델이 더 이상 전체 작업을 혼자 들고 있지는 않다. 모델은 기억하고, 검색하고, 제한하고, 감사하고, 실패에서 복구하는 환경 안에서 동작한다.

즉 신뢰성은 모델의 속성일 뿐 아니라, 작업이 어떻게 외부화되었는지의 결과이기도 하다.

## Memory는 시간을 외부화한다

가장 이해하기 쉬운 외부화는 memory다.

Agent가 context window에만 의존한다면, 매 실행은 임시 작업 공간에 의존한다. 더 긴 context는 도움이 되지만 선택 문제를 없애지는 못한다. 무엇을 넣을 것인가, 무엇을 잊을 것인가, 무엇이 낡은 정보인가, 무엇이 노이즈인가. 더 중요한 점은 context가 기본적으로 일시적이라는 것이다. 외부 상태가 없으면 작업이 끝난 뒤 많은 경험이 사라진다.

외부 memory는 문제를 “모델이 이것을 기억할 수 있는가”에서 “모델이 검색된 올바른 상태를 인식하고 사용할 수 있는가”로 바꾼다. 이것이 cognitive artifacts의 패턴이다. 쇼핑 목록은 생물학적 기억력을 키우지 않는다. 기억 과제를 바꾼다.

Agent에게 memory는 사용자 선호, 프로젝트 규칙, 과거 결정, 실패 궤적, 도메인 지식, 반복되는 제약을 저장할 수 있다. 어려운 부분은 저장 그 자체가 아니다. 무엇을 쓸지, 언제 검색할지, 얼마나 주입할지, 어떻게 압축할지, 오래된 상태가 새 판단을 오염시키지 않게 할지를 정하는 것이다.

따라서 memory는 단순히 더 긴 context의 대체물이 아니다. 시간적 연속성을 위한 인프라다.

## Skills는 절차를 외부화한다

두 번째 외부화는 skills다.

여기서 skill은 “모델이 어떤 도구를 호출할 수 있다”는 뜻이 아니다. 재사용 가능한 절차적 지식이다. step, heuristic, stopping condition, escalation rule, recovery pattern, safety constraint가 포함될 수 있다. 성숙한 skill은 Agent에게 이 종류의 작업을 보통 어떻게 수행해야 하는지 알려준다.

이는 tool use와 다른 추상화 계층이다. Tool은 action을 제공한다. Protocol은 action이 어떻게 발견되고 호출되는지 정의한다. Skill은 그 action들을 어떻게 조직해 반복 가능한 작업으로 만들지를 담는다.

![논문 Figure 5: 외부화된 절차적 전문성으로서의 skills](/images/posts/externalization-in-llm-agents/figure-5-skills-lifecycle.png)

*Figure 5, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

Software engineering agent를 보면 이 점이 매우 구체적이다.

모델은 코드를 수정하고, 테스트를 실행하고, 로그를 읽고, PR 설명을 쓰는 방법을 알고 있을 수 있다. 하지만 안정적인 실행에는 로컬 절차가 필요하다. 어떤 파일을 먼저 읽을지, 사용자 변경을 어떻게 보존할지, 언제 `rg`를 쓸지, 언제 전체 검증을 돌릴지, 이미 staged된 변경은 어떻게 다룰지, 어떤 명령은 조심해야 하는지가 모두 중요하다.

이 모든 것을 매번 모델이 즉석에서 다시 추론하게 두면 행동은 흔들린다. 이것을 skill로 외부화하면 작업은 “workflow를 새로 발명하기”에서 “검증된 workflow를 선택하고 따르기”로 바뀐다.

그래서 skills는 과소평가되기 쉽다. 화려하지 않지만 variance를 직접 낮춘다.

## Protocols는 상호작용 질서를 외부화한다

세 번째 외부화는 protocols다.

Protocol이 없으면 Agent의 상호작용은 대부분 자유 텍스트 협상이다. 모델이 tool call을 원한다고 말한다. 도구가 텍스트를 반환한다. 다른 Agent가 그 텍스트의 의미를 추론한다. 데모에서는 작동할 수 있지만 production에서는 취약하다.

Protocol은 모호한 상호작용을 machine-readable contract로 바꾼다. Tool discovery, argument schema, permission, error, delegation, lifecycle, user approval이 prompt 관습만으로 유지되어서는 안 된다.

가치는 interoperability에만 있지 않다. Protocol은 governance surface도 만든다. 상호작용이 구조화되면 시스템은 validate, audit, replay, monitor, restrict를 할 수 있다. 자유 텍스트는 유연하지만 다루기 어렵다.

## Harness는 인지 환경이다

논문의 가장 강한 정리는 memory, skills, protocols를 harness engineering 안에 놓는 것이다.

Harness는 모델 주변의 얇은 wrapper가 아니다. Agent가 실제로 동작하는 인지 환경이다. control loop, context budget, permission model, sandbox, human approval, log, evaluation hook, failure recovery, sub-agent orchestration이 여기서 작동한다.

![논문 Figure 3: harnessed LLM agent의 외부화 아키텍처](/images/posts/externalization-in-llm-agents/figure-3-harnessed-agent-architecture.png)

*Figure 3, from Zhou et al., [arXiv:2604.08224](https://arxiv.org/abs/2604.08224), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), reproduced unchanged.*

중요한 점은 harness가 memory, skills, protocols 옆에 놓인 또 하나의 모듈이 아니라는 것이다. Harness는 이 외부화 구조들을 호스팅하고 조정하는 runtime이다. Memory는 상태를 제공한다. Skills는 절차를 제공한다. Protocols는 상호작용 구조를 제공한다. Harness는 언제 무엇을 로드할지, 어떻게 상호작용시킬지, 어떤 제약을 적용할지를 결정한다.

그래서 성숙한 Agent는 점점 작은 operating environment처럼 보인다. 자원, 권한, lifecycle, log, policy, recovery path를 가진다. 단순히 “모델 + 도구 목록”이 아니다.

## 능력의 경계가 이동하고 있다

이 논문에서 가장 유용한 생각은 “능력은 어디에 있는가”라는 질문을 바꾸는 것이다.

능력이 weights에만 있다면 Agent 개선은 더 좋은 모델 사용, fine-tuning, retraining이 된다. 능력이 context에도 있다면 prompt와 retrieval이 중요한 engineering surface가 된다. 능력이 harness에도 있다면 system design 자체가 capability의 일부가 된다.

이것은 모델을 덜 중요하게 보는 관점이 아니다. 오히려 강한 모델일수록 좋은 외부 구조 안에 놓을 가치가 크다. LLM은 주어진 정보를 종합하고 판단하고 일반화하는 데 강하다. 그러나 persistent memory, repeatable procedure, permission management, long-lived state, cross-system coordination을 자연스럽게 안정적으로 수행하는 것은 아니다.

Externalization은 편법이 아니다. 실제 경계를 인정하고 작업을 다시 쓰는 engineering이다.

좋은 Agent 시스템은 모델이 매번 처음부터 방법을 찾게 만들지 않는다. 지속 가능한 상태는 memory로, 재사용 가능한 절차는 skills로, 관리 가능한 교환은 protocols로, runtime reliability는 harness로 옮긴다.

## 외부화의 비용도 있다

이 프레임워크는 설득력 있지만 외부화는 공짜가 아니다.

Memory는 오래된 상태, privacy boundary, retrieval pollution을 만든다. Skills는 낡거나 과적합될 수 있고, 잘못 조합되면 위험해질 수 있다. Protocols는 호환되지 않는 표준으로 조각날 수 있고, 시스템을 경직된 interface에 묶을 수도 있다. Harness는 복잡도를 키운다. approval gate, log, sandbox, policy, subroutine이 많아질수록 engineering discipline이 필요하다.

평가 문제도 있다. Agent capability가 모델과 외부 인프라에 분산되어 있다면 우리는 정확히 무엇을 측정하는가? 같은 모델도 다른 harness 안에서는 전혀 다르게 행동할 수 있다. “model capability”와 “agent capability”는 더 이상 같은 말이 아니다.

그래서 이 논문은 유용하다. 외부화가 모든 문제를 해결한다고 말하지 않는다. 신뢰할 수 있는 agency는 모델과 환경의 공동 설계에서 나온다고 말한다.

## 실용적 판단

이 논문은 실용적으로 한 문장으로 줄일 수 있다.

**Agent engineering is increasingly harness engineering.**

모델이 덜 중요해졌기 때문이 아니다. 모델이 더 큰 인지 시스템 안에서 작동하기 때문이다. Memory는 시간을 확장한다. Skills는 절차를 안정화한다. Protocols는 상호작용 질서를 만든다. Harness는 전체를 관찰 가능하고, 권한화되고, 복구 가능하게 만든다.

그래서 어떤 Agent 시스템을 볼 때는 모델 이름만 묻기보다 이런 질문을 해야 한다.

- 어떤 상태를 외부화했는가?
- 어떤 재사용 가능한 skills를 가지고 있는가?
- 도구와 Agent 상호작용은 protocolized되어 있는가?
- Harness는 권한, 실패, 로그, human approval을 어떻게 처리하는가?

이 질문들이 모델 이름보다 실제 시스템 능력에 더 가까운 경우가 많다.

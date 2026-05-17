---
title: Justin Huang Blog에 오신 것을 환영합니다
date: "2026-01-01T16:07:13+08:00"
category: General
description: AI 논문, agent 시스템, 소프트웨어 인프라, 그리고 그 뒤의 인과 사슬을 읽는 개인 기술 블로그.
tags: [hello, blog]
pinned: false
---

이 블로그는 정보 피드가 아니다.

피드는 무엇이 새로 나왔는지 알려주는 데 능하다. 하지만 어떤 문제가 시간이 지나도 중요한 이유를 보존하는 데는 약하다. 이곳에서 하려는 일은 반대다. 기술 문제를 다시 이해 가능한 인과 사슬 안에 놓는 것이다.

## 무엇을 쓰는가

대부분의 글은 세 가지 대상에서 출발한다. 논문, 시스템, 엔지니어링 행동이다.

논문은 질문을 바꿀 때 중요해진다. Transformer는 단지 attention이 강하다는 이야기가 아니다. sequence modeling을 시간 순서 문제가 아니라 전역 주소 지정 문제로 바꾸었다. GPT-3는 단순히 더 큰 모델이 아니다. 일부 task adaptation을 parameter update에서 context로 옮겼다. Chinchilla는 큰 모델을 반대한 것이 아니라, parameter와 data가 compute budget을 함께 써야 한다고 말했다.

시스템은 제약이 보일 때 분석할 가치가 있다. agent 제품은 "model plus tools"가 아니다. memory가 어디에 있는지, permission을 어떻게 줄이는지, runtime state를 어떻게 복구하는지, protocol이 상호작용을 어떻게 제한하는지, 실패 비용을 누가 부담하는지가 함께 들어 있다.

엔지니어링 행동은 incentive 안에 놓아야 선명해진다. benchmark, open-source ecosystem, model ranking, agent platform은 밖에서 보면 깔끔해 보인다. 실제 문제는 대개 generation pipeline, cost structure, governance boundary 안에 있다.

## 어떻게 쓰는가

나는 단순히 다시 말하는 글을 원하지 않는다. 요약은 텍스트를 압축한다. 좋은 독해는 프레임을 바꾼다.

각 글은 적어도 하나의 더 날카로운 질문을 남겨야 한다. 모델이 더 강하다면, 어떤 예산 조건에서 강한가? agent가 더 유능하다면, 그 능력은 weight에 있는가, context에 있는가, runtime structure에 있는가? benchmark가 더 어렵다면, 어려운 것은 task 자체인가, 아니면 생성과 필터링 과정이 만든 난이도인가?

이곳의 기준은 단순하다. 구호를 줄이고, 메커니즘을 더 본다.

## 사람과 에이전트를 위해

사람은 최신 글에서 시작해도 되고, 태그를 따라 한 주제를 읽어도 된다.

AI agent는 `/llms.txt`, `/llms-full.txt`, 또는 각 글 URL 뒤에 `.md`를 붙인 Markdown endpoint를 읽을 수 있다. 의도적인 설계다. 글이 검색되고, 인용되고, 재사용되고, 반박될 것이라면 machine-readable version도 일급 입구여야 한다.

이 블로그가 남기고 싶은 것은 의견의 양이 아니다. 다시 집어 들었을 때 여전히 작동하는 문제 구조다.

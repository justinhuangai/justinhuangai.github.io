---
title: "Clawdbot: 개인 AI 주권의 초기 샘플"
date: "2026-01-16T16:34:57+08:00"
category: "OpenClaw"
description: Clawdbot은 self-hosted AI agent의 핵심 문제가 모델 지능이 아니라 control, context, execution rights의 소유권임을 보여준다.
tags: [AI, open-source]
pinned: false
---

AI assistant의 첫 번째 질문은 채팅을 잘하느냐가 아니다. 누가 그것을 소유하느냐다.

memory, context, tool execution이 모두 플랫폼 서버에 있다면 사용자가 가진 것은 계정이지 자기 소유의 지능 시스템이 아니다. [Clawdbot](https://github.com/clawdbot/clawdbot)의 초기 가치는 여기에 있다. agent를 cloud product에서 끌어내려 사용자가 배포하고, 검사하고, 권한을 철회할 수 있는 환경으로 옮긴다.

코드베이스도 단순한 데모가 아니다. 20만 줄 이상의 TypeScript, macOS, iOS, Android 네이티브 앱, 그리고 50개 이상의 skill module이 있다.

![WhatsApp에서 Clawd와 대화하기](/images/whatsapp-clawd.webp)

## 0. 먼저 몇 가지 용어부터

- `centralized AI assistant`: 대화 기록, memory, control이 주로 플랫폼 쪽에 남는 방식
- `self-hosting`: 소프트웨어를 자신의 machine이나 server에서 실행해 deployment와 data 권한을 되찾는 방식
- `AI agent`: context를 기억하고 tool을 호출하며 task를 실행하는 runtime system
- `chat channel`: WhatsApp, Telegram, Slack, iMessage처럼 이미 쓰고 있는 대화 입구
- `skill`: agent에 설치하는 새 능력으로, 보통 절차, tool call, boundary condition을 포함한다

## 1. Interface보다 Control이 중요하다

대부분의 AI 제품은 문제를 interface 경쟁으로 만든다. 어느 chat window가 더 매끄러운가, 어느 model이 더 잘 답하는가, 어느 subscription이 더 싼가.

Clawdbot은 다른 질문을 던진다. AI가 사용자 대신 오래 일해야 한다면, 왜 그 memory와 execution rights가 기본적으로 third-party platform에 있어야 하는가?

이것은 privacy 구호가 아니다. agent가 file을 읽고, tool을 호출하고, account에 접근하고, long-term preference를 기억하기 시작하면 더 이상 편리한 웹페이지가 아니다. 개인 infrastructure에 가까워진다. 개인 infrastructure의 핵심은 능력만이 아니라 이동 가능성, 감사 가능성, 중지 가능성이다.

## 2. Chat Channel은 장식이 아니다

Clawdbot은 agent를 WhatsApp, Telegram, Slack, iMessage 같은 기존 chat surface에 연결한다.

중요한 것은 platform 수가 아니다. task entry point가 새 앱 하나에 갇히지 않는다는 점이다. 사용자는 AI 제품을 떠올리고, 열고, context를 붙여넣을 필요가 없다. agent가 일이 이미 시작되는 대화 안에 들어온다.

이것은 사용 빈도를 바꾼다. 별도 창 안에만 있는 agent는 attention을 두고 경쟁한다. 기존 channel 안에 있는 agent는 task 자체를 두고 경쟁한다.

## 3. Self-Hosting은 자동으로 안전하지 않다

self-hosting은 power를 되돌려준다. 하지만 그 power가 잘 다뤄진다는 보장은 없다.

command를 실행하고, memory를 저장하고, skill을 설치할 수 있는 agent는 공격면도 넓다. prompt injection, 잘못된 tool call, skill supply chain, 넓은 permission은 "model이 틀렸다"를 "system이 잘못 실행했다"로 바꾼다.

Clawdbot의 유용한 교훈은 단순히 local에서 돈다는 점이 아니다. agent를 permission boundary가 있는 system으로 보게 만든다는 점이다. model은 한 구성요소일 뿐이다. memory, tools, identity, approval, logs가 agent가 실제 workflow에 오래 들어갈 수 있는지를 결정한다.

## 4. 바뀐 질문

Clawdbot은 최종 형태의 제품이 아니다. 초기 샘플에 가깝다.

그것은 질문을 "어느 AI assistant가 더 똑똑한가"에서 "personal agent를 누가 control하는가"로 바꾼다. 이 질문이 더 낮은 층에 있고, 피하기도 어렵다.

언젠가 각자가 장기 실행되는 AI system을 갖게 된다면 분기점은 chat interface가 아닐 것이다. 세 가지 ownership이다. context는 누구의 것인가, execution rights는 누구의 것인가, failure cost는 누가 부담하는가.

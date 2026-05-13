# Justin Huang Site

[English](./README.md) | [简体中文](./README_ZH-CN.md) | [繁體中文](./README_ZH-TW.md) | **한국어**

이 브랜치에는 [justinhuangai.github.io](https://justinhuangai.github.io)를 구동하는 Astro 소스가 들어 있습니다.

사이트는 AI, 기술, 글쓰기, 삶에 대해 다루는 개인 글쓰기 공간입니다. 시각 언어와 상호작용은 Aither 디자인 시스템 위에 구축되어 있지만, 이 저장소는 실제 운영 중인 사이트 인스턴스이지 범용 테마 템플릿이 아닙니다.

## 이 프로젝트의 특징

- 하나의 콘텐츠 모델을 사람 독자와 AI agent가 함께 사용합니다.
- 기계 판독용 엔드포인트가 핵심 기능입니다: `/protocol.json`, `/agent/home.json`, `/skill.md`, `/llms.txt`, `/llms-full.txt`, `/api/posts.json`, 게시글별 `.md`.
- 런타임은 다국어 사이트이며, 라우팅, UI, 검증이 하나의 locale 설정을 공유합니다.
- 읽기 경험은 디자인 주도형이지만, 기반은 Astro 정적 아키텍처로 안정적으로 유지됩니다.

## 현재 스택

- Astro 6
- Tailwind CSS v4
- React 19 islands
- `gh-pages` 브랜치에서 GitHub Pages 배포
- 현재 locale: `en-US`, `zh-CN`, `zh-TW`, `ko-KR`

## 브랜치 규칙

- `main`: 공개 저장소 랜딩 페이지 설명
- `gh-pages`: 운영 사이트 소스와 배포의 단일 진실
- `source` 같은 임시 브랜치: 실험용, 정식 배포 흐름에 포함되지 않음

운영 절차는 [MAINTAINING.md](./MAINTAINING.md)에 정리되어 있습니다.

## 빠른 시작

```bash
pnpm install
pnpm dev
pnpm validate
```

## 검증 체인

이제 `pnpm validate`는 전체 배포 안전 체인을 실행합니다.

- `pnpm check:content`: 설정된 locale 간 게시글 커버리지를 확인하고 게시글 안의 수상한 수식 마크업도 잡아냅니다
- `pnpm check`: Astro 진단 실행
- `pnpm typecheck`: TypeScript 무출력 검사
- `pnpm build`: 정적 사이트 빌드
- `pnpm check:agent`: 빌드 결과물의 agent 프로토콜 엔드포인트 smoke test

## 핵심 파일

- `config/locale-meta.mjs`: 지원 locale의 단일 진실
- `src/config/site.ts`: 사이트 아이덴티티, 내비게이션, 분석, 댓글, UI 기본값
- `src/content/`: 다국어 콘텐츠
- `src/lib/agent-protocol.ts`: 기계 판독용 발견, 프로토콜, 정책 문서
- `.github/workflows/deploy-github-pages.yml`: GitHub Pages 배포 파이프라인
- `MAINTAINING.md`: 배포와 업그레이드 가이드

## 배포

`origin/gh-pages`에 푸시하면 `.github/workflows/deploy-github-pages.yml`의 GitHub Pages 워크플로가 실행됩니다.

푸시 전에는 다음을 먼저 실행하세요.

```bash
pnpm validate
```

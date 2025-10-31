# Platform Architecture Overview

이 섹션에서는 Vision AI Training Platform의 전체 아키텍처와 핵심 설계 패턴을 설명합니다.

## 목차

### [2.1 High-Level Architecture](./high-level-architecture.md)
플랫폼의 전체 구조, End-to-End 플로우, 3-Tier 아키텍처를 설명합니다.

**주요 내용**:
- 사용자 → Frontend → Backend → Training → Results 플로우
- Presentation / Application / Training Execution Layer
- 시스템 컴포넌트 다이어그램

**대상 독자**: 모든 팀원 (플랫폼 전체 이해)

---

### [2.2 Core Design Patterns](./design-patterns.md)
플랫폼의 핵심 설계 패턴과 그 구현 방법을 설명합니다.

**주요 내용**:
- **Adapter Pattern**: 다양한 ML 프레임워크 통합
- **Strategy Pattern**: 실행 모드 선택 (Subprocess vs Docker)
- **Observer Pattern**: Callbacks 시스템 (MLflow, DB, WebSocket)

**대상 독자**: 백엔드, 모델/학습 개발자

---

### [2.3 Data Flow](./data-flow.md)
데이터와 제어 흐름을 단계별로 설명합니다.

**주요 내용**:
- 학습 작업 생성 플로우
- 학습 실행 플로우
- 메트릭 수집 및 전송 플로우
- 실시간 업데이트 메커니즘

**대상 독자**: 백엔드, 데이터 엔지니어

---

## 왜 이 섹션을 읽어야 하나요?

- **새 팀원**: 플랫폼의 전체 구조를 빠르게 파악
- **기능 개발**: 새로운 기능이 어느 레이어에 속하는지 이해
- **디버깅**: 문제가 발생한 지점을 식별
- **설계 리뷰**: 아키텍처 결정의 근거 파악

---

[← 돌아가기](../README.md)

# Decision Log - Vision AI Training Platform

## 2025-10-22: LLM Config 저장 문제 해결 방향

### 🎯 문제 상황

**증상:**
- Multi-step conversation에서 config 데이터가 누락됨
- Step 1: framework, model_name, task_type ✅ 저장
- Step 2: dataset_path ❌ 저장 안됨
- Step 3: epochs, batch_size, learning_rate ❌ 저장 안됨

**근본 원인:**
- LLM(Gemini)이 `current_config` 필드에 이전 값을 포함시키지 않음
- 새로운 정보만 반환하거나 null/빈 객체 반환

### 🔀 검토한 해결 방안

#### Option A: 한 번에 모든 정보 입력
**장점:**
- 즉시 작동 (5분)
- 구현 필요 없음

**단점:**
- 사용자 경험 나쁨
- 자연스러운 대화 불가능
- MVP의 핵심 가치 훼손

**결정:** ❌ 채택 안함 (임시 테스트용으로만 사용)

---

#### Option B: LLM 프롬프트 강화
**장점:**
- 자연어 대화 유지
- 핵심 기능 보존
- 상대적으로 빠른 구현 (1일)

**단점:**
- 100% 확신 불가 (LLM 행동 예측 어려움)
- 여러 번 시도 필요할 수 있음

**결정:** ✅ **먼저 시도** (2025-10-22 ~ 10-23)

---

#### Option C: 프론트엔드 폼 추가
**장점:**
- 100% 확실한 작동
- 좋은 fallback 옵션
- 숙련 사용자에게 편리

**단점:**
- 추가 개발 시간 (3-4시간)
- 자연어 기능과 별도로 관리 필요
- 통합 복잡도 증가

**결정:** ⏸️ **Option B 실패 시 고려**

---

### 💡 최종 결정

**Phase 1 (10/22 ~ 10/23): Option B 집중**
- LLM 프롬프트를 강화해서 `current_config` 반환 강제
- 다양한 프롬프트 엔지니어링 기법 시도
- 충분한 테스트 및 검증

**Phase 2 (10/24 ~ 필요시): Option C**
- Option B가 실패하거나 불안정할 경우
- 프론트엔드 폼을 **LLM과 연동**해서 구현
- Hybrid 방식: Chat으로 config 채우기 → 폼에서 확인/수정 → 제출

### 🎯 핵심 원칙

> **"자연어 기반 AI 모델 학습"은 이 프로젝트의 핵심 차별점이다.**
>
> 단기 편의를 위해 핵심 기능을 우회하면, 나중에 통합 시 큰 리팩토링이 필요하다.
> 지금 제대로 맞추는 것이 장기적으로 효율적이다.

### 📝 구현 계획 (Option B)

#### 1단계: 프롬프트 구조 개선
- `llm_structured.py`의 system prompt 강화
- CRITICAL RULE 추가
- 명시적인 예제 제공

#### 2단계: Few-shot Learning
- 성공/실패 예제를 프롬프트에 포함
- 올바른 동작 패턴 학습

#### 3단계: Validation & Retry
- LLM 응답 검증 로직
- current_config 누락 시 재시도

#### 4단계: 대안 모델 테스트
- Gemini 2.0 Flash
- GPT-4o (비용 고려)
- Claude 3.5 Sonnet (API 제한 고려)

### 📊 성공 기준

**Minimum Viable:**
- 3-step flow에서 config 데이터 100% 보존
- 테스트 케이스 10개 중 8개 이상 성공

**Target:**
- 5-step flow까지 안정적 동작
- 테스트 케이스 10개 중 9개 이상 성공

**Acceptable Fallback:**
- 8개 이상 성공하지만 가끔 실패
- → 프론트엔드에 "수동으로 확인" 버튼 추가

### 🚫 안티패턴 방지

**하지 말아야 할 것:**
- ❌ 단기 해결을 위해 핵심 기능 우회
- ❌ LLM 없이 작동하는 완전 별도 시스템 구축
- ❌ "나중에 합치자"는 생각 (기술 부채 누적)

**해야 할 것:**
- ✅ 핵심 가치에 집중
- ✅ 점진적 개선 (iterative improvement)
- ✅ 실패해도 배운 것을 문서화

### 📅 타임라인

**10/22 (오늘):**
- [x] 문제 원인 분석 완료
- [x] 해결 방향 결정
- [ ] LLM 프롬프트 개선 시작
- [ ] 초기 테스트

**10/23 (내일):**
- [ ] 프롬프트 iteration (최소 3회)
- [ ] 다양한 시나리오 테스트
- [ ] 성공/실패 판단

**10/24 (필요시):**
- [ ] Option C로 전환 (필요시)
- [ ] 또는 학습 로직 구현 시작

### 🔗 관련 문서

- [CONVERSATION_STATE_ARCHITECTURE.md](./CONVERSATION_STATE_ARCHITECTURE.md) - Phase 1+2 아키텍처
- [ANALYSIS.md](../ANALYSIS.md) - 문제 분석 상세
- [CLAUDE.md](../CLAUDE.md) - 프로젝트 전체 가이드

### ✍️ 작성자 노트

이 결정은 "빠른 구현"보다 "올바른 구현"을 선택한 것입니다.
MVP의 핵심 가치를 지키면서도, 실용적인 fallback 계획을 가지고 있습니다.

Option B가 실패하더라도, 시도 과정에서 얻은 지식은 Option C 구현 시 큰 도움이 될 것입니다.

---

**마지막 업데이트:** 2025-10-22 09:30 KST
**다음 리뷰:** 2025-10-23 (Option B 결과 평가)

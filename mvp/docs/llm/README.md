# LLM 제어 구현 문서

Vision AI Training Platform의 자연어 인터페이스 구현을 위한 종합 문서입니다.

---

## 📚 문서 목록

### 1. [LLM_CONTROL_STRATEGY.md](./LLM_CONTROL_STRATEGY.md) ⭐ **시작점**

**전체 구현 전략 및 아키텍처**

- 현재 상태 분석
- 목표 및 요구사항
- Dual-Track 아키텍처 (Gemini + MCP)
- 사용자 시나리오
- 구현 로드맵 (Phase 1-5)
- 기술 스택 및 의사결정

**👉 이 문서부터 읽으세요!**

---

### 2. [MCP_IMPLEMENTATION_GUIDE.md](./MCP_IMPLEMENTATION_GUIDE.md)

**MCP Server 구현 실무 가이드**

- MCP 기초 개념
- 개발 환경 설정
- Tools 구현 (Training, Inference, Dataset, Model)
- Resources 구현
- Prompts 구현
- 인증 및 권한 관리
- 에러 처리
- 테스트 및 배포

**대상**: MCP Server를 구현할 Backend 개발자

---

### 3. [INTENT_MAPPING.md](./INTENT_MAPPING.md)

**발화 패턴 → 기능 매핑 참조**

- 인텐트 분류 체계
- 모든 인텐트별 발화 패턴
- 엔티티 추출 규칙
- API/Tool 매핑
- 컨텍스트 관리 전략
- 복합 Intent 처리
- 예제 대화

**대상**:
- LLM 프롬프트 작성자
- Intent Parser 개발자
- 테스트 케이스 작성자

---

### 4. [GEMINI_TRACK_ENHANCEMENT.md](./GEMINI_TRACK_ENHANCEMENT.md)

**Gemini State Machine 확장 가이드**

- 현재 State Machine 구조 분석
- 새로운 State 추가 방법
- 새로운 Action 추가 방법
- Tool Registry 구현
- Multi-Intent 지원
- Frontend 연동

**대상**:
- Gemini Track (Web UI)을 확장할 개발자
- Phase 1 구현 담당자

---

## 🎯 사용 가이드

### 처음 시작하는 경우

1. **[LLM_CONTROL_STRATEGY.md](./LLM_CONTROL_STRATEGY.md)** 읽기
   - 전체 그림 이해
   - Dual-Track 아키텍처 파악
   - 구현 Phase 확인

2. **역할별 문서 선택**
   - **PM/기획자**: LLM_CONTROL_STRATEGY.md 섹션 2, 6 (요구사항, 로드맵)
   - **Backend (MCP)**: MCP_IMPLEMENTATION_GUIDE.md
   - **Backend (Gemini)**: GEMINI_TRACK_ENHANCEMENT.md
   - **Frontend**: LLM_CONTROL_STRATEGY.md 섹션 3.2, 3.3
   - **QA/테스터**: INTENT_MAPPING.md 섹션 10

3. **구현 시작**
   - Phase 1부터 순차적으로 진행
   - 각 Phase별 체크리스트 활용

### 특정 작업을 위한 경우

| 작업 | 참조 문서 | 섹션 |
|------|----------|------|
| **새로운 인텐트 추가** | INTENT_MAPPING.md | 해당 카테고리 |
| **MCP Tool 추가** | MCP_IMPLEMENTATION_GUIDE.md | 섹션 4 |
| **Gemini State 추가** | GEMINI_TRACK_ENHANCEMENT.md | 섹션 2 |
| **프롬프트 개선** | INTENT_MAPPING.md | 섹션 11 |
| **에러 처리** | MCP_IMPLEMENTATION_GUIDE.md | 섹션 3.3 |
| **테스트 작성** | INTENT_MAPPING.md | 섹션 10 |

---

## 🏗️ 아키텍처 개요

Vision AI Platform은 **Dual-Track 접근 방식**을 사용합니다:

### Track 1: Gemini State Machine (Web UI)

```
사용자 (Web) → ChatPanel → Gemini API → State Machine → Action Handlers → Backend APIs
```

**특징:**
- 초보자 친화적
- 단계별 가이드
- 한국어 최적화
- 명확한 선택지

### Track 2: MCP Server (Advanced/API)

```
사용자 (Claude Code/API) → MCP Client → MCP Server → Tools → Backend Services
```

**특징:**
- 고급 사용자용
- 프로그래매틱 제어
- 자동화 가능
- 유연한 워크플로우

**공통 Backend:**
- Training Service
- Inference Service
- Dataset Analyzer
- Model Registry
- Project Manager

---

## 📊 구현 로드맵

### Phase 1: Gemini Track 확장 (2주)
- [x] 현재 상태 분석
- [ ] State/Action 확장
- [ ] Tool Registry 구현
- [ ] Multi-Intent 지원
- [ ] Frontend 업데이트

### Phase 2: MCP Server 구현 (3주)
- [ ] MCP 기본 구조
- [ ] Training Tools
- [ ] Dataset Tools
- [ ] Model Tools
- [ ] Inference Tools

### Phase 3: Resources & Prompts (1주)
- [ ] Training Resources
- [ ] Validation Resources
- [ ] Model Catalog Resource
- [ ] Recommendation Prompts

### Phase 4: 통합 및 테스트 (2주)
- [ ] E2E 테스트
- [ ] 성능 최적화
- [ ] 문서화
- [ ] Claude Code 통합

### Phase 5: 고급 기능 (Optional, 2주)
- [ ] AutoML 파이프라인
- [ ] 하이퍼파라미터 튜닝
- [ ] 장기 메모리

---

## 🔑 핵심 개념

### Intent (인텐트)
사용자의 의도를 나타내는 카테고리
- 예: `TRAINING.CREATE`, `INFERENCE.QUICK`, `DATASET.ANALYZE`

### Action (액션)
LLM이 수행할 구체적인 작업
- 예: `ASK_CLARIFICATION`, `START_TRAINING`, `SHOW_RESULTS`

### Entity (엔티티)
발화에서 추출한 정보
- 예: `model_name="resnet50"`, `epochs=100`

### Context (컨텍스트)
대화 기록 및 상태
- 현재 작업 ID
- 사용자 선호도
- 임시 설정

### Tool (도구)
MCP에서 LLM이 호출할 수 있는 함수
- 예: `create_training_job()`, `analyze_dataset()`

### Resource (리소스)
MCP에서 LLM이 읽을 수 있는 데이터
- 예: `training://jobs/12345`, `models://catalog`

---

## 🧪 테스트 전략

### Unit Tests
- Intent 인식 정확도
- Entity 추출 정확도
- Tool 실행 검증

### Integration Tests
- 전체 대화 플로우
- Multi-turn 대화
- 복합 Intent 처리

### E2E Tests
- 실제 사용자 시나리오
- 데이터셋 분석 → 학습 → 추론
- 프로젝트 관리 워크플로우

**테스트 케이스**: [INTENT_MAPPING.md - 섹션 10](./INTENT_MAPPING.md#10-테스트-케이스)

---

## 📈 성공 지표 (KPI)

### 사용성
- **Task Completion Rate**: 85% 이상
- **Average Turns to Completion**: 5턴 이내 (학습 설정)

### 성능
- **LLM Response Time**: P95 < 3초
- **End-to-End Latency**: P95 < 5초

### 품질
- **Intent Recognition Accuracy**: 90% 이상
- **Tool Selection Accuracy**: 95% 이상

### 비용
- **Cost per Conversation**: $0.05 이하
- **Cache Hit Rate**: 40% 이상

---

## 🔧 개발 환경

### 필수 도구
- Python 3.11+
- FastAPI
- MCP SDK (`pip install mcp`)
- Google Gemini API Key
- (Optional) Claude API Key

### 설정 파일

**.env:**
```bash
GOOGLE_API_KEY=your_api_key
DATABASE_URL=sqlite:///./vision_platform.db
```

**.claude/mcp.json:**
```json
{
  "mcpServers": {
    "vision-ai-platform": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "C:/path/to/mvp/backend"
    }
  }
}
```

---

## 🤝 기여 가이드

### 새로운 Intent 추가

1. **[INTENT_MAPPING.md](./INTENT_MAPPING.md)에 문서화**
   - 발화 패턴
   - 엔티티
   - API 매핑

2. **Gemini Track: Action 추가**
   - `ActionType` Enum 확장
   - Action Handler 구현

3. **MCP Track: Tool 추가**
   - Tool 함수 구현
   - Docstring 작성
   - 테스트 작성

4. **프롬프트 업데이트**
   - System Prompt에 추가
   - Few-shot Examples 추가

### 문서 업데이트

- 새로운 기능 추가 시 관련 문서 업데이트
- 변경 이력 기록
- 예제 코드 제공

---

## 📞 문의 및 지원

### 문서 관련 문의
- 이 문서에 대한 피드백이나 질문은 GitHub Issues에 등록해주세요

### 참고 자료
- [MCP 공식 문서](https://www.anthropic.com/news/model-context-protocol)
- [Gemini API 문서](https://ai.google.dev/docs)
- [프로젝트 메인 README](../../README.md)
- [아키텍처 문서](../architecture/ARCHITECTURE.md)
- [API 명세](../api/API_SPECIFICATION.md)

---

## 📝 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| 1.0 | 2025-11-01 | 초안 작성 | Claude Code |

---

**Happy Coding! 🚀**

# 기여 가이드

Vision AI Platform 프로젝트에 기여해주셔서 감사합니다! 이 문서는 프로젝트 기여 방법을 안내합니다.

## 목차
- [행동 강령](#행동-강령)
- [기여 방법](#기여-방법)
- [개발 환경 설정](#개발-환경-설정)
- [코딩 컨벤션](#코딩-컨벤션)
- [커밋 메시지 규칙](#커밋-메시지-규칙)
- [Pull Request 프로세스](#pull-request-프로세스)
- [이슈 작성 가이드](#이슈-작성-가이드)
- [코드 리뷰 가이드](#코드-리뷰-가이드)

---

## 행동 강령

### 우리의 약속

우리는 개방적이고 환영받는 환경을 만들기 위해 다음을 약속합니다:

- **존중**: 모든 기여자를 존중하고 배려합니다
- **개방성**: 다양한 관점과 경험을 환영합니다
- **협력**: 건설적인 피드백을 주고받습니다
- **포용성**: 모든 수준의 경험을 가진 기여자를 환영합니다

### 허용되지 않는 행동

- 괴롭힘, 차별, 모욕적 발언
- 개인 정보 공개
- 트롤링이나 정치적/종교적 논쟁
- 프로젝트와 무관한 홍보

---

## 기여 방법

### 1. 버그 리포트

버그를 발견하셨나요? 다음 정보를 포함하여 이슈를 작성해주세요:

**필수 정보:**
- 운영체제 및 버전
- Python, Node.js 버전
- 재현 가능한 단계
- 예상 동작 vs 실제 동작
- 에러 메시지 (있다면)

**예시:**
```markdown
### 버그 설명
학습 중단 버튼을 클릭해도 워크플로우가 계속 실행됩니다.

### 재현 단계
1. 새 학습 워크플로우 생성
2. 학습 시작
3. 5초 후 "중단" 버튼 클릭
4. 워크플로우가 계속 실행됨

### 환경
- OS: macOS 14.1
- Python: 3.11.6
- Browser: Chrome 120

### 에러 로그
```
ERROR: Workflow cancellation failed
WorkflowExecutionAlreadyCompleted
```
```

### 2. 기능 제안

새로운 기능을 제안하고 싶으신가요?

**포함할 내용:**
- 해결하려는 문제
- 제안하는 솔루션
- 대안 (있다면)
- 추가 컨텍스트 (스크린샷, 예시 코드 등)

**예시:**
```markdown
### 문제
현재 데이터셋을 하나씩만 업로드할 수 있어 불편합니다.

### 제안
여러 데이터셋을 한 번에 업로드할 수 있는 배치 업로드 기능

### 해결 방법
- Drag & Drop으로 여러 폴더 선택
- ZIP 파일로 압축하여 업로드
- 업로드 진행률 표시

### 참고
- Kaggle의 데이터셋 업로드 UI
- [스크린샷 첨부]
```

### 3. 문서 개선

문서 오타, 불명확한 설명, 누락된 정보를 발견하면 PR을 보내주세요!

### 4. 코드 기여

코드 기여는 다음 절차를 따릅니다:

1. **이슈 확인**: 작업할 이슈를 찾거나 새로 생성
2. **포크**: 레포지토리를 포크
3. **브랜치 생성**: `feature/your-feature-name`
4. **코드 작성**: 코딩 컨벤션 준수
5. **테스트**: 모든 테스트 통과 확인
6. **커밋**: Conventional Commits 형식 준수
7. **PR 생성**: 템플릿에 따라 설명 작성
8. **코드 리뷰**: 리뷰 피드백 반영

---

## 개발 환경 설정

### 1. 레포지토리 포크 및 클론

```bash
# 포크
# GitHub에서 "Fork" 버튼 클릭

# 클론
git clone https://github.com/YOUR_USERNAME/vision-platform.git
cd vision-platform

# Upstream 추가
git remote add upstream https://github.com/original-org/vision-platform.git
```

### 2. 개발 환경 설정

```bash
# 자동 설정 (권장)
make dev-setup

# 또는 수동 설정
cp .env.example .env
# .env 파일 편집

make infra-up
make frontend-install
make backend-install-all
make db-migrate
make db-seed
```

상세한 내용은 [DEVELOPMENT.md](DEVELOPMENT.md)를 참고하세요.

### 3. 브랜치 생성

```bash
# develop 브랜치에서 최신 코드 받기
git checkout develop
git pull upstream develop

# 새 브랜치 생성
git checkout -b feature/amazing-feature
```

---

## 코딩 컨벤션

### Python (Backend)

**Formatter:**
```bash
# Black으로 자동 포맷팅
black .

# Import 정렬
isort .
```

**Linter:**
```bash
# flake8으로 검사
flake8 app tests

# mypy로 타입 체크
mypy app
```

**스타일 가이드:**
- PEP 8 준수
- 함수/메서드에 docstring 작성 (Google 스타일)
- Type hints 사용
- 최대 줄 길이: 100자

**예시:**
```python
from typing import List, Optional

def calculate_accuracy(
    predictions: List[int],
    targets: List[int],
    ignore_index: Optional[int] = None
) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        ignore_index: Index to ignore in calculation

    Returns:
        Accuracy score between 0 and 1

    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    if ignore_index is not None:
        mask = [t != ignore_index for t in targets]
        predictions = [p for p, m in zip(predictions, mask) if m]
        targets = [t for t, m in zip(targets, mask) if m]

    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets) if len(targets) > 0 else 0.0
```

### TypeScript/JavaScript (Frontend)

**Formatter:**
```bash
# Prettier로 자동 포맷팅
pnpm format
```

**Linter:**
```bash
# ESLint로 검사
pnpm lint
```

**스타일 가이드:**
- 함수 컴포넌트 + Hooks 사용
- Props에 TypeScript 타입 정의
- `const` 우선 사용
- 명확한 변수/함수명

**예시:**
```typescript
interface TrainingProgressProps {
  workflowId: string;
  onComplete?: () => void;
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  workflowId,
  onComplete
}) => {
  const [progress, setProgress] = useState<number>(0);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/workflows/${workflowId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'training_progress') {
        setProgress(data.data.percentage);
        setMetrics(data.data.metrics);
      } else if (data.type === 'training_complete') {
        onComplete?.();
      }
    };

    return () => ws.close();
  }, [workflowId, onComplete]);

  return (
    <div className="space-y-4">
      <ProgressBar value={progress} label="Training Progress" />
      {metrics && <MetricsDisplay metrics={metrics} />}
    </div>
  );
};
```

---

## 커밋 메시지 규칙

### Conventional Commits

모든 커밋은 [Conventional Commits](https://www.conventionalcommits.org/) 형식을 따릅니다.

**형식:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (동작 변경 없음)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경
- `perf`: 성능 개선

**Scope (선택):**
- `frontend`: Frontend 관련
- `backend`: Backend 관련
- `intent-parser`: Intent Parser 서비스
- `orchestrator`: Orchestrator 서비스
- `model-registry`: Model Registry 서비스
- `data-service`: Data Service
- `vm-controller`: VM Controller
- `telemetry`: Telemetry 서비스
- `infra`: 인프라 관련
- `docs`: 문서

**예시:**

```bash
# 새 기능
feat(frontend): add real-time training progress chart

Implement WebSocket-based live chart that updates as training progresses.
Uses Recharts library for smooth animations.

Closes #123

# 버그 수정
fix(orchestrator): handle timeout error in workflow cancellation

The workflow was not properly handling timeout errors when cancelling,
causing the system to hang indefinitely.

Fixes #456

# 문서
docs(api): update endpoint specifications

Add detailed request/response examples for /workflows endpoint.
Include error cases and rate limiting information.

# Breaking Change
feat(backend)!: migrate from REST to GraphQL API

BREAKING CHANGE: All REST endpoints have been removed.
Clients must migrate to GraphQL API.

Migration guide: docs/MIGRATION_GUIDE.md
```

---

## Pull Request 프로세스

### 1. PR 전 체크리스트

- [ ] 코드가 컨벤션을 따르는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 새 기능에 테스트를 추가했는가?
- [ ] 문서를 업데이트했는가?
- [ ] Breaking change가 있다면 CHANGELOG 업데이트했는가?

### 2. PR 생성

**제목 형식:**
```
<type>(<scope>): <description>
```

**예시:**
```
feat(frontend): add WebSocket-based training monitor
fix(orchestrator): resolve workflow cancellation timeout
docs(api): update inference endpoint examples
```

### 3. PR 설명 템플릿

```markdown
## 변경 사항
<!-- 무엇을 변경했는지 명확하게 설명 -->

## 동기
<!-- 왜 이 변경이 필요한지 설명 -->

## 해결 방법
<!-- 어떻게 구현했는지 설명 -->

## 테스트
<!-- 어떻게 테스트했는지 설명 -->
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 수동 테스트 완료

## 스크린샷
<!-- UI 변경이 있다면 Before/After 스크린샷 첨부 -->

## 체크리스트
- [ ] 코드 컨벤션 준수
- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] Breaking change 확인

## 관련 이슈
Closes #123
Related to #456
```

### 4. 코드 리뷰 대응

- 리뷰어의 피드백을 존중하고 건설적으로 대화
- 변경 요청 사항을 반영하고 코멘트로 알림
- 불분명한 피드백은 질문으로 명확히
- 리뷰어에게 감사 표시

### 5. Merge 전 확인

- CI 통과 확인
- 충돌 해결
- Reviewer의 Approval 받기
- 최신 develop 브랜치와 sync

---

## 이슈 작성 가이드

### 버그 이슈 템플릿

```markdown
### 버그 설명
간단명료하게 버그를 설명

### 재현 단계
1. '...'로 이동
2. '...' 클릭
3. '...' 입력
4. 에러 발생

### 예상 동작
무엇이 일어나야 했는지

### 실제 동작
실제로 무엇이 일어났는지

### 환경
- OS: [e.g. macOS 14.1]
- Browser: [e.g. Chrome 120]
- Python: [e.g. 3.11.6]
- Node: [e.g. 20.10.0]

### 에러 로그
```
에러 메시지 붙여넣기
```

### 추가 정보
스크린샷, 비디오 등
```

### 기능 요청 템플릿

```markdown
### 문제/동기
어떤 문제를 해결하고 싶은지

### 제안하는 해결방법
어떻게 구현할지

### 대안
고려한 다른 방법들

### 추가 정보
참고 자료, 예시 등
```

---

## 코드 리뷰 가이드

### 리뷰어를 위한 가이드

**체크 포인트:**

1. **기능성**: 코드가 의도대로 작동하는가?
2. **가독성**: 코드를 이해하기 쉬운가?
3. **유지보수성**: 향후 수정이 용이한가?
4. **성능**: 성능 이슈가 없는가?
5. **보안**: 보안 취약점이 없는가?
6. **테스트**: 충분한 테스트가 있는가?
7. **문서**: 필요한 문서가 있는가?

**피드백 작성 방법:**

✅ **좋은 예:**
```
이 부분은 `Array.reduce()`를 사용하면 더 간결하게 작성할 수 있을 것 같습니다.

예시:
const total = items.reduce((sum, item) => sum + item.value, 0);

이렇게 하면 코드가 더 읽기 쉽고 함수형 프로그래밍 패턴을 따릅니다.
```

❌ **나쁜 예:**
```
이 코드는 별로네요. 다시 작성하세요.
```

**리뷰 종류:**

- **Comment**: 제안이나 질문
- **Approve**: 승인 (Merge 가능)
- **Request Changes**: 변경 요청 (필수 수정 사항)

### PR 작성자를 위한 가이드

**피드백 대응:**

1. 모든 코멘트에 답변
2. 수정 완료 시 "Done" 표시
3. 동의하지 않으면 정중하게 의견 제시
4. 리뷰어에게 감사 표시

**예시:**
```markdown
> 이 함수는 너무 길어서 이해하기 어렵네요. 작은 함수로 나누는 게 어떨까요?

좋은 제안입니다! `validateInput()`, `processData()`, `saveResult()` 3개로 분리했습니다.
커밋: abc1234

감사합니다! 👍
```

---

## 추가 리소스

- [개발 환경 설정](DEVELOPMENT.md)
- [아키텍처 문서](ARCHITECTURE.md)
- [API 명세](API_SPECIFICATION.md)
- [데이터베이스 스키마](DATABASE_SCHEMA.md)
- [디자인 시스템](DESIGN_SYSTEM.md)

---

## 질문이 있으신가요?

- GitHub Issues에 질문 작성
- [Slack 커뮤니티](https://vision-platform.slack.com) 참여
- 이메일: team@vision-platform.com

---

**다시 한 번 기여해주셔서 감사드립니다! 🙏**

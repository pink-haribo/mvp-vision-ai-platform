# 사용자 시나리오 문서

이 폴더는 Vision AI Training Platform의 주요 사용자 시나리오를 **로컬 환경**과 **배포 환경(Railway)** 관점에서 상세히 설명합니다.

## 목적

모델 개발자가 배포 관련 지식이 부족해도 시스템의 동작 흐름을 명확히 이해할 수 있도록 작성되었습니다.

각 시나리오는 다음 내용을 포함합니다:
- ✅ 로컬 환경에서의 상세한 동작 흐름
- ✅ 배포 환경(Railway)에서의 동작 흐름
- ✅ 두 환경의 차이점 비교
- ✅ 관련 코드 파일 및 위치
- ✅ 디버깅 팁

---

## 시나리오 목록

### 1. 유저 로그인 (`01-login-flow.md`)

**요약:** 사용자가 이메일과 비밀번호로 로그인하는 과정

**핵심 차이:**
- 로컬: SQLite 파일 데이터베이스
- 배포: PostgreSQL 관리형 데이터베이스

**배우는 내용:**
- JWT 토큰 생성 및 검증
- LocalStorage 사용
- 데이터베이스 연결 차이 (파일 vs 네트워크)

---

### 2. 프로젝트 조회 (`02-project-list-flow.md`)

**요약:** 로그인한 사용자가 자신의 프로젝트 목록을 조회하는 과정

**핵심 차이:**
- 로컬: SQLite 파일 I/O (~1-2ms)
- 배포: PostgreSQL TCP/IP 연결 (~10-50ms)

**배우는 내용:**
- 인증 미들웨어 (JWT Bearer)
- SQLAlchemy ORM 쿼리
- 연결 풀 (Connection Pool)
- N+1 쿼리 문제 해결

---

### 3. 프로젝트 내 실험 조회 (`03-experiment-list-flow.md`)

**요약:** 특정 프로젝트의 모든 학습 실험 목록을 조회하는 과정

**핵심 차이:**
- 로컬: 로컬 파일시스템 직접 읽기
- 배포: 네트워크 쿼리 (~10-30ms)

**배우는 내용:**
- 프로젝트 권한 확인
- 실험 상태(pending, running, completed, failed)
- 동적 라우팅 (`/projects/[id]`)
- 페이지네이션

---

### 4. 모델 리스트 조회 (`04-model-list-flow.md`)

**요약:** 사용 가능한 모델 목록을 조회하는 과정

**핵심 차이:** (가장 큰 차이!)
- 로컬: Python `import`로 직접 모델 레지스트리 로딩
- 배포: **HTTP API**로 Training Services에서 모델 가져옴

**배우는 내용:**
- 동적 모델 등록 (Dynamic Model Registration)
- 로컬 vs 배포 데이터 출처 차이
- HTTP API 통신
- 프레임워크별 서비스 격리 (timm, ultralytics, huggingface)

**읽어야 할 이유:**
- 이 시나리오가 "왜 배포가 복잡한지" 가장 잘 보여줍니다
- Backend와 Training 코드가 분리된 이유를 이해할 수 있습니다

---

### 5. 학습(실험) 생성 (`05-training-creation-flow.md`)

**요약:** 모델을 선택하고 하이퍼파라미터를 설정해서 학습 작업을 생성하는 과정

**핵심 차이:**
- 로컬: Windows 파일시스템
- 배포: Docker 컨테이너 내부 파일시스템 (ephemeral)

**배우는 내용:**
- TrainingJob 모델 생성
- 출력 디렉토리 생성
- 상태: `pending` (학습 시작 전)
- 데이터셋 경로 검증

**주의:**
- 이 단계에서는 **실제 학습이 시작되지 않습니다!**
- DB에 레코드만 생성됨

---

### 6. 학습(실행) - 학습 버튼 눌렀을 때 (`06-training-execution-flow.md`)

**요약:** "학습 시작" 버튼을 클릭하면 실제 모델 학습이 시작되는 과정

**핵심 차이:** (가장 중요!)
- 로컬: Backend가 **subprocess**로 train.py 직접 실행
- 배포: Backend가 **HTTP POST**로 Training Service에 요청

**배우는 내용:**
- Subprocess vs HTTP API 차이
- Backend와 Training 코드 격리
- 프레임워크별 서비스 라우팅
- 비동기 학습 실행 (BackgroundTasks)
- 학습 완료 후 DB 업데이트

**읽어야 할 이유:**
- 로컬과 배포의 **가장 큰 차이**를 보여줍니다
- 왜 Training Services가 필요한지 이해할 수 있습니다
- 프로덕션 아키텍처의 핵심입니다

---

## 읽는 순서 추천

### 초급: 배포가 처음이라면

```
1. 로그인 (01)
   → 기본적인 HTTP 요청/응답 이해

2. 프로젝트 조회 (02)
   → 데이터베이스 차이 이해

3. 모델 리스트 조회 (04)
   → 로컬 vs 배포의 핵심 차이 이해
```

### 중급: 학습 흐름 이해하고 싶다면

```
3. 실험 조회 (03)
   → 학습 작업의 상태 개념 이해

5. 학습 생성 (05)
   → 학습 작업 생성 과정

6. 학습 실행 (06) ⭐ 가장 중요!
   → 실제 학습이 어떻게 실행되는지
```

### 고급: 전체 흐름 파악

```
모든 시나리오를 순서대로 읽기 (01 → 02 → 03 → 04 → 05 → 06)
```

---

## 주요 차이점 요약표

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **프로토콜** | HTTP | HTTPS (TLS/SSL) |
| **데이터베이스** | SQLite (파일) | PostgreSQL (네트워크) |
| **DB 연결 속도** | ~1-2ms | ~10-50ms |
| **모델 데이터** | Python import | HTTP API |
| **학습 실행** | subprocess | HTTP POST to Training Service |
| **파일시스템** | 로컬 드라이브 | Docker 컨테이너 (ephemeral) |
| **의존성 관리** | 모든 의존성 한 곳 | Backend / Training Services 분리 |
| **스케일링** | 단일 머신 | 수평 확장 가능 (컨테이너 복제) |
| **비용** | 무료 (로컬) | 서비스당 ~$5-10/월 |

---

## 용어 정리

### 로컬 환경 (Development)
- 개발자 컴퓨터에서 실행
- `localhost:3000` (Frontend), `localhost:8000` (Backend)
- SQLite 데이터베이스
- 모든 코드가 같은 컴퓨터에 있음

### 배포 환경 (Production / Railway)
- 클라우드에서 실행
- HTTPS URL (예: `https://backend-production-xxxx.up.railway.app`)
- PostgreSQL 데이터베이스
- Backend와 Training Services가 **별도 컨테이너**

### Subprocess
- 로컬 환경에서 Backend가 Training 스크립트를 **직접 실행**하는 방식
- `subprocess.Popen()` 사용

### HTTP API
- 배포 환경에서 Backend가 Training Services에 **HTTP 요청**으로 학습 시작
- `requests.post()` 사용

### Training Service
- 배포 환경에서 **실제 학습을 실행**하는 별도 서비스
- 프레임워크별로 분리: timm-service, ultralytics-service, huggingface-service

### Docker Container
- 애플리케이션과 의존성을 **격리된 환경**에서 실행
- Railway에서 각 서비스가 독립된 컨테이너로 실행됨

---

## 관련 문서

### 배포 관련
- `docs/production/FRAMEWORK_ISOLATION_DEPLOYMENT.md` - 프레임워크 격리 배포 가이드
- `docs/production/DYNAMIC_MODEL_REGISTRATION.md` - 동적 모델 등록 문서

### 아키텍처
- `docs/architecture/ARCHITECTURE.md` - 전체 시스템 아키텍처
- `docs/architecture/DATABASE_SCHEMA.md` - 데이터베이스 스키마

### 개발
- `CLAUDE.md` - 프로젝트 전체 가이드
- `docs/development/DEVELOPMENT.md` - 개발 환경 설정

---

## 피드백

시나리오 문서에 대한 피드백이나 추가 설명이 필요한 부분이 있다면 이슈를 생성해주세요!

**작성자:** Claude Code
**작성일:** 2024-01-18
**버전:** 1.0

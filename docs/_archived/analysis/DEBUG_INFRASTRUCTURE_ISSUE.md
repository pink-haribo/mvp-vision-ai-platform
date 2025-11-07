# 디버깅 인프라 문제 분석

**날짜**: 2025-10-22
**문제**: print() 및 로깅이 출력되지 않아 디버깅 불가능

---

## 🔴 문제 상황

코드에 다음과 같은 디버깅 코드를 추가했으나 **출력이 전혀 나타나지 않음**:

```python
# conversation_manager.py
print(f"[TRACE-1-LOAD] Session {session_id}")
print(f"  config keys: {list(temp_data['config'].keys())}")

# llm_structured.py
print(f"[TRACE-2-LLM-IN] Passing config to Gemini:")

# action_handlers.py
print(f"[TRACE-4-MERGE] Action handler:")
```

**시도한 방법들 (모두 실패)**:
1. ✗ `print()` 사용
2. ✗ `sys.stderr.write()` + `flush()` 사용
3. ✗ 파일 로깅 (파일이 생성조차 안됨)
4. ✗ `logger.warning()` 사용
5. ✗ Uvicorn `--reload` 사용
6. ✗ `python -u` (unbuffered) 사용
7. ✗ Python 캐시 삭제 (`__pycache__`)

---

## 🔍 근본 원인

### 1. **Uvicorn Reload의 한계**

Uvicorn의 `--reload` 옵션은 파일 변경을 감지하지만:
- 워커 프로세스만 재시작합니다
- **Python import 캐시는 완전히 클리어되지 않습니다**
- 특히 함수 내부의 코드 변경은 제대로 반영되지 않을 수 있습니다

```python
# 이런 변경은 reload가 잘 안됨
def process_message(self, ...):
    print("새로 추가한 디버그 코드")  # ← 이게 반영 안될 수 있음
    existing_code()
```

### 2. **stdout/stderr 버퍼링**

```bash
# 백그라운드 실행 시
nohup venv/Scripts/python.exe -u -m uvicorn app.main:app > log.txt 2>&1 &
```

문제:
- `-u` (unbuffered) 옵션을 써도 일부 출력이 버퍼링됨
- 특히 subprocess로 실행된 워커 프로세스의 출력은 부모로 전달 안될 수 있음

### 3. **여러 프로세스 충돌**

```bash
$ netstat -ano | findstr ":8000"
TCP    127.0.0.1:8000    LISTENING    31156
```

- 우리가 시작한 프로세스가 아닌 **다른 프로세스**가 8000 포트를 사용 중일 수 있음
- `taskkill //F //IM python.exe` 해도 일부 프로세스가 살아남을 수 있음

### 4. **File Logging 실패 원인**

```python
with open("gemini_responses.txt", "a", encoding="utf-8") as f:
    f.write("...")
```

문제:
- 상대 경로 사용 → 작업 디렉토리에 따라 다른 위치에 생성됨
- Uvicorn 워커 프로세스의 작업 디렉토리 != 우리가 생각하는 디렉토리
- 파일 권한 문제
- Exception이 발생해도 우리가 볼 수 없음 (`except: pass`)

---

## ✅ 해결 방법

### 방법 1: 완전한 재시작 (가장 확실)

```bash
# 1. 모든 Python 프로세스 강제 종료
taskkill //F //IM python.exe //T

# 2. 포트 확인
netstat -ano | findstr ":8000"
# → 아무것도 안 나와야 함

# 3. 캐시 완전 삭제
cd mvp/backend
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 4. 새 터미널에서 포그라운드 실행
cd mvp/backend
venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
# → 출력이 직접 보임
```

### 방법 2: 절대 경로 파일 로깅

```python
import os
import datetime

# 절대 경로 사용
LOG_DIR = "C:/Users/flyto/Project/Github/mvp-vision-ai-platform/mvp/data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "trace.log")

def trace_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()  # 즉시 디스크에 쓰기

# 사용
trace_log(f"[TRACE-1] Session {session_id}")
trace_log(f"[TRACE-1] Config: {config}")
```

### 방법 3: HTTP 엔드포인트로 디버그 정보 노출

디버깅 전용 엔드포인트 추가:

```python
# app/api/debug.py
from fastapi import APIRouter

router = APIRouter()

# 전역 변수로 디버그 정보 저장
DEBUG_TRACE = []

def add_trace(message):
    global DEBUG_TRACE
    DEBUG_TRACE.append({
        "timestamp": datetime.now().isoformat(),
        "message": message
    })
    # 최근 100개만 유지
    if len(DEBUG_TRACE) > 100:
        DEBUG_TRACE = DEBUG_TRACE[-100:]

@router.get("/debug/trace")
async def get_trace():
    return {"trace": DEBUG_TRACE}

@router.post("/debug/clear")
async def clear_trace():
    global DEBUG_TRACE
    DEBUG_TRACE = []
    return {"status": "cleared"}
```

사용:
```python
# conversation_manager.py
from app.api.debug import add_trace

add_trace(f"[LOAD] Session {session_id}, config: {config}")

# 테스트
curl http://localhost:8000/api/v1/debug/trace
```

### 방법 4: Database Logging

가장 확실한 방법:

```python
# app/db/models.py
class DebugLog(Base):
    __tablename__ = "debug_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    location = Column(String)  # "conversation_manager.process_message"
    message = Column(Text)
    data = Column(JSON)  # 추가 데이터

# 사용
from app.db.models import DebugLog

db.add(DebugLog(
    location="conversation_manager.process_message",
    message="Loading session",
    data={"session_id": session_id, "config": config}
))
db.commit()

# 조회
SELECT * FROM debug_logs ORDER BY timestamp DESC LIMIT 10;
```

---

## 📋 권장 디버깅 전략

### 단기 (지금 당장)

**Option A - HTTP 엔드포인트 방식** (30분)
- 가장 빠르고 확실
- 브라우저/curl로 즉시 확인 가능
- 코드 변경 최소화

**Option B - 절대 경로 파일 로깅** (15분)
- 간단하지만 파일 확인 필요
- 실시간성이 떨어짐

### 장기 (앞으로)

1. **구조화된 로깅 시스템 구축**
   - Python `logging` 모듈 제대로 설정
   - 로그 레벨별 파일 분리
   - Rotation 설정

2. **전용 디버깅 인프라**
   - Sentry 같은 에러 트래킹 도구
   - OpenTelemetry로 분산 트레이싱
   - 프로덕션과 개발 환경 분리

---

## 🎯 즉시 적용 가능한 해결책

### 현재 config 누락 문제 디버깅용

```python
# app/api/debug.py (새 파일)
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel
import json

router = APIRouter()

@router.get("/debug/session/{session_id}")
async def debug_session(session_id: int, db: Session = Depends(get_db)):
    """세션의 temp_data를 직접 조회"""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        return {"error": "Session not found"}

    return {
        "session_id": session_id,
        "state": session.state,
        "temp_data": session.temp_data,
        "config": session.temp_data.get("config") if session.temp_data else None
    }

@router.get("/debug/last-session")
async def debug_last_session(db: Session = Depends(get_db)):
    """가장 최근 세션 조회"""
    session = db.query(SessionModel).order_by(SessionModel.id.desc()).first()
    if not session:
        return {"error": "No sessions found"}

    return {
        "session_id": session.id,
        "state": session.state,
        "temp_data": session.temp_data,
        "config": session.temp_data.get("config") if session.temp_data else None
    }
```

```python
# app/main.py에 추가
from app.api import debug

app.include_router(debug.router, prefix=f"{settings.API_V1_PREFIX}/debug", tags=["debug"])
```

사용법:
```bash
# Step 1 후
curl http://localhost:8000/api/v1/debug/last-session | python -m json.tool

# Step 2 후
curl http://localhost:8000/api/v1/debug/last-session | python -m json.tool

# 차이를 비교해서 config가 어디서 사라지는지 확인
```

---

## 📝 교훈

1. **로깅 인프라는 프로젝트 초기에 구축해야 함**
   - 문제가 생긴 후에는 디버깅조차 어려움

2. **Uvicorn reload는 완전하지 않음**
   - 중요한 변경은 수동 재시작 필요

3. **print() 디버깅은 신뢰할 수 없음**
   - 특히 백그라운드/프로덕션 환경에서

4. **파일 로깅 시 절대 경로 사용**
   - 상대 경로는 작업 디렉토리에 따라 다름

5. **디버깅 엔드포인트는 매우 유용**
   - HTTP로 상태 조회 가능
   - 실시간 확인 가능

---

## 다음 액션

1. ✅ **HTTP 디버그 엔드포인트 추가** (15분)
2. ✅ **데이터 흐름 추적** (30분)
3. ✅ **버그 수정** (시간 미정)

---

**작성자**: Claude Code
**검토 필요**: 2025-10-23 (디버깅 인프라 개선 계획 수립)

# Platform Quick Start Guide

빠르게 Platform을 실행하는 가이드입니다.

## 필수 요구사항

- Python 3.11+
- Node.js 18+
- pnpm (또는 npm)
- Docker (MinIO용, 선택사항)

## 1. Backend 실행

```bash
cd platform/backend

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

Backend가 http://localhost:8000 에서 실행됩니다.
- API 문서: http://localhost:8000/docs

## 2. Training Service 실행 (선택사항)

실제 학습을 테스트하려면 Training Service도 실행해야 합니다.

```bash
cd platform/training-services/ultralytics

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --port 8001
```

Training Service가 http://localhost:8001 에서 실행됩니다.

## 3. Frontend 실행

```bash
cd platform/frontend

# 의존성 설치
pnpm install

# 개발 서버 실행
pnpm dev
```

Frontend가 http://localhost:3000 에서 실행됩니다.

## 4. MinIO 실행 (선택사항)

S3 스토리지를 사용하려면 MinIO를 실행해야 합니다.

```bash
cd platform/infrastructure

# MinIO 실행
docker-compose -f docker-compose.dev.yml up -d minio

# MinIO 콘솔: http://localhost:9001
# ID: minioadmin
# PW: minioadmin
```

MinIO 콘솔에서 `vision-platform` 버킷을 생성하세요.

## 5. 테스트

1. Backend 확인: http://localhost:8000/docs
2. Frontend 확인: http://localhost:3000
3. Training 페이지: http://localhost:3000/training

## 주의사항

### SUIT 폰트 누락

Frontend가 SUIT 폰트를 찾지 못하면 에러가 발생할 수 있습니다.

**해결 방법 1**: 폰트 다운로드
```bash
cd platform/frontend
mkdir -p fonts
# https://sunn.us/suit/ 에서 SUIT-Variable.woff2 다운로드
# fonts/ 디렉토리에 복사
```

**해결 방법 2**: layout.tsx에서 SUIT 폰트 주석 처리
```tsx
// const suit = localFont({ ... })  // 주석 처리
// className={`${inter.variable} ${suit.variable}`}  // suit.variable 제거
```

### SQLite 데이터베이스

Backend는 자동으로 `platform.db` SQLite 파일을 생성합니다.
초기화하려면 파일을 삭제하고 재시작하세요.

## 트러블슈팅

### Port 충돌

다른 서비스가 이미 포트를 사용 중이라면:

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Python 의존성 설치 실패

torch 설치가 오래 걸리거나 실패하면:

```bash
# CPU 버전으로 설치 (더 빠름)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### pnpm 없음

```bash
npm install -g pnpm
```

또는 npm 사용:
```bash
npm install
npm run dev
```

## 다음 단계

- 학습 작업 생성 테스트
- MinIO에 데이터셋 업로드
- Training Service로 실제 학습 실행

자세한 내용은 각 서비스의 README를 참고하세요.

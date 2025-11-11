# Railway MLflow Service 배포 가이드

MLflow 실험 추적 서버를 Railway에 배포하는 단계별 가이드입니다.

---

## 사전 준비

### 1. S3 Storage 준비

MLflow artifacts를 저장할 S3 호환 스토리지가 필요합니다.

#### 옵션 A: AWS S3 (권장)

1. AWS Console → S3 → Create Bucket
2. Bucket 이름: `vision-platform-prod`
3. Region: `ap-northeast-2` (서울)
4. IAM User 생성 및 Access Key 발급
   - Policy: `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::vision-platform-prod",
        "arn:aws:s3:::vision-platform-prod/*"
      ]
    }
  ]
}
```

**환경 변수:**
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
# AWS_S3_ENDPOINT_URL은 설정하지 않음 (기본 AWS S3 사용)
```

#### 옵션 B: Cloudflare R2 (저렴, S3 호환)

1. Cloudflare Dashboard → R2 → Create Bucket
2. Bucket 이름: `vision-platform-prod`
3. R2 API Token 생성
   - Permissions: Object Read & Write

**환경 변수:**
```bash
AWS_ACCESS_KEY_ID=<R2_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<R2_SECRET_ACCESS_KEY>
AWS_S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
```

**비용 비교:**
- AWS S3: $0.023/GB/월 + 전송 비용
- Cloudflare R2: $0.015/GB/월, egress 무료 ✅

#### 옵션 C: Backblaze B2 (가장 저렴)

1. Backblaze Dashboard → B2 Cloud Storage → Create Bucket
2. Application Key 생성

**환경 변수:**
```bash
AWS_ACCESS_KEY_ID=<B2_KEY_ID>
AWS_SECRET_ACCESS_KEY=<B2_APPLICATION_KEY>
AWS_S3_ENDPOINT_URL=https://s3.us-west-002.backblazeb2.com
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
```

---

## Railway 배포

### 방법 1: Railway Dashboard (권장, 초보자용)

#### Step 1: 새 서비스 추가

1. Railway Dashboard → 프로젝트 선택
2. **+ New Service** 클릭
3. **GitHub Repo** 선택
4. Repository: `mvp-vision-ai-platform` 선택
5. Service Name: `mlflow-service`

#### Step 2: Dockerfile 설정

**Settings → Build:**
- **Builder**: Dockerfile
- **Dockerfile Path**: `docker/mlflow/Dockerfile`
- **Docker Context**: `.` (프로젝트 루트)

#### Step 3: 환경 변수 설정

**Settings → Variables → New Variable:**

**필수 변수:**

```bash
# MLflow Backend Store (PostgreSQL)
MLFLOW_BACKEND_STORE_URI=${{Postgres.DATABASE_URL}}

# S3 Artifact Storage (AWS S3 예시)
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# AWS_S3_ENDPOINT_URL=  # AWS S3 사용 시 비워둠

# MLflow Server Config
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000

# Gunicorn Config
GUNICORN_WORKERS=2
GUNICORN_THREADS=4
```

**Cloudflare R2 사용 시:**
```bash
MLFLOW_BACKEND_STORE_URI=${{Postgres.DATABASE_URL}}
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
AWS_ACCESS_KEY_ID=<R2_ACCESS_KEY>
AWS_SECRET_ACCESS_KEY=<R2_SECRET_KEY>
AWS_S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
```

#### Step 4: 배포 설정

**Settings → Deploy:**
- **Start Command**: `/mlflow/docker-entrypoint.sh`
- **Health Check Path**: `/health`
- **Health Check Timeout**: `30` seconds
- **Restart Policy**: `On Failure`

**Settings → Networking:**
- **Generate Domain** 클릭
- 생성된 URL 복사: `https://mlflow-service-production-xxxx.up.railway.app`

#### Step 5: 배포 실행

1. **Settings → General → Branch**: `production` 선택
2. **Deploy Now** 클릭
3. **View Logs** 에서 배포 진행 확인

**예상 로그:**
```
========================================
Starting MLflow Tracking Server
========================================
Backend Store URI: postgresql://...
Artifact Root: s3://vision-platform-prod/mlflow-artifacts
Host: 0.0.0.0
Port: 5000
Workers: 2
Threads: 4
Waiting for PostgreSQL to be ready...
PostgreSQL is ready!
AWS credentials detected
Starting MLflow server...
[INFO] Listening at: http://0.0.0.0:5000
```

#### Step 6: 검증

**브라우저에서 확인:**
```
https://mlflow-service-production-xxxx.up.railway.app
```

MLflow UI가 표시되어야 함 ✅

**API 테스트:**
```bash
curl https://mlflow-service-production-xxxx.up.railway.app/health
# 응답: {"status": "ok"}
```

---

### 방법 2: Railway CLI (고급 사용자용)

#### Step 1: Railway CLI 설치

```bash
# Windows (PowerShell)
iwr https://railway.app/install.ps1 | iex

# macOS/Linux
curl -fsSL https://railway.app/install.sh | sh
```

#### Step 2: Railway 로그인

```bash
railway login
railway link  # 프로젝트 연결
```

#### Step 3: MLflow 서비스 생성

```bash
# 프로젝트 루트에서 실행
cd C:\Users\flyto\Project\Github\mvp-vision-ai-platform

# MLflow 서비스 생성
railway service create mlflow-service

# 환경 변수 설정
railway variables set \
  MLFLOW_BACKEND_STORE_URI='${{Postgres.DATABASE_URL}}' \
  MLFLOW_DEFAULT_ARTIFACT_ROOT='s3://vision-platform-prod/mlflow-artifacts' \
  AWS_ACCESS_KEY_ID='AKIA...' \
  AWS_SECRET_ACCESS_KEY='...' \
  MLFLOW_HOST='0.0.0.0' \
  MLFLOW_PORT='5000' \
  GUNICORN_WORKERS='2' \
  GUNICORN_THREADS='4'

# 배포
railway up --service mlflow-service
```

---

## Backend 및 Training Services 연동

MLflow 서비스가 배포되면, Backend와 Training Services가 이 MLflow를 사용하도록 환경 변수를 업데이트해야 합니다.

### Backend 환경 변수

**Railway Dashboard → backend-service → Variables:**

```bash
# 추가
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app
```

### Training Services 환경 변수

**각 Training Service (timm, ultralytics, huggingface):**

**Railway Dashboard → {framework}-service → Variables:**

```bash
# 추가
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app

# S3 credentials (MLflow artifacts 접근용)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_ENDPOINT_URL=  # AWS S3 사용 시 비워둠, R2 사용 시 설정
```

---

## 로컬 개발 환경 연동

로컬에서 Railway MLflow를 사용하려면 (선택적):

### Backend `.env`

```bash
# 기존
# MLFLOW_TRACKING_URI=http://localhost:5000

# Railway MLflow 사용
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app
```

### Training `.env`

```bash
# 기존
# MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Railway MLflow + AWS S3 사용
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# MLFLOW_S3_ENDPOINT_URL은 AWS S3 사용 시 설정 안 함
```

**권장:** 로컬 개발은 로컬 MLflow 사용, 프로덕션은 Railway MLflow 사용

---

## 트러블슈팅

### 1. MLflow 서비스가 시작 안 됨

**증상:**
```
Error: Could not connect to backend store
```

**해결:**
1. PostgreSQL 서비스 실행 확인: `railway ps`
2. `MLFLOW_BACKEND_STORE_URI` 값 확인: `railway variables`
3. Railway Dashboard → Postgres → Connection URL 확인

### 2. Artifacts 업로드 실패

**증상:**
```
boto3.exceptions.NoCredentialsError: Unable to locate credentials
```

**해결:**
1. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 설정 확인
2. S3 bucket 존재 확인
3. IAM 권한 확인 (s3:PutObject, s3:GetObject)

### 3. Health Check 실패

**증상:**
```
Health check failed: Connection refused
```

**해결:**
1. `MLFLOW_PORT=5000` 설정 확인
2. Railway → Settings → Networking → Port 확인
3. Dockerfile `EXPOSE 5000` 확인

### 4. PostgreSQL 연결 실패

**증상:**
```
FATAL: password authentication failed
```

**해결:**
1. `MLFLOW_BACKEND_STORE_URI=${{Postgres.DATABASE_URL}}` 정확히 입력
2. Railway Dashboard → Postgres → 연결 확인
3. PostgreSQL 서비스 재시작

---

## 비용 예측

### Railway MLflow Service

- **Instance Type**: Hobby (512MB RAM)
- **예상 비용**: $5/월

### S3 Storage (예시: 10GB artifacts)

- **AWS S3**: ~$0.23/월 + egress 비용
- **Cloudflare R2**: ~$0.15/월 (egress 무료)
- **Backblaze B2**: ~$0.05/월 (egress 무료)

**총 예상 비용:**
- Railway MLflow + Cloudflare R2: **~$5.15/월** ✅
- Railway MLflow + AWS S3: **~$5.23/월**

---

## 다음 단계

MLflow 배포가 완료되면:

1. ✅ Backend 환경 변수 업데이트
2. ✅ Training Services 환경 변수 업데이트
3. ✅ 학습 실행 테스트
4. ✅ MLflow UI에서 실험 추적 확인
5. ✅ Artifacts S3에 업로드 확인

---

**작성일**: 2025-01-18
**작성자**: Claude Code
**버전**: 1.0

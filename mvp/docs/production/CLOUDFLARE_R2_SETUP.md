# Cloudflare R2 설정 가이드 (MLflow Artifacts용)

Cloudflare R2는 S3 호환 오브젝트 스토리지로, **egress(다운로드) 비용이 무료**라서 MLflow artifacts 저장에 이상적입니다.

---

## 왜 Cloudflare R2인가?

### 비용 비교

| 서비스 | 저장 비용 (GB/월) | Egress 비용 (GB) | 월 10GB 기준 총 비용 |
|--------|------------------|------------------|---------------------|
| **Cloudflare R2** | $0.015 | **무료** | **$0.15** ✅ |
| AWS S3 | $0.023 | $0.09 | $0.23 + egress |
| Backblaze B2 | $0.005 | 무료 (1GB/day) | $0.05 |

### 장점
- ✅ Egress 무료 (다운로드 제한 없음)
- ✅ S3 API 완전 호환 (boto3 그대로 사용)
- ✅ 빠른 설정 (5분 이내)
- ✅ 무료 티어: 10GB 저장 + 1백만 Class A 요청

### 단점
- ⚠️ Cloudflare 계정 필요
- ⚠️ 일부 고급 S3 기능 미지원 (하지만 MLflow에는 충분)

---

## 설정 단계

### Step 1: Cloudflare 계정 생성

1. https://dash.cloudflare.com/ 접속
2. **Sign Up** (이미 계정 있으면 로그인)
3. 이메일 인증 완료

**참고:** 도메인이 없어도 R2 사용 가능합니다.

---

### Step 2: R2 활성화

1. Cloudflare Dashboard → 왼쪽 메뉴에서 **R2** 클릭
2. **Purchase R2** 또는 **Get Started** 클릭
3. 결제 수단 등록
   - 무료 티어 사용해도 신용카드 등록 필요
   - 10GB 이하면 **$0** 청구됨

4. **Enable R2** 클릭

---

### Step 3: R2 Bucket 생성

1. R2 Dashboard → **Create bucket** 클릭

2. Bucket 설정:
   - **Bucket name**: `vision-platform-prod`
   - **Location**: `Automatic` (자동 선택) 또는 `Asia Pacific (APAC)` 선택
   - **Storage Class**: `Standard` (기본값)

3. **Create bucket** 클릭

**완료!** Bucket이 생성되었습니다.

---

### Step 4: R2 API Token 생성

1. R2 Dashboard → **Manage R2 API Tokens** 클릭

2. **Create API Token** 클릭

3. Token 설정:
   - **Token name**: `vision-platform-mlflow`
   - **Permissions**:
     - ✅ **Object Read & Write** 선택
   - **TTL**: `Forever` (만료 없음) 또는 원하는 기간
   - **Bucket scope**:
     - **Specific buckets** 선택
     - `vision-platform-prod` 선택

4. **Create API Token** 클릭

5. **중요!** Token 정보 복사 및 저장:

```bash
# 화면에 표시되는 정보 (예시)
Account ID: a1b2c3d4e5f6g7h8i9j0
Access Key ID: 1234567890abcdef1234567890abcdef
Secret Access Key: abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

**⚠️ Secret Access Key는 이 화면에서만 표시됩니다!** 반드시 복사해서 안전한 곳에 저장하세요.

---

### Step 5: S3 Endpoint URL 확인

R2의 S3 호환 endpoint는 다음 형식입니다:

```
https://<account_id>.r2.cloudflarestorage.com
```

**예시:**
```
Account ID: a1b2c3d4e5f6g7h8i9j0
→ Endpoint: https://a1b2c3d4e5f6g7h8i9j0.r2.cloudflarestorage.com
```

**확인 방법:**
1. R2 Dashboard → `vision-platform-prod` bucket 클릭
2. 오른쪽 상단 **Settings** 탭
3. **S3 API** 섹션에서 확인

---

## Railway 환경 변수 설정

위에서 얻은 정보로 Railway 환경 변수를 설정합니다.

### MLflow Service

**Railway Dashboard → mlflow-service → Variables:**

```bash
# R2 Credentials
AWS_ACCESS_KEY_ID=1234567890abcdef1234567890abcdef
AWS_SECRET_ACCESS_KEY=abcdef1234567890abcdef1234567890abcdef1234567890abcdef
AWS_S3_ENDPOINT_URL=https://a1b2c3d4e5f6g7h8i9j0.r2.cloudflarestorage.com

# Artifact Root
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts

# Backend Store (PostgreSQL)
MLFLOW_BACKEND_STORE_URI=${{Postgres.DATABASE_URL}}
```

### Training Services

**Railway Dashboard → timm-service, ultralytics-service, huggingface-service → Variables:**

각 Training Service에 동일하게 추가:

```bash
# MLflow Server URL
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app

# R2 Credentials (artifacts 업로드용)
AWS_ACCESS_KEY_ID=1234567890abcdef1234567890abcdef
AWS_SECRET_ACCESS_KEY=abcdef1234567890abcdef1234567890abcdef1234567890abcdef
AWS_S3_ENDPOINT_URL=https://a1b2c3d4e5f6g7h8i9j0.r2.cloudflarestorage.com
```

---

## 로컬 개발 환경 설정 (선택적)

로컬에서 Railway R2를 사용하려면:

### `mvp/backend/.env`

```bash
# MLflow (로컬 또는 Railway)
# 로컬 MLflow 사용
MLFLOW_TRACKING_URI=http://localhost:5000

# 또는 Railway MLflow 사용
# MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app
```

### `mvp/training/.env` (새로 생성)

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# R2 Credentials (Railway R2 사용)
AWS_ACCESS_KEY_ID=1234567890abcdef1234567890abcdef
AWS_SECRET_ACCESS_KEY=abcdef1234567890abcdef1234567890abcdef
AWS_S3_ENDPOINT_URL=https://a1b2c3d4e5f6g7h8i9j0.r2.cloudflarestorage.com

# 또는 로컬 MinIO 사용
# AWS_ACCESS_KEY_ID=minioadmin
# AWS_SECRET_ACCESS_KEY=minioadmin
# MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

**권장:** 로컬은 로컬 MinIO, 프로덕션은 R2 사용

---

## 연결 테스트

### 1. R2 연결 테스트 (Python)

```python
# test_r2_connection.py
import boto3
from botocore.client import Config

# R2 credentials
r2_endpoint = "https://a1b2c3d4e5f6g7h8i9j0.r2.cloudflarestorage.com"
access_key = "1234567890abcdef1234567890abcdef"
secret_key = "abcdef1234567890abcdef1234567890abcdef"
bucket_name = "vision-platform-prod"

# Create S3 client
s3 = boto3.client(
    's3',
    endpoint_url=r2_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4')
)

# Test: List buckets
print("Listing buckets...")
response = s3.list_buckets()
print(f"Buckets: {[b['Name'] for b in response['Buckets']]}")

# Test: Upload file
print("\nUploading test file...")
s3.put_object(
    Bucket=bucket_name,
    Key='test/hello.txt',
    Body=b'Hello from MLflow!'
)
print("✅ Upload successful!")

# Test: Download file
print("\nDownloading test file...")
response = s3.get_object(Bucket=bucket_name, Key='test/hello.txt')
content = response['Body'].read().decode('utf-8')
print(f"Content: {content}")
print("✅ Download successful!")

# Test: List objects
print("\nListing objects...")
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='test/')
for obj in response.get('Contents', []):
    print(f"  - {obj['Key']} ({obj['Size']} bytes)")

print("\n✅ All tests passed!")
```

**실행:**
```bash
cd mvp/training
python test_r2_connection.py
```

**예상 출력:**
```
Listing buckets...
Buckets: ['vision-platform-prod']

Uploading test file...
✅ Upload successful!

Downloading test file...
Content: Hello from MLflow!
✅ Download successful!

Listing objects...
  - test/hello.txt (19 bytes)

✅ All tests passed!
```

### 2. MLflow Artifacts 업로드 테스트

MLflow 배포 후 학습 실행 시 자동으로 테스트됩니다.

**확인 방법:**
1. 학습 실행
2. MLflow UI → Experiments → Artifacts 탭
3. `s3://vision-platform-prod/mlflow-artifacts/...` 경로 확인

**Cloudflare R2 Dashboard에서 확인:**
1. R2 Dashboard → `vision-platform-prod` bucket
2. `mlflow-artifacts/` 폴더 확인
3. 모델 파일 (`.pth`, `.pt`) 확인

---

## 비용 모니터링

### Cloudflare R2 사용량 확인

1. R2 Dashboard → **Usage** 탭
2. 확인 가능한 지표:
   - **Storage**: 현재 저장된 데이터량 (GB)
   - **Class A Operations**: 쓰기/리스트 작업 (무료: 1백만/월)
   - **Class B Operations**: 읽기 작업 (무료: 10백만/월)
   - **Egress**: 다운로드량 (항상 무료!)

### 예상 비용 (월간)

**시나리오: MLflow로 모델 학습 실험 추적**

- **저장 데이터**: 10GB (체크포인트, artifacts)
- **Class A 작업**: 10,000회 (업로드)
- **Class B 작업**: 50,000회 (다운로드)
- **Egress**: 5GB (모델 다운로드)

**계산:**
```
Storage: 10GB × $0.015 = $0.15
Class A: 10,000회 (무료 티어 내)
Class B: 50,000회 (무료 티어 내)
Egress: 5GB × $0 (무료!) = $0

총 비용: $0.15/월
```

**무료 티어 한도:**
- Storage: 10GB/월 무료
- Class A: 1,000,000회/월 무료
- Class B: 10,000,000회/월 무료
- Egress: 무제한 무료

→ **10GB 이하면 완전 무료!** ✅

---

## 보안 권장사항

### 1. API Token 권한 최소화

```
❌ Admin Read & Write (모든 bucket 접근)
✅ Object Read & Write (특정 bucket만)
```

### 2. Token 교체 주기

- 프로덕션: 3-6개월마다 교체
- 개발: 필요시 교체

### 3. Token 저장 위치

```
❌ 코드에 하드코딩
❌ Git에 커밋
✅ Railway 환경 변수
✅ .env 파일 (gitignore 처리)
```

### 4. Public Access 제한

R2 bucket을 public으로 설정하지 마세요:

1. R2 Dashboard → `vision-platform-prod` → Settings
2. **Public Access**: `Disabled` (기본값)
3. MLflow는 API key로 접근

---

## 트러블슈팅

### 1. "AccessDenied" 에러

**증상:**
```
botocore.exceptions.ClientError: An error occurred (AccessDenied)
```

**원인:**
- API Token 권한 부족
- Bucket 이름 오타

**해결:**
1. R2 Dashboard → Manage R2 API Tokens
2. Token 권한 확인: `Object Read & Write`
3. Bucket scope에 `vision-platform-prod` 포함 확인

### 2. "InvalidAccessKeyId" 에러

**증상:**
```
botocore.exceptions.ClientError: An error occurred (InvalidAccessKeyId)
```

**원인:**
- Access Key ID 오타
- 잘못된 credential 사용

**해결:**
1. R2 Dashboard에서 Access Key ID 재확인
2. Railway Variables에 정확히 복사했는지 확인
3. 공백, 줄바꿈 제거

### 3. "NoSuchBucket" 에러

**증상:**
```
botocore.exceptions.ClientError: An error occurred (NoSuchBucket)
```

**원인:**
- Bucket 이름 오타
- Bucket이 삭제됨

**해결:**
1. R2 Dashboard에서 bucket 존재 확인
2. Bucket 이름 정확히 입력: `vision-platform-prod`

### 4. Endpoint 연결 실패

**증상:**
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**원인:**
- 잘못된 Endpoint URL
- Account ID 오류

**해결:**
1. Endpoint URL 형식 확인: `https://<account_id>.r2.cloudflarestorage.com`
2. R2 Dashboard → Settings → S3 API에서 정확한 endpoint 확인

---

## 대안: AWS S3 사용

Cloudflare R2 대신 AWS S3를 사용하려면:

### AWS S3 설정

1. AWS Console → S3 → Create Bucket
2. Bucket 이름: `vision-platform-prod`
3. Region: `ap-northeast-2` (서울)
4. IAM User 생성 및 정책 연결

**환경 변수 (R2와 차이점):**
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# AWS_S3_ENDPOINT_URL은 설정하지 않음 (기본 AWS S3 사용)
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://vision-platform-prod/mlflow-artifacts
AWS_REGION=ap-northeast-2  # 추가 필요
```

**비용:**
- Storage: $0.023/GB/월
- Egress: $0.09/GB (첫 10TB)
- 예상: 10GB 저장 + 5GB egress = ~$0.68/월

---

## 다음 단계

R2 설정이 완료되면:

1. ✅ Railway MLflow 서비스 배포
2. ✅ MLflow UI 접속 확인
3. ✅ 학습 실행 및 artifacts 업로드 테스트
4. ✅ R2 Dashboard에서 파일 확인

---

**작성일**: 2025-01-18
**작성자**: Claude Code
**버전**: 1.0

# MVP Vision AI - Helm Charts

내부망 Kubernetes에 MVP를 배포하기 위한 Helm Charts입니다.

## 사전 요구사항

외부 서비스들이 이미 설치되어 있어야 합니다:
- PostgreSQL
- Redis
- MinIO (또는 S3 호환 스토리지)

## Charts 구성

```
charts/
├── mvp-backend/    # FastAPI 백엔드 (포트 8000)
├── mvp-frontend/   # Next.js 프론트엔드 (포트 3000)
└── mvp-training/   # ML 학습 서비스 (포트 8001)
```

## Docker 이미지 빌드

```bash
# 프로젝트 루트에서 실행
cd mvp

# Backend 이미지 빌드
docker build -t mvp-backend:latest -f backend/Dockerfile.k8s backend/

# Frontend 이미지 빌드
docker build -t mvp-frontend:latest -f frontend/Dockerfile frontend/

# Training 이미지 빌드
docker build -t mvp-training:latest -f training/Dockerfile.k8s training/
```

## 내부 레지스트리에 Push

```bash
# 태그 지정
docker tag mvp-backend:latest your-registry.internal/mvp-backend:latest
docker tag mvp-frontend:latest your-registry.internal/mvp-frontend:latest
docker tag mvp-training:latest your-registry.internal/mvp-training:latest

# Push
docker push your-registry.internal/mvp-backend:latest
docker push your-registry.internal/mvp-frontend:latest
docker push your-registry.internal/mvp-training:latest
```

## Helm 배포

### 1. values 파일 생성

```bash
# values-production.yaml 생성
cat > values-production.yaml << 'EOF'
# ===========================================
# 공통 이미지 설정
# ===========================================
global:
  imageRegistry: "your-registry.internal"
  imagePullSecrets:
    - name: registry-secret

# ===========================================
# 외부 서비스 연결 정보
# ===========================================
postgresql:
  host: "postgresql.database.svc.cluster.local"
  port: 5432
  database: "vision_platform"
  username: "admin"
  existingSecret: "postgresql-secret"
  existingSecretKey: "password"

redis:
  host: "redis.database.svc.cluster.local"
  port: 6379
  database: 0

minio:
  endpoint: "http://minio.storage.svc.cluster.local:9000"
  existingSecret: "minio-secret"
  bucket: "vision-platform"

# ===========================================
# LLM 설정
# ===========================================
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  existingSecret: "llm-secret"

# ===========================================
# Ingress 설정
# ===========================================
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: vision-ai.your-domain.internal
EOF
```

### 2. Secret 생성

```bash
# PostgreSQL Secret
kubectl create secret generic postgresql-secret \
  --from-literal=password=your-password

# MinIO Secret
kubectl create secret generic minio-secret \
  --from-literal=minio-access-key=your-access-key \
  --from-literal=minio-secret-key=your-secret-key

# LLM Secret
kubectl create secret generic llm-secret \
  --from-literal=openai-api-key=your-openai-key \
  --from-literal=jwt-secret=your-jwt-secret

# 이미지 Pull Secret (필요시)
kubectl create secret docker-registry registry-secret \
  --docker-server=your-registry.internal \
  --docker-username=user \
  --docker-password=password
```

### 3. Chart 설치

```bash
# Namespace 생성
kubectl create namespace vision-ai

# Backend 설치
helm upgrade --install mvp-backend ./charts/mvp-backend \
  -n vision-ai \
  -f values-production.yaml \
  --set image.repository=your-registry.internal/mvp-backend

# Training 설치
helm upgrade --install mvp-training ./charts/mvp-training \
  -n vision-ai \
  -f values-production.yaml \
  --set image.repository=your-registry.internal/mvp-training

# Frontend 설치
helm upgrade --install mvp-frontend ./charts/mvp-frontend \
  -n vision-ai \
  -f values-production.yaml \
  --set image.repository=your-registry.internal/mvp-frontend \
  --set backend.publicUrl=http://vision-ai.your-domain.internal/api
```

### 4. 배포 확인

```bash
# Pod 상태 확인
kubectl get pods -n vision-ai

# 로그 확인
kubectl logs -f deployment/mvp-backend -n vision-ai
kubectl logs -f deployment/mvp-training -n vision-ai
kubectl logs -f deployment/mvp-frontend -n vision-ai

# 서비스 확인
kubectl get svc -n vision-ai
```

## 개별 values.yaml 예시

### Backend (mvp-backend)

```yaml
image:
  repository: your-registry.internal/mvp-backend
  tag: "v1.0.0"

postgresql:
  host: "postgresql.database"
  existingSecret: "postgresql-secret"

llm:
  provider: "gemini"  # Gemini 사용시
  model: "gemini-2.0-flash-exp"
  gemini:
    apiKey: ""  # existingSecret 사용 권장
  existingSecret: "llm-secret"

ingress:
  enabled: true
  hosts:
    - host: api.vision-ai.internal
      paths:
        - path: /
          pathType: Prefix
```

### Frontend (mvp-frontend)

```yaml
image:
  repository: your-registry.internal/mvp-frontend

backend:
  url: "http://mvp-backend:8000"
  publicUrl: "https://api.vision-ai.internal"
  wsUrl: "wss://api.vision-ai.internal"

ingress:
  enabled: true
  hosts:
    - host: vision-ai.internal
      paths:
        - path: /
          pathType: Prefix
```

### Training (mvp-training)

```yaml
image:
  repository: your-registry.internal/mvp-training

# GPU 사용시
gpu:
  enabled: true
  count: 1

resources:
  limits:
    cpu: "8"
    memory: "32Gi"
  requests:
    cpu: "4"
    memory: "16Gi"

persistence:
  enabled: true
  storageClass: "local-storage"
  size: "100Gi"
```

## 삭제

```bash
helm uninstall mvp-frontend -n vision-ai
helm uninstall mvp-training -n vision-ai
helm uninstall mvp-backend -n vision-ai
kubectl delete namespace vision-ai
```

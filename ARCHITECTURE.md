# 시스템 아키텍처

## 목차
- [전체 아키텍처](#전체-아키텍처)
- [컴포넌트별 상세](#컴포넌트별-상세)
- [데이터 플로우](#데이터-플로우)
- [기술 스택](#기술-스택)
- [확장성 고려사항](#확장성-고려사항)
- [보안](#보안)

## 전체 아키텍처

### 계층 구조

```
┌─────────────────────────────────────────────────┐
│         Presentation Layer                       │
│         (Next.js Frontend)                       │
│         - Chat Interface                         │
│         - Training Dashboard                     │
│         - Real-time Monitoring                   │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│           API Gateway Layer                      │
│              (Kong)                              │
│         - Authentication                         │
│         - Rate Limiting                          │
│         - Request Routing                        │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│         Application Layer                        │
│  ┌─────────────────┬─────────────────────────┐  │
│  │ Intent Parser   │   Core Services         │  │
│  │ (LLM Engine)    │   (FastAPI)             │  │
│  │ - NLU           │   - Model Registry      │  │
│  │ - Config Gen    │   - Data Service        │  │
│  │ - Context Mgmt  │   - VM Controller       │  │
│  └─────────────────┴─────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│      Orchestration Layer                         │
│    (Temporal + Celery + Redis)                   │
│    - Workflow Management                         │
│    - Task Scheduling                             │
│    - State Management                            │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│        Execution Layer                           │
│    (Kubernetes + Training Runners)               │
│    - On-demand Pod Creation                      │
│    - GPU Resource Management                     │
│    - Sidecar Pattern (User + Platform)          │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│          Data Layer                              │
│  (PostgreSQL, MongoDB, Redis, S3)                │
│  - User & Metadata                               │
│  - Configurations                                │
│  - Training Artifacts                            │
└─────────────────────────────────────────────────┘
```

## 컴포넌트별 상세

### 1. Frontend (Next.js)

**책임:**
- 사용자 인터페이스 제공
- 실시간 학습 상태 모니터링 (WebSocket)
- 자연어 입력 처리
- 학습 결과 시각화

**주요 기술:**
- Next.js 14 (App Router)
- React 18
- TailwindCSS + SUIT 폰트
- Zustand (상태 관리)
- React Query (서버 상태)
- Socket.io (WebSocket)
- Recharts (데이터 시각화)

**파일 구조:**
```
frontend/
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   └── register/
│   ├── (dashboard)/
│   │   ├── projects/
│   │   ├── models/
│   │   ├── datasets/
│   │   └── settings/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── ui/              # Button, Input, Card 등
│   ├── layout/          # Sidebar, Header 등
│   └── features/        # ChatPanel, ProgressPanel 등
├── lib/
│   ├── design-tokens.ts
│   ├── api-client.ts
│   └── websocket.ts
└── hooks/
    ├── useChat.ts
    ├── useWorkflow.ts
    └── useWebSocket.ts
```

**핵심 기능:**
- **Chat Interface**: LLM과 대화하며 모델 설정
- **Progress Panel**: 실시간 학습 진행상황 표시
- **Model Zoo**: 사용 가능한 모델 탐색
- **Dataset Manager**: 데이터셋 업로드 및 관리

---

### 2. Intent Parser (LLM Engine)

**책임:**
- 자연어 → 구조화된 학습 설정 변환
- 대화 컨텍스트 관리
- Config 자동 생성
- 사용자 질의 응답

**기술:**
- LangChain 0.1.x
- Anthropic Claude 3.5 Sonnet / OpenAI GPT-4 Turbo
- FastAPI
- ChromaDB (벡터 저장소, 선택)

**핵심 클래스:**
```python
class IntentParser:
    """자연어를 TrainingIntent로 파싱"""
    
    async def parse(
        self, 
        user_message: str,
        context: ConversationContext
    ) -> ParseResult:
        """
        자연어 메시지를 파싱하여 의도 추출
        
        Returns:
            ParseResult - complete 또는 needs_clarification
        """
        pass
    
    async def clarify(
        self,
        partial_intent: PartialIntent
    ) -> List[Question]:
        """부족한 정보를 채우기 위한 질문 생성"""
        pass
    
    async def generate_config(
        self,
        intent: TrainingIntent
    ) -> TrainingConfig:
        """완성된 의도를 실행 가능한 Config로 변환"""
        pass
```

**데이터 모델:**
```python
class TrainingIntent(BaseModel):
    task_type: Literal["classification", "detection", "segmentation", "anomaly_detection"]
    model_name: str
    model_source: str  # "timm", "huggingface", "ultralytics", etc.
    num_classes: Optional[int]
    class_names: Optional[List[str]]
    dataset: DatasetInfo
    hyperparameters: HyperParameters
    confidence: float  # 파싱 신뢰도
```

---

### 3. Orchestrator (Temporal)

**책임:**
- 워크플로우 전체 생명주기 관리
- 서비스 간 조율
- 실패 처리 및 재시도
- 리소스 할당 관리

**기술:**
- Temporal 1.22.x
- FastAPI (REST API)
- Celery (비동기 작업)
- Redis (상태 캐시)

**Temporal 워크플로우:**
```python
from temporalio import workflow
from datetime import timedelta

@workflow.defn
class TrainingWorkflow:
    """학습 워크플로우 정의"""
    
    @workflow.run
    async def run(self, config: TrainingConfig) -> WorkflowResult:
        # Step 1: 데이터 검증
        dataset_info = await workflow.execute_activity(
            validate_dataset,
            config.dataset,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Step 2: 모델 준비
        model_info = await workflow.execute_activity(
            prepare_model,
            config.model,
            start_to_close_timeout=timedelta(minutes=10)
        )
        
        # Step 3: 리소스 할당
        vm_info = await workflow.execute_activity(
            allocate_vm,
            config.resources,
            start_to_close_timeout=timedelta(minutes=10)
        )
        
        # Step 4: 학습 실행 (최대 24시간)
        training_result = await workflow.execute_activity(
            run_training,
            {
                "config": config,
                "dataset": dataset_info,
                "model": model_info,
                "vm": vm_info
            },
            start_to_close_timeout=timedelta(hours=24),
            heartbeat_timeout=timedelta(minutes=5)
        )
        
        # Step 5: 리소스 정리
        await workflow.execute_activity(
            cleanup_resources,
            vm_info,
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        return training_result
```

**상태 관리:**
- Workflow State: Temporal이 자동 관리
- 실시간 Progress: Redis에 캐시
- 최종 결과: PostgreSQL에 영구 저장

---

### 4. Model Registry

**책임:**
- 다양한 소스의 모델 통합 (timm, HuggingFace, Ultralytics, MMDetection 등)
- Adapter 패턴으로 통일된 인터페이스 제공
- 모델 메타데이터 관리
- 커스텀 이미지 등록 및 관리

**기술:**
- FastAPI
- MongoDB (모델 메타데이터)
- Docker (커스텀 이미지)

**Adapter 패턴:**
```python
from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """모든 모델 소스의 통일된 인터페이스"""
    
    @abstractmethod
    def load_model(
        self, 
        model_name: str, 
        config: dict
    ) -> ModelInfo:
        """모델 정보 로드"""
        pass
    
    @abstractmethod
    def prepare_training_script(
        self, 
        config: TrainingConfig
    ) -> str:
        """학습 스크립트 생성"""
        pass
    
    @abstractmethod
    def normalize_output(
        self, 
        raw_output: dict
    ) -> TrainingResult:
        """출력 정규화"""
        pass

# 구현 예시
class TimmAdapter(ModelAdapter):
    def load_model(self, model_name: str, config: dict):
        return {
            "source": "timm",
            "model": model_name,
            "available": model_name in timm.list_models(),
            "pretrained_weights": ["imagenet", "imagenet21k"]
        }
    
    def prepare_training_script(self, config: TrainingConfig):
        return f"""
import timm
import torch

model = timm.create_model(
    '{config.model_name}',
    pretrained={config.pretrained},
    num_classes={config.num_classes}
)
# ... 학습 코드
"""

class UltralyticsAdapter(ModelAdapter):
    def load_model(self, model_name: str, config: dict):
        return {
            "source": "ultralytics",
            "model": model_name,
            "task": "detection",
            "variants": ["n", "s", "m", "l", "x"]
        }
```

**지원 프레임워크:**
- timm (PyTorch Image Models)
- HuggingFace Transformers
- Ultralytics YOLO
- MMDetection
- MMSegmentation
- Detectron2
- Custom Docker Images

---

### 5. Data Service

**책임:**
- 데이터셋 업로드 및 검증
- 포맷 변환 (COCO, YOLO, Pascal VOC, ImageNet 등)
- 전처리 및 Augmentation
- 버전 관리 (DVC)

**기술:**
- FastAPI
- boto3 (S3 연동)
- pandas, pyarrow
- albumentations (augmentation)
- pycocotools

**핵심 기능:**
```python
class DataService:
    async def upload_dataset(
        self,
        file: UploadFile,
        format: str,
        metadata: dict
    ) -> DatasetInfo:
        """데이터셋 업로드 및 초기 검증"""
        pass
    
    async def validate_dataset(
        self,
        dataset_id: str,
        expected_format: str
    ) -> ValidationResult:
        """
        데이터셋 구조 검증
        - 파일 개수, 포맷, 클래스 분포 등
        """
        pass
    
    async def convert_format(
        self,
        dataset_id: str,
        from_format: str,
        to_format: str
    ) -> DatasetInfo:
        """
        포맷 변환
        예: COCO → YOLO, Pascal VOC → COCO 등
        """
        pass
    
    async def create_dataloader_config(
        self,
        dataset_id: str,
        config: TrainingConfig
    ) -> dict:
        """학습에 필요한 DataLoader 설정 생성"""
        pass
```

---

### 6. VM Controller

**책임:**
- Kubernetes 클러스터 관리
- GPU 리소스 할당
- Training Runner Pod 생성/삭제
- Auto-scaling

**기술:**
- FastAPI
- kubernetes-client (Python)
- Docker

**On-Demand Pod 생성:**
```python
class VMController:
    async def allocate_runner(
        self,
        workflow_id: str,
        requirements: ResourceRequirements
    ) -> VMInfo:
        """
        학습을 위한 Pod 생성
        
        Returns:
            VMInfo - endpoint, GPU 정보 등
        """
        
        # 1. GPU 노드 선택
        node = await self.select_best_node(requirements)
        
        # 2. Pod 정의
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"trainer-{workflow_id}",
                "namespace": "training",
                "labels": {
                    "app": "training-runner",
                    "workflow_id": workflow_id
                }
            },
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "trainer",
                        "image": requirements.image,
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": requirements.gpu_count,
                                "memory": f"{requirements.memory_gb}Gi"
                            }
                        },
                        "env": [
                            {"name": "WORKFLOW_ID", "value": workflow_id},
                            {"name": "TELEMETRY_URL", "value": "..."}
                        ]
                    },
                    {
                        "name": "sidecar",
                        "image": "platform-sidecar:latest",
                        # Sidecar 설정
                    }
                ]
            }
        }
        
        # 3. Pod 생성
        await k8s_client.create_namespaced_pod(
            namespace="training",
            body=pod_manifest
        )
        
        return VMInfo(...)
```

---

### 7. Training Runner (Kubernetes Pod)

**구조:**
```
Pod: trainer-wf_789xyz
├── Container: trainer (User Code)
│   ├── 사용자 학습 스크립트 실행
│   ├── 모델 학습
│   └── 체크포인트 저장
│
└── Container: sidecar (Platform Logic)
    ├── stdout/stderr 모니터링
    ├── 체크포인트 자동 업로드
    ├── 시스템 메트릭 수집
    └── Telemetry 전송
```

**Sidecar 역할:**
```python
class PlatformSidecar:
    async def run(self):
        await asyncio.gather(
            self.monitor_stdout(),      # 로그 파싱 및 메트릭 추출
            self.watch_checkpoints(),   # 체크포인트 자동 업로드
            self.collect_metrics(),     # GPU/메모리 모니터링
            self.heartbeat(),           # 주기적 상태 전송
        )
    
    async def monitor_stdout(self):
        """
        사용자 코드의 stdout 파싱
        "Epoch 5, Loss: 0.234" → metric 추출
        """
        async for line in self.stream_logs():
            # 자동 메트릭 파싱
            if match := re.search(r'Loss:\s*([\d.]+)', line):
                await self.send_metric('loss', float(match.group(1)))
    
    async def watch_checkpoints(self):
        """
        /workspace/*.pth 파일 감지
        → S3 자동 업로드
        """
        pass
```

---

### 8. Telemetry Service

**책임:**
- 실시간 메트릭 수집
- WebSocket을 통한 Frontend 업데이트
- 시계열 데이터 저장

**기술:**
- FastAPI + WebSocket
- Prometheus (메트릭 저장)
- Redis (실시간 캐시)

**플로우:**
```
Training Runner (Sidecar)
    ↓ HTTP POST
Telemetry Service
    ↓ (저장)
Prometheus + Redis
    ↓ (브로드캐스트)
WebSocket → Frontend
```

---

## 데이터 플로우

### 전체 학습 플로우

```
1. User Input (자연어)
   "ResNet50으로 고양이 3종류 분류"
   
2. Frontend → API Gateway → Intent Parser
   ParseResult { task: "classification", model: "resnet50", ... }
   
3. Intent Parser → Orchestrator
   POST /internal/orchestrator/create-workflow
   
4. Orchestrator (Temporal Workflow 시작)
   └─ Activity: validate_dataset
   └─ Activity: prepare_model
   └─ Activity: allocate_vm
   └─ Activity: run_training
   └─ Activity: cleanup
   
5. VM Controller → Kubernetes
   Create Pod "trainer-wf_789xyz"
   
6. Training Runner
   - User Container: 실제 학습 실행
   - Sidecar Container: 모니터링 및 업로드
   
7. Telemetry Service → Frontend
   실시간 메트릭 WebSocket 전송
   
8. Training Complete
   - 모델 가중치 S3 업로드
   - 메타데이터 PostgreSQL 저장
   - Workflow 상태 "completed"
   
9. Frontend 알림
   "학습 완료! 정확도 95.6%"
```

---

## 기술 스택

### Backend Services

| 서비스 | 언어/프레임워크 | 데이터베이스 | 목적 |
|--------|----------------|-------------|------|
| API Gateway | Kong 3.x | - | 라우팅, 인증 |
| Intent Parser | Python/FastAPI + LangChain | Redis | 자연어 처리 |
| Orchestrator | Python/FastAPI + Temporal | PostgreSQL, Redis | 워크플로우 |
| Model Registry | Python/FastAPI | MongoDB | 모델 관리 |
| Data Service | Python/FastAPI | PostgreSQL, S3 | 데이터 처리 |
| VM Controller | Python/FastAPI | Redis | K8s 관리 |
| Telemetry | Python/FastAPI | Prometheus, Redis | 모니터링 |

### Data Stores

| Store | 버전 | 용도 | 데이터 예시 |
|-------|------|------|-----------|
| PostgreSQL | 16 | Primary DB | 사용자, 프로젝트, 실험 |
| MongoDB | 7 | Document Store | Config, 워크플로우 정의 |
| Redis | 7.2 | Cache & Queue | 세션, 실시간 상태, Celery |
| S3/MinIO | - | Object Storage | 데이터셋, 모델, 체크포인트 |
| Prometheus | 2.48 | Time Series | 메트릭 |

### Infrastructure

- **Container Orchestration**: Kubernetes 1.28+
- **Container Runtime**: Docker 24+, containerd
- **GPU Support**: NVIDIA Container Toolkit, Device Plugin
- **CI/CD**: GitHub Actions, ArgoCD
- **IaC**: Terraform 1.6+
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger (선택)

---

## 확장성 고려사항

### 수평 확장 (Horizontal Scaling)

**Stateless 서비스:**
- Frontend: CDN + 여러 인스턴스
- Backend APIs: Load Balancer + Auto-scaling Group
- Training Runners: Kubernetes Auto-scaling

**Stateful 서비스:**
- PostgreSQL: Primary + Read Replicas
- Redis: Redis Cluster (Sharding)
- MongoDB: Replica Set

### 수직 확장 (Vertical Scaling)

**GPU 리소스:**
- Kubernetes Node Pools (T4, V100, A100 별도 관리)
- Spot Instances 활용 (비용 절감)
- Cluster Autoscaler

### 성능 최적화

**캐싱 전략:**
```
L1: Browser Cache (정적 리소스, 24h)
L2: CDN (Next.js 페이지, 1h)
L3: Redis (API 응답, 5m)
L4: PostgreSQL (쿼리 결과 캐싱)
```

**비동기 처리:**
- 무거운 작업: Celery Queue
- 실시간 업데이트: WebSocket
- Batch Processing: Cron Jobs

**데이터베이스 최적화:**
- 인덱스 전략: B-tree, GiST
- Connection Pooling: PgBouncer
- Query Optimization: EXPLAIN ANALYZE

---

## 보안

### 인증 & 인가

**JWT 토큰:**
```
Access Token: 1시간 유효
Refresh Token: 30일 유효
```

**RBAC:**
- Admin: 모든 권한
- User: 자신의 프로젝트만
- Viewer: 읽기 전용

### 네트워크 보안

**Kubernetes Network Policy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: training-isolation
spec:
  podSelector:
    matchLabels:
      app: training-runner
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: platform
```

### 데이터 보안

- **At Rest**: S3 SSE-S3, PostgreSQL TDE
- **In Transit**: TLS 1.3
- **Secrets**: HashiCorp Vault / AWS Secrets Manager
- **Container Security**: Distroless images, Non-root user

### 격리

- **Namespace**: 사용자별 Kubernetes Namespace
- **Resource Quota**: CPU/Memory/GPU 제한
- **Network Policy**: 사용자 간 통신 차단

---

## 다음 단계

- [API 명세 보기](API_SPECIFICATION.md)
- [개발 환경 설정](DEVELOPMENT.md)
- [데이터베이스 스키마](backend/DATABASE_SCHEMA.md)
- [배포 가이드](infrastructure/DEPLOYMENT.md)

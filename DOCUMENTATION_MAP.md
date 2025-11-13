# Documentation Map - Complete Guide

This file helps you navigate all the design documentation for the production-ready Vision AI Training Platform.

## 📍 Quick Navigation

All design documents are now located in **`platform/docs/`**.

### Core Architecture Documents

Located in `platform/docs/architecture/`:

1. **[OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md)** ⭐ **START HERE**
   - Complete system architecture
   - Component responsibilities
   - Data flows and communication patterns
   - Technology stack overview
   - **Read this first to understand the big picture**

2. **[BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md)**
   - FastAPI service design
   - Database schema (PostgreSQL)
   - API endpoints (REST + WebSocket)
   - LLM integration
   - Temporal integration
   - Authentication & authorization

3. **[TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md)**
   - Framework-specific trainers (Ultralytics, timm, HuggingFace)
   - API contract (environment variables + HTTP callbacks)
   - Callback pattern implementation
   - Storage integration (S3)
   - Format conversion layer
   - Split handling (train/val)
   - Error handling

4. **[DATASET_STORAGE_STRATEGY.md](./platform/docs/architecture/DATASET_STORAGE_STRATEGY.md)** 🔥 **NEW**
   - Individual file storage with meta-based snapshots
   - Space-efficient versioning (99% storage savings)
   - Snapshot system for training reproducibility
   - Integrity management (broken/repair states)
   - Automatic snapshots on training start
   - **Essential for understanding dataset versioning**

5. **[DATASET_SPLIT_STRATEGY.md](./platform/docs/architecture/DATASET_SPLIT_STRATEGY.md)** 🔥 **NEW**
   - 3-level priority system (Job > Dataset > Auto)
   - Dataset-level split metadata
   - Job-level split override
   - Text file approach (train.txt, val.txt)
   - Framework-specific implementation
   - **Critical for multi-user dataset sharing**

6. **[MODEL_WEIGHT_MANAGEMENT.md](./platform/docs/architecture/MODEL_WEIGHT_MANAGEMENT.md)** 🔥 **NEW**
   - Dual storage strategy (datasets vs model weights)
   - Pretrained weight caching and custom uploads
   - Checkpoint management (best/last/intermediate)
   - Retention policy (usage + time based)
   - Organization-based quota management
   - **Essential for pretrained weights and checkpoint lifecycle**

7. **[VALIDATION_METRICS_DESIGN.md](./platform/docs/architecture/VALIDATION_METRICS_DESIGN.md)** 🔥 **NEW**
   - Task-agnostic validation system (all CV tasks)
   - Primary metric for best checkpoint selection
   - Flexible metrics storage (standard + custom)
   - Per-class and per-image validation results
   - Trainer callback integration
   - Hybrid storage strategy (DB + S3)
   - **Essential for validation, metrics tracking, and best model selection**

8. **[INFERENCE_DESIGN.md](./platform/docs/architecture/INFERENCE_DESIGN.md)** 🔥 **NEW**
   - Test Run vs Inference Job (evaluate vs predict)
   - 3-tier execution compatibility (subprocess/Kind/K8s)
   - XAI support (Grad-CAM, LIME, SHAP)
   - LLM-based natural language explanations
   - Real-time progress callbacks
   - Hybrid storage strategy (DB + S3)
   - **Essential for model testing, inference, and explainability**

9. **[EXPORT_DEPLOYMENT_DESIGN.md](./platform/docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md)** 🔥 **NEW**
   - Model export to deployment formats (ONNX, TensorRT, CoreML, TFLite, OpenVINO, TorchScript)
   - Framework capability matrix (quality levels by format)
   - Pre/post processing strategy (3-tier: embedded + wrappers + metadata)
   - Platform inference endpoints with tier-based pricing
   - Deployment strategies (download, platform endpoint, edge, container)
   - Version management and validation
   - **Essential for model deployment and production inference**

10. **[PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md)** 🔥 **NEW**
   - 5-tier user role system (admin, manager, engineer_ii, engineer_i, guest)
   - Project and Experiment collaboration (Owner/Member roles)
   - Dataset membership and visibility (public/private)
   - MLflow integration (Project → Experiment → Run)
   - Experiment comparison, starring, and notes
   - Invitation system and JWT authentication
   - Permission validation (Option C: independent with validation at action time)
   - **Essential for collaboration, permissions, and access control**

11. **[USER_ANALYTICS_DESIGN.md](./platform/docs/architecture/USER_ANALYTICS_DESIGN.md)** 🔥 **NEW**
   - User session tracking (login/logout, active/idle time)
   - Resource usage monitoring (GPU/CPU hours, storage, costs)
   - Behavioral analytics (feature usage, model preferences)
   - Time series aggregation (hourly/daily/weekly/monthly)
   - KPI definitions and calculations
   - Comprehensive audit logging (CRUD operations, permissions changes)
   - Analytics API endpoints and dashboard design
   - **Essential for usage tracking, analytics, compliance, and audit trails**

12. **[ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md)** ⚠️ **CRITICAL**
   - Complete dependency isolation principles
   - No shared file system
   - No direct imports
   - API-only communication
   - **This is the foundation of our architecture - must read!**

### Development Process

Located in `platform/docs/development/`:

13. **[3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)** ⭐ **ESSENTIAL**
   - Tier 1: Subprocess (local dev)
   - Tier 2: Kind (local Kubernetes)
   - Tier 3: Production (K8s cluster)
   - Same code, different configurations
   - Configuration strategy
   - Testing approach per tier

### Component Documentation

Located in `platform/*/README.md`:

14. **[platform/README.md](./platform/README.md)**
   - Production-first approach overview
   - Directory structure
   - Quick start guide

15. **[platform/backend/README.md](./platform/backend/README.md)**
    - Backend service overview
    - Running locally
    - API endpoints summary

16. **[platform/frontend/README.md](./platform/frontend/README.md)**
    - Frontend application overview
    - Development setup
    - Tech stack summary

17. **[platform/trainers/README.md](./platform/trainers/README.md)**
    - Training services overview
    - API contract summary
    - Adding new frameworks

18. **[platform/workflows/README.md](./platform/workflows/README.md)**
    - Temporal workflows overview
    - Workflow definitions
    - Activities

19. **[platform/infrastructure/README.md](./platform/infrastructure/README.md)**
    - Kubernetes + Helm overview
    - Terraform structure
    - Deployment configurations

20. **[platform/observability/README.md](./platform/observability/README.md)**
    - Prometheus metrics
    - Grafana dashboards
    - Loki logs
    - Tracing setup

### Master Index

21. **[platform/docs/README.md](./platform/docs/README.md)**
    - Complete documentation index
    - Reading order recommendations
    - Links to all documents

## 📚 Reading Order for New Developers

If you're new to the project, read in this order:

### Phase 1: Understanding (2-3 hours)
1. ✅ [platform/README.md](./platform/README.md) - Get the production-first vision (15 min)
2. ✅ [OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md) - Understand system architecture (45 min)
3. ✅ [ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md) - Learn critical isolation principles (30 min)
4. ✅ [3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md) - Understand development process (45 min)

### Phase 2: Component Deep-Dive (10-11 hours)
5. ✅ [BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md) - Backend internals (1 hour)
6. ✅ [TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md) - Trainer internals (1 hour)
7. ✅ [DATASET_STORAGE_STRATEGY.md](./platform/docs/architecture/DATASET_STORAGE_STRATEGY.md) - Dataset versioning & snapshots (30 min) 🔥 **NEW**
8. ✅ [DATASET_SPLIT_STRATEGY.md](./platform/docs/architecture/DATASET_SPLIT_STRATEGY.md) - Train/val split management (30 min) 🔥 **NEW**
9. ✅ [MODEL_WEIGHT_MANAGEMENT.md](./platform/docs/architecture/MODEL_WEIGHT_MANAGEMENT.md) - Pretrained weights & checkpoint management (45 min) 🔥 **NEW**
10. ✅ [VALIDATION_METRICS_DESIGN.md](./platform/docs/architecture/VALIDATION_METRICS_DESIGN.md) - Validation & metrics system (45 min) 🔥 **NEW**
11. ✅ [INFERENCE_DESIGN.md](./platform/docs/architecture/INFERENCE_DESIGN.md) - Inference & XAI system (45 min) 🔥 **NEW**
12. ✅ [EXPORT_DEPLOYMENT_DESIGN.md](./platform/docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md) - Model export & deployment (45 min) 🔥 **NEW**
13. ✅ [PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md) - Collaboration & permissions (45 min) 🔥 **NEW**
14. ✅ [USER_ANALYTICS_DESIGN.md](./platform/docs/architecture/USER_ANALYTICS_DESIGN.md) - Usage tracking & audit logging (45 min) 🔥 **NEW**
15. ✅ Component READMEs in `platform/*/README.md` (30 min each)

### Phase 3: Implementation (as needed)
16. Refer to specific design docs when implementing features
17. Use 3_TIER_DEVELOPMENT.md for environment setup
18. Use DATASET_STORAGE_STRATEGY.md for dataset handling
19. Use DATASET_SPLIT_STRATEGY.md for split logic
20. Use MODEL_WEIGHT_MANAGEMENT.md for pretrained weights and checkpoints
21. Use VALIDATION_METRICS_DESIGN.md for validation and metrics tracking
22. Use INFERENCE_DESIGN.md for model testing, inference, and XAI
23. Use EXPORT_DEPLOYMENT_DESIGN.md for model export and deployment
24. Use PROJECT_MEMBERSHIP_DESIGN.md for collaboration, roles, and permissions
25. Use USER_ANALYTICS_DESIGN.md for usage tracking and audit logging
26. Use component READMEs for quick reference

## 🗂️ Complete File Listing

```
platform/
├── docs/
│   ├── README.md                              # Master documentation index
│   ├── architecture/
│   │   ├── OVERVIEW.md                        # ⭐ System architecture
│   │   ├── BACKEND_DESIGN.md                  # Backend service design
│   │   ├── TRAINER_DESIGN.md                  # Training service design
│   │   ├── DATASET_STORAGE_STRATEGY.md        # 🔥 Dataset versioning & snapshots (NEW)
│   │   ├── DATASET_SPLIT_STRATEGY.md          # 🔥 Train/val split management (NEW)
│   │   ├── MODEL_WEIGHT_MANAGEMENT.md         # 🔥 Pretrained weights & checkpoint management (NEW)
│   │   ├── VALIDATION_METRICS_DESIGN.md       # 🔥 Validation & metrics system (NEW)
│   │   ├── INFERENCE_DESIGN.md                # 🔥 Inference & XAI system (NEW)
│   │   ├── EXPORT_DEPLOYMENT_DESIGN.md        # 🔥 Model export & deployment (NEW)
│   │   ├── PROJECT_MEMBERSHIP_DESIGN.md       # 🔥 Collaboration & permissions (NEW)
│   │   ├── USER_ANALYTICS_DESIGN.md           # 🔥 Usage tracking & audit logging (NEW)
│   │   └── ISOLATION_DESIGN.md                # ⚠️ Isolation principles
│   └── development/
│       └── 3_TIER_DEVELOPMENT.md              # ⭐ Development process
├── backend/
│   └── README.md                              # Backend overview
├── frontend/
│   └── README.md                              # Frontend overview
├── trainers/
│   └── README.md                              # Trainers overview
├── workflows/
│   └── README.md                              # Workflows overview
├── infrastructure/
│   └── README.md                              # Infrastructure overview
├── observability/
│   └── README.md                              # Observability overview
└── README.md                                  # Platform overview
```

## 🎯 Quick Reference by Task

### "I want to understand the system"
→ Start with [OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md)

### "I want to set up my development environment"
→ Read [3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)

### "I want to understand why we separate backend and trainers"
→ Read [ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md)

### "I want to add a new API endpoint"
→ Refer to [BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md)

### "I want to add support for a new model framework"
→ Refer to [TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md)

### "I want to manage pretrained weights or checkpoints"
→ Refer to [MODEL_WEIGHT_MANAGEMENT.md](./platform/docs/architecture/MODEL_WEIGHT_MANAGEMENT.md)

### "I want to implement validation and metrics tracking"
→ Refer to [VALIDATION_METRICS_DESIGN.md](./platform/docs/architecture/VALIDATION_METRICS_DESIGN.md)

### "I want to implement model testing, inference, or XAI"
→ Refer to [INFERENCE_DESIGN.md](./platform/docs/architecture/INFERENCE_DESIGN.md)

### "I want to export models or implement deployment"
→ Refer to [EXPORT_DEPLOYMENT_DESIGN.md](./platform/docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md)

### "I want to implement user roles, projects, and collaboration"
→ Refer to [PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md)

### "I want to implement usage tracking, analytics, or audit logging"
→ Refer to [USER_ANALYTICS_DESIGN.md](./platform/docs/architecture/USER_ANALYTICS_DESIGN.md)

### "I want to deploy to Kubernetes"
→ Refer to [platform/infrastructure/README.md](./platform/infrastructure/README.md)

### "I want to add monitoring/metrics"
→ Refer to [platform/observability/README.md](./platform/observability/README.md)

## 📝 Document Status

| Document | Status | Lines | Last Updated |
|----------|--------|-------|--------------|
| OVERVIEW.md | ✅ Complete | 600+ | 2025-01-10 |
| BACKEND_DESIGN.md | ✅ Complete (Updated) | 900+ | 2025-11-10 |
| TRAINER_DESIGN.md | ✅ Complete (Updated) | 700+ | 2025-01-10 |
| ISOLATION_DESIGN.md | ✅ Complete | 400+ | 2025-01-10 |
| 3_TIER_DEVELOPMENT.md | ✅ Complete (Updated) | 800+ | 2025-01-10 |
| PROJECT_MEMBERSHIP_DESIGN.md | ✅ Complete | 900+ | 2025-11-10 |
| USER_ANALYTICS_DESIGN.md | ✅ Complete | 1200+ | 2025-11-10 |
| platform/README.md | ✅ Complete | 200+ | 2025-01-10 |
| Component READMEs | ✅ Complete | 100+ each | 2025-01-10 |

## ✨ Recent Updates

### 2025-11-10: Project Membership & User Analytics

**New Documents**:
- **PROJECT_MEMBERSHIP_DESIGN.md**: Complete collaboration and permission system
  - 5-tier user role system (admin → guest)
  - Project & Experiment collaboration with Owner/Member roles
  - Dataset membership and visibility (public/private)
  - MLflow integration (Project → Experiment → Run)
  - Experiment starring, comparison, and notes
  - Invitation system and JWT authentication
  - Permission validation (Option C: independent with validation at action time)

- **USER_ANALYTICS_DESIGN.md**: Comprehensive usage tracking and audit system
  - User session tracking (active/idle time)
  - Resource usage monitoring (GPU/CPU hours, storage, costs)
  - Behavioral analytics (feature usage, model preferences)
  - Time series aggregation (hourly → monthly)
  - KPI definitions and calculations
  - Audit logging for compliance (CRUD operations, permissions changes)
  - Analytics API endpoints and dashboard design

**Updated**:
- **BACKEND_DESIGN.md**: Updated User, Dataset, TrainingJob, and Organization models with new fields and relationships

### 2025-01-10: Development Environment & Storage

#### Tier 1: Hybrid Development Mode
**Updated**: `3_TIER_DEVELOPMENT.md`

Tier 1 이제 하이브리드 접근을 사용합니다:
- **Local Processes** (빠른 iteration): Backend, Frontend, Trainer
- **Docker Compose** (가벼운 서비스): PostgreSQL, Redis, MinIO
- **Kind Cluster** (모니터링 스택): MLflow, Prometheus, Grafana, Temporal, Loki

**장점**:
- 개발 서비스는 subprocess로 빠르게 재시작
- 모니터링 스택은 한 번 띄워두고 재사용
- Production-like 환경에서 개발 가능

### S3 API 일관 사용
**Updated**: `TRAINER_DESIGN.md`, `3_TIER_DEVELOPMENT.md`

모든 Tier에서 S3-compatible API를 사용합니다:
- **Tier 1**: MinIO (Docker Compose) - `localhost:9000`
- **Tier 2**: MinIO (Kind) - `minio.platform.svc:9000`
- **Tier 3**: Cloudflare R2 or AWS S3

**핵심**: 동일한 boto3 코드, 엔드포인트만 다름
```python
# 모든 Tier에서 동일한 코드
s3.download_file(bucket, key, filename)
s3.upload_file(filename, bucket, key)
```

**제거된 패턴**:
- ❌ `STORAGE_TYPE="local"` 분기
- ❌ LocalStorage 클래스
- ❌ 로컬 파일 시스템 직접 접근

**새로운 원칙**:
- ✅ 모든 Tier에서 S3 API 사용
- ✅ 완전한 코드 일관성
- ✅ Production 버그를 로컬에서 조기 발견

## 🔗 External References

For MVP reference (archived):
- MVP codebase: `mvp/` (kept for reference only)
- MVP documentation: `docs/` (old structure)

## 💡 Tips

1. **Use the search function**: All docs are markdown, so you can use VS Code's search (Ctrl+Shift+F) to find specific topics across all files.

2. **Follow cross-references**: Documents link to each other - follow these links to dive deeper into specific topics.

3. **Check the examples**: Every design doc includes code examples - these are production-ready patterns you should follow.

4. **Start with the overview**: Don't skip OVERVIEW.md - it provides essential context for everything else.

5. **Understand isolation first**: ISOLATION_DESIGN.md is critical - violating these principles will cause major problems later.

## 🚀 Next Steps

1. **Read Phase 1 documents** (OVERVIEW, ISOLATION, 3_TIER)
2. **Set up your development environment** using 3_TIER_DEVELOPMENT.md
3. **Pick a component to work on** (backend, trainer, frontend)
4. **Read the specific design doc** for that component
5. **Start implementing** following the patterns in the docs

## ❓ Still Can't Find Something?

If you can't find what you're looking for:
1. Check [platform/docs/README.md](./platform/docs/README.md) - the master index
2. Use VS Code search (Ctrl+Shift+F) across all `platform/docs/` files
3. Look in the component-specific READMEs (`platform/*/README.md`)

---

**Remember**: This is production-ready architecture. Take time to understand the design before coding. The isolation principles and 3-tier development process are **not optional** - they're the foundation of the entire platform.

Happy coding! 🎉

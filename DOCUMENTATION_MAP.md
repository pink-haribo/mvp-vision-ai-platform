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
   - Error handling

4. **[ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md)** ⚠️ **CRITICAL**
   - Complete dependency isolation principles
   - No shared file system
   - No direct imports
   - API-only communication
   - **This is the foundation of our architecture - must read!**

### Development Process

Located in `platform/docs/development/`:

5. **[3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)** ⭐ **ESSENTIAL**
   - Tier 1: Subprocess (local dev)
   - Tier 2: Kind (local Kubernetes)
   - Tier 3: Production (K8s cluster)
   - Same code, different configurations
   - Configuration strategy
   - Testing approach per tier

### Component Documentation

Located in `platform/*/README.md`:

6. **[platform/README.md](./platform/README.md)**
   - Production-first approach overview
   - Directory structure
   - Quick start guide

7. **[platform/backend/README.md](./platform/backend/README.md)**
   - Backend service overview
   - Running locally
   - API endpoints summary

8. **[platform/frontend/README.md](./platform/frontend/README.md)**
   - Frontend application overview
   - Development setup
   - Tech stack summary

9. **[platform/trainers/README.md](./platform/trainers/README.md)**
   - Training services overview
   - API contract summary
   - Adding new frameworks

10. **[platform/workflows/README.md](./platform/workflows/README.md)**
    - Temporal workflows overview
    - Workflow definitions
    - Activities

11. **[platform/infrastructure/README.md](./platform/infrastructure/README.md)**
    - Kubernetes + Helm overview
    - Terraform structure
    - Deployment configurations

12. **[platform/observability/README.md](./platform/observability/README.md)**
    - Prometheus metrics
    - Grafana dashboards
    - Loki logs
    - Tracing setup

### Master Index

13. **[platform/docs/README.md](./platform/docs/README.md)**
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

### Phase 2: Component Deep-Dive (3-4 hours)
5. ✅ [BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md) - Backend internals (1 hour)
6. ✅ [TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md) - Trainer internals (1 hour)
7. ✅ Component READMEs in `platform/*/README.md` (30 min each)

### Phase 3: Implementation (as needed)
8. Refer to specific design docs when implementing features
9. Use 3_TIER_DEVELOPMENT.md for environment setup
10. Use component READMEs for quick reference

## 🗂️ Complete File Listing

```
platform/
├── docs/
│   ├── README.md                              # Master documentation index
│   ├── architecture/
│   │   ├── OVERVIEW.md                        # ⭐ System architecture
│   │   ├── BACKEND_DESIGN.md                  # Backend service design
│   │   ├── TRAINER_DESIGN.md                  # Training service design
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

### "I want to deploy to Kubernetes"
→ Refer to [platform/infrastructure/README.md](./platform/infrastructure/README.md)

### "I want to add monitoring/metrics"
→ Refer to [platform/observability/README.md](./platform/observability/README.md)

## 📝 Document Status

| Document | Status | Lines | Last Updated |
|----------|--------|-------|--------------|
| OVERVIEW.md | ✅ Complete | 600+ | 2025-01-10 |
| BACKEND_DESIGN.md | ✅ Complete | 800+ | 2025-01-10 |
| TRAINER_DESIGN.md | ✅ Complete (Updated) | 700+ | 2025-01-10 |
| ISOLATION_DESIGN.md | ✅ Complete | 400+ | 2025-01-10 |
| 3_TIER_DEVELOPMENT.md | ✅ Complete (Updated) | 800+ | 2025-01-10 |
| platform/README.md | ✅ Complete | 200+ | 2025-01-10 |
| Component READMEs | ✅ Complete | 100+ each | 2025-01-10 |

## ✨ Recent Updates (2025-01-10)

### Tier 1: Hybrid Development Mode
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

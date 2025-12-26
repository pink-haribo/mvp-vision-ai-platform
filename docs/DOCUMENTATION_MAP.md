# 📚 Documentation Map - Complete Navigation Guide

**Last Updated**: 2025-01-17
**Total Documents**: 50+

This map helps you quickly find the right document for your needs. Documents are organized by purpose and reading priority.

---

## 🚀 Quick Start (Start Here!)

**For Claude Code**:
1. **[CLAUDE.md](./CLAUDE.md)** ⭐ **MOST IMPORTANT**
   - Complete guidance for Claude Code when working on this repository
   - Project overview, architecture, conventions, code quality standards
   - Development commands, API structure, environment setup
   - Model export & deployment, custom slash commands
   - **Read this first before any work!**

**For Developers**:
1. **[README.md](./README.md)** - Project overview and quick start
2. **[CONTRIBUTING.md](./CONTRIBUTING.md)** - Contribution guidelines
3. **[platform/README.md](./platform/README.md)** - Production platform overview

---

## 🏗️ Architecture & Design

### Core Architecture

Located in `platform/docs/architecture/`:

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md)** | ⭐⭐⭐ | Complete system architecture, components, data flows | Understanding the big picture |
| **[ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md)** | ⚠️ CRITICAL | Dependency isolation principles (no shared files, API-only) | **Must read before any implementation** |
| **[BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md)** | ⭐⭐ | FastAPI service, database schema, API endpoints, LLM integration | Working on backend |
| **[TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md)** | ⭐⭐ | Framework-specific trainers, callback pattern, storage integration | Working on trainers |

### Data Management

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[DATASET_STORAGE_STRATEGY.md](./platform/docs/architecture/DATASET_STORAGE_STRATEGY.md)** | ⭐⭐ | Individual file storage, meta-based snapshots, versioning (99% space savings) | Implementing dataset features |
| **[DATASET_SPLIT_STRATEGY.md](./platform/docs/architecture/DATASET_SPLIT_STRATEGY.md)** | ⭐⭐ | 3-level priority (Job > Dataset > Auto), train/val split management | Implementing splits |
| **[MODEL_WEIGHT_MANAGEMENT.md](./platform/docs/architecture/MODEL_WEIGHT_MANAGEMENT.md)** | ⭐⭐ | Dual storage (datasets vs weights), pretrained weights, checkpoints | Managing model weights |

### Training & Inference

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[VALIDATION_METRICS_DESIGN.md](./platform/docs/architecture/VALIDATION_METRICS_DESIGN.md)** | ⭐⭐ | Task-agnostic validation, primary metric, per-class metrics, hybrid storage | Implementing validation |
| **[INFERENCE_DESIGN.md](./platform/docs/architecture/INFERENCE_DESIGN.md)** | ⭐⭐ | Test Run vs Inference Job, 3-tier execution, XAI (Grad-CAM, LIME, SHAP), LLM explanations | Implementing inference |
| **[EXPORT_DEPLOYMENT_DESIGN.md](./platform/docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md)** | ⭐⭐ | Export formats (ONNX, TensorRT, CoreML, TFLite), deployment strategies, platform endpoints | Implementing export/deployment |

### Collaboration & Analytics

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md)** | ⭐⭐ | 5-tier user roles, Project/Experiment collaboration, permissions, MLflow integration | Implementing collaboration |
| **[USER_ANALYTICS_DESIGN.md](./platform/docs/architecture/USER_ANALYTICS_DESIGN.md)** | ⭐ | Session tracking, resource usage, behavioral analytics, audit logging | Implementing analytics |

### Error Handling & Operations

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[ERROR_HANDLING_DESIGN.md](./platform/docs/architecture/ERROR_HANDLING_DESIGN.md)** | ⭐ | Centralized error handling, retry logic, error codes | Implementing error handling |
| **[INTEGRATION_FAILURE_HANDLING.md](./platform/docs/architecture/INTEGRATION_FAILURE_HANDLING.md)** | ⭐ | Handling external service failures (MLflow, S3, etc.) | Implementing resilience |
| **[OPERATIONS_RUNBOOK.md](./platform/docs/architecture/OPERATIONS_RUNBOOK.md)** | ⭐ | Production operations, troubleshooting, monitoring | Operating in production |

---

## 🛠️ Conventions & Systems (Essential Reference)

### Model Management

| Document | Purpose | Read When |
|----------|---------|-----------|
| **[MODEL_CAPABILITIES_SYSTEM.md](./docs/MODEL_CAPABILITIES_SYSTEM.md)** 🔥 **NEW** | Convention-based model registry, dynamic framework discovery, GitHub Actions automation | Adding new trainers or models |
| **[ADVANCED_CONFIG_SCHEMA.md](./docs/ADVANCED_CONFIG_SCHEMA.md)** 🔥 **NEW** | Advanced training configuration schema generation and validation | Implementing advanced training configs |

### Export & Deployment

| Document | Purpose | Read When |
|----------|---------|-----------|
| **[EXPORT_CONVENTION.md](./docs/EXPORT_CONVENTION.md)** 🔥 **NEW** | Convention-based export design (not shared base module!), CLI interface, metadata schema, format guidelines | Implementing export for new trainers |
| **[docs/examples/export_template.py](./docs/examples/export_template.py)** | Copy-paste ready export script template (400+ lines) | Creating export script |
| **[platform/trainers/ultralytics/EXPORT_GUIDE.md](./platform/trainers/ultralytics/EXPORT_GUIDE.md)** 🔥 **NEW** | Complete Ultralytics export guide (800+ lines): all 6 formats, capability matrix, runtime wrappers | Exporting Ultralytics models |

---

## 👨‍💻 Development Process

### Environment Setup

Located in `platform/docs/development/`:

| Document | Priority | Purpose | Read When |
|----------|----------|---------|-----------|
| **[3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)** | ⭐⭐⭐ | Tier 1 (subprocess), Tier 2 (Kind), Tier 3 (K8s) - Same code, different configs | Setting up dev environment |
| **[TIER_STRATEGY.md](./platform/docs/development/TIER_STRATEGY.md)** | ⭐⭐ | Development tier strategy and configuration | Understanding tier approach |
| **[TIER0_SETUP.md](./docs/development/TIER0_SETUP.md)** | ⭐ | Tier 0 setup guide (local development) | Initial setup |
| **[IMPLEMENTATION_PLAN.md](./platform/docs/development/IMPLEMENTATION_PLAN.md)** | ⭐ | Implementation roadmap and milestones | Planning implementation |

---

## 📋 Planning & Roadmap

Located in `docs/planning/`:

| Document | Purpose | Read When |
|----------|---------|-----------|
| **[MVP_TO_PLATFORM_CHECKLIST.md](./docs/planning/MVP_TO_PLATFORM_CHECKLIST.md)** 🔥 **TRACKING** | Complete migration checklist with 270+ tasks across 7 phases, progress tracking | Tracking overall progress |
| **[MVP_TO_PLATFORM_MIGRATION.md](./docs/planning/MVP_TO_PLATFORM_MIGRATION.md)** | Migration strategy (Option A: incremental improvement, 6-week plan) | Understanding migration approach |
| **[PHASE_3_5_INFERENCE_PLAN.md](./docs/planning/PHASE_3_5_INFERENCE_PLAN.md)** | Phase 3.5 inference implementation plan | Working on Phase 3.5 |
| **[PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md](./docs/planning/PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md)** | Phase 3.6 export & deployment implementation plan (4 weeks) | Working on Phase 3.6 |
| **[TRAINER_MARKETPLACE_VISION.md](./docs/planning/TRAINER_MARKETPLACE_VISION.md)** 🔥 **NEW** | Future vision: UI-driven trainer upload, auto validation, marketplace (Phase 7, 15-21 weeks) | Planning future features |

---

## 🔧 Component Documentation

Located in `platform/*/README.md`:

| Component | Document | Purpose |
|-----------|----------|---------|
| Platform | [platform/README.md](./platform/README.md) | Production-first approach, directory structure |
| Backend | [platform/backend/README.md](./platform/backend/README.md) | Backend service overview, running locally, API summary |
| Frontend | [platform/frontend/README.md](./platform/frontend/README.md) | Frontend app, development setup, tech stack |
| Trainers | [platform/trainers/README.md](./platform/trainers/README.md) | Training services, API contract, adding frameworks |
| Workflows | [platform/workflows/README.md](./platform/workflows/README.md) | Temporal workflows, activities |
| Infrastructure | [platform/infrastructure/README.md](./platform/infrastructure/README.md) | Kubernetes + Helm, Terraform, deployment configs |
| Observability | [platform/observability/README.md](./platform/observability/README.md) | Prometheus, Grafana, Loki, tracing |

**Ultralytics Trainer**:
- [platform/trainers/ultralytics/README.md](./platform/trainers/ultralytics/README.md) - Ultralytics trainer overview
- [platform/trainers/ultralytics/EXPORT_GUIDE.md](./platform/trainers/ultralytics/EXPORT_GUIDE.md) - Complete export guide

---

## 📊 Frontend Documentation

Located in `platform/docs/frontend/`:

| Document | Purpose |
|----------|---------|
| **[DESIGN_SYSTEM.md](./platform/docs/frontend/DESIGN_SYSTEM.md)** | UI design system, components, patterns |

---

## 🔍 Reference & Analysis

### Kubernetes Refactoring (Legacy)

Located in `docs/k8s_refactoring/`:

| Document | Purpose | Status |
|----------|---------|--------|
| [README.md](./docs/k8s_refactoring/README.md) | K8s refactoring overview | 📚 Reference |
| [ARCHITECTURE_DECISIONS.md](./docs/k8s_refactoring/ARCHITECTURE_DECISIONS.md) | K8s architecture decisions | 📚 Reference |
| [PLUGIN_GUIDE.md](./docs/k8s_refactoring/PLUGIN_GUIDE.md) | Plugin system guide | 📚 Reference |
| [trainer_architecture.md](./docs/k8s_refactoring/trainer_architecture.md) | Trainer architecture analysis | 📚 Reference |
| [implementation_plan.md](./docs/k8s_refactoring/implementation_plan.md) | K8s implementation plan v1 | 📚 Reference |
| [implementation_plan_v2.md](./docs/k8s_refactoring/implementation_plan_v2.md) | K8s implementation plan v2 | 📚 Reference |

### Final Reviews

Located in `platform/docs/reviews/`:

| Document | Purpose |
|----------|---------|
| **[FINAL_DESIGN_REVIEW_2025-01-11.md](./platform/docs/reviews/FINAL_DESIGN_REVIEW_2025-01-11.md)** | Final design review before implementation |

---

## 📝 Work Logs & Sessions

Located in `docs/`:

| Document | Purpose |
|----------|---------|
| **[CONVERSATION_LOG.md](./docs/CONVERSATION_LOG.md)** | Session timeline log (design decisions, technical discussions, next steps) |
| [SESSION_2025-11-14.md](./docs/SESSION_2025-11-14.md) | Session log 2025-11-14 |
| [SESSION_2025-11-14_FRONTEND_DIAGNOSTICS.md](./docs/SESSION_2025-11-14_FRONTEND_DIAGNOSTICS.md) | Frontend diagnostics session |

---

## 🧹 Maintenance & Cleanup

Root level:

| Document | Purpose |
|----------|---------|
| [DOCS_CLEANUP_PLAN.md](./DOCS_CLEANUP_PLAN.md) | Documentation cleanup plan |
| [REPO_CLEANUP_PLAN.md](./REPO_CLEANUP_PLAN.md) | Repository cleanup plan |

---

## 🎯 Quick Reference by Use Case

### "I'm new to the project"
1. ⭐ [CLAUDE.md](./CLAUDE.md) - Complete overview for Claude Code
2. ⭐ [README.md](./README.md) - Project overview
3. ⭐ [platform/docs/architecture/OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md) - System architecture
4. ⚠️ [platform/docs/architecture/ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md) - **CRITICAL principles**
5. ⭐ [platform/docs/development/3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md) - Development process

### "I'm setting up development environment"
→ [3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)
→ [TIER0_SETUP.md](./docs/development/TIER0_SETUP.md)

### "I'm working on backend"
→ [BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md)
→ [platform/backend/README.md](./platform/backend/README.md)
→ [ERROR_HANDLING_DESIGN.md](./platform/docs/architecture/ERROR_HANDLING_DESIGN.md)

### "I'm adding a new trainer/framework"
→ [TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md)
→ [MODEL_CAPABILITIES_SYSTEM.md](./docs/MODEL_CAPABILITIES_SYSTEM.md) - **Model registry**
→ [EXPORT_CONVENTION.md](./docs/EXPORT_CONVENTION.md) - **Export implementation**
→ [docs/examples/export_template.py](./docs/examples/export_template.py) - **Template**

### "I'm implementing model export"
→ [EXPORT_DEPLOYMENT_DESIGN.md](./platform/docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md)
→ [EXPORT_CONVENTION.md](./docs/EXPORT_CONVENTION.md)
→ [platform/trainers/ultralytics/EXPORT_GUIDE.md](./platform/trainers/ultralytics/EXPORT_GUIDE.md) - **Reference implementation**

### "I'm working on datasets"
→ [DATASET_STORAGE_STRATEGY.md](./platform/docs/architecture/DATASET_STORAGE_STRATEGY.md) - Versioning & snapshots
→ [DATASET_SPLIT_STRATEGY.md](./platform/docs/architecture/DATASET_SPLIT_STRATEGY.md) - Train/val splits

### "I'm implementing validation/metrics"
→ [VALIDATION_METRICS_DESIGN.md](./platform/docs/architecture/VALIDATION_METRICS_DESIGN.md)

### "I'm implementing inference/testing"
→ [INFERENCE_DESIGN.md](./platform/docs/architecture/INFERENCE_DESIGN.md)
→ [PHASE_3_5_INFERENCE_PLAN.md](./docs/planning/PHASE_3_5_INFERENCE_PLAN.md)

### "I'm working on collaboration/permissions"
→ [PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md)

### "I'm implementing analytics/tracking"
→ [USER_ANALYTICS_DESIGN.md](./platform/docs/architecture/USER_ANALYTICS_DESIGN.md)

### "I want to check project progress"
→ [MVP_TO_PLATFORM_CHECKLIST.md](./docs/planning/MVP_TO_PLATFORM_CHECKLIST.md) - **270+ tasks tracked**

### "I want to understand future plans"
→ [TRAINER_MARKETPLACE_VISION.md](./docs/planning/TRAINER_MARKETPLACE_VISION.md) - Phase 7 vision
→ [MVP_TO_PLATFORM_MIGRATION.md](./docs/planning/MVP_TO_PLATFORM_MIGRATION.md) - Overall strategy

---

## 📊 Document Status Summary

### Platform Core (platform/docs/architecture/)
| Category | Count | Status |
|----------|-------|--------|
| Architecture | 12 docs | ✅ Complete |
| Development | 4 docs | ✅ Complete |
| Frontend | 1 doc | ✅ Complete |
| Reviews | 1 doc | ✅ Complete |

### Root Docs (docs/)
| Category | Count | Status |
|----------|-------|--------|
| Conventions | 3 docs | ✅ Complete |
| Planning | 5 docs | ✅ Complete |
| Reference (k8s) | 8 docs | 📚 Reference Only |
| Work Logs | 3 docs | 📝 Active |
| Development | 1 doc | ✅ Complete |

### Component READMEs
| Component | Status |
|-----------|--------|
| Platform | ✅ Complete |
| Backend | ✅ Complete |
| Frontend | ✅ Complete |
| Trainers | ✅ Complete |
| Workflows | ✅ Complete |
| Infrastructure | ✅ Complete |
| Observability | ✅ Complete |

**Total Active Documents**: 50+
**Documentation Coverage**: 95%+

---

## 💡 Documentation Best Practices

1. **Always check CLAUDE.md first** - It's the authoritative guide for Claude Code
2. **Follow cross-references** - Documents link to each other for deeper context
3. **Use VS Code search (Ctrl+Shift+F)** - Search across all markdown files
4. **Check examples** - Every design doc includes production-ready code patterns
5. **Understand isolation principles** - ISOLATION_DESIGN.md is foundation, violations cause problems
6. **Track progress** - MVP_TO_PLATFORM_CHECKLIST.md shows what's done and what's next
7. **Log decisions** - Use CONVERSATION_LOG.md for session summaries

---

## 🔗 Document Relationships

```
CLAUDE.md (Master Guide)
    ↓
README.md → platform/README.md
    ↓
platform/docs/architecture/OVERVIEW.md (Architecture Hub)
    ↓
    ├─→ ISOLATION_DESIGN.md (Critical Principles)
    ├─→ BACKEND_DESIGN.md → Component READMEs
    ├─→ TRAINER_DESIGN.md → MODEL_CAPABILITIES_SYSTEM.md → EXPORT_CONVENTION.md
    ├─→ DATASET_*_STRATEGY.md (Storage & Split)
    ├─→ VALIDATION_METRICS_DESIGN.md
    ├─→ INFERENCE_DESIGN.md
    ├─→ EXPORT_DEPLOYMENT_DESIGN.md → EXPORT_CONVENTION.md
    ├─→ PROJECT_MEMBERSHIP_DESIGN.md
    └─→ USER_ANALYTICS_DESIGN.md

platform/docs/development/3_TIER_DEVELOPMENT.md (Development Hub)
    ↓
    ├─→ TIER_STRATEGY.md
    ├─→ TIER0_SETUP.md
    └─→ IMPLEMENTATION_PLAN.md

docs/planning/ (Planning Hub)
    ↓
    ├─→ MVP_TO_PLATFORM_CHECKLIST.md (Progress Tracking)
    ├─→ MVP_TO_PLATFORM_MIGRATION.md
    ├─→ PHASE_3_5_INFERENCE_PLAN.md
    ├─→ PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md
    └─→ TRAINER_MARKETPLACE_VISION.md (Future)
```

---

## 🚀 Recommended Reading Order

### For Claude Code (AI Assistant)
1. **[CLAUDE.md](./CLAUDE.md)** ⭐ - Read this EVERY TIME before working
2. Task-specific documents as referenced in CLAUDE.md
3. [MVP_TO_PLATFORM_CHECKLIST.md](./docs/planning/MVP_TO_PLATFORM_CHECKLIST.md) - Check progress

### For New Developers

**Phase 1: Understanding (2-3 hours)**
1. [README.md](./README.md) - Project overview (15 min)
2. [platform/docs/architecture/OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md) - System architecture (45 min)
3. [platform/docs/architecture/ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md) - Critical principles (30 min)
4. [platform/docs/development/3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md) - Development process (45 min)

**Phase 2: Component Deep-Dive (6-8 hours)**
5. Component-specific architecture docs (1 hour each)
6. Component READMEs (30 min each)

**Phase 3: Implementation (as needed)**
7. Reference specific design docs when implementing features
8. Use conventions docs (MODEL_CAPABILITIES, EXPORT_CONVENTION) for standards
9. Track progress in MVP_TO_PLATFORM_CHECKLIST.md

---

## ❓ Can't Find Something?

1. **Check this map first** (DOCUMENTATION_MAP.md)
2. **Search with VS Code** (Ctrl+Shift+F) across all `.md` files
3. **Check CLAUDE.md** - Authoritative guide with cross-references
4. **Check component READMEs** in `platform/*/README.md`
5. **Ask in the team** if still unclear

---

## 📌 Key Documents by Priority

### Must Read (Everyone)
- ⭐⭐⭐ [CLAUDE.md](./CLAUDE.md)
- ⭐⭐⭐ [platform/docs/architecture/OVERVIEW.md](./platform/docs/architecture/OVERVIEW.md)
- ⚠️ [platform/docs/architecture/ISOLATION_DESIGN.md](./platform/docs/architecture/ISOLATION_DESIGN.md)
- ⭐⭐⭐ [platform/docs/development/3_TIER_DEVELOPMENT.md](./platform/docs/development/3_TIER_DEVELOPMENT.md)

### Essential Reference (By Role)
**Backend Developer**:
- [BACKEND_DESIGN.md](./platform/docs/architecture/BACKEND_DESIGN.md)
- [ERROR_HANDLING_DESIGN.md](./platform/docs/architecture/ERROR_HANDLING_DESIGN.md)
- [PROJECT_MEMBERSHIP_DESIGN.md](./platform/docs/architecture/PROJECT_MEMBERSHIP_DESIGN.md)

**Trainer Developer**:
- [TRAINER_DESIGN.md](./platform/docs/architecture/TRAINER_DESIGN.md)
- [MODEL_CAPABILITIES_SYSTEM.md](./docs/MODEL_CAPABILITIES_SYSTEM.md)
- [EXPORT_CONVENTION.md](./docs/EXPORT_CONVENTION.md)
- [platform/trainers/ultralytics/EXPORT_GUIDE.md](./platform/trainers/ultralytics/EXPORT_GUIDE.md)

**Frontend Developer**:
- [DESIGN_SYSTEM.md](./platform/docs/frontend/DESIGN_SYSTEM.md)
- [platform/frontend/README.md](./platform/frontend/README.md)

**DevOps/Infrastructure**:
- [platform/infrastructure/README.md](./platform/infrastructure/README.md)
- [platform/observability/README.md](./platform/observability/README.md)
- [OPERATIONS_RUNBOOK.md](./platform/docs/architecture/OPERATIONS_RUNBOOK.md)

---

**Remember**:
- CLAUDE.md is the master guide for Claude Code - always check it first
- ISOLATION_DESIGN.md principles are **not optional** - they're the foundation
- 3-tier development is production-ready from day 1 - no shortcuts
- MODEL_CAPABILITIES_SYSTEM and EXPORT_CONVENTION define key conventions - follow them strictly

Happy coding! 🎉

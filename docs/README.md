# Documentation

This directory contains all project documentation organized by category.

## ⚠️ Important: Document Status

**Current Implementation**: MVP Phase with K8s + Railway + SQLite/PostgreSQL

Many documents were written for the full architecture (Temporal, MongoDB, etc.) but MVP uses a simplified stack.

👉 **See [OUTDATED_FILES.md](./OUTDATED_FILES.md)** for a complete list of outdated documents and recommended alternatives.

---

## 📁 Directory Structure

### ⭐ `/251106` - Current Implementation Specs (LATEST)
- **[01_backend_api_specification.md](./251106/01_backend_api_specification.md)** - Current Backend API (MVP)
- **[02_sdk_adapter_pattern.md](./251106/02_sdk_adapter_pattern.md)** - Platform SDK and Adapter pattern
- **[03_config_schema_guide.md](./251106/03_config_schema_guide.md)** - Training configuration schemas
- **[04_user_flow_scenarios.md](./251106/04_user_flow_scenarios.md)** - User interaction flows
- **[05_annotation_system.md](./251106/05_annotation_system.md)** - Annotation system design
- **[06_model_developer_guide.md](./251106/06_model_developer_guide.md)** - Model developer guide

### ⭐ `/k8s` - Kubernetes Training Setup (LATEST)
- **[20251107_development_workflow_setup.md](./k8s/20251107_development_workflow_setup.md)** - 3-tier dev workflow
- **[20251107_kind_vs_minikube_production_continuity.md](./k8s/20251107_kind_vs_minikube_production_continuity.md)** - K8s environment comparison
- **[20251106_kubernetes_job_migration_plan.md](./k8s/20251106_kubernetes_job_migration_plan.md)** - K8s job migration
- **[K8S_TRAINING_FAQ.md](./k8s/K8S_TRAINING_FAQ.md)** - K8s training FAQ

### ⭐ `/production` - Deployment Guides (LATEST)
- **[RAILWAY_SETUP_GUIDE.md](./production/RAILWAY_SETUP_GUIDE.md)** - Railway deployment
- **[FRAMEWORK_ISOLATION_DEPLOYMENT.md](./production/FRAMEWORK_ISOLATION_DEPLOYMENT.md)** - Framework-specific services
- **[DYNAMIC_MODEL_REGISTRATION.md](./production/DYNAMIC_MODEL_REGISTRATION.md)** - Dynamic model registry
- **[CLOUDFLARE_R2_SETUP.md](./production/CLOUDFLARE_R2_SETUP.md)** - R2 storage setup
- **[RAILWAY_MLFLOW_SETUP.md](./production/RAILWAY_MLFLOW_SETUP.md)** - MLflow on Railway

### ⭐ `/training` - Training Architecture (LATEST)
- **[20251105_checkpoint_management_and_r2_upload_policy.md](./training/20251105_checkpoint_management_and_r2_upload_policy.md)** - Checkpoint management
- **[20251105_inference_api_training_service_integration.md](./training/20251105_inference_api_training_service_integration.md)** - Inference API
- **[20251105_r2_pretrained_weights_management.md](./training/20251105_r2_pretrained_weights_management.md)** - Pretrained weights
- **[20251105_timm_implementation_plan.md](./training/20251105_timm_implementation_plan.md)** - TIMM framework
- **[20251105_training_framework_implementation_guide.md](./training/20251105_training_framework_implementation_guide.md)** - Framework implementation guide

### `/architecture` - System Architecture
- **[DATABASE_SCHEMA.md](./architecture/DATABASE_SCHEMA.md)** - Database schemas and models
- **[ADAPTER_DESIGN.md](./architecture/ADAPTER_DESIGN.md)** - Model adapter pattern

### `/planning` - Project Planning
- **[MVP_PLAN.md](./planning/MVP_PLAN.md)** - MVP implementation plan
- **[MVP_STRUCTURE.md](./planning/MVP_STRUCTURE.md)** - MVP folder structure
- **[MVP_DESIGN_GUIDE.md](./planning/MVP_DESIGN_GUIDE.md)** - MVP design decisions
- **[DECISION_LOG.md](./planning/DECISION_LOG.md)** - Architecture decisions
- Plus other planning documents

### `/datasets` - Dataset Management
- **[DATASET_MANAGEMENT_DESIGN.md](./datasets/DATASET_MANAGEMENT_DESIGN.md)** - Dataset management design
- **[IMPLEMENTATION_PLAN.md](./datasets/IMPLEMENTATION_PLAN.md)** - Implementation plan
- Plus other dataset documents

### `/llm` - LLM Integration
- **[LLM_CONTROL_STRATEGY.md](./llm/LLM_CONTROL_STRATEGY.md)** - LLM control strategy
- **[INTENT_MAPPING.md](./llm/INTENT_MAPPING.md)** - Intent mapping
- Plus other LLM documents

### `/scenarios` - User Flow Scenarios
- **[01-login-flow.md](./scenarios/01-login-flow.md)** through **[06-training-execution-flow.md](./scenarios/06-training-execution-flow.md)**
- **[ENVIRONMENT_VARIABLES.md](./scenarios/ENVIRONMENT_VARIABLES.md)** - Environment variables
- **[INFRASTRUCTURE_COMPARISON.md](./scenarios/INFRASTRUCTURE_COMPARISON.md)** - Infrastructure comparison

### `/analysis` - Technical Analysis
- **[ANALYSIS.md](./analysis/ANALYSIS.md)** - Technical analysis
- **[BREAKTHROUGH.md](./analysis/BREAKTHROUGH.md)** - Key breakthrough findings

### `/features` - Feature Specifications
- **[DATASET_SOURCES_DESIGN.md](./features/DATASET_SOURCES_DESIGN.md)** - Dataset source types

### 📦 `/_archived` - Outdated Documentation
**⚠️ Don't follow these - preserved for reference only**
- See [_archived/README.md](./_archived/README.md) for details
- Contains full architecture docs (Temporal, MongoDB, etc.)
- 23 outdated documents moved here (2025-11-07)

---

## 🔍 Quick Reference

### For New Developers (⭐ Recommended for MVP)
1. Start with [GETTING_STARTED.md](../GETTING_STARTED.md) and [DEV_WORKFLOW.md](../DEV_WORKFLOW.md) in the root
2. Read [251106/01_backend_api_specification.md](./251106/01_backend_api_specification.md) for current API
3. Review [k8s/](./k8s/) folder for K8s training setup
4. Check [production/RAILWAY_SETUP_GUIDE.md](./production/RAILWAY_SETUP_GUIDE.md) for deployment

**⚠️ Avoid**:
- ~~[ARCHITECTURE.md](./architecture/ARCHITECTURE.md)~~ (uses Temporal, MongoDB - not in MVP)
- ~~[DEVELOPMENT.md](./development/DEVELOPMENT.md)~~ (outdated setup instructions)
- ~~[API_SPECIFICATION.md](./api/API_SPECIFICATION.md)~~ (full architecture API)

### For Understanding Current Implementation
1. **Backend API**: [251106/01_backend_api_specification.md](./251106/01_backend_api_specification.md)
2. **SDK/Adapter**: [251106/02_sdk_adapter_pattern.md](./251106/02_sdk_adapter_pattern.md)
3. **K8s Training**: [k8s/20251107_development_workflow_setup.md](./k8s/20251107_development_workflow_setup.md)
4. **Railway Deployment**: [production/RAILWAY_SETUP_GUIDE.md](./production/RAILWAY_SETUP_GUIDE.md)

### For MVP Planning
1. [planning/MVP_PLAN.md](./planning/MVP_PLAN.md) - Implementation roadmap
2. [planning/MVP_STRUCTURE.md](./planning/MVP_STRUCTURE.md) - Folder structure
3. [planning/MVP_DESIGN_GUIDE.md](./planning/MVP_DESIGN_GUIDE.md) - Design decisions

### For Contributors
- See [CONTRIBUTING.md](../CONTRIBUTING.md) in the root
- Check [OUTDATED_FILES.md](./OUTDATED_FILES.md) to avoid using outdated docs

---

## 📝 Document Maintenance

- All documents use **Markdown** format
- Keep documents up-to-date with code changes
- Use relative links when referencing other docs
- Include diagrams where helpful (use Mermaid syntax)

---

**Last Updated**: 2025-11-07

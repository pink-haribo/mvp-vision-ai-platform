# Documentation

This directory contains all project documentation organized by category.

## ⚠️ Important: Document Status

**Current Implementation**: MVP Phase with K8s + Railway + SQLite/PostgreSQL

Many documents were written for the full architecture (Temporal, MongoDB, etc.) but MVP uses a simplified stack.

👉 **See [OUTDATED_FILES.md](./OUTDATED_FILES.md)** for a complete list of outdated documents and recommended alternatives.

---

## 📁 Directory Structure

### `/architecture` - System Architecture & Design
- **[ARCHITECTURE.md](./architecture/ARCHITECTURE.md)** - Comprehensive system architecture, component design, and data flows
- **[DATABASE_SCHEMA.md](./architecture/DATABASE_SCHEMA.md)** - Database schemas, models, and migration guides
- **[ADAPTER_DESIGN.md](./architecture/ADAPTER_DESIGN.md)** - Model adapter pattern for multi-framework support
- **[CONVERSATION_STATE_ARCHITECTURE.md](./architecture/CONVERSATION_STATE_ARCHITECTURE.md)** - Conversation state machine architecture

### `/design` - UI/UX Design
- **[DESIGN_SYSTEM.md](./design/DESIGN_SYSTEM.md)** - Design tokens, colors, typography, and visual guidelines
- **[UI_COMPONENTS.md](./design/UI_COMPONENTS.md)** - Component specifications, props, and usage examples

### `/development` - Development Guides
- **[DEVELOPMENT.md](./development/DEVELOPMENT.md)** - Setup instructions, coding conventions, and testing guidelines
- **[PROJECT_SETUP.md](./development/PROJECT_SETUP.md)** - Initial project setup and configuration

### `/api` - API Documentation
- **[API_SPECIFICATION.md](./api/API_SPECIFICATION.md)** - Complete REST API and WebSocket reference

### `/planning` - Project Planning & Strategy
- **[MVP_PLAN.md](./planning/MVP_PLAN.md)** - 2-week MVP implementation plan and roadmap
- **[MVP_STRUCTURE.md](./planning/MVP_STRUCTURE.md)** - Detailed MVP folder structure
- **[MVP_DESIGN_GUIDE.md](./planning/MVP_DESIGN_GUIDE.md)** - MVP design decisions and guidelines
- **[DECISION_LOG.md](./planning/DECISION_LOG.md)** - Architecture and design decision log

### `/features` - Feature Specifications
- **[DATASET_SOURCES_DESIGN.md](./features/DATASET_SOURCES_DESIGN.md)** - Dataset source types, auto-detection, and UI design

### `/analysis` - Technical Analysis & Research
- **[ANALYSIS.md](./analysis/ANALYSIS.md)** - Technical analysis and investigation notes
- **[BREAKTHROUGH.md](./analysis/BREAKTHROUGH.md)** - Key breakthrough findings and solutions
- **[BUG_FIX_SQLALCHEMY_JSON.md](./analysis/BUG_FIX_SQLALCHEMY_JSON.md)** - SQLAlchemy JSON column change detection issue
- **[DEBUG_INFRASTRUCTURE_ISSUE.md](./analysis/DEBUG_INFRASTRUCTURE_ISSUE.md)** - Infrastructure debugging notes

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

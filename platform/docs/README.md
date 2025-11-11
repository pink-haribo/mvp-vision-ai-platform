# Platform Documentation Index

Complete design documentation for the production-ready Vision AI Training Platform.

## Quick Navigation

### Architecture Design
Located in `platform/docs/architecture/`

- **[OVERVIEW.md](./architecture/OVERVIEW.md)** - Complete system architecture, components, and data flows
- **[BACKEND_DESIGN.md](./architecture/BACKEND_DESIGN.md)** - Backend service design (FastAPI + Temporal + PostgreSQL)
- **[TRAINER_DESIGN.md](./architecture/TRAINER_DESIGN.md)** - Training service design (API contract, callback pattern)
- **[WORKFLOWS_DESIGN.md](./architecture/WORKFLOWS_DESIGN.md)** - Temporal workflow orchestration design
- **[FRONTEND_DESIGN.md](./architecture/FRONTEND_DESIGN.md)** - Frontend design (Next.js + WebSocket)
- **[INFRASTRUCTURE_DESIGN.md](./architecture/INFRASTRUCTURE_DESIGN.md)** - Kubernetes + Helm deployment design
- **[OBSERVABILITY_DESIGN.md](./architecture/OBSERVABILITY_DESIGN.md)** - Monitoring, logging, and tracing design
- **[ISOLATION_DESIGN.md](./architecture/ISOLATION_DESIGN.md)** - Complete isolation principles (Backend ↔ Trainer)

### Development Process
Located in `platform/docs/development/`

- **[3_TIER_DEVELOPMENT.md](./development/3_TIER_DEVELOPMENT.md)** - Complete 3-tier development guide
  - Tier 1: Local subprocess mode
  - Tier 2: Local Kind cluster
  - Tier 3: Production Kubernetes
- **[ENVIRONMENT_PARITY.md](./development/ENVIRONMENT_PARITY.md)** - Ensuring consistency across tiers
- **[TESTING_STRATEGY.md](./development/TESTING_STRATEGY.md)** - Testing approach for each tier

### Migration & Planning
Located in `platform/docs/migration/`

- **[MVP_MIGRATION.md](./migration/MVP_MIGRATION.md)** - Complete migration plan from MVP to Platform
  - Component categorization (REUSE/ADAPT/REBUILD)
  - Code mapping table
  - 5-phase migration strategy
  - Timeline and effort estimates

### Design Reviews
Located in `platform/docs/reviews/`

- **[FINAL_DESIGN_REVIEW_2025-01-11.md](./reviews/FINAL_DESIGN_REVIEW_2025-01-11.md)** - Final design review before implementation
  - Architecture verification
  - P0/P1/P2 action items
  - Design decisions and rationale

### Component-Specific READMEs
Located in each component directory:

- **[platform/backend/README.md](../backend/README.md)** - Backend service overview
- **[platform/frontend/README.md](../frontend/README.md)** - Frontend application overview
- **[platform/trainers/README.md](../trainers/README.md)** - Training services overview
- **[platform/workflows/README.md](../workflows/README.md)** - Temporal workflows overview
- **[platform/infrastructure/README.md](../infrastructure/README.md)** - Infrastructure as Code overview
- **[platform/observability/README.md](../observability/README.md)** - Observability stack overview

## Documentation Structure

```
platform/
├── docs/
│   ├── README.md                    # This file
│   ├── architecture/                # System design documents
│   │   ├── OVERVIEW.md
│   │   ├── BACKEND_DESIGN.md
│   │   ├── TRAINER_DESIGN.md
│   │   ├── WORKFLOWS_DESIGN.md
│   │   ├── FRONTEND_DESIGN.md
│   │   ├── INFRASTRUCTURE_DESIGN.md
│   │   ├── OBSERVABILITY_DESIGN.md
│   │   ├── ISOLATION_DESIGN.md
│   │   ├── ERROR_HANDLING_DESIGN.md
│   │   ├── INTEGRATION_FAILURE_HANDLING.md
│   │   └── OPERATIONS_RUNBOOK.md
│   ├── development/                 # Development process guides
│   │   ├── 3_TIER_DEVELOPMENT.md
│   │   ├── ENVIRONMENT_PARITY.md
│   │   └── TESTING_STRATEGY.md
│   ├── migration/                   # Migration planning
│   │   └── MVP_MIGRATION.md
│   └── reviews/                     # Design reviews
│       └── FINAL_DESIGN_REVIEW_2025-01-11.md
├── backend/
│   └── README.md
├── frontend/
│   └── README.md
├── trainers/
│   └── README.md
├── workflows/
│   └── README.md
├── infrastructure/
│   └── README.md
└── observability/
    └── README.md
```

## Reading Order

If you're new to this project, read in this order:

1. **[platform/README.md](../README.md)** - High-level overview of production approach
2. **[architecture/OVERVIEW.md](./architecture/OVERVIEW.md)** - Complete system architecture
3. **[development/3_TIER_DEVELOPMENT.md](./development/3_TIER_DEVELOPMENT.md)** - How development works
4. **[architecture/ISOLATION_DESIGN.md](./architecture/ISOLATION_DESIGN.md)** - Critical isolation principles
5. **[migration/MVP_MIGRATION.md](./migration/MVP_MIGRATION.md)** - Migration from MVP to Platform

Then dive into specific components as needed:
- Backend: [architecture/BACKEND_DESIGN.md](./architecture/BACKEND_DESIGN.md)
- Trainers: [architecture/TRAINER_DESIGN.md](./architecture/TRAINER_DESIGN.md)
- Workflows: [architecture/WORKFLOWS_DESIGN.md](./architecture/WORKFLOWS_DESIGN.md)
- Frontend: [architecture/FRONTEND_DESIGN.md](./architecture/FRONTEND_DESIGN.md)
- Infrastructure: [architecture/INFRASTRUCTURE_DESIGN.md](./architecture/INFRASTRUCTURE_DESIGN.md)
- Observability: [architecture/OBSERVABILITY_DESIGN.md](./architecture/OBSERVABILITY_DESIGN.md)

## Key Principles

All documents in this directory follow these production-first principles:

1. **No Shortcuts** - Implement correctly from the start, no workarounds
2. **Cloud-Agnostic** - Works on Railway, AWS, or on-premise K8s
3. **Complete Isolation** - Backend and Trainers communicate only via APIs
4. **3-Tier Development** - Same code works across subprocess, Kind, and production K8s
5. **Production-Ready** - Every feature works in both local AND production

## External Documentation

For MVP reference and project context:
- MVP codebase: `mvp/` (kept for reference, not actively developed)
- Project overview: `README.md` (root)
- Development guide: `docs/development/DEVELOPMENT.md`
- Architecture (old): `docs/architecture/ARCHITECTURE.md`
- Claude instructions: `CLAUDE.md`

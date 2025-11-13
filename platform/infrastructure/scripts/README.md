# Infrastructure Management Scripts

This directory contains automation scripts for managing the Vision AI Platform infrastructure across different tiers.

## Tier-0 Scripts (Docker Compose)

### start-tier0.ps1

**Automated startup script for Tier-0 environment**

**Usage:**
```powershell
cd platform/infrastructure
.\scripts\start-tier0.ps1
```

**What it does:**
1. Verifies Docker Desktop is running
2. Creates required data directories in C:\platform-data\
3. Starts all 8 infrastructure services via docker-compose:
   - PostgreSQL (5432)
   - Redis (6379)
   - MinIO (9000, 9001)
   - MLflow (5000)
   - Temporal (7233, 8233)
   - Prometheus (9090)
   - Grafana (3200)
   - Loki (3100, 9096)
4. Waits for all services to become healthy (60s timeout per service)
5. Checks .env configuration (copies from .env.tier0 if needed)
6. Starts Backend server as PowerShell background job
7. Starts Frontend server as PowerShell background job
8. Displays service URLs and management commands

**Expected Duration:** 30-90 seconds

**Output:** Service status, access URLs, PowerShell job IDs, resource usage info

**When to use:**
- After PC reboot
- When starting development session
- After stopping Tier-0 with stop-tier0.ps1
- When switching from Tier-1 to Tier-0

### stop-tier0.ps1

**Clean shutdown script for Tier-0 environment**

**Usage:**
```powershell
cd platform/infrastructure
.\scripts\stop-tier0.ps1
```

**What it does:**
1. Stops all PowerShell background jobs (Backend/Frontend servers)
2. Stops all Docker Compose containers
3. Preserves all data in C:\platform-data\
4. Displays data location summary and restart instructions

**Data Preservation:**
- All database data preserved
- All uploaded files preserved
- All metrics and logs preserved
- Can restart with start-tier0.ps1 without data loss

**When to use:**
- End of development session
- Before PC reboot
- Before switching to Tier-1
- To free up system resources

## Tier-1 Scripts (Kubernetes/Kind)

### start-dev-environment.ps1

**Environment check script for Tier-1 (Kind cluster)**

**Usage:**
```powershell
cd platform/infrastructure
.\scripts\start-dev-environment.ps1
```

**What it does:**
1. Verifies Docker Desktop is running
2. Checks Kind cluster 'platform-dev' exists
3. Waits for cluster to be ready
4. Checks pod status in all namespaces (platform, mlflow, observability, temporal)
5. Displays service access URLs (NodePorts)
6. Shows next steps for starting Backend/Frontend

**Note:** This script checks infrastructure but doesn't start services. Use Helm for deployment.

**When to use:**
- After PC reboot (verify cluster still healthy)
- After creating Kind cluster
- To check service status
- Before starting development

## Helm Deployment Scripts

### deploy-helm-all.ps1

**Deploys all infrastructure services to Kind cluster via Helm**

**Usage:**
```powershell
cd platform/infrastructure
.\scripts\deploy-helm-all.ps1
```

**What it does:**
- Deploys PostgreSQL (Bitnami chart)
- Deploys Redis (Bitnami chart)
- Deploys MinIO (Bitnami chart)
- Deploys MLflow (custom chart)
- Deploys Prometheus + Grafana (kube-prometheus-stack)
- Deploys Loki (Grafana chart)
- Deploys Temporal (temporal chart)

**When to use:**
- Initial cluster setup
- After deleting cluster
- To update all charts to latest versions

### cleanup-raw-manifests.ps1

**Removes raw YAML manifests after Helm migration**

**Usage:**
```powershell
cd platform/infrastructure
.\scripts\cleanup-raw-manifests.ps1
```

**What it does:**
- Archives old raw YAML files
- Cleans up deprecated manifest files
- Prepares for Helm-only workflow

**When to use:**
- During migration to Helm
- Cleanup after verifying Helm deployment works

## Script Comparison

| Script | Tier | Purpose | Duration | Services Affected |
|--------|------|---------|----------|-------------------|
| start-tier0.ps1 | 0 | Full startup | 30-90s | All + BE/FE |
| stop-tier0.ps1 | 0 | Clean shutdown | 10-20s | All + BE/FE |
| start-dev-environment.ps1 | 1 | Health check | 10-30s | None (check only) |
| deploy-helm-all.ps1 | 1 | Deploy infra | 2-5min | All (Helm) |

## Common Workflows

### Starting Fresh Development (Tier-0)

```powershell
# 1. Ensure Docker Desktop is running

# 2. Run startup script
cd platform/infrastructure
.\scripts\start-tier0.ps1

# 3. Everything is ready!
# - Backend: http://localhost:8000
# - Frontend: http://localhost:3000
# - All infrastructure services running
```

### Ending Development Session (Tier-0)

```powershell
cd platform/infrastructure
.\scripts\stop-tier0.ps1

# All data preserved for next session
```

### Switching Tier-0 → Tier-1

```powershell
# 1. Stop Tier-0
cd platform/infrastructure
.\scripts\stop-tier0.ps1

# 2. Create Kind cluster (if not exists)
kind create cluster --config kind-config.yaml

# 3. Deploy infrastructure
.\scripts\deploy-helm-all.ps1

# 4. Switch environment config
cd ../backend
cp .env.tier1 .env

# 5. Start Backend/Frontend manually
# All data from C:\platform-data\ is preserved!
```

### Switching Tier-1 → Tier-0

```powershell
# 1. Delete Kind cluster
kind delete cluster --name platform-dev

# 2. Start Tier-0
cd platform/infrastructure
.\scripts\start-tier0.ps1

# All data from C:\platform-data\ is preserved!
```

## Data Persistence

**Both Tier-0 and Tier-1 use shared storage:**

```
C:\platform-data\
├── postgres/       # Shared PostgreSQL data
├── redis/          # Shared Redis data
├── minio/          # Shared object storage
├── prometheus/     # Shared metrics
├── grafana/        # Shared dashboards
└── loki/           # Shared logs
```

**Benefits:**
- No data loss when switching tiers
- Seamless migration between Docker Compose and Kind
- Persistent across PC reboots

## Troubleshooting

### "Docker Desktop is not running"

**Solution:**
1. Start Docker Desktop
2. Wait for it to fully initialize
3. Re-run the script

### "Failed to start Docker Compose"

**Possible causes:**
- Port conflicts (another service using 5432, 6379, etc.)
- Docker Desktop issues
- Corrupt container state

**Solution:**
```powershell
# Reset Docker Compose
docker-compose -f docker-compose.tier0.yaml down -v
docker system prune -f

# Try again
.\scripts\start-tier0.ps1
```

### "Service health check timeout"

**Possible causes:**
- Service taking longer than 60s to start
- Service configuration error
- System resource constraints

**Solution:**
```powershell
# Check specific service logs
docker logs platform-postgres-tier0
docker logs platform-mlflow-tier0

# Restart specific service
docker-compose -f docker-compose.tier0.yaml restart mlflow
```

### PowerShell Jobs Not Stopping

**Solution:**
```powershell
# Force stop all jobs
Get-Job | Stop-Job -Force
Get-Job | Remove-Job -Force
```

## Development Tips

1. **Use Tier-0 for daily development** - Faster, lighter, easier to debug
2. **Test on Tier-1 before production** - Validate Kubernetes compatibility
3. **Keep scripts updated** - If you modify infrastructure, update scripts
4. **Monitor resource usage** - Use `docker stats` to watch container resources
5. **Regular cleanup** - Run `docker system prune` monthly

## Related Documentation

- [TIER0_SETUP.md](../../docs/development/TIER0_SETUP.md) - Complete Tier-0 guide
- [DEVELOPMENT.md](../../docs/development/DEVELOPMENT.md) - General development guide
- [../docker-compose.tier0.yaml](../docker-compose.tier0.yaml) - Tier-0 infrastructure definition

---

**Last Updated:** 2025-11-13

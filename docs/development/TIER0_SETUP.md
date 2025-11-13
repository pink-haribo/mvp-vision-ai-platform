# Tier-0 Setup Guide

**Lightweight Docker Compose Infrastructure for Resource-Constrained Development**

## Overview

Tier-0 is a lightweight infrastructure configuration designed for local development on resource-constrained machines. Instead of using Kind (Kubernetes in Docker), it uses plain Docker Compose to run only the essential infrastructure services.

### When to Use Tier-0

✅ **Use Tier-0 when:**
- Your PC has limited resources (< 16GB RAM)
- You're doing rapid development iteration
- You don't need Kubernetes-specific features
- Your Kind cluster keeps crashing
- You want faster startup times

❌ **Don't use Tier-0 when:**
- Testing Kubernetes-specific features
- Preparing for production deployment
- Working on infrastructure code
- Testing multi-node scenarios

## Resource Comparison

| Tier | Infrastructure | RAM Usage | CPU Usage | Startup Time |
|------|---------------|-----------|-----------|--------------|
| **Tier-0** | Docker Compose | 500MB-1GB | 5-10% | ~20 seconds |
| **Tier-1** | Kind (Kubernetes) | 2-3GB | 20-30% | ~2 minutes |
| **Tier-2** | Cloud K8s (Railway) | N/A | N/A | N/A |

## Quick Start

### Option A: Automated Startup (Recommended)

**Use this after PC reboot or when starting fresh:**

```powershell
# Navigate to infrastructure directory
cd platform/infrastructure

# Run the startup script
.\scripts\start-tier0.ps1
```

The script will automatically:
1. Check Docker Desktop status
2. Create required data directories (C:\platform-data\*)
3. Start all infrastructure services (PostgreSQL, Redis, MinIO, MLflow, Temporal, Prometheus, Grafana, Loki)
4. Wait for services to be healthy
5. Start Backend server (port 8000)
6. Start Frontend server (port 3000)
7. Display all service URLs

**To stop everything:**

```powershell
# From infrastructure directory
.\scripts\stop-tier0.ps1
```

### Option B: Manual Setup

**If you prefer manual control:**

#### 1. Switch to Tier-0

```bash
# Navigate to backend directory
cd platform/backend

# Copy Tier-0 environment config
cp .env.tier0 .env

# Navigate to infrastructure directory
cd ../infrastructure

# Start infrastructure
docker-compose -f docker-compose.tier0.yaml up -d

# Verify all services are running
docker-compose -f docker-compose.tier0.yaml ps
```

#### 2. Start Backend and Frontend

```bash
# Start Backend (from platform/backend)
cd ../backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Start Frontend (from platform/frontend, new terminal)
cd platform/frontend
pnpm dev
```

### 3. Verify Infrastructure

```bash
# Check PostgreSQL
docker exec -it platform-postgres-tier0 psql -U admin -d platform -c "SELECT version();"

# Check Redis
docker exec -it platform-redis-tier0 redis-cli ping

# Check MinIO (open browser)
# MinIO Console: http://localhost:9001
# Username: minioadmin
# Password: minioadmin

# Check MLflow (open browser)
# MLflow UI: http://localhost:5000
```

## Infrastructure Services

### PostgreSQL (Port 5432)

**Main database for application data**

```bash
# Connect to database
docker exec -it platform-postgres-tier0 psql -U admin -d platform

# Backup database
docker exec platform-postgres-tier0 pg_dump -U admin platform > backup.sql

# Restore database
docker exec -i platform-postgres-tier0 psql -U admin platform < backup.sql
```

**Connection String:**
```
postgresql://admin:devpass@localhost:5432/platform
```

### Redis (Port 6379)

**Cache and session store**

```bash
# Connect to Redis
docker exec -it platform-redis-tier0 redis-cli

# Monitor commands
docker exec -it platform-redis-tier0 redis-cli MONITOR

# Check memory usage
docker exec -it platform-redis-tier0 redis-cli INFO memory
```

**Connection String:**
```
redis://localhost:6379/0
```

### MinIO (Ports 9000, 9001)

**S3-compatible object storage**

```bash
# List buckets
docker exec platform-minio-tier0 mc ls local/

# Upload file
docker exec platform-minio-tier0 mc cp /path/to/file local/training-datasets/

# Download file
docker exec platform-minio-tier0 mc cp local/training-datasets/file /path/to/destination
```

**Access:**
- API Endpoint: http://localhost:9000
- Console UI: http://localhost:9001
- Username: minioadmin
- Password: minioadmin

**Buckets:**
- `training-datasets` - User-uploaded datasets
- `training-checkpoints` - Model checkpoints during training
- `training-results` - Final trained models
- `model-weights` - Pretrained model weights
- `config-schemas` - Model configuration schemas
- `vision-platform-dev` - Development files

### MLflow (Port 5000)

**Experiment tracking and model registry**

```bash
# View MLflow logs
docker logs -f platform-mlflow-tier0

# Restart MLflow
docker-compose -f docker-compose.tier0.yaml restart mlflow
```

**Access:**
- MLflow UI: http://localhost:5000
- Tracking URI: `http://localhost:5000`

## Automated Management Scripts

Tier-0 includes PowerShell scripts for automated environment management:

### start-tier0.ps1

**Complete environment startup script**

```powershell
cd platform/infrastructure
.\scripts\start-tier0.ps1
```

**What it does:**
1. Verifies Docker Desktop is running
2. Creates C:\platform-data\ directories if missing
3. Starts all 8 infrastructure services with docker-compose
4. Waits for each service to be healthy (with timeout)
5. Copies .env.tier0 to .env if needed
6. Starts Backend server (uvicorn) in background job
7. Starts Frontend server (pnpm dev) in background job
8. Displays all service URLs and management commands

**Output includes:**
- Service health status
- Access URLs for all services
- PowerShell job IDs for Backend/Frontend
- Management commands for viewing logs
- Resource usage information

### stop-tier0.ps1

**Clean shutdown script**

```powershell
cd platform/infrastructure
.\scripts\stop-tier0.ps1
```

**What it does:**
1. Stops all PowerShell background jobs (Backend/Frontend)
2. Stops all Docker Compose services
3. Preserves all data in C:\platform-data\
4. Displays data location summary

**Data preservation:**
- All data persists after shutdown
- Can restart anytime with start-tier0.ps1
- Data is shared with Tier-1 (Kind)

### Managing Background Jobs

The startup script runs Backend and Frontend as PowerShell jobs. Use these commands:

```powershell
# View all jobs
Get-Job

# View Backend logs
Receive-Job <BackendJobId> -Keep

# View Frontend logs
Receive-Job <FrontendJobId> -Keep

# Stop specific job
Stop-Job <JobId>
Remove-Job <JobId>

# Stop all jobs
Get-Job | Stop-Job
Get-Job | Remove-Job
```

## Common Tasks

### Reset All Data

**⚠️ WARNING: This deletes all data including databases, files, and experiments!**

```bash
cd platform/infrastructure
docker-compose -f docker-compose.tier0.yaml down -v
docker-compose -f docker-compose.tier0.yaml up -d
```

### View Logs

```bash
# All services
docker-compose -f docker-compose.tier0.yaml logs -f

# Specific service
docker-compose -f docker-compose.tier0.yaml logs -f postgres
docker-compose -f docker-compose.tier0.yaml logs -f minio
docker-compose -f docker-compose.tier0.yaml logs -f mlflow
```

### Stop Infrastructure

```bash
# Stop all services (preserves data)
docker-compose -f docker-compose.tier0.yaml stop

# Start again
docker-compose -f docker-compose.tier0.yaml start

# Shutdown and remove containers (preserves data)
docker-compose -f docker-compose.tier0.yaml down

# Shutdown and DELETE ALL DATA
docker-compose -f docker-compose.tier0.yaml down -v
```

### Health Checks

```bash
# Check service health
docker-compose -f docker-compose.tier0.yaml ps

# Manual health checks
curl http://localhost:5432  # PostgreSQL (should fail with postgres error)
docker exec platform-redis-tier0 redis-cli ping  # Should return PONG
curl http://localhost:9000/minio/health/live  # MinIO (should return 200)
curl http://localhost:5000/health  # MLflow (should return 200)
```

## Migrating Between Tiers

### Tier-0 → Tier-1 (Docker Compose → Kind)

```bash
# 1. Stop Tier-0 infrastructure
cd platform/infrastructure
docker-compose -f docker-compose.tier0.yaml down

# 2. Start Kind cluster
kind create cluster --config kind-config.yaml

# 3. Deploy Tier-1 infrastructure with Helm
./setup-kind-infrastructure.sh

# 4. Switch environment config
cd ../backend
cp .env.tier1 .env

# 5. Port-forward MinIO (if needed)
kubectl port-forward -n platform svc/minio 9000:9000

# 6. Restart Backend (code unchanged!)
# Backend automatically reconnects to new infrastructure
```

### Tier-1 → Tier-0 (Kind → Docker Compose)

```bash
# 1. Stop Kind cluster
kind delete cluster

# 2. Start Tier-0 infrastructure
cd platform/infrastructure
docker-compose -f docker-compose.tier0.yaml up -d

# 3. Switch environment config
cd ../backend
cp .env.tier0 .env

# 4. Restart Backend (code unchanged!)
# Backend automatically reconnects to new infrastructure
```

## Data Persistence and Sharing

### Shared Storage Architecture

**Tier-0 and Tier-1 now share the same data storage location!**

All data is stored in `C:\platform-data\` with the following structure:

```
C:\platform-data\
├── postgres/       # PostgreSQL database files
├── redis/          # Redis persistence files
├── minio/          # MinIO object storage
├── prometheus/     # Prometheus metrics data
├── grafana/        # Grafana dashboards and settings
└── loki/           # Loki log storage
```

**Benefits:**
- ✅ **No data loss** when switching between Tier-0 and Tier-1
- ✅ **No manual export/import** required
- ✅ **Seamless tier migration** - just change .env file
- ✅ **Persistent across reboots** - data survives system restarts

### Switching Between Tiers (With Data Persistence)

#### Tier-0 → Tier-1 (Zero Data Loss)

```bash
# 1. Stop Tier-0 infrastructure
cd platform/infrastructure
docker-compose -f docker-compose.tier0.yaml down

# 2. Start Kind cluster (if not already running)
kind create cluster --config kind-config.yaml

# 3. Deploy Tier-1 infrastructure with Helm
./setup-kind-infrastructure.sh

# 4. Switch environment config
cd ../backend
cp .env.tier1 .env

# 5. Restart Backend
# ✅ All your data (DB, files, etc.) is preserved!
```

#### Tier-1 → Tier-0 (Zero Data Loss)

```bash
# 1. Stop Kind cluster
kind delete cluster --name platform-dev

# 2. Start Tier-0 infrastructure
cd platform/infrastructure
.\scripts\start-tier0.ps1

# ✅ All your data (DB, files, etc.) is preserved!
```

### Data Backup (Optional)

Even though data persists, you may want backups:

```bash
# Backup entire data directory
Compress-Archive -Path C:\platform-data -DestinationPath C:\backups\platform-data-backup.zip

# Restore from backup
Expand-Archive -Path C:\backups\platform-data-backup.zip -DestinationPath C:\platform-data -Force
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
# Linux/Mac:
lsof -i :5432
# Windows:
netstat -ano | findstr :5432

# Kill process
kill -9 <PID>          # Linux/Mac
taskkill /F /PID <PID> # Windows
```

### MinIO Buckets Not Created

```bash
# Check minio-setup logs
docker logs platform-minio-setup-tier0

# Manually create buckets
docker exec platform-minio-tier0 mc alias set myminio http://localhost:9000 minioadmin minioadmin
docker exec platform-minio-tier0 mc mb myminio/training-datasets
```

### PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is ready
docker exec platform-postgres-tier0 pg_isready -U admin

# View PostgreSQL logs
docker logs platform-postgres-tier0

# Restart PostgreSQL
docker-compose -f docker-compose.tier0.yaml restart postgres
```

### MLflow Can't Connect to Database

```bash
# Check PostgreSQL from MLflow container
docker exec platform-mlflow-tier0 nc -zv postgres 5432

# Check MLflow logs
docker logs platform-mlflow-tier0

# Restart MLflow
docker-compose -f docker-compose.tier0.yaml restart mlflow
```

### Disk Space Issues

```bash
# Check Docker disk usage
docker system df

# Clean up unused images and volumes
docker system prune -a

# Remove specific volumes
docker volume rm platform-postgres-tier0
docker volume rm platform-minio-tier0
```

## Performance Tuning

### Reduce RAM Usage

Edit `docker-compose.tier0.yaml`:

```yaml
services:
  redis:
    command: redis-server --appendonly yes --maxmemory 128mb  # Default: 256mb

  postgres:
    environment:
      # Add these to reduce PostgreSQL memory
      POSTGRES_SHARED_BUFFERS: "128MB"  # Default: 25% of RAM
      POSTGRES_EFFECTIVE_CACHE_SIZE: "256MB"
```

### Faster Startup

```bash
# Start only essential services
docker-compose -f docker-compose.tier0.yaml up -d postgres redis minio

# Skip MLflow if not needed
# (comment out mlflow service in docker-compose.tier0.yaml)
```

## Best Practices

### Development Workflow

1. **Keep infrastructure running** - Don't stop Docker Compose between coding sessions
2. **Use .env files** - Never hardcode connection strings
3. **Regular backups** - Export data before major changes
4. **Monitor resources** - Use `docker stats` to watch resource usage
5. **Clean up** - Run `docker system prune` monthly

### Code Compatibility

**✅ DO:**
- Use environment variables for all infrastructure URLs
- Write code that works on both Tier-0 and Tier-1
- Test on Tier-0, validate on Tier-1 before production

**❌ DON'T:**
- Hardcode localhost:5432 or localhost:30543
- Use Kubernetes-specific features in application code
- Skip testing on Tier-1 before deploying to production

## FAQ

### Q: Can I use both Tier-0 and Tier-1 at the same time?

**A:** No, they use the same ports. You must switch between them.

### Q: Will my data be lost when switching tiers?

**A:** No! As of the latest version, both Tier-0 and Tier-1 share the same data storage location (`C:\platform-data\`). Your database, uploaded files, and configurations are preserved when switching between tiers.

### Q: Is Tier-0 production-ready?

**A:** No, Tier-0 is for local development only. Always use Tier-1 (Kind) for staging and Tier-2 (Cloud K8s) for production.

### Q: What's faster for development?

**A:** Tier-0 is faster for:
- Startup time
- Resource usage
- Iteration speed

Tier-1 is better for:
- Production-like environment
- Testing Kubernetes features
- CI/CD pipeline testing

### Q: Can I deploy to production from Tier-0?

**A:** Never! Always test on Tier-1 (Kind) before deploying to Tier-2 (Cloud K8s).

## Related Documentation

- [DEVELOPMENT.md](./DEVELOPMENT.md) - General development guide
- [PROJECT_SETUP.md](./PROJECT_SETUP.md) - Initial project setup
- [../architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - System architecture
- [../infrastructure/KUBERNETES.md](../infrastructure/KUBERNETES.md) - Kubernetes deployment

## Support

If you encounter issues with Tier-0:

1. Check the [Troubleshooting](#troubleshooting) section
2. View service logs: `docker-compose -f docker-compose.tier0.yaml logs -f`
3. Verify environment variables: `cat .env`
4. Try a fresh start: `docker-compose -f docker-compose.tier0.yaml down -v && docker-compose -f docker-compose.tier0.yaml up -d`

---

**Last Updated:** 2025-11-13
**Tier-0 Version:** 1.0.0

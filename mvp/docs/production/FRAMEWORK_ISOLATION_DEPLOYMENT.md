# Framework-Isolated Training Services Deployment Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      Railway Platform                         │
├────────────────┬──────────────┬──────────────┬───────────────┤
│ Backend        │ timm         │ ultralytics  │ huggingface   │
│ Service        │ Service      │ Service      │ Service       │
│                │              │              │               │
│ - FastAPI      │ - PyTorch    │ - PyTorch    │ - PyTorch     │
│ - PostgreSQL   │ - timm       │ - ultralytics│ - transformers│
│ - MLflow       │ - ResNet     │ - YOLO       │ - ViT, CLIP   │
│ - Routing      │ - EfficientNet│ - Segmentation│ - VLMs      │
│                │              │ - Pose       │               │
└────────┬───────┴──────────────┴──────────────┴───────────────┘
         │           ▲               ▲               ▲
         └───────────┴───────────────┴───────────────┘
                     HTTP API Calls
```

## Benefits

### Dependency Isolation
- ✅ **Zero conflicts**: Each framework has its own dependencies
- ✅ **Independent updates**: Update timm without affecting YOLO
- ✅ **Smaller images**: ~800MB each vs 2-3GB combined

### Operational Excellence
- ✅ **Independent scaling**: Scale YOLO service separately if needed
- ✅ **Faster deployments**: Only redeploy changed framework
- ✅ **Better debugging**: Isolated logs per framework
- ✅ **Cost optimization**: Only run services you need

### Production Readiness
- ✅ **Framework versioning**: Lock versions per service
- ✅ **A/B testing**: Test new framework versions independently
- ✅ **Rollback safety**: Roll back one service without affecting others

## Deployment Steps

### Step 1: Deploy Backend Service (Already Done)

Backend service routes requests to framework-specific services.

**Environment Variables:**
```bash
TRAINING_EXECUTION_MODE=api
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

### Step 2: Deploy timm Service

1. **Create New Service in Railway**
   - Service Name: `timm-service`
   - Select GitHub Repo: `mvp-vision-ai-platform`

2. **Configure Service**
   - Root Directory: `mvp/training`
   - Dockerfile Path: `Dockerfile.timm`

3. **Deploy**
   - Wait for build (5-7 minutes - PyTorch + timm)
   - Copy generated URL

4. **Update Backend**
   - Add `TIMM_SERVICE_URL` to Backend service environment variables

### Step 3: Deploy ultralytics Service

1. **Create New Service in Railway**
   - Service Name: `ultralytics-service`
   - Select GitHub Repo: `mvp-vision-ai-platform`

2. **Configure Service**
   - Root Directory: `mvp/training`
   - Dockerfile Path: `Dockerfile.ultralytics`

3. **Deploy**
   - Wait for build (5-7 minutes - PyTorch + ultralytics)
   - Copy generated URL

4. **Update Backend**
   - Add `ULTRALYTICS_SERVICE_URL` to Backend service environment variables

### Step 4: Deploy huggingface Service (Optional)

Follow same steps as above with:
- Service Name: `huggingface-service`
- Dockerfile Path: `Dockerfile.huggingface`
- Environment Variable: `HUGGINGFACE_SERVICE_URL`

## Testing

### Test timm Service (ResNet)
```
Frontend: "/app/datasets/cls-imagenet-10으로 ResNet-50 학습해줘"
```

Backend should route to `TIMM_SERVICE_URL`.

### Test ultralytics Service (YOLO)
```
Frontend: "/app/datasets/det-coco8로 YOLO 모델 학습해줘"
```

Backend should route to `ULTRALYTICS_SERVICE_URL`.

## Cost Optimization

### Development/Testing
Deploy only the framework you're actively using:
- Testing timm? → Deploy only timm-service
- Testing YOLO? → Deploy only ultralytics-service

### Production
Deploy all frameworks for full functionality.

**Railway Costs (Estimate):**
- Backend: $5-10/month (always running)
- Each training service: $5-10/month (can be scaled to zero when not in use)

## Monitoring

### Health Checks

Check service health:
```bash
curl https://timm-service-production-xxxx.up.railway.app/health
curl https://ultralytics-service-production-xxxx.up.railway.app/health
curl https://huggingface-service-production-xxxx.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "training-service"
}
```

### Logs

Each service has independent logs in Railway dashboard:
- Backend logs: Routing decisions, API calls
- timm logs: Training execution, model metrics
- ultralytics logs: YOLO-specific output
- huggingface logs: Transformer training

## Rollback Procedure

If a service deployment fails:

1. **Rollback in Railway**
   - Go to failed service → Deployments tab
   - Click on previous working deployment
   - Click "Redeploy"

2. **No Impact on Other Services**
   - Backend continues running
   - Other training services unaffected
   - Users can still use other frameworks

## Troubleshooting

### Service Not Starting

**Symptom:** Health check fails repeatedly

**Check:**
1. Railway logs for build errors
2. Memory limits (PyTorch needs ~512MB minimum)
3. Dockerfile path configuration

### Backend Can't Reach Training Service

**Symptom:** "Training Service is not healthy" error

**Check:**
1. Environment variable set correctly (`TIMM_SERVICE_URL`, etc.)
2. Training service is deployed and running
3. Network connectivity (Railway internal networking)

### Wrong Service Handling Request

**Symptom:** timm job goes to ultralytics service

**Check:**
1. Backend routing logic in `training_client.py`
2. Framework detection in `training_manager.py`
3. Environment variables match framework names

## Migration from Single Training Service

If you previously deployed a single "training-service":

1. **Deploy New Framework Services**
   - Deploy timm, ultralytics, huggingface as described above

2. **Update Backend Environment Variables**
   - Add `TIMM_SERVICE_URL`, `ULTRALYTICS_SERVICE_URL`, etc.
   - Keep `TRAINING_SERVICE_URL` as fallback

3. **Test Each Framework**
   - Verify routing works correctly

4. **Delete Old Training Service**
   - Once all frameworks work, delete the old combined service
   - Remove `TRAINING_SERVICE_URL` from backend

## Future Enhancements

### Auto-Scaling
Configure Railway auto-scaling based on training demand.

### Service Mesh
Add API Gateway (Kong, Envoy) for advanced routing, rate limiting, and authentication.

### Queue-Based Architecture
Replace synchronous HTTP calls with message queue (RabbitMQ, Redis) for better reliability.

### Distributed Training
Run multiple training service replicas for parallel job execution.

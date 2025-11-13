# Inference API Training Service Integration

**Date**: 2025-11-05
**Status**: In Progress (Implementation 90% complete, image transfer pending)
**Related Issues**: Checkpoint management, production/local environment parity

## Overview

Refactored inference API to use Training Service HTTP endpoints instead of direct subprocess calls, maintaining architectural consistency with training workflow and ensuring production/local environment parity.

## Background / Context

### Problem Discovery

While testing pretrained weight inference, we discovered that the inference API was calling `run_quick_inference.py` directly via subprocess from the Backend, bypassing the Training Service architecture:

```
❌ Previous Flow (Incorrect):
프론트엔드 → Backend API → subprocess → run_quick_inference.py → Adapter
```

This violated our core architectural principle:

> **"로컬과 Railway 배포 환경을 일치화 시키는것에 집중해서 구현을 계속해오고 있어. Backend에서 training service에 속하는 무언가를 import 하거나 direct call 하는건 배포 환경에서 동작하지 않을거야."**

### Why This Matters

**Railway 배포 환경**:
- Backend: Separate container/dyno
- Training Service (Ultralytics): Separate container/dyno
- Training Service (Timm): Separate container/dyno

Backend cannot:
- Import Training Service modules (different containers)
- Execute subprocess in Training Service context
- Access Training Service filesystem directly

**Correct Architecture** (already implemented for training):
```
✅ Training Flow:
Backend API → HTTP → Training Service API → subprocess → train.py → Adapter
```

**Required Change** (for inference):
```
✅ Inference Flow:
Backend API → HTTP → Training Service API → subprocess → run_quick_inference.py → Adapter
```

## Current State

### Before Refactoring

**Backend** (`mvp/backend/app/api/test_inference.py:758-857`):
```python
@router.post("/inference/quick")
async def quick_inference(...):
    # ❌ Direct subprocess call
    training_venv_python = project_root / "training" / "venv" / "Scripts" / "python.exe"
    inference_script = project_root / "training" / "run_quick_inference.py"

    cmd = [str(training_venv_python), str(inference_script), ...]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)
```

**Training Service** (`mvp/training/api_server.py`):
- ❌ No inference endpoint
- Only had: `/training/start`, `/training/status`, `/models/list`

### Architecture Inconsistency

| Operation | Backend → Training Service | Implementation |
|-----------|---------------------------|----------------|
| Training  | ✅ HTTP API               | Correct        |
| Inference | ❌ Direct subprocess      | **Inconsistent** |

## Proposed Solution / Decision

Implement `/inference/quick` endpoint in Training Service API, mirroring the training workflow architecture.

### Key Design Choices

#### 1. **HTTP API over Direct Subprocess**

**Rationale**:
- Production containers cannot share filesystem
- HTTP API provides clear service boundaries
- Same pattern as training workflow
- Environment-agnostic (local/Railway identical)

**Trade-offs**:
- Additional HTTP latency (~10-50ms)
- More complex error handling
- BUT: Architectural consistency >>> minor latency

#### 2. **Framework-Specific Service Routing**

**Rationale**:
- Different frameworks have different dependencies
- Ultralytics: YOLO models, PyTorch
- Timm: Image classification, PyTorch
- Prevents dependency conflicts

**Implementation**:
```python
# Backend uses TrainingServiceClient with framework routing
client = TrainingServiceClient(framework=job.framework)
# Automatically routes to:
# - ULTRALYTICS_SERVICE_URL (port 8002)
# - TIMM_SERVICE_URL (port 8001)
```

**Trade-offs**:
- Multiple service instances required
- BUT: Isolation prevents dependency hell

#### 3. **Subprocess Still Used Within Training Service**

**Rationale**:
- Training Service needs framework-specific venv (`venv-ultralytics`, `venv-timm`)
- Subprocess isolates Python environments
- Backend doesn't need torch/ML dependencies

**Implementation**:
```python
# Training Service api_server.py
venv_python = f"venv-{request.framework}/Scripts/python.exe"
cmd = [venv_python, "run_quick_inference.py", ...]
result = subprocess.run(cmd, capture_output=True)
```

## Implementation Plan

### Phase 1: Training Service Endpoint ✅ COMPLETED

- [x] Add `InferenceRequest` Pydantic schema
- [x] Implement `/inference/quick` POST endpoint
- [x] Framework-specific venv selection logic
- [x] JSON result parsing and error handling

**Code**: `mvp/training/api_server.py:69-440`

### Phase 2: Backend Refactoring ✅ COMPLETED

- [x] Replace subprocess with HTTP client
- [x] Use `TrainingServiceClient` for framework routing
- [x] Convert checkpoint paths (Docker → host)
- [x] Update error handling for HTTP responses

**Code**: `mvp/backend/app/api/test_inference.py:758-921`

### Phase 3: Image Transfer ⏳ IN PROGRESS

**Current blocker**: Image file transfer between services

- [ ] Decide on file transfer method (Multipart/Base64/R2)
- [ ] Implement chosen solution
- [ ] Test batch inference
- [ ] Update documentation

## Technical Details

### Request/Response Flow

**1. Frontend → Backend**:
```http
POST /api/v1/test_inference/inference/quick
Content-Type: application/json

{
  "training_job_id": 20,
  "image_path": "/uploads/temp/abc123.jpg",
  "confidence_threshold": 0.25
}
```

**2. Backend → Training Service**:
```python
# Backend constructs request
client = TrainingServiceClient(framework="ultralytics")
inference_request = {
    "training_job_id": 20,
    "image_path": image_path,  # ⚠️ File transfer issue
    "framework": "ultralytics",
    "model_name": "yolo11n",
    "task_type": "object_detection",
    "num_classes": 80,
    "confidence_threshold": 0.25,
    ...
}

# HTTP POST
response = requests.post(
    f"{client.base_url}/inference/quick",  # http://localhost:8002
    json=inference_request,
    timeout=300
)
```

**3. Training Service → Subprocess**:
```python
# Training Service api_server.py
venv_python = "venv-ultralytics/Scripts/python.exe"
cmd = [
    venv_python,
    "run_quick_inference.py",
    "--training_job_id", "20",
    "--image_path", image_path,
    "--framework", "ultralytics",
    ...
]

result = subprocess.run(cmd, capture_output=True, text=True)
inference_result = json.loads(result.stdout)
return inference_result
```

**4. Training Service → Backend → Frontend**:
```json
{
  "image_path": "...",
  "task_type": "object_detection",
  "predicted_boxes": [
    {
      "class_id": 23,
      "label": "giraffe",
      "confidence": 0.8775,
      "bbox": [487.98, 211.08, 227.69, 294.23]
    }
  ],
  "inference_time_ms": 202.42
}
```

### Data Structures

**InferenceRequest Schema**:
```python
class InferenceRequest(BaseModel):
    training_job_id: int
    image_path: str  # ⚠️ Path vs File transfer
    framework: str
    model_name: str
    task_type: str
    num_classes: int
    dataset_path: str
    output_dir: str
    checkpoint_path: Optional[str] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    top_k: int = 5
```

### Environment Variables

**Required Configuration**:
```bash
# .env (Local)
ULTRALYTICS_SERVICE_URL=http://localhost:8002
TIMM_SERVICE_URL=http://localhost:8001
HUGGINGFACE_SERVICE_URL=http://localhost:8003
TRAINING_SERVICE_URL=http://localhost:8001  # Fallback

# Railway (Production)
ULTRALYTICS_SERVICE_URL=https://training-ultralytics-prod.up.railway.app
TIMM_SERVICE_URL=https://training-timm-prod.up.railway.app
```

### Code Examples

**Backend Client Usage**:
```python
from app.utils.training_client import TrainingServiceClient

# Framework-specific routing
client = TrainingServiceClient(framework="ultralytics")
# → Uses ULTRALYTICS_SERVICE_URL

client = TrainingServiceClient(framework="timm")
# → Uses TIMM_SERVICE_URL

# Fallback for unknown frameworks
client = TrainingServiceClient(framework=None)
# → Uses TRAINING_SERVICE_URL
```

**Training Service Endpoint**:
```python
@app.post("/inference/quick")
async def quick_inference(request: InferenceRequest):
    # Determine venv based on framework
    venv_python = f"venv-{request.framework}/Scripts/python.exe"

    if not os.path.exists(venv_python):
        venv_python = "python"  # Fallback to system python

    # Build subprocess command
    cmd = [
        venv_python,
        "run_quick_inference.py",
        "--training_job_id", str(request.training_job_id),
        "--image_path", request.image_path,
        ...
    ]

    # Execute with timeout
    result = subprocess.run(cmd, capture_output=True, timeout=300)

    # Parse and return JSON
    return json.loads(result.stdout)
```

## Alternatives Considered

### Alternative 1: Keep Direct Subprocess

**Pros**:
- Simpler implementation
- Lower latency (no HTTP overhead)
- Works in local development

**Cons**:
- ❌ **Fails in production** (containers can't exec each other's code)
- ❌ Architectural inconsistency with training flow
- ❌ Violates separation of concerns

**Why Rejected**: Incompatible with Railway deployment architecture

### Alternative 2: Shared Docker Volume

**Pros**:
- Direct filesystem access
- No code changes

**Cons**:
- ❌ Railway doesn't support shared volumes between dynos
- ❌ Not scalable (single point of failure)
- ❌ Still requires subprocess coordination

**Why Rejected**: Platform limitation (Railway)

### Alternative 3: Message Queue (Celery/RabbitMQ)

**Pros**:
- Async processing
- Better for batch jobs
- Industry standard

**Cons**:
- ❌ Overkill for synchronous inference
- ❌ Additional infrastructure (Redis/RabbitMQ)
- ❌ Increased complexity

**Why Rejected**: Synchronous inference doesn't need queue

## Final Design Decisions

### ✅ Image Transfer: Multipart + BytesIO

**Decision**: Use Multipart/Form-Data for image upload, supporting both file-based and memory-based (BytesIO) image data.

**Rationale**:
1. RESTful standard
2. Supports in-memory images (camera streams, PIL/OpenCV objects)
3. No encoding overhead
4. Works identically in local/production

### ✅ Result Images: Base64 Encoding

**Decision**: Encode result images (masks, overlays, upscaled images) as Base64 and include in JSON response.

**Rationale**:
1. Single HTTP response (no additional requests)
2. Simple implementation
3. Frontend can display immediately
4. Acceptable for MVP (typical mask size: 100-500KB → 133-666KB encoded)

## Complete Data Flow

### End-to-End Flow Diagram

```
┌─────────────┐                    ┌──────────────────┐                    ┌──────────────────┐
│  Frontend   │                    │     Backend      │                    │ Training Service │
└──────┬──────┘                    └────────┬─────────┘                    └────────┬─────────┘
       │                                    │                                       │
       │ 1. Upload image                    │                                       │
       │────────────────────────────────────>│                                       │
       │    (file or camera stream)          │                                       │
       │                                    │                                       │
       │                                    │ 2. Multipart POST                     │
       │                                    │   /inference/quick                    │
       │                                    │   - file: image bytes                 │
       │                                    │   - training_job_id: 20               │
       │                                    │   - confidence_threshold: 0.25        │
       │                                    │──────────────────────────────────────>│
       │                                    │                                       │
       │                                    │                                       │ 3. Save temp file
       │                                    │                                       │    /tmp/uuid.jpg
       │                                    │                                       │
       │                                    │                                       │ 4. subprocess
       │                                    │                                       │    run_quick_inference.py
       │                                    │                                       │
       │                                    │                                       │ 5. Adapter.infer_batch()
       │                                    │                                       │    → predictions
       │                                    │                                       │
       │                                    │                                       │ 6. Generate result images
       │                                    │                                       │    (masks, overlays)
       │                                    │                                       │
       │                                    │                                       │ 7. Base64 encode images
       │                                    │                                       │
       │                                    │                                       │ 8. Delete temp files
       │                                    │                                       │
       │                                    │ 9. JSON Response                      │
       │                                    │   {                                   │
       │                                    │     "predicted_boxes": [...],         │
       │                                    │     "predicted_mask_base64": "iVB..." │
       │                                    │   }                                   │
       │                                    │<──────────────────────────────────────│
       │                                    │                                       │
       │ 10. JSON Response                  │                                       │
       │<────────────────────────────────────│                                       │
       │    (same as Training Service)      │                                       │
       │                                    │                                       │
       │ 11. Display results                │                                       │
       │    <img src="data:image/png;       │                                       │
       │         base64,{mask_base64}">     │                                       │
       │                                    │                                       │
```

### Image Upload Implementation

#### Use Case 1: File-Based Upload (Disk)

```python
# Backend: mvp/backend/app/api/test_inference.py
from pathlib import Path

# Local file path
image_path = "/app/uploads/temp/abc123.jpg"

# Send as multipart
files = {'file': open(image_path, 'rb')}
data = {
    'training_job_id': job_id,
    'framework': 'ultralytics',
    'model_name': 'yolo11n',
    'task_type': 'object_detection',
    'confidence_threshold': 0.25,
}

response = requests.post(
    f"{client.base_url}/inference/quick",
    files=files,
    data=data
)
```

#### Use Case 2: Memory-Based Upload (BytesIO)

```python
from io import BytesIO
import requests

# Case A: bytes from another API
image_bytes = requests.get('https://api.example.com/image').content

# Case B: PIL Image
from PIL import Image
pil_image = Image.open(some_source)
buffer = BytesIO()
pil_image.save(buffer, format='JPEG')
buffer.seek(0)
image_bytes = buffer.getvalue()

# Case C: OpenCV (numpy array)
import cv2
cv_image = cv2.imread('image.jpg')
success, encoded = cv2.imencode('.jpg', cv_image)
image_bytes = encoded.tobytes()

# Send as multipart (no disk I/O!)
files = {
    'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')
}
data = {...}

response = requests.post(url, files=files, data=data)
```

#### Use Case 3: Real-Time Camera Stream

```python
import cv2
from io import BytesIO

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame to JPEG (in-memory)
    success, encoded = cv2.imencode('.jpg', frame)
    image_bytes = encoded.tobytes()

    # Inference request (no file save!)
    files = {'file': ('frame.jpg', BytesIO(image_bytes), 'image/jpeg')}
    data = {'training_job_id': 20, 'confidence_threshold': 0.5}

    response = requests.post(url, files=files, data=data, timeout=5)
    result = response.json()

    # Draw bounding boxes on frame
    for box in result.get('predicted_boxes', []):
        # ... draw logic
        pass
```

### Result Handling Implementation

#### Training Service Response Structure

```json
{
  "task_type": "instance_segmentation",
  "image_name": "frame.jpg",
  "inference_time_ms": 245.8,

  // Object Detection results
  "predicted_boxes": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.92,
      "bbox": [x, y, w, h],
      "x1": 100, "y1": 50,
      "x2": 300, "y2": 400
    }
  ],
  "num_detections": 1,

  // Segmentation results (Base64 encoded!)
  "predicted_mask_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
  "overlay_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
}
```

#### Frontend Display

```tsx
// React component
interface InferenceResult {
  predicted_boxes: Box[];
  predicted_mask_base64?: string;
  overlay_base64?: string;
  upscaled_image_base64?: string;
}

function InferencePanel({ result }: { result: InferenceResult }) {
  return (
    <div>
      {/* Original image with bounding boxes */}
      <img src={originalImageUrl} alt="Original" />

      {/* Segmentation mask */}
      {result.predicted_mask_base64 && (
        <img
          src={`data:image/png;base64,${result.predicted_mask_base64}`}
          alt="Segmentation mask"
        />
      )}

      {/* Overlay (boxes + mask combined) */}
      {result.overlay_base64 && (
        <img
          src={`data:image/png;base64,${result.overlay_base64}`}
          alt="Overlay"
        />
      )}
    </div>
  );
}
```

### Implementation Code

#### Training Service: `mvp/training/api_server.py`

```python
from fastapi import File, UploadFile, Form
import uuid
import os
import base64

@app.post("/inference/quick")
async def quick_inference(
    file: UploadFile = File(...),
    training_job_id: int = Form(...),
    framework: str = Form(...),
    model_name: str = Form(...),
    task_type: str = Form(...),
    num_classes: int = Form(...),
    dataset_path: str = Form(""),
    output_dir: str = Form(""),
    checkpoint_path: str = Form(None),
    confidence_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    max_detections: int = Form(100),
    top_k: int = Form(5)
):
    """
    Run quick inference on uploaded image.

    Returns JSON with Base64-encoded result images.
    """
    import subprocess
    import json

    # Save uploaded file to temp location
    temp_image_path = f"/tmp/inference_{uuid.uuid4()}.jpg"

    try:
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # Determine Python interpreter
        training_dir = os.path.dirname(__file__)
        venv_python = os.path.join(training_dir, f"venv-{framework}", "Scripts", "python.exe")

        if not os.path.exists(venv_python):
            venv_python = "python"

        inference_script = os.path.join(training_dir, "run_quick_inference.py")

        # Build command
        cmd = [
            venv_python,
            inference_script,
            "--training_job_id", str(training_job_id),
            "--image_path", temp_image_path,
            "--framework", framework,
            "--model_name", model_name,
            "--task_type", task_type,
            "--num_classes", str(num_classes),
            "--dataset_path", dataset_path,
            "--output_dir", output_dir,
            "--confidence_threshold", str(confidence_threshold),
            "--iou_threshold", str(iou_threshold),
            "--max_detections", str(max_detections),
            "--top_k", str(top_k)
        ]

        if checkpoint_path:
            cmd.extend(["--checkpoint_path", checkpoint_path])

        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            raise HTTPException(500, f"Inference failed: {result.stderr}")

        # Parse JSON result
        inference_result = json.loads(result.stdout)

        # Encode result images as Base64
        result_image_fields = [
            'predicted_mask_path',
            'overlay_path',
            'upscaled_image_path'
        ]

        for field in result_image_fields:
            path_key = field
            base64_key = field.replace('_path', '_base64')

            if inference_result.get(path_key):
                image_path = inference_result[path_key]

                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    # Add Base64 to result
                    inference_result[base64_key] = img_base64

                    # Remove file path
                    del inference_result[path_key]

                    # Delete temp file
                    os.remove(image_path)

        return inference_result

    finally:
        # Cleanup temp input image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
```

#### Backend: `mvp/backend/app/api/test_inference.py`

```python
from io import BytesIO

@router.post("/inference/quick")
async def quick_inference(
    training_job_id: int,
    image_path: str,
    checkpoint_path: Optional[str] = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    top_k: int = 5,
    db: Session = Depends(get_db)
):
    """Quick inference via Training Service API."""

    # Get training job
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == training_job_id
    ).first()

    if not job:
        raise HTTPException(404, f"Job {training_job_id} not found")

    # Initialize Training Service client
    client = TrainingServiceClient(framework=job.framework)

    try:
        # Read image file
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Prepare multipart request
        files = {
            'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')
        }

        data = {
            'training_job_id': training_job_id,
            'framework': job.framework,
            'model_name': job.model_name,
            'task_type': job.task_type,
            'num_classes': job.num_classes or 0,
            'dataset_path': job.dataset_path or "",
            'output_dir': job.output_dir or "",
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'max_detections': max_detections,
            'top_k': top_k
        }

        if checkpoint_path:
            data['checkpoint_path'] = checkpoint_path

        # Call Training Service
        response = requests.post(
            f"{client.base_url}/inference/quick",
            files=files,
            data=data,
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        # Result already has Base64-encoded images
        # Frontend can display directly
        return result

    except requests.exceptions.Timeout:
        raise HTTPException(504, "Inference timeout")
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('detail', str(e))
        raise HTTPException(500, f"Inference failed: {error_detail}")
```

## Migration Path

### Step 1: Implement Multipart File Transfer ⏳

1. Update Training Service endpoint signature:
   ```python
   @app.post("/inference/quick")
   async def quick_inference(
       file: UploadFile = File(...),
       training_job_id: int = Form(...),
       # ... other form fields
   )
   ```

2. Update Backend client:
   ```python
   files = {'file': open(image_path, 'rb')}
   data = {field: value for field, value in inference_request.items()}
   response = requests.post(url, files=files, data=data)
   ```

3. Test locally with both frameworks (ultralytics, timm)

4. Deploy to Railway and verify

### Step 2: Add Batch Inference Endpoint (Future)

```python
@app.post("/inference/batch")
async def batch_inference(
    files: List[UploadFile] = File(...),
    training_job_id: int = Form(...),
    ...
):
    results = []
    for file in files:
        result = await quick_inference_internal(file, ...)
        results.append(result)
    return {"results": results}
```

### Step 3: Update Documentation

- Update API specs with multipart examples
- Add Railway deployment guide for multiple Training Services
- Document environment variable configuration

## References

### Related Files

**Training Service**:
- `mvp/training/api_server.py:69-440` - Inference endpoint implementation
- `mvp/training/run_quick_inference.py` - Subprocess script (unchanged)

**Backend**:
- `mvp/backend/app/api/test_inference.py:758-921` - Refactored client
- `mvp/backend/app/utils/training_client.py:15-78` - Framework routing

**Configuration**:
- `.env` - Service URL configuration
- `CLAUDE.md` - Architecture principles

### Related Docs

- [Checkpoint Management and R2 Upload Policy](./20251105_checkpoint_management_and_r2_upload_policy.md)
- [YOLO Validation Metrics Issue](../issues/yolo_validation_metrics.md)

### External Resources

- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Railway Multi-Service Deployment](https://docs.railway.app/reference/multi-service)
- [Python Requests Multipart](https://requests.readthedocs.io/en/latest/user/quickstart/#post-a-multipart-encoded-file)

## Notes

### Architecture Principles Reinforced

This refactoring reinforces our core architectural principle:

> **Backend and Training Service must communicate via HTTP API only.**

**No exceptions for**:
- Direct imports
- Subprocess calls
- Filesystem sharing

This ensures:
- ✅ Local development = Production deployment
- ✅ Clean service boundaries
- ✅ Scalability (multiple Training Service instances)
- ✅ Framework isolation (no dependency conflicts)

### Port Allocation

**Corrected Assignments**:
- **8001**: Timm Training Service
- **8002**: Ultralytics Training Service
- **8003**: HuggingFace Training Service (future)
- **8000**: Backend API

### Next Steps (After Image Transfer)

1. Implement batch inference endpoint
2. Add inference result caching (Redis)
3. Performance benchmarking (subprocess vs API overhead)
4. Load testing with concurrent requests
5. Add inference metrics to Prometheus/Grafana

### Open Questions

1. Should we cache model loading in Training Service? (Currently loads per request)
2. Should we implement request queue for heavy load? (Celery)
3. How to handle very large images? (> 10MB)
4. Should we add streaming responses for batch inference?

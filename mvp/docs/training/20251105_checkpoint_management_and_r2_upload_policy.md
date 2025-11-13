# Checkpoint Management and R2 Upload Policy

**Date**: 2025-11-05 14:30
**Status**: Proposed
**Related Issues**: Training inference requirements, storage cost optimization

## Overview

This document defines the checkpoint management policy for training jobs, including local storage strategy, R2 upload timing, and database tracking. The policy ensures checkpoints are available for inference while optimizing storage costs and handling training interruptions gracefully.

## Background / Context

### Initial Problem

During discussion of inference testing preparation, we discovered that:
1. Checkpoints are saved locally during training
2. **No R2 upload is happening** - checkpoints remain local only
3. Inference requires checkpoints to be available long-term
4. All checkpoints uploaded to R2 would be cost-prohibitive

### Key Questions Raised

1. **Storage scope**: Are we storing all epoch checkpoints locally?
   - Answer: No, only `best.pt` and `last.pt` (via `save_period=-1`)

2. **Upload timing**: When should checkpoints be uploaded to R2?
   - Upload every epoch? (expensive, impacts performance)
   - Upload on improvement? (still many uploads)
   - Upload once at completion? (simple, but what about interruptions?)

3. **Interruption handling**: What if training is stopped early?
   - User stops intentionally (Ctrl+C)
   - Error crashes training
   - User judges training is "good enough"

4. **UI consistency**: How to reflect R2 upload status in metrics table?
   - Currently shows checkpoint icon if file exists locally
   - Should show upload status to R2 instead

## Current State

### Local Storage (YOLO)

```python
# ultralytics_adapter.py:2027-2028
'save': True,                # Save checkpoints
'save_period': -1,           # Only save last and best
```

**Files saved**:
```
{output_dir}/job_{job_id}/weights/
├── best.pt      # Best validation performance
└── last.pt      # Most recent epoch
```

**Checkpoint sizes**:
- YOLO11n: ~6MB
- YOLO11s: ~20MB
- YOLO11m: ~50MB

### R2 Upload Implementation

**Function exists but not called**:
```python
# storage.py:527 - upload_checkpoint()
# - Supports project-centric paths
# - Non-blocking (warns on failure)
# - Already implemented ✅

# But never called! ❌
```

**Database tracking**:
```python
# models.py:334
checkpoint_path = Column(String(500), nullable=True)

# Currently populated with LOCAL path during training
# Should be populated with R2 path after upload
```

**Frontend display**:
```tsx
// DatabaseMetricsTable.tsx:306
{metric.checkpoint_path ? (
  <CheckCircle2 className="w-3 h-3 text-green-600" />  // Shows if path exists
) : (
  <XCircle className="w-3 h-3 text-gray-300" />
)}
```

### Try-Catch Handling

```python
# ultralytics_adapter.py:1967-1999
try:
    results = self.model.train(**train_args)
    # ... success logic ...
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅ With checkpoint_dir

except KeyboardInterrupt:
    callbacks.on_train_end()  # ❌ No checkpoint_dir!

except Exception as e:
    callbacks.on_train_end()  # ❌ No checkpoint_dir!
```

## Proposed Solution

### Policy: **Upload Once on Completion/Interruption**

**Core principles**:
1. Save only `best.pt` and `last.pt` locally (already done ✅)
2. Upload to R2 once when training ends (normal completion, interruption, or error)
3. Track R2 upload status in database (not local file existence)
4. UI shows checkmark only for epochs with R2-uploaded checkpoints

### Upload Timing Decision Matrix

| Scenario | Best.pt Upload | Last.pt Upload | Rationale |
|----------|---------------|----------------|-----------|
| Normal completion | ✅ | ✅ | Full success |
| User interruption (Ctrl+C) | ✅ | ✅ | Preserve partial results |
| Training error | ✅ | ✅ | Attempt salvage |
| Early epochs (no checkpoint yet) | ❌ | ❌ | Files don't exist |

### Why "Upload Once at End"?

**Considered alternatives**:

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Every epoch** | Max safety | High cost, slow training | ❌ Rejected |
| **Every N epochs** | Balanced | Still redundant uploads | ❌ Rejected |
| **On improvement** | Meaningful uploads | Early training = many uploads | ❌ Rejected |
| **Once at end** ✅ | Simple, fast, cheap | No mid-training backup | ✅ **Selected** |

**Rationale for selection**:
- Most training jobs complete successfully
- Interruptions are rare
- 2 uploads per job (best + last) is cost-effective
- No performance impact during training
- Sufficient for inference use case

## Implementation Plan

### Phase 1: Core Upload Logic

**File**: `mvp/training/platform_sdk/base.py`

```python
def on_train_end(
    self,
    final_metrics: Dict[str, float] = None,
    checkpoint_dir: Optional[str] = None
):
    """
    Called when training ends (completion, interruption, or error).
    Uploads checkpoints to R2 and updates database.
    """
    import mlflow
    from platform_sdk.storage import upload_checkpoint
    import os

    if not self.mlflow_run:
        print(f"[Callbacks] Warning: MLflow run not active")
        return

    # Log final metrics to MLflow
    if final_metrics:
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_{key}", value)

    # Upload checkpoints to R2 and update DB
    uploaded_checkpoints = {}  # {epoch: r2_path}

    if checkpoint_dir:
        best_pt = os.path.join(checkpoint_dir, 'best.pt')
        last_pt = os.path.join(checkpoint_dir, 'last.pt')

        # Upload best.pt
        if os.path.exists(best_pt):
            print(f"[Callbacks] Uploading best.pt to R2...")
            sys.stdout.flush()

            success = upload_checkpoint(
                checkpoint_path=best_pt,
                job_id=self.job_id,
                checkpoint_name='best.pt',
                project_id=self.project_id
            )

            if success:
                # Build R2 path
                if self.project_id:
                    r2_path = f'r2://vision-platform-prod/checkpoints/projects/{self.project_id}/jobs/{self.job_id}/best.pt'
                else:
                    r2_path = f'r2://vision-platform-prod/checkpoints/test-jobs/job_{self.job_id}/best.pt'

                # Find best epoch from database
                best_epoch = self._find_best_epoch()
                if best_epoch:
                    uploaded_checkpoints[best_epoch] = r2_path
                    print(f"[Callbacks] Best checkpoint uploaded: epoch {best_epoch}")
                    sys.stdout.flush()

        # Upload last.pt
        if os.path.exists(last_pt):
            print(f"[Callbacks] Uploading last.pt to R2...")
            sys.stdout.flush()

            success = upload_checkpoint(
                checkpoint_path=last_pt,
                job_id=self.job_id,
                checkpoint_name='last.pt',
                project_id=self.project_id
            )

            if success:
                if self.project_id:
                    r2_path = f'r2://vision-platform-prod/checkpoints/projects/{self.project_id}/jobs/{self.job_id}/last.pt'
                else:
                    r2_path = f'r2://vision-platform-prod/checkpoints/test-jobs/job_{self.job_id}/last.pt'

                # Find last epoch from database
                last_epoch = self._find_last_epoch()
                if last_epoch:
                    uploaded_checkpoints[last_epoch] = r2_path
                    print(f"[Callbacks] Last checkpoint uploaded: epoch {last_epoch}")
                    sys.stdout.flush()

        # Update database with R2 paths
        if uploaded_checkpoints:
            self._update_checkpoint_paths(uploaded_checkpoints)
            print(f"[Callbacks] Database updated with {len(uploaded_checkpoints)} checkpoint paths")
            sys.stdout.flush()

    # End MLflow run
    mlflow.end_run()
    print(f"[Callbacks] MLflow run ended: {self.mlflow_run_id}")
    self.mlflow_run = None

def _find_best_epoch(self) -> Optional[int]:
    """Find epoch with best primary metric value."""
    try:
        import sqlite3
        from pathlib import Path

        training_dir = Path(__file__).parent.parent
        mvp_dir = training_dir.parent
        db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

        if not db_path.exists():
            return None

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Find epoch with highest primary_metric_value
        cursor.execute("""
            SELECT epoch
            FROM validation_results
            WHERE job_id = ?
            ORDER BY primary_metric_value DESC
            LIMIT 1
        """, (self.job_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None
    except Exception as e:
        print(f"[Callbacks WARNING] Failed to find best epoch: {e}")
        return None

def _find_last_epoch(self) -> Optional[int]:
    """Find last (most recent) epoch number."""
    try:
        import sqlite3
        from pathlib import Path

        training_dir = Path(__file__).parent.parent
        mvp_dir = training_dir.parent
        db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

        if not db_path.exists():
            return None

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Find max epoch
        cursor.execute("""
            SELECT MAX(epoch)
            FROM validation_results
            WHERE job_id = ?
        """, (self.job_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else None
    except Exception as e:
        print(f"[Callbacks WARNING] Failed to find last epoch: {e}")
        return None

def _update_checkpoint_paths(self, checkpoints: Dict[int, str]):
    """Update checkpoint_path in database for uploaded checkpoints."""
    try:
        import sqlite3
        from pathlib import Path

        training_dir = Path(__file__).parent.parent
        mvp_dir = training_dir.parent
        db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        for epoch, r2_path in checkpoints.items():
            cursor.execute("""
                UPDATE validation_results
                SET checkpoint_path = ?
                WHERE job_id = ? AND epoch = ?
            """, (r2_path, self.job_id, epoch))

        conn.commit()
        conn.close()

        print(f"[Callbacks] Updated {len(checkpoints)} checkpoint paths in database")
    except Exception as e:
        print(f"[Callbacks WARNING] Failed to update checkpoint paths: {e}")
```

### Phase 2: Fix Exception Handling

**File**: `mvp/training/adapters/ultralytics_adapter.py`

```python
# Line 1995: Define checkpoint_dir BEFORE try block
checkpoint_dir = os.path.join(self.output_dir, f"job_{self.job_id}", "weights")

# YOLO training
try:
    print("[YOLO] Starting YOLO training loop...")
    # ... training setup ...

    results = self.model.train(**train_args)

    print("[YOLO] Training completed!")
    # ... success logging ...

except KeyboardInterrupt:
    print("\n[YOLO] Training interrupted by user")
    print("[YOLO] Uploading current checkpoints before exit...")
    sys.stdout.flush()
    if self.logger:
        self.logger.log_message("Training interrupted by user")
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅ Added!
    raise

except Exception as e:
    print(f"\n[YOLO] ERROR during training: {e}")
    print("[YOLO] Attempting to upload checkpoints despite error...")
    sys.stdout.flush()
    if self.logger:
        self.logger.log_message(f"ERROR: Training failed - {str(e)}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅ Added!
    raise

# Normal completion
print(f"\nTraining completed!")
print("[YOLO] Uploading final checkpoints...")
callbacks.on_train_end(checkpoint_dir=checkpoint_dir)
```

### Phase 3: Remove In-Training Checkpoint Tracking

**File**: `mvp/training/adapters/ultralytics_adapter.py`

```python
# Line 1590-1602: Remove local checkpoint_path assignment during training
# Old code (REMOVE):
# checkpoint_path = None
# best_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'best.pt')
# last_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'last.pt')
# if os.path.exists(best_weights):
#     checkpoint_path = best_weights
# elif os.path.exists(last_weights):
#     checkpoint_path = last_weights

# New code (SIMPLE):
checkpoint_path = None  # Will be set by on_train_end() after R2 upload
```

### Phase 4: Modify upload_checkpoint() Return Value

**File**: `mvp/training/platform_sdk/storage.py`

```python
def upload_checkpoint(
    checkpoint_path: str,
    job_id: int,
    checkpoint_name: str = "best.pt",
    project_id: int = None
) -> bool:  # ✅ Changed from None to bool
    """
    Upload training checkpoint to R2.

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        import boto3
        from pathlib import Path

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            print(f"[R2 WARNING] Checkpoint file not found: {checkpoint_path}")
            sys.stdout.flush()
            return False  # ✅ Changed from return to return False

        # Check R2 credentials
        endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        if not all([endpoint, access_key, secret_key]):
            print(f"[R2] R2 credentials not configured, skipping checkpoint upload")
            sys.stdout.flush()
            return False  # ✅ Changed

        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        bucket = 'vision-platform-prod'

        # Build path based on project_id
        if project_id:
            key = f'checkpoints/projects/{project_id}/jobs/{job_id}/{checkpoint_name}'
        else:
            key = f'checkpoints/test-jobs/job_{job_id}/{checkpoint_name}'

        print(f"[R2] Uploading checkpoint to R2: s3://{bucket}/{key}...")
        sys.stdout.flush()

        # Get file size for progress reporting
        file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f"[R2] Checkpoint size: {file_size_mb:.2f} MB")
        sys.stdout.flush()

        s3.upload_file(str(checkpoint_file), bucket, key)

        print(f"[R2] Checkpoint upload successful!")
        print(f"[R2] Checkpoint available at: s3://{bucket}/{key}")
        sys.stdout.flush()

        return True  # ✅ Added

    except Exception as e:
        # Don't fail training just because upload failed
        warnings.warn(f"[R2 WARNING] Failed to upload checkpoint to R2: {e}", UserWarning)
        print(f"[R2 WARNING] Checkpoint upload failed (non-critical): {e}")
        sys.stdout.flush()
        return False  # ✅ Added
```

## Technical Details

### R2 Path Convention

```
# With project_id
r2://vision-platform-prod/checkpoints/projects/{project_id}/jobs/{job_id}/best.pt
r2://vision-platform-prod/checkpoints/projects/{project_id}/jobs/{job_id}/last.pt

# Without project_id (test jobs)
r2://vision-platform-prod/checkpoints/test-jobs/job_{job_id}/best.pt
r2://vision-platform-prod/checkpoints/test-jobs/job_{job_id}/last.pt
```

### Database Schema

```sql
-- validation_results table
CREATE TABLE validation_results (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    primary_metric_value FLOAT,
    checkpoint_path VARCHAR(500),  -- NULL during training, R2 path after upload
    -- ... other fields ...
);

-- Expected values during lifecycle:
-- Training (epoch 1-99):   checkpoint_path = NULL
-- After completion/stop:   checkpoint_path = 'r2://...' (for best & last epochs only)
```

### Frontend Behavior

```tsx
// DatabaseMetricsTable.tsx
// Checkmark shown ONLY if checkpoint_path starts with 'r2://'
{metric.checkpoint_path?.startsWith('r2://') ? (
  <CheckCircle2 className="w-3 h-3 text-green-600" />
) : (
  <XCircle className="w-3 h-3 text-gray-300" />
)}
```

### Upload Timing Examples

**Example 1: Normal 100-epoch training**
```
Epoch 1-99: No uploads, checkpoint_path = NULL
Epoch 100: Training completes
  → Upload best.pt (epoch 85)
  → Upload last.pt (epoch 100)
  → Update DB: epochs 85, 100 get R2 paths
```

**Example 2: User stops at epoch 20**
```
Epoch 1-19: No uploads
Epoch 20: User presses Ctrl+C
  → KeyboardInterrupt caught
  → Upload best.pt (epoch 18)
  → Upload last.pt (epoch 20)
  → Update DB: epochs 18, 20 get R2 paths
```

**Example 3: Early training (stopped at epoch 2)**
```
Epoch 1: Training starts
Epoch 2: User stops
  → best.pt might not exist yet (no validation done)
  → last.pt exists
  → Upload last.pt only
  → Update DB: epoch 2 gets R2 path
```

## Migration Path

### Step 1: Update Code (Breaking Change)
- Modify `on_train_end()` to upload checkpoints
- Fix exception handlers to pass `checkpoint_dir`
- Remove in-training checkpoint_path assignment

### Step 2: Database Migration (Optional)
```sql
-- Clear existing local paths (they're invalid anyway)
UPDATE validation_results
SET checkpoint_path = NULL
WHERE checkpoint_path IS NOT NULL
  AND checkpoint_path NOT LIKE 'r2://%';
```

### Step 3: Testing
1. Start new training job
2. Check DB during training: all `checkpoint_path` should be NULL
3. Stop training at epoch 10
4. Verify R2 upload happens
5. Check DB: best & last epochs should have R2 paths
6. Frontend should show 2 checkmarks only

### Step 4: Existing Jobs
- Old jobs: No retroactive upload (checkpoints may be lost)
- New jobs: Full R2 upload support

## Cost Analysis

### Storage Cost (Cloudflare R2)

**Per job**:
- Best.pt: ~20MB average
- Last.pt: ~20MB average
- **Total per job**: ~40MB

**At scale** (1000 jobs):
- Total storage: 40GB
- R2 storage: $0.015/GB/month
- **Monthly cost**: $0.60

**Compared to alternatives**:
- Upload every epoch (100 epochs): 2GB per job → $30/month for 1000 jobs (50x more expensive!)
- Upload every 10 epochs: 200MB per job → $3/month (5x more expensive)

### Upload Cost (R2 Operations)

- **Class B Operations** (PUT): Free (10M requests/month)
- **2 uploads per job** = negligible cost

## Alternatives Considered

### Alternative 1: Periodic Upload During Training

```python
# Every 20 epochs
if epoch % 20 == 0:
    upload_checkpoint(best.pt)
    upload_checkpoint(last.pt)
```

**Pros**:
- Mid-training backup
- Safer for very long training

**Cons**:
- Multiple redundant uploads (overwriting same file)
- Training slowdown (waiting for upload)
- Higher cost (10 uploads for 100 epochs)

**Why rejected**: MVP doesn't need mid-training backup. Most jobs complete successfully. Add later if needed.

### Alternative 2: Upload on Best Model Improvement

```python
if is_new_best:
    upload_checkpoint(best.pt)
```

**Pros**:
- Only upload meaningful improvements
- Fewer uploads than every epoch

**Cons**:
- Early training improves almost every epoch
- Still many uploads (20-30 for typical training)
- Complex logic

**Why rejected**: Early training would still cause many uploads. Not significantly better than "once at end".

### Alternative 3: Async Upload with Queue

```python
upload_queue = Queue()
background_worker.start()

# Training code
upload_queue.put(('best.pt', checkpoint_path))
# Training continues immediately

# Background worker
while True:
    item = upload_queue.get()
    upload_checkpoint(*item)
```

**Pros**:
- Zero training slowdown
- Can debounce uploads (wait 5min, then upload latest)

**Cons**:
- Complex implementation (threading/multiprocessing)
- Debugging difficulty
- Overkill for MVP

**Why rejected**: MVP simplicity is priority. Training isn't blocked by 1-2 second upload at the end anyway.

## References

### Related Files
- `mvp/training/platform_sdk/base.py:1724` - `on_train_end()` implementation
- `mvp/training/platform_sdk/storage.py:527` - `upload_checkpoint()` implementation
- `mvp/training/adapters/ultralytics_adapter.py:1967-1999` - Exception handling
- `mvp/training/adapters/ultralytics_adapter.py:1590-1602` - In-training checkpoint tracking
- `mvp/backend/app/db/models.py:334` - `checkpoint_path` field
- `mvp/frontend/components/training/DatabaseMetricsTable.tsx:306` - Checkpoint UI

### Related Documentation
- [Project-Centric Checkpoint Storage](../CONVERSATION_LOG.md#2025-11-04-2130-project-centric-checkpoint-storage-구현) (2025-11-04)
- [Stratified Split & Validation Metrics](../CONVERSATION_LOG.md#2025-11-05-1415-yolo-validation-metrics-이슈-조사-및-stratified-split-구현) (2025-11-05)

### External Resources
- [Cloudflare R2 Pricing](https://developers.cloudflare.com/r2/pricing/)
- [PyTorch Checkpoint Best Practices](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Ultralytics Training Arguments](https://docs.ultralytics.com/modes/train/#arguments)

## Notes

### Open Questions
- Should we add a manual "Upload Checkpoint" API for specific epochs?
- Should we implement checkpoint expiration (delete after 30 days)?
- How to handle inference when checkpoint is not in R2 (fall back to local)?

### Future Considerations

**Phase 2 Enhancements** (post-MVP):
1. **Checkpoint Download API**: For inference to fetch from R2
2. **Lifecycle Policy**: Auto-delete checkpoints after N days if unused
3. **Checkpoint Browser**: UI to view all checkpoints, download, delete
4. **Resume Training**: Fetch `last.pt` from R2 to resume interrupted training

**Monitoring**:
- Add metric: "checkpoint_upload_success_rate"
- Alert if upload failure rate > 5%
- Dashboard: Total checkpoint storage used

### Implementation Priority

1. **P0** (MVP): Basic upload on completion/interruption ✅
2. **P1** (Inference testing): Checkpoint download API
3. **P2** (Cost optimization): Lifecycle policy
4. **P3** (UX improvement): Checkpoint browser UI

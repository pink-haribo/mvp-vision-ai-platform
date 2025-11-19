# Phase 3.1.1: Checkpoint Management Enhancement

이 내용을 `MVP_TO_PLATFORM_CHECKLIST.md`의 **Phase 3.1 섹션 끝 (line 1764 이후)** 에 추가해주세요.

---

#### Phase 3.1.1: Checkpoint Management Enhancement ✅ COMPLETED (2025-11-18)

**Goal**: Complete checkpoint management system by adding last.pt support (resumable training)

**Background**: Original implementation only uploaded best.pt. Platform design requires both:
- `best.pt`: Best validation metrics checkpoint
- `last.pt`: Final epoch state for resumable training

**Implementation** (6 tasks completed):
- [x] **Database Migration** `platform/backend/migrate_add_last_checkpoint.py`
  - [x] Added `last_checkpoint_path VARCHAR(500)` to training_jobs table
  - [x] PostgreSQL migration with rollback safety
  - [x] Verification script for column existence check
- [x] **Backend Data Models** `platform/backend/app/db/models.py:487`
  - [x] Added TrainingJob.last_checkpoint_path field
- [x] **API Schemas** `platform/backend/app/schemas/training.py`
  - [x] Added last_checkpoint_path to TrainingJobResponse (line 118)
  - [x] Added last_checkpoint_path to TrainingCompletionCallback (line 243)
- [x] **API Handler** `platform/backend/app/api/training.py:1732-1737`
  - [x] Updated training_completion_callback() to save last_checkpoint_path
  - [x] Stores both checkpoint paths from completion callback
- [x] **Training Service** `platform/trainers/ultralytics/train.py:371-391`
  - [x] Uploads both best.pt and last.pt to S3
  - [x] S3 paths: `s3://training-checkpoints/checkpoints/{job_id}/best.pt` and `last.pt`
  - [x] completion_data includes both best_checkpoint_path and last_checkpoint_path
- [x] **Documentation** `docs/planning/MVP_TO_PLATFORM_CHECKLIST.md`
  - [x] Updated checkpoint validation section (line 446-449)
  - [x] Added Phase 3.1.1 section

**Flow**:
```
Training Complete
  ↓
Save Locally: /tmp/training/{job_id}/runs/train/weights/
  ├─ best.pt (best metrics)
  └─ last.pt (final state)
  ↓
Upload to S3: s3://training-checkpoints/checkpoints/{job_id}/
  ├─ best.pt
  └─ last.pt
  ↓
Callback → Backend
  ├─ training_jobs.best_checkpoint_path
  └─ training_jobs.last_checkpoint_path
```

**Testing**: Pending E2E test with new training job

**Files Modified**:
1. `platform/backend/app/db/models.py` (+1 field)
2. `platform/backend/app/schemas/training.py` (+2 fields)
3. `platform/backend/app/api/training.py` (+3 lines)
4. `platform/trainers/ultralytics/train.py` (+20 lines)
5. `platform/backend/migrate_add_last_checkpoint.py` (new file, 52 lines)

**Progress**: 6/6 tasks completed (100%) ✅

---

## Phase 3.1의 기존 항목도 업데이트:

Line 1738을 다음과 같이 변경:
```markdown
# 변경 전:
  - [x] S3 checkpoint upload

# 변경 후:
  - [x] S3 checkpoint upload (best.pt + last.pt) ← Updated 2025-11-18
```

Line 1755를 다음과 같이 변경:
```markdown
# 변경 전:
  - [x] S3 checkpoints uploaded

# 변경 후:
  - [x] S3 checkpoints uploaded (best & last) ← Updated 2025-11-18
```


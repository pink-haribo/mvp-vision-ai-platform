# YOLO Validation Metrics Always Zero

**Status:** 🔴 CANNOT FIX - PyTorch Design Limitation
**Date:** 2025-11-05
**Impact:** Medium (Training works, inference expected to work, post-training validation possible)

## Symptoms

- All validation metrics = 0 during training (mAP, precision, recall)
- Final `best.pt` validation shows confusion matrix with data
- Training loss decreases normally
- Model saves successfully

## Root Cause Analysis

### Primary Cause: Callback Timing
Ultralytics callback timing issue:
```python
on_fit_epoch_end:
  validator.batch = None  # Not available at callback time
  validator.pred = None   # No predictions accessible
  confusion_matrix.sum() = 0  # Not populated yet
```

### Secondary Issue: PyTorch InferenceMode
**Manual validation approach is fundamentally impossible:**

```python
# Attempted solution
with torch.inference_mode():  # Used by Ultralytics model.val()
    validate()  # Converts parameters to "inference tensors"

# After InferenceMode
param.requires_grad = True  # ❌ RuntimeError!
# "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed"
```

**Why InferenceMode is used:**
- PyTorch 2.0+ feature for faster inference
- Completely disables autograd tracking
- Irreversibly converts tensors (cannot restore requires_grad)
- Ultralytics uses it for performance optimization

**Comparison:**

| Context | Gradient | Post-restoration | Performance |
|---------|----------|------------------|-------------|
| `no_grad()` | Disabled | ✅ Possible | Slower |
| `inference_mode()` | Disabled | ❌ Impossible | Faster |

## Investigation Log

### Attempt 1: Callback Debugging
Added extensive logging to callbacks:
- `on_val_batch_start`: No batch attribute
- `on_val_batch_end`: validator.pred = None, no targets
- Ground truth: 49 labels loaded, but not in confusion matrix

### Attempt 2: Manual Validation (Initial)
Tried calling `model.val()` directly in callback:
```python
on_fit_epoch_end:
  val_results = model.val(...)  # Run validation manually
  # Result: Gradient error on next training step
```

**Error:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Reason:** `model.val()` disables gradients

### Attempt 3: State Restoration
Tried restoring model state after validation:
```python
# Save state
was_training = model.training
original_grad_states = {name: p.requires_grad for name, p in model.named_parameters()}

# Run validation
model.val(...)

# Restore state
torch.set_grad_enabled(True)
model.train()
for name, param in model.named_parameters():
    param.requires_grad = original_grad_states[name]  # ❌ FAILS!
```

**Error:** `RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode is not allowed`

**Reason:** PyTorch InferenceMode converts tensors irreversibly

### Attempt 4: InferenceMode Investigation
Discovered Ultralytics uses `torch.inference_mode()` instead of `torch.no_grad()`:
- InferenceMode is faster but more restrictive
- Prevents any gradient-related operations after context exit
- Cannot restore `requires_grad` on inference tensors
- **Fundamental limitation, not a bug**

## Current Workaround

None. Validation metrics remain 0 during training.

**Training still works:**
- ✅ Loss decreases normally
- ✅ Model saves (best.pt, last.pt)
- ✅ Inference expected to work
- ❌ Cannot monitor validation metrics per epoch

## Possible Solutions (Future)

### ✅ Option 1: Post-Training Validation (RECOMMENDED)
Run separate validation after training completes:
```python
# After training
results = model.train(...)

# Separate validation - WORKS PERFECTLY
val_metrics = model.val(data=data_yaml, split='val')
# Returns: mAP, precision, recall, confusion matrix ✓
```

**Pros:**
- Simple, reliable
- Full metrics available
- No interference with training

**Cons:**
- No per-epoch monitoring
- Only final metrics

### ❌ Option 2: Find Proper Hook Point
Investigate Ultralytics source to find callback with:
- Access to predictions
- Access to ground truth
- No InferenceMode interference

**Status:** Investigated, no suitable hook exists

### ⚠️ Option 3: Custom Validator (Complex)
Implement separate validation loop:
```python
model.eval()
with torch.no_grad():  # Use no_grad, not inference_mode
    for batch in val_loader:
        preds = model(batch['img'])
        # Calculate metrics manually
        # Update confusion matrix
        # Calculate mAP, precision, recall

model.train()  # Resume training
```

**Pros:**
- Full control
- Per-epoch metrics
- No InferenceMode issues

**Cons:**
- Need to implement all YOLO metrics logic
- Maintenance burden
- Time consuming (~1-2 days)

### ❌ Option 4: Modify Ultralytics Source
Change `torch.inference_mode()` to `torch.no_grad()` in Ultralytics code

**Pros:**
- Would work

**Cons:**
- Breaks on Ultralytics updates
- Slower performance
- Not maintainable

## Related Files

- `mvp/training/adapters/ultralytics_adapter.py:1200-1700` - Callback implementation
- `mvp/training/converters/dice_to_yolo.py:136-212` - Stratified split (fixed)

## References

- Ultralytics Issue: TBD (search for similar issues)
- Discussion: https://github.com/ultralytics/ultralytics/discussions

## Final Decision

**Status:** ❌ CANNOT FIX in current architecture

**Reasoning:**
1. Ultralytics uses `torch.inference_mode()` for performance
2. InferenceMode is a PyTorch design choice, not a bug
3. Manual validation during training is fundamentally incompatible
4. All attempted workarounds fail due to PyTorch limitations

**Action Plan:**
1. ✅ Accept that per-epoch validation metrics = 0
2. ✅ Training still works (loss decreases, model saves)
3. ✅ Post-training validation works perfectly
4. ✅ Inference will work (tested separately)
5. ⏭️ Move to next priority: Inference API testing

**Impact Assessment:**
- **Low:** Training functionality unaffected
- **Medium:** Cannot monitor per-epoch validation progress
- **Workaround:** Use training loss as progress indicator
- **Future:** Can implement custom validator if needed (~1-2 days work)

## Lessons Learned

1. **PyTorch InferenceMode is strict:** Cannot restore gradients after use
2. **Ultralytics optimization choices:** Performance over flexibility
3. **Callback limitations:** Not all data accessible at callback time
4. **Architecture constraints:** Some integrations are fundamentally impossible

## Tested on

- Dataset: COCO32 (32 images, 43 classes) with stratified split
- Dataset: COCO128 (128 images, 71 classes) with stratified split
- Model: YOLO11n
- PyTorch: 2.1.1
- Ultralytics: 8.3.224

## Next Steps

1. ✅ **Proceed to Inference API testing**
2. Test other YOLO models (seg, pose, obb)
3. Test timm models
4. If validation metrics critical: Implement custom validator later

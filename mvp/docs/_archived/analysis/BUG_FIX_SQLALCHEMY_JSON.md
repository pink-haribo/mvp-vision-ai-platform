# Bug Fix: SQLAlchemy JSON Column Change Detection

**Date**: 2025-10-22
**Status**: ✅ FIXED
**Severity**: Critical - Data loss bug

---

## 🐛 Bug Description

Multi-step conversation config was being lost between requests:
- **Step 1**: User selects "ResNet18" → config saved: `{framework, model_name, task_type}` ✅
- **Step 2**: User provides dataset path → config should add `dataset_path` ❌ **LOST!**
- **Step 3**: User provides hyperparameters → config should add `epochs`, `batch_size`, `learning_rate` ❌ **LOST!**

Only the FIRST config from Step 1 was persisting to database.

---

## 🔍 Investigation Process

### Initial Hypothesis: LLM Problem
Suspected Gemini wasn't preserving previous config when parsing new user input.

**Testing**:
- Created `test_gemini_direct.py` to test Gemini API directly
- **Result**: Gemini worked perfectly - preserved all previous fields and added new ones ✅

**Conclusion**: LLM is NOT the problem. Issue is in application code data flow.

### Data Flow Debugging

Added 6 trace points through the entire data flow:

1. **TRACE-1-LOAD**: Loading session from DB
2. **TRACE-2-LLM-IN**: Passing config to Gemini
3. **TRACE-3-LLM-OUT**: Gemini's response config
4. **TRACE-4-MERGE**: Merging LLM config with existing config
5. **TRACE-5-SAVE**: Setting `session.temp_data` before commit
6. **TRACE-6-VERIFY**: Refreshing session after `db.commit()`

### Critical Discovery

Backend logs for Step 2:

```
[TRACE-5-SAVE] Saving to DB:
  config keys: ['framework', 'model_name', 'task_type', 'dataset_path', 'dataset_format']

[TRACE-6-VERIFY] After commit:
  config keys in DB: ['framework', 'model_name', 'task_type']  ← dataset_path GONE!
```

**Before commit**: 5 fields ✅
**After commit**: 3 fields ❌

This proved the data was correct until line 155 (`session.temp_data = updated_temp_data`), but wasn't being persisted by `db.commit()`.

---

## 🎯 Root Cause

**SQLAlchemy doesn't detect changes to mutable objects (dict) in JSON columns when you assign the same object reference back.**

### The Code Flow

**In `action_handlers.py` (_handle_ask_clarification)**:
```python
def _handle_ask_clarification(self, action_response, session, user_message):
    temp_data = session.temp_data or {}  # ← Gets REFERENCE to session.temp_data dict

    # ... modify temp_data in-place ...

    return {
        "temp_data": temp_data  # ← Returns THE SAME OBJECT
    }
```

**In `conversation_manager.py` (process_message)**:
```python
result = await self.action_handlers.handle_action(...)
updated_temp_data = result["temp_data"]  # ← Same dict object as session.temp_data

# This assigns the SAME object back to session.temp_data!
session.temp_data = updated_temp_data  # ← SQLAlchemy sees: same object = no change!

self.db.commit()  # ← Nothing to commit! Change not detected!
```

### Why SQLAlchemy Misses It

SQLAlchemy uses **object identity tracking** for change detection:

```python
# This is detected as a change:
session.temp_data = {"new": "dict"}  # Different object

# This is NOT detected:
session.temp_data = session.temp_data  # Same object
```

Even though we modified the dict's contents, if we assign the same object reference back, SQLAlchemy thinks nothing changed.

---

## ✅ Solution

Use SQLAlchemy's `flag_modified()` to explicitly mark the attribute as changed:

**File**: `mvp/backend/app/services/conversation_manager.py`

```python
# Line 153-160 (AFTER fix)
# 5. Update session in DB
session.state = new_state.value
session.temp_data = updated_temp_data

# CRITICAL FIX: Force SQLAlchemy to detect JSON column change
# When updated_temp_data is the same dict object, SQLAlchemy won't see the change
from sqlalchemy.orm.attributes import flag_modified
flag_modified(session, "temp_data")
```

This explicitly tells SQLAlchemy: "I modified this attribute, please persist it even if you think it's the same object."

---

## 🧪 Verification

### Before Fix

```bash
$ python test_with_debug_endpoint.py

>>> After STEP 2:
Config: {
  "framework": "timm",
  "model_name": "resnet18",
  "task_type": "image_classification"
}
Config keys: ['framework', 'model_name', 'task_type']

FAILED - dataset_path is MISSING!
```

### After Fix

```bash
$ python test_with_debug_endpoint.py

>>> After STEP 2:
Config: {
  "framework": "timm",
  "model_name": "resnet18",
  "task_type": "image_classification",
  "dataset_path": "C:\\datasets\\cls\\imagenet-10",
  "dataset_format": "imagefolder"
}
Config keys: ['framework', 'model_name', 'task_type', 'dataset_path', 'dataset_format']

SUCCESS - dataset_path was added!
```

Backend logs now show:

```
[TRACE-6-VERIFY] After commit:
  config keys in DB: ['framework', 'model_name', 'task_type', 'dataset_path', 'dataset_format']
```

All 5 fields persisted correctly! ✅

---

## 📚 Lessons Learned

### 1. SQLAlchemy JSON Column Gotcha

When working with JSON columns (or any mutable type), be aware:

❌ **Wrong** - SQLAlchemy won't detect change:
```python
data = session.my_json_column
data["new_key"] = "value"
session.my_json_column = data  # Same object - NOT detected!
db.commit()
```

✅ **Correct** - Use flag_modified:
```python
data = session.my_json_column
data["new_key"] = "value"
session.my_json_column = data

from sqlalchemy.orm.attributes import flag_modified
flag_modified(session, "my_json_column")  # Explicitly mark as changed
db.commit()
```

✅ **Alternative** - Create new dict:
```python
data = session.my_json_column
data["new_key"] = "value"
session.my_json_column = {**data}  # New dict object - detected!
db.commit()
```

### 2. Debugging JSON Column Issues

If data isn't persisting to database:

1. Add TRACE point BEFORE commit:
   ```python
   print(f"Before commit: {session.my_column}")
   ```

2. Add TRACE point AFTER commit + refresh:
   ```python
   db.commit()
   db.refresh(session)
   print(f"After commit: {session.my_column}")
   ```

3. Compare - if they differ, you have a change detection issue!

### 3. Alternative Architectures

To avoid this issue entirely, consider:

**Option A**: Always create new dict objects
```python
# In handlers
temp_data = dict(session.temp_data or {})  # Copy, not reference
# ... modify temp_data ...
return {"temp_data": temp_data}
```

**Option B**: Use SQLAlchemy MutableDict
```python
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy import JSON

class Session(Base):
    temp_data = Column(MutableDict.as_mutable(JSON))
```

This automatically tracks changes to the dict.

---

## 🔧 Related Files Modified

### Core Fix
- `mvp/backend/app/services/conversation_manager.py`: Added `flag_modified()` call

### Debug Infrastructure (can be removed after verification)
- Added TRACE-1 through TRACE-6 logging points
- `mvp/backend/test_with_debug_endpoint.py`: HTTP test script
- `mvp/backend/test_gemini_direct.py`: Direct LLM test
- `mvp/backend/app/api/debug.py`: Debug HTTP endpoints

### Documentation
- `docs/DEBUG_INFRASTRUCTURE_ISSUE.md`: Logging/tracing issues
- `docs/DECISION_LOG.md`: Solution options analysis
- `docs/BREAKTHROUGH.md`: Discovery that LLM wasn't the problem
- `docs/BUG_FIX_SQLALCHEMY_JSON.md`: This document

---

## ⚠️ Prevention

Going forward, whenever working with JSON columns:

1. **Always use `flag_modified()` after modifying JSON data**
2. **Test multi-step data accumulation** (not just single-step updates)
3. **Add integration tests** that verify data persists across requests
4. **Consider using `MutableDict`** for automatic change tracking

---

## 📊 Impact

**Before Fix**:
- ❌ Multi-step conversations lost config data
- ❌ Users couldn't complete training configuration via natural language
- ❌ Core feature (natural language training setup) was completely broken

**After Fix**:
- ✅ Config data accumulates correctly across conversation steps
- ✅ Natural language training configuration works end-to-end
- ✅ Users can configure training via multi-turn conversation

**Time to Fix**: ~3 hours of debugging + 5 minutes to implement fix

**Key Insight**: User's guidance was critical - "데이터 흐름 디버깅해보자. LLM 의 문제가 아니라면 문제 해결은 오히려 간단할 수 있어." (Debug the data flow. If it's not the LLM problem, the solution might actually be simple.)

Indeed, once we confirmed LLM wasn't the issue, finding the SQLAlchemy change detection problem was straightforward!

---

**Author**: Claude Code
**Reviewed**: Pending (2025-10-23)

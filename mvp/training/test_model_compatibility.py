"""Test model compatibility with timm and ultralytics libraries."""

import sys
import os

# Set encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

from model_registry.timm_models import TIMM_MODEL_REGISTRY
from model_registry.ultralytics_models import ULTRALYTICS_MODEL_REGISTRY


def test_timm_models():
    """Test if timm model names are valid."""
    print("=" * 60)
    print("TESTING TIMM MODEL COMPATIBILITY")
    print("=" * 60)

    try:
        import timm
        print(f"[OK] timm version: {timm.__version__}")
    except ImportError:
        print("[FAIL] timm not installed")
        return

    available_models = timm.list_models()
    print(f"[OK] Total available timm models: {len(available_models)}\n")

    results = {"P0": [], "P1": [], "P2": []}

    for model_name, metadata in TIMM_MODEL_REGISTRY.items():
        priority = metadata.get("priority", -1)
        display_name = metadata.get("display_name", model_name)
        priority_key = f"P{priority}" if priority in [0, 1, 2] else "Unknown"

        # Check if model is available in timm
        is_available = model_name in available_models
        status = "[OK]" if is_available else "[FAIL]"

        result = {
            "model_name": model_name,
            "display_name": display_name,
            "available": is_available,
            "status": status
        }

        if priority_key in results:
            results[priority_key].append(result)

        print(f"{status} [{priority_key}] {display_name:30s} ({model_name})")

    # Summary
    print("\n" + "=" * 60)
    print("TIMM SUMMARY")
    print("=" * 60)
    for priority in ["P0", "P1", "P2"]:
        total = len(results[priority])
        available = sum(1 for r in results[priority] if r["available"])
        unavailable = total - available
        print(f"{priority}: {available}/{total} available", end="")
        if unavailable > 0:
            print(f" ({unavailable} unavailable)")
            for r in results[priority]:
                if not r["available"]:
                    print(f"  - {r['model_name']}")
        else:
            print()

    return results


def test_ultralytics_models():
    """Test if ultralytics model names are valid."""
    print("\n" + "=" * 60)
    print("TESTING ULTRALYTICS MODEL COMPATIBILITY")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"[OK] ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("[FAIL] ultralytics not installed")
        return

    # Ultralytics models are typically available if the model weights can be downloaded
    # We'll check against known YOLO model patterns
    known_prefixes = [
        "yolov5", "yolov8", "yolo11", "yolov9", "yolov10",
        "yolo_world", "rtdetr", "sam"
    ]

    results = {"P0": [], "P1": [], "P2": []}

    print(f"[OK] Checking against known YOLO patterns: {', '.join(known_prefixes)}\n")

    for model_name, metadata in ULTRALYTICS_MODEL_REGISTRY.items():
        priority = metadata.get("priority", -1)
        display_name = metadata.get("display_name", model_name)
        priority_key = f"P{priority}" if priority in [0, 1, 2] else "Unknown"

        # Check if model name matches known patterns
        is_known = any(model_name.startswith(prefix) for prefix in known_prefixes)

        # Additional check: model_name should end with size indicator or variant
        # e.g., yolov8n, yolov5mu, yolo11l, etc.
        status = "[OK]" if is_known else "[?]"

        result = {
            "model_name": model_name,
            "display_name": display_name,
            "known_pattern": is_known,
            "status": status
        }

        if priority_key in results:
            results[priority_key].append(result)

        print(f"{status} [{priority_key}] {display_name:35s} ({model_name})")

    # Summary
    print("\n" + "=" * 60)
    print("ULTRALYTICS SUMMARY")
    print("=" * 60)
    for priority in ["P0", "P1", "P2"]:
        total = len(results[priority])
        known = sum(1 for r in results[priority] if r["known_pattern"])
        unknown = total - known
        print(f"{priority}: {known}/{total} known patterns", end="")
        if unknown > 0:
            print(f" ({unknown} unknown)")
            for r in results[priority]:
                if not r["known_pattern"]:
                    print(f"  - {r['model_name']}")
        else:
            print()

    return results


def main():
    """Run all compatibility tests."""
    timm_results = test_timm_models()
    ultralytics_results = test_ultralytics_models()

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    if timm_results:
        timm_total = sum(len(results) for results in timm_results.values())
        timm_available = sum(
            sum(1 for r in results if r["available"])
            for results in timm_results.values()
        )
        print(f"timm: {timm_available}/{timm_total} models available")

    if ultralytics_results:
        ultra_total = sum(len(results) for results in ultralytics_results.values())
        ultra_known = sum(
            sum(1 for r in results if r["known_pattern"])
            for results in ultralytics_results.values()
        )
        print(f"ultralytics: {ultra_known}/{ultra_total} known model patterns")

    print("\nNote: Ultralytics models may need to download weights on first use.")
    print("Actual availability will be confirmed during training/inference.")


if __name__ == "__main__":
    main()

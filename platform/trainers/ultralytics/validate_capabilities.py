#!/usr/bin/env python3
"""
Validate Ultralytics model capabilities.

Tests each model in capabilities.json to verify it can be loaded by Ultralytics.
Optionally updates the capabilities.json with validation results.

Usage:
    python validate_capabilities.py                    # Validate only
    python validate_capabilities.py --update           # Validate and update capabilities.json
    python validate_capabilities.py --model yolo11n    # Validate single model
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics package not found. Install with: pip install ultralytics")
    sys.exit(1)


def validate_model(model_name: str, verbose: bool = True) -> Tuple[bool, str]:
    """
    Validate that a model can be loaded by Ultralytics.

    Args:
        model_name: Model name (e.g., "yolo11n")
        verbose: Print validation progress

    Returns:
        (success: bool, error_message: str)
    """
    try:
        if verbose:
            print(f"  Testing {model_name}...", end=" ", flush=True)

        # Attempt to load the model (will auto-download if needed)
        model = YOLO(f"{model_name}.pt")

        if verbose:
            print("✓ OK")

        return True, ""

    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"✗ FAILED: {error_msg}")

        return False, error_msg


def validate_all_models(capabilities: Dict, verbose: bool = True) -> Dict[str, Tuple[bool, str]]:
    """
    Validate all models in capabilities.

    Args:
        capabilities: Loaded capabilities dict
        verbose: Print progress

    Returns:
        Dict mapping model_name to (success, error_message)
    """
    models = capabilities.get("models", [])

    if verbose:
        print(f"\nValidating {len(models)} models from capabilities.json...")
        print("-" * 80)

    results = {}

    for model in models:
        model_name = model["model_name"]
        success, error = validate_model(model_name, verbose=verbose)
        results[model_name] = (success, error)

    return results


def update_capabilities_with_results(
    capabilities: Dict,
    validation_results: Dict[str, Tuple[bool, str]],
    capabilities_file: Path
) -> None:
    """
    Update capabilities.json with validation results.

    Adds "validated" and "validation_date" fields to each model.
    Sets "supported" to False if validation failed.

    Args:
        capabilities: Loaded capabilities dict
        validation_results: Results from validate_all_models
        capabilities_file: Path to capabilities.json
    """
    validation_date = datetime.now().isoformat()

    for model in capabilities["models"]:
        model_name = model["model_name"]
        success, error = validation_results.get(model_name, (False, "Not tested"))

        # Update validation status
        model["validated"] = success
        model["validation_date"] = validation_date

        # If validation failed, mark as unsupported
        if not success:
            model["supported"] = False
            model["validation_error"] = error
        else:
            # Remove error field if exists
            model.pop("validation_error", None)

    # Write updated capabilities
    with open(capabilities_file, 'w', encoding='utf-8') as f:
        json.dump(capabilities, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Updated {capabilities_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Ultralytics model capabilities"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update capabilities.json with validation results"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Validate single model only"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Load capabilities.json
    script_dir = Path(__file__).parent
    capabilities_file = script_dir / "capabilities.json"

    if not capabilities_file.exists():
        print(f"ERROR: capabilities.json not found at {capabilities_file}")
        sys.exit(1)

    with open(capabilities_file, 'r', encoding='utf-8') as f:
        capabilities = json.load(f)

    if verbose:
        print("=" * 80)
        print("Ultralytics Model Capabilities Validator")
        print("=" * 80)
        print(f"Capabilities file: {capabilities_file}")
        print(f"Framework: {capabilities.get('framework')}")
        print(f"Version: {capabilities.get('version')}")

    # Validate single model or all
    if args.model:
        # Find model in capabilities
        model_found = None
        for m in capabilities["models"]:
            if m["model_name"] == args.model:
                model_found = m
                break

        if not model_found:
            print(f"ERROR: Model '{args.model}' not found in capabilities.json")
            sys.exit(1)

        success, error = validate_model(args.model, verbose=verbose)

        if success:
            print(f"\n✓ Model '{args.model}' validated successfully!")
            sys.exit(0)
        else:
            print(f"\n✗ Model '{args.model}' validation failed: {error}")
            sys.exit(1)

    else:
        # Validate all models
        validation_results = validate_all_models(capabilities, verbose=verbose)

        # Count results
        total = len(validation_results)
        passed = sum(1 for success, _ in validation_results.values() if success)
        failed = total - passed

        # Print summary
        if verbose:
            print("\n" + "=" * 80)
            print("Validation Summary")
            print("=" * 80)
            print(f"Total models: {total}")
            print(f"✓ Passed: {passed} ({passed/total*100:.1f}%)")
            print(f"✗ Failed: {failed} ({failed/total*100:.1f}%)")

        # Show failed models
        if failed > 0:
            print(f"\nFailed models:")
            for model_name, (success, error) in validation_results.items():
                if not success:
                    print(f"  - {model_name}: {error}")

        # Update capabilities.json if requested
        if args.update:
            update_capabilities_with_results(
                capabilities,
                validation_results,
                capabilities_file
            )

        # Exit with error if any failed
        if failed > 0:
            print(f"\n✗ Validation failed for {failed} model(s)")
            sys.exit(1)
        else:
            print(f"\n✓ All {total} models validated successfully!")
            sys.exit(0)


if __name__ == "__main__":
    main()

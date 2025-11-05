"""
Utility script to proactively upload pretrained model weights to R2.

This script downloads pretrained weights from original sources (Ultralytics, timm)
and uploads them to R2 for faster access during training.

Usage:
    python upload_pretrained_weights.py --framework ultralytics
    python upload_pretrained_weights.py --framework timm
    python upload_pretrained_weights.py --framework all  # Upload all frameworks
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from model_registry.ultralytics_models import ULTRALYTICS_MODEL_REGISTRY
from model_registry.timm_models import TIMM_MODEL_REGISTRY


def upload_to_r2(local_path: str, framework: str, model_name: str, file_extension: str) -> bool:
    """
    Upload model weights to R2.

    Args:
        local_path: Path to local model file
        framework: Framework name (ultralytics, timm)
        model_name: Model name
        file_extension: File extension (pt, pth, safetensors)

    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Check if R2 credentials are available
        endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        if not all([endpoint, access_key, secret_key]):
            print(f"[ERROR] R2 credentials not configured. Set environment variables:")
            print(f"  - AWS_S3_ENDPOINT_URL")
            print(f"  - AWS_ACCESS_KEY_ID")
            print(f"  - AWS_SECRET_ACCESS_KEY")
            return False

        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        bucket = 'vision-platform-prod'
        key = f'models/pretrained/{framework}/{model_name}.{file_extension}'

        # Check if already exists
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"[SKIP] Already exists in R2: s3://{bucket}/{key}")
            return True
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise

        # Get file size
        file_size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"[UPLOAD] Uploading to R2: s3://{bucket}/{key} ({file_size_mb:.2f} MB)...")

        # Upload
        s3.upload_file(local_path, bucket, key)

        print(f"[SUCCESS] Uploaded to R2: s3://{bucket}/{key}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to upload to R2: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_ultralytics_weights():
    """Download and upload Ultralytics pretrained weights."""
    print("\n" + "="*80)
    print("UPLOADING ULTRALYTICS PRETRAINED WEIGHTS")
    print("="*80 + "\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Install with: pip install ultralytics")
        return False

    # Mapping from registry model names to actual weight file names
    MODEL_WEIGHT_MAPPING = {
        "yolo11n": "yolo11n.pt",
        "yolo11n-seg": "yolo11n-seg.pt",
        "yolo11n-pose": "yolo11n-pose.pt",
        "yolo_world_v2_s": "yolov8s-worldv2.pt",  # YOLO-World naming convention
        "sam2_t": "sam2_t.pt",  # SAM2 naming convention
    }

    success_count = 0
    fail_count = 0

    for model_name, info in ULTRALYTICS_MODEL_REGISTRY.items():
        if not info.get("pretrained_available", False):
            print(f"[SKIP] {model_name}: No pretrained weights available")
            continue

        weight_filename = MODEL_WEIGHT_MAPPING.get(model_name)
        if not weight_filename:
            print(f"[SKIP] {model_name}: Unknown weight file mapping")
            continue

        print(f"\n[{model_name}] Downloading pretrained weights...")

        try:
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp())

            try:
                # Load model (this triggers download)
                model = YOLO(weight_filename)

                # Find downloaded weights
                # Ultralytics caches to ~/.cache/torch/hub/ultralytics/ or similar
                cache_dirs = [
                    Path.home() / ".cache" / "torch" / "hub" / "ultralytics",
                    Path.home() / ".cache" / "ultralytics",
                    Path.home() / ".ultralytics",
                ]

                weight_path = None
                for cache_dir in cache_dirs:
                    potential_path = cache_dir / weight_filename
                    if potential_path.exists():
                        weight_path = potential_path
                        break

                if not weight_path:
                    # Try finding in model object
                    if hasattr(model, 'ckpt_path') and model.ckpt_path:
                        weight_path = Path(model.ckpt_path)
                    elif hasattr(model, 'model_name') and model.model_name:
                        weight_path = Path(model.model_name)

                if not weight_path or not weight_path.exists():
                    print(f"[ERROR] {model_name}: Could not locate downloaded weights")
                    fail_count += 1
                    continue

                print(f"[FOUND] {model_name}: {weight_path}")

                # Upload to R2
                if upload_to_r2(str(weight_path), "ultralytics", model_name, "pt"):
                    success_count += 1
                else:
                    fail_count += 1

            finally:
                # Cleanup temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"ULTRALYTICS SUMMARY: {success_count} uploaded, {fail_count} failed")
    print(f"{'='*80}\n")

    return fail_count == 0


def upload_timm_weights():
    """Download and upload timm pretrained weights."""
    print("\n" + "="*80)
    print("UPLOADING TIMM PRETRAINED WEIGHTS")
    print("="*80 + "\n")

    try:
        import timm
        import torch
    except ImportError:
        print("[ERROR] timm or torch not installed. Install with: pip install timm torch")
        return False

    success_count = 0
    fail_count = 0

    for model_name, info in TIMM_MODEL_REGISTRY.items():
        if not info.get("pretrained_available", False):
            print(f"[SKIP] {model_name}: No pretrained weights available")
            continue

        model_id = info.get("model_id", model_name)
        print(f"\n[{model_name}] Downloading pretrained weights (timm ID: {model_id})...")

        try:
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            temp_file = temp_dir / f"{model_name}.pth"

            try:
                # Create model (this triggers weight download)
                model = timm.create_model(model_id, pretrained=True)

                # Save weights to temp file
                torch.save(model.state_dict(), temp_file)

                print(f"[SAVED] {model_name}: {temp_file} ({temp_file.stat().st_size / (1024*1024):.2f} MB)")

                # Upload to R2
                if upload_to_r2(str(temp_file), "timm", model_name, "pth"):
                    success_count += 1
                else:
                    fail_count += 1

            finally:
                # Cleanup temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"TIMM SUMMARY: {success_count} uploaded, {fail_count} failed")
    print(f"{'='*80}\n")

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Upload pretrained model weights to R2 storage"
    )
    parser.add_argument(
        "--framework",
        choices=["ultralytics", "timm", "all"],
        default="all",
        help="Which framework weights to upload (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models without uploading"
    )

    args = parser.parse_args()

    # Check R2 credentials (skip for dry-run)
    endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not args.dry_run and not all([endpoint, access_key, secret_key]):
        print("\n[ERROR] R2 credentials not configured!")
        print("\nRequired environment variables:")
        print("  - AWS_S3_ENDPOINT_URL")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("\nSet them in your .env file or environment.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"R2 PRETRAINED WEIGHT UPLOAD UTILITY")
    print(f"{'='*80}")
    print(f"Endpoint: {endpoint if endpoint else 'N/A (dry-run)'}")
    print(f"Bucket: vision-platform-prod")
    print(f"Framework: {args.framework}")
    print(f"Dry Run: {args.dry_run}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("\n[DRY RUN MODE] Listing models without uploading:\n")

        if args.framework in ["ultralytics", "all"]:
            print("\nUltralytics Models:")
            for model_name, info in ULTRALYTICS_MODEL_REGISTRY.items():
                if info.get("pretrained_available"):
                    print(f"  - {model_name}: {info.get('display_name')}")

        if args.framework in ["timm", "all"]:
            print("\nTimm Models:")
            for model_name, info in TIMM_MODEL_REGISTRY.items():
                if info.get("pretrained_available"):
                    print(f"  - {model_name}: {info.get('display_name')}")

        return

    # Upload weights
    all_success = True

    if args.framework in ["ultralytics", "all"]:
        if not upload_ultralytics_weights():
            all_success = False

    if args.framework in ["timm", "all"]:
        if not upload_timm_weights():
            all_success = False

    if all_success:
        print("\n✅ All pretrained weights uploaded successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some uploads failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

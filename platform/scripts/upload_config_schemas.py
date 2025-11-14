#!/usr/bin/env python3
"""
Upload Trainer Configuration Schemas to S3/R2

This script auto-discovers trainers in platform/trainers/ and uploads
their config schemas to object storage (S3/R2).

Usage:
    python upload_config_schemas.py --all --dry-run  # Validate all schemas
    python upload_config_schemas.py --framework ultralytics  # Upload single schema
    python upload_config_schemas.py --all  # Upload all schemas

Environment Variables:
    AWS_S3_ENDPOINT_URL: S3-compatible endpoint (MinIO/R2)
    AWS_ACCESS_KEY_ID: Access key
    AWS_SECRET_ACCESS_KEY: Secret key
    S3_BUCKET_RESULTS: Results bucket name (default: vision-platform-results)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError


def discover_trainers(trainers_dir: Path) -> List[str]:
    """
    Auto-discover trainers with config_schema.py.

    Args:
        trainers_dir: Path to platform/trainers/ directory

    Returns:
        List of framework names
    """
    frameworks = []

    if not trainers_dir.exists():
        print(f"[ERROR] Trainers directory not found: {trainers_dir}")
        return frameworks

    for trainer_dir in trainers_dir.iterdir():
        if not trainer_dir.is_dir():
            continue

        schema_file = trainer_dir / "config_schema.py"
        if schema_file.exists():
            frameworks.append(trainer_dir.name)
            print(f"[OK] Found trainer: {trainer_dir.name}")

    return frameworks


def extract_schema(framework: str, trainers_dir: Path) -> Dict[str, Any]:
    """
    Extract configuration schema from trainer.

    Args:
        framework: Framework name (e.g., 'ultralytics')
        trainers_dir: Path to platform/trainers/ directory

    Returns:
        Configuration schema dict

    Raises:
        ImportError: If config_schema module not found
        AttributeError: If get_config_schema() function not found
    """
    trainer_path = trainers_dir / framework

    # Add trainer directory to Python path
    sys.path.insert(0, str(trainer_path))

    try:
        # Import config_schema module
        import config_schema

        # Call get_config_schema() function
        if not hasattr(config_schema, 'get_config_schema'):
            raise AttributeError(
                f"{framework}/config_schema.py must have get_config_schema() function"
            )

        schema = config_schema.get_config_schema()

        # Validate schema structure
        required_keys = ['framework', 'description', 'version', 'fields', 'presets']
        for key in required_keys:
            if key not in schema:
                raise ValueError(f"Schema missing required key: {key}")

        print(f"[OK] Extracted schema for {framework}")
        print(f"  - Fields: {len(schema['fields'])}")
        print(f"  - Presets: {list(schema['presets'].keys())}")

        return schema

    except Exception as e:
        print(f"[ERROR] Failed to extract schema for {framework}: {e}")
        raise

    finally:
        # Remove from Python path
        sys.path.pop(0)


def upload_schema_to_s3(
    schema: Dict[str, Any],
    framework: str,
    bucket: str,
    s3_client: boto3.client,
    dry_run: bool = False
) -> bool:
    """
    Upload schema to Internal Storage (Results MinIO).

    Schemas are stored directly in config-schemas bucket (no schemas/ prefix).

    Args:
        schema: Configuration schema dict
        framework: Framework name
        bucket: S3 bucket name (e.g., config-schemas)
        s3_client: Boto3 S3 client
        dry_run: If True, only validate without uploading

    Returns:
        True if successful, False otherwise
    """
    # Schema is stored directly in config-schemas bucket (no schemas/ prefix)
    s3_key = f"{framework}.json"
    schema_json = json.dumps(schema, indent=2)

    if dry_run:
        print(f"[DRY RUN] Would upload to s3://{bucket}/{s3_key}")
        print(f"[DRY RUN] Schema size: {len(schema_json)} bytes")
        return True

    try:
        # Upload to Internal Storage
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=schema_json,
            ContentType='application/json',
            CacheControl='max-age=300',  # 5 minutes cache
        )

        print(f"[OK] Uploaded schema to s3://{bucket}/{s3_key}")
        return True

    except ClientError as e:
        print(f"[ERROR] Failed to upload schema: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Upload trainer configuration schemas to S3/R2'
    )
    parser.add_argument(
        '--framework',
        type=str,
        help='Framework name to upload (e.g., ultralytics)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Upload all discovered trainers'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate schemas without uploading'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default=os.getenv('INTERNAL_BUCKET_SCHEMAS', 'config-schemas'),
        help='Internal Storage bucket name (default: config-schemas)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.framework and not args.all:
        parser.error("Either --framework or --all must be specified")

    # Get paths
    script_dir = Path(__file__).parent
    platform_dir = script_dir.parent
    trainers_dir = platform_dir / "trainers"

    print("=" * 80)
    print("Config Schema Upload Script")
    print("=" * 80)
    print(f"Platform directory: {platform_dir}")
    print(f"Trainers directory: {trainers_dir}")
    print(f"Target bucket: {args.bucket}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    # Discover trainers
    if args.all:
        frameworks = discover_trainers(trainers_dir)
        if not frameworks:
            print("[ERROR] No trainers found with config_schema.py")
            sys.exit(1)
    else:
        frameworks = [args.framework]

    # Initialize S3 client for Internal Storage (Results MinIO - port 9002)
    # Config schemas are stored in Internal Storage (same as checkpoints, weights)
    s3_endpoint = os.getenv('INTERNAL_STORAGE_ENDPOINT') or os.getenv('AWS_S3_ENDPOINT_URL')
    s3_access_key = os.getenv('INTERNAL_STORAGE_ACCESS_KEY') or os.getenv('AWS_ACCESS_KEY_ID')
    s3_secret_key = os.getenv('INTERNAL_STORAGE_SECRET_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('INTERNAL_BUCKET_SCHEMAS', args.bucket)

    if not args.dry_run:
        if not all([s3_endpoint, s3_access_key, s3_secret_key]):
            print("[ERROR] Missing Internal Storage credentials in environment variables:")
            print("   - INTERNAL_STORAGE_ENDPOINT (or AWS_S3_ENDPOINT_URL)")
            print("   - INTERNAL_STORAGE_ACCESS_KEY (or AWS_ACCESS_KEY_ID)")
            print("   - INTERNAL_STORAGE_SECRET_KEY (or AWS_SECRET_ACCESS_KEY)")
            print("")
            print("For Tier-0 development, config schemas are stored in:")
            print("  Internal Storage (Results MinIO): http://localhost:9002")
            print("  Bucket: config-schemas")
            sys.exit(1)

        print(f"[INFO] Using Internal Storage: {s3_endpoint}")
        print(f"[INFO] Target bucket: {bucket_name}")

        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )

        # Check bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"[OK] Bucket exists: {bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"[ERROR] Bucket not found: {bucket_name}")
                print(f"[ERROR] Make sure Internal Storage (Results MinIO) is running on port 9002")
                sys.exit(1)
            else:
                print(f"[ERROR] Cannot access bucket: {e}")
                sys.exit(1)

        # Update args.bucket to use the detected bucket name
        args.bucket = bucket_name
    else:
        s3_client = None

    # Process each framework
    success_count = 0
    fail_count = 0

    for framework in frameworks:
        print(f"\nProcessing {framework}...")

        try:
            # Extract schema
            schema = extract_schema(framework, trainers_dir)

            # Upload schema
            if args.dry_run or upload_schema_to_s3(
                schema, framework, args.bucket, s3_client, args.dry_run
            ):
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[ERROR] Error processing {framework}: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"[OK] Success: {success_count}/{len(frameworks)}")
    print(f"[ERROR] Failed: {fail_count}/{len(frameworks)}")

    if fail_count > 0:
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] No schemas were uploaded (validation only)")
    else:
        print("\n[OK] All schemas uploaded successfully!")


if __name__ == "__main__":
    main()

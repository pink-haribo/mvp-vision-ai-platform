#!/usr/bin/env python3
"""
Upload training configuration schemas to storage.

This script extracts configuration schemas from each training framework
and uploads them to MinIO/S3 storage for Backend consumption.

Usage:
    python upload_schema_to_storage.py [--framework FRAMEWORK] [--all]

Examples:
    python upload_schema_to_storage.py --framework ultralytics
    python upload_schema_to_storage.py --all
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_schemas import get_ultralytics_schema, get_timm_schema
import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_s3_client():
    """Create S3 client for MinIO/S3."""
    endpoint_url = os.getenv('AWS_S3_ENDPOINT_URL')
    access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not all([endpoint_url, access_key_id, secret_access_key]):
        raise ValueError(
            "Storage credentials not set. Required:\n"
            "  AWS_S3_ENDPOINT_URL\n"
            "  AWS_ACCESS_KEY_ID\n"
            "  AWS_SECRET_ACCESS_KEY"
        )

    logger.info(f"Connecting to storage: {endpoint_url}")

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4')
    )


def extract_schema(framework: str) -> dict:
    """Extract schema from framework."""
    logger.info(f"Extracting schema for framework: {framework}")

    if framework == 'ultralytics':
        schema = get_ultralytics_schema()
    elif framework == 'timm':
        schema = get_timm_schema()
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Convert to dictionary
    schema_dict = {
        'framework': framework,
        'description': f"{framework.title()} Training Configuration",
        'version': '1.0',
        'fields': [field.to_dict() for field in schema.fields],
        'presets': schema.presets
    }

    logger.info(f"  - Extracted {len(schema_dict['fields'])} fields")
    logger.info(f"  - Extracted {len(schema_dict['presets'])} presets")

    return schema_dict


def upload_schema(s3_client, framework: str, schema_dict: dict, bucket: str):
    """Upload schema to storage."""
    schema_key = f"schemas/{framework}.json"
    schema_json = json.dumps(schema_dict, indent=2)

    logger.info(f"Uploading schema to {bucket}/{schema_key}")

    s3_client.put_object(
        Bucket=bucket,
        Key=schema_key,
        Body=schema_json.encode('utf-8'),
        ContentType='application/json'
    )

    logger.info(f"✓ Successfully uploaded {framework} schema ({len(schema_json)} bytes)")


def save_local_copy(framework: str, schema_dict: dict, output_dir: Path):
    """Save local copy for reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{framework}-schema.json"

    with open(output_file, 'w') as f:
        json.dump(schema_dict, f, indent=2)

    logger.info(f"✓ Saved local copy: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Upload training schemas to storage')
    parser.add_argument(
        '--framework',
        choices=['ultralytics', 'timm', 'huggingface'],
        help='Specific framework to upload'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Upload all frameworks'
    )
    parser.add_argument(
        '--bucket',
        default=os.getenv('S3_BUCKET_RESULTS', 'training-results'),
        help='S3 bucket name (default: training-results)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'schemas',
        help='Local output directory for schema copies'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Extract schemas but do not upload'
    )

    args = parser.parse_args()

    # Determine frameworks to process
    if args.all:
        frameworks = ['ultralytics', 'timm']
    elif args.framework:
        frameworks = [args.framework]
    else:
        parser.error("Must specify --framework or --all")

    logger.info("=" * 60)
    logger.info("Schema Upload Tool")
    logger.info("=" * 60)
    logger.info(f"Frameworks: {', '.join(frameworks)}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Get S3 client (unless dry-run)
    s3_client = None if args.dry_run else get_s3_client()

    # Process each framework
    success_count = 0
    for framework in frameworks:
        try:
            logger.info("")
            logger.info(f"Processing: {framework}")
            logger.info("-" * 60)

            # Extract schema
            schema_dict = extract_schema(framework)

            # Save local copy
            save_local_copy(framework, schema_dict, args.output_dir)

            # Upload to storage
            if not args.dry_run:
                upload_schema(s3_client, framework, schema_dict, args.bucket)
            else:
                logger.info(f"[DRY RUN] Would upload to {args.bucket}/schemas/{framework}.json")

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to process {framework}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Upload Summary: {success_count}/{len(frameworks)} successful")
    logger.info("=" * 60)

    if success_count < len(frameworks):
        sys.exit(1)


if __name__ == '__main__':
    main()

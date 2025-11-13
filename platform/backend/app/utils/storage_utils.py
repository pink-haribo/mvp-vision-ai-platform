"""
Storage utilities for object storage management.

Provides centralized storage client management for both local (MinIO)
and production (Cloudflare R2) environments.

Both MinIO and R2 are S3-compatible, so we use the same boto3 client.
The only difference is the endpoint URL and credentials.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_storage_type() -> str:
    """
    Determine current storage type based on environment variables.

    Detection logic:
    1. Check AWS_S3_ENDPOINT_URL
    2. If contains 'r2.cloudflarestorage.com' -> 'r2'
    3. If contains 'localhost' or '127.0.0.1' -> 'minio'
    4. Otherwise -> 's3' (generic S3)

    Returns:
        Storage type string: 'r2', 'minio', or 's3'

    Raises:
        ValueError: If AWS_S3_ENDPOINT_URL is not set
    """
    endpoint_url = os.getenv('AWS_S3_ENDPOINT_URL')

    if not endpoint_url:
        raise ValueError(
            "AWS_S3_ENDPOINT_URL environment variable not set. "
            "Storage configuration requires explicit endpoint URL. "
            "Set AWS_S3_ENDPOINT_URL to either:\n"
            "  - MinIO (local): http://localhost:30900\n"
            "  - Cloudflare R2 (production): https://...r2.cloudflarestorage.com\n"
            "  - Generic S3: https://s3.amazonaws.com"
        )

    # Cloudflare R2
    if 'r2.cloudflarestorage.com' in endpoint_url:
        logger.info(f"[STORAGE] Detected Cloudflare R2: {endpoint_url}")
        return 'r2'

    # Local MinIO
    if 'localhost' in endpoint_url or '127.0.0.1' in endpoint_url or ':30900' in endpoint_url:
        logger.info(f"[STORAGE] Detected MinIO: {endpoint_url}")
        return 'minio'

    # Generic S3
    logger.info(f"[STORAGE] Using generic S3: {endpoint_url}")
    return 's3'


def get_storage_config() -> dict:
    """
    Get storage configuration from environment variables.

    Returns:
        Dictionary with storage configuration:
        - endpoint_url: S3 endpoint URL
        - access_key_id: Access key ID
        - secret_access_key: Secret access key
        - bucket_name: S3 bucket name
        - storage_type: Storage type ('r2', 'minio', 's3')

    Raises:
        ValueError: If required environment variables are not set
    """
    endpoint_url = os.getenv('AWS_S3_ENDPOINT_URL')
    access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('S3_BUCKET', 'vision-platform-dev')

    # Validate required variables
    missing_vars = []
    if not endpoint_url:
        missing_vars.append('AWS_S3_ENDPOINT_URL')
    if not access_key_id:
        missing_vars.append('AWS_ACCESS_KEY_ID')
    if not secret_access_key:
        missing_vars.append('AWS_SECRET_ACCESS_KEY')

    if missing_vars:
        raise ValueError(
            f"Missing required storage environment variables: {', '.join(missing_vars)}\n"
            f"Please set the following in your .env file:\n"
            f"  AWS_S3_ENDPOINT_URL=<endpoint>\n"
            f"  AWS_ACCESS_KEY_ID=<access_key>\n"
            f"  AWS_SECRET_ACCESS_KEY=<secret_key>\n"
            f"  S3_BUCKET=<bucket_name>\n\n"
            f"For local development (MinIO):\n"
            f"  AWS_S3_ENDPOINT_URL=http://localhost:30900\n"
            f"  AWS_ACCESS_KEY_ID=minioadmin\n"
            f"  AWS_SECRET_ACCESS_KEY=minioadmin\n"
            f"  S3_BUCKET=vision-platform-dev\n\n"
            f"For production (Cloudflare R2):\n"
            f"  AWS_S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com\n"
            f"  AWS_ACCESS_KEY_ID=<your_r2_access_key>\n"
            f"  AWS_SECRET_ACCESS_KEY=<your_r2_secret_key>\n"
            f"  S3_BUCKET=vision-platform-prod"
        )

    # Determine storage type
    storage_type = get_storage_type()

    config = {
        'endpoint_url': endpoint_url,
        'access_key_id': access_key_id,
        'secret_access_key': secret_access_key,
        'bucket_name': bucket_name,
        'storage_type': storage_type
    }

    logger.info(
        f"[STORAGE] Configuration loaded:\n"
        f"  Type: {storage_type}\n"
        f"  Endpoint: {endpoint_url}\n"
        f"  Bucket: {bucket_name}"
    )

    return config


def get_storage_client():
    """
    Get S3-compatible storage client instance.

    This function imports and returns the global storage client.
    The client is configured based on environment variables.

    Returns:
        S3Storage instance configured for current environment

    Raises:
        ValueError: If storage configuration is invalid
    """
    from app.utils.s3_storage import s3_storage

    if s3_storage.client is None:
        raise ValueError(
            "Storage client not initialized. "
            "This usually means storage environment variables are not set correctly. "
            "Check AWS_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY."
        )

    return s3_storage

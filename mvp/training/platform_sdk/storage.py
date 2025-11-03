"""
Model Storage Management with Auto-Caching.

Provides smart model weight management with fallback strategy:
1. Local cache
2. R2 (platform shared cache)
3. Original source (auto-upload to R2 for future use)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable
import warnings


def get_model_weights(
    model_name: str,
    framework: str,
    download_fn: Callable[[], str],
    file_extension: str = "pt"
) -> str:
    """
    Get model weights with auto-caching and fallback.

    Priority:
    1. Local cache (/workspace/data/.cache/models/)
    2. R2 (platform shared cache)
    3. Original source (Ultralytics, timm, HuggingFace)
       → Auto-upload to R2 on success

    Args:
        model_name: Model name (e.g., "yolo11n")
        framework: Framework name (e.g., "ultralytics", "timm")
        download_fn: Function to download from original source
        file_extension: File extension (default: "pt")

    Returns:
        Local path to model weights

    Raises:
        FileNotFoundError: If all download methods fail
    """
    cache_dir = Path("/workspace/data/.cache/models") / framework
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / f"{model_name}.{file_extension}"

    # 1. Check local cache
    if local_path.exists():
        print(f"[CACHE] Using local cache: {local_path}")
        sys.stdout.flush()
        return str(local_path)

    # 2. Try R2
    print(f"[R2] Checking R2 for {framework}/{model_name}...")
    sys.stdout.flush()

    r2_path = _try_download_from_r2(model_name, framework, local_path, file_extension)
    if r2_path:
        print(f"[R2] Downloaded from R2: {r2_path}")
        sys.stdout.flush()
        return r2_path

    # 3. Download from original source
    print(f"[ORIGINAL] Not found in R2, downloading from original source...")
    sys.stdout.flush()

    try:
        # Call original download function
        downloaded_path = download_fn()

        if not downloaded_path or not Path(downloaded_path).exists():
            raise FileNotFoundError(f"Download function returned invalid path: {downloaded_path}")

        print(f"[ORIGINAL] Download successful: {downloaded_path}")
        sys.stdout.flush()

        # Copy to local cache
        import shutil
        shutil.copy(downloaded_path, local_path)
        print(f"[CACHE] Copied to local cache: {local_path}")
        sys.stdout.flush()

        # Auto-upload to R2 for future use
        _upload_to_r2(local_path, model_name, framework, file_extension)

        return str(local_path)

    except Exception as e:
        print(f"[ERROR] Failed to download from original source: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        raise FileNotFoundError(
            f"Could not download model '{model_name}' from any source. "
            f"Tried: local cache, R2, original source."
        )


def _try_download_from_r2(
    model_name: str,
    framework: str,
    dest_path: Path,
    file_extension: str
) -> Optional[str]:
    """
    Try to download model from R2.

    Returns:
        Local path if successful, None otherwise
    """
    try:
        import boto3

        # Check if R2 credentials are available
        endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        if not all([endpoint, access_key, secret_key]):
            print(f"[R2] R2 credentials not configured, skipping")
            sys.stdout.flush()
            return None

        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        bucket = 'vision-platform-prod'
        key = f'models/pretrained/{framework}/{model_name}.{file_extension}'

        print(f"[R2] Downloading from s3://{bucket}/{key}...")
        sys.stdout.flush()

        s3.download_file(bucket, key, str(dest_path))

        print(f"[R2] Download successful")
        sys.stdout.flush()

        return str(dest_path)

    except Exception as e:
        print(f"[R2] Not found in R2 or download failed: {e}")
        sys.stdout.flush()
        return None


def _upload_to_r2(
    local_path: Path,
    model_name: str,
    framework: str,
    file_extension: str
):
    """
    Upload model to R2 for future users.

    Non-blocking: Warns on failure but doesn't crash.
    """
    try:
        import boto3

        # Check if R2 credentials are available
        endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        if not all([endpoint, access_key, secret_key]):
            print(f"[R2] R2 credentials not configured, skipping upload")
            sys.stdout.flush()
            return

        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        bucket = 'vision-platform-prod'
        key = f'models/pretrained/{framework}/{model_name}.{file_extension}'

        print(f"[R2] Uploading to R2 for future use: s3://{bucket}/{key}...")
        sys.stdout.flush()

        s3.upload_file(str(local_path), bucket, key)

        print(f"[R2] Upload successful! Future users will download from R2.")
        sys.stdout.flush()

    except Exception as e:
        # Don't fail training just because upload failed
        warnings.warn(f"[R2 WARNING] Failed to upload to R2: {e}", UserWarning)
        print(f"[R2 WARNING] Upload failed (non-critical): {e}")
        sys.stdout.flush()

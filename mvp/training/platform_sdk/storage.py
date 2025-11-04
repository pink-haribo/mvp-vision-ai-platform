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


# ========== Dataset Management ==========

def get_dataset(
    dataset_id: str,
    download_fn: Optional[Callable[[], str]] = None
) -> str:
    """
    Get dataset with auto-caching and fallback.

    Priority:
    1. Local cache (/workspace/data/.cache/datasets/)
    2. R2 (platform shared cache)
    3. Original source (if download_fn provided)
       → Auto-upload to R2 on success

    Args:
        dataset_id: Dataset identifier (e.g., "coco8", "user123_custom_dataset")
        download_fn: Optional function to download dataset (returns path to zip file)

    Returns:
        Local path to extracted dataset directory

    Raises:
        FileNotFoundError: If all download methods fail

    Example:
        >>> def download_custom_dataset():
        ...     # Download from external source
        ...     return "/tmp/my_dataset.zip"
        >>> dataset_path = get_dataset("my_dataset", download_fn=download_custom_dataset)
        >>> # dataset_path: "/workspace/data/.cache/datasets/my_dataset"
    """
    cache_dir = Path("/workspace/data/.cache/datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = cache_dir / dataset_id

    # 1. Check local cache (already extracted)
    if dataset_dir.exists() and _is_valid_dataset(dataset_dir):
        print(f"[CACHE] Using local cached dataset: {dataset_dir}")
        sys.stdout.flush()
        return str(dataset_dir)

    # 2. Try R2
    print(f"[R2] Checking R2 for dataset: {dataset_id}...")
    sys.stdout.flush()

    r2_path = _try_download_dataset_from_r2(dataset_id, dataset_dir)
    if r2_path:
        print(f"[R2] Dataset downloaded and extracted: {r2_path}")
        sys.stdout.flush()
        return r2_path

    # 3. Download from original source
    if download_fn is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found in local cache or R2, "
            f"and no download function provided."
        )

    print(f"[ORIGINAL] Not found in R2, downloading from original source...")
    sys.stdout.flush()

    try:
        # Call original download function
        zip_path = download_fn()

        if not zip_path or not Path(zip_path).exists():
            raise FileNotFoundError(f"Download function returned invalid path: {zip_path}")

        print(f"[ORIGINAL] Download successful: {zip_path}")
        sys.stdout.flush()

        # Extract to local cache
        _extract_dataset(zip_path, dataset_dir)
        print(f"[CACHE] Extracted to local cache: {dataset_dir}")
        sys.stdout.flush()

        # Auto-upload to R2 for future use
        _upload_dataset_to_r2(zip_path, dataset_id)

        return str(dataset_dir)

    except Exception as e:
        print(f"[ERROR] Failed to download from original source: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        raise FileNotFoundError(
            f"Could not download dataset '{dataset_id}' from any source. "
            f"Tried: local cache, R2, original source."
        )


def _is_valid_dataset(dataset_dir: Path) -> bool:
    """
    Check if dataset directory contains valid data.

    Returns:
        True if dataset appears valid (has images or labels)
    """
    # Check for common dataset structure indicators
    indicators = [
        dataset_dir / "images",
        dataset_dir / "labels",
        dataset_dir / "train",
        dataset_dir / "val",
        dataset_dir / "annotations",
    ]

    for indicator in indicators:
        if indicator.exists():
            return True

    # Fallback: check if directory is not empty
    try:
        return any(dataset_dir.iterdir())
    except:
        return False


def _try_download_dataset_from_r2(
    dataset_id: str,
    dest_dir: Path
) -> Optional[str]:
    """
    Try to download and extract dataset from R2.

    Supports two formats:
    1. Zip file: datasets/{dataset_id}.zip
    2. Directory: datasets/{dataset_id}/ (with multiple files)

    Returns:
        Local directory path if successful, None otherwise
    """
    try:
        import boto3
        import zipfile
        import tempfile

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

        # Try 1: Check for zip file
        zip_key = f'datasets/{dataset_id}.zip'
        print(f"[R2] Trying zip file: s3://{bucket}/{zip_key}...")
        sys.stdout.flush()

        try:
            # Check if zip file exists
            s3.head_object(Bucket=bucket, Key=zip_key)

            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_path = tmp_file.name

            try:
                s3.download_file(bucket, zip_key, tmp_path)
                print(f"[R2] Zip download successful, extracting...")
                sys.stdout.flush()

                # Extract to destination
                _extract_dataset(tmp_path, dest_dir)

                print(f"[R2] Extraction successful")
                sys.stdout.flush()

                return str(dest_dir)

            finally:
                # Clean up temporary file
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()

        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Zip not found, try directory structure
                print(f"[R2] Zip not found, trying directory structure...")
                sys.stdout.flush()
            else:
                raise

        # Try 2: Check for directory structure
        dir_prefix = f'datasets/{dataset_id}/'
        print(f"[R2] Trying directory: s3://{bucket}/{dir_prefix}...")
        sys.stdout.flush()

        # List all objects with this prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=dir_prefix)

        if 'Contents' not in response or len(response['Contents']) == 0:
            print(f"[R2] No files found in directory")
            sys.stdout.flush()
            return None

        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Download all files
        total_files = len(response['Contents'])
        print(f"[R2] Found {total_files} files, downloading...")
        sys.stdout.flush()

        for idx, obj in enumerate(response['Contents'], 1):
            key = obj['Key']
            # Get relative path (remove prefix)
            relative_path = key[len(dir_prefix):]

            if not relative_path:  # Skip directory marker
                continue

            local_file = dest_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"[R2] [{idx}/{total_files}] Downloading {relative_path}...")
            sys.stdout.flush()

            s3.download_file(bucket, key, str(local_file))

        print(f"[R2] Directory download successful: {dest_dir}")
        sys.stdout.flush()

        return str(dest_dir)

    except Exception as e:
        print(f"[R2] Download failed: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return None


def _extract_dataset(zip_path: str, dest_dir: Path):
    """
    Extract dataset zip file to destination directory.

    Args:
        zip_path: Path to zip file
        dest_dir: Destination directory

    Raises:
        Exception: If extraction fails
    """
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT] Extracting {zip_path} to {dest_dir}...")
    sys.stdout.flush()

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    print(f"[EXTRACT] Extraction complete")
    sys.stdout.flush()


def _upload_dataset_to_r2(
    zip_path: str,
    dataset_id: str
):
    """
    Upload dataset to R2 for future users.

    Non-blocking: Warns on failure but doesn't crash.

    Args:
        zip_path: Path to dataset zip file
        dataset_id: Dataset identifier
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
        key = f'datasets/{dataset_id}.zip'

        print(f"[R2] Uploading to R2 for future use: s3://{bucket}/{key}...")
        sys.stdout.flush()

        s3.upload_file(zip_path, bucket, key)

        print(f"[R2] Upload successful! Future users will download from R2.")
        sys.stdout.flush()

    except Exception as e:
        # Don't fail training just because upload failed
        warnings.warn(f"[R2 WARNING] Failed to upload dataset to R2: {e}", UserWarning)
        print(f"[R2 WARNING] Dataset upload failed (non-critical): {e}")
        sys.stdout.flush()

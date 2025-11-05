"""
R2 Dataset Downloader for Training Service.

Downloads and extracts datasets from Cloudflare R2 storage before training.
"""

import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
import zipfile
import tempfile
import shutil


class R2Downloader:
    """Cloudflare R2 downloader for training datasets."""

    def __init__(self):
        """Initialize R2 client with credentials from environment."""
        self.endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET", "vision-platform-dev")

        if not all([self.endpoint_url, self.access_key_id, self.secret_access_key]):
            print(f"[R2Downloader] WARNING: R2 credentials not configured")
            self.client = None
            return

        # Initialize S3 client (R2 is S3-compatible)
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version="s3v4"),
        )

        print(f"[R2Downloader] Initialized: bucket={self.bucket_name}")

    def download_and_extract_dataset(
        self,
        dataset_id: str,
        extract_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Download dataset zip from R2 and extract it.

        Args:
            dataset_id: Dataset identifier (UUID or simple ID)
            extract_dir: Directory to extract to (default: temp dir)

        Returns:
            Path to extracted dataset directory, or None if failed
        """
        if not self.client:
            print(f"[R2Downloader] ERROR: R2 client not initialized")
            return None

        # Create temporary directory for zip file
        temp_dir = Path(tempfile.mkdtemp(prefix="r2_dataset_"))
        zip_path = temp_dir / f"{dataset_id}.zip"

        try:
            # Download zip file from R2
            object_key = f"datasets/{dataset_id}.zip"
            print(f"[R2Downloader] Downloading: {object_key}")

            self.client.download_file(
                self.bucket_name,
                object_key,
                str(zip_path)
            )

            print(f"[R2Downloader] Downloaded: {zip_path} ({zip_path.stat().st_size / 1024:.2f} KB)")

            # Determine extraction directory
            if extract_dir is None:
                extract_dir = Path(tempfile.mkdtemp(prefix=f"dataset_{dataset_id}_"))
            else:
                extract_dir = Path(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract zip file
            print(f"[R2Downloader] Extracting to: {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # List extracted files
            extracted_files = list(extract_dir.rglob("*"))
            print(f"[R2Downloader] Extracted {len(extracted_files)} files/directories")

            # Clean up zip file
            zip_path.unlink()
            temp_dir.rmdir()

            print(f"[R2Downloader] ✓ Dataset ready: {extract_dir}")
            return extract_dir

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                print(f"[R2Downloader] ERROR: Dataset not found in R2: {object_key}")
            else:
                print(f"[R2Downloader] ERROR: Failed to download from R2: {e}")
            return None
        except zipfile.BadZipFile:
            print(f"[R2Downloader] ERROR: Downloaded file is not a valid zip: {zip_path}")
            return None
        except Exception as e:
            print(f"[R2Downloader] ERROR: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temp files if they still exist
            if zip_path.exists():
                zip_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

    def is_uuid_path(self, path: str) -> bool:
        """
        Check if a path looks like a UUID (dataset ID) rather than a local path.

        Args:
            path: Path string to check

        Returns:
            True if path looks like a UUID
        """
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        # Simple ID format: det-coco8, cls-imagenet, etc.
        path_str = str(path)

        # Check if it's a UUID (8-4-4-4-12 hex digits with hyphens)
        if len(path_str) == 36 and path_str.count('-') == 4:
            parts = path_str.split('-')
            expected_lengths = [8, 4, 4, 4, 12]
            if len(parts) == 5 and all(len(p) == expected_lengths[i] for i, p in enumerate(parts)):
                try:
                    # Try to parse as hex
                    all(int(p, 16) for p in parts)
                    return True
                except ValueError:
                    pass

        # Check if it's a simple ID (no path separators, short length)
        if '/' not in path_str and '\\' not in path_str and len(path_str) < 50:
            return True

        return False


def download_dataset_if_needed(dataset_path: str) -> str:
    """
    Download dataset from R2 if path is a UUID, otherwise return path as-is.

    Args:
        dataset_path: Dataset path or UUID

    Returns:
        Local dataset path (either original or downloaded path)
    """
    downloader = R2Downloader()

    # Check if path looks like a UUID or simple ID (not a local path)
    if downloader.is_uuid_path(dataset_path):
        print(f"[R2Downloader] Detected dataset ID (not local path): {dataset_path}")
        print(f"[R2Downloader] Attempting to download from R2...")

        # Download and extract
        extracted_path = downloader.download_and_extract_dataset(dataset_path)

        if extracted_path:
            return str(extracted_path)
        else:
            # If download fails, return original path (will fail later with clear error)
            print(f"[R2Downloader] WARNING: Download failed, using original path: {dataset_path}")
            return dataset_path
    else:
        # Local path, use as-is
        print(f"[R2Downloader] Using local path: {dataset_path}")
        return dataset_path

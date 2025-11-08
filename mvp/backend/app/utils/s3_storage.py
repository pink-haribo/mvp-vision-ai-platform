"""
S3-Compatible Storage Utility for Dataset Management.

Supports multiple S3-compatible storage backends:
- Cloudflare R2 (production)
- MinIO (local development)
- AWS S3 (generic)

Handles:
- Dataset upload (zip files)
- Dataset download
- File operations (upload, delete)
- Presigned URL generation
"""

import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional, BinaryIO
import logging
import zipfile
import tempfile
import json
from io import BytesIO

logger = logging.getLogger(__name__)


class S3Storage:
    """
    S3-compatible storage client for dataset management.

    Supports Cloudflare R2, MinIO, AWS S3, and other S3-compatible backends.
    """

    def __init__(self):
        """Initialize S3-compatible client with credentials from environment."""
        self.endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # Separate buckets for different purposes
        self.bucket_datasets = os.getenv("S3_BUCKET_DATASETS", "training-datasets")
        self.bucket_checkpoints = os.getenv("S3_BUCKET_CHECKPOINTS", "training-checkpoints")
        self.bucket_results = os.getenv("S3_BUCKET_RESULTS", "training-results")

        if not all([self.endpoint_url, self.access_key_id, self.secret_access_key]):
            logger.warning(
                "S3 storage credentials not configured. Storage operations will fail. "
                "Set AWS_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY."
            )
            self.client = None
            return

        # Initialize S3 client (works with R2, MinIO, S3, etc.)
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={
                    'max_attempts': 3,  # Reduce from default (unlimited)
                    'mode': 'standard'  # Use exponential backoff
                },
                connect_timeout=5,  # 5 seconds connection timeout
                read_timeout=10,    # 10 seconds read timeout
            ),
        )

        # Detect storage type for logging
        storage_type = "unknown"
        if "r2.cloudflarestorage.com" in self.endpoint_url:
            storage_type = "Cloudflare R2"
        elif "localhost" in self.endpoint_url or "127.0.0.1" in self.endpoint_url:
            storage_type = "MinIO (local)"
        elif "amazonaws.com" in self.endpoint_url:
            storage_type = "AWS S3"

        logger.info(f"S3 Storage initialized: type={storage_type}, buckets=[datasets={self.bucket_datasets}, checkpoints={self.bucket_checkpoints}, results={self.bucket_results}], endpoint={self.endpoint_url}")

    def upload_file(
        self,
        file_path: Path,
        object_key: str,
        content_type: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Upload a file to S3 storage.

        Args:
            file_path: Local file path
            object_key: S3 object key (e.g., "datasets/my-dataset.zip")
            content_type: MIME type (optional, auto-detected if not provided)
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return False

        bucket_name = bucket or self.bucket_datasets

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_file(
                str(file_path),
                bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded file to storage: {bucket_name}/{object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload file to storage: {e}")
            return False

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Upload a file object to storage.

        Args:
            file_obj: File-like object
            object_key: S3 object key
            content_type: MIME type
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return False

        bucket_name = bucket or self.bucket_datasets

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_fileobj(
                file_obj,
                bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded file object to storage: {bucket_name}/{object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload file object to storage: {e}")
            return False

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Upload bytes data to storage.

        Args:
            data: Bytes data to upload
            object_key: S3 object key
            content_type: MIME type

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return False

        try:
            # Wrap bytes in BytesIO
            file_obj = BytesIO(data)
            return self.upload_fileobj(file_obj, object_key, content_type)

        except Exception as e:
            logger.error(f"Failed to upload bytes to storage: {e}")
            return False

    def download_file(
        self,
        object_key: str,
        file_path: Path,
        bucket: Optional[str] = None,
    ) -> bool:
        """
        Download a file from storage.

        Args:
            object_key: S3 object key
            file_path: Local destination path
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return False

        bucket_name = bucket or self.bucket_datasets

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(
                bucket_name,
                object_key,
                str(file_path)
            )

            logger.info(f"Downloaded file from storage: {bucket_name}/{object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to download file from storage: {e}")
            return False

    def get_file_content(self, object_key: str, bucket: Optional[str] = None) -> Optional[bytes]:
        """
        Get file content from storage as bytes.

        Args:
            object_key: S3 object key
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            File content as bytes, or None if not found
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return None

        bucket_name = bucket or self.bucket_datasets

        try:
            response = self.client.get_object(
                Bucket=bucket_name,
                Key=object_key
            )

            content = response['Body'].read()
            logger.info(f"Retrieved file content from storage: {bucket_name}/{object_key} ({len(content)} bytes)")
            return content

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"File not found in storage: {object_key}")
            else:
                logger.error(f"Failed to get file content from storage: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting file content: {e}")
            return None

    def delete_file(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """
        Delete a file from storage.

        Args:
            object_key: S3 object key
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return False

        bucket_name = bucket or self.bucket_datasets

        try:
            self.client.delete_object(
                Bucket=bucket_name,
                Key=object_key
            )

            logger.info(f"Deleted file from storage: {bucket_name}/{object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete file from storage: {e}")
            return False

    def file_exists(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """
        Check if a file exists in storage.

        Args:
            object_key: S3 object key
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            True if file exists
        """
        if not self.client:
            return False

        bucket_name = bucket or self.bucket_datasets

        try:
            self.client.head_object(
                Bucket=bucket_name,
                Key=object_key
            )
            return True

        except ClientError:
            return False

    def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600,
        bucket: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.

        Args:
            object_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            Presigned URL or None if failed
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return None

        bucket_name = bucket or self.bucket_datasets

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": bucket_name,
                    "Key": object_key
                },
                ExpiresIn=expiration
            )

            logger.info(f"Generated presigned URL for {bucket_name}/{object_key}")
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    def upload_dataset_zip(
        self,
        zip_file_path: Path,
        dataset_id: str
    ) -> bool:
        """
        Upload a dataset zip file to storage.

        Args:
            zip_file_path: Path to dataset zip file
            dataset_id: Unique dataset identifier

        Returns:
            True if successful
        """
        object_key = f"datasets/{dataset_id}.zip"
        return self.upload_file(
            zip_file_path,
            object_key,
            content_type="application/zip"
        )

    def download_dataset_zip(
        self,
        dataset_id: str,
        dest_dir: Path
    ) -> Optional[Path]:
        """
        Download a dataset zip file from storage.

        Args:
            dataset_id: Dataset identifier
            dest_dir: Destination directory

        Returns:
            Path to downloaded zip file or None if failed
        """
        object_key = f"datasets/{dataset_id}.zip"
        zip_path = dest_dir / f"{dataset_id}.zip"

        if self.download_file(object_key, zip_path):
            return zip_path
        return None

    def validate_dice_format(self, zip_file_path: Path) -> tuple[bool, Optional[dict], Optional[str]]:
        """
        Validate that a zip file contains DICE Format dataset.

        Args:
            zip_file_path: Path to zip file

        Returns:
            (is_valid, metadata, error_message)
        """
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Check for required files
                namelist = zip_ref.namelist()

                # Must have annotations.json
                annotations_files = [f for f in namelist if f.endswith('annotations.json')]
                if not annotations_files:
                    return False, None, "Missing annotations.json"

                # Must have meta.json
                meta_files = [f for f in namelist if f.endswith('meta.json')]
                if not meta_files:
                    return False, None, "Missing meta.json"

                # Must have images/ directory
                image_files = [f for f in namelist if 'images/' in f and f.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'))]
                if not image_files:
                    return False, None, "No images found in images/ directory"

                # Parse meta.json
                meta_path = meta_files[0]
                with zip_ref.open(meta_path) as meta_file:
                    meta_data = json.load(meta_file)

                # Parse annotations.json (just to validate it's valid JSON)
                annotations_path = annotations_files[0]
                with zip_ref.open(annotations_path) as ann_file:
                    annotations_data = json.load(ann_file)

                # Validate format version
                if annotations_data.get("format_version") not in ["1.0", "1.1"]:
                    return False, None, f"Unsupported format version: {annotations_data.get('format_version')}"

                # Return metadata
                metadata = {
                    "dataset_id": meta_data.get("dataset_id"),
                    "dataset_name": meta_data.get("dataset_name"),
                    "task_type": meta_data.get("task_type"),
                    "num_classes": meta_data.get("num_classes"),
                    "total_images": meta_data.get("total_images"),
                    "format_version": annotations_data.get("format_version"),
                    "content_hash": meta_data.get("content_hash"),
                }

                return True, metadata, None

        except zipfile.BadZipFile:
            return False, None, "Invalid zip file"
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON in metadata: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"

    def upload_image(
        self,
        file_obj,
        dataset_id: str,
        image_filename: str,
        content_type: str = "image/jpeg"
    ) -> bool:
        """
        Upload an individual image to storage.

        Args:
            file_obj: File-like object
            dataset_id: Dataset identifier
            image_filename: Image filename (e.g., "000001.jpg")
            content_type: MIME type

        Returns:
            True if successful
        """
        object_key = f"datasets/{dataset_id}/images/{image_filename}"
        return self.upload_fileobj(
            file_obj,
            object_key,
            content_type=content_type
        )

    def list_images(
        self,
        dataset_id: str,
        prefix: str = "images/",
        bucket: Optional[str] = None,
    ) -> list[str]:
        """
        List all images in a dataset.

        Args:
            dataset_id: Dataset identifier
            prefix: Prefix within dataset (default: "images/")
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            List of image keys (relative to dataset root)
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return []

        bucket_name = bucket or self.bucket_datasets

        try:
            full_prefix = f"datasets/{dataset_id}/{prefix}"
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=full_prefix
            )

            images = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract relative path (remove "datasets/{dataset_id}/")
                    key = obj['Key']
                    relative_key = key.replace(f"datasets/{dataset_id}/", "")
                    images.append(relative_key)

            logger.info(f"Listed {len(images)} images for dataset {dataset_id}")
            return images

        except Exception as e:
            logger.error(f"Failed to list images for {dataset_id}: {str(e)}")
            return []

    def delete_all_with_prefix(self, prefix: str, bucket: Optional[str] = None) -> int:
        """
        Delete all objects with a given prefix.

        Args:
            prefix: Prefix to filter objects (e.g., "datasets/abc-123/")
            bucket: Bucket name (defaults to datasets bucket)

        Returns:
            Number of objects deleted
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return 0

        bucket_name = bucket or self.bucket_datasets

        try:
            deleted_count = 0
            continuation_token = None

            while True:
                # List objects with prefix
                list_params = {
                    'Bucket': bucket_name,
                    'Prefix': prefix
                }
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token

                response = self.client.list_objects_v2(**list_params)

                # Delete objects if any found
                if 'Contents' in response:
                    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]

                    if objects_to_delete:
                        delete_response = self.client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
                        deleted_count += len(delete_response.get('Deleted', []))

                # Check if there are more objects to delete
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

            logger.info(f"Deleted {deleted_count} objects from {bucket_name} with prefix: {prefix}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete objects with prefix {prefix}: {str(e)}")
            return 0


# Global S3-compatible storage client instance
# Works with Cloudflare R2, MinIO, AWS S3, etc.
s3_storage = S3Storage()

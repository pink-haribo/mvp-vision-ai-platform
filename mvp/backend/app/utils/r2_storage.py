"""
Cloudflare R2 Storage Utility for Dataset Management.

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

logger = logging.getLogger(__name__)


class R2Storage:
    """Cloudflare R2 Storage client for dataset management."""

    def __init__(self):
        """Initialize R2 client with credentials from environment."""
        self.endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET", "vision-platform-dev")

        if not all([self.endpoint_url, self.access_key_id, self.secret_access_key]):
            logger.warning("R2 credentials not configured. R2 operations will fail.")
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

        logger.info(f"R2 Storage initialized: bucket={self.bucket_name}")

    def upload_file(
        self,
        file_path: Path,
        object_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Upload a file to R2.

        Args:
            file_path: Local file path
            object_key: R2 object key (e.g., "datasets/my-dataset.zip")
            content_type: MIME type (optional, auto-detected if not provided)

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return False

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_file(
                str(file_path),
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded file to R2: {object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload file to R2: {e}")
            return False

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Upload a file object to R2.

        Args:
            file_obj: File-like object
            object_key: R2 object key
            content_type: MIME type

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return False

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded file object to R2: {object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload file object to R2: {e}")
            return False

    def download_file(
        self,
        object_key: str,
        file_path: Path
    ) -> bool:
        """
        Download a file from R2.

        Args:
            object_key: R2 object key
            file_path: Local destination path

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return False

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(
                self.bucket_name,
                object_key,
                str(file_path)
            )

            logger.info(f"Downloaded file from R2: {object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to download file from R2: {e}")
            return False

    def delete_file(self, object_key: str) -> bool:
        """
        Delete a file from R2.

        Args:
            object_key: R2 object key

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return False

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )

            logger.info(f"Deleted file from R2: {object_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete file from R2: {e}")
            return False

    def file_exists(self, object_key: str) -> bool:
        """
        Check if a file exists in R2.

        Args:
            object_key: R2 object key

        Returns:
            True if file exists
        """
        if not self.client:
            return False

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True

        except ClientError:
            return False

    def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.

        Args:
            object_key: R2 object key
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL or None if failed
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return None

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": object_key
                },
                ExpiresIn=expiration
            )

            logger.info(f"Generated presigned URL for {object_key}")
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
        Upload a dataset zip file to R2.

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
        Download a dataset zip file from R2.

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

    def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for downloading an object from R2.

        Args:
            object_key: R2 object key (e.g., "datasets/{id}/images/000001.jpg")
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL string or None if failed
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return None

        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL for: {object_key} (expires in {expiration}s)")
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {object_key}: {str(e)}")
            return None

    def upload_image(
        self,
        file_obj,
        dataset_id: str,
        image_filename: str,
        content_type: str = "image/jpeg"
    ) -> bool:
        """
        Upload an individual image to R2.

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
        prefix: str = "images/"
    ) -> list[str]:
        """
        List all images in a dataset.

        Args:
            dataset_id: Dataset identifier
            prefix: Prefix within dataset (default: "images/")

        Returns:
            List of image keys (relative to dataset root)
        """
        if not self.client:
            logger.error("R2 client not initialized")
            return []

        try:
            full_prefix = f"datasets/{dataset_id}/{prefix}"
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
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


# Global R2 client instance
r2_storage = R2Storage()

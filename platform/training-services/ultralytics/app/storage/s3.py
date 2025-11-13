"""
S3 Storage Client

Handles all S3 operations (download datasets, upload checkpoints).
"""

import asyncio
from pathlib import Path
from typing import Optional

import aioboto3


class S3Client:
    """Async S3 client for MinIO/S3."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint = endpoint
        self.bucket = bucket
        self.session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    async def upload_file(self, local_path: Path, s3_key: str) -> None:
        """Upload a file to S3."""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
        ) as s3:
            await s3.upload_file(str(local_path), self.bucket, s3_key)

    async def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download a file from S3."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
        ) as s3:
            await s3.download_file(self.bucket, s3_key, str(local_path))

    async def download_directory(self, s3_prefix: str, local_dir: Path) -> None:
        """Download all files under an S3 prefix to a local directory."""
        local_dir.mkdir(parents=True, exist_ok=True)

        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
        ) as s3:
            # List all objects under prefix
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    # Get relative path
                    relative_path = s3_key.replace(s3_prefix, "", 1).lstrip("/")

                    if not relative_path:  # Skip if it's the prefix itself
                        continue

                    local_file = local_dir / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    await s3.download_file(self.bucket, s3_key, str(local_file))

    def get_s3_uri(self, key: str) -> str:
        """Get S3 URI for a key."""
        return f"s3://{self.bucket}/{key}"

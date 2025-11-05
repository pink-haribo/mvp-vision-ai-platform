"""
Dataset Images API - Individual image management for datasets.

This module provides APIs for:
- Uploading individual images to datasets
- Listing images in a dataset
- Generating presigned URLs for image access
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Path as PathParam
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import Dataset, User
from app.utils.r2_storage import r2_storage
from app.utils.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Response Models ====================

class ImageUploadResponse(BaseModel):
    """Response model for image upload"""
    status: str
    message: str
    image_path: Optional[str] = None
    dataset_id: str


class ImageInfo(BaseModel):
    """Information about a single image"""
    filename: str
    presigned_url: str
    size: Optional[int] = None


class ImageListResponse(BaseModel):
    """Response model for image list"""
    status: str
    dataset_id: str
    total_images: int
    images: List[ImageInfo]


class PresignedUrlResponse(BaseModel):
    """Response model for presigned URL"""
    status: str
    image_path: str
    presigned_url: str
    expires_in: int  # seconds


# ==================== API Endpoints ====================

@router.post("/{dataset_id}/images", response_model=ImageUploadResponse)
async def upload_image_to_dataset(
    dataset_id: str = PathParam(..., description="Dataset ID"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload an individual image to a dataset.

    **Use Case:**
    - Labeling tool: Upload unlabeled images
    - Dataset augmentation: Add more images to existing dataset

    **Storage Structure:**
    ```
    datasets/{dataset_id}/
    └── images/
        ├── {filename}.jpg
        ├── {filename}.png
        └── ...
    ```

    **Request:**
    - `file`: Image file (JPEG, PNG, etc.)

    **Response:**
    - `status`: "success" or "error"
    - `message`: Description
    - `image_path`: Relative path in R2 (e.g., "images/photo.jpg")
    - `dataset_id`: Dataset identifier
    """
    try:
        # 1. Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # 2. Check ownership permission
        if dataset.owner_id != current_user.id and dataset.visibility != 'public':
            logger.warning(f"User {current_user.id} attempted to upload to dataset {dataset_id} owned by {dataset.owner_id}")
            raise HTTPException(status_code=403, detail="Permission denied: You can only upload to your own datasets")

        # 3. Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return ImageUploadResponse(
                status="error",
                message=f"Invalid file type: {file.content_type}. Only images are allowed.",
                dataset_id=dataset_id
            )

        # 3. Generate safe filename
        filename = file.filename or "image.jpg"
        # TODO: Add unique suffix if filename already exists

        # 4. Upload to R2
        logger.info(f"Uploading image {filename} to dataset {dataset_id}")

        file_content = await file.read()
        from io import BytesIO
        file_obj = BytesIO(file_content)

        success = r2_storage.upload_image(
            file_obj=file_obj,
            dataset_id=dataset_id,
            image_filename=filename,
            content_type=file.content_type
        )

        if not success:
            return ImageUploadResponse(
                status="error",
                message="Failed to upload image to R2",
                dataset_id=dataset_id
            )

        # 5. Update dataset metadata (num_images)
        # TODO: Increment num_images count

        logger.info(f"Successfully uploaded {filename} to dataset {dataset_id}")

        return ImageUploadResponse(
            status="success",
            message=f"Image '{filename}' uploaded successfully",
            image_path=f"images/{filename}",
            dataset_id=dataset_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        return ImageUploadResponse(
            status="error",
            message=f"Upload failed: {str(e)}",
            dataset_id=dataset_id
        )


@router.get("/{dataset_id}/images", response_model=ImageListResponse)
async def list_dataset_images(
    dataset_id: str = PathParam(..., description="Dataset ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all images in a dataset with presigned URLs.

    **Use Case:**
    - Labeling tool: Display images for annotation
    - Dataset preview: Show dataset contents
    - Image gallery: Browse uploaded images

    **Response:**
    - `status`: "success" or "error"
    - `dataset_id`: Dataset identifier
    - `total_images`: Number of images
    - `images`: List of image information with presigned URLs

    **Presigned URL:**
    - Valid for 1 hour (3600 seconds)
    - Direct browser access to R2
    - No server load for image delivery
    """
    try:
        # 1. Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # 2. Check view permission (owner or public dataset)
        if dataset.owner_id != current_user.id and dataset.visibility != 'public':
            logger.warning(f"User {current_user.id} attempted to list images from dataset {dataset_id} owned by {dataset.owner_id}")
            raise HTTPException(status_code=403, detail="Permission denied: You can only view your own datasets or public datasets")

        # 3. List images from R2
        logger.info(f"Listing images for dataset {dataset_id}")
        image_keys = r2_storage.list_images(dataset_id, prefix="images/")

        # 3. Generate presigned URLs for each image
        images = []
        for image_key in image_keys:
            # Generate presigned URL
            presigned_url = r2_storage.generate_presigned_url(
                object_key=f"datasets/{dataset_id}/{image_key}",
                expiration=3600  # 1 hour
            )

            if presigned_url:
                # Extract just the filename
                filename = image_key.replace("images/", "")
                images.append(ImageInfo(
                    filename=filename,
                    presigned_url=presigned_url
                ))

        logger.info(f"Found {len(images)} images in dataset {dataset_id}")

        return ImageListResponse(
            status="success",
            dataset_id=dataset_id,
            total_images=len(images),
            images=images
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list images: {str(e)}")


@router.get("/{dataset_id}/images/{image_filename}/url", response_model=PresignedUrlResponse)
async def get_image_presigned_url(
    dataset_id: str = PathParam(..., description="Dataset ID"),
    image_filename: str = PathParam(..., description="Image filename"),
    expiration: int = 3600,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a presigned URL for a specific image.

    **Use Case:**
    - Labeling tool: Load single image for annotation
    - Image viewer: Display full-resolution image
    - Download: Allow direct download link

    **Parameters:**
    - `dataset_id`: Dataset identifier
    - `image_filename`: Image filename (e.g., "photo.jpg")
    - `expiration`: URL validity in seconds (default: 3600 = 1 hour)

    **Response:**
    - `status`: "success"
    - `image_path`: Relative path in dataset
    - `presigned_url`: Direct R2 URL (valid for specified time)
    - `expires_in`: Expiration time in seconds
    """
    try:
        # 1. Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # 2. Check view permission (owner or public dataset)
        if dataset.owner_id != current_user.id and dataset.visibility != 'public':
            logger.warning(f"User {current_user.id} attempted to access image from dataset {dataset_id} owned by {dataset.owner_id}")
            raise HTTPException(status_code=403, detail="Permission denied: You can only access your own datasets or public datasets")

        # 3. Generate presigned URL
        image_path = f"images/{image_filename}"
        object_key = f"datasets/{dataset_id}/{image_path}"

        logger.info(f"Generating presigned URL for {object_key}")

        presigned_url = r2_storage.generate_presigned_url(
            object_key=object_key,
            expiration=expiration
        )

        if not presigned_url:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL")

        return PresignedUrlResponse(
            status="success",
            image_path=image_path,
            presigned_url=presigned_url,
            expires_in=expiration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")

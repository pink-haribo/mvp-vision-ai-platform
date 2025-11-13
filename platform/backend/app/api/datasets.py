"""
Datasets API endpoints for dataset analysis and management.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging
import os
import tempfile
import zipfile
import shutil
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from app.utils.dataset_analyzer import DatasetAnalyzer
from app.utils.storage_utils import get_storage_client, get_storage_type
from app.db.database import get_db
from app.db.models import Dataset, User
from app.utils.dependencies import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


class DatasetAnalyzeRequest(BaseModel):
    """Request model for dataset analysis"""
    path: str
    format_hint: Optional[str] = None  # 'imagefolder', 'yolo', 'coco', or None for auto-detect


class DatasetAnalyzeResponse(BaseModel):
    """Response model for dataset analysis"""
    status: str  # 'success' or 'error'
    dataset_info: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    suggestions: Optional[List[str]] = None


def _get_dataset_metadata_from_db(dataset_id: str, db: Session) -> Optional[DatasetAnalyzeResponse]:
    """
    Check if dataset_id exists in database and return metadata.

    Works for all public datasets (including platform samples).
    """
    # Query dataset from database
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.visibility == 'public').first()

    if not dataset:
        return None

    logger.info(f"Found dataset in database: {dataset_id}")

    # Build response from database record
    dataset_info = {
        "format": dataset.format,
        "confidence": 1.0,  # DB datasets are pre-validated
        "labeled": dataset.labeled,  # Whether dataset has annotations
        "structure": {
            "num_classes": dataset.num_classes,
            "num_images": dataset.num_images,
            "classes": dataset.class_names[:10] if dataset.class_names else [],  # First 10 classes
        },
        "statistics": {
            "total_images": dataset.num_images,
            "source": dataset.storage_type,
            "validated": True,
        },
        "samples_per_class": {},  # Not detailed for DB datasets
        "quality_checks": {
            "corrupted_files": [],
            "missing_labels": [],
            "class_imbalance": False,
            "resolution_variance": "uniform",
            "overall_status": "excellent",
        },
        "preview_images": [],  # No preview for DB datasets
    }

    return DatasetAnalyzeResponse(
        status="success",
        dataset_info=dataset_info
    )


@router.post("/analyze", response_model=DatasetAnalyzeResponse)
async def analyze_dataset(request: DatasetAnalyzeRequest, db: Session = Depends(get_db)):
    """
    Analyze a dataset and return structure, statistics, and quality checks.

    - Automatically detects format (ImageFolder, YOLO, COCO)
    - Counts classes and samples
    - Calculates statistics (resolution, size, etc.)
    - Performs quality checks (corrupted files, class imbalance, etc.)
    - Supports public datasets from database (pre-validated metadata)
    """
    try:
        # Check if this is a public dataset in database
        db_dataset = _get_dataset_metadata_from_db(request.path, db)
        if db_dataset:
            logger.info(f"Found public dataset in database: {request.path}")
            return db_dataset

        # Validate path exists
        path = Path(request.path)
        if not path.exists():
            return DatasetAnalyzeResponse(
                status="error",
                error_type="path_not_found",
                message=f"경로를 찾을 수 없습니다: {request.path}",
                suggestions=[
                    "경로가 올바른지 확인하세요",
                    "절대 경로를 사용하세요 (예: C:\\datasets\\imagenet-10)",
                    "네트워크 드라이브의 경우 연결 상태를 확인하세요"
                ]
            )

        if not path.is_dir():
            return DatasetAnalyzeResponse(
                status="error",
                error_type="not_a_directory",
                message=f"경로가 디렉토리가 아닙니다: {request.path}",
                suggestions=[
                    "데이터셋 폴더 경로를 입력하세요",
                    "파일이 아닌 폴더를 선택하세요"
                ]
            )

        # Initialize analyzer
        analyzer = DatasetAnalyzer(path)

        # Detect format
        logger.info(f"Analyzing dataset at: {path}")
        detected_format = analyzer.detect_format(hint=request.format_hint)
        logger.info(f"[DEBUG] detected_format = {detected_format}")

        if detected_format['format'] == 'unknown':
            return DatasetAnalyzeResponse(
                status="error",
                error_type="unknown_format",
                message="데이터셋 형식을 인식할 수 없습니다",
                suggestions=[
                    "지원 형식: ImageFolder, YOLO, COCO",
                    "ImageFolder: dataset/class1/img1.jpg",
                    "YOLO: images/*.jpg + labels/*.txt",
                    "COCO: annotations/*.json + images/"
                ]
            )

        # Collect statistics
        logger.info(f"Collecting statistics for {detected_format['format']} format")
        stats = analyzer.collect_statistics(detected_format['format'])

        # Perform quality checks
        logger.info("Performing quality checks")
        quality_checks = analyzer.check_quality(stats)

        # Build response
        dataset_info = {
            "format": detected_format['format'],
            "confidence": detected_format['confidence'],
            "task_type": detected_format.get('task_type'),
            "structure": stats.get('structure', {}),
            "statistics": stats.get('statistics', {}),
            "samples_per_class": stats.get('samples_per_class', {}),
            "quality_checks": quality_checks,
            "preview_images": stats.get('preview_images', [])
        }

        return DatasetAnalyzeResponse(
            status="success",
            dataset_info=dataset_info
        )

    except PermissionError:
        return DatasetAnalyzeResponse(
            status="error",
            error_type="permission_denied",
            message="경로에 대한 접근 권한이 없습니다",
            suggestions=[
                "폴더의 권한을 확인하세요",
                "관리자 권한으로 실행해보세요"
            ]
        )
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)
        return DatasetAnalyzeResponse(
            status="error",
            error_type="analysis_error",
            message=f"데이터셋 분석 중 오류가 발생했습니다: {str(e)}",
            suggestions=[
                "데이터셋 구조가 올바른지 확인하세요",
                "일부 파일이 손상되었을 수 있습니다"
            ]
        )


class SampleDatasetInfo(BaseModel):
    """Sample dataset information"""
    id: str
    name: str
    description: str
    format: str
    labeled: bool  # Changed from task_type - indicates if dataset has annotations
    num_items: int
    size_mb: Optional[float] = None
    source: str
    path: str
    visibility: str  # 'public', 'private', 'organization'
    owner_id: Optional[int] = None
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    owner_badge_color: Optional[str] = None


class DatasetListItem(BaseModel):
    """Dataset list item"""
    name: str
    path: str
    size_mb: Optional[float] = None
    num_items: Optional[int] = None


class DatasetListResponse(BaseModel):
    """Response model for dataset list"""
    base_path: str
    datasets: List[DatasetListItem]


@router.get("/available", response_model=List[SampleDatasetInfo])
async def list_sample_datasets(
    labeled: Optional[bool] = Query(default=None, description="Filter by labeled status (true=has annotations, false=unlabeled)"),
    tags: Optional[str] = Query(default=None, description="Filter by tags (comma-separated)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List available datasets for the current user.

    Returns:
    - User's own datasets (where owner_id == user_id)
    - Public datasets (visibility == 'public')

    Args:
        labeled: Optional filter by labeled status
        tags: Optional filter by tags (comma-separated, e.g., "platform-sample,coco")
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of datasets with metadata
    """
    from sqlalchemy import or_

    # Query datasets that are either:
    # 1. Owned by the current user
    # 2. Public (visible to everyone)
    query = db.query(Dataset).filter(
        or_(
            Dataset.owner_id == current_user.id,
            Dataset.visibility == 'public'
        )
    )

    # Filter by labeled status
    if labeled is not None:
        query = query.filter(Dataset.labeled == labeled)

    # Filter by tags
    if tags:
        tag_list = [t.strip() for t in tags.split(',')]
        # Check if dataset has any of the specified tags
        for tag in tag_list:
            query = query.filter(Dataset.tags.contains([tag]))

    datasets = query.all()

    logger.info(f"User {current_user.id} ({current_user.email}) querying datasets: found {len(datasets)} datasets")

    # Convert to response format
    result = []
    for ds in datasets:
        result.append({
            "id": ds.id,
            "name": ds.name,
            "description": ds.description or f"Dataset - {ds.format} format",
            "format": ds.format,
            "labeled": ds.labeled,
            "num_items": ds.num_images,
            "size_mb": None,  # Size not stored in DB yet
            "source": ds.storage_type,
            "path": ds.id,  # Use ID as path
            "visibility": ds.visibility,
            "owner_id": ds.owner_id,
            "owner_name": ds.owner.full_name if ds.owner else None,
            "owner_email": ds.owner.email if ds.owner else None,
            "owner_badge_color": ds.owner.badge_color if ds.owner else None,
        })

    return result


@router.get("/list", response_model=DatasetListResponse)
async def list_datasets(
    base_path: str = Query(default="C:\\datasets", description="Base directory to scan for datasets")
):
    """
    List available datasets in the specified base directory.

    Scans for subdirectories that appear to be datasets based on their structure.
    """
    try:
        base = Path(base_path)

        if not base.exists():
            # Try common dataset locations
            alternative_paths = [
                Path("C:\\datasets"),
                Path("D:\\datasets"),
                Path.home() / "datasets",
                Path.cwd() / "datasets"
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    base = alt_path
                    break
            else:
                # None found, return empty list
                return DatasetListResponse(
                    base_path=str(base),
                    datasets=[]
                )

        if not base.is_dir():
            return DatasetListResponse(
                base_path=str(base),
                datasets=[]
            )

        datasets = []

        # Scan subdirectories
        try:
            for item in base.iterdir():
                if not item.is_dir():
                    continue

                # Skip hidden directories
                if item.name.startswith('.'):
                    continue

                # Calculate directory size and item count
                size_bytes = 0
                num_items = 0

                try:
                    for root, dirs, files in os.walk(item):
                        # Count image files
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))]
                        num_items += len(image_files)

                        # Calculate size
                        for file in files:
                            try:
                                size_bytes += os.path.getsize(os.path.join(root, file))
                            except:
                                pass
                except:
                    pass

                size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes > 0 else None

                datasets.append(DatasetListItem(
                    name=item.name,
                    path=str(item),
                    size_mb=size_mb,
                    num_items=num_items if num_items > 0 else None
                ))

        except PermissionError:
            logger.warning(f"Permission denied accessing {base}")

        # Sort by name
        datasets.sort(key=lambda x: x.name.lower())

        return DatasetListResponse(
            base_path=str(base),
            datasets=datasets
        )

    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


class CreateDatasetRequest(BaseModel):
    """Request model for creating a new dataset"""
    name: str
    description: Optional[str] = None
    visibility: str = "private"  # 'public', 'private', 'organization'


class CreateDatasetResponse(BaseModel):
    """Response model for dataset creation"""
    status: str
    dataset_id: Optional[str] = None
    message: str
    dataset: Optional[Dict[str, Any]] = None


@router.post("", response_model=CreateDatasetResponse)
async def create_dataset(
    request: CreateDatasetRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new empty dataset (metadata only).

    Images can be uploaded later via the folder upload endpoint.

    Args:
        request: Dataset creation request with name, description, visibility
        current_user: Current authenticated user
        db: Database session

    Returns:
        Created dataset information
    """
    try:
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())

        # Create dataset record (empty, no images yet)
        new_dataset = Dataset(
            id=dataset_id,
            name=request.name,
            description=request.description or f"Dataset: {request.name}",
            format="dice",  # Default format
            labeled=False,  # No annotations yet
            storage_type=get_storage_type(),  # Auto-detect from environment (r2, minio, s3)
            storage_path=f"datasets/{dataset_id}/",  # Reserve storage path
            visibility=request.visibility,
            owner_id=current_user.id,  # Set owner to current user
            num_classes=0,
            num_images=0,  # Empty dataset
            class_names=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        logger.info(f"Created empty dataset: {dataset_id} - {request.name}")

        return CreateDatasetResponse(
            status="success",
            dataset_id=dataset_id,
            message=f"Dataset '{request.name}' created successfully",
            dataset={
                "id": dataset_id,
                "name": request.name,
                "description": new_dataset.description,
                "visibility": request.visibility,
                "num_images": 0,
                "created_at": new_dataset.created_at.isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}", exc_info=True)
        return CreateDatasetResponse(
            status="error",
            message=f"Failed to create dataset: {str(e)}"
        )


class DeleteDatasetResponse(BaseModel):
    """Response model for dataset deletion"""
    status: str
    message: str


@router.delete("/{dataset_id}", response_model=DeleteDatasetResponse)
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a dataset and all its associated data.

    Only the dataset owner can delete it.

    This will:
    1. Delete all images from R2 storage
    2. Delete the dataset record from database

    Args:
        dataset_id: Dataset ID to delete
        current_user: Current authenticated user
        db: Database session

    Returns:
        Deletion status
    """
    try:
        # Find dataset in database
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            return DeleteDatasetResponse(
                status="error",
                message=f"Dataset not found: {dataset_id}"
            )

        # Check if user has permission to delete (must be owner)
        if dataset.owner_id != current_user.id:
            logger.warning(f"User {current_user.id} attempted to delete dataset {dataset_id} owned by {dataset.owner_id}")
            return DeleteDatasetResponse(
                status="error",
                message="Permission denied: You can only delete your own datasets"
            )

        # Delete all images from storage
        try:
            storage = get_storage_client()
            prefix = f"datasets/{dataset_id}/"
            deleted_count = storage.delete_all_with_prefix(prefix)
            logger.info(f"Deleted {deleted_count} objects from storage with prefix: {prefix}")
        except Exception as e:
            logger.warning(f"Error deleting storage objects for dataset {dataset_id}: {str(e)}")
            # Continue with database deletion even if storage deletion fails

        # Delete dataset record from database
        db.delete(dataset)
        db.commit()

        logger.info(f"Deleted dataset: {dataset_id} - {dataset.name}")

        return DeleteDatasetResponse(
            status="success",
            message=f"Dataset '{dataset.name}' deleted successfully"
        )

    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}", exc_info=True)
        db.rollback()
        return DeleteDatasetResponse(
            status="error",
            message=f"Failed to delete dataset: {str(e)}"
        )


@router.get("/{dataset_id}/file/{filename}")
async def get_dataset_file(
    dataset_id: str,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific file from dataset (e.g., annotations.json).

    This endpoint allows downloading dataset metadata files like annotations.json
    to check which images have labels without downloading all images.
    """
    try:
        # Verify dataset exists and user has access
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Check user ownership or public access
        if dataset.owner_id != current_user.id and dataset.visibility != 'public':
            raise HTTPException(status_code=403, detail="Access denied")

        # Construct storage key for the file
        storage_key = f"datasets/{dataset_id}/{filename}"

        logger.info(f"Fetching file from storage: {storage_key}")

        # Get file content from storage
        storage = get_storage_client()
        file_content = storage.get_file_content(storage_key)

        if file_content is None:
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found in dataset"
            )

        # Parse JSON if it's a JSON file
        if filename.endswith('.json'):
            try:
                # Decode bytes to string, then parse JSON
                json_str = file_content.decode('utf-8')
                json_data = json.loads(json_str)
                return JSONResponse(content=json_data)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse JSON file {filename}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse JSON file: {str(e)}"
                )

        # For non-JSON files, return raw content as base64
        import base64
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        return JSONResponse(content={"content": encoded_content})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dataset file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch file: {str(e)}"
        )

"""
Datasets API endpoints for dataset analysis and management.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging
import os
import tempfile
import zipfile
import shutil
from datetime import datetime
from sqlalchemy.orm import Session

from app.utils.dataset_analyzer import DatasetAnalyzer
from app.utils.r2_storage import r2_storage
from app.db.database import get_db
from app.db.models import Dataset

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
        "task_type": dataset.task_type,
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
    task_type: str
    num_items: int
    size_mb: Optional[float] = None
    source: str
    path: str


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
    task_type: Optional[str] = Query(default=None, description="Filter by task type (image_classification, object_detection, etc.)"),
    tags: Optional[str] = Query(default=None, description="Filter by tags (comma-separated)"),
    db: Session = Depends(get_db)
):
    """
    List available public datasets from database.

    Returns all public datasets, including platform-provided samples.
    Platform sample datasets have 'platform-sample' tag.

    Args:
        task_type: Optional filter by task type
        tags: Optional filter by tags (comma-separated, e.g., "platform-sample,coco")
        db: Database session

    Returns:
        List of datasets with metadata
    """
    # Query public datasets
    query = db.query(Dataset).filter(Dataset.visibility == 'public')

    # Filter by task type
    if task_type:
        query = query.filter(Dataset.task_type == task_type)

    # Filter by tags
    if tags:
        tag_list = [t.strip() for t in tags.split(',')]
        # Check if dataset has any of the specified tags
        for tag in tag_list:
            query = query.filter(Dataset.tags.contains([tag]))

    datasets = query.all()

    # Convert to response format
    result = []
    for ds in datasets:
        result.append({
            "id": ds.id,
            "name": ds.name,
            "description": ds.description or f"Dataset for {ds.task_type}",
            "format": ds.format,
            "task_type": ds.task_type,
            "num_items": ds.num_images,
            "size_mb": None,  # Size not stored in DB yet
            "source": ds.storage_type,
            "path": ds.id,  # Use ID as path
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


class DatasetUploadResponse(BaseModel):
    """Response model for dataset upload"""
    status: str  # 'success' or 'error'
    dataset_id: Optional[str] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    visibility: str = Query("private", regex="^(public|private|organization)$"),
    user_id: str = Query("platform"),  # TODO: Get from auth token
    db: Session = Depends(get_db)
):
    """
    Upload a DICE Format dataset (zip file) to R2 and register in database.

    Steps:
    1. Validate uploaded file is a zip
    2. Save to temporary location
    3. Validate DICE Format (annotations.json, meta.json, images/)
    4. Upload to R2
    5. Register metadata in database
    6. Clean up temporary files

    Args:
        file: Uploaded zip file (DICE Format dataset)
        visibility: Dataset visibility (public/private/organization)
        user_id: User ID (from auth token)
        db: Database session

    Returns:
        Upload status and dataset metadata
    """
    logger.info(f"Uploading dataset: {file.filename}")

    # Validate file extension
    if not file.filename.endswith('.zip'):
        return DatasetUploadResponse(
            status="error",
            message="Only zip files are supported"
        )

    temp_zip_path = None
    try:
        # Create temporary file for upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            temp_zip_path = Path(tmp_file.name)

            # Save uploaded file
            content = await file.read()
            tmp_file.write(content)
            logger.info(f"Saved uploaded file to {temp_zip_path} ({len(content)} bytes)")

        # Validate DICE Format
        is_valid, metadata, error_msg = r2_storage.validate_dice_format(temp_zip_path)

        if not is_valid:
            logger.error(f"Invalid DICE Format: {error_msg}")
            return DatasetUploadResponse(
                status="error",
                message=f"Invalid DICE Format: {error_msg}"
            )

        logger.info(f"Validated DICE Format: {metadata}")

        # Check if dataset ID already exists
        dataset_id = metadata['dataset_id']
        existing = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if existing:
            return DatasetUploadResponse(
                status="error",
                message=f"Dataset with ID '{dataset_id}' already exists"
            )

        # Upload to R2
        logger.info(f"Uploading to R2: datasets/{dataset_id}.zip")
        upload_success = r2_storage.upload_dataset_zip(temp_zip_path, dataset_id)

        if not upload_success:
            return DatasetUploadResponse(
                status="error",
                message="Failed to upload to R2 storage"
            )

        # Extract class names from metadata (if available)
        # For this, we need to read annotations.json from zip
        class_names = []
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                annotations_files = [f for f in zip_ref.namelist() if f.endswith('annotations.json')]
                if annotations_files:
                    import json
                    with zip_ref.open(annotations_files[0]) as ann_file:
                        annotations_data = json.load(ann_file)
                        classes = annotations_data.get('classes', [])
                        class_names = [c.get('name') for c in classes if 'name' in c]
        except Exception as e:
            logger.warning(f"Could not extract class names: {e}")

        # Register in database
        new_dataset = Dataset(
            id=dataset_id,
            name=metadata['dataset_name'],
            description=f"Uploaded via web UI - {metadata['task_type']}",
            format="dice",  # DICE Format
            task_type=metadata['task_type'],
            storage_type="r2",
            storage_path=f"datasets/{dataset_id}.zip",
            visibility=visibility,
            owner_id=None,  # TODO: Get from auth token
            num_classes=metadata['num_classes'],
            num_images=metadata['total_images'],
            class_names=class_names,
            content_hash=metadata.get('content_hash'),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        logger.info(f"Dataset registered in database: {dataset_id}")

        return DatasetUploadResponse(
            status="success",
            dataset_id=dataset_id,
            message=f"Dataset '{metadata['dataset_name']}' uploaded successfully",
            metadata={
                "dataset_id": dataset_id,
                "dataset_name": metadata['dataset_name'],
                "task_type": metadata['task_type'],
                "num_classes": metadata['num_classes'],
                "total_images": metadata['total_images'],
                "visibility": visibility,
                "storage_path": f"datasets/{dataset_id}.zip"
            }
        )

    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}", exc_info=True)
        return DatasetUploadResponse(
            status="error",
            message=f"Upload failed: {str(e)}"
        )

    finally:
        # Clean up temporary file
        if temp_zip_path and temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
                logger.info(f"Cleaned up temporary file: {temp_zip_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")

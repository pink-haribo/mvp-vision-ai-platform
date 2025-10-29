"""
Datasets API endpoints for dataset analysis and management.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging
import os

from app.utils.dataset_analyzer import DatasetAnalyzer

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


@router.post("/analyze", response_model=DatasetAnalyzeResponse)
async def analyze_dataset(request: DatasetAnalyzeRequest):
    """
    Analyze a dataset and return structure, statistics, and quality checks.

    - Automatically detects format (ImageFolder, YOLO, COCO)
    - Counts classes and samples
    - Calculates statistics (resolution, size, etc.)
    - Performs quality checks (corrupted files, class imbalance, etc.)
    """
    try:
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

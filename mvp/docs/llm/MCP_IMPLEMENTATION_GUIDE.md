# MCP Server 구현 가이드

## 개요

이 문서는 Vision AI Training Platform의 MCP (Model Context Protocol) 서버 구현을 위한 실무 가이드입니다.

**대상**: Backend 개발자
**난이도**: 중급
**예상 소요 시간**: 3주

---

## 목차

1. [MCP 기초](#1-mcp-기초)
2. [개발 환경 설정](#2-개발-환경-설정)
3. [서버 구조 설계](#3-서버-구조-설계)
4. [Tools 구현](#4-tools-구현)
5. [Resources 구현](#5-resources-구현)
6. [Prompts 구현](#6-prompts-구현)
7. [테스트](#7-테스트)
8. [배포](#8-배포)

---

## 1. MCP 기초

### 1.1 MCP란?

Model Context Protocol (MCP)은 Anthropic이 개발한 **AI 시스템과 외부 도구/데이터를 연결하는 표준 프로토콜**입니다.

**핵심 개념:**

```
┌──────────────┐         MCP Protocol         ┌──────────────┐
│  MCP Client  │ ←────────────────────────→  │  MCP Server  │
│   (Claude)   │    JSON-RPC over stdio       │  (Backend)   │
└──────────────┘                              └──────────────┘
       │                                              │
       │ Tool Calls                                   │ Tool Implementations
       │ Resource Requests                            │ Resource Providers
       │ Prompt Requests                              │ Prompt Templates
       ▼                                              ▼
```

### 1.2 MCP의 세 가지 프리미티브

#### **1. Tools (함수)**

LLM이 호출할 수 있는 함수입니다.

```python
@server.tool()
async def create_training_job(
    model_name: str,
    dataset_path: str,
    epochs: int = 100
) -> dict:
    """
    Create a new training job.

    Args:
        model_name: Model to train
        dataset_path: Path to dataset
        epochs: Number of epochs

    Returns:
        dict: Job details
    """
    # Implementation
    pass
```

**특징:**
- LLM이 자동으로 선택하고 호출
- 파라미터는 타입 힌트와 docstring으로 정의
- 반환값은 JSON 직렬화 가능해야 함

#### **2. Resources (데이터 소스)**

LLM이 읽을 수 있는 구조화된 데이터입니다.

```python
@server.resource("training://jobs/{job_id}")
async def training_job_resource(uri: str) -> str:
    """Get training job details"""
    job_id = extract_id(uri)
    job = await get_job(job_id)

    return f"""
    Training Job #{job.id}

    Model: {job.model_name}
    Status: {job.status}
    Progress: {job.current_epoch}/{job.epochs}
    """
```

**특징:**
- URI 스킴 기반 (`training://`, `models://` 등)
- 텍스트 또는 이미지 반환 가능
- LLM이 필요 시 조회

#### **3. Prompts (템플릿)**

재사용 가능한 프롬프트 템플릿입니다.

```python
@server.prompt()
async def model_recommendation(
    task_type: str,
    dataset_size: int
) -> list[Message]:
    """Generate model recommendation prompt"""

    return [
        Message(
            role="user",
            content=f"Recommend a model for {task_type} with {dataset_size} images"
        )
    ]
```

**특징:**
- 파라미터화된 프롬프트
- 멀티턴 대화 구성 가능
- LLM이 선택하여 사용

### 1.3 통신 방식

MCP는 **JSON-RPC 2.0** over **stdio** 또는 **HTTP**를 사용합니다.

**요청 예시:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "create_training_job",
    "arguments": {
      "model_name": "resnet50",
      "dataset_path": "C:/datasets/cats",
      "epochs": 100
    }
  }
}
```

**응답 예시:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"job_id\": 12345, \"status\": \"created\"}"
      }
    ]
  }
}
```

---

## 2. 개발 환경 설정

### 2.1 의존성 설치

**requirements.txt 추가:**
```txt
# MCP SDK
mcp>=0.9.0

# 기존 의존성
fastapi>=0.104.0
sqlalchemy>=2.0.0
# ...
```

**설치:**
```bash
cd mvp/backend
pip install -r requirements.txt
```

### 2.2 프로젝트 구조

```
mvp/backend/app/mcp/
├── __init__.py
├── server.py                 # MCP Server 진입점
├── config.py                 # MCP 설정
├── tools/
│   ├── __init__.py
│   ├── base.py              # Base Tool 클래스
│   ├── training.py          # Training tools
│   ├── inference.py         # Inference tools
│   ├── dataset.py           # Dataset tools
│   ├── model.py             # Model tools
│   └── project.py           # Project tools
├── resources/
│   ├── __init__.py
│   ├── base.py              # Base Resource 클래스
│   ├── training.py          # Training resources
│   ├── validation.py        # Validation resources
│   └── model.py             # Model resources
├── prompts/
│   ├── __init__.py
│   ├── base.py              # Base Prompt 클래스
│   ├── recommendation.py    # Recommendation prompts
│   └── troubleshooting.py   # Troubleshooting prompts
└── utils/
    ├── __init__.py
    ├── auth.py              # Authentication helpers
    └── formatting.py        # Output formatting
```

### 2.3 MCP Server 초기화

**`mvp/backend/app/mcp/server.py`:**

```python
"""MCP Server for Vision AI Training Platform"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server 인스턴스
mcp_server = Server("vision-ai-training-platform")

def register_all_components():
    """모든 Tools, Resources, Prompts 등록"""

    # Tools 등록
    from .tools import training, inference, dataset, model, project
    logger.info("Registering tools...")
    training.register_tools(mcp_server)
    inference.register_tools(mcp_server)
    dataset.register_tools(mcp_server)
    model.register_tools(mcp_server)
    project.register_tools(mcp_server)

    # Resources 등록
    from .resources import training as training_res, validation, model as model_res
    logger.info("Registering resources...")
    training_res.register_resources(mcp_server)
    validation.register_resources(mcp_server)
    model_res.register_resources(mcp_server)

    # Prompts 등록
    from .prompts import recommendation, troubleshooting
    logger.info("Registering prompts...")
    recommendation.register_prompts(mcp_server)
    troubleshooting.register_prompts(mcp_server)

    logger.info("All components registered successfully")

async def main():
    """MCP Server 실행 (stdio 모드)"""

    # 컴포넌트 등록
    register_all_components()

    # stdio 서버 실행
    logger.info("Starting MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.4 Claude Code 설정

**`.claude/mcp.json`:**

```json
{
  "mcpServers": {
    "vision-ai-platform": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "C:/Users/flyto/Project/Github/mvp-vision-ai-platform/mvp/backend",
      "env": {
        "PYTHONPATH": ".",
        "DATABASE_URL": "sqlite:///./vision_platform.db",
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
      }
    }
  }
}
```

**환경 변수 설정 (`.env`):**
```bash
GOOGLE_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./vision_platform.db
```

---

## 3. 서버 구조 설계

### 3.1 Base Classes

#### **Base Tool**

**`mvp/backend/app/mcp/tools/base.py`:**

```python
"""Base classes for MCP tools"""

from abc import ABC, abstractmethod
from typing import Any
from mcp.server import Server
from sqlalchemy.orm import Session
from app.db.database import get_db
import logging

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Base class for all MCP tools"""

    def __init__(self, server: Server):
        self.server = server
        self.logger = logger

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Any: Tool result (must be JSON serializable)
        """
        pass

    async def get_db(self) -> Session:
        """Get database session"""
        async for db in get_db():
            return db

    def log_execution(self, tool_name: str, params: dict, result: Any):
        """Log tool execution"""
        self.logger.info(
            f"Tool executed: {tool_name}",
            extra={"params": params, "result_type": type(result).__name__}
        )

    def handle_error(self, error: Exception, tool_name: str) -> dict:
        """Handle tool execution error"""
        self.logger.error(f"Tool error in {tool_name}: {str(error)}")
        return {
            "error": str(error),
            "tool": tool_name,
            "type": type(error).__name__
        }
```

#### **Base Resource**

**`mvp/backend/app/mcp/resources/base.py`:**

```python
"""Base classes for MCP resources"""

from abc import ABC, abstractmethod
from typing import Any

class BaseResource(ABC):
    """Base class for all MCP resources"""

    @abstractmethod
    async def get_content(self, uri: str) -> str:
        """
        Get resource content.

        Args:
            uri: Resource URI (e.g., "training://jobs/12345")

        Returns:
            str: Formatted resource content
        """
        pass

    def extract_id_from_uri(self, uri: str, position: int = -1) -> int:
        """Extract ID from URI"""
        parts = uri.split("/")
        return int(parts[position])

    def format_as_markdown(self, data: dict) -> str:
        """Format data as markdown"""
        # Implementation
        pass
```

### 3.2 인증 및 권한

**`mvp/backend/app/mcp/utils/auth.py`:**

```python
"""Authentication and authorization for MCP tools"""

from typing import Optional
from app.models import User, TrainingJob, Project
from sqlalchemy.orm import Session

class MCPAuth:
    """MCP 인증 및 권한 관리"""

    @staticmethod
    async def get_current_user(db: Session, user_id: int) -> Optional[User]:
        """Get current user from database"""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    async def check_job_permission(
        db: Session,
        user_id: int,
        job_id: int,
        action: str = "read"
    ) -> bool:
        """
        Check if user has permission for training job.

        Args:
            db: Database session
            user_id: User ID
            job_id: Training job ID
            action: Action to perform (read, write, delete)

        Returns:
            bool: True if permitted
        """
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return False

        # Creator has full access
        if job.creator_id == user_id:
            return True

        # Check project membership
        if job.project_id:
            project = db.query(Project).filter(Project.id == job.project_id).first()
            if project:
                # Check if user is project member
                return user_id in [m.user_id for m in project.members]

        return False

    @staticmethod
    async def check_project_permission(
        db: Session,
        user_id: int,
        project_id: int,
        action: str = "read"
    ) -> bool:
        """Check if user has permission for project"""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return False

        # Owner has full access
        if project.owner_id == user_id:
            return True

        # Check membership
        member = next((m for m in project.members if m.user_id == user_id), None)
        if member:
            # Check role for write/delete actions
            if action in ["write", "delete"]:
                return member.role in ["owner", "editor"]
            return True

        return False
```

### 3.3 에러 처리

**공통 에러 클래스:**

```python
"""MCP-specific exceptions"""

class MCPError(Exception):
    """Base MCP error"""
    pass

class PermissionDeniedError(MCPError):
    """User doesn't have permission"""
    pass

class ResourceNotFoundError(MCPError):
    """Resource not found"""
    pass

class ValidationError(MCPError):
    """Input validation failed"""
    pass

class ToolExecutionError(MCPError):
    """Tool execution failed"""
    pass
```

**에러 핸들러:**

```python
async def safe_execute_tool(func, **kwargs):
    """Safely execute tool with error handling"""
    try:
        return await func(**kwargs)
    except PermissionDeniedError as e:
        return {"error": "Permission denied", "message": str(e)}
    except ResourceNotFoundError as e:
        return {"error": "Resource not found", "message": str(e)}
    except ValidationError as e:
        return {"error": "Validation failed", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Internal error", "message": "An unexpected error occurred"}
```

---

## 4. Tools 구현

### 4.1 Training Tools

**`mvp/backend/app/mcp/tools/training.py`:**

```python
"""Training-related MCP tools"""

from mcp.server import Server
from mcp.types import Tool
from app.services.training_service import TrainingService
from app.db.database import get_db
from .base import BaseTool
from ..utils.auth import MCPAuth

training_service = TrainingService()
auth = MCPAuth()

def register_tools(server: Server):
    """Register all training tools"""

    @server.tool()
    async def create_training_job(
        model_name: str,
        task_type: str,
        dataset_path: str,
        user_id: int,
        framework: str = "timm",
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_classes: int | None = None,
        project_id: int | None = None,
        experiment_name: str | None = None,
        advanced_config: dict | None = None
    ) -> dict:
        """
        Create a new training job.

        This tool creates a new training job with the specified configuration.
        The job is created but not started automatically - use start_training to begin.

        Args:
            model_name: Model name (e.g., resnet50, yolov8m, vit-base)
            task_type: Task type (classification, object_detection, instance_segmentation, etc.)
            dataset_path: Absolute path to dataset directory
            user_id: User ID creating the job
            framework: Framework to use (timm, ultralytics, transformers). Default: timm
            epochs: Number of training epochs. Default: 100
            batch_size: Batch size for training. Default: 32
            learning_rate: Learning rate. Default: 0.001
            num_classes: Number of classes (auto-detected if None)
            project_id: Project ID to associate with (optional)
            experiment_name: Name for this experiment (optional)
            advanced_config: Advanced configuration (optimizer, scheduler, augmentation)

        Returns:
            dict: Created job details including job_id, status, and configuration

        Raises:
            ValidationError: If parameters are invalid
            PermissionDeniedError: If user doesn't have permission
        """
        try:
            async with get_db() as db:
                # Verify user
                user = await auth.get_current_user(db, user_id)
                if not user:
                    raise PermissionDeniedError("Invalid user")

                # Verify project permission if provided
                if project_id:
                    has_permission = await auth.check_project_permission(
                        db, user_id, project_id, "write"
                    )
                    if not has_permission:
                        raise PermissionDeniedError("No permission for this project")

                # Create job
                job = await training_service.create_job(
                    db=db,
                    user_id=user_id,
                    model_name=model_name,
                    task_type=task_type,
                    dataset_path=dataset_path,
                    framework=framework,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_classes=num_classes,
                    project_id=project_id,
                    experiment_name=experiment_name,
                    advanced_config=advanced_config or {}
                )

                return {
                    "job_id": job.id,
                    "status": job.status,
                    "model": job.model_name,
                    "framework": job.framework,
                    "task_type": job.task_type,
                    "dataset": job.dataset_path,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "learning_rate": job.learning_rate,
                    "created_at": job.created_at.isoformat(),
                    "message": f"Training job #{job.id} created successfully. Use start_training to begin."
                }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def start_training(
        job_id: int,
        user_id: int,
        resume_from_checkpoint: str | None = None
    ) -> dict:
        """
        Start a training job.

        Args:
            job_id: Training job ID
            user_id: User ID starting the job
            resume_from_checkpoint: Path to checkpoint to resume from (optional)

        Returns:
            dict: Job status and estimated completion time

        Raises:
            PermissionDeniedError: If user doesn't have permission
            ResourceNotFoundError: If job doesn't exist
        """
        try:
            async with get_db() as db:
                # Check permission
                has_permission = await auth.check_job_permission(
                    db, user_id, job_id, "write"
                )
                if not has_permission:
                    raise PermissionDeniedError("No permission for this job")

                # Start job
                result = await training_service.start_job(
                    db=db,
                    job_id=job_id,
                    checkpoint_path=resume_from_checkpoint
                )

                return {
                    "job_id": job_id,
                    "status": "running",
                    "process_id": result.process_id,
                    "started_at": result.started_at.isoformat(),
                    "estimated_duration_minutes": result.epochs * 2,
                    "message": f"Training started successfully. Monitor progress with get_training_status."
                }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def get_training_status(job_id: int, user_id: int) -> dict:
        """
        Get current status and progress of a training job.

        Args:
            job_id: Training job ID
            user_id: User ID requesting status

        Returns:
            dict: Detailed status including progress, metrics, and ETA

        Raises:
            PermissionDeniedError: If user doesn't have permission
        """
        try:
            async with get_db() as db:
                # Check permission
                has_permission = await auth.check_job_permission(
                    db, user_id, job_id, "read"
                )
                if not has_permission:
                    raise PermissionDeniedError("No permission for this job")

                # Get status
                status = await training_service.get_job_status(db, job_id)

                return {
                    "job_id": job_id,
                    "status": status.status,
                    "current_epoch": status.current_epoch,
                    "total_epochs": status.total_epochs,
                    "progress_percent": (
                        status.current_epoch / status.total_epochs * 100
                        if status.total_epochs > 0 else 0
                    ),
                    "latest_metrics": status.latest_metrics,
                    "estimated_remaining_minutes": status.eta_minutes,
                    "started_at": status.started_at.isoformat() if status.started_at else None,
                    "updated_at": status.updated_at.isoformat()
                }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def stop_training(
        job_id: int,
        user_id: int,
        save_checkpoint: bool = True
    ) -> dict:
        """
        Stop a running training job.

        Args:
            job_id: Training job ID
            user_id: User ID stopping the job
            save_checkpoint: Whether to save checkpoint before stopping. Default: True

        Returns:
            dict: Final status and checkpoint path if saved

        Raises:
            PermissionDeniedError: If user doesn't have permission
        """
        try:
            async with get_db() as db:
                # Check permission
                has_permission = await auth.check_job_permission(
                    db, user_id, job_id, "write"
                )
                if not has_permission:
                    raise PermissionDeniedError("No permission for this job")

                # Stop job
                result = await training_service.stop_job(
                    db=db,
                    job_id=job_id,
                    save_checkpoint=save_checkpoint
                )

                return {
                    "job_id": job_id,
                    "status": "stopped",
                    "final_epoch": result.final_epoch,
                    "checkpoint_path": result.checkpoint_path if save_checkpoint else None,
                    "message": "Training stopped successfully."
                }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def list_training_jobs(
        user_id: int,
        status: str | None = None,
        project_id: int | None = None,
        limit: int = 20
    ) -> list[dict]:
        """
        List training jobs with optional filters.

        Args:
            user_id: User ID requesting the list
            status: Filter by status (running, completed, failed, etc.)
            project_id: Filter by project ID
            limit: Maximum number of jobs to return. Default: 20

        Returns:
            list[dict]: List of training jobs

        Raises:
            PermissionDeniedError: If user doesn't have permission for project
        """
        try:
            async with get_db() as db:
                # If project_id provided, check permission
                if project_id:
                    has_permission = await auth.check_project_permission(
                        db, user_id, project_id, "read"
                    )
                    if not has_permission:
                        raise PermissionDeniedError("No permission for this project")

                # List jobs
                jobs = await training_service.list_jobs(
                    db=db,
                    user_id=user_id,
                    status=status,
                    project_id=project_id,
                    limit=limit
                )

                return [
                    {
                        "job_id": job.id,
                        "model": job.model_name,
                        "task_type": job.task_type,
                        "status": job.status,
                        "created_at": job.created_at.isoformat(),
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "final_metric": job.final_accuracy,
                        "project_id": job.project_id,
                        "experiment_name": job.experiment_name
                    }
                    for job in jobs
                ]

        except Exception as e:
            return [{"error": str(e), "type": type(e).__name__}]
```

### 4.2 Dataset Tools

**`mvp/backend/app/mcp/tools/dataset.py`:**

```python
"""Dataset-related MCP tools"""

from mcp.server import Server
from app.services.dataset_service import DatasetService
from pathlib import Path

dataset_service = DatasetService()

def register_tools(server: Server):

    @server.tool()
    async def analyze_dataset(dataset_path: str) -> dict:
        """
        Analyze dataset structure, format, and quality.

        This tool examines a dataset directory and provides comprehensive analysis
        including format detection, class distribution, quality metrics, and recommendations.

        Args:
            dataset_path: Absolute path to dataset directory

        Returns:
            dict: Comprehensive dataset analysis including:
                - format: Detected format (imagefolder, yolo, coco)
                - task_type: Inferred task type
                - num_classes: Number of classes
                - classes: List of class names
                - total_images: Total image count
                - class_distribution: Images per class
                - imbalance_ratio: Max/min class ratio
                - corrupted_files: List of corrupted files
                - quality_score: Overall quality (0-100)
                - recommendations: List of recommendations

        Raises:
            ValidationError: If path doesn't exist or is invalid
        """
        try:
            # Validate path
            if not Path(dataset_path).exists():
                raise ValidationError(f"Dataset path does not exist: {dataset_path}")

            # Analyze
            analysis = await dataset_service.analyze(dataset_path)

            return {
                "path": dataset_path,
                "format": analysis.format,
                "task_type": analysis.inferred_task_type,
                "num_classes": len(analysis.classes),
                "classes": analysis.classes,
                "total_images": analysis.total_images,
                "class_distribution": analysis.class_distribution,
                "imbalance_ratio": analysis.imbalance_ratio,
                "avg_image_size": analysis.avg_image_size,
                "corrupted_files": analysis.corrupted_files,
                "quality_score": analysis.quality_score,
                "recommendations": analysis.recommendations,
                "message": (
                    f"Dataset analyzed successfully. "
                    f"Found {len(analysis.classes)} classes with {analysis.total_images} total images."
                )
            }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def list_datasets(base_path: str = "C:/datasets") -> list[dict]:
        """
        List available datasets in a directory.

        Args:
            base_path: Base directory to search for datasets. Default: C:/datasets

        Returns:
            list[dict]: List of found datasets with basic info

        Raises:
            ValidationError: If base_path doesn't exist
        """
        try:
            if not Path(base_path).exists():
                raise ValidationError(f"Base path does not exist: {base_path}")

            datasets = await dataset_service.list_datasets(base_path)

            return [
                {
                    "name": ds.name,
                    "path": ds.path,
                    "format": ds.format,
                    "num_classes": ds.num_classes,
                    "total_images": ds.total_images
                }
                for ds in datasets
            ]

        except Exception as e:
            return [{"error": str(e), "type": type(e).__name__}]

    @server.tool()
    async def validate_dataset(
        dataset_path: str,
        expected_format: str,
        expected_classes: list[str] | None = None
    ) -> dict:
        """
        Validate dataset against expected structure.

        Args:
            dataset_path: Path to dataset
            expected_format: Expected format (imagefolder, yolo, coco)
            expected_classes: Expected class names (optional)

        Returns:
            dict: Validation results with issues and warnings

        Raises:
            ValidationError: If path doesn't exist
        """
        try:
            if not Path(dataset_path).exists():
                raise ValidationError(f"Dataset path does not exist: {dataset_path}")

            result = await dataset_service.validate(
                dataset_path,
                expected_format,
                expected_classes
            )

            return {
                "is_valid": result.is_valid,
                "format_match": result.format_match,
                "classes_match": result.classes_match,
                "issues": result.issues,
                "warnings": result.warnings,
                "message": "Validation passed" if result.is_valid else "Validation failed"
            }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}
```

### 4.3 Model Tools

**`mvp/backend/app/mcp/tools/model.py`:**

```python
"""Model-related MCP tools"""

from mcp.server import Server
from app.services.model_registry import ModelRegistry

registry = ModelRegistry()

def register_tools(server: Server):

    @server.tool()
    async def search_models(
        task_type: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        min_priority: int = 0
    ) -> list[dict]:
        """
        Search available models by filters.

        Args:
            task_type: Filter by task type (classification, detection, etc.)
            framework: Filter by framework (timm, ultralytics, transformers)
            tags: Filter by tags (e.g., ["lightweight", "sota"])
            min_priority: Minimum priority score (0-10). Default: 0

        Returns:
            list[dict]: Matching models with metadata
        """
        try:
            models = registry.search(
                task_type=task_type,
                framework=framework,
                tags=tags,
                min_priority=min_priority
            )

            return [
                {
                    "name": m.name,
                    "framework": m.framework,
                    "task_type": m.task_type,
                    "description": m.description,
                    "priority": m.priority,
                    "tags": m.tags,
                    "typical_accuracy": m.typical_accuracy,
                    "training_speed": m.training_speed
                }
                for m in models
            ]

        except Exception as e:
            return [{"error": str(e), "type": type(e).__name__}]

    @server.tool()
    async def get_model_guide(framework: str, model_name: str) -> dict:
        """
        Get comprehensive guide for a specific model.

        Args:
            framework: Framework name (timm, ultralytics, transformers)
            model_name: Model name (e.g., resnet50, yolov8m)

        Returns:
            dict: Complete guide including benchmarks, pros/cons, alternatives

        Raises:
            ResourceNotFoundError: If model not found
        """
        try:
            guide = registry.get_guide(framework, model_name)

            if not guide:
                raise ResourceNotFoundError(
                    f"Model {framework}/{model_name} not found"
                )

            return {
                "model": guide.name,
                "framework": guide.framework,
                "description": guide.description,
                "benchmarks": guide.benchmarks,
                "pros": guide.pros,
                "cons": guide.cons,
                "best_for": guide.best_for,
                "not_recommended_for": guide.not_recommended_for,
                "alternatives": guide.alternatives,
                "typical_hyperparams": guide.typical_hyperparams,
                "references": guide.references
            }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def compare_models(model_specs: list[dict]) -> dict:
        """
        Compare multiple models side by side.

        Args:
            model_specs: List of model specs, each with "framework" and "name" keys
                Example: [{"framework": "timm", "name": "resnet50"}, ...]

        Returns:
            dict: Comparison table with metrics, speed, accuracy, etc.
        """
        try:
            comparison = registry.compare_models(model_specs)

            return {
                "models": [spec["name"] for spec in model_specs],
                "comparison": comparison,
                "recommendation": comparison.get("recommendation")
            }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @server.tool()
    async def recommend_model(
        task_type: str,
        dataset_size: int,
        priority: str = "balanced",
        constraints: dict | None = None
    ) -> dict:
        """
        Recommend best model based on requirements.

        Args:
            task_type: Task type (classification, detection, etc.)
            dataset_size: Number of images in dataset
            priority: What to optimize for (speed, accuracy, balanced). Default: balanced
            constraints: Additional constraints (max_params, max_latency, etc.)

        Returns:
            dict: Recommended model with detailed reasoning
        """
        try:
            recommendation = await registry.recommend_model(
                task_type=task_type,
                dataset_size=dataset_size,
                priority=priority,
                constraints=constraints or {}
            )

            return {
                "recommended_model": recommendation.model_name,
                "framework": recommendation.framework,
                "reasoning": recommendation.reasoning,
                "expected_accuracy": recommendation.expected_accuracy,
                "expected_training_time": recommendation.expected_training_time,
                "alternatives": recommendation.alternatives,
                "suggested_hyperparams": recommendation.suggested_hyperparams
            }

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}
```

---

## 5. Resources 구현

### 5.1 Training Resources

**`mvp/backend/app/mcp/resources/training.py`:**

```python
"""Training-related MCP resources"""

from mcp.server import Server
from app.services.training_service import TrainingService
from app.db.database import get_db

training_service = TrainingService()

def register_resources(server: Server):

    @server.resource("training://jobs/{job_id}")
    async def training_job_resource(uri: str) -> str:
        """
        Get detailed information about a training job.

        URI format: training://jobs/{job_id}
        """
        job_id = int(uri.split("/")[-1])

        async with get_db() as db:
            job = await training_service.get_job(db, job_id)

        return f"""
# Training Job #{job.id}

## Configuration
- **Model**: {job.model_name} ({job.framework})
- **Task**: {job.task_type}
- **Dataset**: {job.dataset_path}
- **Format**: {job.dataset_format}
- **Classes**: {job.num_classes}

## Hyperparameters
- **Epochs**: {job.epochs}
- **Batch Size**: {job.batch_size}
- **Learning Rate**: {job.learning_rate}
- **Optimizer**: {job.advanced_config.get('optimizer', {}).get('type', 'adam')}
- **Scheduler**: {job.advanced_config.get('scheduler', {}).get('type', 'step')}

## Status
- **Current Status**: {job.status}
- **Progress**: {job.current_epoch}/{job.epochs} epochs ({job.current_epoch/job.epochs*100:.1f}%)
- **Started**: {job.started_at.isoformat() if job.started_at else 'Not started'}
- **Updated**: {job.updated_at.isoformat()}

## Performance
- **Primary Metric**: {job.primary_metric} = {job.primary_metric_value:.4f}
- **Final Accuracy**: {job.final_accuracy:.4f if job.final_accuracy else 'N/A'}
- **Best Checkpoint**: {job.best_checkpoint_path or 'Not available'}

## Project & Experiment
- **Project ID**: {job.project_id or 'Uncategorized'}
- **Experiment Name**: {job.experiment_name or 'Default'}
- **Tags**: {', '.join(job.tags) if job.tags else 'None'}

## Outputs
- **Output Directory**: {job.output_dir}
- **MLflow Run**: {job.mlflow_run_id or 'N/A'}

## Notes
{job.notes or 'No notes'}
"""

    @server.resource("training://jobs/{job_id}/metrics")
    async def training_metrics_resource(uri: str) -> str:
        """
        Get training metrics history.

        URI format: training://jobs/{job_id}/metrics
        """
        job_id = int(uri.split("/")[-2])

        async with get_db() as db:
            metrics = await training_service.get_metrics(db, job_id, limit=100)

        if not metrics:
            return f"# Training Metrics for Job #{job_id}\n\nNo metrics available yet."

        output = f"# Training Metrics for Job #{job_id}\n\n"
        output += "| Epoch | Loss | Accuracy | Val Loss | Val Acc | LR |\n"
        output += "|-------|------|----------|----------|---------|----|\n"

        for m in metrics:
            output += (
                f"| {m.epoch} | {m.loss:.4f} | {m.accuracy:.4f} | "
                f"{m.val_loss:.4f} | {m.val_accuracy:.4f} | {m.learning_rate:.6f} |\n"
            )

        return output

    @server.resource("training://jobs/{job_id}/logs")
    async def training_logs_resource(uri: str) -> str:
        """
        Get training logs.

        URI format: training://jobs/{job_id}/logs
        """
        job_id = int(uri.split("/")[-2])

        async with get_db() as db:
            logs = await training_service.get_logs(db, job_id, limit=500)

        if not logs:
            return f"# Training Logs for Job #{job_id}\n\nNo logs available."

        output = f"# Training Logs for Job #{job_id}\n\n"
        output += "```\n"
        for log in logs:
            timestamp = log.created_at.strftime("%Y-%m-%d %H:%M:%S")
            output += f"[{timestamp}] {log.message}\n"
        output += "```\n"

        return output
```

---

## 6. Prompts 구현

**`mvp/backend/app/mcp/prompts/recommendation.py`:**

```python
"""Recommendation prompts"""

from mcp.server import Server
from mcp.types import PromptMessage, TextContent

def register_prompts(server: Server):

    @server.prompt()
    async def model_recommendation(
        task_type: str,
        dataset_size: int,
        target_metric: str = "accuracy"
    ) -> list[PromptMessage]:
        """
        Generate model recommendation prompt.

        Args:
            task_type: Task type
            dataset_size: Number of images
            target_metric: Metric to optimize for
        """
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.search(task_type=task_type)

        models_text = "\n".join([
            f"- {m.name} ({m.framework}): {m.description}"
            for m in models[:10]
        ])

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""
I need help selecting a model for a computer vision task.

Task Type: {task_type}
Dataset Size: {dataset_size} images
Target Metric: {target_metric}

Available models:
{models_text}

Please recommend the best model and explain:
1. Why this model is suitable
2. Expected performance
3. Training time estimate
4. Any concerns or limitations
5. Alternative options
"""
                )
            )
        ]
```

---

## 7. 테스트

### 7.1 Unit Tests

**`tests/mcp/test_training_tools.py`:**

```python
import pytest
from app.mcp.tools.training import create_training_job

@pytest.mark.asyncio
async def test_create_training_job():
    """Test creating a training job via MCP tool"""

    result = await create_training_job(
        model_name="resnet50",
        task_type="classification",
        dataset_path="C:/datasets/test",
        user_id=1,
        epochs=10
    )

    assert "job_id" in result
    assert result["status"] == "created"
    assert result["model"] == "resnet50"
```

### 7.2 Integration Tests

**Claude Code에서 직접 테스트:**

```
User: Analyze my dataset at C:/datasets/cats

Claude: [Calls analyze_dataset tool]
        [Returns analysis results]
```

---

## 8. 배포

### 8.1 Production 설정

**환경 변수:**
```bash
MCP_LOG_LEVEL=INFO
MCP_AUTH_ENABLED=true
MCP_RATE_LIMIT_PER_MINUTE=60
```

### 8.2 모니터링

- Tool 호출 로그
- 응답 시간 추적
- 에러율 모니터링

---

## 다음 단계

1. Phase 1의 나머지 Tools 구현
2. Resources 구현
3. Prompts 구현
4. End-to-end 테스트


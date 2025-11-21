"""
Tool Registry for Gemini Track

This module provides a central registry of all available tools that the LLM can use.
Each tool is a function that can be called by ActionHandlers to perform specific tasks.

Phase 1 Implementation:
- Dataset tools
- Model tools
- Inference tools
- Training status tools
"""

from typing import Callable, Dict, Any, Optional
from enum import Enum
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories for organization"""
    TRAINING = "training"
    INFERENCE = "inference"
    DATASET = "dataset"
    MODEL = "model"
    PROJECT = "project"
    RESULTS = "results"


class Tool:
    """
    Tool definition

    A tool is a callable function that can be invoked by ActionHandlers.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        handler: Callable,
        parameters: Dict[str, Any],
        requires_auth: bool = True
    ):
        self.name = name
        self.description = description
        self.category = category
        self.handler = handler
        self.parameters = parameters
        self.requires_auth = requires_auth

    def to_dict(self) -> dict:
        """Convert to LLM-friendly format for prompt"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Central registry of all available tools

    This class manages all tools that can be used by the LLM/ActionHandlers.
    Tools are organized by category for easy discovery.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_by_category(self, category: ToolCategory) -> list[Tool]:
        """List all tools in a category"""
        return [t for t in self.tools.values() if t.category == category]

    def get_all_descriptions(self) -> str:
        """
        Get all tool descriptions for LLM prompt

        Returns formatted string with all available tools.
        """
        desc = "Available tools:\n\n"
        for category in ToolCategory:
            tools = self.list_by_category(category)
            if tools:
                desc += f"## {category.value.upper()}\n"
                for tool in tools:
                    desc += f"- **{tool.name}**: {tool.description}\n"
                    param_str = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
                    desc += f"  Parameters: {param_str}\n\n"
        return desc

    async def call_tool(
        self,
        tool_name: str,
        parameters: dict,
        db: Session,
        user_id: Optional[int] = None
    ) -> Any:
        """
        Call a tool with parameters

        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            db: Database session
            user_id: User ID (for permission checks)

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
            PermissionError: If user doesn't have permission
        """
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Permission check (if required)
        if tool.requires_auth and user_id is None:
            raise PermissionError(f"Tool {tool_name} requires authentication")

        # Execute tool
        logger.info(f"Executing tool: {tool_name} with params: {parameters}")
        try:
            result = await tool.handler(parameters, db, user_id)
            logger.info(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}", exc_info=True)
            raise

    def _register_default_tools(self):
        """Register all default tools"""

        # ========== DATASET TOOLS ==========

        self.register(Tool(
            name="analyze_dataset",
            description="Analyze dataset structure, format, and quality",
            category=ToolCategory.DATASET,
            handler=self._analyze_dataset,
            parameters={
                "dataset_path": "str (required) - Path to dataset directory"
            },
            requires_auth=False
        ))

        self.register(Tool(
            name="list_datasets",
            description="List available datasets in a directory",
            category=ToolCategory.DATASET,
            handler=self._list_datasets,
            parameters={
                "base_path": "str (optional) - Base directory to search, defaults to C:/datasets"
            },
            requires_auth=False
        ))

        # ========== MODEL TOOLS ==========

        self.register(Tool(
            name="search_models",
            description="Search available models by filters",
            category=ToolCategory.MODEL,
            handler=self._search_models,
            parameters={
                "task_type": "str (optional) - Task type filter",
                "framework": "str (optional) - Framework filter (timm, ultralytics, transformers)",
                "tags": "list[str] (optional) - Tag filters"
            },
            requires_auth=False
        ))

        self.register(Tool(
            name="get_model_guide",
            description="Get comprehensive guide for a specific model",
            category=ToolCategory.MODEL,
            handler=self._get_model_guide,
            parameters={
                "framework": "str (required) - Framework name",
                "model_name": "str (required) - Model name"
            },
            requires_auth=False
        ))

        self.register(Tool(
            name="compare_models",
            description="Compare multiple models side by side",
            category=ToolCategory.MODEL,
            handler=self._compare_models,
            parameters={
                "model_specs": "list[dict] (required) - List of {framework, name} dicts"
            },
            requires_auth=False
        ))

        # ========== TRAINING TOOLS ==========

        self.register(Tool(
            name="get_training_status",
            description="Get current status and progress of a training job",
            category=ToolCategory.TRAINING,
            handler=self._get_training_status,
            parameters={
                "job_id": "int (required) - Training job ID"
            },
            requires_auth=False
        ))

        self.register(Tool(
            name="list_training_jobs",
            description="List training jobs with optional filters",
            category=ToolCategory.TRAINING,
            handler=self._list_training_jobs,
            parameters={
                "status": "str (optional) - Filter by status",
                "project_id": "int (optional) - Filter by project ID",
                "limit": "int (optional) - Maximum number of jobs, default 20"
            },
            requires_auth=False
        ))

        self.register(Tool(
            name="stop_training",
            description="Stop a running training job",
            category=ToolCategory.TRAINING,
            handler=self._stop_training,
            parameters={
                "job_id": "int (required) - Training job ID",
                "save_checkpoint": "bool (optional) - Save checkpoint before stopping, default True"
            },
            requires_auth=False
        ))

        # ========== INFERENCE TOOLS ==========

        self.register(Tool(
            name="run_quick_inference",
            description="Run quick inference on a single image",
            category=ToolCategory.INFERENCE,
            handler=self._run_quick_inference,
            parameters={
                "job_id": "int (required) - Training job ID to use for inference",
                "image_path": "str (required) - Path to image file"
            },
            requires_auth=False
        ))

        logger.info(f"Registered {len(self.tools)} default tools")

    # ========== TOOL HANDLERS ==========

    async def _analyze_dataset(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for analyze_dataset tool"""
        from app.utils.dataset_analyzer import analyze_dataset
        from pathlib import Path

        dataset_path = params.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required")

        if not Path(dataset_path).exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # Call existing dataset analyzer
        analysis = analyze_dataset(dataset_path)

        return {
            "path": dataset_path,
            "format": analysis.get("format", "unknown"),
            "num_classes": len(analysis.get("classes", [])),
            "classes": analysis.get("classes", []),
            "total_images": analysis.get("total_samples", 0),
            "class_distribution": analysis.get("class_distribution", {}),
            "dataset_info": analysis.get("dataset_info", {}),
            "suggestions": analysis.get("suggestions", [])
        }

    async def _list_datasets(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> list[dict]:
        """Handler for list_datasets tool"""
        from pathlib import Path
        import os

        datasets = []

        # Scan built-in datasets directory (supports Windows/Linux)
        # Default: C:/datasets (Windows dev), /app/datasets (Linux prod)
        datasets_path = os.getenv("DATASETS_PATH", "C:/datasets")
        builtin_path = Path(datasets_path)

        if builtin_path.exists():
            for item in builtin_path.iterdir():
                if item.is_dir():
                    datasets.append({
                        "name": f"{item.name} (기본 제공)",
                        "path": str(item.absolute()),
                        "category": "built-in",
                        "exists": True
                    })

        # Also scan user-provided base_path if different from default
        base_path = params.get("base_path")
        if base_path and base_path != datasets_path:
            base_dir = Path(base_path)
            if base_dir.exists():
                for item in base_dir.iterdir():
                    if item.is_dir():
                        datasets.append({
                            "name": item.name,
                            "path": str(item.absolute()),
                            "category": "user",
                            "exists": True
                        })

        return datasets

    async def _search_models(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> list[dict]:
        """Handler for search_models tool"""
        # For now, return static model list from capabilities
        # In future, this can query a model registry database

        task_type = params.get("task_type")
        framework = params.get("framework")

        # Import capabilities
        from app.api.chat import get_capabilities

        capabilities = await get_capabilities()
        models = capabilities["models"]

        # Filter by task_type
        if task_type:
            models = [m for m in models if task_type in m.get("task_types", [])]

        # Filter by framework
        if framework:
            models = [m for m in models if m.get("framework") == framework]

        return models

    async def _get_model_guide(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for get_model_guide tool"""
        framework = params.get("framework")
        model_name = params.get("model_name")

        if not framework or not model_name:
            raise ValueError("framework and model_name are required")

        # TODO: Implement model guide retrieval
        # For now, return basic info

        return {
            "framework": framework,
            "model_name": model_name,
            "description": f"Model guide for {model_name}",
            "available": True
        }

    async def _compare_models(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for compare_models tool"""
        model_specs = params.get("model_specs", [])

        if not model_specs:
            raise ValueError("model_specs is required")

        # TODO: Implement model comparison
        return {
            "models": model_specs,
            "comparison": "Model comparison feature coming soon"
        }

    async def _get_training_status(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for get_training_status tool"""
        from app.db.models import TrainingJob

        job_id = params.get("job_id")
        if not job_id:
            raise ValueError("job_id is required")

        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        # Get latest metrics
        latest_metrics = {}
        if job.metrics:
            latest = sorted(job.metrics, key=lambda m: m.epoch, reverse=True)[0] if job.metrics else None
            if latest:
                latest_metrics = {
                    "epoch": latest.epoch,
                    "loss": latest.loss,
                    "accuracy": latest.accuracy,
                    "val_loss": latest.val_loss,
                    "val_accuracy": latest.val_accuracy
                }

        return {
            "job_id": job.id,
            "status": job.status,
            "model": job.model_name,
            "framework": job.framework,
            "current_epoch": job.current_epoch or 0,
            "total_epochs": job.epochs,
            "progress_percent": (job.current_epoch / job.epochs * 100) if job.epochs > 0 else 0,
            "latest_metrics": latest_metrics,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None
        }

    async def _list_training_jobs(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> list[dict]:
        """Handler for list_training_jobs tool"""
        from app.db.models import TrainingJob

        status = params.get("status")
        project_id = params.get("project_id")
        limit = params.get("limit", 20)

        query = db.query(TrainingJob)

        if status:
            query = query.filter(TrainingJob.status == status)

        if project_id:
            query = query.filter(TrainingJob.project_id == project_id)

        if user_id:
            query = query.filter(TrainingJob.creator_id == user_id)

        jobs = query.order_by(TrainingJob.created_at.desc()).limit(limit).all()

        return [
            {
                "job_id": job.id,
                "model": job.model_name,
                "task_type": job.task_type,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "final_metric": job.final_accuracy
            }
            for job in jobs
        ]

    async def _stop_training(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for stop_training tool"""
        from app.db.models import TrainingJob
        import signal

        job_id = params.get("job_id")
        save_checkpoint = params.get("save_checkpoint", True)

        if not job_id:
            raise ValueError("job_id is required")

        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        if job.status != "running":
            return {
                "job_id": job_id,
                "status": job.status,
                "message": f"Job is not running (current status: {job.status})"
            }

        # Stop the training process
        if job.process_id:
            try:
                import os
                import psutil

                process = psutil.Process(job.process_id)
                process.send_signal(signal.SIGTERM)

                job.status = "stopped"
                db.commit()

                return {
                    "job_id": job_id,
                    "status": "stopped",
                    "message": "Training stopped successfully"
                }
            except Exception as e:
                logger.error(f"Failed to stop training: {e}")
                return {
                    "job_id": job_id,
                    "error": str(e),
                    "message": "Failed to stop training"
                }
        else:
            return {
                "job_id": job_id,
                "error": "No process ID found",
                "message": "Cannot stop training - no process ID"
            }

    async def _run_quick_inference(
        self,
        params: dict,
        db: Session,
        user_id: Optional[int]
    ) -> dict:
        """Handler for run_quick_inference tool"""
        from app.db.models import TrainingJob
        from pathlib import Path

        job_id = params.get("job_id")
        image_path = params.get("image_path")

        if not job_id or not image_path:
            raise ValueError("job_id and image_path are required")

        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        if not Path(image_path).exists():
            raise ValueError(f"Image not found: {image_path}")

        # TODO: Implement actual inference
        # For now, return placeholder

        return {
            "job_id": job_id,
            "image_path": image_path,
            "predictions": [],
            "message": "Quick inference feature coming soon"
        }


# Global instance
tool_registry = ToolRegistry()

"""
Unit tests for Tool Registry

Tests tool registration, retrieval, and execution.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from sqlalchemy.orm import Session

from app.utils.tool_registry import (
    Tool,
    ToolCategory,
    ToolRegistry,
    tool_registry
)


class TestToolClass:
    """Test Tool class"""

    def test_tool_creation(self):
        """Test creating a tool"""
        handler = AsyncMock()
        tool = Tool(
            name="test_tool",
            description="Test tool description",
            category=ToolCategory.DATASET,
            handler=handler,
            parameters={"param1": "string"},
            requires_auth=True
        )

        assert tool.name == "test_tool"
        assert tool.description == "Test tool description"
        assert tool.category == ToolCategory.DATASET
        assert tool.handler == handler
        assert tool.parameters == {"param1": "string"}
        assert tool.requires_auth is True

    def test_tool_to_dict(self):
        """Test tool to_dict conversion"""
        handler = AsyncMock()
        tool = Tool(
            name="test_tool",
            description="Test description",
            category=ToolCategory.MODEL,
            handler=handler,
            parameters={"param1": "string", "param2": "int"}
        )

        result = tool.to_dict()

        assert result["name"] == "test_tool"
        assert result["description"] == "Test description"
        assert result["category"] == "model"
        assert result["parameters"] == {"param1": "string", "param2": "int"}
        assert "handler" not in result  # Handler should not be in dict


class TestToolRegistry:
    """Test ToolRegistry class"""

    @pytest.fixture
    def fresh_registry(self):
        """Create a fresh registry for testing"""
        registry = ToolRegistry()
        # Clear default tools for isolated testing
        registry.tools.clear()
        return registry

    def test_registry_initialization(self):
        """Test registry initializes with default tools"""
        registry = ToolRegistry()

        # Should have default tools registered
        assert len(registry.tools) > 0

        # Check specific tools exist
        assert registry.get("analyze_dataset") is not None
        assert registry.get("list_datasets") is not None
        assert registry.get("search_models") is not None
        assert registry.get("get_training_status") is not None

    def test_register_tool(self, fresh_registry):
        """Test registering a new tool"""
        handler = AsyncMock()
        tool = Tool(
            name="custom_tool",
            description="Custom test tool",
            category=ToolCategory.TRAINING,
            handler=handler,
            parameters={"test": "param"}
        )

        fresh_registry.register(tool)

        assert fresh_registry.get("custom_tool") is not None
        assert fresh_registry.get("custom_tool").name == "custom_tool"

    def test_get_tool(self, fresh_registry):
        """Test retrieving a tool by name"""
        handler = AsyncMock()
        tool = Tool(
            name="test_tool",
            description="Test",
            category=ToolCategory.DATASET,
            handler=handler,
            parameters={}
        )
        fresh_registry.register(tool)

        retrieved = fresh_registry.get("test_tool")

        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_nonexistent_tool(self, fresh_registry):
        """Test retrieving a tool that doesn't exist"""
        result = fresh_registry.get("nonexistent_tool")
        assert result is None

    def test_list_by_category(self, fresh_registry):
        """Test listing tools by category"""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        handler3 = AsyncMock()

        tool1 = Tool("tool1", "desc1", ToolCategory.DATASET, handler1, {})
        tool2 = Tool("tool2", "desc2", ToolCategory.DATASET, handler2, {})
        tool3 = Tool("tool3", "desc3", ToolCategory.MODEL, handler3, {})

        fresh_registry.register(tool1)
        fresh_registry.register(tool2)
        fresh_registry.register(tool3)

        dataset_tools = fresh_registry.list_by_category(ToolCategory.DATASET)
        model_tools = fresh_registry.list_by_category(ToolCategory.MODEL)

        assert len(dataset_tools) == 2
        assert len(model_tools) == 1
        assert all(t.category == ToolCategory.DATASET for t in dataset_tools)

    def test_get_all_descriptions(self, fresh_registry):
        """Test getting all tool descriptions for LLM prompt"""
        handler = AsyncMock()

        tool1 = Tool("tool1", "First tool", ToolCategory.DATASET, handler, {"p1": "str"})
        tool2 = Tool("tool2", "Second tool", ToolCategory.MODEL, handler, {"p2": "int"})

        fresh_registry.register(tool1)
        fresh_registry.register(tool2)

        descriptions = fresh_registry.get_all_descriptions()

        assert "Available tools:" in descriptions
        assert "DATASET" in descriptions
        assert "MODEL" in descriptions
        assert "tool1" in descriptions
        assert "First tool" in descriptions
        assert "tool2" in descriptions
        assert "Second tool" in descriptions

    @pytest.mark.asyncio
    async def test_call_tool_success(self, fresh_registry, db_session):
        """Test successfully calling a tool"""
        # Create mock handler that returns expected result
        async def mock_handler(params, db, user_id):
            return {"result": "success", "data": params}

        tool = Tool(
            name="test_tool",
            description="Test",
            category=ToolCategory.DATASET,
            handler=mock_handler,
            parameters={"param1": "string"},
            requires_auth=False
        )
        fresh_registry.register(tool)

        result = await fresh_registry.call_tool(
            "test_tool",
            {"param1": "value1"},
            db_session,
            user_id=None
        )

        assert result["result"] == "success"
        assert result["data"]["param1"] == "value1"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, fresh_registry, db_session):
        """Test calling a non-existent tool raises error"""
        with pytest.raises(ValueError, match="Tool not found"):
            await fresh_registry.call_tool(
                "nonexistent_tool",
                {},
                db_session,
                user_id=None
            )

    @pytest.mark.asyncio
    async def test_call_tool_requires_auth(self, fresh_registry, db_session):
        """Test calling a tool that requires auth without user_id"""
        handler = AsyncMock()
        tool = Tool(
            name="auth_tool",
            description="Requires auth",
            category=ToolCategory.TRAINING,
            handler=handler,
            parameters={},
            requires_auth=True
        )
        fresh_registry.register(tool)

        with pytest.raises(PermissionError, match="requires authentication"):
            await fresh_registry.call_tool(
                "auth_tool",
                {},
                db_session,
                user_id=None
            )

    @pytest.mark.asyncio
    async def test_call_tool_with_auth(self, fresh_registry, db_session):
        """Test calling a tool with proper authentication"""
        async def mock_handler(params, db, user_id):
            return {"user": user_id, "authenticated": True}

        tool = Tool(
            name="auth_tool",
            description="Requires auth",
            category=ToolCategory.TRAINING,
            handler=mock_handler,
            parameters={},
            requires_auth=True
        )
        fresh_registry.register(tool)

        result = await fresh_registry.call_tool(
            "auth_tool",
            {},
            db_session,
            user_id=123
        )

        assert result["user"] == 123
        assert result["authenticated"] is True

    @pytest.mark.asyncio
    async def test_call_tool_handler_error(self, fresh_registry, db_session):
        """Test tool handler raises error"""
        async def failing_handler(params, db, user_id):
            raise RuntimeError("Tool execution failed")

        tool = Tool(
            name="failing_tool",
            description="Will fail",
            category=ToolCategory.DATASET,
            handler=failing_handler,
            parameters={},
            requires_auth=False
        )
        fresh_registry.register(tool)

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await fresh_registry.call_tool(
                "failing_tool",
                {},
                db_session,
                user_id=None
            )


class TestDefaultTools:
    """Test default tools registered in tool_registry"""

    @pytest.mark.asyncio
    async def test_analyze_dataset_tool(self, db_session, tmp_path):
        """Test analyze_dataset tool"""
        # Create mock dataset directory
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        class1_dir = dataset_dir / "class1"
        class1_dir.mkdir()
        class2_dir = dataset_dir / "class2"
        class2_dir.mkdir()

        # Create some dummy image files
        (class1_dir / "img1.jpg").write_text("fake image")
        (class2_dir / "img2.jpg").write_text("fake image")

        with patch("app.utils.dataset_analyzer.analyze_dataset") as mock_analyze:
            mock_analyze.return_value = {
                "format": "imagefolder",
                "classes": ["class1", "class2"],
                "total_samples": 2,
                "class_distribution": {"class1": 1, "class2": 1},
                "dataset_info": {},
                "suggestions": []
            }

            result = await tool_registry.call_tool(
                "analyze_dataset",
                {"dataset_path": str(dataset_dir)},
                db_session,
                user_id=None
            )

            assert result["format"] == "imagefolder"
            assert result["num_classes"] == 2
            assert "class1" in result["classes"]
            assert "class2" in result["classes"]

    @pytest.mark.asyncio
    async def test_list_datasets_tool(self, db_session, tmp_path):
        """Test list_datasets tool"""
        # Create mock base directory with datasets
        base_dir = tmp_path / "datasets"
        base_dir.mkdir()
        (base_dir / "dataset1").mkdir()
        (base_dir / "dataset2").mkdir()

        result = await tool_registry.call_tool(
            "list_datasets",
            {"base_path": str(base_dir)},
            db_session,
            user_id=None
        )

        assert len(result) == 2
        assert any(d["name"] == "dataset1" for d in result)
        assert any(d["name"] == "dataset2" for d in result)

    @pytest.mark.asyncio
    async def test_search_models_tool(self, db_session):
        """Test search_models tool"""
        with patch("app.api.chat.get_capabilities") as mock_cap:
            mock_cap.return_value = {
                "models": [
                    {
                        "name": "resnet50",
                        "framework": "timm",
                        "task_types": ["classification"]
                    },
                    {
                        "name": "yolov8n",
                        "framework": "ultralytics",
                        "task_types": ["object_detection"]
                    }
                ]
            }

            # Search by task type
            result = await tool_registry.call_tool(
                "search_models",
                {"task_type": "classification"},
                db_session,
                user_id=None
            )

            assert len(result) >= 1
            assert all("classification" in m.get("task_types", []) for m in result)

    @pytest.mark.asyncio
    async def test_get_training_status_tool(self, db_session):
        """Test get_training_status tool"""
        from app.db.models import TrainingJob

        # Create a training job
        job = TrainingJob(
            session_id=1,
            framework="timm",
            model_name="resnet50",
            task_type="classification",
            dataset_path="/test/path",
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            status="running",
            current_epoch=5
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)

        result = await tool_registry.call_tool(
            "get_training_status",
            {"job_id": job.id},
            db_session,
            user_id=None
        )

        assert result["job_id"] == job.id
        assert result["status"] == "running"
        assert result["model"] == "resnet50"
        assert result["current_epoch"] == 5
        assert result["total_epochs"] == 10
        assert result["progress_percent"] == 50.0

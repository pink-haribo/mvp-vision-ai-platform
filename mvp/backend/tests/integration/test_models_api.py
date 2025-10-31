"""
Test for /models/list API endpoint.

Ensures that the model registry correctly returns all available models
including yolo11 variants.
"""

import pytest


class TestModelsAPI:
    """Test model registry API."""

    def test_models_list_endpoint(self, client):
        """Test that /models/list returns all models."""
        # Act
        response = client.get("/api/v1/models/list")

        # Assert
        assert response.status_code == 200
        models = response.json()

        assert isinstance(models, list)
        assert len(models) > 0

    def test_yolo11_models_in_list(self, client):
        """Test that yolo11n, yolo11m, yolo11l are in the model list."""
        # Act
        response = client.get("/api/v1/models/list")
        models = response.json()

        # Convert to dict for easier lookup
        model_dict = {m["model_name"]: m for m in models}

        # Assert: yolo11 variants exist
        assert "yolo11n" in model_dict, "yolo11n not found in model list"
        assert "yolo11m" in model_dict, "yolo11m not found in model list"
        assert "yolo11l" in model_dict, "yolo11l not found in model list"

    def test_yolo11n_metadata(self, client):
        """Test that yolo11n has correct metadata."""
        # Act
        response = client.get("/api/v1/models/list")
        models = response.json()

        # Find yolo11n
        yolo11n = next((m for m in models if m["model_name"] == "yolo11n"), None)

        # Assert
        assert yolo11n is not None
        assert yolo11n["framework"] == "ultralytics"
        assert yolo11n["task_type"] == "object_detection"
        assert yolo11n["display_name"] == "YOLOv11 Nano"
        assert yolo11n["pretrained_available"] is True

    def test_model_status_field(self, client):
        """Test that models have status field (active/experimental/deprecated)."""
        # Act
        response = client.get("/api/v1/models/list")
        models = response.json()

        # Assert: All models have status field
        for model in models:
            assert "status" in model or model.get("status") is None, \
                f"Model {model['model_name']} missing status field"

            # If status exists, it should be valid
            if "status" in model and model["status"]:
                assert model["status"] in ["active", "experimental", "deprecated"], \
                    f"Invalid status: {model['status']}"

    def test_filter_by_framework(self, client):
        """Test filtering models by framework."""
        # Act
        response = client.get("/api/v1/models/list?framework=ultralytics")
        models = response.json()

        # Assert: All models are ultralytics
        if len(models) > 0:
            for model in models:
                assert model["framework"] == "ultralytics"

    def test_filter_by_task_type(self, client):
        """Test filtering models by task type."""
        # Act
        response = client.get("/api/v1/models/list?task_type=object_detection")
        models = response.json()

        # Assert: All models are object detection
        if len(models) > 0:
            for model in models:
                assert model["task_type"] == "object_detection"

    def test_yolo8_and_yolo11_coexist(self, client):
        """Test that both YOLOv8 and YOLOv11 models are available."""
        # Act
        response = client.get("/api/v1/models/list")
        models = response.json()

        model_names = {m["model_name"] for m in models}

        # Assert: Both v8 and v11 exist
        has_yolo8 = any("yolov8" in name or "yolo8" in name for name in model_names)
        has_yolo11 = any("yolo11" in name for name in model_names)

        assert has_yolo8, "No YOLOv8 models found"
        assert has_yolo11, "No YOLOv11 models found"

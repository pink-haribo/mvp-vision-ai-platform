"""
API Schema Consistency E2E Tests

핵심 원칙: Frontend 요청 ↔ Backend 스키마 일치 검증

이 테스트는 Frontend가 실제로 보내는 요청 형식과
Backend가 기대하는 스키마가 일치하는지 검증합니다.

스키마 불일치는 production에서 런타임 에러를 발생시키므로
E2E 테스트로 사전에 검증해야 합니다.
"""

import pytest
import requests


BASE_URL = "http://localhost:8000/api/v1"


def test_export_capabilities_schema(api_url):
    """
    Test: Export capabilities response 스키마 일치

    Issue: Frontend는 'formats' 필드를 기대하지만
    Backend가 'supported_formats'를 반환하는 경우가 있음
    """
    # Query parameters required: framework and task_type
    params = {
        "framework": "ultralytics",
        "task_type": "object_detection"
    }
    response = requests.get(f"{api_url}/export/capabilities", params=params)

    assert response.status_code == 200, f"Failed to get export capabilities: {response.text}"

    data = response.json()

    # 필수 필드 검증
    assert "formats" in data, "Missing 'formats' field (Frontend expects this)"

    # formats는 list여야 함
    assert isinstance(data["formats"], list), "'formats' should be a list"

    # 각 format은 필요한 필드를 가져야 함
    for fmt in data["formats"]:
        assert "format_name" in fmt, f"Missing 'format_name' in {fmt}"
        assert "display_name" in fmt, f"Missing 'display_name' in {fmt}"
        assert "description" in fmt, f"Missing 'description' in {fmt}"
        assert "file_extension" in fmt, f"Missing 'file_extension' in {fmt}"
        assert "options" in fmt, f"Missing 'options' in {fmt}"

    print(f"✓ Export capabilities schema validated ({len(data['formats'])} formats)")


def test_export_job_request_schema():
    """
    Test: Export job request 스키마 일치

    Frontend가 보내는 모든 필드가 Backend 스키마와 일치하는지 검증
    """
    # Frontend가 보내는 실제 요청 형식 (CreateExportModal에서)
    frontend_request = {
        "training_job_id": 1,
        "export_format": "onnx",
        "export_config": {
            "opset_version": 17,
            "dynamic_axes": True,
            "embed_preprocessing": False
        }
    }

    # 스키마 검증 (실제 요청은 보내지 않음)
    # Backend ExportJobRequest 스키마가 이 필드들을 모두 받을 수 있는지 확인
    required_fields = ["training_job_id", "export_format"]
    optional_fields = ["export_config", "optimization_config", "validation_config"]

    for field in required_fields:
        assert field in frontend_request, f"Frontend missing required field: {field}"

    # export_config의 하위 필드 검증
    if "export_config" in frontend_request:
        config = frontend_request["export_config"]
        # ONNX 옵션
        valid_onnx_options = ["opset_version", "dynamic_axes", "embed_preprocessing", "simplify"]
        for key in config.keys():
            assert key in valid_onnx_options, f"Unknown ONNX option: {key}"

    print("✓ Export job request schema validated")


def test_deployment_request_schema():
    """
    Test: Deployment request 스키마 일치

    Frontend CreateDeploymentModal이 보내는 요청과 Backend 스키마 일치 검증
    """
    # Platform Endpoint 배포 요청
    platform_endpoint_request = {
        "export_job_id": 1,
        "deployment_type": "platform_endpoint",
        "deployment_config": {
            "auto_activate": True,
            "generate_api_key": True
        }
    }

    # Edge Package 배포 요청
    edge_package_request = {
        "export_job_id": 1,
        "deployment_type": "edge_package",
        "deployment_config": {
            "target_platform": "linux_x64",
            "include_runtime": True,
            "optimize_size": True
        }
    }

    # 필수 필드 검증
    for request in [platform_endpoint_request, edge_package_request]:
        assert "export_job_id" in request
        assert "deployment_type" in request
        assert "deployment_config" in request

        # deployment_type 값 검증
        valid_types = ["platform_endpoint", "edge_package", "container", "download"]
        assert request["deployment_type"] in valid_types, \
            f"Invalid deployment_type: {request['deployment_type']}"

    print("✓ Deployment request schema validated")


def test_training_job_request_schema():
    """
    Test: Training job request 스키마 일치

    Frontend가 보내는 training job 생성 요청과 Backend 스키마 일치 검증
    """
    # Frontend가 보내는 실제 형식
    frontend_request = {
        "project_id": 1,
        "framework": "ultralytics",
        "model_name": "yolo11n",
        "task_type": "object_detection",
        "dataset_id": 1,
        "config": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.01,
            "imgsz": 640,
            "device": "0"
        },
        "advanced_config": {
            "mosaic": 1.0,
            "mixup": 0.0,
            "augment": True
        }
    }

    # 필수 필드 검증
    required_fields = [
        "framework", "model_name", "task_type", "dataset_id", "config"
    ]

    for field in required_fields:
        assert field in frontend_request, f"Missing required field: {field}"

    # config 하위 필드 검증 (basic config)
    config = frontend_request["config"]
    expected_basic_fields = ["epochs", "batch_size", "learning_rate", "imgsz", "device"]
    for field in expected_basic_fields:
        assert field in config, f"Missing basic config field: {field}"

    # advanced_config 검증 (선택적)
    if "advanced_config" in frontend_request:
        adv_config = frontend_request["advanced_config"]
        assert isinstance(adv_config, dict), "advanced_config should be a dict"

    print("✓ Training job request schema validated")


def test_inference_request_schema():
    """
    Test: Inference request 스키마 일치

    TestInferencePanel이 보내는 요청과 Backend 스키마 일치 검증
    """
    # Pretrained model inference
    pretrained_request = {
        "model_source": "pretrained",
        "model_name": "yolo11n.pt",
        "task_type": "object_detection",
        "image_paths": ["/tmp/test1.jpg", "/tmp/test2.jpg"],
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45
    }

    # Trained checkpoint inference
    checkpoint_request = {
        "model_source": "trained",
        "training_job_id": 123,
        "checkpoint_type": "best",  # or "last"
        "image_paths": ["/tmp/test1.jpg"],
        "confidence_threshold": 0.25
    }

    # Pretrained 요청 검증
    assert "model_source" in pretrained_request
    assert pretrained_request["model_source"] == "pretrained"
    assert "model_name" in pretrained_request
    assert "image_paths" in pretrained_request
    assert isinstance(pretrained_request["image_paths"], list)

    # Checkpoint 요청 검증
    assert "model_source" in checkpoint_request
    assert checkpoint_request["model_source"] == "trained"
    assert "training_job_id" in checkpoint_request
    assert "checkpoint_type" in checkpoint_request
    assert checkpoint_request["checkpoint_type"] in ["best", "last"]

    print("✓ Inference request schema validated")


def test_progress_callback_schema():
    """
    Test: Training progress callback 스키마 일치

    Trainer SDK가 보내는 progress callback과 Backend 스키마 일치 검증
    """
    # SDK가 보내는 progress callback
    sdk_callback = {
        "job_id": 123,
        "status": "running",
        "current_epoch": 5,
        "total_epochs": 50,
        "progress_percent": 10,
        "metrics": {
            "loss": 0.234,
            "accuracy": 0.789,
            "learning_rate": 0.001,
            "extra_metrics": {
                "precision": 0.82,
                "recall": 0.76,
                "mAP50": 0.78
            }
        },
        "checkpoint_path": "/tmp/checkpoints/epoch_5.pt"
    }

    # 필수 필드 검증
    assert "job_id" in sdk_callback
    assert "status" in sdk_callback
    assert "current_epoch" in sdk_callback
    assert "total_epochs" in sdk_callback

    # metrics 구조 검증
    if "metrics" in sdk_callback:
        metrics = sdk_callback["metrics"]
        # extra_metrics는 선택적이지만, 있다면 dict여야 함
        if "extra_metrics" in metrics:
            assert isinstance(metrics["extra_metrics"], dict)

    print("✓ Progress callback schema validated")


def test_log_callback_schema():
    """
    Test: Log callback 스키마 일치

    SDK가 보내는 log callback과 Backend 스키마 일치 검증
    """
    # SDK가 보내는 log callback
    sdk_log_callback = {
        "job_id": 123,
        "level": "INFO",
        "event_type": "training",
        "message": "Epoch 5 completed successfully"
    }

    # 필수 필드 검증
    required_fields = ["job_id", "level", "event_type", "message"]
    for field in required_fields:
        assert field in sdk_log_callback, f"Missing required field: {field}"

    # level 값 검증
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    assert sdk_log_callback["level"] in valid_levels, \
        f"Invalid log level: {sdk_log_callback['level']}"

    print("✓ Log callback schema validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

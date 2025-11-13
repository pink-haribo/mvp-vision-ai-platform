# Test Coverage Plan

이 문서는 Vision AI Training Platform의 포괄적인 테스트 커버리지 계획입니다.

## 현재 상태 (2025-01-31)

### ✅ 완료된 테스트 (P0/P1)

**P0: 버그 회귀 테스트**
- ✅ test_yolo11n_bug.py (5 tests)
- ✅ test_encoding.py (5 tests)
- ✅ test_adapter_imports.py (9 tests, 2 skipped)

**P1: 기본 API 테스트**
- ✅ test_models_api.py (7 tests)

**커버리지: ~10%** (24 passed / 예상 240+ tests)

---

## P2: 모델 레지스트리 심화 테스트 (필수)

### test_model_loading.py
실제 모델 로딩 및 초기화를 테스트합니다.

```python
class TestModelLoading:
    """Test actual model loading and initialization."""

    def test_yolov8n_loads_successfully(self):
        """Test that yolov8n model can be loaded."""
        # Arrange
        model_name = "yolov8n"

        # Act
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")

        # Assert
        assert model is not None
        assert model.model is not None

    def test_yolo11n_loads_successfully(self):
        """Test that yolo11n model can be loaded."""
        # Similar to yolov8n

    def test_pretrained_weight_paths(self):
        """Test that all pretrained weight paths are correct."""
        # Verify weight file naming conventions
        models = ["yolov8n", "yolo11n", "yolov8s-seg", "yolo11m"]
        for model in models:
            # Check if weight file exists or can be downloaded
            pass

    def test_invalid_model_name_error(self):
        """Test that invalid model name raises appropriate error."""
        with pytest.raises(Exception):
            YOLO("invalid_model_name.pt")

    def test_model_registry_consistency(self):
        """Test that model registry has no duplicate or conflicting entries."""
        # Check for duplicates, naming conflicts
        pass
```

**우선순위:** P2
**예상 테스트 수:** 15+
**예상 구현 시간:** 2-3시간

---

## P2: Advanced Config 테스트 (필수)

### test_training_config.py
고급 학습 설정이 올바르게 적용되는지 테스트합니다.

```python
class TestAdvancedConfig:
    """Test advanced training configuration."""

    def test_optimizer_config_applied(self):
        """Test that optimizer config is correctly applied."""
        config = {
            "optimizer": {
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0001
            }
        }
        # Verify optimizer is created with correct params

    def test_scheduler_config_applied(self):
        """Test learning rate scheduler configuration."""
        config = {
            "scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": 100
            }
        }
        # Verify scheduler works correctly

    def test_augmentation_config_applied(self):
        """Test data augmentation configuration."""
        config = {
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "degrees": 10.0,
                "translate": 0.1
            }
        }
        # Verify augmentation is applied to training data

    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = {
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001
            }
        }
        # Verify early stopping works

    def test_checkpoint_config(self):
        """Test checkpoint saving configuration."""
        config = {
            "save_period": 5,
            "save_best": True
        }
        # Verify checkpoints are saved correctly
```

**우선순위:** P2
**예상 테스트 수:** 20+
**예상 구현 시간:** 3-4시간

---

## P2: Inference 함수 테스트 (필수)

### test_inference_pretrained.py
Pretrained 모델로 추론을 테스트합니다.

```python
class TestPretrainedInference:
    """Test inference with pretrained models."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        return str(img_path)

    def test_yolov8n_detection_pretrained(self, sample_image):
        """Test YOLOv8n detection with pretrained weights."""
        # Run inference
        # Verify results format
        pass

    def test_yolo11n_detection_pretrained(self, sample_image):
        """Test YOLO11n detection with pretrained weights."""
        pass

    def test_yolov8n_seg_pretrained(self, sample_image):
        """Test YOLOv8n-seg segmentation with pretrained weights."""
        pass

    def test_inference_batch(self, tmp_path):
        """Test batch inference with multiple images."""
        # Create multiple images
        # Run batch inference
        # Verify all results returned
        pass

    def test_inference_single(self, sample_image):
        """Test single image inference."""
        pass
```

### test_inference_checkpoint.py
학습된 체크포인트로 추론을 테스트합니다.

```python
class TestCheckpointInference:
    """Test inference with trained checkpoints."""

    @pytest.fixture
    def trained_checkpoint(self):
        """Create or use a trained checkpoint."""
        # Mock a trained checkpoint or use a small pre-trained one
        pass

    def test_inference_with_best_checkpoint(self, trained_checkpoint, sample_image):
        """Test inference using best.pt checkpoint."""
        pass

    def test_inference_with_last_checkpoint(self, trained_checkpoint, sample_image):
        """Test inference using last.pt checkpoint."""
        pass

    def test_checkpoint_not_found_error(self, sample_image):
        """Test appropriate error when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            # Try to run inference with non-existent checkpoint
            pass
```

**우선순위:** P2
**예상 테스트 수:** 25+
**예상 구현 시간:** 4-5시간

---

## P2: Task별 결과 형식 테스트 (필수)

### test_inference_output_format.py
각 task의 결과 형식을 검증합니다.

```python
class TestInferenceOutputFormat:
    """Test that inference outputs are in correct format for each task."""

    def test_detection_output_format(self):
        """Test object detection output format."""
        # Expected format:
        # {
        #     "boxes": [[x1, y1, x2, y2], ...],
        #     "confidences": [0.95, ...],
        #     "class_ids": [0, ...],
        #     "class_names": ["person", ...]
        # }
        result = run_detection_inference()

        assert "boxes" in result
        assert "confidences" in result
        assert "class_ids" in result
        assert len(result["boxes"]) == len(result["confidences"])

        # Validate data types
        assert all(isinstance(box, list) for box in result["boxes"])
        assert all(len(box) == 4 for box in result["boxes"])
        assert all(isinstance(conf, float) for conf in result["confidences"])

    def test_segmentation_output_format(self):
        """Test segmentation output format."""
        # Expected format:
        # {
        #     "masks": [np.array([[0, 1, ...], ...]), ...],
        #     "boxes": [[x1, y1, x2, y2], ...],
        #     "confidences": [0.95, ...]
        # }
        pass

    def test_classification_output_format(self):
        """Test classification output format."""
        # Expected format:
        # {
        #     "class_id": 5,
        #     "class_name": "dog",
        #     "confidence": 0.98,
        #     "top_k": [
        #         {"class_id": 5, "class_name": "dog", "confidence": 0.98},
        #         {"class_id": 3, "class_name": "cat", "confidence": 0.01},
        #         ...
        #     ]
        # }
        pass

    def test_pose_output_format(self):
        """Test pose estimation output format."""
        # Expected format:
        # {
        #     "keypoints": [[[x1, y1, conf], [x2, y2, conf], ...], ...],
        #     "boxes": [[x1, y1, x2, y2], ...],
        #     "confidences": [0.95, ...]
        # }
        pass
```

**우선순위:** P2
**예상 테스트 수:** 15+
**예상 구현 시간:** 2-3시간

---

## P3: 학습 프로세스 전체 테스트 (중요, 시간 소요)

### test_training_lifecycle.py
전체 학습 lifecycle을 테스트합니다.

```python
class TestTrainingLifecycle:
    """Test complete training lifecycle."""

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Create a tiny dataset for fast training."""
        # Create 10 images with annotations
        # YOLO format: images/ and labels/
        pass

    def test_complete_training_flow(self, tiny_dataset):
        """Test complete training from start to finish."""
        # 1. Create training job
        response = client.post("/api/v1/training/jobs", json={
            "config": {
                "model_name": "yolov8n",
                "dataset_path": str(tiny_dataset),
                "epochs": 2,  # Very short for testing
                "batch_size": 2
            }
        })
        job_id = response.json()["id"]

        # 2. Wait for training to complete (with timeout)
        # Poll status until "completed"

        # 3. Verify job completed successfully
        job = get_job(job_id)
        assert job["status"] == "completed"

    def test_training_metrics_collected(self, tiny_dataset):
        """Test that training metrics are collected during training."""
        # Run short training
        # Check that metrics are saved to database
        metrics = get_metrics(job_id)

        assert len(metrics) > 0
        assert "loss" in metrics[0]
        assert "epoch" in metrics[0]

    def test_checkpoint_saved(self, tiny_dataset):
        """Test that checkpoints are saved during training."""
        # Run training
        # Verify best.pt and last.pt exist
        job = get_job(job_id)

        best_checkpoint = Path(job["checkpoint_dir"]) / "best.pt"
        last_checkpoint = Path(job["checkpoint_dir"]) / "last.pt"

        assert best_checkpoint.exists()
        assert last_checkpoint.exists()

    def test_mlflow_artifacts_saved(self, tiny_dataset):
        """Test that MLflow artifacts are saved."""
        # Run training
        # Check MLflow for artifacts
        import mlflow

        run = mlflow.get_run(job["mlflow_run_id"])
        artifacts = mlflow.list_artifacts(run.info.run_id)

        assert len(artifacts) > 0

    def test_validation_metrics_computed(self, tiny_dataset):
        """Test that validation metrics are computed."""
        # Run training with validation split
        # Check validation metrics
        val_metrics = get_validation_metrics(job_id)

        assert "val_loss" in val_metrics
        assert "mAP" in val_metrics
```

### test_training_metrics.py
학습 메트릭의 정확성을 테스트합니다.

```python
class TestTrainingMetrics:
    """Test training metrics accuracy and storage."""

    def test_loss_decreases_over_epochs(self, tiny_dataset):
        """Test that loss generally decreases over training."""
        # Run training
        metrics = get_all_metrics(job_id)

        # Get loss values
        losses = [m["loss"] for m in metrics]

        # Loss should generally trend downward
        # (Not strict because of small dataset)
        assert losses[-1] < losses[0] * 1.5  # Allow some variance

    def test_metrics_saved_per_epoch(self, tiny_dataset):
        """Test that metrics are saved for each epoch."""
        epochs = 3
        # Run training with 3 epochs

        metrics = get_all_metrics(job_id)

        # Should have metrics for each epoch
        assert len(metrics) >= epochs

    def test_image_paths_in_metrics(self, tiny_dataset):
        """Test that image paths are correctly stored in metrics."""
        # Run training
        metrics = get_all_metrics(job_id)

        # Check that image paths are valid
        if "image_paths" in metrics[0]:
            for path in metrics[0]["image_paths"]:
                assert Path(path).exists()
```

**우선순위:** P3
**예상 테스트 수:** 30+
**예상 구현 시간:** 8-10시간 (학습이 시간이 걸리므로)

---

## P3: 통합 테스트 (End-to-End)

### test_e2e_workflows.py
실제 사용자 워크플로우를 테스트합니다.

```python
class TestEndToEndWorkflows:
    """Test complete user workflows."""

    def test_complete_detection_workflow(self):
        """Test: Upload dataset → Train → Inference → Validate."""
        # 1. Upload dataset
        # 2. Create training job
        # 3. Wait for completion
        # 4. Run inference on test images
        # 5. Validate results
        pass

    def test_complete_classification_workflow(self):
        """Test classification from start to finish."""
        pass

    def test_multiple_jobs_concurrent(self):
        """Test running multiple training jobs concurrently."""
        # Create 3 jobs
        # Verify they all complete successfully
        pass
```

**우선순위:** P3
**예상 테스트 수:** 10+
**예상 구현 시간:** 4-5시간

---

## 테스트 구현 우선순위 및 일정

### Week 1: P2 Core Functionality
1. **Day 1-2:** test_model_loading.py (모델 레지스트리)
2. **Day 3-4:** test_inference_pretrained.py (Pretrained 추론)
3. **Day 5:** test_inference_output_format.py (결과 형식)

### Week 2: P2 Advanced Features
1. **Day 1-2:** test_training_config.py (Advanced Config)
2. **Day 3-4:** test_inference_checkpoint.py (체크포인트 추론)
3. **Day 5:** 통합 테스트 및 버그 수정

### Week 3: P3 Complete Training
1. **Day 1-3:** test_training_lifecycle.py (학습 전체 프로세스)
2. **Day 4-5:** test_training_metrics.py (메트릭 검증)

### Week 4: P3 Integration
1. **Day 1-3:** test_e2e_workflows.py (E2E 테스트)
2. **Day 4-5:** CI/CD 통합 및 문서화

---

## 예상 최종 커버리지

- **P0 (완료):** 26 tests
- **P2 (Week 1-2):** 75 tests
- **P3 (Week 3-4):** 40 tests
- **총 예상:** ~140 tests

**예상 커버리지:** 60-70% (핵심 기능 위주)

---

## 테스트 실행 전략

### 빠른 테스트 (CI/CD)
```bash
# P0 tests only (< 2초)
pytest tests/unit/ tests/integration/test_models_api.py -v

# P2 tests without training (< 30초)
pytest tests/ -m "not slow" -v
```

### 전체 테스트 (로컬)
```bash
# 모든 테스트 (5-10분)
pytest tests/ -v

# 학습 포함 전체 테스트 (30분+)
pytest tests/ -m "all" -v
```

---

## 다음 단계

1. ✅ P0 테스트 완료 (현재)
2. ⬜ P2 테스트 구현 시작
3. ⬜ CI/CD 파이프라인 구축
4. ⬜ P3 테스트 구현
5. ⬜ 커버리지 리포트 생성

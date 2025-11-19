"""
Test pretrained model weight loading for all active models.

This script verifies that all 10 active models in the registry can successfully
load their pretrained weights.

Priority: P1 (Critical for production readiness)

Note: D-FINE model is excluded as it's not yet available on HuggingFace Hub.

Usage:
    # Run in training environment (with framework dependencies installed)
    cd mvp/training
    python -m pytest ../backend/tests/integration/test_pretrained_loading.py -v -s
"""

import pytest
import sys
from pathlib import Path

# Add training directory to path
training_dir = Path(__file__).parent.parent.parent.parent / "training"
sys.path.insert(0, str(training_dir))


def _is_ultralytics_available():
    """Check if ultralytics is available."""
    try:
        import ultralytics
        return True
    except ImportError:
        return False


def _is_timm_available():
    """Check if timm is available."""
    try:
        import timm
        return True
    except ImportError:
        return False


def _is_transformers_available():
    """Check if transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


# ============================================================
# Ultralytics Models (5 models)
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_ultralytics_available(),
    reason="ultralytics not installed"
)
class TestUltralyticsPretrainedLoading:
    """Test Ultralytics pretrained model loading."""

    def test_yolo11n_pretrained_loading(self):
        """Test YOLO11n detection model pretrained weight loading."""
        from ultralytics import YOLO

        print("\n[TEST] Loading yolo11n.pt...")
        model = YOLO("yolo11n.pt")

        assert model is not None, "Model is None"
        assert hasattr(model, "model"), "Model has no 'model' attribute"
        assert model.model is not None, "model.model is None"
        assert model.task == "detect", f"Expected task='detect', got '{model.task}'"

        # Verify model has weights loaded
        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count > 0, "Model has no parameters"
        print(f"[OK] yolo11n loaded successfully ({param_count:,} parameters)")

    def test_yolo11n_seg_pretrained_loading(self):
        """Test YOLO11n-seg instance segmentation model pretrained weight loading."""
        from ultralytics import YOLO

        print("\n[TEST] Loading yolo11n-seg.pt...")
        model = YOLO("yolo11n-seg.pt")

        assert model is not None
        assert hasattr(model, "model")
        assert model.model is not None
        assert model.task == "segment", f"Expected task='segment', got '{model.task}'"

        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count > 0
        print(f"[OK] yolo11n-seg loaded successfully ({param_count:,} parameters)")

    def test_yolo11n_pose_pretrained_loading(self):
        """Test YOLO11n-pose keypoint detection model pretrained weight loading."""
        from ultralytics import YOLO

        print("\n[TEST] Loading yolo11n-pose.pt...")
        model = YOLO("yolo11n-pose.pt")

        assert model is not None
        assert hasattr(model, "model")
        assert model.model is not None
        assert model.task == "pose", f"Expected task='pose', got '{model.task}'"

        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count > 0
        print(f"[OK] yolo11n-pose loaded successfully ({param_count:,} parameters)")

    def test_yolo_world_v2_s_pretrained_loading(self):
        """Test YOLO-World v2 small zero-shot detection model pretrained weight loading."""
        from ultralytics import YOLO

        print("\n[TEST] Loading yolo_world_v2_s.pt...")
        # YOLO-World uses different weight name format
        model = YOLO("yolov8s-worldv2.pt")

        assert model is not None
        assert hasattr(model, "model")
        assert model.model is not None

        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count > 0
        print(f"[OK] yolo_world_v2_s loaded successfully ({param_count:,} parameters)")

    def test_sam2_t_pretrained_loading(self):
        """Test SAM2 Tiny zero-shot segmentation model pretrained weight loading."""
        from ultralytics import SAM

        print("\n[TEST] Loading sam2_t.pt...")
        # SAM2 uses SAM class, not YOLO
        model = SAM("sam2_t.pt")

        assert model is not None
        assert hasattr(model, "model")
        assert model.model is not None

        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count > 0
        print(f"[OK] sam2_t loaded successfully ({param_count:,} parameters)")


# ============================================================
# TIMM Models (3 models)
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_timm_available(),
    reason="timm not installed"
)
class TestTIMMPretrainedLoading:
    """Test TIMM pretrained model loading."""

    def test_tf_efficientnetv2_s_pretrained_loading(self):
        """Test EfficientNetV2-S pretrained weight loading."""
        import timm
        import torch

        print("\n[TEST] Loading tf_efficientnetv2_s.in1k...")
        model = timm.create_model("tf_efficientnetv2_s.in1k", pretrained=True)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"[OK] tf_efficientnetv2_s.in1k loaded successfully ({param_count:,} parameters)")

        # Test forward pass with dummy input
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)  # EfficientNetV2-S default size
            output = model(dummy_input)
            assert output is not None
            assert output.shape[0] == 1  # Batch size
            assert output.shape[1] == 1000  # ImageNet classes
            print(f"[OK] Forward pass successful, output shape: {output.shape}")

    def test_convnext_tiny_pretrained_loading(self):
        """Test ConvNeXt Tiny pretrained weight loading."""
        import timm
        import torch

        print("\n[TEST] Loading convnext_tiny.in12k_ft_in1k...")
        model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"[OK] convnext_tiny loaded successfully ({param_count:,} parameters)")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            assert output is not None
            assert output.shape[0] == 1
            assert output.shape[1] == 1000
            print(f"[OK] Forward pass successful, output shape: {output.shape}")

    def test_vit_base_patch16_224_pretrained_loading(self):
        """Test Vision Transformer Base pretrained weight loading."""
        import timm
        import torch

        print("\n[TEST] Loading vit_base_patch16_224.augreg_in21k_ft_in1k...")
        model = timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"[OK] vit_base_patch16_224 loaded successfully ({param_count:,} parameters)")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            assert output is not None
            assert output.shape[0] == 1
            assert output.shape[1] == 1000
            print(f"[OK] Forward pass successful, output shape: {output.shape}")


# ============================================================
# HuggingFace Models (2 models)
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_transformers_available(),
    reason="transformers not installed"
)
class TestHuggingFacePretrainedLoading:
    """Test HuggingFace pretrained model loading."""

    # NOTE: D-FINE model test is excluded as the model is not yet available on HuggingFace Hub
    # def test_dfine_x_coco_pretrained_loading(self):
    #     """Test D-FINE detection model pretrained weight loading."""
    #     from transformers import AutoModel

    #     print("\n[TEST] Loading ustc-community/dfine-x-coco...")
    #     model = AutoModel.from_pretrained("ustc-community/dfine-x-coco", trust_remote_code=True)

    #     assert model is not None

    #     param_count = sum(p.numel() for p in model.parameters())
    #     assert param_count > 0
    #     print(f"[OK] dfine-x-coco loaded successfully ({param_count:,} parameters)")

    def test_segformer_b0_pretrained_loading(self):
        """Test SegFormer-B0 semantic segmentation model pretrained weight loading."""
        from transformers import SegformerForSemanticSegmentation

        print("\n[TEST] Loading nvidia/segformer-b0-finetuned-ade-512-512...")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        assert model is not None

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"[OK] segformer-b0 loaded successfully ({param_count:,} parameters)")

    def test_swin2sr_x2_pretrained_loading(self):
        """Test Swin2SR super-resolution model pretrained weight loading."""
        from transformers import Swin2SRForImageSuperResolution

        print("\n[TEST] Loading caidas/swin2SR-classical-sr-x2-64...")
        model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64"
        )

        assert model is not None

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"[OK] swin2SR-x2 loaded successfully ({param_count:,} parameters)")


# ============================================================
# Summary Test
# ============================================================

@pytest.mark.slow
def test_all_models_summary():
    """Summary of all model loading tests."""
    print("\n" + "=" * 80)
    print("MODEL LOADING TEST SUMMARY")
    print("=" * 80)
    print("\nTarget: 10 models across 3 frameworks")
    print("\nFrameworks:")
    print("  - Ultralytics: 5 models (yolo11n, yolo11n-seg, yolo11n-pose, yolo_world_v2_s, sam2_t)")
    print("  - TIMM: 3 models (tf_efficientnetv2_s, convnext_tiny, vit_base_patch16_224)")
    print("  - HuggingFace: 2 models (segformer-b0, swin2SR-x2)")
    print("\nNote: D-FINE model is excluded as it's not yet available on HuggingFace Hub")
    print("Note: Run with individual framework environments to test specific models")
    print("=" * 80)

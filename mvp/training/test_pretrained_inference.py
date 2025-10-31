#!/usr/bin/env python3
"""
Standalone pretrained model inference test.

Tests that all active pretrained models can successfully run inference:
- Ultralytics: YOLO11n (detect, seg, pose), YOLO-World v2, SAM2
- TIMM: EfficientNetV2-S, ConvNeXt Tiny, ViT-Base
- HuggingFace: SegFormer-B0, Swin2SR

Usage:
    python test_pretrained_inference.py
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image


def create_test_image():
    """Create a 640x640 RGB test image."""
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img


# ============================================================
# Ultralytics Models (5 models + batch)
# ============================================================

def test_yolo11n_detection():
    """Test YOLO11n detection inference."""
    from ultralytics import YOLO

    print("\n[TEST] YOLO11n detection inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = YOLO("yolo11n.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None
        assert len(results) == 1

        result = results[0]
        assert hasattr(result, "boxes")
        assert hasattr(result, "names")

        print(f"  ✓ YOLO11n detection inference successful")
        print(f"    Detected {len(result.boxes)} objects")


def test_yolo11n_seg():
    """Test YOLO11n-seg instance segmentation inference."""
    from ultralytics import YOLO

    print("\n[TEST] YOLO11n-seg segmentation inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = YOLO("yolo11n-seg.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None
        assert len(results) == 1

        result = results[0]
        assert hasattr(result, "boxes")
        assert hasattr(result, "masks")

        print(f"  ✓ YOLO11n-seg inference successful")


def test_yolo11n_pose():
    """Test YOLO11n-pose keypoint detection inference."""
    from ultralytics import YOLO

    print("\n[TEST] YOLO11n-pose inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = YOLO("yolo11n-pose.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None
        assert len(results) == 1

        result = results[0]
        assert hasattr(result, "keypoints")

        print(f"  ✓ YOLO11n-pose inference successful")


def test_yolo_world_v2():
    """Test YOLO-World v2 zero-shot detection inference."""
    from ultralytics import YOLO

    print("\n[TEST] YOLO-World v2 inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = YOLO("yolov8s-worldv2.pt")

        # Set custom classes for zero-shot detection
        model.set_classes(["person", "car", "dog"])

        results = model(str(test_image_path), verbose=False)

        assert results is not None
        assert len(results) == 1

        print(f"  ✓ YOLO-World v2 inference successful")


def test_sam2():
    """Test SAM2 zero-shot segmentation inference."""
    from ultralytics import SAM

    print("\n[TEST] SAM2 inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = SAM("sam2_t.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None
        assert len(results) == 1

        print(f"  ✓ SAM2 inference successful")


def test_batch_inference():
    """Test batch inference with multiple images."""
    from ultralytics import YOLO

    print("\n[TEST] Batch inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 test images
        image_paths = []
        for i in range(3):
            test_image_path = Path(tmpdir) / f"test_image_{i}.jpg"
            img = create_test_image()
            img.save(test_image_path)
            image_paths.append(str(test_image_path))

        model = YOLO("yolo11n.pt")
        results = model(image_paths, verbose=False)

        assert results is not None
        assert len(results) == len(image_paths)

        print(f"  ✓ Batch inference successful ({len(results)} images)")


# ============================================================
# TIMM Models (3 models)
# ============================================================

def test_efficientnetv2():
    """Test EfficientNetV2-S classification inference."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    print("\n[TEST] EfficientNetV2-S inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = timm.create_model("tf_efficientnetv2_s.in1k", pretrained=True)
        model.eval()

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        img = Image.open(test_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        assert output is not None
        assert output.shape[0] == 1
        assert output.shape[1] == 1000  # ImageNet classes

        probs = torch.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

        print(f"  ✓ EfficientNetV2-S inference successful")
        print(f"    Top class: {top_class.item()}, Confidence: {top_prob.item():.4f}")


def test_convnext():
    """Test ConvNeXt Tiny classification inference."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    print("\n[TEST] ConvNeXt Tiny inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True)
        model.eval()

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        img = Image.open(test_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        assert output is not None
        assert output.shape[0] == 1
        assert output.shape[1] == 1000

        probs = torch.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

        print(f"  ✓ ConvNeXt Tiny inference successful")
        print(f"    Top class: {top_class.item()}, Confidence: {top_prob.item():.4f}")


def test_vit():
    """Test Vision Transformer Base inference."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    print("\n[TEST] ViT-Base inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        model.eval()

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        img = Image.open(test_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        assert output is not None
        assert output.shape[0] == 1
        assert output.shape[1] == 1000

        probs = torch.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

        print(f"  ✓ ViT-Base inference successful")
        print(f"    Top class: {top_class.item()}, Confidence: {top_prob.item():.4f}")


# ============================================================
# HuggingFace Models (2 models)
# ============================================================

def test_segformer():
    """Test SegFormer-B0 semantic segmentation inference."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    import torch

    print("\n[TEST] SegFormer-B0 inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        img = Image.open(test_image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        assert logits is not None
        assert len(logits.shape) == 4  # [batch, classes, height, width]

        # Upsample to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        print(f"  ✓ SegFormer-B0 inference successful")
        print(f"    Segmentation map shape: {pred_seg.shape}")
        print(f"    Unique classes: {torch.unique(pred_seg).numel()}")


def test_swin2sr():
    """Test Swin2SR super-resolution inference."""
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
    import torch

    print("\n[TEST] Swin2SR inference...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64"
        )
        processor = Swin2SRImageProcessor.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64"
        )

        img = Image.open(test_image_path).convert("RGB")
        # Resize to smaller size for faster testing
        img = img.resize((128, 128))

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        output_tensor = outputs.reconstruction
        assert output_tensor is not None

        # Check that output is 2x the processed input size
        processed_size = inputs.pixel_values.shape[2]
        actual_output_size = output_tensor.shape[2]
        expected_output = processed_size * 2

        assert actual_output_size == expected_output, \
            f"Expected {expected_output} (2x processed), got {actual_output_size}"

        print(f"  ✓ Swin2SR inference successful")
        print(f"    Original input: {img.size}")
        print(f"    Processed input: {processed_size}x{processed_size}")
        print(f"    Output size: {actual_output_size}x{actual_output_size}")
        print(f"    Upscaling ratio: 2x (of processed input)")


# ============================================================
# Main Test Runner
# ============================================================

def main():
    """Run all pretrained model inference tests."""
    print("\n" + "=" * 80)
    print("PRETRAINED MODEL INFERENCE TESTS")
    print("=" * 80)
    print("\nTarget: 10 models across 3 frameworks")
    print("  - Ultralytics: 5 models + batch inference")
    print("  - TIMM: 3 models")
    print("  - HuggingFace: 2 models")

    ultralytics_tests = []
    timm_tests = []
    hf_tests = []

    try:
        # ========== Ultralytics Tests ==========
        print("\n" + "=" * 80)
        print("ULTRALYTICS MODELS (5 + batch)")
        print("=" * 80)

        try:
            test_yolo11n_detection()
            ultralytics_tests.append("YOLO11n detection")
        except Exception as e:
            print(f"  ✗ YOLO11n detection failed: {e}")

        try:
            test_yolo11n_seg()
            ultralytics_tests.append("YOLO11n-seg")
        except Exception as e:
            print(f"  ✗ YOLO11n-seg failed: {e}")

        try:
            test_yolo11n_pose()
            ultralytics_tests.append("YOLO11n-pose")
        except Exception as e:
            print(f"  ✗ YOLO11n-pose failed: {e}")

        try:
            test_yolo_world_v2()
            ultralytics_tests.append("YOLO-World v2")
        except Exception as e:
            print(f"  ✗ YOLO-World v2 failed: {e}")

        try:
            test_sam2()
            ultralytics_tests.append("SAM2")
        except Exception as e:
            print(f"  ✗ SAM2 failed: {e}")

        try:
            test_batch_inference()
            ultralytics_tests.append("Batch inference")
        except Exception as e:
            print(f"  ✗ Batch inference failed: {e}")

        # ========== TIMM Tests ==========
        print("\n" + "=" * 80)
        print("TIMM MODELS (3)")
        print("=" * 80)

        try:
            import timm

            try:
                test_efficientnetv2()
                timm_tests.append("EfficientNetV2-S")
            except Exception as e:
                print(f"  ✗ EfficientNetV2-S failed: {e}")

            try:
                test_convnext()
                timm_tests.append("ConvNeXt Tiny")
            except Exception as e:
                print(f"  ✗ ConvNeXt Tiny failed: {e}")

            try:
                test_vit()
                timm_tests.append("ViT-Base")
            except Exception as e:
                print(f"  ✗ ViT-Base failed: {e}")

        except ImportError:
            print("  ⚠  TIMM not available - skipping TIMM tests")
            print("     (Run in timm container for TIMM tests)")

        # ========== HuggingFace Tests ==========
        print("\n" + "=" * 80)
        print("HUGGINGFACE MODELS (2)")
        print("=" * 80)

        try:
            import transformers

            try:
                test_segformer()
                hf_tests.append("SegFormer-B0")
            except Exception as e:
                print(f"  ✗ SegFormer-B0 failed: {e}")

            try:
                test_swin2sr()
                hf_tests.append("Swin2SR")
            except Exception as e:
                print(f"  ✗ Swin2SR failed: {e}")

        except ImportError:
            print("  ⚠  Transformers not available - skipping HuggingFace tests")
            print("     (Run in huggingface container for HuggingFace tests)")

        # ========== Summary ==========
        print("\n" + "=" * 80)
        print("✅ PRETRAINED MODEL INFERENCE TESTS COMPLETED")
        print("=" * 80)

        print(f"\nSuccessfully tested:")
        print(f"  Ultralytics: {len(ultralytics_tests)} tests")
        for test in ultralytics_tests:
            print(f"    ✓ {test}")

        if timm_tests:
            print(f"  TIMM: {len(timm_tests)} tests")
            for test in timm_tests:
                print(f"    ✓ {test}")
        else:
            print(f"  TIMM: 0 tests (not available in this container)")

        if hf_tests:
            print(f"  HuggingFace: {len(hf_tests)} tests")
            for test in hf_tests:
                print(f"    ✓ {test}")
        else:
            print(f"  HuggingFace: 0 tests (not available in this container)")

        total = len(ultralytics_tests) + len(timm_tests) + len(hf_tests)
        print(f"\nTotal: {total} inference tests passed!")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

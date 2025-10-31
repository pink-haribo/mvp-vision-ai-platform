#!/usr/bin/env python3
"""
Standalone inference output format test.

Tests that inference outputs have the correct format for each task type:
- Object detection (YOLO11n)
- Instance segmentation (YOLO11n-seg)
- Classification (TIMM)
- Pose estimation (YOLO11n-pose)

Usage:
    python test_inference_output_format.py
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


def test_detection_output_format():
    """Test YOLO11n detection output format."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TEST 1: YOLO11N DETECTION OUTPUT FORMAT")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        # Run detection
        print(f"\n[TEST] Running YOLO11n detection...")
        model = YOLO("yolo11n.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None, "Results are None"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]

        # Check required attributes
        assert hasattr(result, "boxes"), "Result missing 'boxes' attribute"
        assert hasattr(result, "names"), "Result missing 'names' attribute"

        # Check boxes format
        boxes = result.boxes
        assert hasattr(boxes, "xyxy"), "Boxes missing 'xyxy' attribute"
        assert hasattr(boxes, "conf"), "Boxes missing 'conf' (confidence) attribute"
        assert hasattr(boxes, "cls"), "Boxes missing 'cls' (class) attribute"

        # Verify data types if objects detected
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            assert xyxy.shape[1] == 4, f"Expected 4 coordinates per box, got {xyxy.shape[1]}"
            assert len(conf) == len(xyxy), "Confidence array length mismatch"
            assert len(cls) == len(xyxy), "Class array length mismatch"

            # Check coordinate ranges
            assert np.all(xyxy >= 0), "Negative coordinates found"

            # Check confidence ranges (0-1)
            assert np.all((conf >= 0) & (conf <= 1)), "Confidence out of range [0, 1]"

            print(f"  ✓ Detected {len(boxes)} objects with valid format")
        else:
            print(f"  ✓ No objects detected (valid empty result)")

        # Check names dictionary
        assert isinstance(result.names, dict), "Names should be a dictionary"
        assert len(result.names) > 0, "Names dictionary is empty"

        print(f"\n✅ YOLO11n detection output format test passed!")
        print(f"   Boxes: {len(boxes)} objects")
        print(f"   Classes: {len(result.names)} total classes")


def test_segmentation_output_format():
    """Test YOLO11n-seg segmentation output format."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TEST 2: YOLO11N-SEG SEGMENTATION OUTPUT FORMAT")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        # Run segmentation
        print(f"\n[TEST] Running YOLO11n-seg segmentation...")
        model = YOLO("yolo11n-seg.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None, "Results are None"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]

        # Check required attributes
        assert hasattr(result, "boxes"), "Result missing 'boxes' attribute"
        assert hasattr(result, "masks"), "Result missing 'masks' attribute"

        # Check boxes (same as detection)
        boxes = result.boxes
        assert hasattr(boxes, "xyxy"), "Boxes missing 'xyxy'"
        assert hasattr(boxes, "conf"), "Boxes missing 'conf'"
        assert hasattr(boxes, "cls"), "Boxes missing 'cls'"

        # Check masks
        masks = result.masks
        if masks is not None and len(masks) > 0:
            assert hasattr(masks, "data"), "Masks missing 'data' attribute"
            mask_data = masks.data.cpu().numpy()

            # Masks should be 2D or 3D (batch, height, width)
            assert len(mask_data.shape) >= 2, f"Invalid mask shape: {mask_data.shape}"

            # Mask values should be binary or float [0, 1]
            assert np.all((mask_data >= 0) & (mask_data <= 1)), "Mask values out of range"

            # Number of masks should match number of boxes
            assert len(mask_data) == len(boxes), "Mask count != box count"

            print(f"  ✓ {len(masks)} masks with valid format")
            print(f"  ✓ Mask shape: {mask_data.shape}")
        else:
            print(f"  ✓ No masks (valid empty result)")

        print(f"\n✅ YOLO11n-seg segmentation output format test passed!")


def test_pose_output_format():
    """Test YOLO11n-pose pose estimation output format."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TEST 3: YOLO11N-POSE OUTPUT FORMAT")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        # Run pose estimation
        print(f"\n[TEST] Running YOLO11n-pose...")
        model = YOLO("yolo11n-pose.pt")
        results = model(str(test_image_path), verbose=False)

        assert results is not None, "Results are None"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]

        # Check required attributes
        assert hasattr(result, "keypoints"), "Result missing 'keypoints' attribute"
        assert hasattr(result, "boxes"), "Result missing 'boxes' attribute"

        # Check keypoints
        keypoints = result.keypoints
        if keypoints is not None and len(keypoints) > 0:
            assert hasattr(keypoints, "data"), "Keypoints missing 'data' attribute"
            assert hasattr(keypoints, "conf"), "Keypoints missing 'conf' attribute"

            kp_data = keypoints.data.cpu().numpy()

            # Keypoints should be 3D: (num_persons, num_keypoints, 2 or 3)
            # Format: [x, y] or [x, y, confidence]
            assert len(kp_data.shape) == 3, f"Expected 3D keypoints, got {kp_data.shape}"

            num_persons = kp_data.shape[0]
            num_keypoints = kp_data.shape[1]
            coords_per_kp = kp_data.shape[2]

            assert coords_per_kp >= 2, f"Expected at least 2 coords per keypoint, got {coords_per_kp}"
            assert num_keypoints == 17, f"Expected 17 COCO keypoints, got {num_keypoints}"

            # Check coordinate ranges
            xy_coords = kp_data[:, :, :2]
            assert np.all(xy_coords >= 0), "Negative keypoint coordinates found"

            # If confidence is included in data
            if coords_per_kp == 3:
                kp_conf_in_data = kp_data[:, :, 2]
                assert np.all((kp_conf_in_data >= 0) & (kp_conf_in_data <= 1)), \
                    "Keypoint confidence out of range"

            # Check boxes match number of persons
            boxes = result.boxes
            assert len(boxes) == num_persons, \
                f"Box count ({len(boxes)}) != person count ({num_persons})"

            print(f"  ✓ {num_persons} person(s) detected with {num_keypoints} keypoints each")
        else:
            print(f"  ✓ No persons detected (valid empty result)")

        print(f"\n✅ YOLO11n-pose output format test passed!")


def test_classification_output_format():
    """Test TIMM classification output format."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    print("\n" + "=" * 80)
    print("TEST 4: TIMM CLASSIFICATION OUTPUT FORMAT")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        img = create_test_image()
        img.save(test_image_path)

        # Load model
        print(f"\n[TEST] Running TIMM classification (EfficientNetV2)...")
        model = timm.create_model("tf_efficientnetv2_s.in1k", pretrained=True)
        model.eval()

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        img = Image.open(test_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        # Check output format
        assert output is not None, "Output is None"
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        assert len(output.shape) == 2, f"Expected 2D output, got {len(output.shape)}D"
        assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
        assert output.shape[1] == 1000, f"Expected 1000 classes, got {output.shape[1]}"

        # Check logits format
        logits = output[0]
        assert torch.all(torch.isfinite(logits)), "Output contains inf or nan"

        # Get probabilities
        probs = torch.softmax(output, dim=1)[0]
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities out of range"
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), \
            f"Probabilities don't sum to 1: {probs.sum()}"

        # Get top-k predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)

        assert len(top_probs) == top_k, f"Expected {top_k} predictions"
        assert len(top_indices) == top_k, f"Expected {top_k} indices"

        # Top probabilities should be sorted in descending order
        assert torch.all(top_probs[:-1] >= top_probs[1:]), "Top probs not sorted"

        print(f"\n✅ TIMM classification output format test passed!")
        print(f"   Output shape: {output.shape}")
        print(f"   Top-1 class: {top_indices[0].item()}")
        print(f"   Top-1 confidence: {top_probs[0].item():.4f}")
        print(f"   Top-5 classes: {top_indices.tolist()}")


def main():
    """Run all output format tests."""
    print("\n" + "=" * 80)
    print("INFERENCE OUTPUT FORMAT TESTS")
    print("=" * 80)
    print("\nTesting inference output formats for all task types.")
    print("This verifies:")
    print("  - Detection outputs have boxes, confidence, classes")
    print("  - Segmentation outputs have boxes and masks")
    print("  - Classification outputs have logits and probabilities")
    print("  - Pose outputs have keypoints and boxes")

    try:
        # Test 1: Detection (Ultralytics)
        test_detection_output_format()

        # Test 2: Segmentation (Ultralytics)
        test_segmentation_output_format()

        # Test 3: Pose (Ultralytics)
        test_pose_output_format()

        # Test 4: Classification (TIMM)
        try:
            test_classification_output_format()
        except ImportError:
            print("\n⚠️  TIMM not available - skipping classification test")
            print("   (Run in timm container for classification tests)")

        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL INFERENCE OUTPUT FORMAT TESTS PASSED")
        print("=" * 80)
        print("\nSuccessfully tested:")
        print("  ✓ YOLO11n detection output format")
        print("  ✓ YOLO11n-seg segmentation output format")
        print("  ✓ YOLO11n-pose pose estimation output format")
        try:
            import timm
            print("  ✓ TIMM classification output format")
        except ImportError:
            print("  ⚠  TIMM classification (not available in this container)")

        print("\nAll output formats verified!")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

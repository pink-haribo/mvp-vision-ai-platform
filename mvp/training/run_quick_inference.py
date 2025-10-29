"""
Quick inference script for running inference in training venv.

This script is called by the backend API via subprocess to run inference
using the training environment (which has torch installed).
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_inference(
    training_job_id: int,
    checkpoint_path: str,
    image_path: str,
    framework: str,
    model_name: str,
    task_type: str,
    num_classes: int,
    dataset_path: str,
    output_dir: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    top_k: int = 5
):
    """Run inference on a single image."""
    from training.adapters.base import TaskType

    # Create adapter based on framework
    if framework == "timm":
        from training.adapters.timm_adapter import TimmAdapter
        adapter = TimmAdapter(
            model_name=model_name,
            task_type=TaskType(task_type),
            num_classes=num_classes,
            dataset_path=dataset_path,
            output_dir=output_dir,
            job_id=training_job_id
        )
    elif framework == "ultralytics":
        from training.adapters.ultralytics_adapter import UltralyticsAdapter
        adapter = UltralyticsAdapter(
            model_name=model_name,
            task_type=TaskType(task_type),
            num_classes=num_classes,
            dataset_path=dataset_path,
            output_dir=output_dir,
            job_id=training_job_id
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Load checkpoint
    adapter.load_checkpoint(
        checkpoint_path=checkpoint_path,
        inference_mode=True
    )

    # Run inference
    results = adapter.infer_batch([image_path])

    if not results or len(results) == 0:
        raise RuntimeError("Inference failed - no results returned")

    result = results[0]

    # Convert result to dict
    result_dict = {
        "image_path": result.image_path,
        "image_name": result.image_name,
        "task_type": result.task_type.value if hasattr(result.task_type, 'value') else str(result.task_type),
        "inference_time_ms": result.inference_time_ms,
        "preprocessing_time_ms": result.preprocessing_time_ms,
        "postprocessing_time_ms": result.postprocessing_time_ms,
    }

    # Add task-specific results
    if result.task_type == TaskType.IMAGE_CLASSIFICATION:
        result_dict["predicted_label"] = result.predicted_label
        result_dict["predicted_label_id"] = result.predicted_label_id
        result_dict["confidence"] = result.confidence
        result_dict["top5_predictions"] = result.top5_predictions or []

    elif result.task_type == TaskType.OBJECT_DETECTION:
        result_dict["predicted_boxes"] = result.predicted_boxes or []
        result_dict["num_detections"] = len(result.predicted_boxes) if result.predicted_boxes else 0

    elif result.task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
        result_dict["predicted_boxes"] = result.predicted_boxes or []
        result_dict["predicted_mask_path"] = result.predicted_mask_path
        result_dict["num_instances"] = len(result.predicted_boxes) if result.predicted_boxes else 0

    elif result.task_type == TaskType.POSE_ESTIMATION:
        result_dict["predicted_keypoints"] = result.predicted_keypoints or []
        result_dict["num_persons"] = len(result.predicted_keypoints) if result.predicted_keypoints else 0

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quick inference")
    parser.add_argument("--training_job_id", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.45)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    try:
        result = run_inference(
            training_job_id=args.training_job_id,
            checkpoint_path=args.checkpoint_path,
            image_path=args.image_path,
            framework=args.framework,
            model_name=args.model_name,
            task_type=args.task_type,
            num_classes=args.num_classes,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            max_detections=args.max_detections,
            top_k=args.top_k
        )

        # Output result as JSON
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        import traceback
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_info), file=sys.stderr)
        sys.exit(1)

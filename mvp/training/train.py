"""Main training script with Adapter pattern support."""

import argparse
import json
import sys
import os

# Add training directory to path
sys.path.insert(0, os.path.dirname(__file__))

from adapters.base import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    TaskType,
    DatasetFormat
)
from adapters.timm_adapter import TimmAdapter
from adapters.ultralytics_adapter import UltralyticsAdapter

# Configure MLflow tracking URI
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

# Adapter registry
ADAPTER_REGISTRY = {
    'timm': TimmAdapter,
    'ultralytics': UltralyticsAdapter,
    # 'transformers': TransformersAdapter,  # TODO: Implement
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models with adapter pattern')

    # New arguments for adapter pattern
    parser.add_argument('--framework', type=str, required=True,
                        choices=['timm', 'ultralytics', 'transformers'],
                        help='Training framework to use')
    parser.add_argument('--task_type', type=str, required=True,
                        help='Task type (e.g., image_classification, object_detection)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., resnet50, yolov8n)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset_format', type=str, default='imagefolder',
                        choices=['imagefolder', 'coco', 'yolo', 'voc', 'custom'],
                        help='Dataset format')
    parser.add_argument('--num_classes', type=int, required=False,
                        help='Number of classes (required for classification)')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')

    # Other arguments
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--job_id', type=int, required=True,
                        help='Training job ID')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("\n" + "="*80)
    print("Vision AI Training Platform - Adapter Pattern")
    print("="*80)
    print(f"[CONFIG] {json.dumps(vars(args), indent=2)}")

    # Create configuration objects
    try:
        model_config = ModelConfig(
            framework=args.framework,
            task_type=TaskType(args.task_type),
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            image_size=args.image_size,
        )

        dataset_config = DatasetConfig(
            dataset_path=args.dataset_path,
            format=DatasetFormat(args.dataset_format),
        )

        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            device=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create configuration: {str(e)}")
        sys.exit(1)

    # Select adapter
    adapter_class = ADAPTER_REGISTRY.get(args.framework)
    if not adapter_class:
        print(f"[ERROR] Framework '{args.framework}' not supported")
        print(f"[ERROR] Available frameworks: {list(ADAPTER_REGISTRY.keys())}")
        sys.exit(1)

    # Create adapter instance
    print(f"\n[INFO] Creating {args.framework} adapter for {args.task_type}")
    adapter = adapter_class(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=args.output_dir,
        job_id=args.job_id
    )

    # Train
    try:
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")

        metrics = adapter.train()

        print("\n" + "="*80)
        print("Training Completed Successfully!")
        print("="*80)
        print(f"[FINAL_RESULTS] {len(metrics)} epochs completed")
        if metrics:
            final_metrics = metrics[-1]
            print(f"[FINAL_METRICS] Train Loss: {final_metrics.train_loss:.4f}")
            if final_metrics.val_loss:
                print(f"[FINAL_METRICS] Val Loss: {final_metrics.val_loss:.4f}")
            for key, value in final_metrics.metrics.items():
                print(f"[FINAL_METRICS] {key}: {value:.4f}")

        print("\n[STATUS] SUCCESS")
        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("Training Failed!")
        print("="*80)
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n[STATUS] FAILED")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main() or 0)

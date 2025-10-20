"""Main training script."""

import argparse
import json
import sys
import os

# Add training directory to path
sys.path.insert(0, os.path.dirname(__file__))

import torch

from models.resnet import create_resnet50, get_model_info
from data.dataset import create_dataloaders, get_dataset_info
from training.trainer import Trainer

# Configure MLflow tracking URI
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ResNet50 for image classification')

    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--job_id', type=int, required=True,
                        help='Training job ID')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print(f"[ARGS] {json.dumps(vars(args))}")

    # Check dataset
    print(f"[INFO] Checking dataset at {args.dataset_path}")
    dataset_info = get_dataset_info(args.dataset_path)
    print(f"[DATASET_INFO] {json.dumps(dataset_info)}")

    if not dataset_info['train_exists'] or not dataset_info['val_exists']:
        print("[ERROR] Dataset structure is invalid. Need train/ and val/ directories.")
        sys.exit(1)

    # Create dataloaders
    print(f"[INFO] Creating dataloaders with batch_size={args.batch_size}")
    try:
        train_loader, val_loader, num_classes = create_dataloaders(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create dataloaders: {str(e)}")
        sys.exit(1)

    # Verify num_classes matches
    if num_classes != args.num_classes:
        print(f"[WARNING] Dataset has {num_classes} classes but {args.num_classes} was specified")
        args.num_classes = num_classes

    # Create model
    print(f"[INFO] Creating ResNet50 model with {args.num_classes} classes")
    try:
        model = create_resnet50(
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
        model_info = get_model_info(model)
        print(f"[MODEL_INFO] {json.dumps(model_info)}")
    except Exception as e:
        print(f"[ERROR] Failed to create model: {str(e)}")
        sys.exit(1)

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=device,
    )

    # Train
    try:
        # Get dataset name from path
        dataset_name = os.path.basename(os.path.normpath(args.dataset_path))

        results = trainer.train(
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            mlflow_experiment_name="vision-ai-training",
            mlflow_run_name=f"job-{args.job_id}",
            model_name="resnet50",
            dataset_name=dataset_name,
        )
        print(f"[FINAL_RESULTS] {json.dumps(results)}")
        print("[STATUS] SUCCESS")
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        print("[STATUS] FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

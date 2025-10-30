"""Main training script with Adapter pattern support."""

import argparse
import json
import sys
import os
from pathlib import Path

# Disable output buffering to see logs immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add training directory to path
sys.path.insert(0, os.path.dirname(__file__))

from platform_sdk import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    TaskType,
    DatasetFormat
)
from adapters import ADAPTER_REGISTRY, TimmAdapter, UltralyticsAdapter

# Configure MLflow tracking URI
# Use MLflow server instead of local file storage
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")


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

    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from checkpoint (restore optimizer/scheduler state)')

    return parser.parse_args()


def load_advanced_config_from_db(job_id: int):
    """
    Load advanced_config from database for given job_id.

    Args:
        job_id: Training job ID

    Returns:
        Advanced config dict or None
    """
    try:
        from sqlalchemy import create_engine, Column, Integer, JSON
        from sqlalchemy.orm import declarative_base, sessionmaker

        # Get database path relative to project root
        # train.py is at: mvp/training/train.py
        # DB is at: mvp/data/db/vision_platform.db
        training_dir = os.path.dirname(os.path.abspath(__file__))  # mvp/training
        mvp_dir = os.path.dirname(training_dir)  # mvp
        db_path = os.path.join(mvp_dir, 'data', 'db', 'vision_platform.db')

        if not os.path.exists(db_path):
            print(f"[WARNING] Database not found at: {db_path}")
            return None

        database_url = f"sqlite:///{db_path}"
        print(f"[INFO] Connecting to database: {db_path}")

        # Create engine and session
        engine = create_engine(database_url, echo=False)
        SessionLocal = sessionmaker(bind=engine)

        # Define minimal TrainingJob model
        Base = declarative_base()

        class TrainingJob(Base):
            __tablename__ = "training_jobs"
            id = Column(Integer, primary_key=True)
            advanced_config = Column(JSON, nullable=True)

        # Query database
        db = SessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job and job.advanced_config:
                print(f"[INFO] SUCCESS: Loaded advanced_config from database for job {job_id}")
                print(f"[INFO]   Optimizer: {job.advanced_config.get('optimizer', {}).get('type', 'N/A')}")
                print(f"[INFO]   Scheduler: {job.advanced_config.get('scheduler', {}).get('type', 'N/A')}")
                print(f"[INFO]   Augmentation: {'enabled' if job.advanced_config.get('augmentation', {}).get('enabled') else 'disabled'}")
                return job.advanced_config
            else:
                print(f"[INFO] No advanced_config found for job {job_id}, using defaults")
                return None
        finally:
            db.close()
    except Exception as e:
        print(f"[WARNING] Failed to load advanced_config from DB: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training function."""
    args = parse_args()

    print("\n" + "="*80)
    print("Vision AI Training Platform - Adapter Pattern")
    print("="*80)
    print(f"[CONFIG] {json.dumps(vars(args), indent=2)}")

    # Load advanced_config from database if available
    advanced_config = load_advanced_config_from_db(args.job_id)
    if advanced_config:
        print(f"[CONFIG] Advanced config loaded: {json.dumps(advanced_config, indent=2)}")

    # Create configuration objects
    try:
        # Determine appropriate image size based on framework
        image_size = args.image_size
        if args.framework == 'ultralytics' and image_size == 224:
            # YOLO models typically use 640x640
            image_size = 640
            print(f"[CONFIG] Adjusting image_size from 224 to 640 for YOLO")

        model_config = ModelConfig(
            framework=args.framework,
            task_type=TaskType(args.task_type),
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            image_size=image_size,
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
            advanced_config=advanced_config,  # Pass advanced_config to TrainingConfig
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

    # Print complete configuration summary
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*80)
    print(f"[CONFIG] Job ID: {args.job_id}")
    print(f"[CONFIG] Framework: {args.framework}")
    print(f"[CONFIG] Task Type: {args.task_type}")
    print(f"[CONFIG] Model: {args.model_name}")
    print(f"[CONFIG] Pretrained: {args.pretrained}")
    print(f"[CONFIG] Image Size: {model_config.image_size}")  # Use actual config value
    if args.num_classes:
        print(f"[CONFIG] Number of Classes: {args.num_classes}")
    print(f"\n[CONFIG] Training Parameters:")
    print(f"         - Epochs: {args.epochs}")
    print(f"         - Batch Size: {args.batch_size}")
    print(f"         - Base Learning Rate: {args.learning_rate}")
    print(f"         - Device: {args.device}")
    print(f"\n[CONFIG] Dataset:")
    print(f"         - Path: {args.dataset_path}")
    print(f"         - Format: {args.dataset_format}")
    print(f"\n[CONFIG] Output Directory: {args.output_dir}")

    if advanced_config:
        print(f"\n[CONFIG] Advanced Configuration: ENABLED")
        if 'optimizer' in advanced_config:
            print(f"         - Optimizer will be configured from advanced_config")
        if 'scheduler' in advanced_config:
            print(f"         - Scheduler will be configured from advanced_config")
        if 'augmentation' in advanced_config and advanced_config['augmentation'].get('enabled'):
            print(f"         - Custom augmentation will be applied")
        if advanced_config.get('mixed_precision'):
            print(f"         - Mixed Precision Training: ENABLED")
        if advanced_config.get('gradient_clip_value'):
            print(f"         - Gradient Clipping: {advanced_config['gradient_clip_value']}")
    else:
        print(f"\n[CONFIG] Advanced Configuration: DISABLED (using defaults)")

    # Create adapter instance
    print(f"\n[INFO] Creating {args.framework} adapter for {args.task_type}")
    adapter = adapter_class(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=args.output_dir,
        job_id=args.job_id
    )

    # Load checkpoint if provided
    start_epoch = 0
    checkpoint_to_load = None
    resume_training = False

    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
            sys.exit(1)

        print(f"\n[INFO] Will load checkpoint from: {args.checkpoint_path}")
        print(f"[INFO] Resume training: {args.resume}")

        checkpoint_to_load = args.checkpoint_path
        resume_training = args.resume

    # Train
    try:
        print("\n" + "="*80)
        if checkpoint_to_load:
            if resume_training:
                print("RESUMING TRAINING FROM CHECKPOINT")
            else:
                print("STARTING TRAINING WITH PRETRAINED WEIGHTS")
        else:
            print("STARTING TRAINING")
        print("="*80 + "\n")

        metrics = adapter.train(
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_to_load,
            resume_training=resume_training
        )

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

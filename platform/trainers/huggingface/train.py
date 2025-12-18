#!/usr/bin/env python3
"""
HuggingFace Transformers Trainer

Simple CLI script for training vision models using HuggingFace Transformers.
All observability (MLflow, Loki, Prometheus) is handled by Backend.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name google/vit-base-patch16-224 \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 10, "batch_size": 16, "learning_rate": 0.00005}'

Environment Variables (alternative to CLI args):
    JOB_ID, MODEL_NAME, DATASET_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Training failure
    2 = Callback failure
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, Features, Image as HFImage
from dotenv import load_dotenv
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from trainer_sdk import ErrorType, TrainerSDK

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerSDKLogHandler(logging.Handler):
    """
    Custom logging handler that forwards logs to Backend via TrainerSDK.

    This allows existing logger.info() calls to automatically send logs
    to Backend → WebSocket → Frontend without code changes.
    """

    def __init__(self, sdk: TrainerSDK):
        super().__init__()
        self.sdk = sdk
        self._enabled = True

    def emit(self, record: logging.LogRecord):
        if not self._enabled:
            return
        try:
            level_map = {
                logging.DEBUG: 'DEBUG',
                logging.INFO: 'INFO',
                logging.WARNING: 'WARNING',
                logging.ERROR: 'ERROR',
                logging.CRITICAL: 'ERROR',
            }
            level = level_map.get(record.levelno, 'INFO')
            message = self.format(record)
            self.sdk.log(message, level=level, source='trainer')
        except Exception:
            pass

    def disable(self):
        """Disable the handler (used during shutdown)"""
        self._enabled = False


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='HuggingFace Transformers Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='HuggingFace model name')
    parser.add_argument('--dataset-s3-uri', type=str, help='S3 URI to dataset')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Training config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to training config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from args or environment"""
    job_id = args.job_id or os.getenv('JOB_ID')
    model_name = args.model_name or os.getenv('MODEL_NAME')
    dataset_s3_uri = args.dataset_s3_uri or os.getenv('DATASET_S3_URI')
    callback_url = args.callback_url or os.getenv('CALLBACK_URL')

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    elif os.getenv('CONFIG'):
        config = json.loads(os.getenv('CONFIG'))
    else:
        config = {}

    if not all([job_id, model_name, dataset_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: job_id, model_name, dataset_s3_uri, callback_url")

    os.environ['JOB_ID'] = str(job_id)
    os.environ['CALLBACK_URL'] = callback_url

    return {
        'job_id': job_id,
        'model_name': model_name,
        'dataset_s3_uri': dataset_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


def load_imagefolder_as_hf_dataset(
    split_dir: Path,
    class_names: list,
    processor: AutoImageProcessor
) -> Dataset:
    """Load ImageFolder structure as HuggingFace Dataset"""
    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.append(str(img_path))
                labels.append(class_idx)

    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })

    dataset = dataset.cast_column("image", HFImage())

    return dataset


def preprocess_function(examples, processor):
    """Preprocess images using HuggingFace processor"""
    images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    import numpy as np
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


class ProgressCallback:
    """Callback to report progress to backend"""

    def __init__(self, sdk: TrainerSDK, total_epochs: int):
        self.sdk = sdk
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch += 1
        metrics = {}

        if state.log_history:
            for log in reversed(state.log_history):
                if 'loss' in log:
                    metrics['train_loss'] = log['loss']
                    break

        self.sdk.report_progress(
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            metrics=metrics
        )


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        cfg = load_config(args)
        job_id = cfg['job_id']
        model_name = cfg['model_name']
        dataset_s3_uri = cfg['dataset_s3_uri']
        config = cfg['config']

        logger.info(f"Starting training job {job_id}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_s3_uri}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")

        # Initialize SDK
        sdk = TrainerSDK()

        # Add SDK log handler to forward logs to Backend
        sdk_handler = TrainerSDKLogHandler(sdk)
        sdk_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sdk_handler)

        sdk.report_started(
            model_name=model_name,
            config=config,
            dataset_uri=dataset_s3_uri
        )

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Download dataset
        logger.info("Downloading dataset...")
        dataset_path = sdk.download_dataset(dataset_s3_uri)
        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Create output directory
        output_dir = Path(f"./outputs/job_{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load processor
        logger.info(f"Loading processor for {model_name}...")
        processor = AutoImageProcessor.from_pretrained(model_name)

        # Detect classes
        train_dir = dataset_path / "train"
        if not train_dir.exists():
            train_dir = dataset_path

        class_names = sorted([
            d.name for d in train_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        num_classes = len(class_names)
        logger.info(f"Detected {num_classes} classes: {class_names}")

        # Load datasets
        logger.info("Loading train dataset...")
        train_dataset = load_imagefolder_as_hf_dataset(train_dir, class_names, processor)

        val_dir = dataset_path / "val"
        if val_dir.exists():
            logger.info("Loading validation dataset...")
            val_dataset = load_imagefolder_as_hf_dataset(val_dir, class_names, processor)
        else:
            logger.info("No val folder, splitting train 80/20...")
            split = train_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split['train']
            val_dataset = split['test']

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, processor),
            batched=True,
            remove_columns=["image"]
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_function(x, processor),
            batched=True,
            remove_columns=["image"]
        )

        # Load model
        logger.info(f"Loading model {model_name}...")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label={i: name for i, name in enumerate(class_names)},
            label2id={name: i for i, name in enumerate(class_names)}
        )
        model = model.to(device)

        # Training configuration
        epochs = config.get('epochs', 10)
        batch_size = config.get('batch_size', config.get('batch', 16))
        learning_rate = config.get('learning_rate', config.get('lr', 5e-5))

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=config.get('weight_decay', 0.01),
            warmup_ratio=config.get('warmup_ratio', 0.1),
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        train_result = trainer.train()

        # Evaluate
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        logger.info(f"Final evaluation: {eval_result}")

        # Save best model
        best_model_dir = output_dir / "best_model"
        trainer.save_model(str(best_model_dir))
        processor.save_pretrained(str(best_model_dir))
        logger.info(f"Best model saved to {best_model_dir}")

        # Upload checkpoint
        logger.info("Uploading best model...")
        checkpoint_s3_uri = sdk.upload_checkpoint(best_model_dir)
        logger.info(f"Model uploaded to: {checkpoint_s3_uri}")

        # Report completion
        final_metrics = {
            'accuracy': eval_result.get('eval_accuracy', 0) * 100,
            'eval_loss': eval_result.get('eval_loss', 0),
            'train_loss': train_result.metrics.get('train_loss', 0),
        }

        sdk.report_completed(
            metrics=final_metrics,
            checkpoint_path=str(best_model_dir),
            output_dir=str(output_dir)
        )

        logger.info(f"Training completed! Accuracy: {final_metrics['accuracy']:.2f}%")

        # Flush remaining logs and cleanup handler
        sdk.flush_logs()
        sdk_handler.disable()
        logger.removeHandler(sdk_handler)

        sdk.close()
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

        # Cleanup handler if it exists
        try:
            if 'sdk_handler' in dir() and sdk_handler is not None:
                sdk.flush_logs()
                sdk_handler.disable()
                logger.removeHandler(sdk_handler)
                sdk.close()
        except Exception:
            pass

        try:
            error_sdk = TrainerSDK()
            error_sdk.report_failed(
                error_type=ErrorType.TRAINING_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            error_sdk.close()
        except Exception as callback_error:
            logger.error(f"Failed to report error: {callback_error}")
            return 2

        return 1


if __name__ == "__main__":
    sys.exit(main())

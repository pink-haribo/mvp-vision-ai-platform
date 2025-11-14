"""
CLI Entry Point for Training Service

Allows executing training jobs via subprocess (local dev) or K8s Job (production).

Usage:
    # Basic usage with all arguments
    python -m app \\
        --job-id "job_123" \\
        --model-name "yolo11n" \\
        --dataset-s3-uri "s3://bucket/datasets/my-dataset" \\
        --callback-url "http://localhost:8000/api/v1/training" \\
        --config '{"epochs": 50, "batch_size": 16}'

    # Using config file (recommended for complex configs)
    python -m app \\
        --job-id "job_123" \\
        --model-name "yolo11n" \\
        --dataset-s3-uri "s3://bucket/datasets/my-dataset" \\
        --callback-url "http://localhost:8000/api/v1/training" \\
        --config-file /path/to/config.json

Exit Codes:
    0: Training completed successfully
    1: Training failed (exception in training logic)
    2: Callback failure (training succeeded but notification failed)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

from app.trainer.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_str: str = None, config_file: str = None) -> Dict[str, Any]:
    """
    Load training configuration from JSON string or file.

    Args:
        config_str: JSON string containing config
        config_file: Path to JSON config file

    Returns:
        Parsed configuration dictionary

    Raises:
        ValueError: If neither config_str nor config_file provided, or if JSON invalid
    """
    if config_file:
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.info(f"Loaded config from file: {config_file}")
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_file}: {e}")

    elif config_str:
        try:
            config = json.loads(config_str)
            logger.info("Loaded config from command-line argument")
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --config argument: {e}")

    else:
        # Empty config is acceptable (will use defaults)
        logger.info("No config provided, using defaults")
        return {}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with environment variable fallback.

    For K8s Job compatibility, each argument can be provided via:
    1. CLI argument (e.g., --job-id 123)
    2. Environment variable (e.g., JOB_ID=123)

    CLI arguments take precedence over environment variables.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics Training Service - CLI Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # K8s Job compatible: Read from CLI args or environment variables
    parser.add_argument(
        "--job-id",
        default=os.getenv("JOB_ID"),
        required=not os.getenv("JOB_ID"),
        help="Training job ID (env: JOB_ID)"
    )

    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME"),
        required=not os.getenv("MODEL_NAME"),
        help="Model name (env: MODEL_NAME)"
    )

    parser.add_argument(
        "--dataset-s3-uri",
        default=os.getenv("DATASET_S3_URI"),
        required=not os.getenv("DATASET_S3_URI"),
        help="S3 URI of dataset (env: DATASET_S3_URI)"
    )

    parser.add_argument(
        "--callback-url",
        default=os.getenv("CALLBACK_URL"),
        required=not os.getenv("CALLBACK_URL"),
        help="Backend API base URL (env: CALLBACK_URL)"
    )

    # Config can be provided as JSON string, file, or environment variable
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config",
        default=os.getenv("TRAINING_CONFIG"),
        help="Training configuration as JSON string (env: TRAINING_CONFIG)"
    )
    config_group.add_argument(
        "--config-file",
        help="Path to training configuration JSON file"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (env: LOG_LEVEL, default: INFO)"
    )

    return parser.parse_args()


def main():
    """
    Main CLI entry point.

    Parses arguments, loads configuration, and executes training.
    Exit code is handled by train_model() when EXECUTION_MODE="job".
    """
    try:
        # Parse arguments
        args = parse_args()

        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))

        logger.info("=" * 80)
        logger.info(f"Ultralytics Training Service - Job {args.job_id}")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset_s3_uri}")
        logger.info(f"Callback URL: {args.callback_url}")

        # Load configuration
        config = load_config(
            config_str=args.config,
            config_file=args.config_file
        )

        logger.info(f"Training config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

        # Run async training
        asyncio.run(train_model(
            job_id=args.job_id,
            model_name=args.model_name,
            dataset_s3_uri=args.dataset_s3_uri,
            callback_url=args.callback_url,
            config=config
        ))

        # If we reach here in "job" mode, train_model() called sys.exit()
        # If we reach here in "service" mode, training completed successfully
        logger.info(f"Training job {args.job_id} completed successfully")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error in training job: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

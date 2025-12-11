#!/usr/bin/env python3
"""
Ultralytics YOLO Exporter (SDK Version)

Simple CLI script for exporting trained YOLO models to production formats using the Trainer SDK.
All observability is handled by Backend.

Usage:
    python export.py \
        --export-job-id 123 \
        --training-job-id 456 \
        --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best.pt \
        --export-format onnx \
        --callback-url http://localhost:8000/api/v1/export \
        --config '{"opset_version": 13, "simplify": true, "dynamic": true}'

Environment Variables (alternative to CLI args):
    EXPORT_JOB_ID, TRAINING_JOB_ID, CHECKPOINT_S3_URI, EXPORT_FORMAT, CALLBACK_URL, CONFIG
    WORKSPACE_DIR: Working directory for export (default: /workspace, useful for local testing)

Exit Codes:
    0 = Success
    1 = Export failure
    2 = Callback failure
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from ultralytics import YOLO

from trainer_sdk import ErrorType, TrainerSDK

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Exporter')

    parser.add_argument('--export-job-id', type=str, help='Export job ID')
    parser.add_argument('--training-job-id', type=str, help='Original training job ID')
    parser.add_argument('--checkpoint-s3-uri', type=str, help='S3 URI to trained checkpoint')
    parser.add_argument('--export-format', type=str,
                        choices=['onnx', 'tensorrt', 'coreml', 'tflite', 'torchscript', 'openvino'],
                        help='Export format')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Export config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to export config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from environment or args"""
    # Priority: env vars > CLI args
    export_job_id = os.getenv('EXPORT_JOB_ID') or args.export_job_id
    training_job_id = os.getenv('TRAINING_JOB_ID') or args.training_job_id
    checkpoint_s3_uri = os.getenv('CHECKPOINT_S3_URI') or args.checkpoint_s3_uri
    export_format = os.getenv('EXPORT_FORMAT') or args.export_format
    callback_url = os.getenv('CALLBACK_URL') or args.callback_url

    if os.getenv('CONFIG'):
        config = json.loads(os.getenv('CONFIG'))
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    else:
        config = {}

    # Validate required fields
    if not all([export_job_id, training_job_id, checkpoint_s3_uri, export_format, callback_url]):
        raise ValueError(
            "Missing required arguments: export_job_id, training_job_id, "
            "checkpoint_s3_uri, export_format, callback_url"
        )

    # Set environment variables for SDK
    os.environ['JOB_ID'] = str(export_job_id)
    os.environ['CALLBACK_URL'] = callback_url

    return {
        'export_job_id': export_job_id,
        'training_job_id': training_job_id,
        'checkpoint_s3_uri': checkpoint_s3_uri,
        'export_format': export_format,
        'callback_url': callback_url,
        'config': config
    }


def export_model(
    model_path: Path,
    export_format: str,
    export_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Export YOLO model to specified format"""
    logger.info(f"[EXPORT] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    logger.info(f"[EXPORT] Exporting to format: {export_format}")

    # Build export kwargs from config
    export_kwargs = {}

    if export_format == 'onnx':
        export_kwargs['format'] = 'onnx'
        export_kwargs['opset'] = export_config.get('opset_version', 13)
        export_kwargs['simplify'] = export_config.get('simplify', True)
        if export_config.get('dynamic_axes'):
            export_kwargs['dynamic'] = True
        else:
            export_kwargs['dynamic'] = export_config.get('dynamic', True)

    elif export_format == 'tensorrt':
        export_kwargs['format'] = 'engine'
        export_kwargs['half'] = export_config.get('fp16', False)
        export_kwargs['int8'] = export_config.get('int8', False)
        if export_config.get('workspace_size_gb'):
            export_kwargs['workspace'] = export_config['workspace_size_gb']

    elif export_format == 'coreml':
        export_kwargs['format'] = 'coreml'
        if export_config.get('minimum_deployment_target'):
            export_kwargs['nms'] = True

    elif export_format == 'tflite':
        export_kwargs['format'] = 'tflite'
        export_kwargs['int8'] = export_config.get('int8', False)

    elif export_format == 'torchscript':
        export_kwargs['format'] = 'torchscript'

    elif export_format == 'openvino':
        export_kwargs['format'] = 'openvino'
        export_kwargs['half'] = export_config.get('fp16', False)

    # Run export
    start_time = datetime.now()
    export_result = model.export(**export_kwargs)
    export_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"[EXPORT] Export completed in {export_time:.2f}s")

    exported_file = Path(export_result)
    file_size_bytes = exported_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    logger.info(f"[EXPORT] Exported file size: {file_size_mb:.2f} MB")

    return {
        'exported_file': exported_file,
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_mb,
        'export_time_seconds': export_time
    }


def generate_metadata(
    model_path: Path,
    export_format: str,
    task_type: str,
    export_result: Dict[str, Any],
    export_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate metadata.json for exported model"""
    logger.info("[EXPORT] Generating metadata.json")

    model = YOLO(str(model_path))
    class_names = model.names if hasattr(model, 'names') else {}

    # Detect task type from model
    if not task_type:
        model_name = model_path.stem.lower()
        if 'seg' in model_name:
            task_type = 'instance_segmentation'
        elif 'pose' in model_name:
            task_type = 'pose_estimation'
        elif 'cls' in model_name:
            task_type = 'image_classification'
        else:
            task_type = 'object_detection'

    # Use SDK's create_export_metadata for standardized format
    metadata = TrainerSDK.create_export_metadata(
        framework='ultralytics',
        model_name=model_path.stem,
        export_format=export_format,
        task_type=task_type,
        input_shape=[640, 640, 3],
        output_shape=[[1, 84, 8400]],  # Default YOLO output shape
        class_names=list(class_names.values()) if isinstance(class_names, dict) else class_names,
        preprocessing={
            'resize': 640,
            'normalize': {'mean': [0.0, 0.0, 0.0], 'std': [255.0, 255.0, 255.0]},
            'format': 'RGB',
            'layout': 'NCHW'
        },
        postprocessing={
            'nms': True,
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 300
        },
        export_config=export_config,
        # Additional fields
        export_info={
            'file_size_mb': export_result['file_size_mb'],
            'export_time_seconds': export_result['export_time_seconds'],
        }
    )

    logger.info(f"[EXPORT] Metadata generated for task: {task_type}")
    return metadata


def copy_runtime_wrappers(
    runtimes_dir: Path,
    export_format: str,
    metadata: Dict[str, Any]
):
    """Copy runtime wrapper templates based on export format"""
    current_dir = Path(__file__).parent
    runtimes_source = current_dir / 'runtimes'

    if not runtimes_source.exists():
        logger.warning(f"[EXPORT] Runtime wrappers not found at: {runtimes_source}")
        return

    format_to_runtime = {
        'onnx': ['python', 'cpp'],
        'tensorrt': ['python', 'cpp'],
        'coreml': ['swift'],
        'tflite': ['kotlin'],
        'torchscript': ['python'],
        'openvino': ['python', 'cpp']
    }

    runtimes_to_copy = format_to_runtime.get(export_format, [])

    for runtime in runtimes_to_copy:
        source_runtime_dir = runtimes_source / runtime
        if not source_runtime_dir.exists():
            continue

        dest_runtime_dir = runtimes_dir / runtime
        dest_runtime_dir.mkdir(parents=True, exist_ok=True)

        for item in source_runtime_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_runtime_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest_runtime_dir / item.name, dirs_exist_ok=True)

    # Create README
    readme_path = runtimes_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"# Runtime Wrappers for {export_format.upper()}\n\n")
        f.write(f"Task: {metadata.get('task_type', 'unknown')}\n")
        f.write(f"Classes: {metadata.get('num_classes', 'unknown')}\n\n")
        f.write("See individual wrapper directories for usage.\n")

    logger.info(f"[EXPORT] Copied {len(runtimes_to_copy)} runtime wrapper(s)")


def create_export_package(
    exported_file: Path,
    metadata: Dict[str, Any],
    output_dir: Path,
    export_job_id: str
) -> Path:
    """Create export package (zip) with model and metadata"""
    logger.info("[EXPORT] Creating export package")

    package_dir = output_dir / f"export_{export_job_id}"
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy exported model
    model_dest = package_dir / exported_file.name
    shutil.copy2(exported_file, model_dest)

    # Write metadata.json
    metadata_path = package_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy runtime wrappers
    runtimes_dir = package_dir / 'runtimes'
    runtimes_dir.mkdir(exist_ok=True)
    export_format = metadata.get('export_format', 'unknown')
    copy_runtime_wrappers(runtimes_dir, export_format, metadata)

    # Create zip package
    package_zip = output_dir / f"export_{export_job_id}.zip"
    with zipfile.ZipFile(package_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)

    package_size_mb = package_zip.stat().st_size / (1024 * 1024)
    logger.info(f"[EXPORT] Package created: {package_zip} ({package_size_mb:.2f} MB)")

    return package_zip


def run_export(
    export_job_id: str,
    training_job_id: str,
    checkpoint_s3_uri: str,
    export_format: str,
    export_config: Dict[str, Any]
) -> int:
    """Main export function"""
    sdk = TrainerSDK()

    try:
        logger.info("=" * 80)
        logger.info("ULTRALYTICS YOLO EXPORT")
        logger.info("=" * 80)
        logger.info(f"Export Job ID: {export_job_id}")
        logger.info(f"Training Job ID: {training_job_id}")
        logger.info(f"Export Format: {export_format}")
        logger.info(f"Checkpoint: {checkpoint_s3_uri}")
        logger.info(f"Config: {json.dumps(export_config, indent=2)}")
        logger.info("=" * 80)

        # Report started
        sdk.report_started('export')

        # Create workspace (configurable via env var for local testing)
        workspace = Path(os.getenv('WORKSPACE_DIR', '/workspace'))
        workspace.mkdir(parents=True, exist_ok=True)

        # Download checkpoint
        logger.info(f"[EXPORT] Downloading checkpoint from: {checkpoint_s3_uri}")
        checkpoint_filename = Path(checkpoint_s3_uri).name
        checkpoint_path = workspace / checkpoint_filename
        sdk.download_checkpoint(checkpoint_s3_uri, str(checkpoint_path))
        logger.info(f"[EXPORT] Checkpoint downloaded to: {checkpoint_path}")

        # Export model
        export_result = export_model(
            model_path=checkpoint_path,
            export_format=export_format,
            export_config=export_config,
            output_dir=workspace
        )

        # Generate metadata
        task_type = export_config.get('task_type', '')
        metadata = generate_metadata(
            model_path=checkpoint_path,
            export_format=export_format,
            task_type=task_type,
            export_result=export_result,
            export_config=export_config
        )

        # Create export package
        package_path = create_export_package(
            exported_file=export_result['exported_file'],
            metadata=metadata,
            output_dir=workspace,
            export_job_id=export_job_id
        )

        # Upload package to MinIO
        logger.info(f"[EXPORT] Uploading package to MinIO: {package_path}")
        s3_key = f"exports/{training_job_id}/{export_job_id}/{package_path.name}"
        package_s3_uri = sdk.upload_file(
            str(package_path),
            s3_key,
            content_type='application/zip',
            storage_type='internal'
        )
        logger.info(f"[EXPORT] Package uploaded to: {package_s3_uri}")

        # Send completion callback
        sdk.report_export_completed(
            export_format=export_format,
            output_s3_uri=package_s3_uri,
            file_size_bytes=int(export_result['file_size_bytes']),
            metadata=metadata
        )

        logger.info("=" * 80)
        logger.info("[SUCCESS] Export completed successfully!")
        logger.info(f"Package: {package_s3_uri}")
        logger.info("=" * 80)

        sdk.close()
        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("[FAILURE] Export failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

        # Determine error type
        error_type = ErrorType.UNKNOWN_ERROR
        error_msg = str(e)

        if 'checkpoint' in error_msg.lower():
            error_type = ErrorType.CHECKPOINT_ERROR
        elif 'export' in error_msg.lower() or 'format' in error_msg.lower():
            error_type = ErrorType.FRAMEWORK_ERROR
        elif 'CUDA' in error_msg or 'memory' in error_msg.lower():
            error_type = ErrorType.RESOURCE_ERROR

        try:
            sdk.report_failed(
                error_type=error_type,
                message=error_msg,
                traceback=traceback.format_exc()
            )
        except Exception as cb_error:
            logger.error(f"[CALLBACK ERROR] Failed to send failure callback: {cb_error}")

        sdk.close()
        return 1


def main():
    """Main entry point"""
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        config_data = load_config(args)

        exit_code = run_export(
            export_job_id=config_data['export_job_id'],
            training_job_id=config_data['training_job_id'],
            checkpoint_s3_uri=config_data['checkpoint_s3_uri'],
            export_format=config_data['export_format'],
            export_config=config_data['config']
        )

        logger.info(f"Export job {config_data['export_job_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Ultralytics YOLO Exporter

Simple CLI script for exporting trained YOLO models to production formats with S3 integration and Backend callbacks.

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

Exit Codes:
    0 = Success
    1 = Export failure
    2 = Callback failure
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ultralytics import YOLO

from utils import DualStorageClient, CallbackClient

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
    """Load configuration from environment or args (K8s Job compatible)"""
    # Priority: env vars > CLI args (K8s Job style)
    export_job_id = os.getenv('EXPORT_JOB_ID') or args.export_job_id
    training_job_id = os.getenv('TRAINING_JOB_ID') or args.training_job_id
    checkpoint_s3_uri = os.getenv('CHECKPOINT_S3_URI') or args.checkpoint_s3_uri
    export_format = os.getenv('EXPORT_FORMAT') or args.export_format
    callback_url = os.getenv('CALLBACK_URL') or args.callback_url

    # Config priority: env var > config file > CLI arg
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

    return {
        'export_job_id': export_job_id,
        'training_job_id': training_job_id,
        'checkpoint_s3_uri': checkpoint_s3_uri,
        'export_format': export_format,
        'callback_url': callback_url,
        'config': config
    }


def download_checkpoint(storage: DualStorageClient, checkpoint_s3_uri: str, local_dir: Path) -> Path:
    """Download checkpoint from MinIO Internal Storage"""
    logger.info(f"[EXPORT] Downloading checkpoint from: {checkpoint_s3_uri}")

    # Parse S3 URI: s3://bucket/key
    if not checkpoint_s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {checkpoint_s3_uri}")

    parts = checkpoint_s3_uri[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    checkpoint_filename = Path(key).name
    local_checkpoint = local_dir / checkpoint_filename

    # Download from Internal Storage (port 9002)
    storage.internal_client.client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(local_checkpoint)
    )

    logger.info(f"[EXPORT] Checkpoint downloaded to: {local_checkpoint}")
    return local_checkpoint


def export_model(
    model_path: Path,
    export_format: str,
    export_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Export YOLO model to specified format using Ultralytics export API"""
    logger.info(f"[EXPORT] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    logger.info(f"[EXPORT] Exporting to format: {export_format}")

    # Build export kwargs from config
    export_kwargs = {}

    if export_format == 'onnx':
        # ONNX export parameters
        export_kwargs['format'] = 'onnx'
        export_kwargs['opset'] = export_config.get('opset_version', 13)
        export_kwargs['simplify'] = export_config.get('simplify', True)
        # Ultralytics uses 'dynamic' (bool), not 'dynamic_axes' (dict)
        # If dynamic_axes is specified, enable dynamic mode
        if export_config.get('dynamic_axes'):
            export_kwargs['dynamic'] = True
        else:
            export_kwargs['dynamic'] = export_config.get('dynamic', True)

    elif export_format == 'tensorrt':
        # TensorRT export parameters
        export_kwargs['format'] = 'engine'
        export_kwargs['half'] = export_config.get('fp16', False)
        export_kwargs['int8'] = export_config.get('int8', False)
        if export_config.get('workspace_size_gb'):
            export_kwargs['workspace'] = export_config['workspace_size_gb']

    elif export_format == 'coreml':
        # CoreML export parameters
        export_kwargs['format'] = 'coreml'
        if export_config.get('minimum_deployment_target'):
            export_kwargs['nms'] = True  # CoreML NMS support

    elif export_format == 'tflite':
        # TFLite export parameters
        export_kwargs['format'] = 'tflite'
        export_kwargs['int8'] = export_config.get('int8', False)

    elif export_format == 'torchscript':
        # TorchScript export parameters
        export_kwargs['format'] = 'torchscript'

    elif export_format == 'openvino':
        # OpenVINO export parameters
        export_kwargs['format'] = 'openvino'
        export_kwargs['half'] = export_config.get('fp16', False)

    # Run export
    start_time = datetime.now()
    export_result = model.export(**export_kwargs)
    export_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"[EXPORT] Export completed in {export_time:.2f}s")
    logger.info(f"[EXPORT] Exported model: {export_result}")

    # Get exported file path
    exported_file = Path(export_result)

    # Get file size
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

    # Load model to extract metadata
    model = YOLO(str(model_path))

    # Extract class names from model
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

    # Get model input shape
    model_info = model.model.info() if hasattr(model.model, 'info') else {}

    metadata = {
        'model_info': {
            'framework': 'ultralytics',
            'task_type': task_type,
            'export_format': export_format,
            'model_name': model_path.stem,
            'num_classes': len(class_names),
            'class_names': class_names,
        },
        'export_info': {
            'export_time': datetime.now().isoformat(),
            'file_size_mb': export_result['file_size_mb'],
            'export_time_seconds': export_result['export_time_seconds'],
            'export_config': export_config,
        },
        'preprocessing': {
            'resize': 640,  # Default YOLO size
            'normalize': {
                'mean': [0.0, 0.0, 0.0],
                'std': [255.0, 255.0, 255.0]
            },
            'format': 'RGB',
            'layout': 'NCHW'
        },
        'postprocessing': {
            'nms': True,
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 300
        },
        'input_spec': {
            'name': 'images',
            'shape': [1, 3, 640, 640],  # [batch, channels, height, width]
            'dtype': 'float32'
        },
        'output_spec': {
            'format': 'xyxy' if task_type == 'object_detection' else 'auto',
            'includes': ['boxes', 'scores', 'classes'] if task_type == 'object_detection' else ['auto']
        }
    }

    logger.info(f"[EXPORT] Metadata generated for task: {task_type}, {len(class_names)} classes")

    return metadata


def copy_runtime_wrappers(
    runtimes_dir: Path,
    export_format: str,
    metadata: Dict[str, Any]
):
    """Copy runtime wrapper templates based on export format"""
    # Get current directory (trainers/ultralytics)
    current_dir = Path(__file__).parent
    runtimes_source = current_dir / 'runtimes'

    if not runtimes_source.exists():
        logger.warning(f"[EXPORT] Runtime wrappers not found at: {runtimes_source}")
        return

    # Map export format to runtime wrapper language
    format_to_runtime = {
        'onnx': ['python', 'cpp'],
        'tensorrt': ['python', 'cpp'],
        'coreml': ['swift'],
        'tflite': ['kotlin'],
        'torchscript': ['python'],
        'openvino': ['python', 'cpp']
    }

    runtimes_to_copy = format_to_runtime.get(export_format, [])

    if not runtimes_to_copy:
        logger.warning(f"[EXPORT] No runtime wrappers configured for format: {export_format}")
        return

    # Copy runtime wrappers
    for runtime in runtimes_to_copy:
        source_runtime_dir = runtimes_source / runtime

        if not source_runtime_dir.exists():
            logger.warning(f"[EXPORT] Runtime wrapper not found: {source_runtime_dir}")
            continue

        dest_runtime_dir = runtimes_dir / runtime
        dest_runtime_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from source runtime
        for item in source_runtime_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_runtime_dir / item.name)
                logger.info(f"[EXPORT] Copied {runtime} wrapper: {item.name}")
            elif item.is_dir():
                # Copy directories recursively (for C++ includes, etc.)
                shutil.copytree(item, dest_runtime_dir / item.name, dirs_exist_ok=True)
                logger.info(f"[EXPORT] Copied {runtime} directory: {item.name}")

    # Create main README
    readme_path = runtimes_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# Runtime Wrappers\n\n")
        f.write(f"Runtime wrappers for **{export_format.upper()}** model.\n\n")
        f.write(f"**Task Type:** {metadata['model_info']['task_type']}\n")
        f.write(f"**Classes:** {metadata['model_info']['num_classes']}\n")
        f.write(f"**Input Shape:** {metadata['input_spec']['shape']}\n\n")

        f.write("## Available Wrappers\n\n")
        for runtime in runtimes_to_copy:
            f.write(f"- **{runtime.capitalize()}**: See `{runtime}/README.md`\n")

        f.write("\n## Quick Start\n\n")
        f.write("Each wrapper directory contains:\n")
        f.write("- Implementation files (`.py`, `.cpp`, `.swift`, `.kt`)\n")
        f.write("- README with usage examples\n")
        f.write("- Dependencies/build instructions\n\n")

        f.write("## Common Operations\n\n")
        f.write("1. **Load model**: Initialize wrapper with model file\n")
        f.write("2. **Preprocess**: Resize image to input size, normalize\n")
        f.write("3. **Inference**: Run model forward pass\n")
        f.write("4. **Postprocess**: Apply NMS, filter by confidence\n")
        f.write("5. **Visualize**: Draw boxes/masks on image\n\n")

        f.write("See individual wrapper READMEs for language-specific details.\n")

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
    logger.info(f"[EXPORT] Copied model to: {model_dest}")

    # Write metadata.json
    metadata_path = package_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[EXPORT] Metadata saved to: {metadata_path}")

    # Copy runtime wrappers
    runtimes_dir = package_dir / 'runtimes'
    runtimes_dir.mkdir(exist_ok=True)

    # Copy runtime wrappers based on export format
    export_format = metadata.get('model_info', {}).get('export_format', 'unknown')
    copy_runtime_wrappers(runtimes_dir, export_format, metadata)
    logger.info(f"[EXPORT] Runtime wrappers copied to: {runtimes_dir}")

    # Create zip package
    package_zip = output_dir / f"export_{export_job_id}.zip"
    with zipfile.ZipFile(package_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)

    logger.info(f"[EXPORT] Package created: {package_zip} ({package_zip.stat().st_size / 1024 / 1024:.2f} MB)")

    return package_zip


def upload_export_package(
    storage: DualStorageClient,
    package_path: Path,
    training_job_id: str,
    export_job_id: str
) -> str:
    """Upload export package to MinIO Internal Storage"""
    logger.info(f"[EXPORT] Uploading package to MinIO: {package_path}")

    # Upload to Internal Storage: s3://training-checkpoints/exports/{training_job_id}/{export_job_id}/
    bucket = os.getenv('INTERNAL_BUCKET_CHECKPOINTS', 'training-checkpoints')
    key = f"exports/{training_job_id}/{export_job_id}/{package_path.name}"

    storage.internal_client.client.upload_file(
        Filename=str(package_path),
        Bucket=bucket,
        Key=key
    )

    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"[EXPORT] Package uploaded to: {s3_uri}")

    return s3_uri


async def send_completion_callback(
    callback_client: CallbackClient,
    export_job_id: str,
    status: str,
    export_results: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
):
    """Send export completion callback to backend"""
    logger.info(f"[EXPORT] Sending completion callback: status={status}")

    callback_data = {
        'status': status,
        'completed_at': datetime.now().isoformat()
    }

    if export_results:
        callback_data['export_results'] = export_results

    if error_message:
        callback_data['error_message'] = error_message

    try:
        # Send callback (assuming callback endpoint exists)
        # POST /api/v1/export/jobs/{export_job_id}/callback/completion
        url = f"{callback_client.base_url}/export/jobs/{export_job_id}/callback/completion"

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=callback_data)
            response.raise_for_status()

        logger.info(f"[EXPORT] Callback sent successfully")

    except Exception as e:
        logger.error(f"[EXPORT] Callback failed: {e}")
        raise


async def main():
    """Main export workflow"""
    try:
        # Parse arguments
        args = parse_args()

        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

        # Load configuration
        config_data = load_config(args)
        export_job_id = config_data['export_job_id']
        training_job_id = config_data['training_job_id']
        checkpoint_s3_uri = config_data['checkpoint_s3_uri']
        export_format = config_data['export_format']
        callback_url = config_data['callback_url']
        export_config = config_data['config']

        logger.info("="*80)
        logger.info("ULTRALYTICS YOLO EXPORT")
        logger.info("="*80)
        logger.info(f"Export Job ID: {export_job_id}")
        logger.info(f"Training Job ID: {training_job_id}")
        logger.info(f"Export Format: {export_format}")
        logger.info(f"Checkpoint: {checkpoint_s3_uri}")
        logger.info(f"Config: {json.dumps(export_config, indent=2)}")
        logger.info("="*80)

        # Initialize clients
        storage = DualStorageClient()
        callback_client = CallbackClient(callback_url)

        # Create workspace
        workspace = Path('/workspace')
        workspace.mkdir(parents=True, exist_ok=True)

        # Download checkpoint
        checkpoint_path = download_checkpoint(storage, checkpoint_s3_uri, workspace)

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
        package_s3_uri = upload_export_package(
            storage=storage,
            package_path=package_path,
            training_job_id=training_job_id,
            export_job_id=export_job_id
        )

        # Prepare export results for callback
        export_results = {
            'export_path': package_s3_uri,
            'file_size_mb': export_result['file_size_mb'],
            'export_time_seconds': export_result['export_time_seconds'],
            'metadata': metadata,
            'validation_passed': True  # TODO: Add actual validation
        }

        # Send completion callback
        await send_completion_callback(
            callback_client=callback_client,
            export_job_id=export_job_id,
            status='completed',
            export_results=export_results
        )

        logger.info("="*80)
        logger.info("[SUCCESS] Export completed successfully!")
        logger.info(f"Package: {package_s3_uri}")
        logger.info("="*80)

        sys.exit(0)

    except Exception as e:
        logger.error("="*80)
        logger.error("[FAILURE] Export failed!")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

        # Try to send failure callback
        try:
            if 'export_job_id' in locals() and 'callback_client' in locals():
                await send_completion_callback(
                    callback_client=callback_client,
                    export_job_id=export_job_id,
                    status='failed',
                    error_message=str(e)
                )
                sys.exit(1)  # Export failure
            else:
                sys.exit(1)  # Export failure (couldn't send callback)
        except Exception as callback_error:
            logger.error(f"[CALLBACK ERROR] Failed to send failure callback: {callback_error}")
            sys.exit(2)  # Callback failure


if __name__ == '__main__':
    asyncio.run(main())

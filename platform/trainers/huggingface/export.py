#!/usr/bin/env python3
"""
HuggingFace Transformers Exporter (SDK Version)

Simple CLI script for exporting trained HuggingFace models to production formats using the Trainer SDK.
All observability is handled by Backend.

Supported Export Formats:
- ONNX: Cross-platform, optimized for inference
- TorchScript: PyTorch native deployment

Usage:
    python export.py \
        --export-job-id 123 \
        --training-job-id 456 \
        --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best_model/ \
        --export-format onnx \
        --callback-url http://localhost:8000/api/v1/export \
        --config '{"opset_version": 17, "dynamic_axes": true}'

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

import torch
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
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


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='HuggingFace Transformers Exporter')

    parser.add_argument('--export-job-id', type=str, help='Export job ID')
    parser.add_argument('--training-job-id', type=str, help='Original training job ID')
    parser.add_argument('--checkpoint-s3-uri', type=str, help='S3 URI to trained checkpoint')
    parser.add_argument('--export-format', type=str,
                        choices=['onnx', 'torchscript'],
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


def load_model(model_path: Path):
    """Load HuggingFace model from checkpoint directory"""
    logger.info(f"[EXPORT] Loading model from: {model_path}")

    # Load the model
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    model.eval()

    # Load the image processor if available
    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(str(model_path))
        logger.info("[EXPORT] Image processor loaded successfully")
    except Exception as e:
        logger.warning(f"[EXPORT] Could not load image processor: {e}")

    # Load config for metadata
    config = AutoConfig.from_pretrained(str(model_path))

    return model, processor, config


def get_input_shape(processor, config) -> tuple:
    """Determine input shape from processor or config"""
    # Default shape for ViT models
    default_size = 224

    if processor is not None:
        # Try to get size from processor
        if hasattr(processor, 'size'):
            size_info = processor.size
            if isinstance(size_info, dict):
                height = size_info.get('height', size_info.get('shortest_edge', default_size))
                width = size_info.get('width', size_info.get('shortest_edge', default_size))
            else:
                height = width = size_info
        else:
            height = width = default_size
    else:
        # Fallback to config or default
        height = width = getattr(config, 'image_size', default_size)

    return (3, height, width)  # CHW format


def export_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    export_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Export model to ONNX format"""
    logger.info("[EXPORT] Exporting to ONNX format...")

    # Get export options
    opset_version = export_config.get('opset_version', 17)
    dynamic_axes = export_config.get('dynamic_axes', True)

    # Create dummy input (batch, channels, height, width)
    dummy_input = torch.randn(1, *input_shape)

    # Output path
    output_path = output_dir / 'model.onnx'

    # Configure dynamic axes
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    # Export
    start_time = datetime.now()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config,
        do_constant_folding=True,
        export_params=True,
    )

    export_time = (datetime.now() - start_time).total_seconds()

    # Get file size
    file_size_bytes = output_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    logger.info(f"[EXPORT] ONNX export completed in {export_time:.2f}s")
    logger.info(f"[EXPORT] Output: {output_path} ({file_size_mb:.2f} MB)")

    return {
        'exported_file': output_path,
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_mb,
        'export_time_seconds': export_time
    }


def export_torchscript(
    model: torch.nn.Module,
    input_shape: tuple,
    export_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Export model to TorchScript format"""
    logger.info("[EXPORT] Exporting to TorchScript format...")

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)

    # Output path
    output_path = output_dir / 'model.torchscript'

    # Export using trace (more compatible with HuggingFace models)
    start_time = datetime.now()

    # Use trace for better compatibility
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(output_path))

    export_time = (datetime.now() - start_time).total_seconds()

    # Get file size
    file_size_bytes = output_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    logger.info(f"[EXPORT] TorchScript export completed in {export_time:.2f}s")
    logger.info(f"[EXPORT] Output: {output_path} ({file_size_mb:.2f} MB)")

    return {
        'exported_file': output_path,
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_mb,
        'export_time_seconds': export_time
    }


def export_model(
    model: torch.nn.Module,
    input_shape: tuple,
    export_format: str,
    export_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Export model to specified format"""
    if export_format == 'onnx':
        return export_onnx(model, input_shape, export_config, output_dir)
    elif export_format == 'torchscript':
        return export_torchscript(model, input_shape, export_config, output_dir)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def generate_metadata(
    model_path: Path,
    model_config,
    processor,
    input_shape: tuple,
    export_format: str,
    export_result: Dict[str, Any],
    export_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate metadata.json for exported model"""
    logger.info("[EXPORT] Generating metadata.json")

    # Get class information from config
    id2label = getattr(model_config, 'id2label', {})
    label2id = getattr(model_config, 'label2id', {})
    num_classes = getattr(model_config, 'num_labels', len(id2label))

    # Get class names
    if id2label:
        class_names = [id2label.get(str(i), id2label.get(i, f'class_{i}'))
                       for i in range(num_classes)]
    else:
        class_names = [f'class_{i}' for i in range(num_classes)]

    # Get model name from config
    model_name = getattr(model_config, '_name_or_path', 'unknown')

    # Get preprocessing info
    preprocessing = {
        'resize': input_shape[1],  # Height
        'normalize': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet defaults
            'std': [0.229, 0.224, 0.225]
        },
        'format': 'RGB',
        'layout': 'NCHW'
    }

    # Override with processor values if available
    if processor is not None:
        if hasattr(processor, 'image_mean'):
            preprocessing['normalize']['mean'] = list(processor.image_mean)
        if hasattr(processor, 'image_std'):
            preprocessing['normalize']['std'] = list(processor.image_std)

    # Use SDK's create_export_metadata for standardized format
    metadata = TrainerSDK.create_export_metadata(
        framework='huggingface',
        model_name=model_name,
        export_format=export_format,
        task_type='image_classification',
        input_shape=list(input_shape),  # [C, H, W]
        output_shape=[[1, num_classes]],
        class_names=class_names,
        preprocessing=preprocessing,
        postprocessing={
            'apply_softmax': True,
            'top_k': 5
        },
        export_config=export_config,
        export_info={
            'file_size_mb': export_result['file_size_mb'],
            'export_time_seconds': export_result['export_time_seconds'],
        }
    )

    logger.info(f"[EXPORT] Metadata generated for {num_classes} classes")
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
        'onnx': ['python'],
        'torchscript': ['python'],
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
        f.write(f"Framework: HuggingFace Transformers\n")
        f.write(f"Task: {metadata.get('task_type', 'image_classification')}\n")
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
    logger.info(f"[EXPORT] Copied file: {exported_file} -> {model_dest}")

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
        logger.info("HUGGINGFACE TRANSFORMERS EXPORT")
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

        # Download checkpoint (HuggingFace saves as directory)
        logger.info(f"[EXPORT] Downloading checkpoint from: {checkpoint_s3_uri}")
        checkpoint_path = workspace / "checkpoint"
        sdk.download_checkpoint(checkpoint_s3_uri, str(checkpoint_path))
        logger.info(f"[EXPORT] Checkpoint downloaded to: {checkpoint_path}")

        # Load model
        model, processor, model_config = load_model(checkpoint_path)

        # Determine input shape
        input_shape = get_input_shape(processor, model_config)
        logger.info(f"[EXPORT] Input shape: {input_shape}")

        # Export model
        export_result = export_model(
            model=model,
            input_shape=input_shape,
            export_format=export_format,
            export_config=export_config,
            output_dir=workspace
        )

        # Generate metadata
        metadata = generate_metadata(
            model_path=checkpoint_path,
            model_config=model_config,
            processor=processor,
            input_shape=input_shape,
            export_format=export_format,
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
        elif 'export' in error_msg.lower() or 'onnx' in error_msg.lower():
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

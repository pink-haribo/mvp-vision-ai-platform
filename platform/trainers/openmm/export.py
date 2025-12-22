#!/usr/bin/env python3
"""
OpenMMLab Model Export Script

Exports MMDetection/MMSegmentation/MMPose models to various formats.

Supported Formats:
- ONNX
- TorchScript
- TensorRT (via ONNX)

Environment Variables:
    CALLBACK_URL: Backend API URL
    JOB_ID: Export job ID
    CHECKPOINT_PATH: S3 URI of trained checkpoint
    EXPORT_FORMAT: onnx, torchscript, tensorrt
    EXPORT_CONFIG: JSON-encoded export configuration
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

from trainer_sdk import TrainerSDK, ErrorType

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def export_to_onnx(model, checkpoint_path: str, output_path: str, config: dict):
    """
    Export MMDetection model to ONNX.

    Args:
        model: MMDetection model
        checkpoint_path: Path to checkpoint
        output_path: Output ONNX file path
        config: Export configuration
    """
    import torch
    from mmdet.apis import init_detector

    # Initialize model with checkpoint
    model = init_detector(config.get('config_file', 'default.py'), checkpoint_path)
    model.eval()

    # Get input shape
    input_shape = config.get('input_shape', [1, 3, 640, 640])
    dummy_input = torch.randn(*input_shape).cuda()

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=config.get('opset_version', 17),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )
    logger.info(f"Model exported to ONNX: {output_path}")


def export_to_torchscript(model, checkpoint_path: str, output_path: str, config: dict):
    """
    Export MMDetection model to TorchScript.

    Args:
        model: MMDetection model
        checkpoint_path: Path to checkpoint
        output_path: Output TorchScript file path
        config: Export configuration
    """
    import torch
    from mmdet.apis import init_detector

    model = init_detector(config.get('config_file', 'default.py'), checkpoint_path)
    model.eval()

    # Get input shape
    input_shape = config.get('input_shape', [1, 3, 640, 640])
    dummy_input = torch.randn(*input_shape).cuda()

    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    logger.info(f"Model exported to TorchScript: {output_path}")


def main():
    """Main export function."""
    sdk = TrainerSDK()

    try:
        sdk.report_started('export')
        logger.info(f"Export job {sdk.job_id} started")

        # Get export format
        export_format = os.getenv('EXPORT_FORMAT', 'onnx')
        logger.info(f"Export format: {export_format}")

        # Download checkpoint
        checkpoint_s3_uri = os.getenv('CHECKPOINT_PATH')
        if not checkpoint_s3_uri:
            raise ValueError("CHECKPOINT_PATH environment variable required")

        checkpoint_path = "/tmp/model.pth"
        sdk.download_checkpoint(checkpoint_s3_uri, checkpoint_path)
        logger.info(f"Checkpoint downloaded: {checkpoint_path}")

        # Load export config
        export_config_str = os.getenv('EXPORT_CONFIG', '{}')
        export_config = json.loads(export_config_str)

        # Export model
        output_path = f"/tmp/model.{export_format}"
        logger.info(f"Exporting to {export_format}...")

        if export_format == 'onnx':
            export_to_onnx(None, checkpoint_path, output_path, export_config)
        elif export_format == 'torchscript':
            export_to_torchscript(None, checkpoint_path, output_path, export_config)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        # Upload exported model
        logger.info("Uploading exported model...")
        s3_key = f"exports/{sdk.job_id}/model.{export_format}"
        output_s3_uri = sdk.upload_file(
            local_path=output_path,
            s3_key=s3_key,
            storage_type='internal'
        )

        # Get file size
        file_size = Path(output_path).stat().st_size

        # Create metadata
        metadata = sdk.create_export_metadata(
            framework='openmm',
            model_name=sdk.model_name,
            export_format=export_format,
            task_type=sdk.task_type,
            input_shape=export_config.get('input_shape', [640, 640, 3]),
            output_shape=[[1, 84, 8400]],  # TODO: Dynamic based on model
            export_config=export_config
        )

        # Report completion
        sdk.report_export_completed(
            export_format=export_format,
            output_s3_uri=output_s3_uri,
            file_size_bytes=file_size,
            metadata=metadata
        )
        logger.info("Export completed successfully")

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Export failed: {error_msg}")
        logger.error(error_trace)

        error_type = ErrorType.FRAMEWORK_ERROR
        if "checkpoint" in error_msg.lower():
            error_type = ErrorType.CHECKPOINT_ERROR

        sdk.report_failed(
            error_type=error_type,
            message=error_msg,
            traceback=error_trace
        )
        sys.exit(1)

    finally:
        sdk.close()


if __name__ == "__main__":
    main()

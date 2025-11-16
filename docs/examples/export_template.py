#!/usr/bin/env python3
"""
Export Script Template for Vision AI Training Platform

This template follows the Export Convention v1.0 for multi-framework trainers.

USAGE:
1. Copy this file to your trainer directory:
   cp docs/examples/export_template.py platform/trainers/your_framework/export.py

2. Modify the framework-specific functions:
   - load_model()
   - get_metadata()
   - export_onnx() and other export_*() functions

3. Test your implementation:
   python export.py --checkpoint_path ... --export_format onnx --output_dir ...

CONVENTION COMPLIANCE:
✅ CLI Interface: Follows standard argparse with required arguments
✅ Output Files: Produces model.{format} and metadata.json
✅ Exit Codes: 0=success, 1=export failed, 2=validation failed, 3=config error
✅ Metadata Schema: Follows standard schema with required fields
✅ Logging: Uses standard logging format for Backend monitoring

See docs/EXPORT_CONVENTION.md for full specification.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging (DO NOT MODIFY - Backend captures this)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLI ARGUMENT PARSING (DO NOT MODIFY)
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments following Export Convention v1.0.

    DO NOT MODIFY this function. Backend's export_subprocess.py expects
    these exact argument names and formats.
    """
    parser = argparse.ArgumentParser(
        description='Export trained model to production format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--export_format',
        type=str,
        required=True,
        choices=['onnx', 'tensorrt', 'coreml', 'tflite', 'torchscript', 'openvino'],
        help='Export format'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for exported files'
    )

    parser.add_argument(
        '--export_config',
        type=str,
        default='{}',
        help='JSON string with format-specific configuration (optional)'
    )

    return parser.parse_args()


# ============================================================================
# FRAMEWORK-SPECIFIC FUNCTIONS (MODIFY THESE)
# ============================================================================

def load_model(checkpoint_path: str) -> Any:
    """
    Load model from checkpoint.

    MODIFY THIS: Implement your framework's model loading logic.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded model object (framework-specific)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted or incompatible

    Example for Ultralytics:
        from ultralytics import YOLO
        return YOLO(checkpoint_path)

    Example for timm:
        import torch
        import timm
        checkpoint = torch.load(checkpoint_path)
        model = timm.create_model(checkpoint['model_name'], num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    Example for HuggingFace:
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained(checkpoint_path)
    """
    raise NotImplementedError(
        "You must implement load_model() for your framework.\n"
        "See docs/EXPORT_CONVENTION.md for examples."
    )


def get_metadata(model: Any, export_format: str, export_config: Dict) -> Dict:
    """
    Extract metadata from model.

    MODIFY THIS: Extract framework-specific metadata.

    Args:
        model: Loaded model object
        export_format: Target export format
        export_config: Export configuration dict

    Returns:
        dict: Metadata following standard schema

    Required fields:
        - framework: str (e.g., 'ultralytics', 'timm', 'huggingface')
        - model_name: str (e.g., 'yolo11n', 'resnet50', 'vit-base-patch16-224')
        - export_format: str
        - task_type: str ('detection', 'classification', 'segmentation', 'pose')
        - input_shape: list[int] (e.g., [640, 640, 3] or [3, 224, 224])
        - input_dtype: str (e.g., 'float32', 'uint8')
        - output_shape: list[list[int]] (e.g., [[1, 1000]] for classification)
        - created_at: str (ISO 8601 timestamp)

    Optional but recommended:
        - class_names: list[str] (for classification/detection)
        - num_classes: int
        - preprocessing: dict (mean, std, color_space)
        - postprocessing: dict (nms, threshold)

    Example for Detection Model:
        return {
            'framework': 'ultralytics',
            'model_name': 'yolo11n',
            'export_format': export_format,
            'task_type': 'detection',
            'input_shape': [640, 640, 3],
            'input_dtype': 'float32',
            'output_shape': [[1, 84, 8400]],
            'class_names': model.names,
            'num_classes': len(model.names),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'export_config': export_config,
            'preprocessing': {
                'mean': [0.0, 0.0, 0.0],
                'std': [255.0, 255.0, 255.0],
                'color_space': 'RGB',
                'resize_mode': 'letterbox'
            },
            'postprocessing': {
                'apply_nms': True,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            }
        }

    Example for Classification Model:
        return {
            'framework': 'timm',
            'model_name': 'resnet50',
            'export_format': export_format,
            'task_type': 'classification',
            'input_shape': [3, 224, 224],
            'input_dtype': 'float32',
            'output_shape': [[1, 1000]],
            'class_names': imagenet_classes,
            'num_classes': 1000,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'export_config': export_config,
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'color_space': 'RGB',
                'resize_mode': 'center_crop'
            }
        }
    """
    raise NotImplementedError(
        "You must implement get_metadata() for your framework.\n"
        "See docs/EXPORT_CONVENTION.md for schema requirements."
    )


# ============================================================================
# FORMAT-SPECIFIC EXPORT FUNCTIONS (MODIFY THESE)
# ============================================================================

def export_onnx(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to ONNX format.

    MODIFY THIS: Call your framework's ONNX export method.

    Args:
        model: Loaded model object
        config: Export configuration dict (opset_version, dynamic_axes, simplify)
        output_dir: Output directory path

    Returns:
        str: Path to exported ONNX file

    Raises:
        RuntimeError: If ONNX export fails
        NotImplementedError: If framework doesn't support ONNX

    Example for Ultralytics:
        output_path = model.export(
            format='onnx',
            opset=config.get('opset_version', 17),
            dynamic=config.get('dynamic_axes', False),
            simplify=config.get('simplify', True)
        )
        return str(output_path)

    Example for PyTorch-based frameworks:
        import torch
        output_path = output_dir / 'model.onnx'
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=config.get('opset_version', 17),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
                if config.get('dynamic_axes', False) else None
        )
        return str(output_path)
    """
    raise NotImplementedError(
        "ONNX export not implemented for this framework.\n"
        "Implement export_onnx() or raise NotImplementedError if unsupported."
    )


def export_tensorrt(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to TensorRT format.

    MODIFY THIS or raise NotImplementedError if not supported.

    Args:
        model: Loaded model object
        config: Export configuration dict (fp16, int8, max_batch_size, workspace_size)
        output_dir: Output directory path

    Returns:
        str: Path to exported TensorRT engine file

    Raises:
        RuntimeError: If TensorRT export fails
        NotImplementedError: If framework doesn't support TensorRT

    Example for Ultralytics:
        output_path = model.export(
            format='engine',
            half=config.get('fp16', False),
            int8=config.get('int8', False),
            workspace=config.get('workspace_size', 4)
        )
        return str(output_path)
    """
    raise NotImplementedError(
        "TensorRT export not available for this framework.\n"
        "Please use ONNX format and convert to TensorRT externally."
    )


def export_coreml(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to CoreML format.

    MODIFY THIS or raise NotImplementedError if not supported.

    Args:
        model: Loaded model object
        config: Export configuration dict (minimum_deployment_target, compute_units)
        output_dir: Output directory path

    Returns:
        str: Path to exported CoreML model (.mlpackage directory)

    Raises:
        RuntimeError: If CoreML export fails
        NotImplementedError: If framework doesn't support CoreML

    Example for Ultralytics:
        output_path = model.export(
            format='coreml',
            nms=True
        )
        return str(output_path)
    """
    raise NotImplementedError(
        "CoreML export not available for this framework."
    )


def export_tflite(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to TFLite format.

    MODIFY THIS or raise NotImplementedError if not supported.

    Args:
        model: Loaded model object
        config: Export configuration dict (quantize, int8)
        output_dir: Output directory path

    Returns:
        str: Path to exported TFLite file

    Raises:
        RuntimeError: If TFLite export fails
        NotImplementedError: If framework doesn't support TFLite

    Example for Ultralytics:
        output_path = model.export(
            format='tflite',
            int8=config.get('int8', False)
        )
        return str(output_path)
    """
    raise NotImplementedError(
        "TFLite export not available for this framework."
    )


def export_torchscript(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to TorchScript format.

    MODIFY THIS or raise NotImplementedError if not supported.

    Args:
        model: Loaded model object
        config: Export configuration dict
        output_dir: Output directory path

    Returns:
        str: Path to exported TorchScript file

    Raises:
        RuntimeError: If TorchScript export fails
        NotImplementedError: If framework doesn't support TorchScript

    Example for PyTorch models:
        import torch
        output_path = output_dir / 'model.torchscript'
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(output_path))
        return str(output_path)
    """
    raise NotImplementedError(
        "TorchScript export not available for this framework."
    )


def export_openvino(model: Any, config: Dict, output_dir: Path) -> str:
    """
    Export model to OpenVINO format.

    MODIFY THIS or raise NotImplementedError if not supported.

    Args:
        model: Loaded model object
        config: Export configuration dict
        output_dir: Output directory path

    Returns:
        str: Path to exported OpenVINO IR file (.xml)

    Raises:
        RuntimeError: If OpenVINO export fails
        NotImplementedError: If framework doesn't support OpenVINO

    Example for Ultralytics:
        output_path = model.export(format='openvino')
        return str(output_path)
    """
    raise NotImplementedError(
        "OpenVINO export not available for this framework."
    )


# ============================================================================
# VALIDATION (OPTIONAL BUT RECOMMENDED)
# ============================================================================

def validate_export(model_path: str, original_model: Any, metadata: Dict) -> bool:
    """
    Validate exported model.

    OPTIONAL: Implement validation logic to catch export issues early.

    Args:
        model_path: Path to exported model file
        original_model: Original loaded model
        metadata: Metadata dict

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails

    Example for ONNX:
        import onnxruntime as ort
        import numpy as np

        # Load exported model
        session = ort.InferenceSession(model_path)

        # Run dummy inference
        input_shape = metadata['input_shape']
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input})

        # Check output shape matches metadata
        assert len(output) == len(metadata['output_shape'])
        logger.info("ONNX validation passed")
        return True
    """
    logger.warning("Validation not implemented - skipping")
    return True


# ============================================================================
# MAIN EXPORT WORKFLOW (DO NOT MODIFY)
# ============================================================================

def main():
    """
    Main export workflow.

    DO NOT MODIFY this function. It follows Export Convention v1.0.
    Backend's export_subprocess.py expects this exact workflow.
    """
    try:
        # 1. Parse arguments
        args = parse_args()
        logger.info(f"Starting export to {args.export_format} format")
        logger.info(f"Checkpoint: {args.checkpoint_path}")
        logger.info(f"Output directory: {args.output_dir}")

        # Parse export config
        try:
            export_config = json.loads(args.export_config)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid export_config JSON: {e}")
            sys.exit(3)  # Configuration error

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load model
        logger.info("Loading model from checkpoint...")
        try:
            model = load_model(args.checkpoint_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)  # Export failed

        # 3. Export to target format
        logger.info(f"Exporting to {args.export_format} format...")
        export_functions = {
            'onnx': export_onnx,
            'tensorrt': export_tensorrt,
            'coreml': export_coreml,
            'tflite': export_tflite,
            'torchscript': export_torchscript,
            'openvino': export_openvino
        }

        try:
            export_func = export_functions[args.export_format]
            model_path = export_func(model, export_config, output_dir)
            logger.info(f"Export completed: {model_path}")
        except NotImplementedError as e:
            logger.error(f"Format not supported: {e}")
            sys.exit(3)  # Configuration error
        except Exception as e:
            logger.error(f"Export failed: {e}")
            sys.exit(1)  # Export failed

        # 4. Generate metadata
        logger.info("Generating metadata...")
        try:
            metadata = get_metadata(model, args.export_format, export_config)

            # Validate required fields
            required_fields = [
                'framework', 'model_name', 'export_format', 'task_type',
                'input_shape', 'input_dtype', 'output_shape', 'created_at'
            ]
            missing_fields = [f for f in required_fields if f not in metadata]
            if missing_fields:
                raise ValueError(f"Metadata missing required fields: {missing_fields}")

            # Save metadata
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to generate metadata: {e}")
            sys.exit(1)  # Export failed

        # 5. Validate export (optional)
        logger.info("Validating exported model...")
        try:
            validate_export(model_path, model, metadata)
            logger.info("Validation passed")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            sys.exit(2)  # Validation failed

        # 6. Success
        logger.info("Export completed successfully!")
        logger.info(f"Model: {model_path}")
        logger.info(f"Metadata: {metadata_path}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

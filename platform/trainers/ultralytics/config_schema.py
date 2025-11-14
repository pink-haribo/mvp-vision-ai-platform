"""
Ultralytics YOLO Configuration Schema

This module defines the configuration schema for YOLO models.
Used by Frontend to dynamically generate configuration UI.
"""

from typing import Dict, Any, List


def get_config_schema() -> Dict[str, Any]:
    """
    Return configuration schema for Ultralytics YOLO models.

    Returns:
        Dict with framework, description, version, fields, and presets
    """
    fields = [
        # ========== Optimizer Settings ==========
        {
            "name": "optimizer_type",
            "type": "select",
            "default": "Adam",
            "options": ["Adam", "AdamW", "SGD", "RMSprop"],
            "description": "Optimizer algorithm",
            "group": "optimizer",
            "required": False,
            "advanced": False
        },
        {
            "name": "weight_decay",
            "type": "float",
            "default": 0.0005,
            "min": 0.0,
            "max": 0.01,
            "step": 0.0001,
            "description": "Weight decay (L2 regularization)",
            "group": "optimizer",
            "required": False,
            "advanced": True
        },
        {
            "name": "momentum",
            "type": "float",
            "default": 0.937,
            "min": 0.0,
            "max": 1.0,
            "step": 0.001,
            "description": "Momentum for SGD",
            "group": "optimizer",
            "required": False,
            "advanced": True
        },

        # ========== Scheduler Settings ==========
        {
            "name": "cos_lr",
            "type": "bool",
            "default": True,
            "description": "Use cosine learning rate scheduler",
            "group": "scheduler",
            "required": False,
            "advanced": False
        },
        {
            "name": "lrf",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "description": "Final learning rate (lr0 * lrf)",
            "group": "scheduler",
            "required": False,
            "advanced": False
        },
        {
            "name": "warmup_epochs",
            "type": "int",
            "default": 3,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Number of warmup epochs",
            "group": "scheduler",
            "required": False,
            "advanced": False
        },
        {
            "name": "warmup_momentum",
            "type": "float",
            "default": 0.8,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "Initial warmup momentum",
            "group": "scheduler",
            "required": False,
            "advanced": True
        },
        {
            "name": "warmup_bias_lr",
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "description": "Warmup initial bias learning rate",
            "group": "scheduler",
            "required": False,
            "advanced": True
        },

        # ========== Augmentation Settings (YOLO-specific) ==========
        {
            "name": "mosaic",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "Mosaic augmentation probability (4-image blend)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "mixup",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Mixup augmentation probability",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "copy_paste",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Copy-Paste augmentation probability",
            "group": "augmentation",
            "required": False,
            "advanced": True
        },
        {
            "name": "degrees",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 180.0,
            "step": 5.0,
            "description": "Rotation degrees (+/- deg)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "translate",
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Translation (+/- fraction)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "scale",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "description": "Scaling (+/- gain)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "shear",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 45.0,
            "step": 5.0,
            "description": "Shear degrees (+/- deg)",
            "group": "augmentation",
            "required": False,
            "advanced": True
        },
        {
            "name": "perspective",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 0.001,
            "step": 0.0001,
            "description": "Perspective distortion",
            "group": "augmentation",
            "required": False,
            "advanced": True
        },
        {
            "name": "flipud",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "Vertical flip probability",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "fliplr",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "Horizontal flip probability",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "hsv_h",
            "type": "float",
            "default": 0.015,
            "min": 0.0,
            "max": 1.0,
            "step": 0.005,
            "description": "HSV-Hue augmentation (fraction)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "hsv_s",
            "type": "float",
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "HSV-Saturation augmentation (fraction)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },
        {
            "name": "hsv_v",
            "type": "float",
            "default": 0.4,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "HSV-Value augmentation (fraction)",
            "group": "augmentation",
            "required": False,
            "advanced": False
        },

        # ========== Optimization Settings ==========
        {
            "name": "amp",
            "type": "bool",
            "default": True,
            "description": "Automatic Mixed Precision training",
            "group": "optimization",
            "required": False,
            "advanced": False
        },
        {
            "name": "close_mosaic",
            "type": "int",
            "default": 10,
            "min": 0,
            "max": 50,
            "step": 1,
            "description": "Disable mosaic augmentation for final N epochs",
            "group": "optimization",
            "required": False,
            "advanced": True
        },

        # ========== Validation Settings ==========
        {
            "name": "val_interval",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Validate every N epochs",
            "group": "validation",
            "required": False,
            "advanced": False
        },
    ]

    presets = {
        "easy": {
            "mosaic": 1.0,
            "fliplr": 0.5,
            "amp": True,
        },
        "medium": {
            "mosaic": 1.0,
            "mixup": 0.1,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10,
            "translate": 0.1,
            "scale": 0.5,
            "amp": True,
        },
        "advanced": {
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "fliplr": 0.5,
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "degrees": 15,
            "translate": 0.2,
            "scale": 0.9,
            "shear": 5.0,
            "perspective": 0.0005,
            "amp": True,
            "close_mosaic": 15,
        }
    }

    return {
        "framework": "ultralytics",
        "description": "Ultralytics YOLO Training Configuration",
        "version": "1.0",
        "fields": fields,
        "presets": presets
    }


if __name__ == "__main__":
    """Test schema generation"""
    import json
    schema = get_config_schema()
    print(json.dumps(schema, indent=2))
    print(f"\nTotal fields: {len(schema['fields'])}")
    print(f"Presets: {list(schema['presets'].keys())}")

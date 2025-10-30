"""
HuggingFace Transformers Model Registry

Supports:
- ViT (Vision Transformer) - Image Classification
- D-FINE - Object Detection
- EoMT (Encoder-only Mask Transformer) - Semantic Segmentation
- Swin2SR - Super-Resolution
"""

from typing import Dict, Any

HUGGINGFACE_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ========================================
    # Image Classification
    # ========================================

    "google/vit-base-patch16-224": {
        "display_name": "Vision Transformer (ViT) Base",
        "description": "Transformer-based image classification - Attention-based global context (86M params)",
        "framework": "huggingface",
        "task_type": "image_classification",
        "model_id": "google/vit-base-patch16-224",
        "params": "86M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 3e-4,
        "tags": ["p1", "transformer", "attention", "imagenet", "2021", "classification"],
        "priority": 1,

        "architecture": {
            "type": "Vision Transformer",
            "patch_size": 16,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
        },

        "performance": {
            "imagenet_top1": "81.3%",
            "imagenet_top5": "96.5%",
            "inference_speed": "~50 images/sec (V100)",
        },

        "training_tips": {
            "optimizer": "AdamW",
            "lr": "3e-4",
            "warmup": "10k steps",
            "batch_size": "32-64 (depends on GPU)",
            "epochs": "30-100 (fine-tuning)",
        },

        "use_cases": [
            {
                "title": "Fine-grained Image Classification",
                "description": "Classify images with global context understanding using attention mechanisms",
                "industry": "E-commerce, Medical Imaging, Agriculture",
                "dataset": "Custom ImageNet-style dataset",
                "metrics": {
                    "before": "ResNet-50: 76.1% accuracy",
                    "after": "ViT-Base: 81.3% accuracy with attention visualization"
                }
            },
            {
                "title": "Product Categorization",
                "description": "Automatic product category assignment for e-commerce",
                "industry": "E-commerce",
                "dataset": "Product images with categories",
                "metrics": {
                    "before": "Manual tagging: 1,000 products/day",
                    "after": "Automated: 100,000 products/day with 95% accuracy"
                }
            }
        ]
    },

    # ========================================
    # Object Detection
    # ========================================

    "ustc-community/dfine-x-coco": {
        "display_name": "D-FINE (Detection Fine-grained)",
        "description": "SOTA real-time detector - Fine-grained bbox refinement (57.1% AP on COCO)",
        "framework": "huggingface",
        "task_type": "object_detection",
        "model_id": "ustc-community/dfine-x-coco",
        "params": "67M",
        "input_size": 640,
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 1e-4,
        "tags": ["p1", "detection", "detr", "real-time", "sota", "2024"],
        "priority": 1,

        "architecture": {
            "type": "DETR-based",
            "backbone": "ResNet-50",
            "features": ["FDR (Fine-grained Distribution Refinement)", "GO-LSD (Global Optimal Localization Self-Distillation)"],
            "innovation": "Redefines bbox regression as distribution refinement",
        },

        "performance": {
            "coco_map50": "57.1%",
            "coco_map50-95": "40.8%",
            "inference_speed": "Real-time (>30 FPS on V100)",
            "vs_yolov8": "+7% mAP50 improvement",
        },

        "training_tips": {
            "optimizer": "AdamW",
            "lr": "1e-4",
            "warmup": "1k steps",
            "batch_size": "8-16 (GPU memory intensive)",
            "epochs": "50-300 (COCO scale)",
            "dataset_format": "COCO JSON annotations required",
        },

        "use_cases": [
            {
                "title": "Precision Object Localization",
                "description": "High-precision bounding box detection for industrial quality inspection",
                "industry": "Manufacturing, Quality Control",
                "dataset": "Custom COCO-format defect dataset",
                "metrics": {
                    "before": "YOLOv8: 50.2% mAP50, occasional missed defects",
                    "after": "D-FINE: 57.1% mAP50 with fine-grained localization, 0.1% false negative rate"
                }
            },
            {
                "title": "Autonomous Driving Object Detection",
                "description": "Real-time detection of vehicles, pedestrians, traffic signs",
                "industry": "Autonomous Vehicles",
                "dataset": "Road scene dataset (COCO format)",
                "metrics": {
                    "before": "DETR: 45% mAP50, 15 FPS",
                    "after": "D-FINE: 57% mAP50, 35 FPS (real-time capable)"
                }
            }
        ]
    },

    # ========================================
    # Semantic Segmentation
    # ========================================

    "tue-mps/eomt-vit-large": {
        "display_name": "EoMT (Encoder-only Mask Transformer)",
        "description": "CVPR 2025 Highlight - Segmentation without task-specific components (4x faster)",
        "framework": "huggingface",
        "task_type": "semantic_segmentation",
        "model_id": "tue-mps/eomt-vit-large",
        "params": "304M",
        "input_size": 518,
        "pretrained_available": True,
        "recommended_batch_size": 4,
        "recommended_lr": 1e-4,
        "tags": ["p1", "segmentation", "vit", "encoder-only", "cvpr2025", "fast"],
        "priority": 1,

        "architecture": {
            "type": "Encoder-only ViT",
            "backbone": "ViT-Large",
            "innovation": "No task-specific decoder - learns segmentation with pure ViT",
            "key_insight": "Inductive biases can be learned through large-scale pretraining",
        },

        "performance": {
            "ade20k_miou": "53.0%",
            "cityscapes_miou": "82.5%",
            "inference_speed": "4x faster than Mask2Former (0.6s vs 2.5s per image)",
            "model_simplicity": "No complex decoder architecture",
        },

        "training_tips": {
            "optimizer": "AdamW",
            "lr": "1e-4",
            "warmup": "1.5k steps",
            "batch_size": "4-8 (large model)",
            "epochs": "100-160 (ADE20K scale)",
            "dataset_format": "Segmentation masks (PNG) required",
            "mixed_precision": "Recommended (fp16)",
        },

        "use_cases": [
            {
                "title": "Fast Semantic Segmentation for Autonomous Driving",
                "description": "Real-time pixel-wise scene understanding for road scenes",
                "industry": "Autonomous Vehicles",
                "dataset": "Cityscapes or custom road dataset",
                "metrics": {
                    "before": "Mask2Former: 50.1% mIoU, 2.5s/image (too slow for real-time)",
                    "after": "EoMT: 53.0% mIoU, 0.6s/image (4x faster, real-time capable)"
                }
            },
            {
                "title": "Medical Image Segmentation",
                "description": "Segment organs, tumors, lesions in medical scans",
                "industry": "Healthcare, Medical Imaging",
                "dataset": "Custom medical image dataset with masks",
                "metrics": {
                    "before": "U-Net: 78% Dice, requires task-specific architecture",
                    "after": "EoMT: 82% Dice with simpler architecture and faster inference"
                }
            }
        ]
    },

    # ========================================
    # Super-Resolution
    # ========================================

    "caidas/swin2SR-classical-sr-x2-64": {
        "display_name": "Swin2SR (2x Super-Resolution)",
        "description": "Image restoration and super-resolution - 2x upscaling with artifact removal",
        "framework": "huggingface",
        "task_type": "super_resolution",
        "model_id": "caidas/swin2SR-classical-sr-x2-64",
        "params": "11.9M",
        "input_size": 512,
        "upscale_factor": 2,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 2e-4,
        "tags": ["p1", "super-resolution", "restoration", "swin", "2022", "image-quality"],
        "priority": 1,

        "architecture": {
            "type": "Swin Transformer V2",
            "window_size": 8,
            "features": ["Residual Swin Transformer Blocks", "Continuous position bias"],
            "innovation": "Training stability improvements from SwinV2",
        },

        "performance": {
            "psnr": "33.89 dB (Set5 dataset)",
            "ssim": "0.9195",
            "inference_speed": "~20 images/sec (512x512 → 1024x1024 on V100)",
        },

        "training_tips": {
            "optimizer": "Adam",
            "lr": "2e-4",
            "warmup": "Not typically used",
            "batch_size": "16-32",
            "epochs": "500-1000 (SR requires many iterations)",
            "loss": "L1 pixel loss",
            "dataset_format": "HR-LR paired images (DIV2K style)",
        },

        "use_cases": [
            {
                "title": "Medical Image Quality Enhancement",
                "description": "Upscale low-resolution medical images for better diagnosis",
                "industry": "Healthcare, Radiology",
                "dataset": "HR-LR paired medical images",
                "metrics": {
                    "before": "Bicubic interpolation: 30.1 dB PSNR, blurry details",
                    "after": "Swin2SR: 33.9 dB PSNR with sharp details and artifact removal"
                }
            },
            {
                "title": "Surveillance Video Enhancement",
                "description": "Enhance low-resolution surveillance footage for better identification",
                "industry": "Security, Law Enforcement",
                "dataset": "Surveillance image pairs",
                "metrics": {
                    "before": "480p footage: Hard to identify faces/plates",
                    "after": "Upscaled to 960p: Clear facial features, readable license plates"
                }
            },
            {
                "title": "E-commerce Product Image Enhancement",
                "description": "Upscale low-quality product images from suppliers",
                "industry": "E-commerce, Retail",
                "dataset": "Product image pairs (LR from suppliers, HR reference)",
                "metrics": {
                    "before": "Low-res images: 25% conversion rate",
                    "after": "Enhanced images: 40% conversion rate (+60% improvement)"
                }
            }
        ]
    },

    # ========================================
    # Additional Models (x4 upscaling variant)
    # ========================================

    "caidas/swin2SR-classical-sr-x4-64": {
        "display_name": "Swin2SR (4x Super-Resolution)",
        "description": "Image restoration and super-resolution - 4x upscaling with artifact removal",
        "framework": "huggingface",
        "task_type": "super_resolution",
        "model_id": "caidas/swin2SR-classical-sr-x4-64",
        "params": "11.9M",
        "input_size": 256,
        "upscale_factor": 4,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 2e-4,
        "tags": ["p1", "super-resolution", "restoration", "swin", "2022", "4x"],
        "priority": 1,

        "architecture": {
            "type": "Swin Transformer V2",
            "window_size": 8,
            "features": ["Residual Swin Transformer Blocks", "Continuous position bias"],
            "innovation": "Training stability improvements from SwinV2",
        },

        "performance": {
            "psnr": "30.77 dB (Set5 dataset, 4x upscaling)",
            "ssim": "0.8812",
            "inference_speed": "~15 images/sec (256x256 → 1024x1024 on V100)",
        },

        "training_tips": {
            "optimizer": "Adam",
            "lr": "2e-4",
            "warmup": "Not typically used",
            "batch_size": "16-32",
            "epochs": "500-1000",
            "loss": "L1 pixel loss",
            "dataset_format": "HR-LR paired images (DIV2K style)",
        },

        "use_cases": [
            {
                "title": "Legacy Document Digitization",
                "description": "Restore and upscale scanned documents from low-resolution sources",
                "industry": "Archives, Libraries",
                "dataset": "Historical document pairs",
                "metrics": {
                    "before": "Original scans: 300 DPI, hard to read",
                    "after": "Enhanced to 1200 DPI equivalent with OCR accuracy +25%"
                }
            }
        ]
    },
}


# Helper functions

def get_huggingface_model(model_name: str) -> Dict[str, Any]:
    """Get model info by name."""
    return HUGGINGFACE_MODEL_REGISTRY.get(model_name, {})


def list_huggingface_models(
    task_type: str = None,
    priority: int = None,
    tags: list = None
) -> Dict[str, Dict[str, Any]]:
    """
    List models with optional filtering.

    Args:
        task_type: Filter by task type (e.g., 'image_classification')
        priority: Filter by priority level (0, 1, 2)
        tags: Filter by tags (e.g., ['transformer', 'sota'])

    Returns:
        Filtered dictionary of models
    """
    filtered = {}

    for model_name, model_info in HUGGINGFACE_MODEL_REGISTRY.items():
        # Task type filter
        if task_type and model_info.get('task_type') != task_type:
            continue

        # Priority filter
        if priority is not None and model_info.get('priority') != priority:
            continue

        # Tags filter
        if tags:
            model_tags = model_info.get('tags', [])
            if not any(tag in model_tags for tag in tags):
                continue

        filtered[model_name] = model_info

    return filtered


def get_huggingface_models_by_task(task_type: str) -> Dict[str, Dict[str, Any]]:
    """Get all models for a specific task type."""
    return list_huggingface_models(task_type=task_type)


# Export
__all__ = [
    'HUGGINGFACE_MODEL_REGISTRY',
    'get_huggingface_model',
    'list_huggingface_models',
    'get_huggingface_models_by_task',
]

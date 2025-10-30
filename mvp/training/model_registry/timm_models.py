"""timm (PyTorch Image Models) model registry."""

from typing import Dict, Any, List, Optional

# P0: Priority 0 models (Quick Win validation)
# P1: Priority 1 models (Core expansion)
# P2: Priority 2 models (Full coverage)

TIMM_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ============================================================
    # P0: Quick Win (2 models)
    # ============================================================

    "resnet50": {
        "display_name": "ResNet-50",
        "description": "Most popular baseline CNN - Industry standard for benchmarking",
        "params": "25.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.001,
        "tags": ["p0", "baseline", "classic", "standard", "popular"],
        "priority": 0,
        "task_type": "image_classification",

        # Benchmark performance
        "benchmark": {
            "imagenet_top1": 80.4,
            "imagenet_top5": 95.1,
            "inference_speed_v100": 140,  # images/second
            "inference_speed_unit": "img/s",
            "training_time_epoch": "~2 hours",
            "training_hardware": "ImageNet, 8x V100",
            "flops": "4.1G",
        },

        # Use cases
        "use_cases": [
            "Baseline comparison for benchmarking",
            "Transfer learning starting point",
            "Educational and research purposes",
            "Production-ready classification",
            "Feature extraction backbone"
        ],

        # Pros and cons
        "pros": [
            "Well-documented and extensively tested",
            "Excellent transfer learning performance",
            "Balanced accuracy and speed trade-off",
            "Widely supported across frameworks",
            "Large community and resources"
        ],

        "cons": [
            "Not the most parameter-efficient model",
            "Larger than modern mobile-optimized models",
            "Lower accuracy than Vision Transformers",
            "Fixed architecture, less flexible"
        ],

        # When to use
        "when_to_use": "Use ResNet-50 when you need a reliable, well-understood baseline for comparison or as a starting point for transfer learning. Ideal for educational purposes and production systems where stability is critical.",

        "when_not_to_use": "Avoid if you need maximum efficiency (use EfficientNet), highest accuracy (use ViT), or deployment on resource-constrained devices (use MobileNet).",

        # Alternatives
        "alternatives": [
            {
                "model": "efficientnetv2_s",
                "reason": "More efficient, higher accuracy, faster training"
            },
            {
                "model": "vit_base_patch16_224",
                "reason": "Higher accuracy with transformer architecture"
            },
            {
                "model": "resnet18",
                "reason": "Lighter version, faster inference"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 32,
                "range": [16, 64],
                "note": "Adjust based on GPU memory"
            },
            "learning_rate": {
                "value": 0.001,
                "range": [0.0001, 0.01],
                "note": "Use 0.1 for training from scratch, 0.001 for fine-tuning"
            },
            "epochs": {
                "value": 50,
                "range": [20, 100],
                "note": "More epochs for training from scratch"
            },
            "optimizer": "Adam or AdamW",
            "scheduler": "Cosine annealing with warmup",
            "weight_decay": 0.0001,
            "image_size": 224,
            "augmentation": "Standard (RandomResizedCrop, RandomHorizontalFlip, ColorJitter)"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Medical Image Classification",
                "description": "Stanford University used ResNet-50 for pneumonia detection from chest X-rays",
                "metrics": {
                    "after": "93% accuracy, comparable to radiologists"
                },
                "link": "https://stanfordmlgroup.github.io/projects/chexnet/"
            },
            {
                "title": "E-commerce Product Categorization",
                "description": "Major retailers use ResNet-50 for automatic product tagging and categorization",
                "metrics": {
                    "before": "Manual tagging: 100 products/hour",
                    "after": "Automated: 10,000 products/hour with 95% accuracy"
                }
            }
        ],

        # Training modes
        "training_modes": {
            "supported": ["full_training", "fine_tuning"],
            "recommended": "fine_tuning",
            "default": "fine_tuning",
            "notes": "Fine-tuning is highly recommended for most use cases. Full training only for large-scale datasets (100K+ images).",
            "requirements": {
                "full_training": {
                    "min_dataset_size": 100000,
                    "min_images": "100K+",
                    "gpu_recommendation": "1x A100 or 4x V100",
                    "estimated_time": "3-7 days",
                    "notes": "Only recommended for large-scale datasets. Requires significant computational resources."
                },
                "fine_tuning": {
                    "min_dataset_size": 1000,
                    "min_images": "1K-100K",
                    "gpu_recommendation": "1x V100 or RTX 3090",
                    "estimated_time": "2-24 hours",
                    "notes": "Works well with limited data. Most cost-effective approach."
                }
            }
        }
    },

    "tf_efficientnetv2_s.in1k": {
        "display_name": "EfficientNetV2-Small",
        "description": "Modern efficient CNN - Best accuracy/speed trade-off with fast training",
        "params": "21.5M",
        "input_size": 384,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p0", "modern", "efficient", "balanced", "2021", "fast-training"],
        "priority": 0,
        "task_type": "image_classification",

        # Benchmark performance
        "benchmark": {
            "imagenet_top1": 84.3,
            "imagenet_top5": 97.0,
            "inference_speed_v100": 200,
            "inference_speed_unit": "img/s",
            "training_time_epoch": "~1.5 hours",
            "training_hardware": "ImageNet, 8x V100",
            "training_speedup": "11x faster than EfficientNet-B7",
            "flops": "8.4G",
        },

        # Use cases
        "use_cases": [
            "Production deployment with efficiency requirements",
            "Resource-constrained cloud environments",
            "Fast training iterations needed",
            "High accuracy with moderate compute budget",
            "Transfer learning with modern architecture"
        ],

        # Pros and cons
        "pros": [
            "Training up to 11x faster than EfficientNet-B7",
            "Better accuracy than ResNet-50 with fewer params",
            "Progressive learning for training stability",
            "Optimized for modern hardware (TPU/GPU)",
            "Fused-MBConv for efficiency"
        ],

        "cons": [
            "Larger input size (384) increases memory usage",
            "Slightly more complex than ResNet",
            "Less documentation than ResNet-50",
            "Requires more careful hyperparameter tuning"
        ],

        # When to use
        "when_to_use": "Use EfficientNetV2-S when you want state-of-the-art efficiency and accuracy balance, especially when training speed matters or you have limited compute budget.",

        "when_not_to_use": "Avoid if you need maximum simplicity (use ResNet), smallest model size (use MobileNet), or are doing educational/baseline comparisons (use ResNet-50).",

        # Alternatives
        "alternatives": [
            {
                "model": "resnet50",
                "reason": "More stable, better documentation, easier to understand"
            },
            {
                "model": "efficientnet_b0",
                "reason": "Smaller, even more efficient (original EfficientNet)"
            },
            {
                "model": "mobilenetv4_conv_medium",
                "reason": "Even lighter, optimized for mobile deployment"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 64,
                "range": [32, 128],
                "note": "Larger batches work well with EfficientNetV2"
            },
            "learning_rate": {
                "value": 0.001,
                "range": [0.0005, 0.005],
                "note": "Can use higher LR with progressive learning"
            },
            "epochs": {
                "value": 50,
                "range": [30, 100],
                "note": "Converges faster than V1"
            },
            "optimizer": "AdamW (recommended) or RMSprop",
            "scheduler": "Cosine with linear warmup (5 epochs)",
            "weight_decay": 0.00001,
            "image_size": 384,
            "augmentation": "AutoAugment or RandAugment (strong augmentation works well)"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Satellite Image Analysis",
                "description": "Used for land use classification with 10m resolution satellite imagery",
                "metrics": {
                    "after": "88% accuracy with 3x faster training than ResNet-101"
                }
            },
            {
                "title": "Manufacturing Quality Control",
                "description": "Real-time defect detection on production lines",
                "metrics": {
                    "after": "95% defect detection rate at 200 FPS (V100)"
                }
            }
        ]
    },

    "resnet18": {
        "display_name": "ResNet-18",
        "description": "Lightweight baseline CNN - Faster training and inference than ResNet-50",
        "params": "11.7M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p0", "baseline", "lightweight", "fast", "classic"],
        "priority": 0,
        "task_type": "image_classification",

        # Benchmark performance
        "benchmark": {
            "imagenet_top1": 69.8,
            "imagenet_top5": 89.1,
            "inference_speed_v100": 240,  # images/second
            "inference_speed_unit": "img/s",
            "training_time_epoch": "~1.2 hours",
            "training_hardware": "ImageNet, 8x V100",
            "flops": "1.8G",
        },

        # Use cases
        "use_cases": [
            "Quick prototyping and experimentation",
            "Resource-constrained environments",
            "Educational purposes and learning",
            "Fast iteration during development",
            "Baseline for lighter models"
        ],

        # Pros and cons
        "pros": [
            "Fast training (half the time of ResNet-50)",
            "Lower memory footprint",
            "Good for quick experiments",
            "Faster inference for real-time applications",
            "Well-tested and stable"
        ],

        "cons": [
            "Lower accuracy than deeper models",
            "Limited representation power",
            "Not suitable for complex tasks",
            "Less feature extraction capability"
        ],

        # When to use
        "when_to_use": "Use ResNet-18 for rapid prototyping, resource-constrained deployments, or when you need faster training iterations. Ideal for simpler classification tasks or as a baseline before trying heavier models.",

        "when_not_to_use": "Avoid if you need maximum accuracy, working with complex datasets, or have sufficient compute resources (use ResNet-50 or larger instead).",

        # Alternatives
        "alternatives": [
            {
                "model": "resnet50",
                "reason": "Better accuracy with more parameters"
            },
            {
                "model": "mobilenetv4_conv_medium",
                "reason": "Even lighter, optimized for mobile"
            },
            {
                "model": "efficientnet_b0",
                "reason": "Better accuracy at similar size"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 64,
                "range": [32, 128],
                "note": "Can use larger batches due to smaller model"
            },
            "learning_rate": {
                "value": 0.001,
                "range": [0.0005, 0.01],
                "note": "Similar to ResNet-50"
            },
            "epochs": {
                "value": 50,
                "range": [30, 100],
                "note": "Converges faster than deeper models"
            },
            "optimizer": "Adam or SGD with momentum",
            "scheduler": "Cosine annealing or step decay",
            "weight_decay": 0.0001,
            "image_size": 224,
            "augmentation": "Standard (RandomCrop, HorizontalFlip)"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Real-time Edge Deployment",
                "description": "Used in IoT cameras for real-time object recognition with limited compute",
                "metrics": {
                    "inference_speed": "30 FPS on Raspberry Pi 4",
                    "accuracy": "85% on custom dataset"
                }
            },
            {
                "title": "Rapid Prototyping",
                "description": "Startups use ResNet-18 for quick MVP validation before scaling to larger models",
                "metrics": {
                    "training_time": "2 hours vs 8 hours (ResNet-50)",
                    "development_cycle": "3x faster iterations"
                }
            }
        ]
    },

    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "description": "Efficient baseline CNN - Best accuracy per parameter with compound scaling",
        "params": "5.3M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p0", "efficient", "lightweight", "baseline", "2019"],
        "priority": 0,
        "task_type": "image_classification",

        # Benchmark performance
        "benchmark": {
            "imagenet_top1": 77.3,
            "imagenet_top5": 93.5,
            "inference_speed_v100": 180,  # images/second
            "inference_speed_unit": "img/s",
            "training_time_epoch": "~1.5 hours",
            "training_hardware": "ImageNet, 8x V100",
            "flops": "0.4G",
        },

        # Use cases
        "use_cases": [
            "Mobile and edge device deployment",
            "Efficient cloud inference at scale",
            "Transfer learning with limited data",
            "Cost-effective production systems",
            "Balanced accuracy and efficiency"
        ],

        # Pros and cons
        "pros": [
            "Smallest model in EfficientNet family",
            "Better accuracy than ResNet-18 with fewer params",
            "Excellent parameter efficiency",
            "Good transfer learning performance",
            "Fast inference on modern hardware"
        ],

        "cons": [
            "More complex architecture than ResNet",
            "Requires careful hyperparameter tuning",
            "Slightly slower than ResNet-18",
            "Less community resources than ResNet"
        ],

        # When to use
        "when_to_use": "Use EfficientNet-B0 when you need a good balance of accuracy and efficiency, especially for mobile deployment or when parameter count is a constraint. Ideal for transfer learning on small datasets.",

        "when_not_to_use": "Avoid if you need absolute simplicity (use ResNet), fastest possible inference (use MobileNet), or maximum accuracy regardless of size (use larger models).",

        # Alternatives
        "alternatives": [
            {
                "model": "resnet18",
                "reason": "Simpler architecture, faster training"
            },
            {
                "model": "efficientnetv2_s",
                "reason": "Improved version, faster training"
            },
            {
                "model": "mobilenetv4_conv_small",
                "reason": "Even more efficient for mobile"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 64,
                "range": [32, 128],
                "note": "Can use larger batches due to small size"
            },
            "learning_rate": {
                "value": 0.001,
                "range": [0.0005, 0.005],
                "note": "Use warmup for stability"
            },
            "epochs": {
                "value": 50,
                "range": [30, 100],
                "note": "May need more epochs than ResNet"
            },
            "optimizer": "AdamW (recommended) or RMSprop",
            "scheduler": "Cosine with warmup (5 epochs)",
            "weight_decay": 0.00001,
            "image_size": 224,
            "augmentation": "AutoAugment or RandAugment recommended"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Mobile App Integration",
                "description": "Plant identification app using EfficientNet-B0 for on-device inference",
                "metrics": {
                    "model_size": "20MB on mobile",
                    "inference_time": "100ms on iPhone 12",
                    "accuracy": "92% on 1000 plant species"
                }
            },
            {
                "title": "Cost-effective Cloud Serving",
                "description": "E-commerce platform using EfficientNet-B0 for product categorization at scale",
                "metrics": {
                    "throughput": "10x higher than ResNet-50",
                    "cost_savings": "60% reduction in inference costs",
                    "accuracy": "89% on 500 categories"
                }
            }
        ]
    },

    # ============================================================
    # P1: Core Expansion (6 models)
    # ============================================================

    "vgg16": {
        "display_name": "VGG-16",
        "description": "Classic deep CNN - Simple architecture, excellent for transfer learning",
        "params": "138.4M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.001,
        "tags": ["p1", "classic", "simple", "transfer-learning"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 71.6,
            "imagenet_top5": 90.6,
            "inference_speed_v100": 120,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Transfer learning baseline",
            "Feature extraction for other tasks",
            "Educational purposes",
        ],
        "pros": [
            "Very simple and intuitive architecture",
            "Excellent transfer learning performance",
            "Strong feature extractor",
        ],
        "cons": [
            "Large model size (138M params)",
            "Slower than modern architectures",
            "Memory intensive",
        ],
        "when_to_use": "Use VGG-16 for transfer learning or when you need a simple, well-understood architecture with strong feature extraction capabilities.",
        "alternatives": [
            {"model": "resnet50", "reason": "More efficient, better accuracy"},
            {"model": "efficientnet_b0", "reason": "Much smaller, faster"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 32, "range": [16, 64]},
            "learning_rate": {"value": 0.001, "range": [0.0001, 0.01]},
            "epochs": {"value": 50, "range": [20, 100]},
            "optimizer": "Adam or SGD",
            "scheduler": "Step decay or cosine",
        },
    },

    "mobilenetv3_large_100": {
        "display_name": "MobileNetV3-Large",
        "description": "Mobile-optimized CNN - Best for edge deployment and real-time inference",
        "params": "5.5M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 128,
        "recommended_lr": 0.001,
        "tags": ["p1", "mobile", "efficient", "edge", "realtime"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 75.8,
            "imagenet_top5": 92.7,
            "inference_speed_v100": 250,
            "inference_speed_unit": "img/s",
            "mobile_latency": "25ms on iPhone 11",
        },
        "use_cases": [
            "Mobile app integration",
            "Edge device deployment",
            "Real-time inference on CPU",
            "IoT and embedded systems",
        ],
        "pros": [
            "Very small and fast",
            "Optimized for mobile hardware",
            "Low power consumption",
            "Good accuracy for size",
        ],
        "cons": [
            "Lower accuracy than large models",
            "Complex architecture (hard to modify)",
            "May underperform on complex tasks",
        ],
        "when_to_use": "Use MobileNetV3 for mobile/edge deployment where inference speed and model size are critical constraints.",
        "alternatives": [
            {"model": "efficientnet_b0", "reason": "Better accuracy, slightly larger"},
            {"model": "resnet18", "reason": "Simpler architecture, similar speed"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 128, "range": [64, 256]},
            "learning_rate": {"value": 0.001, "range": [0.0005, 0.005]},
            "epochs": {"value": 100, "range": [50, 200]},
            "optimizer": "RMSprop or Adam",
            "scheduler": "Cosine with warmup",
        },
    },

    "densenet121": {
        "display_name": "DenseNet-121",
        "description": "Dense connection CNN - Parameter efficient with excellent gradient flow",
        "params": "8.0M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p1", "efficient", "dense-connections", "gradient-flow"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 74.9,
            "imagenet_top5": 92.2,
            "inference_speed_v100": 160,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Medical image analysis",
            "Small dataset scenarios",
            "Feature reuse applications",
        ],
        "pros": [
            "Very parameter efficient",
            "Excellent gradient flow (no vanishing gradients)",
            "Strong feature reuse",
            "Good for small datasets",
        ],
        "cons": [
            "Memory intensive during training",
            "Slower inference than ResNet",
            "Complex computational graph",
        ],
        "when_to_use": "Use DenseNet-121 when working with limited data or when you need strong gradient flow for deep networks.",
        "alternatives": [
            {"model": "resnet50", "reason": "Faster inference, simpler"},
            {"model": "efficientnet_b0", "reason": "More efficient overall"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 64, "range": [32, 128]},
            "learning_rate": {"value": 0.001, "range": [0.0001, 0.01]},
            "epochs": {"value": 100, "range": [50, 200]},
            "optimizer": "SGD with Nesterov or Adam",
            "scheduler": "Cosine annealing",
        },
    },

    "convnext_tiny": {
        "display_name": "ConvNeXt-Tiny",
        "description": "Modern CNN (2022) - Transformer-inspired design, state-of-the-art accuracy",
        "params": "28.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.0004,
        "tags": ["p1", "modern", "2022", "sota", "transformer-inspired"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 82.1,
            "imagenet_top5": 96.0,
            "inference_speed_v100": 140,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "State-of-the-art CNN baseline",
            "Production systems requiring high accuracy",
            "Research and benchmarking",
        ],
        "pros": [
            "Modern architecture with transformer insights",
            "Excellent accuracy for CNN",
            "Pure convolution (no attention)",
            "Good training stability",
        ],
        "cons": [
            "Larger than ResNet-50",
            "Requires careful hyperparameter tuning",
            "Less community resources than ResNet",
        ],
        "when_to_use": "Use ConvNeXt when you need state-of-the-art CNN performance without using transformers.",
        "alternatives": [
            {"model": "efficientnetv2_s", "reason": "Faster training, more efficient"},
            {"model": "vit_base_patch16_224", "reason": "Pure transformer, higher accuracy"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 64, "range": [32, 128]},
            "learning_rate": {"value": 0.0004, "range": [0.0001, 0.001]},
            "epochs": {"value": 100, "range": [50, 300]},
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "vit_base_patch16_224": {
        "display_name": "ViT-Base/16",
        "description": "Vision Transformer - Pure attention-based architecture, no convolutions",
        "params": "86.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0003,
        "tags": ["p1", "transformer", "attention", "sota", "2021"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 84.5,
            "imagenet_top5": 97.2,
            "inference_speed_v100": 100,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "High-accuracy requirements",
            "Large dataset scenarios",
            "Research on attention mechanisms",
        ],
        "pros": [
            "State-of-the-art accuracy",
            "Scales well with data size",
            "Global receptive field from start",
            "Interpretable attention maps",
        ],
        "cons": [
            "Requires large datasets",
            "Slower than CNNs",
            "High memory usage",
            "Poor with small datasets",
        ],
        "when_to_use": "Use ViT when you have large datasets and need maximum accuracy with attention mechanisms.",
        "when_not_to_use": "Avoid with small datasets (< 100K images) or resource-constrained environments.",
        "alternatives": [
            {"model": "convnext_tiny", "reason": "CNN alternative, faster, works with small data"},
            {"model": "swin_tiny_patch4_window7_224", "reason": "More efficient transformer"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 32, "range": [16, 64]},
            "learning_rate": {"value": 0.0003, "range": [0.0001, 0.001]},
            "epochs": {"value": 100, "range": [50, 300]},
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup (10-20 epochs)",
        },
    },

    "swin_tiny_patch4_window7_224": {
        "display_name": "Swin-Tiny",
        "description": "Hierarchical Vision Transformer - Efficient attention with shifted windows",
        "params": "28.3M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.0005,
        "tags": ["p1", "transformer", "hierarchical", "efficient", "2021"],
        "priority": 1,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 81.2,
            "imagenet_top5": 95.5,
            "inference_speed_v100": 130,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Vision tasks requiring spatial hierarchy",
            "Backbone for detection/segmentation",
            "Balanced accuracy and efficiency",
        ],
        "pros": [
            "More efficient than ViT",
            "Hierarchical features (like CNNs)",
            "Good for downstream tasks",
            "Shifted window attention reduces compute",
        ],
        "cons": [
            "Complex architecture",
            "Still requires substantial data",
            "Slower than pure CNNs",
        ],
        "when_to_use": "Use Swin Transformer when you need transformer benefits with better efficiency than ViT.",
        "alternatives": [
            {"model": "vit_base_patch16_224", "reason": "Simpler, higher accuracy"},
            {"model": "convnext_tiny", "reason": "Pure CNN, similar accuracy, faster"},
        ],
        "recommended_settings": {
            "batch_size": {"value": 64, "range": [32, 128]},
            "learning_rate": {"value": 0.0005, "range": [0.0001, 0.001]},
            "epochs": {"value": 100, "range": [50, 300]},
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    # ============================================================
    # P2: Full Coverage (8 models)
    # Specialized and advanced architectures for comprehensive coverage
    # ============================================================

    "regnety_008": {
        "display_name": "RegNetY-8GF",
        "description": "RegNet Y-8GF - Efficient architecture with optimal depth/width ratio",
        "params": "39.2M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p2", "regnet", "efficient", "modern"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 81.7,
            "imagenet_top5": 95.8,
            "inference_speed_v100": 280,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "General image classification with efficiency focus",
            "Applications requiring balanced accuracy/speed",
            "Transfer learning for custom datasets",
            "Mobile/edge deployment (optimized architecture)",
        ],
        "pros": [
            "Excellent accuracy/efficiency trade-off",
            "Fast inference speed (280 img/s)",
            "Optimized network design (NAS-inspired)",
            "Good generalization capability",
        ],
        "cons": [
            "Less popular than ResNet/EfficientNet",
            "Limited community resources",
            "Not as parameter-efficient as MobileNet",
        ],
        "when_to_use": "Choose RegNetY when you need a good balance between accuracy (81.7%) and inference speed, especially for production deployment where efficiency matters.",
        "when_not_to_use": "Avoid if you need state-of-the-art accuracy (>83%) or ultra-lightweight models (<10M params) for mobile devices.",
        "alternatives": [
            "ResNet-50 (similar accuracy, more widely used)",
            "EfficientNet-B2 (better parameter efficiency)",
            "ConvNeXt-Tiny (modern alternative)",
        ],
        "recommended_settings": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "maxvit_tiny_tf_224": {
        "display_name": "MaxViT-Tiny",
        "description": "MaxViT Tiny - Hybrid CNN+Transformer with multi-axis attention",
        "params": "31.0M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.0005,
        "tags": ["p2", "transformer", "hybrid", "attention"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 83.6,
            "imagenet_top5": 96.7,
            "inference_speed_v100": 150,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "High-accuracy image classification",
            "Vision tasks requiring global context",
            "Fine-grained classification (birds, products)",
            "Transfer learning for complex patterns",
        ],
        "pros": [
            "State-of-the-art accuracy (83.6%)",
            "Hybrid CNN+Transformer architecture",
            "Multi-axis attention mechanism",
            "Strong feature representation",
        ],
        "cons": [
            "Slower inference than pure CNNs",
            "Higher memory consumption",
            "Requires more training data",
            "Complex architecture to debug",
        ],
        "when_to_use": "Choose MaxViT when accuracy is critical (83.6%) and you have sufficient computational resources and training data.",
        "when_not_to_use": "Avoid for real-time applications or when training data is limited (<10K images). Use pure CNNs instead.",
        "alternatives": [
            "ViT-Base (pure transformer)",
            "Swin-Small (hierarchical transformer)",
            "ConvNeXt-Base (pure CNN alternative)",
        ],
        "recommended_settings": {
            "epochs": 150,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "beit_base_patch16_224": {
        "display_name": "BEiT-Base/16",
        "description": "BEiT Base - Self-supervised Vision Transformer with masked image modeling",
        "params": "86.5M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0003,
        "tags": ["p2", "transformer", "self-supervised", "bert-style"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 85.2,
            "imagenet_top5": 97.5,
            "inference_speed_v100": 110,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "State-of-the-art image classification",
            "Self-supervised pre-training",
            "Transfer learning with limited labels",
            "Vision-language tasks (multimodal)",
        ],
        "pros": [
            "Excellent accuracy (85.2%)",
            "Self-supervised pre-training (robust features)",
            "Strong transfer learning capability",
            "BERT-style masked image modeling",
        ],
        "cons": [
            "Very slow inference (110 img/s)",
            "Large model size (86.5M params)",
            "High memory requirements",
            "Requires significant compute for training",
        ],
        "when_to_use": "Choose BEiT when you need state-of-the-art accuracy (85.2%) and have access to GPU resources, especially for research or semi-supervised learning.",
        "when_not_to_use": "Avoid for production deployment with latency constraints or limited compute. Use EfficientNet or ResNet instead.",
        "alternatives": [
            "ViT-Base (similar architecture, supervised)",
            "Swin-Base (better efficiency)",
            "EfficientNetV2-M (faster inference)",
        ],
        "recommended_settings": {
            "epochs": 200,
            "learning_rate": 0.0003,
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "convnext_base": {
        "display_name": "ConvNeXt-Base",
        "description": "ConvNeXt Base - Modernized CNN with Transformer design principles",
        "params": "88.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0004,
        "tags": ["p2", "modern-cnn", "large", "high-accuracy"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 85.8,
            "imagenet_top5": 97.7,
            "inference_speed_v100": 140,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "State-of-the-art classification accuracy",
            "Production systems with GPU infrastructure",
            "Transfer learning for challenging datasets",
            "Research experiments requiring high baseline",
        ],
        "pros": [
            "State-of-the-art CNN accuracy (85.8%)",
            "Faster than Transformers at inference",
            "Modern design (depthwise conv, LayerNorm)",
            "Excellent transfer learning performance",
        ],
        "cons": [
            "Large model size (88.6M params)",
            "Higher memory consumption",
            "Slower training than smaller models",
        ],
        "when_to_use": "Choose ConvNeXt-Base when you need the best possible accuracy (85.8%) while keeping CNN's efficiency advantage over Transformers.",
        "when_not_to_use": "Avoid for edge devices or real-time applications. Use ConvNeXt-Tiny or EfficientNet instead.",
        "alternatives": [
            "BEiT-Base (transformer alternative)",
            "Swin-Base (hierarchical transformer)",
            "EfficientNetV2-L (more efficient)",
        ],
        "recommended_settings": {
            "epochs": 200,
            "learning_rate": 0.0004,
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "nfnet_f0": {
        "display_name": "NFNet-F0",
        "description": "NFNet F0 - Normalizer-Free Network with adaptive gradient clipping",
        "params": "71.5M",
        "input_size": 256,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0008,
        "tags": ["p2", "normalizer-free", "stable-training", "research"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 83.6,
            "imagenet_top5": 96.8,
            "inference_speed_v100": 90,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Research on BatchNorm alternatives",
            "Stable training at large batch sizes",
            "High-accuracy classification",
            "Experiments with normalization techniques",
        ],
        "pros": [
            "No BatchNorm required (simpler architecture)",
            "Stable training with large batches",
            "Good accuracy (83.6%)",
            "Adaptive gradient clipping",
        ],
        "cons": [
            "Slow inference (90 img/s)",
            "Less widely used than standard architectures",
            "Requires careful hyperparameter tuning",
            "Not optimized for edge deployment",
        ],
        "when_to_use": "Choose NFNet when researching normalization-free architectures or training with very large batch sizes (>2048).",
        "when_not_to_use": "Avoid for production systems or when using standard batch sizes (<256). Use ResNet or EfficientNet instead.",
        "alternatives": [
            "ResNet-101 (with BatchNorm)",
            "EfficientNetV2-M (better efficiency)",
            "ConvNeXt-Base (modern alternative)",
        ],
        "recommended_settings": {
            "epochs": 150,
            "learning_rate": 0.0008,
            "optimizer": "SGD",
            "scheduler": "Cosine",
        },
    },

    "coatnet_0_rw_224": {
        "display_name": "CoAtNet-0",
        "description": "CoAtNet-0 - Hybrid Convolution+Attention network with vertical layout",
        "params": "23.8M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.0005,
        "tags": ["p2", "hybrid", "attention", "efficient"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 81.6,
            "imagenet_top5": 95.7,
            "inference_speed_v100": 200,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Balanced accuracy/efficiency classification",
            "Hybrid CNN-Transformer exploration",
            "Transfer learning for medium-scale datasets",
            "Production with moderate compute budget",
        ],
        "pros": [
            "Good accuracy with modest params (81.6%, 23.8M)",
            "Hybrid architecture (best of CNNs+Transformers)",
            "Faster than pure transformers (200 img/s)",
            "Efficient parameter usage",
        ],
        "cons": [
            "Less popular than mainstream models",
            "Limited community support",
            "Complex architecture to customize",
        ],
        "when_to_use": "Choose CoAtNet when you want hybrid CNN+Transformer benefits with <25M parameters and good inference speed.",
        "when_not_to_use": "Avoid if you need ultra-high accuracy (>84%) or ultra-lightweight models (<10M params).",
        "alternatives": [
            "MaxViT-Tiny (more advanced hybrid)",
            "ConvNeXt-Tiny (pure CNN)",
            "ViT-Small (pure transformer)",
        ],
        "recommended_settings": {
            "epochs": 120,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "Cosine with warmup",
        },
    },

    "efficientformerv2_s0": {
        "display_name": "EfficientFormerV2-S0",
        "description": "EfficientFormerV2 S0 - Mobile-friendly Vision Transformer with 4D block",
        "params": "3.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 128,
        "recommended_lr": 0.002,
        "tags": ["p2", "mobile", "lightweight", "transformer", "edge"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 76.2,
            "imagenet_top5": 93.0,
            "inference_speed_v100": 450,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Mobile and edge device deployment",
            "Real-time classification with transformers",
            "Resource-constrained environments",
            "IoT and embedded vision systems",
        ],
        "pros": [
            "Ultra-lightweight (3.6M params)",
            "Very fast inference (450 img/s)",
            "Transformer architecture for mobile",
            "Low latency on CPU/mobile devices",
        ],
        "cons": [
            "Lower accuracy than larger models (76.2%)",
            "Limited capacity for complex tasks",
            "Newer model (less community support)",
        ],
        "when_to_use": "Choose EfficientFormerV2-S0 when deploying transformers on mobile/edge devices where latency is critical (<10ms).",
        "when_not_to_use": "Avoid when accuracy is paramount (need >80%). Use larger transformers or CNNs instead.",
        "alternatives": [
            "MobileNetV3-Small (pure CNN, even lighter)",
            "EfficientNet-B0 (better accuracy/params ratio)",
            "ViT-Tiny (standard small transformer)",
        ],
        "recommended_settings": {
            "epochs": 100,
            "learning_rate": 0.002,
            "optimizer": "AdamW",
            "scheduler": "Cosine",
        },
    },

    "repvgg_b0": {
        "display_name": "RepVGG-B0",
        "description": "RepVGG B0 - Re-parameterized VGG with structural re-param at inference",
        "params": "15.8M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["p2", "reparameterized", "simple", "fast-inference"],
        "priority": 2,
        "task_type": "image_classification",
        "benchmark": {
            "imagenet_top1": 75.1,
            "imagenet_top5": 92.4,
            "inference_speed_v100": 400,
            "inference_speed_unit": "img/s",
        },
        "use_cases": [
            "Production deployment prioritizing speed",
            "Hardware with optimized conv implementations",
            "Edge inference with simple architecture",
            "Transfer learning for simple patterns",
        ],
        "pros": [
            "Very fast inference (400 img/s)",
            "Simple inference-time architecture (3x3 conv only)",
            "Hardware-friendly (optimized kernels)",
            "Structural re-parameterization technique",
        ],
        "cons": [
            "Moderate accuracy (75.1%)",
            "Training-time architecture more complex",
            "Less flexible than modern architectures",
        ],
        "when_to_use": "Choose RepVGG when inference speed is critical and architecture simplicity matters for hardware optimization.",
        "when_not_to_use": "Avoid when accuracy is more important than speed, or when training efficiency matters more than inference.",
        "alternatives": [
            "MobileNetV3-Large (mobile-optimized)",
            "EfficientNet-B0 (better accuracy/params)",
            "ResNet-18 (more widely used)",
        ],
        "recommended_settings": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "SGD",
            "scheduler": "Cosine",
        },
    },
}


def get_timm_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model metadata by model name."""
    return TIMM_MODEL_REGISTRY.get(model_name)


def list_timm_models(
    tags: Optional[List[str]] = None,
    priority: Optional[int] = None
) -> List[str]:
    """
    List timm models, optionally filtered by tags or priority.

    Args:
        tags: Filter by tags (e.g., ["p0", "baseline"])
        priority: Filter by priority (0, 1, or 2)

    Returns:
        List of model names
    """
    models = []

    for name, info in TIMM_MODEL_REGISTRY.items():
        # Filter by priority
        if priority is not None and info.get("priority") != priority:
            continue

        # Filter by tags
        if tags:
            model_tags = info.get("tags", [])
            if not any(tag in model_tags for tag in tags):
                continue

        models.append(name)

    return models

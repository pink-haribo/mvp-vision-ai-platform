"""Ultralytics YOLO model registry."""

from typing import Dict, Any, List, Optional

# P0: Priority 0 models (Quick Win validation)
# P1: Priority 1 models (Core expansion)
# P2: Priority 2 models (Full coverage)

ULTRALYTICS_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ============================================================
    # P0: Quick Win (4 models - including YOLO-World!)
    # ============================================================

    "yolo11n": {
        "display_name": "YOLOv11 Nano",
        "description": "Latest YOLO (Sep 2024) - Ultra-lightweight real-time detection",
        "params": "2.6M",
        "input_size": 640,
        "task_types": ["object_detection"],
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.01,
        "tags": ["p0", "latest", "2024", "ultralight", "realtime", "edge", "yolo11"],
        "priority": 0,

        # 🆕 Phase 1: Model metadata
        "status": "active",
        "inference_only": False,
        "recommended": True,  # P0 latest YOLO for edge
        "performance_tier": "fast",  # Ultra-lightweight, fast inference

        # Benchmark performance
        "benchmark": {
            "coco_map50": 52.1,
            "coco_map50_95": 39.5,
            "inference_speed_v100": 120,  # FPS
            "inference_speed_unit": "FPS",
            "inference_speed_jetson_nano": 15,
            "inference_speed_cpu": 25,
            "model_size_mb": 5.8,
            "vs_yolov8n": "-22% params, +1.2 mAP",
            "flops": "6.5G",
        },

        # Use cases
        "use_cases": [
            "Edge devices (Raspberry Pi, Jetson Nano)",
            "Mobile deployment (iOS, Android apps)",
            "Real-time video processing on CPU",
            "Resource-constrained cloud servers",
            "IoT cameras and embedded systems"
        ],

        # Pros and cons
        "pros": [
            "22% fewer parameters than YOLOv8n",
            "Latest YOLO architecture (Sep 2024)",
            "Fast inference even on CPU (25 FPS)",
            "Very small model size (5.8 MB)",
            "Real-time capable on edge devices"
        ],

        "cons": [
            "Lower accuracy than larger models",
            "May struggle with small or crowded objects",
            "Less suitable for high-precision tasks",
            "Newer model, less battle-tested"
        ],

        # When to use
        "when_to_use": "Use YOLOv11n when deployment on edge/mobile devices is critical, or when real-time speed is more important than maximum accuracy. Ideal for resource-constrained environments.",

        "when_not_to_use": "Avoid if you need maximum detection accuracy, complex scene understanding, or precise small object detection (use YOLOv11m or YOLOv11l instead).",

        # Alternatives
        "alternatives": [
            {
                "model": "yolo11m",
                "reason": "Better accuracy (+12 mAP), still good speed"
            },
            {
                "model": "yolov8n",
                "reason": "More stable, well-tested (but 22% more params)"
            },
            {
                "model": "yolo_world_v2_s",
                "reason": "Flexible classes with zero-shot detection"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 64,
                "range": [32, 128],
                "note": "Can use large batches due to small model size"
            },
            "learning_rate": {
                "value": 0.01,
                "range": [0.001, 0.02],
                "note": "YOLO uses higher LR than classification models"
            },
            "epochs": {
                "value": 100,
                "range": [50, 300],
                "note": "Detection models benefit from longer training"
            },
            "optimizer": "AdamW or SGD with momentum",
            "scheduler": "Cosine annealing",
            "weight_decay": 0.0005,
            "image_size": 640,
            "augmentation": "Mosaic, MixUp, HSV augmentation (built-in)"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Smart Home Security Camera",
                "description": "Person detection on Raspberry Pi 4 for home security system",
                "metrics": {
                    "after": "15 FPS real-time detection, 92% accuracy"
                }
            },
            {
                "title": "Retail People Counting",
                "description": "Customer counting at store entrances with edge devices",
                "metrics": {
                    "after": "98% counting accuracy, 25 FPS on CPU"
                }
            }
        ]
    },

    "yolo11n-seg": {
        "display_name": "YOLOv11n-seg",
        "description": "Latest YOLO11 Nano Segmentation - Ultra-lightweight instance segmentation",
        "params": "2.9M",
        "input_size": 640,
        "task_types": ["instance_segmentation"],
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["p0", "latest", "2024", "segmentation", "instance", "edge", "yolo11"],
        "priority": 0,

        "status": "active",
        "inference_only": False,
        "recommended": True,
        "performance_tier": "fast",

        "benchmark": {
            "coco_seg_map50_95": 38.4,
            "coco_box_map50_95": 38.9,
            "inference_speed_v100": 90,  # FPS
            "inference_speed_unit": "FPS",
            "model_size_mb": 6.5,
            "flops": "12.9G",
        },

        "use_cases": [
            "Edge device instance segmentation",
            "Real-time segmentation on mobile devices",
            "Object counting with precise masks",
            "Product detection in retail",
            "Autonomous navigation obstacle detection"
        ],

        "pros": [
            "Latest YOLO11 architecture (Sep 2024)",
            "Ultra-lightweight for segmentation (2.9M)",
            "Fast inference even on CPU",
            "Instance segmentation + bounding boxes",
            "Edge device compatible"
        ],

        "cons": [
            "Lower accuracy than larger seg models",
            "May struggle with overlapping objects",
            "Limited fine-grained detail",
            "Newer model, less battle-tested"
        ],

        "when_to_use": "Use YOLO11n-seg when you need instance segmentation on edge devices or real-time applications where speed is critical.",

        "when_not_to_use": "Avoid if you need precise segmentation masks, high overlap handling, or maximum accuracy (use YOLOv11m-seg or SAM2 instead).",

        "alternatives": [
            {
                "model": "yolov8n-seg",
                "reason": "More stable, similar performance"
            },
            {
                "model": "sam2_t",
                "reason": "Zero-shot segmentation with prompts"
            },
            {
                "model": "yolo11m-seg",
                "reason": "Higher accuracy, slower"
            }
        ],

        "recommended_settings": {
            "batch_size": {
                "value": 32,
                "range": [16, 64],
                "note": "Segmentation requires more memory than detection"
            },
            "learning_rate": {
                "value": 0.01,
                "range": [0.001, 0.02],
                "note": "Standard YOLO learning rate"
            },
            "epochs": {
                "value": 150,
                "range": [100, 300],
                "note": "Segmentation benefits from longer training"
            },
            "optimizer": "AdamW",
            "scheduler": "Cosine annealing",
            "image_size": 640,
        },
    },

    "yolo11n-pose": {
        "display_name": "YOLOv11n-pose",
        "description": "Latest YOLO11 Nano Pose - Ultra-lightweight human pose estimation",
        "params": "2.9M",
        "input_size": 640,
        "task_types": ["pose_estimation"],
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["p0", "latest", "2024", "pose", "keypoints", "edge", "yolo11"],
        "priority": 0,

        "status": "active",
        "inference_only": False,
        "recommended": True,
        "performance_tier": "fast",

        "benchmark": {
            "coco_pose_map50_95": 50.5,
            "inference_speed_v100": 95,  # FPS
            "inference_speed_unit": "FPS",
            "model_size_mb": 6.5,
            "keypoints": 17,  # COCO format
            "flops": "7.6G",
        },

        "use_cases": [
            "Real-time fitness tracking apps",
            "Sports analytics on edge devices",
            "Gesture recognition systems",
            "Human-robot interaction",
            "Mobile AR applications"
        ],

        "pros": [
            "Latest YOLO11 architecture (Sep 2024)",
            "Ultra-lightweight for pose estimation",
            "Fast inference on CPU/edge devices",
            "17 keypoints (COCO format)",
            "Multi-person pose detection"
        ],

        "cons": [
            "Lower accuracy than larger pose models",
            "May struggle with occlusions",
            "Limited to 17 keypoints (COCO)",
            "Newer model, less proven"
        ],

        "when_to_use": "Use YOLO11n-pose when you need real-time pose estimation on edge/mobile devices or CPU-only systems.",

        "when_not_to_use": "Avoid if you need maximum pose accuracy, handle heavy occlusions, or need more than 17 keypoints (use YOLOv11m-pose or specialized models).",

        "alternatives": [
            {
                "model": "yolov8n-pose",
                "reason": "More stable, similar performance"
            },
            {
                "model": "yolo11m-pose",
                "reason": "Higher accuracy, slower"
            },
            {
                "model": "OpenPose",
                "reason": "More keypoints, much slower"
            }
        ],

        "recommended_settings": {
            "batch_size": {
                "value": 32,
                "range": [16, 64],
                "note": "Pose estimation memory requirements"
            },
            "learning_rate": {
                "value": 0.01,
                "range": [0.001, 0.02],
                "note": "Standard YOLO learning rate"
            },
            "epochs": {
                "value": 200,
                "range": [150, 300],
                "note": "Pose models need longer training"
            },
            "optimizer": "AdamW",
            "scheduler": "Cosine annealing",
            "image_size": 640,
        },
    },

    # "yolo11m": {
    #     "display_name": "YOLOv11 Medium",
    #     "description": "Latest YOLO (Sep 2024) - Best accuracy/speed balance for production",
    #     "params": "20.1M",
    #     "input_size": 640,
    #     "task_types": ["object_detection"],
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p0", "latest", "2024", "balanced", "production", "sota", "yolo11"],
    #     "priority": 0,

    #     # 🆕 Phase 1: Model metadata
    #     "status": "active",
    #     "inference_only": False,
    #     "recommended": True,  # P0 best balance for production
    #     "performance_tier": "balanced",  # Best accuracy/speed tradeoff

    #     # Benchmark performance
    #     "benchmark": {
    #         "coco_map50": 67.8,
    #         "coco_map50_95": 51.5,
    #         "inference_speed_v100": 60,  # FPS
    #         "inference_speed_unit": "FPS",
    #         "inference_speed_t4": 35,
    #         "inference_speed_cpu": 5,
    #         "model_size_mb": 40.2,
    #         "vs_yolov8m": "-22% params, +1.3 mAP",
    #         "flops": "68.6G",
    #     },

    #     # Use cases
    #     "use_cases": [
    #         "Production object detection systems",
    #         "Autonomous vehicles and robotics",
    #         "Security and surveillance systems",
    #         "Industrial quality inspection",
    #         "Retail analytics and people counting"
    #     ],

    #     # Pros and cons
    #     "pros": [
    #         "Best accuracy/speed trade-off in YOLO series",
    #         "22% fewer params than YOLOv8m",
    #         "Higher mAP than YOLOv8m (+1.3)",
    #         "Production-ready and well-optimized",
    #         "Excellent balance for GPU deployment"
    #     ],

    #     "cons": [
    #         "Requires GPU for real-time performance",
    #         "Larger model size than nano (40 MB)",
    #         "Higher compute requirements than nano",
    #         "Not suitable for edge devices"
    #     ],

    #     # When to use
    #     "when_to_use": "Use YOLOv11m when you need the best balance of accuracy and speed for production deployment with GPU available. The go-to choice for most object detection tasks.",

    #     "when_not_to_use": "Avoid if deploying on edge/mobile (use YOLOv11n), need maximum accuracy regardless of speed (use YOLOv11l/x), or doing zero-shot detection (use YOLO-World).",

    #     # Alternatives
    #     "alternatives": [
    #         {
    #             "model": "yolo11n",
    #             "reason": "Much faster, suitable for edge devices"
    #         },
    #         {
    #             "model": "yolo11l",
    #             "reason": "Higher accuracy (+2-3 mAP) if speed is less critical"
    #         },
    #         {
    #             "model": "yolo_world_v2_m",
    #             "reason": "Similar size but with zero-shot capability"
    #         }
    #     ],

    #     # Recommended settings
    #     "recommended_settings": {
    #         "batch_size": {
    #             "value": 16,
    #             "range": [8, 32],
    #             "note": "Adjust based on GPU memory"
    #         },
    #         "learning_rate": {
    #             "value": 0.01,
    #             "range": [0.001, 0.02],
    #             "note": "Standard YOLO learning rate"
    #         },
    #         "epochs": {
    #             "value": 100,
    #             "range": [50, 300],
    #             "note": "100 epochs usually sufficient"
    #         },
    #         "optimizer": "AdamW (recommended) or SGD",
    #         "scheduler": "Cosine annealing with warmup",
    #         "weight_decay": 0.0005,
    #         "image_size": 640,
    #         "augmentation": "Mosaic, MixUp, CopyPaste, HSV (built-in)"
    #     },

    #     # Real-world examples
    #     "real_world_examples": [
    #         {
    #             "title": "Warehouse Automation",
    #             "description": "Package detection and sorting in logistics center",
    #             "metrics": {
    #                 "after": "99.2% detection accuracy at 60 FPS (V100)"
    #             }
    #         },
    #         {
    #             "title": "Traffic Monitoring",
    #             "description": "Vehicle and pedestrian detection for smart city",
    #             "metrics": {
    #                 "after": "96% accuracy in complex urban scenes, 35 FPS (T4)"
    #             }
    #         }
    #     ]
    # },

    "yolo_world_v2_s": {
        "display_name": "YOLO-World v2 Small",
        "description": "Open-vocabulary detection (CVPR 2024) - Detect ANY object with text prompts",
        "params": "22M",
        "input_size": 640,
        "task_types": ["zero_shot_detection"],  # Using existing enum
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["p0", "cvpr2024", "open-vocab", "zero-shot", "innovative", "text-prompt", "yolo-world"],
        "priority": 0,

        # Benchmark performance
        "benchmark": {
            "lvis_map": 26.2,
            "lvis_map_rare": 17.8,  # Performance on rare classes
            "coco_map50": 62.3,  # Zero-shot on COCO
            "coco_map50_95": 44.3,
            "inference_speed_v100": 52,  # FPS
            "inference_speed_unit": "FPS",
            "model_size_mb": 44,
            "custom_classes_support": "Unlimited",
            "vs_traditional": "No retraining needed for new classes",
        },

        # 🌟 Special features for YOLO-World
        "special_features": {
            "type": "open_vocabulary",
            "capabilities": [
                "Detect objects without training on them",
                "Define classes using natural language text",
                "Zero-shot detection capability",
                "Dynamic class definition at runtime",
                "Support for rare and long-tail objects"
            ],
            "example_prompts": [
                "a red apple",
                "damaged product packaging",
                "person wearing a hat",
                "car with visible license plate",
                "ripe banana vs unripe banana"
            ],
            "usage_example": {
                "traditional_yolo": "model.predict('image.jpg')  # Detects 80 COCO classes",
                "yolo_world": "model.set_classes(['cat', 'dog', 'my custom object']).predict('image.jpg')  # Detects custom classes!"
            },
            "prompt_engineering_tips": [
                "Be specific: 'red apple' works better than 'apple'",
                "Use descriptive attributes: 'damaged', 'ripe', 'vintage'",
                "Combine object + state: 'person wearing mask'",
                "Avoid ambiguity: specify color, size, condition"
            ]
        },

        # Use cases
        "use_cases": [
            "Retail: Detect new products without retraining",
            "Security: Custom threat detection with flexible classes",
            "Quality control: Find specific defects described in text",
            "Research: Rapid prototyping with new object categories",
            "E-commerce: Flexible product detection and cataloging"
        ],

        # Pros and cons
        "pros": [
            "No retraining needed for new object classes",
            "Natural language class definition (very intuitive)",
            "Fast adaptation to new detection scenarios",
            "Handles rare and custom objects well",
            "Real-time speed maintained (52 FPS)"
        ],

        "cons": [
            "Lower accuracy than specialized models (~15-20% less)",
            "Requires careful prompt engineering",
            "Slower than standard YOLO (text encoding overhead)",
            "Limited to detection only (no segmentation yet)",
            "Newer technology, fewer examples available"
        ],

        # When to use
        "when_to_use": "Use YOLO-World when you need flexibility to detect new object types without retraining, dealing with long-tail or rare objects, or want to rapidly prototype detection systems with changing requirements.",

        "when_not_to_use": "Avoid if you need maximum accuracy on fixed classes (use YOLOv11), doing segmentation (use YOLOv11-seg), or have well-defined static class sets (traditional YOLO will be more accurate).",

        # Alternatives
        "alternatives": [
            {
                "model": "yolo11m",
                "reason": "Higher accuracy for fixed 80 COCO classes"
            },
            {
                "model": "yolo_world_v2_m",
                "reason": "Larger, more accurate open-vocab model"
            }
        ],

        # Recommended settings
        "recommended_settings": {
            "batch_size": {
                "value": 16,
                "range": [8, 32],
                "note": "Similar to YOLOv11m"
            },
            "learning_rate": {
                "value": 0.01,
                "range": [0.001, 0.02],
                "note": "Standard YOLO learning rate"
            },
            "epochs": {
                "value": 100,
                "range": [50, 200],
                "note": "Open-vocab models may need longer training"
            },
            "optimizer": "AdamW",
            "scheduler": "Cosine annealing",
            "weight_decay": 0.0005,
            "image_size": 640,
            "custom_prompts": "List of text descriptions for classes",
            "prompt_mode": "offline (pre-computed) or dynamic (runtime)"
        },

        # Real-world examples
        "real_world_examples": [
            {
                "title": "Retail Inventory Management",
                "description": "Detect thousands of product SKUs without individual model training",
                "metrics": {
                    "after": "95% detection accuracy for 5000+ products, instant new product support"
                }
            },
            {
                "title": "Wildlife Monitoring",
                "description": "Detect rare animal species with text descriptions only",
                "metrics": {
                    "after": "87% detection rate for 200+ species, no training data needed"
                }
            }
        ],

        # 🔧 Special configuration
        "requires_custom_prompts": True,
        "prompt_input_required": True
    },

    # "yolo_world_v2_m": {
    #     "display_name": "YOLO-World v2 Medium",
    #     "description": "Open-vocabulary detection (CVPR 2024) - More accurate zero-shot detection",
    #     "params": "42M",
    #     "input_size": 640,
    #     "task_types": ["zero_shot_detection"],
    #     "pretrained_available": True,
    #     "recommended_batch_size": 8,
    #     "recommended_lr": 0.01,
    #     "tags": ["p0", "cvpr2024", "open-vocab", "zero-shot", "accurate", "yolo-world"],
    #     "priority": 0,

    #     # Benchmark performance
    #     "benchmark": {
    #         "lvis_map": 35.4,  # State-of-the-art on LVIS
    #         "lvis_map_rare": 26.8,
    #         "coco_map50": 68.1,
    #         "coco_map50_95": 48.1,
    #         "inference_speed_v100": 52,  # FPS (same as small due to optimizations)
    #         "inference_speed_unit": "FPS",
    #         "model_size_mb": 84,
    #         "custom_classes_support": "Unlimited",
    #         "vs_yolo_world_s": "+9.2 mAP on LVIS",
    #     },

    #     # Special features
    #     "special_features": {
    #         "type": "open_vocabulary",
    #         "capabilities": [
    #             "State-of-the-art open-vocabulary performance",
    #             "Best-in-class rare object detection",
    #             "Robust to prompt variations",
    #             "Multi-language support (experimental)",
    #             "Better understanding of complex descriptions"
    #         ],
    #         "example_prompts": [
    #             "vintage car from 1950s",
    #             "person with blue backpack",
    #             "damaged packaging box with torn corner",
    #             "ripe banana vs unripe banana",
    #             "dog wearing a collar"
    #         ],
    #         "usage_example": {
    #             "traditional_yolo": "model.predict('image.jpg')  # 80 classes only",
    #             "yolo_world": "model.set_classes(['specific object description']).predict('image.jpg')  # Anything!"
    #         },
    #         "prompt_engineering_tips": [
    #             "More robust to prompt variations than small version",
    #             "Can handle compound descriptions better",
    #             "Supports multi-attribute queries effectively",
    #             "Better with detailed, specific prompts"
    #         ]
    #     },

    #     # Use cases
    #     "use_cases": [
    #         "Large-scale retail inventory systems",
    #         "Advanced security and surveillance",
    #         "Medical imaging with custom conditions",
    #         "Autonomous vehicles (rare scenario detection)",
    #         "Wildlife monitoring and species detection"
    #     ],

    #     # Pros and cons
    #     "pros": [
    #         "Best-in-class open-vocabulary accuracy",
    #         "Excellent rare object detection performance",
    #         "More robust prompt understanding",
    #         "Still maintains real-time speed (52 FPS)",
    #         "Better generalization to unseen classes"
    #     ],

    #     "cons": [
    #         "2x parameters vs small version (42M)",
    #         "Higher memory usage (84 MB model)",
    #         "Slower than standard YOLO",
    #         "Requires more compute resources",
    #         "Still lower accuracy than specialized models"
    #     ],

    #     # When to use
    #     "when_to_use": "Use YOLO-World-v2-M when you need maximum accuracy for open-vocabulary detection and have sufficient GPU resources. Best for production systems requiring flexible class definitions with high accuracy.",

    #     "when_not_to_use": "Avoid if deploying on edge devices (use small version), need maximum speed (use standard YOLO), or have fixed well-defined classes (YOLOv11 will be more accurate).",

    #     # Alternatives
    #     "alternatives": [
    #         {
    #             "model": "yolo_world_v2_s",
    #             "reason": "Faster, lighter, still good for most use cases"
    #         },
    #         {
    #             "model": "yolo11l",
    #             "reason": "Higher accuracy on fixed 80 COCO classes"
    #         }
    #     ],

    #     # Recommended settings
    #     "recommended_settings": {
    #         "batch_size": {
    #             "value": 8,
    #             "range": [4, 16],
    #             "note": "Larger model requires more memory"
    #         },
    #         "learning_rate": {
    #             "value": 0.01,
    #             "range": [0.001, 0.02],
    #             "note": "Standard YOLO learning rate"
    #         },
    #         "epochs": {
    #             "value": 100,
    #             "range": [50, 200],
    #             "note": "Larger models may need more epochs"
    #         },
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine annealing with warmup",
    #         "weight_decay": 0.0005,
    #         "image_size": 640,
    #         "custom_prompts": "List of detailed text descriptions",
    #         "prompt_mode": "offline recommended for best performance"
    #     },

    #     # Real-world examples
    #     "real_world_examples": [
    #         {
    #             "title": "Airport Security Screening",
    #             "description": "Detect custom prohibited items using text descriptions",
    #             "metrics": {
    #                 "after": "98% detection rate for 500+ prohibited item types"
    #             }
    #         },
    #         {
    #             "title": "Agricultural Monitoring",
    #             "description": "Detect various crop diseases with descriptive prompts",
    #             "metrics": {
    #                 "after": "93% disease detection across 50+ crop types"
    #             }
    #         }
    #     ],

    #     # Special configuration
    #     "requires_custom_prompts": True,
    #     "prompt_input_required": True
    # },

    # YOLOv8 Series (proven, stable baseline)
    # "yolov8n": {
    #     "display_name": "YOLOv8 Nano",
    #     "description": "Proven YOLO baseline (2023) - Stable and widely tested",
    #     "params": "3.2M",
    #     "input_size": 640,
    #     "task_types": ["object_detection"],
    #     "pretrained_available": True,
    #     "recommended_batch_size": 64,
    #     "recommended_lr": 0.01,
    #     "tags": ["p0", "baseline", "2023", "stable", "proven", "yolov8"],
    #     "priority": 0,

    #     # 🆕 Phase 1: Model metadata
    #     "status": "active",
    #     "inference_only": False,
    #     "recommended": False,  # Superseded by yolo11n, but kept for stability
    #     "performance_tier": "fast",  # Fast nano model

    #     # Benchmark performance
    #     "benchmark": {
    #         "coco_map50": 50.2,
    #         "coco_map50_95": 37.3,
    #         "inference_speed_v100": 100,
    #         "inference_speed_unit": "FPS",
    #         "inference_speed_jetson_nano": 12,
    #         "inference_speed_cpu": 20,
    #         "model_size_mb": 6.2,
    #         "flops": "8.7G",
    #     },

    #     # Use cases
    #     "use_cases": [
    #         "Production deployments requiring stability",
    #         "Baseline comparison for new models",
    #         "Educational and research purposes",
    #         "Known-good starting point for projects",
    #         "When you need extensive community support"
    #     ],

    #     # Pros and cons
    #     "pros": [
    #         "Extensively tested and proven in production",
    #         "Large community and resources",
    #         "Stable training behavior",
    #         "Well-documented edge cases",
    #         "Many pretrained checkpoints available"
    #     ],

    #     "cons": [
    #         "Slightly larger than YOLOv11n",
    #         "Not the latest architecture",
    #         "Lower mAP than YOLOv11",
    #         "Higher computational cost than v11"
    #     ],

    #     # When to use
    #     "when_to_use": "Use YOLOv8n when you need a proven, stable baseline with extensive community support. Ideal for production systems where reliability is more important than cutting-edge performance.",

    #     "when_not_to_use": "Avoid if you need absolute best performance (use YOLOv11) or smallest model size (use YOLOv11n).",

    #     # Alternatives
    #     "alternatives": [
    #         {
    #             "model": "yolo11n",
    #             "reason": "22% smaller with better accuracy"
    #         },
    #         {
    #             "model": "yolov8s",
    #             "reason": "Better accuracy, slightly larger"
    #         }
    #     ],

    #     # Recommended settings
    #     "recommended_settings": {
    #         "batch_size": {
    #             "value": 64,
    #             "range": [32, 128],
    #             "note": "Adjust based on GPU memory"
    #         },
    #         "learning_rate": {
    #             "value": 0.01,
    #             "range": [0.001, 0.1],
    #             "note": "Use warmup for large batches"
    #         },
    #         "epochs": {
    #             "value": 100,
    #             "range": [50, 300],
    #             "note": "More epochs for small datasets"
    #         },
    #         "optimizer": "SGD or AdamW",
    #         "image_size": 640,
    #         "augmentation": "Mosaic, MixUp, HSV augmentation"
    #     },

    #     # Real-world examples
    #     "real_world_examples": [
    #         {
    #             "title": "Security System Deployment",
    #             "description": "Deployed in 500+ retail stores for real-time theft detection",
    #             "metrics": {
    #                 "uptime": "99.9% over 6 months",
    #                 "detection_rate": "94% on custom dataset",
    #                 "false_positives": "<2% per day"
    #             }
    #         }
    #     ]
    # },

    # "yolov8s": {
    #     "display_name": "YOLOv8 Small",
    #     "description": "Balanced YOLO model (2023) - Good accuracy-speed tradeoff",
    #     "params": "11.2M",
    #     "input_size": 640,
    #     "task_types": ["object_detection"],
    #     "pretrained_available": True,
    #     "recommended_batch_size": 32,
    #     "recommended_lr": 0.01,
    #     "tags": ["p0", "baseline", "2023", "balanced", "proven", "yolov8"],
    #     "priority": 0,

    #     # Benchmark performance
    #     "benchmark": {
    #         "coco_map50": 61.8,
    #         "coco_map50_95": 44.9,
    #         "inference_speed_v100": 80,
    #         "inference_speed_unit": "FPS",
    #         "inference_speed_jetson_nano": 8,
    #         "inference_speed_cpu": 12,
    #         "model_size_mb": 22,
    #         "flops": "28.6G",
    #     },

    #     # Use cases
    #     "use_cases": [
    #         "Standard detection tasks on cloud GPUs",
    #         "Good balance of speed and accuracy",
    #         "Transfer learning baseline",
    #         "Production systems with GPU access",
    #         "When nano model accuracy is insufficient"
    #     ],

    #     # Pros and cons
    #     "pros": [
    #         "Better accuracy than nano models",
    #         "Still fast enough for real-time",
    #         "Good for standard GPU setups",
    #         "Proven performance in production",
    #         "Extensive pretrained weights"
    #     ],

    #     "cons": [
    #         "Too large for edge devices",
    #         "Slower than nano variants",
    #         "Higher memory requirements",
    #         "Not optimized for mobile"
    #     ],

    #     # When to use
    #     "when_to_use": "Use YOLOv8s when you have GPU resources and need better accuracy than nano models. Ideal for cloud deployments where speed is still important but not critical.",

    #     "when_not_to_use": "Avoid for edge/mobile deployment (use nano) or when you need maximum accuracy (use medium/large).",

    #     # Alternatives
    #     "alternatives": [
    #         {
    #             "model": "yolo11m",
    #             "reason": "Latest architecture with similar size"
    #         },
    #         {
    #             "model": "yolov8n",
    #             "reason": "Much faster, lower accuracy"
    #         },
    #         {
    #             "model": "yolov8m",
    #             "reason": "Better accuracy, slower"
    #         }
    #     ],

    #     # Recommended settings
    #     "recommended_settings": {
    #         "batch_size": {
    #             "value": 32,
    #             "range": [16, 64],
    #             "note": "Balance between speed and memory"
    #         },
    #         "learning_rate": {
    #             "value": 0.01,
    #             "range": [0.001, 0.1],
    #             "note": "Standard YOLO learning rate"
    #         },
    #         "epochs": {
    #             "value": 100,
    #             "range": [50, 300],
    #             "note": "Standard training duration"
    #         },
    #         "optimizer": "SGD with momentum 0.937",
    #         "image_size": 640,
    #         "augmentation": "Full YOLO augmentation suite"
    #     },

    #     # Real-world examples
    #     "real_world_examples": [
    #         {
    #             "title": "Manufacturing Quality Control",
    #             "description": "Defect detection on production lines with 95% accuracy",
    #             "metrics": {
    #                 "throughput": "50 FPS on RTX 3080",
    #                 "accuracy": "95% defect detection",
    #                 "deployment_time": "2 weeks from POC to production"
    #             }
    #         }
    #     ]
    # },

    # "yolov8m": {
    #     "display_name": "YOLOv8 Medium",
    #     "description": "High-accuracy YOLO model (2023) - Production-ready performance",
    #     "params": "25.9M",
    #     "input_size": 640,
    #     "task_types": ["object_detection"],
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p0", "baseline", "2023", "accurate", "proven", "yolov8"],
    #     "priority": 0,

    #     # Benchmark performance
    #     "benchmark": {
    #         "coco_map50": 67.2,
    #         "coco_map50_95": 50.2,
    #         "inference_speed_v100": 60,
    #         "inference_speed_unit": "FPS",
    #         "inference_speed_jetson_nano": 4,
    #         "inference_speed_cpu": 6,
    #         "model_size_mb": 52,
    #         "flops": "78.9G",
    #     },

    #     # Use cases
    #     "use_cases": [
    #         "High-accuracy requirements (medical, safety-critical)",
    #         "Cloud inference with GPU availability",
    #         "Offline processing pipelines",
    #         "When accuracy > speed priority",
    #         "Complex multi-class detection"
    #     ],

    #     # Pros and cons
    #     "pros": [
    #         "High accuracy on COCO dataset",
    #         "Good generalization to custom data",
    #         "Proven in production systems",
    #         "Excellent for fine-tuning",
    #         "Strong baseline for comparisons"
    #     ],

    #     "cons": [
    #         "Slower inference than smaller models",
    #         "Requires powerful GPU",
    #         "Higher training cost",
    #         "Not suitable for real-time on CPU",
    #         "Large model size for deployment"
    #     ],

    #     # When to use
    #     "when_to_use": "Use YOLOv8m when accuracy is the primary concern and you have adequate GPU resources. Ideal for applications where detection quality directly impacts business value.",

    #     "when_not_to_use": "Avoid for real-time mobile/edge applications, CPU inference, or when model size is constrained.",

    #     # Alternatives
    #     "alternatives": [
    #         {
    #             "model": "yolo11m",
    #             "reason": "Latest architecture, similar size"
    #         },
    #         {
    #             "model": "yolov8s",
    #             "reason": "Faster, lower accuracy"
    #         },
    #         {
    #             "model": "yolov8l",
    #             "reason": "Even higher accuracy (P1)"
    #         }
    #     ],

    #     # Recommended settings
    #     "recommended_settings": {
    #         "batch_size": {
    #             "value": 16,
    #             "range": [8, 32],
    #             "note": "Requires 16GB+ GPU memory"
    #         },
    #         "learning_rate": {
    #             "value": 0.01,
    #             "range": [0.001, 0.1],
    #             "note": "Standard YOLO learning rate"
    #         },
    #         "epochs": {
    #             "value": 100,
    #             "range": [50, 300],
    #             "note": "Longer training for best results"
    #         },
    #         "optimizer": "SGD with momentum 0.937",
    #         "image_size": 640,
    #         "augmentation": "Full augmentation recommended"
    #     },

    #     # Real-world examples
    #     "real_world_examples": [
    #         {
    #             "title": "Medical Imaging Analysis",
    #             "description": "Tumor detection in CT scans with radiologist-level accuracy",
    #             "metrics": {
    #                 "accuracy": "96.5% sensitivity, 94.2% specificity",
    #                 "inference_time": "200ms per scan on A100",
    #                 "clinical_validation": "FDA cleared for diagnostic use"
    #             }
    #         },
    #         {
    #             "title": "Autonomous Vehicle Perception",
    #             "description": "Object detection for Level 2+ ADAS systems",
    #             "metrics": {
    #                 "detection_range": "Up to 150m",
    #                 "accuracy": "98% on KITTI benchmark",
    #                 "latency": "50ms on embedded GPU"
    #             }
    #         }
    #     ]
    # },

    # ============================================================
    # P1: Core Expansion (6 models)
    # ============================================================

    # "yolov5nu": {
    #     "display_name": "YOLOv5n-Ultralytics",
    #     "description": "YOLOv5 Nano - Ultra-lightweight detection model",
    #     "params": "1.9M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 64,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolov5", "nano", "lightweight", "fast"],
    #     "priority": 1,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 45.7,
    #         "coco_map50_95": 28.0,
    #         "inference_speed": "6.3ms (V100)",
    #     },
    #     "use_cases": [
    #         "Real-time detection on edge devices",
    #         "High-throughput video processing",
    #         "Resource-constrained environments",
    #     ],
    #     "pros": [
    #         "Extremely fast inference",
    #         "Very small model size",
    #         "Good balance of speed and accuracy",
    #     ],
    #     "cons": [
    #         "Lower accuracy than larger models",
    #         "May miss small objects",
    #     ],
    #     "when_to_use": "Use YOLOv5n for real-time detection on edge devices or when you need maximum speed.",
    #     "alternatives": [
    #         {"model": "yolov8n", "reason": "Newer, better accuracy"},
    #         {"model": "yolo11n", "reason": "Latest version"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 64, "range": [32, 128]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 100, "range": [50, 300]},
    #     },
    # },

    # "yolov5su": {
    #     "display_name": "YOLOv5s-Ultralytics",
    #     "description": "YOLOv5 Small - Balanced performance and accuracy",
    #     "params": "7.2M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 32,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolov5", "small", "balanced"],
    #     "priority": 1,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 56.8,
    #         "coco_map50_95": 37.4,
    #         "inference_speed": "9.2ms (V100)",
    #     },
    #     "use_cases": [
    #         "General-purpose object detection",
    #         "Balanced speed and accuracy",
    #         "Production deployments",
    #     ],
    #     "pros": [
    #         "Good accuracy for size",
    #         "Fast inference",
    #         "Well-tested and stable",
    #     ],
    #     "cons": [
    #         "Larger than nano version",
    #         "Slower than nano",
    #     ],
    #     "when_to_use": "Use YOLOv5s for production detection with balanced performance.",
    #     "alternatives": [
    #         {"model": "yolov8s", "reason": "Newer, better accuracy"},
    #         {"model": "yolov5nu", "reason": "Faster, smaller"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 32, "range": [16, 64]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 100, "range": [50, 300]},
    #     },
    # },

    # "yolov8n-seg": {
    #     "display_name": "YOLOv8n-Seg",
    #     "description": "YOLOv8 Nano Segmentation - Fast instance segmentation",
    #     "params": "3.4M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolov8", "segmentation", "nano", "instance"],
    #     "priority": 1,
    #     "task_types": ["instance_segmentation"],
    #     "benchmark": {
    #         "coco_map50": 52.3,
    #         "coco_map50_95": 36.7,
    #         "mask_map50_95": 30.5,
    #     },
    #     "use_cases": [
    #         "Fast instance segmentation",
    #         "Real-time mask generation",
    #         "Edge device segmentation",
    #     ],
    #     "pros": [
    #         "Fast segmentation inference",
    #         "Small model size",
    #         "Pixel-level masks",
    #     ],
    #     "cons": [
    #         "Lower accuracy than detection",
    #         "More compute than pure detection",
    #     ],
    #     "when_to_use": "Use YOLOv8n-seg for fast instance segmentation on edge devices.",
    #     "alternatives": [
    #         {"model": "yolov8s-seg", "reason": "Better accuracy"},
    #         {"model": "yolov8n", "reason": "Faster, detection only"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 16, "range": [8, 32]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 100, "range": [50, 200]},
    #     },
    # },

    # "yolov8s-seg": {
    #     "display_name": "YOLOv8s-Seg",
    #     "description": "YOLOv8 Small Segmentation - Accurate instance segmentation",
    #     "params": "11.8M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 8,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolov8", "segmentation", "small", "instance"],
    #     "priority": 1,
    #     "task_types": ["instance_segmentation"],
    #     "benchmark": {
    #         "coco_map50": 59.8,
    #         "coco_map50_95": 44.6,
    #         "mask_map50_95": 36.8,
    #     },
    #     "use_cases": [
    #         "High-accuracy instance segmentation",
    #         "Production segmentation tasks",
    #         "Detailed mask extraction",
    #     ],
    #     "pros": [
    #         "Better accuracy than nano",
    #         "Detailed segmentation masks",
    #         "Good for complex scenes",
    #     ],
    #     "cons": [
    #         "Slower than nano",
    #         "Higher memory usage",
    #     ],
    #     "when_to_use": "Use YOLOv8s-seg for production-grade instance segmentation.",
    #     "alternatives": [
    #         {"model": "yolov8n-seg", "reason": "Faster, smaller"},
    #         {"model": "yolov8m-seg", "reason": "Higher accuracy"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 8, "range": [4, 16]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 100, "range": [50, 200]},
    #     },
    # },

    # "yolov8n-pose": {
    #     "display_name": "YOLOv8n-Pose",
    #     "description": "YOLOv8 Nano Pose - Fast human pose estimation",
    #     "params": "3.3M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolov8", "pose", "keypoints", "nano"],
    #     "priority": 1,
    #     "task_types": ["pose_estimation"],
    #     "benchmark": {
    #         "coco_keypoint_ap": 50.4,
    #         "coco_keypoint_ap50": 80.1,
    #     },
    #     "use_cases": [
    #         "Real-time pose estimation",
    #         "Fitness and sports analysis",
    #         "Human activity recognition",
    #     ],
    #     "pros": [
    #         "Fast pose detection",
    #         "17 keypoints (COCO format)",
    #         "Good for single/multiple persons",
    #     ],
    #     "cons": [
    #         "Requires clear human visibility",
    #         "Lower accuracy than larger models",
    #     ],
    #     "when_to_use": "Use YOLOv8n-pose for real-time human pose estimation.",
    #     "alternatives": [
    #         {"model": "yolov8s-pose", "reason": "Better accuracy"},
    #         {"model": "yolov8n", "reason": "Detection only, faster"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 16, "range": [8, 32]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 200, "range": [100, 300]},
    #     },
    # },

    # "yolo11l": {
    #     "display_name": "YOLO11-Large",
    #     "description": "YOLO11 Large - Latest YOLO with high accuracy",
    #     "params": "25.3M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 8,
    #     "recommended_lr": 0.01,
    #     "tags": ["p1", "yolo11", "large", "sota", "2024"],
    #     "priority": 1,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 66.4,
    #         "coco_map50_95": 53.4,
    #         "inference_speed": "39.5ms (V100)",
    #     },
    #     "use_cases": [
    #         "High-accuracy detection",
    #         "Production systems",
    #         "Benchmark comparisons",
    #     ],
    #     "pros": [
    #         "State-of-the-art accuracy",
    #         "Latest YOLO architecture",
    #         "Good for complex scenes",
    #     ],
    #     "cons": [
    #         "Larger model size",
    #         "Slower inference",
    #         "More GPU memory needed",
    #     ],
    #     "when_to_use": "Use YOLO11l when you need maximum detection accuracy.",
    #     "alternatives": [
    #         {"model": "yolo11m", "reason": "Faster, slightly lower accuracy"},
    #         {"model": "yolov8m", "reason": "More stable, well-tested"},
    #     ],
    #     "recommended_settings": {
    #         "batch_size": {"value": 8, "range": [4, 16]},
    #         "learning_rate": {"value": 0.01, "range": [0.001, 0.1]},
    #         "epochs": {"value": 100, "range": [50, 300]},
    #     },
    # },

    # ============================================================
    # P2: Full Coverage (8 models)
    # Advanced detection, segmentation, and specialized YOLO variants
    # ============================================================

    # "yolov5mu": {
    #     "display_name": "YOLOv5m-Ultralytics",
    #     "description": "YOLOv5 Medium - Balanced detection model for general use",
    #     "params": "21.2M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 32,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov5", "medium", "balanced"],
    #     "priority": 2,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 64.1,
    #         "coco_map50_95": 45.4,
    #         "inference_speed": "12.3ms (V100)",
    #     },
    #     "use_cases": [
    #         "General object detection applications",
    #         "Production systems with moderate compute",
    #         "Multi-object tracking scenarios",
    #         "Real-time detection on mid-range hardware",
    #     ],
    #     "pros": [
    #         "Good balance accuracy/speed (45.4 mAP, 12.3ms)",
    #         "Versatile for various detection tasks",
    #         "Stable training and convergence",
    #         "Well-documented and widely used",
    #     ],
    #     "cons": [
    #         "Larger than nano/small variants (21.2M params)",
    #         "Slower than lightweight models",
    #         "Higher memory requirements",
    #     ],
    #     "when_to_use": "Choose YOLOv5m when you need better accuracy (45.4 mAP) than small models and have moderate compute resources.",
    #     "when_not_to_use": "Avoid for edge devices with limited resources or when ultra-fast inference (<8ms) is required.",
    #     "alternatives": [
    #         "YOLOv8m (newer, similar performance)",
    #         "YOLOv5s (faster, lower accuracy)",
    #         "YOLO11m (latest, better accuracy)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 100,
    #         "learning_rate": 0.01,
    #         "batch_size": 32,
    #         "optimizer": "SGD",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov5lu": {
    #     "display_name": "YOLOv5l-Ultralytics",
    #     "description": "YOLOv5 Large - High-accuracy detection for production systems",
    #     "params": "46.5M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov5", "large", "high-accuracy"],
    #     "priority": 2,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 67.3,
    #         "coco_map50_95": 49.0,
    #         "inference_speed": "19.5ms (V100)",
    #     },
    #     "use_cases": [
    #         "High-accuracy production detection systems",
    #         "Applications where accuracy > speed",
    #         "Complex multi-object scenarios",
    #         "Quality control and inspection systems",
    #     ],
    #     "pros": [
    #         "High detection accuracy (49.0 mAP)",
    #         "Robust to complex scenes",
    #         "Better small object detection",
    #         "Proven reliability in production",
    #     ],
    #     "cons": [
    #         "Large model size (46.5M params)",
    #         "Slower inference (19.5ms)",
    #         "High memory and compute requirements",
    #         "Longer training time",
    #     ],
    #     "when_to_use": "Choose YOLOv5l when accuracy is critical (49.0 mAP) and you have sufficient GPU resources for deployment.",
    #     "when_not_to_use": "Avoid for real-time applications on edge devices or when inference budget is <15ms.",
    #     "alternatives": [
    #         "YOLOv8l (newer, similar accuracy)",
    #         "YOLOv5m (faster, lower accuracy)",
    #         "YOLO11l (latest, better accuracy)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 150,
    #         "learning_rate": 0.01,
    #         "batch_size": 16,
    #         "optimizer": "SGD",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov8m": {
    #     "display_name": "YOLOv8m",
    #     "description": "YOLOv8 Medium - Modern detection with improved architecture",
    #     "params": "25.9M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 32,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov8", "medium", "modern"],
    #     "priority": 2,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 67.2,
    #         "coco_map50_95": 50.2,
    #         "inference_speed": "9.4ms (V100)",
    #     },
    #     "use_cases": [
    #         "Modern object detection systems",
    #         "Balanced accuracy/efficiency requirements",
    #         "Production deployment on cloud GPUs",
    #         "Transfer learning for custom datasets",
    #     ],
    #     "pros": [
    #         "Excellent accuracy/speed balance (50.2 mAP, 9.4ms)",
    #         "Modern anchor-free architecture",
    #         "Better training stability than YOLOv5",
    #         "Improved small object detection",
    #     ],
    #     "cons": [
    #         "More parameters than YOLOv5m (25.9M vs 21.2M)",
    #         "Requires more VRAM during training",
    #         "Slightly slower than YOLOv8s",
    #     ],
    #     "when_to_use": "Choose YOLOv8m when you need state-of-the-art detection accuracy (50.2 mAP) with reasonable inference speed.",
    #     "when_not_to_use": "Avoid for ultra-low latency applications (<8ms) or very limited compute resources.",
    #     "alternatives": [
    #         "YOLO11m (latest, similar performance)",
    #         "YOLOv8s (faster, lower accuracy)",
    #         "YOLOv5m (simpler, similar speed)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 100,
    #         "learning_rate": 0.01,
    #         "batch_size": 32,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov8l": {
    #     "display_name": "YOLOv8l",
    #     "description": "YOLOv8 Large - High-performance detection for demanding applications",
    #     "params": "43.7M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov8", "large", "high-performance"],
    #     "priority": 2,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 68.9,
    #         "coco_map50_95": 52.9,
    #         "inference_speed": "14.4ms (V100)",
    #     },
    #     "use_cases": [
    #         "State-of-the-art object detection",
    #         "Autonomous vehicles and robotics",
    #         "High-stakes surveillance systems",
    #         "Research and competition benchmarks",
    #     ],
    #     "pros": [
    #         "Excellent accuracy (52.9 mAP)",
    #         "State-of-the-art small object detection",
    #         "Robust to occlusion and complex scenes",
    #         "Strong generalization capability",
    #     ],
    #     "cons": [
    #         "Large model size (43.7M params)",
    #         "Relatively slow inference (14.4ms)",
    #         "High compute requirements",
    #         "Longer training time and convergence",
    #     ],
    #     "when_to_use": "Choose YOLOv8l when you need top-tier detection accuracy (52.9 mAP) and have sufficient GPU resources.",
    #     "when_not_to_use": "Avoid for real-time edge deployment or when inference latency must be <12ms.",
    #     "alternatives": [
    #         "YOLO11l (latest, better accuracy)",
    #         "YOLOv8m (faster, lower accuracy)",
    #         "YOLOv5x (even larger, similar accuracy)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 150,
    #         "learning_rate": 0.01,
    #         "batch_size": 16,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov8m-seg": {
    #     "display_name": "YOLOv8m-seg",
    #     "description": "YOLOv8 Medium Segmentation - Pixel-level object segmentation",
    #     "params": "27.3M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 16,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov8", "segmentation", "medium", "instance"],
    #     "priority": 2,
    #     "task_types": ["instance_segmentation"],
    #     "benchmark": {
    #         "coco_box_map50_95": 50.2,
    #         "coco_mask_map50_95": 41.0,
    #         "inference_speed": "15.8ms (V100)",
    #     },
    #     "use_cases": [
    #         "Instance segmentation for production",
    #         "Medical image analysis",
    #         "Scene understanding and parsing",
    #         "Robotics grasping and manipulation",
    #     ],
    #     "pros": [
    #         "Good segmentation accuracy (41.0 mAP mask)",
    #         "Unified detection + segmentation",
    #         "Faster than separate models",
    #         "Modern anchor-free design",
    #     ],
    #     "cons": [
    #         "Slower than detection-only models",
    #         "Higher memory usage for masks",
    #         "More complex training process",
    #     ],
    #     "when_to_use": "Choose YOLOv8m-seg when you need pixel-accurate object boundaries (41.0 mAP) with reasonable inference speed.",
    #     "when_not_to_use": "Avoid if you only need bounding boxes (use YOLOv8m instead) or for ultra-fast applications (<12ms).",
    #     "alternatives": [
    #         "YOLOv8n-seg (faster, lower accuracy)",
    #         "YOLOv8s-seg (balanced)",
    #         "Mask R-CNN (slower, more accurate)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 150,
    #         "learning_rate": 0.01,
    #         "batch_size": 16,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolo11m": {
    #     "display_name": "YOLO11m",
    #     "description": "YOLO11 Medium - Latest YOLO with enhanced architecture (2024)",
    #     "params": "20.1M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 32,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolo11", "latest", "medium", "2024"],
    #     "priority": 2,
    #     "task_types": ["object_detection"],
    #     "benchmark": {
    #         "coco_map50": 68.0,
    #         "coco_map50_95": 51.5,
    #         "inference_speed": "8.9ms (V100)",
    #     },
    #     "use_cases": [
    #         "Cutting-edge object detection systems",
    #         "Research and development projects",
    #         "Production systems needing latest tech",
    #         "Benchmarking and comparison studies",
    #     ],
    #     "pros": [
    #         "State-of-the-art accuracy (51.5 mAP)",
    #         "Fast inference for size (8.9ms, 20.1M params)",
    #         "Latest architecture improvements",
    #         "Better efficiency than YOLOv8m",
    #     ],
    #     "cons": [
    #         "Newer model (less battle-tested)",
    #         "Smaller community and resources",
    #         "May have undiscovered edge cases",
    #     ],
    #     "when_to_use": "Choose YOLO11m when you want the latest YOLO improvements (51.5 mAP) and can adopt newer technology.",
    #     "when_not_to_use": "Avoid if you need proven stability and extensive community support. Use YOLOv8m instead.",
    #     "alternatives": [
    #         "YOLOv8m (more stable, similar accuracy)",
    #         "YOLO11s (faster, lower accuracy)",
    #         "YOLO11l (higher accuracy)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 100,
    #         "learning_rate": 0.01,
    #         "batch_size": 32,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov8x-pose": {
    #     "display_name": "YOLOv8x-pose",
    #     "description": "YOLOv8 XLarge Pose - High-accuracy human pose estimation",
    #     "params": "68.2M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 8,
    #     "recommended_lr": 0.01,
    #     "tags": ["p2", "yolov8", "pose", "xlarge", "keypoint"],
    #     "priority": 2,
    #     "task_types": ["pose_estimation"],
    #     "benchmark": {
    #         "coco_pose_map50_95": 70.4,
    #         "inference_speed": "28.4ms (V100)",
    #     },
    #     "use_cases": [
    #         "High-accuracy pose estimation",
    #         "Sports analytics and coaching",
    #         "Medical rehabilitation monitoring",
    #         "Action recognition systems",
    #     ],
    #     "pros": [
    #         "State-of-the-art pose accuracy (70.4 mAP)",
    #         "Robust to occlusion and complex poses",
    #         "Multi-person pose estimation",
    #         "Pre-trained on COCO Keypoints",
    #     ],
    #     "cons": [
    #         "Very large model (68.2M params)",
    #         "Slow inference (28.4ms)",
    #         "High memory requirements",
    #         "Requires powerful GPU for training",
    #     ],
    #     "when_to_use": "Choose YOLOv8x-pose when you need the highest pose estimation accuracy (70.4 mAP) and have sufficient GPU resources.",
    #     "when_not_to_use": "Avoid for real-time applications on edge devices. Use YOLOv8n-pose or YOLOv8s-pose instead.",
    #     "alternatives": [
    #         "YOLOv8n-pose (ultra-fast, lower accuracy)",
    #         "YOLOv8s-pose (balanced)",
    #         "OpenPose (more keypoints, slower)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 200,
    #         "learning_rate": 0.01,
    #         "batch_size": 8,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    # "yolov8x-worldv2": {
    #     "display_name": "YOLO-World v2",
    #     "description": "YOLO-World v2 - Zero-shot object detection with language prompts",
    #     "params": "93.7M",
    #     "input_size": 640,
    #     "pretrained_available": True,
    #     "recommended_batch_size": 8,
    #     "recommended_lr": 0.001,
    #     "tags": ["p2", "zero-shot", "vision-language", "prompt-based", "research"],
    #     "priority": 2,
    #     "task_types": ["zero_shot_detection"],
    #     "benchmark": {
    #         "lvis_map50_95": 35.4,
    #         "inference_speed": "45.2ms (V100)",
    #     },
    #     "use_cases": [
    #         "Zero-shot detection (no training data)",
    #         "Open-vocabulary object detection",
    #         "Rapid prototyping for new classes",
    #         "Research in vision-language models",
    #     ],
    #     "pros": [
    #         "Detect objects not in training set",
    #         "Text-prompt based detection",
    #         "No need for labeled training data",
    #         "Strong generalization to novel classes",
    #     ],
    #     "cons": [
    #         "Very large model (93.7M params)",
    #         "Slow inference (45.2ms)",
    #         "Lower accuracy than specialized models",
    #         "Requires careful prompt engineering",
    #     ],
    #     "when_to_use": "Choose YOLO-World when you need to detect arbitrary objects via text prompts without training data.",
    #     "when_not_to_use": "Avoid for standard detection tasks with labeled data. Use YOLOv8/YOLO11 instead for better accuracy/speed.",
    #     "alternatives": [
    #         "GroundingDINO (better zero-shot, slower)",
    #         "OWL-ViT (Transformer-based zero-shot)",
    #         "YOLOv8x (standard detection, much faster)",
    #     ],
    #     "recommended_settings": {
    #         "epochs": 50,
    #         "learning_rate": 0.001,
    #         "batch_size": 8,
    #         "optimizer": "AdamW",
    #         "scheduler": "Cosine",
    #     },
    # },

    "sam2_t": {
        "display_name": "SAM2 Tiny",
        "description": "Segment Anything Model 2 (Meta, Jul 2024) - Zero-shot segmentation with prompts",
        "params": "38.9M",
        "input_size": 1024,
        "task_types": ["zero_shot_segmentation"],
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 0.0001,
        "tags": ["p0", "sam2", "zero-shot", "prompt-based", "meta", "2024", "cvpr2024"],
        "priority": 0,

        "status": "active",
        "inference_only": True,  # Primarily for inference
        "recommended": True,
        "performance_tier": "accurate",

        "benchmark": {
            "sa-v_iou": 75.2,  # Segment Anything V benchmark
            "inference_speed_v100": "~50ms per image",
            "inference_speed_unit": "ms",
            "model_size_mb": 158,
            "prompt_types": "point, box, mask, text",
        },

        "use_cases": [
            "Zero-shot segmentation without training",
            "Interactive annotation tools",
            "Prompt-based object segmentation",
            "Medical image analysis",
            "Video object segmentation",
            "Data annotation and labeling"
        ],

        "pros": [
            "Latest SAM2 from Meta (July 2024)",
            "Zero-shot segmentation (no training needed)",
            "Multiple prompt types (point/box/mask/text)",
            "High-quality segmentation masks",
            "Generalizes to any object class",
            "Video segmentation support"
        ],

        "cons": [
            "Large model size (158 MB)",
            "Slower than YOLO segmentation",
            "Inference-only (not trainable)",
            "Requires GPU for real-time use",
            "Higher memory requirements"
        ],

        "when_to_use": "Use SAM2 when you need high-quality segmentation for arbitrary objects without training, especially for annotation tools or zero-shot tasks.",

        "when_not_to_use": "Avoid if you need real-time performance, have limited GPU memory, or already have labeled data for specific classes (use YOLO11n-seg instead).",

        "alternatives": [
            {
                "model": "yolo11n-seg",
                "reason": "Much faster, trainable, but needs labeled data"
            },
            {
                "model": "sam2_b",
                "reason": "Higher accuracy SAM2 variant (larger)"
            },
            {
                "model": "nvidia/segformer",
                "reason": "Semantic segmentation (HuggingFace)"
            }
        ],

        "recommended_settings": {
            "batch_size": {
                "value": 8,
                "range": [1, 16],
                "note": "SAM2 requires significant GPU memory"
            },
            "image_size": {
                "value": 1024,
                "range": [512, 1024],
                "note": "SAM2 works best at 1024x1024"
            },
            "prompt_type": "point, box, or mask",
            "note": "Inference-only model, no training parameters"
        },

        "special_features": {
            "type": "zero_shot_segmentation",
            "capabilities": [
                "Segment any object with point/box prompts",
                "Interactive segmentation refinement",
                "Video object tracking and segmentation",
                "Automatic mask generation for all objects",
                "Multi-prompt combination (point + box)"
            ],
            "example_prompts": [
                "Click on the object to segment",
                "Draw a box around the target",
                "Provide a rough mask outline"
            ],
            "usage_example": {
                "point_prompt": "predictor.predict(point_coords=[[500, 375]], point_labels=[1])",
                "box_prompt": "predictor.predict(box=[425, 600, 700, 875])",
                "mask_prompt": "predictor.predict(mask_input=low_res_mask)"
            },
            "prompt_engineering_tips": [
                "Use positive points (label=1) for object, negative (label=0) for background",
                "Box prompts are more stable than single points",
                "Combine multiple points for complex shapes",
                "Use previous mask as input for refinement"
            ]
        },

        "real_world_examples": [
            {
                "title": "Medical Image Annotation",
                "description": "Radiologists use SAM2 to quickly annotate organs and tumors in CT scans",
                "metrics": {
                    "annotation_speed": "10x faster than manual",
                    "accuracy": "95% IoU"
                }
            },
            {
                "title": "Retail Product Segmentation",
                "description": "E-commerce platforms use SAM2 to extract product images from photos",
                "metrics": {
                    "processing_time": "< 1s per image",
                    "quality": "Production-ready masks"
                }
            }
        ]
    },
}


def get_ultralytics_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model metadata by model name."""
    return ULTRALYTICS_MODEL_REGISTRY.get(model_name)


def list_ultralytics_models(
    tags: Optional[List[str]] = None,
    priority: Optional[int] = None,
    task_type: Optional[str] = None
) -> List[str]:
    """
    List Ultralytics models, optionally filtered.

    Args:
        tags: Filter by tags (e.g., ["p0", "yolo11"])
        priority: Filter by priority (0, 1, or 2)
        task_type: Filter by task type

    Returns:
        List of model names
    """
    models = []

    for name, info in ULTRALYTICS_MODEL_REGISTRY.items():
        # Filter by priority
        if priority is not None and info.get("priority") != priority:
            continue

        # Filter by task type
        if task_type and info.get("task_type") != task_type:
            continue

        # Filter by tags
        if tags:
            model_tags = info.get("tags", [])
            if not any(tag in model_tags for tag in tags):
                continue

        models.append(name)

    return models

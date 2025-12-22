"""
OpenMMLab Trainer Utilities

Helper functions for OpenMMLab trainers.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def get_mmdet_config_path(model_name: str) -> str:
    """
    Get MMDetection config file path for given model.

    Args:
        model_name: Model name

    Returns:
        Config file path

    Raises:
        ValueError: If model not supported
    """
    model_config_map = {
        # Faster R-CNN family
        'faster-rcnn-r50': 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'faster-rcnn-r101': 'faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py',

        # Mask R-CNN family
        'mask-rcnn-r50': 'mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
        'mask-rcnn-r101': 'mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py',

        # RetinaNet
        'retinanet-r50': 'retinanet/retinanet_r50_fpn_1x_coco.py',

        # FCOS
        'fcos-r50': 'fcos/fcos_r50_fpn_1x_coco.py',

        # YOLOX
        'yolox-s': 'yolox/yolox_s_8xb8-300e_coco.py',
        'yolox-m': 'yolox/yolox_m_8xb8-300e_coco.py',
        'yolox-l': 'yolox/yolox_l_8xb8-300e_coco.py',

        # RTMDet
        'rtmdet-s': 'rtmdet/rtmdet_s_8xb32-300e_coco.py',
        'rtmdet-m': 'rtmdet/rtmdet_m_8xb32-300e_coco.py',
        'rtmdet-l': 'rtmdet/rtmdet_l_8xb32-300e_coco.py',
    }

    config_path = model_config_map.get(model_name)
    if not config_path:
        raise ValueError(f"Unsupported model: {model_name}")

    return config_path


def parse_coco_metrics(eval_results: Dict) -> Dict[str, float]:
    """
    Parse COCO evaluation results to standard format.

    Args:
        eval_results: Raw evaluation results from MMDetection

    Returns:
        Standardized metrics
    """
    return {
        'mAP': eval_results.get('bbox_mAP', 0.0),
        'mAP50': eval_results.get('bbox_mAP_50', 0.0),
        'mAP75': eval_results.get('bbox_mAP_75', 0.0),
        'mAP_small': eval_results.get('bbox_mAP_s', 0.0),
        'mAP_medium': eval_results.get('bbox_mAP_m', 0.0),
        'mAP_large': eval_results.get('bbox_mAP_l', 0.0),
    }


def validate_config(config: Dict) -> bool:
    """
    Validate training configuration.

    Args:
        config: Training configuration

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    basic = config.get('basic', {})

    # Validate epochs
    epochs = basic.get('epochs', 12)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"Invalid epochs: {epochs}")

    # Validate batch size
    batch = basic.get('batch', 2)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError(f"Invalid batch size: {batch}")

    # Validate learning rate
    lr = basic.get('lr0', 0.02)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"Invalid learning rate: {lr}")

    return True


def get_class_names_from_coco(annotations_file: str) -> List[str]:
    """
    Extract class names from COCO annotations file.

    Args:
        annotations_file: Path to annotations JSON

    Returns:
        List of class names
    """
    import json

    with open(annotations_file) as f:
        data = json.load(f)

    categories = data.get('categories', [])
    # Filter out __background__
    real_categories = [cat for cat in categories if cat.get('name') != '__background__']
    return [cat['name'] for cat in real_categories]

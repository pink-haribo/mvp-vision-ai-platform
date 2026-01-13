"""
VFM-v1 Default Config

This module provides the MMEngine Config for VFM training.
Loads base config directly and merges custom settings - no lazy import issues.

Usage:
    from default_config import get_vfm_config

    cfg = get_vfm_config(
        model_name='vfm_v1_l',
        dataset_dir='/path/to/dataset',
        work_dir='/path/to/work_dir',
        ...
    )
    # cfg is a ready-to-use MMEngine Config object
"""

import sys
from pathlib import Path
from typing import List, Tuple

from mmengine.config import Config


# VFM-v1 root directory (where this file is located)
VFM_ROOT = Path(__file__).parent.resolve()

# Base config path (absolute)
BASE_CONFIG_PATH = VFM_ROOT / 'third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'


def get_vfm_config(
    model_name: str,
    dataset_dir: str,
    work_dir: str,
    ann_file_train: str,
    ann_file_val: str,
    img_prefix: str,
    texts_file: str,
    class_names: List[str],
    num_classes: int,
    pretrained_path: str,
    # Training parameters
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    weight_decay: float = 0.05,
    val_interval: int = 10,
    save_epoch_intervals: int = 20,
    close_mosaic_epochs: int = 2,
    img_scale: Tuple[int, int] = (640, 640),
    text_model_name: str = 'openai/clip-vit-base-patch32',
) -> Config:
    """
    Generate VFM MMEngine Config object.

    Loads base config directly using Config.fromfile() and merges
    custom settings. This avoids lazy import / relative path issues.

    Args:
        model_name: Model name (e.g., 'vfm_v1_l')
        dataset_dir: Path to dataset directory
        work_dir: Working directory for outputs
        ann_file_train: Path to training annotation file
        ann_file_val: Path to validation annotation file
        img_prefix: Path to images directory
        texts_file: Path to class text prompts JSON
        class_names: List of class names
        num_classes: Number of classes
        pretrained_path: Path to pretrained weights
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
        val_interval: Validation interval (epochs)
        save_epoch_intervals: Checkpoint save interval
        close_mosaic_epochs: Epochs before end to close mosaic
        img_scale: Image scale (height, width)
        text_model_name: CLIP text encoder model name

    Returns:
        MMEngine Config object ready for Runner.from_cfg()
    """
    # Load base config directly (no lazy import)
    cfg = Config.fromfile(str(BASE_CONFIG_PATH))

    # Get values from base config
    last_stage_out_channels = cfg.last_stage_out_channels
    pre_transform = cfg.pre_transform
    copypaste_prob = cfg.copypaste_prob
    affine_scale = cfg.affine_scale
    min_area_ratio = cfg.min_area_ratio
    use_mask2refine = cfg.use_mask2refine
    mixup_prob = cfg.mixup_prob
    last_transform = cfg.last_transform
    train_pipeline_stage2_base = cfg.train_pipeline_stage2
    test_pipeline_base = cfg.test_pipeline
    val_interval_stage2 = cfg.val_interval_stage2
    base_model_backbone = cfg.model.backbone

    # Custom imports for YOLO World
    cfg.custom_imports = dict(
        imports=['yolo_world'],
        allow_failed_imports=False
    )

    # Hyper-parameters
    text_channels = 512
    neck_embed_channels = [128, 256, last_stage_out_channels // 2]
    neck_num_heads = [4, 8, last_stage_out_channels // 2 // 32]

    # Dataset metadata
    metainfo_classes = dict(classes=tuple(class_names))

    # Model settings
    cfg.model = dict(
        type='YOLOWorldDetector',
        mm_neck=True,
        num_train_classes=num_classes,
        num_test_classes=num_classes,
        data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
        backbone=dict(
            type='MultiModalYOLOBackbone',
            image_model=base_model_backbone,
            text_model=dict(
                type='HuggingCLIPLanguageBackbone',
                model_name=text_model_name,
                frozen_modules=['all']
            )
        ),
        neck=dict(
            type='YOLOWorldPAFPN',
            in_channels=[256, 512, last_stage_out_channels],
            out_channels=[256, 512, last_stage_out_channels],
            guide_channels=text_channels,
            embed_channels=neck_embed_channels,
            num_heads=neck_num_heads,
            block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')
        ),
        bbox_head=dict(
            type='YOLOWorldHead',
            head_module=dict(
                type='YOLOWorldHeadModule',
                use_bn_head=True,
                embed_dims=text_channels,
                num_classes=num_classes,
                in_channels=[256, 512, last_stage_out_channels]
            )
        ),
        train_cfg=dict(assigner=dict(num_classes=num_classes))
    )

    # Dataset transforms
    text_transform = [
        dict(
            type='RandomLoadText',
            num_neg_samples=(num_classes, num_classes),
            max_num_samples=num_classes,
            padding_to_max=True,
            padding_value=''
        ),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                       'flip_direction', 'texts')
        )
    ]

    mosaic_affine_transform = [
        dict(
            type='MultiModalMosaic',
            img_scale=img_scale,
            pad_val=114.0,
            pre_transform=pre_transform
        ),
        dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_aspect_ratio=100.,
            scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
            border=(-img_scale[0] // 2, -img_scale[1] // 2),
            border_val=(114, 114, 114),
            min_area_ratio=min_area_ratio,
            use_mask_refine=use_mask2refine
        )
    ]

    train_pipeline = [
        *pre_transform,
        *mosaic_affine_transform,
        dict(
            type='YOLOv5MultiModalMixUp',
            prob=mixup_prob,
            pre_transform=[*pre_transform, *mosaic_affine_transform]
        ),
        *last_transform[:-1],
        *text_transform
    ]

    train_pipeline_stage2 = [
        *train_pipeline_stage2_base[:-1],
        *text_transform
    ]

    test_pipeline = [
        *test_pipeline_base[:-1],
        dict(type='LoadText'),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'pad_param', 'texts')
        )
    ]

    # Train dataset
    cfg.train_dataloader = dict(
        persistent_workers=True,
        batch_size=batch_size,
        collate_fn=dict(type='yolow_collate'),
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='YOLOv5CocoDataset',
                metainfo=metainfo_classes,
                data_root=dataset_dir,
                ann_file=ann_file_train,
                data_prefix=dict(img=img_prefix),
                filter_cfg=dict(filter_empty_gt=False, min_size=32)
            ),
            class_text_path=texts_file,
            pipeline=train_pipeline
        )
    )

    # Val dataset
    cfg.val_dataloader = dict(
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='YOLOv5CocoDataset',
                metainfo=metainfo_classes,
                data_root=dataset_dir,
                ann_file=ann_file_val,
                data_prefix=dict(img=img_prefix),
                filter_cfg=dict(filter_empty_gt=False, min_size=32)
            ),
            class_text_path=texts_file,
            pipeline=test_pipeline
        )
    )
    cfg.test_dataloader = cfg.val_dataloader

    # Training settings
    cfg.default_hooks = dict(
        param_scheduler=dict(
            scheduler_type='linear',
            lr_factor=0.01,
            max_epochs=epochs
        ),
        checkpoint=dict(
            max_keep_ckpts=-1,
            save_best='coco/bbox_mAP',
            rule='greater',
            interval=save_epoch_intervals
        )
    )

    cfg.custom_hooks = [
        dict(
            type='EMAHook',
            ema_type='ExpMomentumEMA',
            momentum=0.0001,
            update_buffers=True,
            strict_load=False,
            priority=49
        ),
        dict(
            type='mmdet.PipelineSwitchHook',
            switch_epoch=epochs - close_mosaic_epochs,
            switch_pipeline=train_pipeline_stage2
        )
    ]

    cfg.train_cfg = dict(
        max_epochs=epochs,
        val_interval=val_interval,
        dynamic_intervals=[((epochs - close_mosaic_epochs), val_interval_stage2)]
    )

    cfg.optim_wrapper = dict(
        optimizer=dict(
            type='AdamW',
            lr=learning_rate,
            weight_decay=weight_decay,
            batch_size_per_gpu=batch_size
        ),
        paramwise_cfg=dict(
            custom_keys={
                'backbone.text_model': dict(lr_mult=0.01),
                'logit_scale': dict(weight_decay=0.0)
            }
        ),
        constructor='YOLOWv5OptimizerConstructor'
    )

    # Evaluation settings
    cfg.val_evaluator = dict(
        type='mmdet.CocoMetric',
        proposal_nums=(100, 1, 10),
        ann_file=ann_file_val,
        metric='bbox',
        classwise=True
    )
    cfg.test_evaluator = cfg.val_evaluator

    # Pretrained model and work directory
    cfg.load_from = pretrained_path
    cfg.work_dir = work_dir

    return cfg

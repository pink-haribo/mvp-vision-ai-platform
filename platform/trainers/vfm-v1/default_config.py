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

    # Model architecture parameters (for L model)
    deepen_factor = 1.0
    widen_factor = 1.0
    last_stage_out_channels = 512
    text_channels = 512
    neck_embed_channels = [128, 256, 256]
    neck_num_heads = [4, 8, 8]
    strides = [8, 16, 32]

    # Common configs
    norm_cfg = dict(type='BN', eps=0.001, momentum=0.03)
    act_cfg = dict(type='SiLU', inplace=True)

    # Loss weights
    loss_cls_weight = 0.5
    loss_bbox_weight = 7.5
    loss_dfl_weight = 0.375

    # Task-aligned assigner
    tal_topk = 10
    tal_alpha = 0.5
    tal_beta = 6.0

    # Augmentation parameters
    affine_scale = 0.9
    mixup_prob = 0.15
    copypaste_prob = 0.3
    min_area_ratio = 0.01
    max_aspect_ratio = 100.0
    use_mask2refine = True

    # Custom imports for YOLO World
    cfg.custom_imports = dict(
        imports=['yolo_world'],
        allow_failed_imports=False
    )

    # Dataset metadata
    metainfo_classes = dict(classes=tuple(class_names))

    # Pre-transform pipeline
    pre_transform = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True, mask2bbox=True)
    ]

    # Albu augmentations
    albu_train_transforms = [
        dict(type='Blur', p=0.01),
        dict(type='MedianBlur', p=0.01),
        dict(type='ToGray', p=0.01),
        dict(type='CLAHE', p=0.01),
    ]

    # Text transform for multimodal
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

    # Last transform (before text_transform)
    last_transform = [
        dict(type='RemoveDataElement', keys=['gt_masks']),
        dict(
            type='mmdet.Albu',
            transforms=albu_train_transforms,
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']
            ),
            keymap=dict(img='image', gt_bboxes='bboxes')
        ),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
    ]

    # Mosaic + Affine transform
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
            max_aspect_ratio=max_aspect_ratio,
            scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
            border=(-img_scale[0] // 2, -img_scale[1] // 2),
            border_val=(114, 114, 114),
            min_area_ratio=min_area_ratio,
            use_mask_refine=use_mask2refine
        )
    ]

    # Full train pipeline
    train_pipeline = [
        *pre_transform,
        *mosaic_affine_transform,
        dict(
            type='YOLOv5MultiModalMixUp',
            prob=mixup_prob,
            pre_transform=[*pre_transform, *mosaic_affine_transform]
        ),
        *last_transform,
        *text_transform
    ]

    # Stage 2 train pipeline (no mosaic/mixup)
    train_pipeline_stage2 = [
        *pre_transform,
        dict(type='YOLOv5KeepRatioResize', scale=img_scale),
        dict(
            type='LetterResize',
            scale=img_scale,
            allow_scale_up=True,
            pad_val=dict(img=114.0)
        ),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_aspect_ratio=max_aspect_ratio,
            scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
            border_val=(114, 114, 114),
            min_area_ratio=min_area_ratio,
            use_mask_refine=use_mask2refine
        ),
        *last_transform,
        *text_transform
    ]

    # Test/Val pipeline
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='YOLOv5KeepRatioResize', scale=img_scale),
        dict(
            type='LetterResize',
            scale=img_scale,
            allow_scale_up=False,
            pad_val=dict(img=114)
        ),
        dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
        dict(type='LoadText'),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'pad_param', 'texts')
        )
    ]

    # ============================================================
    # Model Configuration (YOLO World)
    # ============================================================
    cfg.model = dict(
        type='YOLOWorldDetector',
        mm_neck=True,
        num_train_classes=num_classes,
        num_test_classes=num_classes,
        data_preprocessor=dict(
            type='YOLOWDetDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True
        ),
        backbone=dict(
            type='MultiModalYOLOBackbone',
            image_model=dict(
                type='YOLOv8CSPDarknet',
                arch='P5',
                last_stage_out_channels=last_stage_out_channels,
                deepen_factor=deepen_factor,
                widen_factor=widen_factor,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
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
            num_csp_blocks=3,
            block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        ),
        bbox_head=dict(
            type='YOLOWorldHead',
            head_module=dict(
                type='YOLOWorldHeadModule',
                num_classes=num_classes,
                in_channels=[256, 512, last_stage_out_channels],
                embed_dims=text_channels,
                featmap_strides=strides,
                reg_max=16,
                use_bn_head=True,
                widen_factor=widen_factor,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='none',
                loss_weight=loss_cls_weight
            ),
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                reduction='sum',
                loss_weight=loss_bbox_weight,
                return_iou=False
            ),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=loss_dfl_weight
            ),
            prior_generator=dict(
                type='mmdet.MlvlPointGenerator',
                offset=0.5,
                strides=strides
            )
        ),
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=num_classes,
                topk=tal_topk,
                alpha=tal_alpha,
                beta=tal_beta,
                eps=1e-9,
                use_ciou=True
            )
        ),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300
        )
    )

    # ============================================================
    # Dataset Configuration
    # ============================================================
    cfg.train_dataloader = dict(
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='yolow_collate'),
        dataset=dict(
            _delete_=True,
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

    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            _delete_=True,
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

    # ============================================================
    # Training Configuration
    # ============================================================
    cfg.default_scope = 'mmyolo'

    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(
            type='YOLOv5ParamSchedulerHook',
            scheduler_type='linear',
            lr_factor=0.01,
            max_epochs=epochs
        ),
        checkpoint=dict(
            type='CheckpointHook',
            interval=save_epoch_intervals,
            max_keep_ckpts=-1,
            save_best=None
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='mmdet.DetVisualizationHook')
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
        type='EpochBasedTrainLoop',
        max_epochs=epochs,
        val_interval=val_interval,
        dynamic_intervals=[(epochs - close_mosaic_epochs, 1)]
    )
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')

    cfg.optim_wrapper = dict(
        type='AmpOptimWrapper',
        loss_scale='dynamic',
        clip_grad=dict(max_norm=10.0),
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

    cfg.param_scheduler = None

    # ============================================================
    # Evaluation Configuration
    # ============================================================
    cfg.val_evaluator = dict(
        type='mmdet.CocoMetric',
        ann_file=f'{dataset_dir}/{ann_file_val}',
        metric='bbox',
        classwise=True,
        proposal_nums=(100, 1, 10)
    )
    cfg.test_evaluator = cfg.val_evaluator

    # ============================================================
    # Environment Configuration
    # ============================================================
    cfg.env_cfg = dict(
        cudnn_benchmark=True,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl')
    )

    cfg.log_level = 'INFO'
    cfg.log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

    cfg.visualizer = dict(
        type='mmdet.DetLocalVisualizer',
        name='visualizer',
        vis_backends=[dict(type='LocalVisBackend')]
    )

    cfg.vis_backends = [dict(type='LocalVisBackend')]

    # ============================================================
    # Load & Save Configuration
    # ============================================================
    cfg.load_from = pretrained_path
    cfg.work_dir = work_dir
    cfg.resume = False

    return cfg

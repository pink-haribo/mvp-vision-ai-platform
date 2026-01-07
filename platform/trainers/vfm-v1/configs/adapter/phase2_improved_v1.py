# Phase 2 Improved V1 - Tier 1 개선사항 적용
# 1. Adapter capacity 증가 (reduction_ratio: 4->2)
# 2. Learning rate 증가 (1e-4 -> 5e-4)
# 3. 더 많은 레이어 unfreeze (마지막 2개)
# 4. 학습 epoch 증가 (50 -> 100)

# _base_ = '../finetune_coco/vfm_v1_l_mvtec.py'
_base_ = './vfm_v1_l_mvtec.py'

# Adapter settings - IMPROVED
use_adapter = True
adapter_type = 'HierarchicalAdapter'
adapter_reduction_ratio = 2  # 4 -> 2 (4배 파라미터 증가)
adapter_num_heads = 8
adapter_mlp_ratio = 4
adapter_positions = ['top_down', 'bottom_up']

# Phase 2: Unfreeze MORE layers - IMPROVED
unfreeze_neck_patterns = [
    'top_down_layers.1',   # 마지막 2개 (기존: 1개)
    'top_down_layers.2',
    'bottom_up_layers.1',  # 마지막 2개 (기존: 1개)
    'bottom_up_layers.2',
    'out_layers.1',        # 마지막 2개 (기존: 1개)
    'out_layers.2',
]

# Training settings - IMPROVED
max_epochs = 100  # 50 -> 100
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 5e-4  # 1e-4 -> 5e-4 (5배 증가)

# Load from pretrained model
load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
resume = False

# Model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=_base_.num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=4),
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            reduction_ratio=adapter_reduction_ratio,  # IMPROVED: 2
            num_heads=adapter_num_heads,
            mlp_ratio=adapter_mlp_ratio),
        adapter_positions=adapter_positions,
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),  # IMPROVED: 더 많은 레이어
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            freeze_all=True)),
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer - IMPROVED LR
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Adapters: VERY HIGH LR (2x base = 1e-3)
            'adapter': dict(lr_mult=2.0),
            
            # Unfrozen neck layers: HIGH LR (1x base = 5e-4)
            'top_down_layers.1': dict(lr_mult=1.0),
            'top_down_layers.2': dict(lr_mult=1.0),
            'bottom_up_layers.1': dict(lr_mult=1.0),
            'bottom_up_layers.2': dict(lr_mult=1.0),
            'out_layers.1': dict(lr_mult=1.0),
            'out_layers.2': dict(lr_mult=1.0),
        },
        bias_lr_mult=2.0,
        bias_decay_mult=0.0,
        norm_decay_mult=0.0))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Runtime settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))

# DDP settings
find_unused_parameters = True

# Visualization
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]


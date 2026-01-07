# Phase 2 - Option 1 (BottleneckAdapter) - Strategy A (Neck only + unfreeze)
# Load from pretrained epoch_100.pth and train adapters + selected layers

_base_ = '../finetune_coco/vfm_v1_l_mvtec.py'

# Adapter settings
use_adapter = True
adapter_type = 'BottleneckAdapter'  # Option 1: Simple bottleneck
adapter_reduction_ratio = 4
adapter_positions = ['top_down', 'bottom_up']  # Strategy A: Neck only

# Phase 2: Unfreeze specific layers in Neck
unfreeze_neck_patterns = [
    'top_down_layers.2',   # Last top-down layer
    'bottom_up_layers.2',  # Last bottom-up layer
    'out_layers.2',        # Last output layer
]

# Training settings
max_epochs = 50
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 1e-4  # Base learning rate

# Load from pretrained model (NOT Phase 1)
load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
resume = False

# Model settings with adapter + Phase 2 unfreezing
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
        frozen_stages=4),  # Freeze all backbone stages
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',  # Use adapter-enabled neck
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            reduction_ratio=adapter_reduction_ratio),
        adapter_positions=adapter_positions,
        freeze_all=True,  # Freeze original neck parameters
        unfreeze_patterns=unfreeze_neck_patterns),  # Phase 2: Unfreeze selected layers
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            freeze_all=True)),  # Freeze head
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer with different learning rates for different parts
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Adapters: high LR (base_lr)
            'adapter': dict(lr_mult=1.0),
            
            # Unfrozen neck layers: medium LR (0.5 * base_lr)
            'top_down_layers.2': dict(lr_mult=0.5),
            'bottom_up_layers.2': dict(lr_mult=0.5),
            'out_layers.2': dict(lr_mult=0.5),
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
        max_keep_ckpts=3,
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


# Selective Fine-tuning V1 - Adapter 없이 선택적 레이어만 fine-tuning
# Backbone freeze + Neck full fine-tuning + Head full fine-tuning
# 
# 이 방법은 adapter의 제약 없이 더 빠르고 효과적일 수 있습니다.
# MVTec은 작은 데이터셋이므로 overfitting 위험이 낮습니다.

# _base_ = '../finetune_coco/vfm_v1_l_mvtec.py'
_base_ = './vfm_v1_l_mvtec.py'

# Training settings
max_epochs = 100
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 2e-4  # Moderate LR for fine-tuning

# Load from pretrained model
load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
resume = False

# Model settings - NO ADAPTER
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
        type='YOLOWorldPAFPN',  # NO ADAPTER - original neck
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',  # NO ADAPTER - original head
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer with layer-wise LR
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Backbone: FROZEN (lr_mult=0)
            'backbone.image_model': dict(lr_mult=0.0),
            'backbone.text_model': dict(lr_mult=0.0),
            
            # Neck: FULL FINE-TUNING (lr_mult=1.0)
            'neck': dict(lr_mult=1.0),
            
            # Head: FULL FINE-TUNING (lr_mult=1.0)
            'bbox_head': dict(lr_mult=1.0),
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


# Phase 2 Dense LoRA V1 - SAM2-UNet 스타일 Dense LoRA
# 모든 주요 위치에 LoRA 추가: Backbone + Neck + Head
# 
# 전략:
# - Backbone (stage3, stage4): rank=8 (작게, feature extraction)
# - Neck (all layers): rank=16 (중간, feature fusion)
# - Head (cls, reg): rank=32 (크게, task-specific)
# - ImagePoolingAttention: rank=32 (크게, text-image fusion)

_base_ = '../adapter/vfm_v1_l_mvtec.py'

# LoRA settings - Multi-scale ranks
use_adapter = True
adapter_type = 'LoRAAdapter'

# Different ranks for different components
backbone_rank = 8      # Small: feature extraction
neck_rank = 16         # Medium: feature fusion
head_rank = 32         # Large: task-specific prediction
attention_rank = 32    # Large: text-image fusion (important!)

adapter_alpha = 16.0   # Scaling factor
adapter_dropout = 0.1  # Regularization

# Training settings
max_epochs = 100
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 1e-3  # High LR for LoRA

# Load from pretrained model
# load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
# load_From = 'work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth'
resume = False

# Model settings - Dense LoRA
model = dict(
    # Backbone with LoRA
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackboneWithAdapter',
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=4,
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            rank=backbone_rank,  # rank=8 for backbone
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_stages=[3, 4],  # Only stage3, stage4 (high-level features)
        freeze_all=True,
        unfreeze_patterns=[]),  # No unfreezing, only LoRA
    
    # Neck with LoRA (all positions)
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            rank=neck_rank,  # rank=16 for neck
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_positions=['reduce', 'top_down', 'bottom_up', 'out'],  # ALL positions!
        freeze_all=True,
        unfreeze_patterns=[]),  # No unfreezing, only LoRA
    
    # Head with LoRA
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModuleWithAdapter',
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            use_adapter=use_adapter,
            adapter_cfg=dict(
                type=adapter_type,
                rank=head_rank,  # rank=32 for head (task-specific)
                alpha=adapter_alpha,
                dropout=adapter_dropout),
            adapter_positions=['both'],  # Both cls and reg
            freeze_all=True,
            unfreeze_patterns=[])),  # No unfreezing, only LoRA
    
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer - High LR for LoRA
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # LoRA parameters: NO weight decay
            'lora_A': dict(lr_mult=1.0, weight_decay=0.0),
            'lora_B': dict(lr_mult=1.0, weight_decay=0.0),
            
            # Different LR for different components
            'backbone.adapters': dict(lr_mult=0.5),  # Lower for backbone
            'neck.adapters': dict(lr_mult=1.0),      # Medium for neck
            'bbox_head.adapters': dict(lr_mult=1.5), # Higher for head (task-specific)
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


# Phase 2 Hybrid V1 - Dense LoRA + Selective Fine-tuning
# 
# 전략:
# 1. Dense LoRA: 모든 위치에 LoRA 추가 (parameter-efficient)
# 2. Selective Fine-tuning: 중요한 레이어는 full fine-tuning
#    - Neck 마지막 레이어 (out_layers.2)
#    - Head 마지막 레이어 (cls_preds.2, reg_preds.2)
# 
# 이 방식은 LoRA의 효율성과 Full fine-tuning의 성능을 결합합니다.

# _base_ = '../finetune_coco/vfm_v1_l_mvtec.py'
_base_ = '../adapter/vfm_v1_l_mvtec.py'

# LoRA settings
use_adapter = True
adapter_type = 'LoRAAdapter'

# Multi-scale ranks
backbone_rank = 8
neck_rank = 16
head_rank = 32
adapter_alpha = 16.0
adapter_dropout = 0.1

# Selective fine-tuning: unfreeze important layers
unfreeze_neck_patterns = [
    'out_layers.2',        # Neck 마지막 output layer (가장 중요)
]

unfreeze_head_patterns = [
    'cls_preds.2',         # Head 마지막 cls layer
    'reg_preds.2',         # Head 마지막 reg layer
]

# Training settings
max_epochs = 100
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 1e-3  # High LR for LoRA

# Load from pretrained model
# load_from = 'work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth'
resume = False

# Model settings - Hybrid (Dense LoRA + Selective Fine-tuning)
model = dict(
    # Backbone with LoRA (frozen except LoRA)
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
            rank=backbone_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_stages=[3, 4],
        freeze_all=True,
        unfreeze_patterns=[]),  # Backbone: only LoRA, no unfreezing
    
    # Neck with LoRA + Selective unfreezing
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            rank=neck_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_positions=['reduce', 'top_down', 'bottom_up', 'out'],  # Dense LoRA
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),  # + Selective unfreezing
    
    # Head with LoRA + Selective unfreezing
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
                rank=head_rank,
                alpha=adapter_alpha,
                dropout=adapter_dropout),
            adapter_positions=['both'],  # Dense LoRA
            freeze_all=True,
            unfreeze_patterns=unfreeze_head_patterns)),  # + Selective unfreezing
    
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer - Different LR for LoRA and unfrozen layers
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # LoRA parameters: HIGH LR, NO weight decay
            'lora_A': dict(lr_mult=1.0, weight_decay=0.0),
            'lora_B': dict(lr_mult=1.0, weight_decay=0.0),
            
            # Backbone LoRA: lower LR
            'backbone.adapters': dict(lr_mult=0.5, weight_decay=0.0),
            
            # Neck LoRA: medium LR
            'neck.adapters': dict(lr_mult=1.0, weight_decay=0.0),
            
            # Head LoRA: higher LR
            'bbox_head.adapters': dict(lr_mult=1.5, weight_decay=0.0),
            
            # Unfrozen neck layers: MEDIUM LR with weight decay
            'out_layers.2': dict(lr_mult=0.5, weight_decay=0.05),
            
            # Unfrozen head layers: MEDIUM LR with weight decay
            'cls_preds.2': dict(lr_mult=0.5, weight_decay=0.05),
            'reg_preds.2': dict(lr_mult=0.5, weight_decay=0.05),
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


# Phase 2 Hybrid Aggressive V1 - Dense LoRA + Aggressive Fine-tuning
# 
# 전략:
# 1. Dense LoRA: 모든 위치에 LoRA 추가
# 2. Aggressive Fine-tuning: 더 많은 레이어 unfreeze
#    - Backbone stage4 (일부)
#    - Neck 마지막 2개 레이어
#    - Head 전체
# 
# 이 방식은 성능을 최대한 끌어올리기 위한 공격적인 접근입니다.
# 메모리 사용량이 높지만 최고 성능을 기대할 수 있습니다.

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
adapter_dropout = 0.0 # default: 0.1

# Aggressive unfreezing
unfreeze_backbone_patterns = [
    'image_model.stage4',  # Backbone 마지막 stage
]

unfreeze_neck_patterns = [
    'top_down_layers.1',   # 마지막 2개
    'top_down_layers.2',
    'bottom_up_layers.1',  # 마지막 2개
    'bottom_up_layers.2',
    'out_layers.1',        # 마지막 2개
    'out_layers.2',
]

unfreeze_head_patterns = [
    'cls_preds',           # Head 전체 (모든 scale)
    'reg_preds',           # Head 전체 (모든 scale)
]

# Training settings
max_epochs = 200
close_mosaic_epochs = 2
save_epoch_intervals = 20
base_lr = 5e-4  # Moderate LR (LoRA + full fine-tuning)

# Load from pretrained model
load_from = 'work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth'
resume = False

# Model settings - Aggressive Hybrid
model = dict(
    # Backbone with LoRA + stage4 unfreezing
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
        unfreeze_patterns=unfreeze_backbone_patterns),  # Unfreeze stage4
    
    # Neck with Dense LoRA + aggressive unfreezing
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
        adapter_positions=['reduce', 'top_down', 'bottom_up', 'out'],
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),  # Unfreeze 마지막 2개
    
    # Head with Dense LoRA + full unfreezing
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
            adapter_positions=['both'],
            freeze_all=True,
            unfreeze_patterns=unfreeze_head_patterns)),  # Unfreeze 전체
    
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Training loop settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=20,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

# Optimizer - Balanced LR for LoRA and unfrozen layers
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=10.0, norm_type=2),  # Gradient clipping
    paramwise_cfg=dict(
        custom_keys={
            # LoRA parameters: HIGH LR, NO weight decay
            'lora_A': dict(lr_mult=2.0, weight_decay=0.0),  # 1e-3
            'lora_B': dict(lr_mult=2.0, weight_decay=0.0),
            
            # Backbone LoRA
            'backbone.adapters': dict(lr_mult=1.0, weight_decay=0.0),  # 5e-4
            
            # Neck LoRA
            'neck.adapters': dict(lr_mult=2.0, weight_decay=0.0),  # 1e-3
            
            # Head LoRA
            'bbox_head.adapters': dict(lr_mult=2.0, weight_decay=0.0),  # 1e-3
            
            # Unfrozen backbone: LOW LR
            'image_model.stage4': dict(lr_mult=0.1),  # 5e-5
            
            # Unfrozen neck: MEDIUM LR
            'top_down_layers': dict(lr_mult=0.5),  # 2.5e-4
            'bottom_up_layers': dict(lr_mult=0.5),
            'out_layers': dict(lr_mult=0.5),
            
            # Unfrozen head: MEDIUM LR
            'cls_preds': dict(lr_mult=0.5),  # 2.5e-4
            'reg_preds': dict(lr_mult=0.5),
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


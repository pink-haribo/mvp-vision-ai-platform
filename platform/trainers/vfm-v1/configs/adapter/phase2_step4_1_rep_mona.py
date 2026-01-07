# Phase 2 Step 4-1: Conservative + Spatial Context
# Rep-MoNA LoRA (Neck only) + Moderate Unfreezing
# 
# 전략:
# 1. Backbone: Standard LoRA (rank=8) - CLIP 사전학습 충분
# 2. Neck: Rep-MoNA LoRA (rank=16) - 핵심! Multi-scale spatial context
# 3. Head: Standard LoRA (rank=32) - Task-specific
# 4. Unfreezing: Step 2와 동일 (Neck out_layers.2, Head cls/reg_preds.2)
# 
# 목표:
# - Step 2의 안정성 유지
# - Neck에서 공간 문맥 인식 → 미세 결함 탐지 향상
# - 추론 속도 동일 (re-parameterization)
# - 파라미터 증가 무시 가능 (+0.03M)

_base_ = '../adapter/vfm_v1_l_mvtec.py'

# LoRA settings
use_adapter = True

# Multi-scale ranks
backbone_rank = 8
neck_rank = 16
head_rank = 32
adapter_alpha = 16.0
adapter_dropout = 0.1

# Rep-MoNA settings
rep_mona_kernel_sizes = [3, 5, 7]  # Multi-scale
rep_mona_use_layer_norm = True     # LayerNorm (안정성 우선)

# Selective fine-tuning: Step 2와 동일
unfreeze_neck_patterns = [
    'out_layers.2',        # Neck 마지막 output layer만
]

unfreeze_head_patterns = [
    'cls_preds.2',         # Head 마지막 scale만
    'reg_preds.2',
]

# Model settings - Step 4-1 (Conservative + Spatial Context)
model = dict(
    # Backbone with Standard LoRA (rank=8)
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
            type='LoRAAdapter',  # Standard LoRA
            rank=backbone_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_stages=[3, 4],
        freeze_all=True,
        unfreeze_patterns=[]),  # Backbone: only LoRA, no unfreezing
    
    # Neck with Rep-MoNA LoRA (rank=16) ⭐ 핵심 개선!
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type='RepMoNAAdapter',  # Rep-MoNA LoRA!
            rank=neck_rank,
            kernel_sizes=rep_mona_kernel_sizes,
            use_layer_norm=rep_mona_use_layer_norm),
        adapter_positions=['reduce', 'top_down', 'bottom_up', 'out'],  # Dense
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),  # + Selective unfreezing
    
    # Head with Standard LoRA (rank=32)
    bbox_head=dict(
        head_module=dict(
            type='YOLOWorldHeadModuleWithAdapter',
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            use_adapter=use_adapter,
            adapter_cfg=dict(
                type='LoRAAdapter',  # Standard LoRA
                rank=head_rank,
                alpha=adapter_alpha,
                dropout=adapter_dropout),
            adapter_positions=['both'],  # cls + reg
            freeze_all=True,
            unfreeze_patterns=unfreeze_head_patterns))
)

# Training settings
max_epochs = 160
close_mosaic_epochs = 10
save_epoch_intervals = 10

# Optimizer - Discriminative Learning Rate
base_lr = 5e-4

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Backbone Standard LoRA
            'backbone.adapters': dict(lr_mult=1.0),  # 5e-4
            
            # Neck Rep-MoNA LoRA (약간 높게)
            'neck.adapters': dict(lr_mult=1.4),  # 7e-4
            
            # Head Standard LoRA
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.0),  # 5e-4
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.0),  # 5e-4
            
            # Unfrozen layers (낮게)
            'neck.out_layers.2': dict(lr_mult=0.5),  # 2.5e-4
            'bbox_head.head_module.cls_preds.2': dict(lr_mult=0.5),  # 2.5e-4
            'bbox_head.head_module.reg_preds.2': dict(lr_mult=0.5),  # 2.5e-4
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

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
        begin=1,  # Fixed: prevent negative LR
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Runtime settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]

# Logging
default_hooks.update(
    dict(
        logger=dict(
            type='LoggerHook',
            interval=50)))

# Visualization
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend')
    ],
    name='visualizer')

# Load from Step 2 checkpoint (optional)
# load_from = 'work_dirs/phase2_hybrid_v1/best_coco_bbox_mAP_epoch_XXX.pth'


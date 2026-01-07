# Phase 2 Step 4-2: Aggressive + Stabilized
# Rep-MoNA LoRA (Neck + Head) + Reduced Unfreezing
# 
# 전략:
# 1. Backbone: Standard LoRA (rank=8) - Freeze (Step 3는 stage4 unfrozen)
# 2. Neck: Rep-MoNA LoRA (rank=16) - 핵심! Multi-scale spatial context
# 3. Head: Rep-MoNA LoRA (rank=32) - 추가 개선! Spatial context
# 4. Unfreezing: Step 3보다 대폭 축소 (Rep-MoNA로 표현력 확보)
# 
# 목표:
# - Step 3의 gradient explosion 해결
# - Rep-MoNA로 표현력 확보 → Unfreezing 범위 축소
# - 파라미터 효율성 20.78M → 5.3M (3.8배 개선)
# - Step 2보다 높은 표현력 기대

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

# Selective fine-tuning: Step 3보다 축소
unfreeze_neck_patterns = [
    'top_down_layers.2',   # 마지막만 (Step 3는 1,2 모두)
    'bottom_up_layers.2',  # 마지막만
]

unfreeze_head_patterns = [
    'cls_preds.2',         # 마지막만 (Step 3는 전체)
    'reg_preds.2',         # 마지막만
]

# Model settings - Step 4-2 (Aggressive + Stabilized)
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
        unfreeze_patterns=[]),  # Backbone: Freeze (Step 3는 stage4 unfrozen)
    
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
        unfreeze_patterns=unfreeze_neck_patterns),  # 축소된 unfreezing
    
    # Head with Rep-MoNA LoRA (rank=32) ⭐ 추가 개선!
    bbox_head=dict(
        head_module=dict(
            type='YOLOWorldHeadModuleWithAdapter',
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            use_adapter=use_adapter,
            adapter_cfg=dict(
                type='RepMoNAAdapter',  # Rep-MoNA LoRA!
                rank=head_rank,
                kernel_sizes=rep_mona_kernel_sizes,
                use_layer_norm=rep_mona_use_layer_norm),
            adapter_positions=['both'],  # cls + reg
            freeze_all=True,
            unfreeze_patterns=unfreeze_head_patterns))  # 축소된 unfreezing
)

# Training settings
max_epochs = 160
close_mosaic_epochs = 10
save_epoch_intervals = 10
warmup_epochs = 10  # Rep-MoNA 초기화 안정화

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
            
            # Head Rep-MoNA LoRA (약간 높게)
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.4),  # 7e-4
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.4),  # 7e-4
            
            # Unfrozen layers (낮게, Step 3 안정성 개선)
            'neck.top_down_layers.2': dict(lr_mult=0.5),  # 2.5e-4
            'neck.bottom_up_layers.2': dict(lr_mult=0.5),  # 2.5e-4
            'bbox_head.head_module.cls_preds.2': dict(lr_mult=0.5),  # 2.5e-4
            'bbox_head.head_module.reg_preds.2': dict(lr_mult=0.5),  # 2.5e-4
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# Learning rate scheduler (with warmup)
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

# Custom hooks - 안정화 기법
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,  # EMA for stability
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

# Loss weight balancing (Step 3 gradient explosion 완화)
model.update(
    dict(
        bbox_head=dict(
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=0.5),  # Step 3 실패 원인 완화
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                reduction='sum',
                loss_weight=7.5,
                return_iou=False),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=1.5 / 4))))

# Load from Step 2 checkpoint (optional)
# load_from = 'work_dirs/phase2_hybrid_v1/best_coco_bbox_mAP_epoch_XXX.pth'


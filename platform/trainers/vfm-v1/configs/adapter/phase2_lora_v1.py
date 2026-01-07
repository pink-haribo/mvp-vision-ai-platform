# Phase 2 LoRA V1 - LoRA adapter 사용
# LoRA는 Bottleneck adapter보다 더 효과적인 PEFT 방법입니다.
# 
# 장점:
# 1. 더 적은 파라미터로 더 높은 표현력
# 2. Parallel 구조로 gradient flow 개선
# 3. LLM fine-tuning에서 검증된 효과

# _base_ = '../finetune_coco/vfm_v1_l_mvtec.py'
_base_ = './vfm_v1_l_mvtec.py'

# LoRA settings
use_adapter = True
adapter_type = 'LoRAAdapter'  # NEW: LoRA instead of Bottleneck
adapter_rank = 32  # Rank of low-rank decomposition (16, 32, 64)
adapter_alpha = 32.0  # Scaling factor (usually same as rank)
adapter_dropout = 0.1  # Dropout for regularization
adapter_positions = ['top_down', 'bottom_up']

# Phase 2: Unfreeze more layers
unfreeze_neck_patterns = [
    'top_down_layers.1',
    'top_down_layers.2',
    'bottom_up_layers.1',
    'bottom_up_layers.2',
    'out_layers.1',
    'out_layers.2',
]

# Training settings
max_epochs = 100
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 1e-3  # Higher LR for LoRA (1e-3 is common)

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
            rank=adapter_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout),
        adapter_positions=adapter_positions,
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),
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

# Optimizer - Higher LR for LoRA
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),  # Lower weight decay for LoRA
    paramwise_cfg=dict(
        custom_keys={
            # LoRA adapters: HIGH LR (1e-3)
            'lora_A': dict(lr_mult=1.0, weight_decay=0.0),  # No weight decay for LoRA
            'lora_B': dict(lr_mult=1.0, weight_decay=0.0),
            
            # Unfrozen neck layers: MEDIUM LR (5e-4)
            'top_down_layers': dict(lr_mult=0.5),
            'bottom_up_layers': dict(lr_mult=0.5),
            'out_layers': dict(lr_mult=0.5),
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


# Phase 2 Step 5-1: True MoNA Adapter (MVTec) - Base
# 원본 MoNA 논문 (CVPR 2025) 구조 충실히 재현 + Pretrained 로드
#
# Step 5 시리즈:
# - step5_1 (이 파일): load_from 추가 (기본)
# - step5_2: + Backbone adapter 적용 및 stage4 unfreeze
# - step5_3: + Neck unfreeze 확대 (layers 1,2)
# - step5_4: + Head unfreeze 확대 (전체)
#
# 참고: https://github.com/LeiyiHU/mona

_base_ = './vfm_v1_l_mvtec.py'

# DDP 설정
find_unused_parameters = True

# ⭐ Pretrained 체크포인트 로드 (핵심!)
load_from = 'work_dirs/vfm_v1_l_mvtec/best_coco_bbox_mAP_epoch_30.pth'
resume = False

# Adapter settings
use_adapter = True

# True MoNA settings (원본 논문 기반)
mona_hidden_dim = 64        # 원본 고정값
mona_kernel_sizes = [3, 5, 7]  # Multi-scale
mona_dropout = 0.1          # 원본 값
mona_gamma_init = 1e-6      # 핵심! 초기 identity 보장

# Selective fine-tuning (최소화)
unfreeze_neck_patterns = [
    'top_down_layers.2',
    'bottom_up_layers.2',
]

unfreeze_head_patterns = [
    'cls_preds.2',
    'reg_preds.2',
]

# Model settings - Step 5-1 (True MoNA + Pretrained)
model = dict(
    # Backbone: 완전 Freeze (adapter 없음)
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackboneWithAdapter',
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=4,
        use_adapter=False,
        freeze_all=True,
        unfreeze_patterns=[]),

    # Neck with True MoNA
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type='TrueMoNAAdapter',
            hidden_dim=mona_hidden_dim,
            kernel_sizes=mona_kernel_sizes,
            dropout=mona_dropout,
            gamma_init=mona_gamma_init),
        adapter_positions=['reduce', 'top_down', 'bottom_up', 'out'],
        freeze_all=True,
        unfreeze_patterns=unfreeze_neck_patterns),

    # Head with True MoNA
    bbox_head=dict(
        _delete_=True,
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModuleWithAdapter',
            num_classes=_base_.num_training_classes,
            in_channels=[256, 512, 512],
            widen_factor=1.0,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32],
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            use_adapter=use_adapter,
            adapter_cfg=dict(
                type='TrueMoNAAdapter',
                hidden_dim=mona_hidden_dim,
                kernel_sizes=mona_kernel_sizes,
                dropout=mona_dropout,
                gamma_init=mona_gamma_init),
            adapter_positions=['both'],
            freeze_all=True,
            unfreeze_patterns=unfreeze_head_patterns),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5,
            strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=_base_.num_training_classes,
                use_ciou=True,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-9)),
        test_cfg=dict(
            multi_label=True,
            nms_pre=1000,
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
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
            loss_weight=0.375)),

    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes))
)

# Training settings
max_epochs = 160
close_mosaic_epochs = 10
save_epoch_intervals = 10
warmup_epochs = 10

# Optimizer
base_lr = 5e-4

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Neck/Head True MoNA adapters
            'neck.reduce_adapters': dict(lr_mult=1.4),
            'neck.top_down_adapters': dict(lr_mult=1.4),
            'neck.bottom_up_adapters': dict(lr_mult=1.4),
            'neck.out_adapters': dict(lr_mult=1.4),
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.4),
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.4),
            # Unfrozen layers (낮은 lr)
            'neck.top_down_layers.2': dict(lr_mult=0.5),
            'neck.bottom_up_layers.2': dict(lr_mult=0.5),
            'bbox_head.head_module.cls_preds.2': dict(lr_mult=0.5),
            'bbox_head.head_module.reg_preds.2': dict(lr_mult=0.5),
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
        begin=1,
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
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))

# Custom hooks
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

# Visualization
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')


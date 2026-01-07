# Phase 1 - Option 2 (HierarchicalAdapter) - Strategy B (Multi-stage: Backbone + Neck + Head)
# Resume from epoch_100.pth and train only adapters for 50 epochs

_base_ = '../finetune_coco/vfm_v1_l_mvtec.py'

# Adapter settings
use_adapter = True
adapter_type = 'HierarchicalAdapter'  # Option 2: Hierarchical with attention
adapter_reduction_ratio = 4
adapter_num_heads = 8
adapter_mlp_ratio = 4

# Adapter positions for each component
backbone_adapter_stages = [2, 3, 4]  # Add adapters to stage 2, 3, 4
neck_adapter_positions = ['top_down', 'bottom_up']
head_adapter_positions = ['both']  # Both cls and reg

# Training settings
max_epochs = 50
close_mosaic_epochs = 2
save_epoch_intervals = 10
base_lr = 1e-4  # Lower learning rate for adapter fine-tuning

# Resume from previous training
load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
resume = False

# Model settings with multi-stage adapters
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=_base_.num_training_classes,
    num_test_classes=_base_.num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackboneWithAdapter',  # Use adapter-enabled backbone
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=-1,  # Don't use frozen_stages (we use freeze_all instead)
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            reduction_ratio=adapter_reduction_ratio,
            num_heads=adapter_num_heads,
            mlp_ratio=adapter_mlp_ratio),
        adapter_stages=backbone_adapter_stages,
        freeze_all=True),  # Freeze all backbone parameters except adapters
    neck=dict(
        type='YOLOWorldPAFPNWithAdapter',  # Use adapter-enabled neck
        guide_channels=_base_.text_channels,
        embed_channels=_base_.neck_embed_channels,
        num_heads=_base_.neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        use_adapter=use_adapter,
        adapter_cfg=dict(
            type=adapter_type,
            reduction_ratio=adapter_reduction_ratio,
            num_heads=adapter_num_heads,
            mlp_ratio=adapter_mlp_ratio),
        adapter_positions=neck_adapter_positions,
        freeze_all=True),  # Freeze original neck parameters
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModuleWithAdapter',  # Use adapter-enabled head
            use_bn_head=True,
            embed_dims=_base_.text_channels,
            num_classes=_base_.num_training_classes,
            use_adapter=use_adapter,
            adapter_cfg=dict(
                type=adapter_type,
                reduction_ratio=adapter_reduction_ratio,
                num_heads=adapter_num_heads,
                mlp_ratio=adapter_mlp_ratio),
            adapter_positions=head_adapter_positions,
            freeze_all=True)),  # Freeze original head parameters
    train_cfg=dict(assigner=dict(num_classes=_base_.num_training_classes)))

# Optimizer: Only train adapter parameters
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
        batch_size_per_gpu=_base_.train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            # Freeze all original parameters
            'backbone.image_model': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone.text_model': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.reduce_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.top_down_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.bottom_up_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.out_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.upsample_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.downsample_layers': dict(lr_mult=0.0, decay_mult=0.0),
            'bbox_head.head_module.cls_preds': dict(lr_mult=0.0, decay_mult=0.0),
            'bbox_head.head_module.reg_preds': dict(lr_mult=0.0, decay_mult=0.0),
            'bbox_head.head_module.cls_contrasts': dict(lr_mult=0.0, decay_mult=0.0),
            # Train all adapters
            'backbone.adapters': dict(lr_mult=1.0),
            'neck.top_down_adapters': dict(lr_mult=1.0),
            'neck.bottom_up_adapters': dict(lr_mult=1.0),
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.0),
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.0),
            'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# Training settings
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best='auto',
        interval=save_epoch_intervals))

# Work directory
work_dir = './work_dirs/phase1_option2_strategy_b'

# DDP settings - required for adapter training with frozen parameters
find_unused_parameters = True


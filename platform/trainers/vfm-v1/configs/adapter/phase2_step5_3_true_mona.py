# Phase 2 Step 5-3: True MoNA + Backbone + Extended Neck (MVTec)
# 
# Step 5-1 기반 + Backbone adapter + Neck unfreeze 확대
#
# 포함된 것:
# - [Step 5-2] Backbone에 TrueMoNA adapter 적용 (stage 3,4) + stage4 unfreeze
# - [Step 5-3] Neck layers 1,2 모두 unfreeze + out_layers unfreeze
#
# 참고: https://github.com/LeiyiHU/mona

_base_ = './phase2_step5_1_true_mona.py'

# ⭐ Backbone unfreeze (Step 5-2)
unfreeze_backbone_patterns = [
    'image_model.stage4',
]

# ⭐ Neck unfreeze 확대 (Step 5-3)
unfreeze_neck_patterns = [
    'top_down_layers.1',
    'top_down_layers.2',
    'bottom_up_layers.1',
    'bottom_up_layers.2',
    'out_layers.1',
    'out_layers.2',
]

# Model settings - Step 5-3
model = dict(
    # Backbone with TrueMoNA + stage4 unfreeze (Step 5-2)
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackboneWithAdapter',
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=4,
        use_adapter=True,
        adapter_cfg=dict(
            type='TrueMoNAAdapter',
            hidden_dim=_base_.mona_hidden_dim,
            kernel_sizes=_base_.mona_kernel_sizes,
            dropout=_base_.mona_dropout,
            gamma_init=_base_.mona_gamma_init),
        adapter_stages=[3, 4],
        freeze_all=True,
        unfreeze_patterns=unfreeze_backbone_patterns),
    # Neck unfreeze 확대 (Step 5-3)
    neck=dict(
        unfreeze_patterns=unfreeze_neck_patterns),
)

# Optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            # Backbone
            'backbone.adapters': dict(lr_mult=0.5, weight_decay=0.0),
            'image_model.stage4': dict(lr_mult=0.1),
            # Neck adapters
            'neck.reduce_adapters': dict(lr_mult=1.4),
            'neck.top_down_adapters': dict(lr_mult=1.4),
            'neck.bottom_up_adapters': dict(lr_mult=1.4),
            'neck.out_adapters': dict(lr_mult=1.4),
            # Neck layers
            'neck.top_down_layers.1': dict(lr_mult=0.5),
            'neck.top_down_layers.2': dict(lr_mult=0.5),
            'neck.bottom_up_layers.1': dict(lr_mult=0.5),
            'neck.bottom_up_layers.2': dict(lr_mult=0.5),
            'neck.out_layers.1': dict(lr_mult=0.5),
            'neck.out_layers.2': dict(lr_mult=0.5),
            # Head adapters
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.4),
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.4),
            # Head layers
            'bbox_head.head_module.cls_preds.2': dict(lr_mult=0.5),
            'bbox_head.head_module.reg_preds.2': dict(lr_mult=0.5),
        }
    )
)


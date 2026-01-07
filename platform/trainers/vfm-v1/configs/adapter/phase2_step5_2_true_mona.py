# Phase 2 Step 5-2: True MoNA + Backbone Adapter (MVTec)
# 
# Step 5-1 기반 + Backbone adapter 적용 및 stage4 unfreeze
#
# 추가된 것:
# - Backbone에 TrueMoNA adapter 적용 (stage 3,4)
# - Backbone stage4 unfreeze
#
# 참고: https://github.com/LeiyiHU/mona

_base_ = './phase2_step5_1_true_mona.py'

# ⭐ Step 5-2: Backbone adapter 활성화 + stage4 unfreeze
unfreeze_backbone_patterns = [
    'image_model.stage4',  # Backbone 마지막 stage
]

# Model settings - Step 5-2
model = dict(
    # Backbone with True MoNA + stage4 unfreeze
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackboneWithAdapter',
        image_model={{_base_.model.backbone.image_model}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=_base_.text_model_name,
            frozen_modules=['all']),
        frozen_stages=4,
        use_adapter=True,  # ⭐ Backbone adapter 활성화!
        adapter_cfg=dict(
            type='TrueMoNAAdapter',
            hidden_dim=_base_.mona_hidden_dim,
            kernel_sizes=_base_.mona_kernel_sizes,
            dropout=_base_.mona_dropout,
            gamma_init=_base_.mona_gamma_init),
        adapter_stages=[3, 4],  # stage 3,4에 adapter
        freeze_all=True,
        unfreeze_patterns=unfreeze_backbone_patterns),  # stage4 unfreeze
)

# Optimizer - Backbone 학습률 조정
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            # Backbone adapter: 낮은 lr
            'backbone.adapters': dict(lr_mult=0.5, weight_decay=0.0),
            # Backbone stage4: 매우 낮은 lr
            'image_model.stage4': dict(lr_mult=0.1),
            # Neck/Head adapters
            'neck.reduce_adapters': dict(lr_mult=1.4),
            'neck.top_down_adapters': dict(lr_mult=1.4),
            'neck.bottom_up_adapters': dict(lr_mult=1.4),
            'neck.out_adapters': dict(lr_mult=1.4),
            'bbox_head.head_module.cls_adapters': dict(lr_mult=1.4),
            'bbox_head.head_module.reg_adapters': dict(lr_mult=1.4),
            # Unfrozen layers
            'neck.top_down_layers.2': dict(lr_mult=0.5),
            'neck.bottom_up_layers.2': dict(lr_mult=0.5),
            'bbox_head.head_module.cls_preds.2': dict(lr_mult=0.5),
            'bbox_head.head_module.reg_preds.2': dict(lr_mult=0.5),
        }
    )
)


# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World Head with Adapter support

import copy
from typing import Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptConfigType
from mmyolo.registry import MODELS
from torch.nn.modules.batchnorm import _BatchNorm

from .yolo_world_head import YOLOWorldHeadModule


@MODELS.register_module()
class YOLOWorldHeadModuleWithAdapter(YOLOWorldHeadModule):
    """YOLO-World Head Module with Adapter support.
    
    This head module adds adapter modules to the prediction heads
    for parameter-efficient fine-tuning.
    
    Args:
        use_adapter (bool): Whether to use adapters. Default: False.
        adapter_cfg (ConfigType): Config for adapter modules. Default: None.
        adapter_positions (List[str]): Where to insert adapters.
            Options: ['cls', 'reg', 'both']. Default: ['both'].
        freeze_all (bool): Whether to freeze all original parameters. Default: True.
    """
    
    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 use_adapter: bool = False,
                 adapter_cfg: ConfigType = None,
                 adapter_positions: List[str] = ['both'],
                 freeze_all: bool = False,
                 unfreeze_patterns: List[str] = [],  # Phase 2: patterns to unfreeze
                 **kwargs) -> None:

        self.use_adapter = use_adapter
        self.adapter_positions = adapter_positions
        self.unfreeze_patterns = unfreeze_patterns

        # Initialize parent class
        super().__init__(
            *args,
            embed_dims=embed_dims,
            use_bn_head=use_bn_head,
            use_einsum=use_einsum,
            freeze_all=freeze_all,
            **kwargs)
        
        # Build adapters if enabled
        if self.use_adapter:
            if adapter_cfg is None:
                adapter_cfg = dict(type='BottleneckAdapter', reduction_ratio=4)
            
            self._build_adapters(adapter_cfg)
    
    def _build_adapters(self, adapter_cfg: ConfigType) -> None:
        """Build adapter modules for prediction heads."""
        
        # Adapters for classification heads
        if 'cls' in self.adapter_positions or 'both' in self.adapter_positions:
            self.cls_adapters = nn.ModuleList()
            for in_channel in self.in_channels:
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = self.embed_dims
                self.cls_adapters.append(MODELS.build(cfg))
        
        # Adapters for regression heads
        if 'reg' in self.adapter_positions or 'both' in self.adapter_positions:
            self.reg_adapters = nn.ModuleList()
            reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
            for _ in self.in_channels:
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = reg_out_channels
                self.reg_adapters.append(MODELS.build(cfg))
    
    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       txt_masks: Tensor, cls_pred: nn.Module,
                       reg_pred: nn.Module,
                       cls_contrast: nn.Module, idx: int = 0) -> Tuple:
        """Forward feature of a single scale level with adapters.

        Args:
            idx: Index of the current scale level (for adapter selection)
        """
        b, _, h, w = img_feat.shape

        # Classification branch
        cls_embed = cls_pred(img_feat)

        # Apply adapter to cls_embed if enabled
        if self.use_adapter and ('cls' in self.adapter_positions or 'both' in self.adapter_positions):
            cls_embed = self.cls_adapters[idx](cls_embed)

        cls_logit = cls_contrast(cls_embed, txt_feat)

        if txt_masks is not None:
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = -10e6
            else:
                cls_logit[txt_masks == 0] = -10e6

        # Regression branch
        reg_feat = reg_pred(img_feat)

        # Apply adapter to reg_feat if enabled
        if self.use_adapter and ('reg' in self.adapter_positions or 'both' in self.adapter_positions):
            reg_feat = self.reg_adapters[idx](reg_feat)

        if self.reg_max > 1:
            reg_dist_pred = reg_feat.reshape([-1, 4, self.reg_max, h, w])
            reg_dist_pred = reg_dist_pred.permute(0, 3, 4, 1, 2).reshape(b, h, w, 4, self.reg_max)
            reg_dist_pred = reg_dist_pred.softmax(4).matmul(self.proj.view([-1, 1])).squeeze(-1)
            reg_pred_out = reg_dist_pred.permute(0, 3, 1, 2)
        else:
            reg_pred_out = reg_feat

        if self.training:
            reg_dist_pred_for_loss = reg_feat.reshape([-1, 4, self.reg_max, h, w])
            return cls_logit, reg_pred_out, reg_dist_pred_for_loss
        else:
            return cls_logit, reg_pred_out

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor = None) -> Tuple[List]:
        """Forward features from the upstream network with adapters."""
        assert len(img_feats) == self.num_levels

        # Prepare text features and masks for each level
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        if txt_masks is not None:
            txt_masks = [txt_masks for _ in range(self.num_levels)]
        else:
            txt_masks = [None for _ in range(self.num_levels)]

        # Create index list for adapter selection
        indices = list(range(self.num_levels))

        # Use multi_apply with index parameter
        from mmdet.models.utils import multi_apply
        return multi_apply(self.forward_single, img_feats, txt_feats,
                          txt_masks, self.cls_preds, self.reg_preds,
                          self.cls_contrasts, indices)

    def train(self, mode=True):
        """Convert the model into training mode while keeping frozen parts in eval mode.

        Phase 2: Also unfreezes layers matching unfreeze_patterns.
        """
        super(YOLOWorldHeadModule, self).train(mode)

        if self.freeze_all:
            # Freeze all base parameters
            for name, module in self.named_modules():
                # Skip adapter modules - keep them in training mode
                if 'adapter' in name:
                    module.train(mode)
                    for param in module.parameters():
                        param.requires_grad = True
                # Phase 2: Unfreeze layers matching patterns
                elif any(pattern in name for pattern in self.unfreeze_patterns):
                    module.train(mode)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    # Freeze other modules
                    if isinstance(module, _BatchNorm):
                        module.eval()
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False


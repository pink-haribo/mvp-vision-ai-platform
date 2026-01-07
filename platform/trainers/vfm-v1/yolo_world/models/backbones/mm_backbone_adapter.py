# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Modal Backbone with Adapter support

import copy
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS

from .mm_backbone import MultiModalYOLOBackbone


@MODELS.register_module()
class MultiModalYOLOBackboneWithAdapter(MultiModalYOLOBackbone):
    """Multi-Modal YOLO Backbone with Adapter modules.
    
    This backbone adds adapter modules to the image model stages
    for parameter-efficient fine-tuning.
    
    Args:
        use_adapter (bool): Whether to use adapters. Default: False.
        adapter_cfg (ConfigType): Config for adapter modules. Default: None.
        adapter_stages (List[int]): Which stages to add adapters. 
            Default: [1, 2, 3, 4] (all stages).
        freeze_all (bool): Whether to freeze all original parameters. Default: True.
    """
    
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 use_adapter: bool = False,
                 adapter_cfg: ConfigType = None,
                 adapter_stages: List[int] = [1, 2, 3, 4],
                 freeze_all: bool = False,
                 unfreeze_patterns: List[str] = [],  # Phase 2: patterns to unfreeze
                 init_cfg: OptMultiConfig = None) -> None:
        
        self.use_adapter = use_adapter
        self.adapter_stages = adapter_stages
        self.freeze_all_params = freeze_all
        self.unfreeze_patterns = unfreeze_patterns

        # Initialize parent class
        super().__init__(
            image_model=image_model,
            text_model=text_model,
            frozen_stages=frozen_stages,
            with_text_model=with_text_model,
            init_cfg=init_cfg)
        
        # Build adapters if enabled
        if self.use_adapter:
            if adapter_cfg is None:
                adapter_cfg = dict(type='BottleneckAdapter', reduction_ratio=4)
            
            self._build_adapters(adapter_cfg)
        
        # Freeze all parameters if requested
        if self.freeze_all_params:
            self._freeze_all()
    
    def _build_adapters(self, adapter_cfg: ConfigType) -> None:
        """Build adapter modules for specified stages."""
        self.adapters = nn.ModuleDict()
        
        for stage_idx in self.adapter_stages:
            stage_name = f'stage{stage_idx}'
            if hasattr(self.image_model, stage_name):
                # Get the output channels of this stage
                stage = getattr(self.image_model, stage_name)
                
                # Find the output channels from the last layer in the stage
                if hasattr(stage, 'out_channels'):
                    out_channels = stage.out_channels
                elif isinstance(stage, nn.Sequential) and len(stage) > 0:
                    # Try to infer from the last module
                    last_module = stage[-1]
                    if hasattr(last_module, 'out_channels'):
                        out_channels = last_module.out_channels
                    else:
                        # Default fallback - will be set dynamically
                        out_channels = None
                else:
                    out_channels = None
                
                if out_channels is not None:
                    cfg = copy.deepcopy(adapter_cfg)
                    cfg['in_channels'] = out_channels
                    self.adapters[stage_name] = MODELS.build(cfg)
    
    def _freeze_all(self):
        """Freeze all parameters except adapters."""
        # Freeze image model
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # Freeze text model (already frozen in parent)
        if self.text_model is not None:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Ensure adapters are trainable
        if self.use_adapter:
            for adapter in self.adapters.values():
                for param in adapter.parameters():
                    param.requires_grad = True
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen parts frozen.

        Phase 2: Also unfreezes layers matching unfreeze_patterns.
        """
        super().train(mode)

        if self.freeze_all_params:
            # Freeze all parameters first
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
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False
    
    def forward(self, image: Tensor, text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        """Forward with adapters."""
        # Forward through image model with adapters
        img_feats = []
        x = image
        
        for i, layer_name in enumerate(self.image_model.layers):
            layer = getattr(self.image_model, layer_name)
            x = layer(x)
            
            # Apply adapter if this stage has one
            if self.use_adapter and layer_name in self.adapters:
                x = self.adapters[layer_name](x)
            
            # Collect output features
            if i in self.image_model.out_indices:
                img_feats.append(x)
        
        img_feats = tuple(img_feats)
        
        # Forward through text model
        if text is not None and self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None


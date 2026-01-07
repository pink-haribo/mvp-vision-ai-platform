# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World PAFPN with Adapter support

import copy
from typing import List, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS

from .yolo_world_pafpn import YOLOWorldPAFPN, YOLOWorldDualPAFPN


@MODELS.register_module()
class YOLOWorldPAFPNWithAdapter(YOLOWorldPAFPN):
    """YOLO-World PAFPN with Adapter modules.
    
    This neck adds adapter modules to the PAFPN for parameter-efficient fine-tuning.
    Adapters can be inserted at different positions:
    - 'reduce': After reduce layers
    - 'top_down': After top-down layers
    - 'bottom_up': After bottom-up layers
    - 'out': After output layers
    
    Args:
        use_adapter (bool): Whether to use adapters. Default: False.
        adapter_cfg (ConfigType): Config for adapter modules. Default: None.
        adapter_positions (List[str]): Where to insert adapters. 
            Options: ['reduce', 'top_down', 'bottom_up', 'out']. 
            Default: ['top_down', 'bottom_up'].
        freeze_all (bool): Whether to freeze all original parameters. Default: True.
    """
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 use_adapter: bool = False,
                 adapter_cfg: ConfigType = None,
                 adapter_positions: List[str] = ['top_down', 'bottom_up'],
                 freeze_all: bool = False,
                 unfreeze_patterns: List[str] = [],  # Phase 2: patterns to unfreeze
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        
        self.use_adapter = use_adapter
        self.adapter_positions = adapter_positions
        self.unfreeze_patterns = unfreeze_patterns

        # Initialize parent class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            block_cfg=block_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        
        # Build adapters if enabled
        if self.use_adapter:
            if adapter_cfg is None:
                adapter_cfg = dict(type='BottleneckAdapter', reduction_ratio=4)
            
            self._build_adapters(adapter_cfg)
    
    def _build_adapters(self, adapter_cfg: ConfigType) -> None:
        """Build adapter modules for different positions.

        Channel flow in PAFPN:
        - reduce_layers[idx]: in_channels[idx] -> out_channels[idx]
        - top_down_layers are built in __init__ with: for idx in range(len-1, 0, -1)
          - build_top_down_layer(idx) outputs out_channels[idx-1]
          - So top_down_layers[0] = build_top_down_layer(len-1) outputs out_channels[len-2]
          - And top_down_layers[1] = build_top_down_layer(len-2) outputs out_channels[len-3]
        - bottom_up_layers[idx] outputs out_channels[idx + 1]
        - out_layers[idx]: out_channels[idx] -> out_channels[idx]
        """

        # Adapters for reduce layers (output: out_channels[idx])
        if 'reduce' in self.adapter_positions:
            self.reduce_adapters = nn.ModuleList()
            for idx in range(len(self.in_channels)):
                out_ch = make_divisible(self.out_channels[idx], self.widen_factor)
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = out_ch
                self.reduce_adapters.append(MODELS.build(cfg))

        # Adapters for top-down layers
        # In __init__: for idx in range(len(in_channels) - 1, 0, -1):
        #     top_down_layers.append(build_top_down_layer(idx))
        # build_top_down_layer(idx) outputs out_channels[idx - 1]
        # So we need to build adapters in the same order
        if 'top_down' in self.adapter_positions:
            self.top_down_adapters = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1, 0, -1):
                # build_top_down_layer(idx) outputs out_channels[idx - 1]
                out_ch = make_divisible(self.out_channels[idx - 1], self.widen_factor)
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = out_ch
                self.top_down_adapters.append(MODELS.build(cfg))

        # Adapters for bottom-up layers
        # bottom_up_layers[idx] outputs out_channels[idx + 1]
        if 'bottom_up' in self.adapter_positions:
            self.bottom_up_adapters = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1):
                out_ch = make_divisible(self.out_channels[idx + 1], self.widen_factor)
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = out_ch
                self.bottom_up_adapters.append(MODELS.build(cfg))

        # Adapters for output layers (output: out_channels[idx])
        if 'out' in self.adapter_positions:
            self.out_adapters = nn.ModuleList()
            for idx in range(len(self.in_channels)):
                out_ch = make_divisible(self.out_channels[idx], self.widen_factor)
                cfg = copy.deepcopy(adapter_cfg)
                cfg['in_channels'] = out_ch
                self.out_adapters.append(MODELS.build(cfg))

    def _freeze_all(self):
        """Freeze all parameters except adapters."""
        # Freeze all base parameters
        for name, module in self.named_modules():
            # Skip adapter modules
            if 'adapter' in name:
                continue

            # Freeze batch norm
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

            # Freeze parameters
            for param in module.parameters(recurse=False):
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keeping frozen parts in eval mode.

        Phase 2: Also unfreezes layers matching unfreeze_patterns.
        """
        super(YOLOWorldPAFPN, self).train(mode)  # Skip YOLOWorldPAFPN's train() to avoid double freezing

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
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function with adapters."""
        assert len(img_feats) == len(self.in_channels)
        
        # Reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            out = self.reduce_layers[idx](img_feats[idx])
            if self.use_adapter and 'reduce' in self.adapter_positions:
                out = self.reduce_adapters[idx](out)
            reduce_outs.append(out)
        
        # Top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            
            if self.use_adapter and 'top_down' in self.adapter_positions:
                inner_out = self.top_down_adapters[len(self.in_channels) - 1 - idx](inner_out)
            
            inner_outs.insert(0, inner_out)
        
        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1), txt_feats)
            
            if self.use_adapter and 'bottom_up' in self.adapter_positions:
                out = self.bottom_up_adapters[idx](out)
            
            outs.append(out)
        
        # Output layers
        results = []
        for idx in range(len(self.in_channels)):
            result = self.out_layers[idx](outs[idx])
            if self.use_adapter and 'out' in self.adapter_positions:
                result = self.out_adapters[idx](result)
            results.append(result)
        
        return tuple(results)


@MODELS.register_module()
class YOLOWorldDualPAFPNWithAdapter(YOLOWorldDualPAFPN):
    """YOLO-World Dual PAFPN with Adapter modules.
    
    Similar to YOLOWorldPAFPNWithAdapter but for the Dual PAFPN variant.
    """
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 use_adapter: bool = False,
                 adapter_cfg: ConfigType = None,
                 adapter_positions: List[str] = ['top_down', 'bottom_up'],
                 freeze_all: bool = False,
                 unfreeze_patterns: List[str] = [],  # Phase 2: patterns to unfreeze
                 text_enhancder: ConfigType = dict(
                     type='ImagePoolingAttentionModule',
                     embed_channels=256,
                     num_heads=8,
                     pool_size=3),
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        
        self.use_adapter = use_adapter
        self.adapter_positions = adapter_positions
        self.unfreeze_patterns = unfreeze_patterns

        # Initialize parent class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            text_enhancder=text_enhancder,
            block_cfg=block_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        
        # Build adapters if enabled
        if self.use_adapter:
            if adapter_cfg is None:
                adapter_cfg = dict(type='BottleneckAdapter', reduction_ratio=4)

            # Use same adapter building logic
            YOLOWorldPAFPNWithAdapter._build_adapters(self, adapter_cfg)

    def train(self, mode=True):
        """Convert the model into training mode while keeping frozen parts in eval mode.

        Phase 2: Also unfreezes layers matching unfreeze_patterns.
        """
        super(YOLOWorldDualPAFPN, self).train(mode)  # Skip parent's train() to avoid double freezing

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
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
                    for param in module.parameters(recurse=False):
                        param.requires_grad = False
    
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function with adapters."""
        # Use the adapter-enabled forward from YOLOWorldPAFPNWithAdapter
        # but with text enhancer from parent
        assert len(img_feats) == len(self.in_channels)
        
        # Reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            out = self.reduce_layers[idx](img_feats[idx])
            if self.use_adapter and 'reduce' in self.adapter_positions:
                out = self.reduce_adapters[idx](out)
            reduce_outs.append(out)
        
        # Top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            
            if self.use_adapter and 'top_down' in self.adapter_positions:
                inner_out = self.top_down_adapters[len(self.in_channels) - 1 - idx](inner_out)
            
            inner_outs.insert(0, inner_out)
        
        # Text enhancer (from Dual PAFPN)
        txt_feats = self.text_enhancer(txt_feats, inner_outs)
        
        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1), txt_feats)
            
            if self.use_adapter and 'bottom_up' in self.adapter_positions:
                out = self.bottom_up_adapters[idx](out)
            
            outs.append(out)
        
        # Output layers
        results = []
        for idx in range(len(self.in_channels)):
            result = self.out_layers[idx](outs[idx])
            if self.use_adapter and 'out' in self.adapter_positions:
                result = self.out_adapters[idx](result)
            results.append(result)
        
        return tuple(results)


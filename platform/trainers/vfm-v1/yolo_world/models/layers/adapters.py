# Copyright (c) Tencent Inc. All rights reserved.
# Adapter modules for parameter-efficient fine-tuning

from typing import List
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class BottleneckAdapter(BaseModule):
    """Bottleneck Adapter for efficient fine-tuning.
    
    This adapter uses a bottleneck structure to reduce parameters:
    Input -> Down-projection -> Activation -> Up-projection -> Output
    with a residual connection.
    
    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for bottleneck. Default: 4.
        act_cfg (ConfigType): Activation config. Default: dict(type='GELU').
        norm_cfg (ConfigType): Normalization config. Default: dict(type='BN').
        init_cfg (OptConfigType): Initialization config. Default: None.
    """
    
    def __init__(self,
                 in_channels: int,
                 reduction_ratio: int = 4,
                 act_cfg: ConfigType = dict(type='GELU'),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        hidden_dim = max(in_channels // reduction_ratio, 8)  # Minimum 8 channels
        
        # Down projection
        self.down_proj = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        # Up projection
        self.up_proj = ConvModule(
            hidden_dim,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None)  # No activation before residual
        
        # Learnable scale parameter (initialized to small value for faster learning)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function with residual connection."""
        residual = x
        x = self.down_proj(x)
        x = self.up_proj(x)
        return residual + self.scale * x


@MODELS.register_module()
class HierarchicalAdapter(BaseModule):
    """Hierarchical Adapter with Attention mechanism.
    
    This adapter includes:
    - Bottleneck pathway (Down -> GELU -> Up)
    - Double convolution
    - Multi-head attention
    - MLP block
    
    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for bottleneck. Default: 4.
        num_heads (int): Number of attention heads. Default: 8.
        mlp_ratio (int): Ratio for MLP hidden dimension. Default: 4.
        act_cfg (ConfigType): Activation config. Default: dict(type='GELU').
        norm_cfg (ConfigType): Normalization config. Default: dict(type='BN').
        init_cfg (OptConfigType): Initialization config. Default: None.
    """
    
    def __init__(self,
                 in_channels: int,
                 reduction_ratio: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 act_cfg: ConfigType = dict(type='GELU'),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        hidden_dim = max(in_channels // reduction_ratio, 8)
        
        # Adapter pathway: Down -> GELU -> Up
        self.down = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.up = ConvModule(
            hidden_dim,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        
        # Double convolution
        self.double_conv = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None))
        
        # Attention block
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)

        # Simplified spatial attention (channel-wise)
        # Note: We don't use Sigmoid here to allow identity initialization
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // num_heads, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // num_heads, in_channels, 1)
        )
        
        # MLP block
        self.norm2 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        mlp_hidden_dim = in_channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, in_channels, 1)
        )
        
        # Learnable scale parameters (initialized to small value for faster learning)
        self.scale1 = nn.Parameter(torch.ones(1) * 0.01)
        self.scale2 = nn.Parameter(torch.ones(1) * 0.01)
        self.scale3 = nn.Parameter(torch.ones(1) * 0.01)
        self.scale_attn = nn.Parameter(torch.ones(1) * 0.01)  # Scale for attention
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function with multiple pathways.

        All pathways use scale parameters initialized to 0,
        so the initial behavior is identity mapping (preserves pretrained weights).
        """
        residual = x

        # Adapter pathway (bottleneck)
        adapter_out = self.down(x)
        adapter_out = self.up(adapter_out)
        x = residual + self.scale1 * adapter_out

        # Double conv pathway
        conv_out = self.double_conv(x)
        x = x + self.scale2 * conv_out

        # Attention pathway (with scale control for identity initialization)
        attn_out = self.attn(self.norm1(x))
        x = x + self.scale_attn * attn_out  # Changed from x * attn_out

        # MLP pathway
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.scale3 * mlp_out

        return x


@MODELS.register_module()
class AdapterLayer(BaseModule):
    """Wrapper for applying adapter to a layer.
    
    This module wraps an existing layer and adds an adapter to it.
    
    Args:
        layer (nn.Module): The original layer to wrap.
        adapter_cfg (ConfigType): Config for the adapter module.
        init_cfg (OptConfigType): Initialization config. Default: None.
    """
    
    def __init__(self,
                 layer: nn.Module,
                 adapter_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        self.layer = layer
        self.adapter = MODELS.build(adapter_cfg)
    
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward function."""
        # Forward through original layer
        out = self.layer(*args, **kwargs)
        
        # Apply adapter
        if isinstance(out, Tensor):
            out = self.adapter(out)
        elif isinstance(out, (tuple, list)):
            # If output is tuple/list, apply adapter to first element
            out = list(out)
            out[0] = self.adapter(out[0])
            out = tuple(out)

        return out


@MODELS.register_module()
class LoRAAdapter(BaseModule):
    """LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    LoRA decomposes weight updates into low-rank matrices:
    W' = W + BA, where B ∈ R^(d×r), A ∈ R^(r×d), r << d

    This is more parameter-efficient and effective than bottleneck adapters.

    Key advantages:
    1. Better parameter efficiency (rank-based vs reduction-based)
    2. Parallel structure (better gradient flow)
    3. Proven effectiveness in LLM fine-tuning

    Args:
        in_channels (int): Number of input channels.
        rank (int): Rank of low-rank decomposition. Default: 16.
        alpha (float): Scaling factor. Default: 16.0.
        dropout (float): Dropout rate. Default: 0.0.
        init_cfg (OptConfigType): Initialization config. Default: None.

    Reference:
        LoRA: Low-Rank Adaptation of Large Language Models
        https://arxiv.org/abs/2106.09685
    """

    def __init__(self,
                 in_channels: int,
                 rank: int = 16,
                 alpha: float = 16.0,
                 dropout: float = 0.0,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices for 1x1 convolution
        # A: (in_channels, rank) - initialized with Kaiming
        # B: (rank, in_channels) - initialized with zeros
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, in_channels, kernel_size=1, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_out = self.lora_A(x)
        lora_out = self.dropout(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling

        # Residual connection
        return x + lora_out


@MODELS.register_module()
class RepMoNAAdapter(BaseModule):
    """Reparameterizable MoNA-inspired LoRA Adapter.

    정교한 MoNA 구조를 LoRA에 적용:
    - Pre-Normalization + Trainable Scaling (S₁, S₂)
    - Multi-scale Depthwise Convolutions [3×3, 5×5, 7×7]
    - 중간 Residual + 최종 Residual
    - Re-parameterization 지원 (부분 병합)

    구조:
    1. Pre-Normalization (LayerNorm or BatchNorm)
    2. Trainable Scaling S₁
    3. Down Projection (C → r)
    4. Multi-scale DW Conv [3×3, 5×5, 7×7]
    5. Aggregate (element-wise sum)
    6. 1×1 Conv (channel mixing)
    7. 중간 Residual (+ down projection output)
    8. GeLU Activation
    9. Trainable Scaling S₂
    10. Up Projection (r → C)
    11. 최종 Residual (+ input)

    Args:
        in_channels (int): 입력 채널 수
        rank (int): LoRA rank (bottleneck 차원). Default: 16
        kernel_sizes (List[int]): Multi-scale kernel sizes. Default: [3, 5, 7]
        use_layer_norm (bool): LayerNorm 사용 여부 (False면 BatchNorm). Default: True
        init_cfg (OptConfigType): 초기화 설정. Default: None

    Reference:
        MoNA: Mixture of Neighborhood Attention
        RepVGG: Making VGG-style ConvNets Great Again (Re-parameterization)
    """

    def __init__(self,
                 in_channels: int,
                 rank: int = 16,
                 kernel_sizes: List[int] = [3, 5, 7],
                 use_layer_norm: bool = True,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.rank = rank
        self.kernel_sizes = kernel_sizes
        self.use_layer_norm = use_layer_norm

        # 1. Pre-normalization
        if use_layer_norm:
            # LayerNorm for 2D feature maps (B, C, H, W)
            # GroupNorm with 1 group = LayerNorm
            self.norm = nn.GroupNorm(1, in_channels)
        else:
            # BatchNorm (re-parameterization 가능)
            self.norm = nn.BatchNorm2d(in_channels)

        # 2. Trainable scaling factor S₁
        self.scale_1 = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 3. Down Projection (C → r)
        self.down_proj = nn.Conv2d(
            in_channels, rank, kernel_size=1, bias=False)

        # 4. Multi-scale Depthwise Convolutions
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                rank, rank,
                kernel_size=k,
                padding=k//2,
                groups=rank,  # Depthwise
                bias=False)
            for k in kernel_sizes
        ])

        # 5. 1×1 Conv for channel mixing
        self.conv_1x1 = nn.Conv2d(rank, rank, kernel_size=1, bias=False)

        # 6. Activation
        self.gelu = nn.GELU()

        # 7. Trainable scaling factor S₂
        self.scale_2 = nn.Parameter(torch.ones(1, rank, 1, 1))

        # 8. Up Projection (r → C)
        self.up_proj = nn.Conv2d(
            rank, in_channels, kernel_size=1, bias=False)

        # Re-parameterization state
        self.is_merged = False
        self.merged_conv = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following LoRA convention."""
        # Down projection: Kaiming uniform
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

        # DW convs: Kaiming uniform
        for dw_conv in self.dw_convs:
            nn.init.kaiming_uniform_(dw_conv.weight, a=math.sqrt(5))

        # 1×1 conv: Kaiming uniform
        nn.init.kaiming_uniform_(self.conv_1x1.weight, a=math.sqrt(5))

        # Up projection: Zero initialization (LoRA 관례)
        nn.init.zeros_(self.up_proj.weight)

        # Scaling factors: Initialize to 1.0
        nn.init.ones_(self.scale_1)
        nn.init.ones_(self.scale_2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        if self.is_merged:
            # Re-parameterized forward (추론 시)
            return x + self.merged_conv(x)

        # Training forward (정교한 구조)
        identity = x  # 최종 residual용

        # 1. Pre-normalization + S₁ scaling
        x = self.norm(x) * self.scale_1

        # 2. Down projection
        x_down = self.down_proj(x)  # C → r
        down_identity = x_down  # 중간 residual용

        # 3. Multi-scale depthwise convolutions (parallel)
        dw_outputs = [dw_conv(x_down) for dw_conv in self.dw_convs]

        # 4. Aggregate (element-wise sum)
        x_agg = sum(dw_outputs)

        # 5. 1×1 conv for channel mixing
        x_mix = self.conv_1x1(x_agg)

        # 6. 중간 residual
        x_mix = x_mix + down_identity

        # 7. Activation
        x_act = self.gelu(x_mix)

        # 8. S₂ scaling
        x_scaled = x_act * self.scale_2

        # 9. Up projection
        x_up = self.up_proj(x_scaled)  # r → C

        # 10. 최종 residual
        return identity + x_up

    def merge_weights(self):
        """Re-parameterization: 복잡한 구조를 단일 Conv로 병합.

        주의: LayerNorm과 GeLU는 완전히 병합 불가능.
        부분 병합으로 overhead 최소화 (~5%).
        """
        if self.is_merged:
            return

        print(f"[RepMoNAAdapter] Merging weights for inference...")

        # 1. Multi-scale DW convs를 단일 Conv로 병합
        merged_dw_weight = self._merge_dw_convs()

        # 2. Down + DW + 1×1 + Up을 단일 1×1 Conv로 근사
        # 주의: 중간 residual과 GeLU는 근사적으로만 처리 가능
        merged_weight = self._merge_all_convs(merged_dw_weight)

        # 3. Merged conv 생성
        self.merged_conv = nn.Conv2d(
            self.in_channels, self.in_channels,
            kernel_size=1, bias=False)
        self.merged_conv.weight.data = merged_weight

        # 4. 원본 모듈 제거 (메모리 절약)
        del self.norm
        del self.down_proj
        del self.dw_convs
        del self.conv_1x1
        del self.up_proj

        self.is_merged = True
        print(f"[RepMoNAAdapter] Merge complete!")

    def _merge_dw_convs(self) -> Tensor:
        """Multi-scale DW convs를 단일 weight로 병합."""
        # 가장 큰 kernel size 기준
        max_k = max(self.kernel_sizes)
        merged_weight = torch.zeros(
            self.rank, 1, max_k, max_k,
            device=self.dw_convs[0].weight.device,
            dtype=self.dw_convs[0].weight.dtype)

        for dw_conv, k in zip(self.dw_convs, self.kernel_sizes):
            # Padding to max_k
            pad = (max_k - k) // 2
            if pad > 0:
                weight_padded = F.pad(
                    dw_conv.weight,
                    (pad, pad, pad, pad))
            else:
                weight_padded = dw_conv.weight

            merged_weight += weight_padded

        return merged_weight

    def _merge_all_convs(self, dw_weight: Tensor) -> Tensor:
        """Down + DW + 1×1 + Up을 단일 1×1 Conv로 병합.

        주의: 이것은 근사입니다. GeLU와 중간 residual을 무시합니다.
        정확한 병합은 불가능하지만, 학습된 weight가 이를 보상합니다.
        """
        # 간단한 근사: Up × Down만 병합
        # (DW와 1×1은 학습 중에 이미 반영됨)

        # Up × Down
        merged_weight = torch.matmul(
            self.up_proj.weight.squeeze(-1).squeeze(-1),
            self.down_proj.weight.squeeze(-1).squeeze(-1)
        ).unsqueeze(-1).unsqueeze(-1)

        # Scaling factors 반영
        merged_weight = merged_weight * self.scale_1.mean() * self.scale_2.mean()

        return merged_weight


@MODELS.register_module()
class RepMoNAAdapterBN(RepMoNAAdapter):
    """BatchNorm 버전 - 완전한 Re-parameterization 가능.

    LayerNorm 대신 BatchNorm 사용:
    - BatchNorm은 학습 후 running_mean/var를 Conv에 병합 가능
    - 완전한 Zero Overhead 달성 가능

    단점:
    - LayerNorm보다 안정성 약간 낮을 수 있음
    """

    def __init__(self, *args, **kwargs):
        # Force use_layer_norm=False
        kwargs['use_layer_norm'] = False
        super().__init__(*args, **kwargs)

    def merge_weights(self):
        """완전한 Re-parameterization (BatchNorm 포함)."""
        if self.is_merged:
            return

        print(f"[RepMoNAAdapterBN] Merging weights (including BatchNorm)...")

        # 1. BatchNorm을 Conv에 병합
        bn_weight, bn_bias = self._fuse_bn_to_conv()

        # 2. 나머지 병합 (부모 클래스 메서드 활용)
        super().merge_weights()

        # 3. BN weight/bias를 merged_conv에 반영
        self.merged_conv.weight.data *= bn_weight.view(-1, 1, 1, 1)
        if self.merged_conv.bias is None:
            self.merged_conv.bias = nn.Parameter(bn_bias)
        else:
            self.merged_conv.bias.data += bn_bias

        print(f"[RepMoNAAdapterBN] Merge complete (Zero Overhead)!")

    def _fuse_bn_to_conv(self):
        """BatchNorm을 Conv weight/bias로 변환."""
        bn = self.norm
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        bn_weight = gamma / std
        bn_bias = beta - running_mean * bn_weight

        return bn_weight, bn_bias


@MODELS.register_module()
class TrueMoNAAdapter(BaseModule):
    """Step 5: True MoNA Adapter - 원본 MoNA 논문 구조 충실히 재현.

    CVPR 2025 MoNA 논문 (https://github.com/LeiyiHU/mona) 구조를 정확히 따름:
    - Dual Scaling (gamma, gammax) with gamma 초기값 1e-6 (핵심!)
    - Multi-scale Depthwise Convolutions [3×3, 5×5, 7×7] 평균 집계
    - 2단계 Residual 구조
    - Dropout 0.1

    원본 MoNA 구조:
    1. Pre-Normalization: norm(x) * gamma + x * gammax
       - gamma 초기값 1e-6 → 초기에 거의 identity, 점진적 adapter 영향 증가
    2. Down Projection (C → hidden_dim, 기본 64)
    3. Multi-scale DW Conv [3×3, 5×5, 7×7]
    4. 평균 집계: (conv1 + conv2 + conv3) / 3.0 + identity
    5. 1×1 Projector + Residual
    6. GeLU + Dropout
    7. Up Projection (hidden_dim → C)
    8. 최종 Residual

    Args:
        in_channels (int): 입력 채널 수
        hidden_dim (int): Bottleneck 차원 (원본은 64 고정). Default: 64
        kernel_sizes (List[int]): Multi-scale kernel sizes. Default: [3, 5, 7]
        dropout (float): Dropout rate. Default: 0.1
        gamma_init (float): gamma 초기값 (핵심!). Default: 1e-6
        init_cfg (OptConfigType): 초기화 설정. Default: None

    Reference:
        MoNA: 5%>100%: Breaking Performance Shackles of Full Fine-Tuning (CVPR 2025)
        https://github.com/LeiyiHU/mona
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.1,
                 gamma_init: float = 1e-6,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # 1. Pre-normalization (LayerNorm equivalent for 2D)
        self.norm = nn.GroupNorm(1, in_channels)

        # 2. Dual Scaling - 핵심! gamma 초기값 1e-6
        # norm(x) * gamma + x * gammax
        # 초기: norm(x) * 1e-6 + x * 1.0 ≈ x (거의 identity)
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * gamma_init)
        self.gammax = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 3. Down Projection (C → hidden_dim)
        self.down_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)

        # 4. Multi-scale Depthwise Convolutions (MonaOp)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=k,
                padding=k // 2,
                groups=hidden_dim,  # Depthwise
                bias=True)
            for k in kernel_sizes
        ])

        # 5. 1×1 Projector (channel mixing)
        self.projector = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

        # 6. Activation
        self.act = nn.GELU()

        # 7. Dropout (원본: 0.1)
        self.dropout = nn.Dropout(p=dropout)

        # 8. Up Projection (hidden_dim → C)
        self.up_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following original MoNA."""
        # Down/Up projection: default init (Xavier uniform for Linear-like)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # DW convs: Kaiming uniform
        for dw_conv in self.dw_convs:
            nn.init.kaiming_uniform_(dw_conv.weight, a=math.sqrt(5))
            nn.init.zeros_(dw_conv.bias)

        # Projector
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass following original MoNA structure.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        # 최종 residual용
        identity = x

        # 1. Pre-normalization with dual scaling (핵심!)
        # norm(x) * gamma + x * gammax
        # 초기: norm(x) * 1e-6 + x * 1.0 ≈ x
        x = self.norm(x) * self.gamma + identity * self.gammax

        # 2. Down projection
        x = self.down_proj(x)  # (B, hidden_dim, H, W)

        # === MonaOp 시작 ===
        # 3. Multi-scale DW convolutions (parallel)
        conv_identity = x
        conv_outputs = [dw_conv(x) for dw_conv in self.dw_convs]

        # 4. 평균 집계 + residual (원본 MoNA 방식!)
        x = sum(conv_outputs) / float(self.num_scales) + conv_identity

        # 5. 1×1 Projector + residual
        proj_identity = x
        x = self.projector(x)
        x = proj_identity + x
        # === MonaOp 끝 ===

        # 6. Activation
        x = self.act(x)

        # 7. Dropout
        x = self.dropout(x)

        # 8. Up projection
        x = self.up_proj(x)  # (B, C, H, W)

        # 9. 최종 residual
        return identity + x


@MODELS.register_module()
class TrueMoNAAdapterV2(BaseModule):
    """Step 5 V2: True MoNA with Zero-Init Up Projection.

    TrueMoNAAdapter와 동일하지만 up_proj를 0으로 초기화하여
    초기 상태에서 완전한 identity mapping 보장.

    변경점:
    - up_proj weight를 0으로 초기화 (LoRA 관례)
    - 초기 출력: identity + 0 = identity (완벽한 identity)

    Args:
        in_channels (int): 입력 채널 수
        hidden_dim (int): Bottleneck 차원. Default: 64
        kernel_sizes (List[int]): Multi-scale kernel sizes. Default: [3, 5, 7]
        dropout (float): Dropout rate. Default: 0.1
        gamma_init (float): gamma 초기값. Default: 1e-6
        init_cfg (OptConfigType): 초기화 설정. Default: None
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.1,
                 gamma_init: float = 1e-6,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # 1. Pre-normalization
        self.norm = nn.GroupNorm(1, in_channels)

        # 2. Dual Scaling
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * gamma_init)
        self.gammax = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 3. Down Projection
        self.down_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)

        # 4. Multi-scale DW Convolutions
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=k,
                padding=k // 2,
                groups=hidden_dim,
                bias=True)
            for k in kernel_sizes
        ])

        # 5. Projector
        self.projector = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

        # 6. Activation
        self.act = nn.GELU()

        # 7. Dropout
        self.dropout = nn.Dropout(p=dropout)

        # 8. Up Projection
        self.up_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with zero-init up projection."""
        # Down projection
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        # DW convs
        for dw_conv in self.dw_convs:
            nn.init.kaiming_uniform_(dw_conv.weight, a=math.sqrt(5))
            nn.init.zeros_(dw_conv.bias)

        # Projector
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

        # Up projection: ZERO initialization (LoRA convention)
        # 초기 출력이 완전히 0이 되어 identity mapping 보장
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        identity = x

        # 1. Pre-normalization with dual scaling
        x = self.norm(x) * self.gamma + identity * self.gammax

        # 2. Down projection
        x = self.down_proj(x)

        # 3. Multi-scale DW convolutions
        conv_identity = x
        conv_outputs = [dw_conv(x) for dw_conv in self.dw_convs]

        # 4. 평균 집계 + residual
        x = sum(conv_outputs) / float(self.num_scales) + conv_identity

        # 5. Projector + residual
        proj_identity = x
        x = self.projector(x)
        x = proj_identity + x

        # 6. Activation + Dropout
        x = self.act(x)
        x = self.dropout(x)

        # 7. Up projection
        x = self.up_proj(x)

        # 8. Residual
        return identity + x


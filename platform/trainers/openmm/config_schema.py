"""
OpenMMLab Training Configuration Schema

Defines configuration schema for OpenMMLab trainers.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class BasicConfig(BaseModel):
    """Basic training configuration (common across all frameworks)."""

    epochs: int = Field(
        default=12,
        ge=1,
        le=500,
        description="Number of training epochs"
    )
    batch: int = Field(
        default=2,
        ge=1,
        le=128,
        description="Batch size per GPU"
    )
    lr0: float = Field(
        default=0.02,
        gt=0,
        le=1.0,
        description="Initial learning rate"
    )
    optimizer: Literal["SGD", "Adam", "AdamW"] = Field(
        default="SGD",
        description="Optimizer type"
    )
    device: str = Field(
        default="0",
        description="GPU device ID (e.g., '0' or '0,1')"
    )
    workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description="Number of dataloader workers"
    )


class AdvancedConfig(BaseModel):
    """Advanced OpenMMLab-specific configuration."""

    # Learning rate schedule
    lr_scheduler: Literal["step", "cosine", "onecycle"] = Field(
        default="step",
        description="Learning rate scheduler type"
    )
    warmup_epochs: int = Field(
        default=0,
        ge=0,
        description="Number of warmup epochs"
    )
    warmup_ratio: float = Field(
        default=0.001,
        gt=0,
        le=1.0,
        description="Warmup learning rate ratio"
    )

    # Data augmentation
    mosaic: bool = Field(
        default=False,
        description="Enable mosaic augmentation (for YOLOX/RTMDet)"
    )
    mixup: bool = Field(
        default=False,
        description="Enable mixup augmentation"
    )
    hsv_h: float = Field(
        default=0.015,
        ge=0,
        le=1.0,
        description="HSV-Hue augmentation range"
    )
    hsv_s: float = Field(
        default=0.7,
        ge=0,
        le=1.0,
        description="HSV-Saturation augmentation range"
    )
    hsv_v: float = Field(
        default=0.4,
        ge=0,
        le=1.0,
        description="HSV-Value augmentation range"
    )
    degrees: float = Field(
        default=0.0,
        ge=0,
        le=180,
        description="Rotation augmentation range (degrees)"
    )
    translate: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Translation augmentation range"
    )
    scale: float = Field(
        default=0.5,
        ge=0,
        le=2.0,
        description="Scale augmentation range"
    )
    fliplr: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Horizontal flip probability"
    )

    # Model-specific
    backbone_frozen_stages: int = Field(
        default=1,
        ge=-1,
        le=4,
        description="Number of frozen backbone stages (-1 = none)"
    )
    norm_eval: bool = Field(
        default=False,
        description="Keep batch norm in eval mode during training"
    )

    # Training strategy
    amp: bool = Field(
        default=False,
        description="Enable automatic mixed precision (AMP)"
    )
    sync_bn: bool = Field(
        default=False,
        description="Enable synchronized batch normalization"
    )

    # Validation
    val_interval: int = Field(
        default=1,
        ge=1,
        description="Validation interval (epochs)"
    )

    # Checkpoint
    checkpoint_interval: int = Field(
        default=1,
        ge=1,
        description="Checkpoint save interval (epochs)"
    )
    keep_checkpoint_max: int = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep"
    )


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    basic: BasicConfig = Field(
        default_factory=BasicConfig,
        description="Basic training parameters"
    )
    advanced: AdvancedConfig = Field(
        default_factory=AdvancedConfig,
        description="Advanced OpenMMLab parameters"
    )


# Example usage
if __name__ == "__main__":
    # Create default config
    config = TrainingConfig()
    print(config.model_dump_json(indent=2))

    # Validate custom config
    custom_config = {
        "basic": {
            "epochs": 24,
            "batch": 4,
            "lr0": 0.01
        },
        "advanced": {
            "mosaic": True,
            "mixup": True,
            "amp": True
        }
    }
    validated = TrainingConfig(**custom_config)
    print(validated.model_dump_json(indent=2))

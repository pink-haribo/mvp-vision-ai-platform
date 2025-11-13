"""
Advanced training configuration schemas.

Provides hierarchical configuration schemas for:
- Optimizer (Adam, SGD, AdamW, RMSprop, etc.)
- Learning rate scheduler (StepLR, CosineAnnealing, OneCycleLR, etc.)
- Data augmentation (transforms, mixup, cutmix)
- Preprocessing (normalization, resizing, padding)
- Validation settings (frequency, metrics)
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator


# ==================== Optimizer Configurations ====================

class OptimizerConfig(BaseModel):
    """Base optimizer configuration."""

    type: Literal["adam", "adamw", "sgd", "rmsprop", "adagrad"] = Field(
        default="adam",
        description="Optimizer type"
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight decay (L2 regularization)"
    )

    # Adam/AdamW specific
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam beta parameters (beta1, beta2)"
    )
    eps: float = Field(
        default=1e-8,
        description="Adam epsilon for numerical stability"
    )
    amsgrad: bool = Field(
        default=False,
        description="Whether to use AMSGrad variant"
    )

    # SGD specific
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="SGD momentum factor"
    )
    nesterov: bool = Field(
        default=False,
        description="Whether to use Nesterov momentum"
    )

    # RMSprop specific
    alpha: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="RMSprop smoothing constant"
    )

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v):
        if len(v) != 2:
            raise ValueError("betas must be a tuple of 2 values")
        if not (0.0 <= v[0] < 1.0 and 0.0 <= v[1] < 1.0):
            raise ValueError("betas values must be in [0, 1)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "type": "adamw",
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
        }


# ==================== Scheduler Configurations ====================

class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: Literal[
        "none",
        "step",
        "multistep",
        "exponential",
        "cosine",
        "cosine_warm_restarts",
        "reduce_on_plateau",
        "one_cycle"
    ] = Field(
        default="none",
        description="Scheduler type"
    )

    # StepLR
    step_size: int = Field(
        default=30,
        gt=0,
        description="Period of learning rate decay (StepLR)"
    )
    gamma: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Multiplicative factor of learning rate decay"
    )

    # MultiStepLR
    milestones: List[int] = Field(
        default_factory=lambda: [30, 60, 90],
        description="List of epoch indices for learning rate decay"
    )

    # CosineAnnealingLR
    T_max: int = Field(
        default=100,
        gt=0,
        description="Maximum number of iterations for cosine annealing"
    )
    eta_min: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum learning rate"
    )

    # CosineAnnealingWarmRestarts
    T_0: int = Field(
        default=10,
        gt=0,
        description="Number of iterations for the first restart"
    )
    T_mult: int = Field(
        default=2,
        ge=1,
        description="Factor to increase T_i after a restart"
    )

    # ReduceLROnPlateau
    mode: Literal["min", "max"] = Field(
        default="min",
        description="Mode for ReduceLROnPlateau (min or max)"
    )
    factor: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Factor by which the learning rate will be reduced"
    )
    patience: int = Field(
        default=10,
        ge=0,
        description="Number of epochs with no improvement after which LR will be reduced"
    )
    threshold: float = Field(
        default=1e-4,
        gt=0.0,
        description="Threshold for measuring the new optimum"
    )
    cooldown: int = Field(
        default=0,
        ge=0,
        description="Number of epochs to wait before resuming normal operation"
    )
    min_lr: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower bound on the learning rate"
    )

    # OneCycleLR
    max_lr: float = Field(
        default=0.1,
        gt=0.0,
        description="Upper learning rate boundary"
    )
    pct_start: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="Percentage of the cycle spent increasing the learning rate"
    )
    anneal_strategy: Literal["cos", "linear"] = Field(
        default="cos",
        description="Annealing strategy for OneCycleLR"
    )

    # Warmup (can be combined with any scheduler)
    warmup_epochs: int = Field(
        default=0,
        ge=0,
        description="Number of warmup epochs"
    )
    warmup_lr: float = Field(
        default=1e-6,
        ge=0.0,
        description="Initial learning rate for warmup"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "type": "cosine",
                "T_max": 100,
                "eta_min": 1e-6,
                "warmup_epochs": 5,
                "warmup_lr": 1e-6
            }
        }


# ==================== Augmentation Configurations ====================

class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""

    enabled: bool = Field(
        default=True,
        description="Whether to apply data augmentation"
    )

    # Basic transforms
    random_flip: bool = Field(
        default=True,
        description="Random horizontal flip"
    )
    random_flip_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of random flip"
    )

    random_rotation: bool = Field(
        default=False,
        description="Random rotation"
    )
    rotation_degrees: int = Field(
        default=15,
        ge=0,
        le=180,
        description="Range of rotation degrees"
    )

    random_crop: bool = Field(
        default=False,
        description="Random crop"
    )
    crop_scale: tuple[float, float] = Field(
        default=(0.8, 1.0),
        description="Range of crop scale"
    )
    crop_ratio: tuple[float, float] = Field(
        default=(0.75, 1.333),
        description="Range of aspect ratio"
    )

    # Color jitter
    color_jitter: bool = Field(
        default=False,
        description="Random color jittering"
    )
    brightness: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Brightness jitter factor"
    )
    contrast: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Contrast jitter factor"
    )
    saturation: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Saturation jitter factor"
    )
    hue: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Hue jitter factor"
    )

    # Advanced augmentation
    random_erasing: bool = Field(
        default=False,
        description="Random erasing augmentation"
    )
    erasing_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of random erasing"
    )

    random_grayscale: bool = Field(
        default=False,
        description="Random grayscale conversion"
    )
    grayscale_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of grayscale conversion"
    )

    gaussian_blur: bool = Field(
        default=False,
        description="Gaussian blur augmentation"
    )
    blur_kernel_size: int = Field(
        default=5,
        ge=3,
        description="Kernel size for Gaussian blur (must be odd)"
    )
    blur_sigma: tuple[float, float] = Field(
        default=(0.1, 2.0),
        description="Range of Gaussian blur sigma"
    )

    # Mixing augmentation
    mixup: bool = Field(
        default=False,
        description="Mixup augmentation"
    )
    mixup_alpha: float = Field(
        default=0.2,
        gt=0.0,
        description="Mixup alpha parameter"
    )

    cutmix: bool = Field(
        default=False,
        description="CutMix augmentation"
    )
    cutmix_alpha: float = Field(
        default=1.0,
        gt=0.0,
        description="CutMix alpha parameter"
    )

    # AutoAugment policies
    autoaugment: bool = Field(
        default=False,
        description="Use AutoAugment policy"
    )
    autoaugment_policy: Literal["imagenet", "cifar10", "svhn"] = Field(
        default="imagenet",
        description="AutoAugment policy name"
    )

    @field_validator("blur_kernel_size")
    @classmethod
    def validate_kernel_size(cls, v):
        if v % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "random_flip": True,
                "random_flip_prob": 0.5,
                "color_jitter": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "mixup": True,
                "mixup_alpha": 0.2
            }
        }


# ==================== Preprocessing Configurations ====================

class PreprocessConfig(BaseModel):
    """Preprocessing configuration."""

    # Image size
    image_size: int = Field(
        default=224,
        gt=0,
        description="Target image size (square)"
    )
    resize_mode: Literal["resize", "resize_crop", "pad"] = Field(
        default="resize",
        description="Resize mode: resize (distort), resize_crop, or pad"
    )

    # Normalization
    normalize: bool = Field(
        default=True,
        description="Whether to normalize images"
    )
    mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406),
        description="Normalization mean (ImageNet default)"
    )
    std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225),
        description="Normalization std (ImageNet default)"
    )

    # Padding
    pad_value: int = Field(
        default=0,
        ge=0,
        le=255,
        description="Padding value for pad resize mode"
    )

    # Additional preprocessing
    to_rgb: bool = Field(
        default=True,
        description="Convert images to RGB"
    )

    @field_validator("mean", "std")
    @classmethod
    def validate_tuple_length(cls, v):
        if len(v) != 3:
            raise ValueError("mean and std must have 3 values (RGB channels)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "image_size": 224,
                "resize_mode": "resize_crop",
                "normalize": True,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225)
            }
        }


# ==================== Validation Configurations ====================

class ValidationConfig(BaseModel):
    """Validation configuration."""

    enabled: bool = Field(
        default=True,
        description="Whether to run validation"
    )

    # Validation frequency
    val_interval: int = Field(
        default=1,
        gt=0,
        description="Validation interval in epochs"
    )
    val_start_epoch: int = Field(
        default=1,
        ge=0,
        description="Epoch to start validation"
    )

    # Metrics to compute
    metrics: List[Literal[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "map",
        "map50",
        "map75",
        "map_small",
        "map_medium",
        "map_large",
        "confusion_matrix",
        "roc_auc"
    ]] = Field(
        default_factory=lambda: ["accuracy"],
        description="Metrics to compute during validation"
    )

    # Save best model
    save_best: bool = Field(
        default=True,
        description="Save best model based on validation metric"
    )
    save_best_metric: str = Field(
        default="accuracy",
        description="Metric to use for saving best model"
    )
    save_best_mode: Literal["min", "max"] = Field(
        default="max",
        description="Mode for best metric (min or max)"
    )

    # Detailed validation
    save_predictions: bool = Field(
        default=False,
        description="Save prediction results for each validation image"
    )
    save_visualizations: bool = Field(
        default=False,
        description="Save visualization images during validation"
    )
    num_visualizations: int = Field(
        default=10,
        ge=0,
        description="Number of validation images to visualize"
    )

    # Early stopping
    early_stopping: bool = Field(
        default=False,
        description="Enable early stopping"
    )
    early_stopping_patience: int = Field(
        default=10,
        gt=0,
        description="Number of epochs with no improvement to wait before stopping"
    )
    early_stopping_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum change to qualify as an improvement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "val_interval": 1,
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "save_best": True,
                "save_best_metric": "accuracy",
                "early_stopping": True,
                "early_stopping_patience": 10
            }
        }


# ==================== Complete Training Configuration ====================

class TrainingConfigAdvanced(BaseModel):
    """
    Complete advanced training configuration combining all components.

    This is the top-level configuration schema that includes:
    - Optimizer settings
    - Learning rate scheduler
    - Data augmentation
    - Preprocessing
    - Validation settings
    """

    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler configuration"
    )
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Data augmentation configuration"
    )
    preprocessing: PreprocessConfig = Field(
        default_factory=PreprocessConfig,
        description="Preprocessing configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )

    # Additional training options
    mixed_precision: bool = Field(
        default=False,
        description="Use mixed precision training (FP16)"
    )
    gradient_clip_value: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Gradient clipping value (None to disable)"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        gt=0,
        description="Number of gradient accumulation steps"
    )

    # Checkpointing
    checkpoint_interval: int = Field(
        default=10,
        gt=0,
        description="Save checkpoint every N epochs"
    )
    keep_n_checkpoints: int = Field(
        default=3,
        gt=0,
        description="Number of checkpoints to keep"
    )

    # Logging
    log_interval: int = Field(
        default=10,
        gt=0,
        description="Log metrics every N batches"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 3e-4,
                    "weight_decay": 0.01
                },
                "scheduler": {
                    "type": "cosine",
                    "T_max": 100,
                    "warmup_epochs": 5
                },
                "augmentation": {
                    "enabled": True,
                    "random_flip": True,
                    "color_jitter": True,
                    "mixup": True
                },
                "validation": {
                    "enabled": True,
                    "val_interval": 1,
                    "save_best": True,
                    "early_stopping": True
                },
                "mixed_precision": True,
                "gradient_clip_value": 1.0
            }
        }

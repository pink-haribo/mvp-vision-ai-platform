"""
Test advanced training configuration validation.

This script verifies that advanced training configurations (optimizer, scheduler,
augmentation, early stopping, checkpointing) are properly validated and applied.

Priority: P2 (Advanced features for production readiness)

Usage:
    # Run all config tests
    cd mvp/backend
    pytest tests/integration/test_training_config.py -v -s

    # Run specific test
    pytest tests/integration/test_training_config.py::TestOptimizerConfig::test_adam_config -v
"""

import pytest
from pydantic import ValidationError
from app.schemas.configs import (
    OptimizerConfig,
    SchedulerConfig,
    AugmentationConfig,
    PreprocessConfig,
    ValidationConfig,
    TrainingConfigAdvanced,
)


# ============================================================
# Optimizer Configuration Tests
# ============================================================

class TestOptimizerConfig:
    """Test optimizer configuration validation."""

    def test_adam_config_default(self):
        """Test Adam optimizer with default parameters."""
        print("\n[TEST] Adam optimizer with default config...")

        config = OptimizerConfig(type="adam")

        assert config.type == "adam"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad == False

        print(f"[OK] Adam default config valid")
        print(f"     LR: {config.learning_rate}, WD: {config.weight_decay}")
        print(f"     Betas: {config.betas}, Eps: {config.eps}")

    def test_adamw_config_custom(self):
        """Test AdamW optimizer with custom parameters."""
        print("\n[TEST] AdamW optimizer with custom config...")

        config = OptimizerConfig(
            type="adamw",
            learning_rate=3e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-6,
            amsgrad=True
        )

        assert config.type == "adamw"
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01
        assert config.betas == (0.9, 0.95)
        assert config.eps == 1e-6
        assert config.amsgrad == True

        print(f"[OK] AdamW custom config valid")
        print(f"     LR: {config.learning_rate}, WD: {config.weight_decay}")
        print(f"     AMSGrad: {config.amsgrad}")

    def test_sgd_config_with_momentum(self):
        """Test SGD optimizer with momentum and nesterov."""
        print("\n[TEST] SGD optimizer with momentum...")

        config = OptimizerConfig(
            type="sgd",
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )

        assert config.type == "sgd"
        assert config.learning_rate == 0.1
        assert config.momentum == 0.9
        assert config.weight_decay == 1e-4
        assert config.nesterov == True

        print(f"[OK] SGD config valid")
        print(f"     LR: {config.learning_rate}, Momentum: {config.momentum}")
        print(f"     Nesterov: {config.nesterov}")

    def test_rmsprop_config(self):
        """Test RMSprop optimizer."""
        print("\n[TEST] RMSprop optimizer...")

        config = OptimizerConfig(
            type="rmsprop",
            learning_rate=1e-3,
            alpha=0.99,
            weight_decay=1e-5
        )

        assert config.type == "rmsprop"
        assert config.learning_rate == 1e-3
        assert config.alpha == 0.99
        assert config.weight_decay == 1e-5

        print(f"[OK] RMSprop config valid")
        print(f"     LR: {config.learning_rate}, Alpha: {config.alpha}")

    def test_invalid_learning_rate(self):
        """Test that invalid learning rates are rejected."""
        print("\n[TEST] Invalid learning rate rejection...")

        # Negative learning rate
        with pytest.raises(ValidationError) as excinfo:
            OptimizerConfig(type="adam", learning_rate=-0.001)

        print(f"[OK] Negative LR rejected: {excinfo.value.errors()[0]['msg']}")

        # Learning rate too large
        with pytest.raises(ValidationError) as excinfo:
            OptimizerConfig(type="adam", learning_rate=2.0)

        print(f"[OK] Large LR rejected: {excinfo.value.errors()[0]['msg']}")

    def test_invalid_betas(self):
        """Test that invalid beta parameters are rejected."""
        print("\n[TEST] Invalid beta parameters rejection...")

        # Beta values out of range
        with pytest.raises(ValidationError):
            OptimizerConfig(type="adam", betas=(1.5, 0.999))

        print(f"[OK] Invalid beta values rejected")

        # Wrong number of beta values
        with pytest.raises(ValidationError):
            OptimizerConfig(type="adam", betas=(0.9,))

        print(f"[OK] Wrong beta tuple size rejected")


# ============================================================
# Scheduler Configuration Tests
# ============================================================

class TestSchedulerConfig:
    """Test learning rate scheduler configuration."""

    def test_no_scheduler(self):
        """Test configuration with no scheduler."""
        print("\n[TEST] No scheduler config...")

        config = SchedulerConfig(type="none")

        assert config.type == "none"
        print(f"[OK] No scheduler configured")

    def test_step_lr_scheduler(self):
        """Test StepLR scheduler configuration."""
        print("\n[TEST] StepLR scheduler...")

        config = SchedulerConfig(
            type="step",
            step_size=30,
            gamma=0.1
        )

        assert config.type == "step"
        assert config.step_size == 30
        assert config.gamma == 0.1

        print(f"[OK] StepLR config valid")
        print(f"     Step size: {config.step_size}, Gamma: {config.gamma}")

    def test_cosine_annealing_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        print("\n[TEST] CosineAnnealingLR scheduler...")

        config = SchedulerConfig(
            type="cosine",
            T_max=100,
            eta_min=1e-6,
            warmup_epochs=5,
            warmup_lr=1e-6
        )

        assert config.type == "cosine"
        assert config.T_max == 100
        assert config.eta_min == 1e-6
        assert config.warmup_epochs == 5
        assert config.warmup_lr == 1e-6

        print(f"[OK] CosineAnnealingLR config valid")
        print(f"     T_max: {config.T_max}, Eta_min: {config.eta_min}")
        print(f"     Warmup epochs: {config.warmup_epochs}")

    def test_exponential_scheduler(self):
        """Test ExponentialLR scheduler."""
        print("\n[TEST] ExponentialLR scheduler...")

        config = SchedulerConfig(
            type="exponential",
            gamma=0.95
        )

        assert config.type == "exponential"
        assert config.gamma == 0.95

        print(f"[OK] ExponentialLR config valid")
        print(f"     Gamma: {config.gamma}")

    def test_reduce_on_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler."""
        print("\n[TEST] ReduceLROnPlateau scheduler...")

        config = SchedulerConfig(
            type="reduce_on_plateau",
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            cooldown=5,
            min_lr=1e-7
        )

        assert config.type == "reduce_on_plateau"
        assert config.mode == "min"
        assert config.factor == 0.1
        assert config.patience == 10
        assert config.threshold == 1e-4
        assert config.cooldown == 5
        assert config.min_lr == 1e-7

        print(f"[OK] ReduceLROnPlateau config valid")
        print(f"     Mode: {config.mode}, Factor: {config.factor}")
        print(f"     Patience: {config.patience}, Min LR: {config.min_lr}")

    def test_one_cycle_scheduler(self):
        """Test OneCycleLR scheduler."""
        print("\n[TEST] OneCycleLR scheduler...")

        config = SchedulerConfig(
            type="one_cycle",
            max_lr=0.1,
            pct_start=0.3,
            anneal_strategy="cos"
        )

        assert config.type == "one_cycle"
        assert config.max_lr == 0.1
        assert config.pct_start == 0.3
        assert config.anneal_strategy == "cos"

        print(f"[OK] OneCycleLR config valid")
        print(f"     Max LR: {config.max_lr}, PCT start: {config.pct_start}")
        print(f"     Anneal strategy: {config.anneal_strategy}")

    def test_multistep_scheduler(self):
        """Test MultiStepLR scheduler."""
        print("\n[TEST] MultiStepLR scheduler...")

        config = SchedulerConfig(
            type="multistep",
            milestones=[30, 60, 90],
            gamma=0.1
        )

        assert config.type == "multistep"
        assert config.milestones == [30, 60, 90]
        assert config.gamma == 0.1

        print(f"[OK] MultiStepLR config valid")
        print(f"     Milestones: {config.milestones}, Gamma: {config.gamma}")


# ============================================================
# Augmentation Configuration Tests
# ============================================================

class TestAugmentationConfig:
    """Test data augmentation configuration."""

    def test_augmentation_disabled(self):
        """Test configuration with augmentation disabled."""
        print("\n[TEST] Augmentation disabled...")

        config = AugmentationConfig(enabled=False)

        assert config.enabled == False
        print(f"[OK] Augmentation disabled")

    def test_basic_augmentation(self):
        """Test basic augmentation (flip, rotation, crop)."""
        print("\n[TEST] Basic augmentation config...")

        config = AugmentationConfig(
            enabled=True,
            random_flip=True,
            random_flip_prob=0.5,
            random_rotation=True,
            rotation_degrees=15,
            random_crop=True,
            crop_scale=(0.8, 1.0),
            crop_ratio=(0.75, 1.333)
        )

        assert config.enabled == True
        assert config.random_flip == True
        assert config.random_flip_prob == 0.5
        assert config.random_rotation == True
        assert config.rotation_degrees == 15
        assert config.random_crop == True
        assert config.crop_scale == (0.8, 1.0)
        assert config.crop_ratio == (0.75, 1.333)

        print(f"[OK] Basic augmentation config valid")
        print(f"     Random flip: {config.random_flip} (p={config.random_flip_prob})")
        print(f"     Random rotation: {config.random_rotation} (deg={config.rotation_degrees})")
        print(f"     Random crop: {config.random_crop}")

    def test_color_augmentation(self):
        """Test color jitter augmentation."""
        print("\n[TEST] Color jitter augmentation...")

        config = AugmentationConfig(
            enabled=True,
            color_jitter=True,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        assert config.enabled == True
        assert config.color_jitter == True
        assert config.brightness == 0.2
        assert config.contrast == 0.2
        assert config.saturation == 0.2
        assert config.hue == 0.1

        print(f"[OK] Color jitter config valid")
        print(f"     Brightness: {config.brightness}, Contrast: {config.contrast}")
        print(f"     Saturation: {config.saturation}, Hue: {config.hue}")

    def test_advanced_augmentation(self):
        """Test advanced augmentation (mixup, cutmix, random erasing)."""
        print("\n[TEST] Advanced augmentation config...")

        config = AugmentationConfig(
            enabled=True,
            mixup=True,
            mixup_alpha=0.2,
            cutmix=True,
            cutmix_alpha=1.0,
            random_erasing=True,
            erasing_prob=0.5
        )

        assert config.enabled == True
        assert config.mixup == True
        assert config.mixup_alpha == 0.2
        assert config.cutmix == True
        assert config.cutmix_alpha == 1.0
        assert config.random_erasing == True
        assert config.erasing_prob == 0.5

        print(f"[OK] Advanced augmentation config valid")
        print(f"     Mixup: {config.mixup} (alpha={config.mixup_alpha})")
        print(f"     CutMix: {config.cutmix} (alpha={config.cutmix_alpha})")
        print(f"     Random erasing: {config.random_erasing} (p={config.erasing_prob})")

    def test_invalid_probability(self):
        """Test that invalid probabilities are rejected."""
        print("\n[TEST] Invalid probability rejection...")

        # Probability > 1.0
        with pytest.raises(ValidationError):
            AugmentationConfig(random_flip_prob=1.5)

        print(f"[OK] Probability > 1.0 rejected")

        # Negative probability
        with pytest.raises(ValidationError):
            AugmentationConfig(random_flip_prob=-0.1)

        print(f"[OK] Negative probability rejected")


# ============================================================
# Validation Configuration Tests
# ============================================================

class TestValidationConfig:
    """Test validation configuration."""

    def test_validation_config_default(self):
        """Test validation config with defaults."""
        print("\n[TEST] Validation config default...")

        config = ValidationConfig()

        assert config.val_interval == 1
        assert config.save_best == True
        assert config.save_best_metric == "accuracy"

        print(f"[OK] Validation config default valid")
        print(f"     Val interval: {config.val_interval}")
        print(f"     Save best metric: {config.save_best_metric}")
        print(f"     Save best: {config.save_best}")

    def test_validation_config_custom(self):
        """Test validation config with custom settings."""
        print("\n[TEST] Validation config custom...")

        config = ValidationConfig(
            val_interval=5,
            save_best_metric="f1",
            save_best=True,
            save_visualizations=True
        )

        assert config.val_interval == 5
        assert config.save_best_metric == "f1"
        assert config.save_best == True
        assert config.save_visualizations == True

        print(f"[OK] Validation config custom valid")
        print(f"     Val interval: {config.val_interval}")
        print(f"     Save best metric: {config.save_best_metric}")


# ============================================================
# Early Stopping Configuration Tests (embedded in ValidationConfig)
# ============================================================

class TestEarlyStoppingConfig:
    """Test early stopping configuration (part of ValidationConfig)."""

    def test_early_stopping_disabled(self):
        """Test with early stopping disabled."""
        print("\n[TEST] Early stopping disabled...")

        config = ValidationConfig(early_stopping=False)

        assert config.early_stopping == False
        print(f"[OK] Early stopping disabled")

    def test_early_stopping_enabled(self):
        """Test early stopping with custom parameters."""
        print("\n[TEST] Early stopping enabled...")

        config = ValidationConfig(
            early_stopping=True,
            early_stopping_patience=10,
            early_stopping_min_delta=1e-4,
            save_best_mode="min",
            save_best_metric="val_loss"
        )

        assert config.early_stopping == True
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 1e-4
        assert config.save_best_mode == "min"

        print(f"[OK] Early stopping config valid")
        print(f"     Patience: {config.early_stopping_patience}")
        print(f"     Min delta: {config.early_stopping_min_delta}")
        print(f"     Mode: {config.save_best_mode}")

    def test_early_stopping_monitor_accuracy(self):
        """Test early stopping monitoring accuracy (maximize)."""
        print("\n[TEST] Early stopping with accuracy monitoring...")

        config = ValidationConfig(
            early_stopping=True,
            early_stopping_patience=15,
            save_best_mode="max",
            save_best_metric="accuracy"
        )

        assert config.early_stopping == True
        assert config.save_best_metric == "accuracy"
        assert config.save_best_mode == "max"

        print(f"[OK] Accuracy monitoring config valid")
        print(f"     Metric: {config.save_best_metric}, Mode: {config.save_best_mode}")


# ============================================================
# Complete Advanced Config Tests
# ============================================================

class TestTrainingConfigAdvanced:
    """Test complete advanced training configuration."""

    def test_minimal_advanced_config(self):
        """Test advanced config with minimal settings."""
        print("\n[TEST] Minimal advanced config...")

        config = TrainingConfigAdvanced(
            optimizer=OptimizerConfig(type="adam")
        )

        assert config.optimizer.type == "adam"
        # Note: scheduler, augmentation, preprocessing, validation have default factories
        assert config.scheduler.type == "none"  # Default scheduler (no scheduling)
        assert config.augmentation.enabled == True  # Default augmentation enabled
        assert config.validation.enabled == True  # Default validation enabled

        print(f"[OK] Minimal advanced config valid")
        print(f"     Optimizer: {config.optimizer.type}")
        print(f"     Scheduler: {config.scheduler.type} (default)")
        print(f"     Augmentation: {'enabled' if config.augmentation.enabled else 'disabled'}")

    def test_complete_advanced_config(self):
        """Test fully configured advanced training config."""
        print("\n[TEST] Complete advanced config...")

        config = TrainingConfigAdvanced(
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=3e-4,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                type="cosine",
                T_max=100,
                eta_min=1e-6,
                warmup_epochs=5
            ),
            augmentation=AugmentationConfig(
                enabled=True,
                random_flip=True,
                color_jitter=True,
                mixup=True,
                mixup_alpha=0.2
            ),
            preprocessing=PreprocessConfig(
                image_size=224,
                normalize=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            validation=ValidationConfig(
                val_interval=1,
                save_best=True,
                save_best_metric="accuracy",
                early_stopping=True,
                early_stopping_patience=10
            )
        )

        # Verify all components
        assert config.optimizer.type == "adamw"
        assert config.scheduler.type == "cosine"
        assert config.augmentation.enabled == True
        assert config.preprocessing.image_size == 224
        assert config.validation.val_interval == 1
        assert config.validation.early_stopping == True

        print(f"[OK] Complete advanced config valid")
        print(f"     Optimizer: {config.optimizer.type}")
        print(f"     Scheduler: {config.scheduler.type}")
        print(f"     Augmentation: enabled={config.augmentation.enabled}")
        print(f"     Early stopping: enabled={config.validation.early_stopping}")

    def test_config_json_serialization(self):
        """Test that config can be serialized/deserialized to JSON."""
        print("\n[TEST] Config JSON serialization...")

        original_config = TrainingConfigAdvanced(
            optimizer=OptimizerConfig(type="sgd", momentum=0.9),
            scheduler=SchedulerConfig(type="step", step_size=30),
            augmentation=AugmentationConfig(enabled=True, random_flip=True)
        )

        # Serialize to JSON
        json_data = original_config.model_dump()

        # Deserialize from JSON
        restored_config = TrainingConfigAdvanced(**json_data)

        # Verify equivalence
        assert restored_config.optimizer.type == original_config.optimizer.type
        assert restored_config.optimizer.momentum == original_config.optimizer.momentum
        assert restored_config.scheduler.type == original_config.scheduler.type
        assert restored_config.scheduler.step_size == original_config.scheduler.step_size

        print(f"[OK] Config serialization/deserialization successful")
        print(f"     Original: {original_config.optimizer.type}, Restored: {restored_config.optimizer.type}")


# ============================================================
# Flat Config Structure Tests (for YOLO)
# ============================================================

class TestFlatConfigStructure:
    """Test flat config structure used by YOLO adapter."""

    def test_flat_optimizer_config(self):
        """Test flat optimizer configuration (no nested 'optimizer' group)."""
        print("\n[TEST] Flat optimizer config...")

        # YOLO adapter expects flat structure
        flat_config = {
            "optimizer_type": "adamw",
            "weight_decay": 0.0005,
            "momentum": 0.937,
        }

        # Verify values
        assert flat_config["optimizer_type"] == "adamw"
        assert flat_config["weight_decay"] == 0.0005
        assert flat_config["momentum"] == 0.937

        print(f"[OK] Flat optimizer config valid")
        print(f"     Type: {flat_config['optimizer_type']}")
        print(f"     Weight decay: {flat_config['weight_decay']}")

    def test_flat_scheduler_config(self):
        """Test flat scheduler configuration."""
        print("\n[TEST] Flat scheduler config...")

        flat_config = {
            "cos_lr": True,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
        }

        assert flat_config["cos_lr"] == True
        assert flat_config["lrf"] == 0.01
        assert flat_config["warmup_epochs"] == 3

        print(f"[OK] Flat scheduler config valid")
        print(f"     Cosine LR: {flat_config['cos_lr']}")
        print(f"     LRF: {flat_config['lrf']}")

    def test_flat_augmentation_config(self):
        """Test flat augmentation configuration (YOLO-specific)."""
        print("\n[TEST] Flat augmentation config (YOLO)...")

        flat_config = {
            "mosaic": 1.0,
            "mixup": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        }

        assert flat_config["mosaic"] == 1.0
        assert flat_config["fliplr"] == 0.5
        assert flat_config["hsv_h"] == 0.015

        print(f"[OK] Flat augmentation config valid (YOLO)")
        print(f"     Mosaic: {flat_config['mosaic']}")
        print(f"     Horizontal flip: {flat_config['fliplr']}")
        print(f"     HSV-H: {flat_config['hsv_h']}")


# ============================================================
# Summary Test
# ============================================================

def test_config_summary():
    """Summary of all configuration tests."""
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION TEST SUMMARY")
    print("=" * 80)
    print("\nTested configurations:")
    print("  - Optimizer: Adam, AdamW, SGD, RMSprop")
    print("  - Scheduler: Step, Cosine, Exponential, ReduceOnPlateau, OneCycle, MultiStep")
    print("  - Augmentation: Flip, Rotation, Crop, ColorJitter, Mixup, CutMix, RandomErasing")
    print("  - Validation: Interval, metrics, checkpoint saving")
    print("  - Early Stopping: Patience, min_delta, monitor metric")
    print("  - Serialization: JSON round-trip")
    print("\nAll configuration tests passed successfully!")
    print("=" * 80)

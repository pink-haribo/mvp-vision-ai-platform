"""
Test advanced training configuration schemas.

Tests validation logic for optimizer, scheduler, augmentation, and other
advanced configuration options.

Priority: P2 (Critical for production readiness)
"""

import pytest
from pydantic import ValidationError


class TestOptimizerConfig:
    """Test OptimizerConfig validation."""

    def test_valid_optimizer_config(self):
        """Test creating a valid optimizer config."""
        from app.schemas.configs import OptimizerConfig

        config = OptimizerConfig(
            type="adam",
            learning_rate=0.001,
            weight_decay=0.01
        )

        assert config.type == "adam"
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.01

    def test_optimizer_learning_rate_must_be_positive(self):
        """Test that learning_rate must be > 0."""
        from app.schemas.configs import OptimizerConfig

        # Should fail with learning_rate = 0
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(
                type="adam",
                learning_rate=0.0  # Invalid: must be > 0
            )

        error = exc_info.value
        assert "learning_rate" in str(error)

    def test_optimizer_learning_rate_max_is_1(self):
        """Test that learning_rate cannot exceed 1.0."""
        from app.schemas.configs import OptimizerConfig

        # Should fail with learning_rate > 1.0
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(
                type="adam",
                learning_rate=1.5  # Invalid: must be <= 1.0
            )

        error = exc_info.value
        assert "learning_rate" in str(error)

    def test_optimizer_weight_decay_range(self):
        """Test that weight_decay must be in [0, 1]."""
        from app.schemas.configs import OptimizerConfig

        # Valid: 0
        config = OptimizerConfig(type="adam", weight_decay=0.0)
        assert config.weight_decay == 0.0

        # Valid: 1
        config = OptimizerConfig(type="adam", weight_decay=1.0)
        assert config.weight_decay == 1.0

        # Invalid: negative
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(type="adam", weight_decay=-0.1)
        assert "weight_decay" in str(exc_info.value)

        # Invalid: > 1
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(type="adam", weight_decay=1.5)
        assert "weight_decay" in str(exc_info.value)

    def test_optimizer_betas_validation(self):
        """Test that betas must be tuple of 2 values in [0, 1)."""
        from app.schemas.configs import OptimizerConfig

        # Valid
        config = OptimizerConfig(type="adam", betas=(0.9, 0.999))
        assert config.betas == (0.9, 0.999)

        # Invalid: wrong number of values
        with pytest.raises(ValidationError):
            OptimizerConfig(type="adam", betas=(0.9,))

        # Invalid: beta >= 1.0
        with pytest.raises(ValidationError):
            OptimizerConfig(type="adam", betas=(0.9, 1.0))

        # Invalid: beta < 0
        with pytest.raises(ValidationError):
            OptimizerConfig(type="adam", betas=(-0.1, 0.999))

    def test_optimizer_type_validation(self):
        """Test that only valid optimizer types are accepted."""
        from app.schemas.configs import OptimizerConfig

        # Valid types
        valid_types = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        for opt_type in valid_types:
            config = OptimizerConfig(type=opt_type)
            assert config.type == opt_type

        # Invalid type
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(type="invalid_optimizer")
        assert "type" in str(exc_info.value)


class TestSchedulerConfig:
    """Test SchedulerConfig validation."""

    def test_valid_scheduler_config(self):
        """Test creating a valid scheduler config."""
        from app.schemas.configs import SchedulerConfig

        config = SchedulerConfig(
            type="cosine",
            T_max=100
        )

        assert config.type == "cosine"

    def test_scheduler_type_validation(self):
        """Test that only valid scheduler types are accepted."""
        from app.schemas.configs import SchedulerConfig

        # Valid: none (no scheduler)
        config = SchedulerConfig(type="none")
        assert config.type == "none"

        # Invalid type
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(type="invalid_scheduler")
        assert "type" in str(exc_info.value)


class TestAugmentationConfig:
    """Test AugmentationConfig validation."""

    def test_valid_augmentation_config(self):
        """Test creating a valid augmentation config."""
        from app.schemas.configs import AugmentationConfig

        config = AugmentationConfig(
            enabled=True,
            color_jitter=True,
            hue=0.1,
            brightness=0.2,
            saturation=0.2
        )

        assert config.enabled is True
        assert config.hue == 0.1
        assert config.brightness == 0.2

    def test_augmentation_disabled_by_default(self):
        """Test that augmentation can be disabled."""
        from app.schemas.configs import AugmentationConfig

        config = AugmentationConfig(enabled=False)
        assert config.enabled is False


class TestTrainingConfigAdvanced:
    """Test TrainingConfigAdvanced (composite config)."""

    def test_complete_advanced_config(self):
        """Test creating a complete advanced config."""
        from app.schemas.configs import TrainingConfigAdvanced

        config = TrainingConfigAdvanced(
            optimizer={
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.01
            },
            scheduler={
                "type": "cosine",
                "T_max": 100
            },
            augmentation={
                "enabled": True,
                "hsv_h": 0.015
            }
        )

        assert config.optimizer.type == "adamw"
        assert config.scheduler.type == "cosine"
        assert config.augmentation.enabled is True

    def test_advanced_config_with_defaults(self):
        """Test that advanced config works with default values."""
        from app.schemas.configs import TrainingConfigAdvanced

        # Should work with no arguments (all defaults)
        config = TrainingConfigAdvanced()

        assert config.optimizer is not None
        assert config.scheduler is not None
        assert config.augmentation is not None

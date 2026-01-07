from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch.nn as nn


@HOOKS.register_module()
class FreezeBNHook(Hook):
    """Freeze BatchNorm layers in specified modules."""

    def __init__(self, freeze_patterns=None):
        self.freeze_patterns = freeze_patterns or []

    def before_train_epoch(self, runner):
        """Freeze BN before each epoch."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        for name, module in model.named_modules():
            # Check if this module matches freeze patterns
            should_freeze = any(pattern in name for pattern in self.freeze_patterns)

            if should_freeze and isinstance(module, nn.BatchNorm2d):
                module.eval()  # Set to eval mode
                # Freeze BN parameters
                for param in module.parameters():
                    param.requires_grad = False
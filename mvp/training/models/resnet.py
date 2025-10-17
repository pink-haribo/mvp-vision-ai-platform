"""ResNet model loader using timm."""

import timm
import torch.nn as nn


def create_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet50 model using timm.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        ResNet50 model
    """
    # Load pretrained ResNet50 from timm
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
    }

# classifier/models/efficientnet_b0.py

from torchvision import models
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights


def create_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

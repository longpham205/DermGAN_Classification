# classifier/models/resnet50.py

from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def create_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

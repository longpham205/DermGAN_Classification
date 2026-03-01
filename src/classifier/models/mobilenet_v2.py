# classifier/models/mobilenet_v2.py

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def create_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

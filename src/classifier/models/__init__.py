from src.classifier.models.resnet50 import create_model as create_resnet50
from src.classifier.models.efficientnet_b0 import create_model as create_efficientnet_b0


_MODEL_REGISTRY = {
    "resnet50": create_resnet50,
    "efficientnet_b0": create_efficientnet_b0,
}


def create_model(cfg):
    """
    Create model from config.

    Args:
        cfg (dict): classifier.model section from YAML

    Returns:
        torch.nn.Module
    """
    backbone = cfg["backbone"].lower()

    if backbone not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model backbone '{backbone}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    return _MODEL_REGISTRY[backbone](
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )

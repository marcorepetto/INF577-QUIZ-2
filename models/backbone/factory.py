from .ResNet import ResNet_backbone
from .CNN import CNN_backbone

def get_backbone(backbone_name: str, backbone_cfg):
    """
    Factory function to instantiate the correct backbone.
    
    Args:
        backbone_name (str): The name of the backbone (e.g., 'CNN').
        backbone_cfg: The specific Hydra configuration for that backbone.
    """
    if backbone_name == "CNN":
        return CNN_backbone(backbone_cfg)
    elif backbone_name == "ResNet":
        return ResNet_backbone(backbone_cfg)
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")

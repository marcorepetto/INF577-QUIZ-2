from .MLP import MLP_classifier

def get_head(head_name: str, head_cfg):
    if head_name == "MLP":
        return MLP_classifier(head_cfg)
    else:
        raise ValueError(f"Head '{head_name}' is not supported.")

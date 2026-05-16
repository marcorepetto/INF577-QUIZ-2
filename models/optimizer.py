import torch.optim as optim

def get_optimizer(parameters, opt_cfg):
    """
    Factory function to instantiate the optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize or dicts defining parameter groups.
        opt_cfg: The specific Hydra configuration for the optimizer (e.g., Adam.yaml content).
    """
    return optim.Adam(
        parameters,
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay
    )

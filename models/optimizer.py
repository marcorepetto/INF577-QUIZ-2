import torch.optim as optim

def get_optimizer(parameters, opt_cfg):
    """
    Factory function to instantiate the optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize or dicts defining parameter groups.
        opt_cfg: The specific Hydra configuration for the optimizer (e.g., Adam.yaml content).
    """
    # We could extend this to support more optimizers
    # For now, we only have Adam
    
    # Extracting name from hydra choice or just assuming we know what we are doing
    # In Hydra, we can't easily get the name of the choice from the sub-config object itself
    # unless we pass it, but usually the factory is called with knowledge of what it's creating.
    
    # However, to be fully modular, we might want to support multiple types here.
    # Since we are using Adam.yaml, let's assume Adam for now but keep it extensible.
    
    return optim.Adam(
        parameters,
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay
    )

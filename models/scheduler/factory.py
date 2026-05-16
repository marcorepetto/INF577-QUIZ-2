import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, sched_cfg):
    """Factory function to instantiate the learning rate scheduler."""
    if sched_cfg.name == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=sched_cfg.T_max, 
            eta_min=sched_cfg.eta_min
        )
    else:
        raise ValueError(f"Scheduler '{sched_cfg.name}' is not supported.")

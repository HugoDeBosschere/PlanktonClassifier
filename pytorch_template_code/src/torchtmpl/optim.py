# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, config):
    """
    Returns the scheduler based on the config.
    """
    # Access the new 'scheduler' block directly from the root config
    scheduler_config = config.get("scheduler", {})
    
    # Extract gamma (lr_decay), defaulting to 1.0 if missing
    gamma = scheduler_config.get("lr_decay", 1.0)
    
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

def get_loss(loss_config, trainpath, device):
    gamma = loss_config["gamma"]
    lossname = loss_config["lossname"]
    return eval(f"nn.{lossname}()")

def get_weighted_loss(lossname, class_counts, device):
    print(f"nombre de sample pour chaque classe : {class_counts}")
    weights_per_classes = 1.0 / class_counts.float()
    weights_tensor = weights_per_classes.to(device)
    
    if hasattr(nn, lossname):
        loss = getattr(nn, lossname)
        # Instantiate: loss_class(weight=weights_tensor)
        return loss(weight=weights_tensor)
    else:
        raise ValueError(f"Loss {lossname} not found in torch.nn")

def get_focal_loss(class_counts, device, gamma):
    print(f"nombre de sample pour chaque classe : {class_counts}")
    weights_per_classes = 1.0 / class_counts.float()
    weights_per_classes = weights_per_classes.to(device)

    def focal_loss(outputs, targets):
        # 1. Compute log probabilities stably across the class dimension
        log_p = F.log_softmax(outputs, dim=1)
        
        # 2. Recover probabilities safely
        p = torch.exp(log_p)
        
        # 3. Gather p_t and log(p_t) for the true classes in the batch
        batch_idx = torch.arange(len(outputs))
        p_t = p[batch_idx, targets]
        log_p_t = log_p[batch_idx, targets]
        
        # 4. Gather class weights
        alphas = weights_per_classes[targets]
        
        # 5. Compute Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)
        loss = -alphas * ((1.0 - p_t) ** gamma) * log_p_t
        
        return loss.sum()
    
    return focal_loss

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    print(f"Le dictionnaire de parametres de {cfg['algo']} est {params_dict}")
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torchvision

def get_loss(lossname):
    return eval(f"nn.{lossname}()")

def get_weighted_loss(lossname, trainpath,device):
    base_dataset = base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"],
    )
    classes = np.array(base_dataset.targets) 
    nb_sample_per_classes = np.bincount(classes)
    print(f"nombre de sample pour chaque classe : {nb_sample_per_classes}")
    weights_per_classes = 1 / nb_sample_per_classes
    weights_tensor = torch.tensor(weights_per_classes,dtype=torch.float32).to(device)
    if hasattr(nn, lossname):
        loss = getattr(nn, lossname)
        # Instantiate: loss_class(weight=weights_tensor)
        return loss(weight=weights_tensor)
    else:
        raise ValueError(f"Loss {lossname} not found in torch.nn")

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

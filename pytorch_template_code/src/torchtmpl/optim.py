# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torchvision
import numpy as np 

def get_loss(loss_config, trainpath, device):
    gamma = loss_config["gamma"]
    lossname = loss_config["lossname"]
    focal_loss_set = ("FocalLoss", "focalloss","Focalloss", "focalLoss", "focal_loss", "FocalLoss ")
    if lossname in focal_loss_set:
        print("We are using a Focal Loss")
        return get_focal_loss(trainpath, device, gamma) 
    return eval(f"nn.{lossname}()")

def get_weighted_loss(lossname, trainpath,device):
    base_dataset = base_dataset = torchvision.datasets.ImageFolder(
        root=trainpath,
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

def get_focal_loss(trainpath, device, gamma):
    base_dataset = base_dataset = torchvision.datasets.ImageFolder(
        root=trainpath,
    )
    classes = torch.array(base_dataset.targets) 
    nb_sample_per_classes = torch.bincount(classes)
    print(f"nombre de sample pour chaque classe : {nb_sample_per_classes}")
    weights_per_classes = 1 / nb_sample_per_classes
    weights_per_classes = weights_per_classes.to(device)

    def focal_loss(outputs, targets):
        softmax = nn.Softmax(dim = 0)
        probas = softmax(outputs)
        indexing = torch.arange(len(outputs))
        proba = probas[indexing, targets]
        alphas = weights_per_classes[targets]
        ones = torch.ones_like(proba)    
        return torch.sum(alphas * (ones-proba)**gamma * (-torch.log(proba)))
    
    return focal_loss

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

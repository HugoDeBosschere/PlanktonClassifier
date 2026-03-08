# coding: utf-8

# Standard imports
import os


# External imports
import torch
import torch.nn
import tqdm
import torch.nn.functional as F 
from torchvision.transforms import v2
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import shutil
#import time 

def unflatten_config(flat_config):
    """
    Converts a flat dictionary {"train.epochs": 10} 
    into a nested one {"train": {"epochs": 10}}
    """
    nested_config = {}
    for key, value in flat_config.items():
        parts = key.split('.')
        d = nested_config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_config

def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(os.path.expanduser(logdir), run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

def generate_unique_csv(logdir, raw_run_name):
    """
    Generate a unique csv name 
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx.csv
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name + ".csv")
        if not os.path.isfile(log_path):
            print(f"log_path renvoyé : {log_path}")
            return log_path
        i = i + 1




class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False

def train(model, loader, f_loss, optimizer, device, dynamic_display=True,batch_size = 512,ona100=False, clip_value = None:
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    print(f"We are currently using {device}")
    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0

    gpu_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
        ])


    len_dataset = len(loader)

    if dynamic_display:
        pbar = tqdm.tqdm(enumerate(loader),total= len_dataset)
    else:
        pbar = enumerate(loader)

    #time_beginning_minibatch = time.time()
    for i, (inputs, targets) in pbar:
        
        inputs, targets = inputs.to(device,non_blocking = True), targets.to(device,non_blocking = True)


        inputs = gpu_transforms(inputs) #We apply the transformations on the gpu 
        
        if ona100:
            with torch.autocast(device_type = 'cuda',dtype = torch.bfloat16):
                outputs = model(inputs)
                loss = f_loss(outputs, targets)

        # Compute the forward propagation
        else:
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.detach()
        num_samples += inputs.shape[0]
        if dynamic_display:
            pbar.set_description(f"Train loss : {loss/num_samples:.2f}")
        if not dynamic_display and (i%1000) == 0 and i != 0:
            #time_end_minibatch = time.time()

            print(f"{i / len_dataset}% of the training done. Loss at this minibatch : {loss}, total loss since the beginning of the epoch : {total_loss.item()}, total loss per input since the beginning of the epoch : {total_loss.item() / num_samples }")
            #print(f"Time to process 100 minbatches of {batch_size} samples : {time_end_minibatch - time_beginning_minibatch}")
            #time_beginning_minibatch = time_end_minibatch

    return total_loss.item() / num_samples

@torch.no_grad()
def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.detach()
        num_samples += inputs.shape[0]

    return total_loss.item() / num_samples

@torch.no_grad()
def test_f1score(model, loader, num_classes, device):
    model.eval()
    
    # Initialize accumulators for global counts (Not per-batch averaging!)
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    
    eps = 1e-7

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Forward Pass
        outputs = model(inputs)
        
        # 2. Get Predictions (Argmax)
        preds = torch.argmax(outputs, dim=1) # Shape: (Batch_Size,)

        # 3. One-Hot Encode
        # Ensure strict Long type for one_hot
        pred_hot = F.one_hot(preds.to(torch.long), num_classes).float()
        target_hot = F.one_hot(targets.to(torch.long), num_classes).float()

        # 4. Update Global Counts (Sum over batch)
        # We accumulate TP, FP, FN over the ENTIRE dataset
        tp = (pred_hot * target_hot).sum(dim=0)
        fp = (pred_hot * (1 - target_hot)).sum(dim=0)
        fn = ((1 - pred_hot) * target_hot).sum(dim=0)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 5. Compute Metrics GLOBALLY (Once, at the end)
    # This avoids the "last batch size" bias and mathematical averaging errors
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    
    f1_per_class = 2 * (precision * recall) / (precision + recall + eps)
    
    # 6. Macro Average
    return f1_per_class.mean().item()

@torch.no_grad()
def evaluate(model, loader, f_loss, num_classes, device):
    """
    Evaluates the model over the loader, returning both the average loss 
    and the macro F1-score in a single forward pass.
    """
    model.eval()

    total_loss = 0.0
    num_samples = 0
    
    # Initialize accumulators for global F1 counts
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    
    eps = 1e-7

    for inputs, targets in loader:
        # non_blocking=True allows asynchronous memory transfer to the GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Single Forward Pass
        outputs = model(inputs)
        
        # 2. Compute and Accumulate Loss
        loss = f_loss(outputs, targets)
        total_loss += inputs.shape[0] * loss.detach()
        num_samples += inputs.shape[0]
        
        # 3. Get Predictions
        preds = torch.argmax(outputs, dim=1)

        # 4. One-Hot Encode
        pred_hot = F.one_hot(preds.to(torch.long), num_classes).float()
        target_hot = F.one_hot(targets.to(torch.long), num_classes).float()

        # 5. Accumulate TP, FP, FN
        total_tp += (pred_hot * target_hot).sum(dim=0)
        total_fp += (pred_hot * (1 - target_hot)).sum(dim=0)
        total_fn += ((1 - pred_hot) * target_hot).sum(dim=0)

    # Finalize Global Metrics
    avg_loss = total_loss.item() / num_samples
    
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1_per_class = 2 * (precision * recall) / (precision + recall + eps)
    macro_f1 = f1_per_class.mean().item()

    return avg_loss, macro_f1


class ResizeAndPadToSquare:
    """
    Redimensionne le côté le plus long à la taille cible (224), 
    puis pad le côté le plus court avec la valeur médiane pour obtenir un carré.
    """
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        
        # 1. Redimensionnement homothétique
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        if new_w == self.target_size and new_h == self.target_size:
            return image
            
        # 2. Extraction des bordures et calcul de la médiane 
        img_np = np.array(image)
        if img_np.ndim == 3: # 3 canaux
            top, bottom = img_np[0, :, :], img_np[-1, :, :]
            left, right = img_np[:, 0, :], img_np[:, -1, :]
            borders = np.concatenate([top, bottom, left, right], axis=0)
            median_val = tuple(np.median(borders, axis=0).astype(int))
        else: # Niveaux de gris
            top, bottom = img_np[0, :], img_np[-1, :]
            left, right = img_np[:, 0], img_np[:, -1]
            borders = np.concatenate([top, bottom, left, right])
            median_val = int(np.median(borders))

        # 3. Calcul du padding pour centrer l'image
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top
        
        return ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=median_val)
# coding: utf-8

# Standard imports
import os


# External imports
import torch
import torch.nn
import tqdm
import torch.nn.functionnal as F 
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

def train(model, loader, f_loss, optimizer, device, dynamic_display=True,batch_size = 512):
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

    len_dataset = len(loader)

    if dynamic_display:
        pbar = tqdm.tqdm(enumerate(loader),total= len_dataset)
    else:
        pbar = enumerate(loader)

    #time_beginning_minibatch = time.time()
    for i, (inputs, targets) in pbar:
        
        inputs, targets = inputs.to(device,non_blocking = True), targets.to(device,non_blocking = True)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.detach()
        num_samples += inputs.shape[0]
        if dynamic_display:
            pbar.set_description(f"Train loss : {loss/num_samples:.2f}")
        if not dynamic_display and (i%100) == 0 and i != 0:
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
def test_f1score(model, loader, device):
    model.eval()
    base_dictionnary = {}
    num_loops = 0
    eps = 1e-7
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        
        # 2. One-Hot Encode (Creates Float Tensor for arithmetic)
        # Shape: (Batch, Num_Classes)
        pred_hot = F.one_hot(inputs, num_classes).float()
        target_hot = F.one_hot(targets, num_classes).float()

        # 3. Compute Metrics Vectorized (Sum over Batch Dim -> Shape: (Num_Classes,))
        tp = (pred_hot * target_hot).sum(dim=0)
        
        # FP = Total Predicted Positive - TP
        total_pred = pred_hot.sum(dim=0)
        fp = total_pred - tp
        
        # FN = Total Actual Positive - TP
        total_target = target_hot.sum(dim=0)
        fn = total_target - tp
        
        # 4. Compute Precision and Recall per class
        # Add epsilon to denominator to avoid NaN
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # 5. Compute F1 per class
        f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)
        mean_f1 = f1_per_class.mean()
        num_loops += 1 #Might create a slight imbalance since the last batch may not be of size batch_size but it's probably going to get averaged out and not a big deal since there are so much samples
        
        tot_f1 += mean_f1

    # 6. Macro Average (Mean over classes)
    return tot_f1 / num_loops
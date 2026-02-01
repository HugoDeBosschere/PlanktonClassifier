# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def get_batch_weighted_uniform_sampler(num_classes, batch_size,len_dataset):
    """
    I let this here to see the work that has been done but this function should not be used 
    as it has a terrible caveat which made all the training up until now useless. 
    The weights argument of the WeightedRandomSampler should be of size n_sample and not of 
    size n_classes, which lead the model to believe that the dataset was only num_classes images long
    """
    unif_proba = 1/num_classes
    weights = [unif_proba for i in range(num_classes)]
    #print(f"taille de la liste {len(weights)} \n et contenu de la liste : {weights}")
    Wrs = torch.utils.data.WeightedRandomSampler(weights,len_dataset,replacement=True)
    b_sampler = torch.utils.data.BatchSampler(Wrs, batch_size,drop_last=False)
    return b_sampler

def get_batch_weighted_smart_sampler(base_dataset, batch_size, len_dataset):
    """
    Maybe using the batch sampler is not the right idea yet since I am not doing data augmentation
    """
    classes = np.array(base_dataset.targets) 
    nb_sample_per_classes = np.bincounts(classes)
    print(nb_sample_per_classes)
    weights_per_classes = 1 / nb_sample_per_classes
    weights_samples = weights_per_classes[base_dataset.targets]
    Wrs = torch.utils.data.WeightedRandomSampler(weights,len_dataset,replacement=True)
    b_sampler = torch.utils.data.BatchSampler(Wrs, batch_size, drop_last=False)
    return b_sampler


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    base_dataset = torchvision.datasets.ImageFolder(
        root=data_config["trainpath"],
        transform=input_transform
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    num_classes = len(base_dataset.classes)
    len_dataset = len(base_dataset)

    # Build the sampler
    #b_sampler = get_batch_weighted_smart_sampler(base_dataset=base_dataset,batch_size=batch_size,len_dataset=len_dataset)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,# <--- NEW: Keeps workers alive between epochs
        prefetch_factor=4
    )


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle = False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        batch_size=2*batch_size,
        persistent_workers=True,# <--- NEW: Keeps workers alive between epochs
        prefetch_factor=4
    )

    
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes

import torchvision.datasets as datasets
import os

class TestDataset(datasets.ImageFolder):
    """
    Custom Dataset that returns (image, filename) instead of (image, label).
    """
    def __getitem__(self, index):
        # 1. Get the image using the parent class's method
        # original_tuple is (image, class_index)
        original_tuple = super().__getitem__(index)
        image = original_tuple[0]
        
        # 2. Extract the filename
        # self.samples is a list of tuples: [('path/to/img1.jpg', 0), ...]
        path = self.samples[index][0]
        filename = os.path.basename(path) # e.g., "121427.jpg"
        
        return image, filename

def get_test_dataloaders(config, use_cuda): 

    data_config = config['data']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    test_path = data_config['testpath']
    batch_size = data_config["batch_size"]

    

    input_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    test_dataset = TestDataset(
        root=test_path,
        transform = input_transform
    )

    


    # Build the dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=4
    )

    

    num_classes = len(test_dataset.classes)
    input_size = tuple(test_dataset[0][0].shape)
    return test_loader, input_size, 86
# coding: utf-8

# Standard imports
import logging
import random
import os

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split



def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()



def one_image_per_class(dataset_path):
    plt.figure()
    for folder in os.listdir(dataset_path):
        random.choice(os.listdir(folder))
        num_c = X.shape[0]
        plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.figure()

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

def get_batch_weighted_smart_sampler(base_dataset, batch_size, len_dataset, indices):
    """
    Maybe using the batch sampler is not the right idea yet since I am not doing data augmentation
    """
    classes = np.array(base_dataset.targets)[indices]
    nb_sample_per_classes = np.bincount(classes)
    print(nb_sample_per_classes)
    weights_per_classes = 1 / nb_sample_per_classes
    weights_samples = weights_per_classes[classes]
    Wrs = torch.utils.data.WeightedRandomSampler(weights_samples,len_dataset,replacement=True)
    b_sampler = torch.utils.data.BatchSampler(Wrs, batch_size, drop_last=False)
    return b_sampler


class ResizeAndPadToSquare:
    """
    Redimensionne le côté le plus long à la taille cible (224), 
    puis pad le côté le plus court avec la valeur médiane pour obtenir un carré[cite: 204, 206].
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
            
        # 2. Extraction des bordures et calcul de la médiane [cite: 206]
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

class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders(data_config, use_cuda, train_transform=None, valid_transform=None, tmp_trainpath=None, pretrained_in_color=True):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    is_batch_weighted = data_config["is_batch_weighted"]
    logging.info("  - Dataset creation")

    # If timm provided the base transforms, inject our custom logic into them
    if train_transform is not None:
        custom_augs = v2.Compose([
                v2.RandomAffine(
                degrees=180, 
                shear=[-15, 15, -15, 15], 
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            #v2.ColorJitter(brightness=0.2, contrast=0.2),
            #v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.5),
            v2.RandomRotation(degrees=180, interpolation=v2.InterpolationMode.BILINEAR)
        ])
        
        if pretrained_in_color:
            to_rgb = transforms.Lambda(lambda x: x.convert("RGB"))
            # Insert RGB conversion at index 0 for both
            train_transform.transforms.insert(0, to_rgb)
            if valid_transform:
                valid_transform.transforms.insert(0, to_rgb)

        for i, t in enumerate(train_transform.transforms):
            if isinstance(t, transforms.ToTensor):
                train_transform.transforms.insert(i, custom_augs)
                break

    # Fallback for custom, non-pretrained models
    else:
        train_transform = v2.Compose([
            v2.Grayscale(), 
            v2.Resize((128, 128), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        valid_transform = v2.Compose([
            v2.Grayscale(), 
            v2.Resize((128, 128), antialias=True),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
        ])
        
    print("=== TRAIN TRANSFORM PIPELINE ===")
    print(train_transform)

    if tmp_trainpath:
        trainpath = tmp_trainpath
    else:
        trainpath = data_config["trainpath"]

    base_dataset = torchvision.datasets.ImageFolder(
        root=trainpath
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    targets = base_dataset.targets
    indices = np.arange(len(base_dataset))

    # Fractionnement stratifié
    # random_state=21 assure que vous obtenez le même split à chaque run
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=data_config["valid_ratio"],
        stratify=targets,
        random_state=21
    )

    train_subset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_subset = torch.utils.data.Subset(base_dataset, valid_indices)

    train_dataset = DatasetTransformer(train_subset,train_transform)
    valid_dataset = DatasetTransformer(valid_subset,valid_transform)

    num_classes = len(base_dataset.classes)
    len_dataset_train = len(train_dataset)
    len_dataset_valid = len(valid_dataset)

    if is_batch_weighted:
        # Build the train sampler (not sure it's useful having two different samplers)
        b_sampler_train = get_batch_weighted_smart_sampler(base_dataset=base_dataset,batch_size=batch_size,len_dataset=len_dataset_train, indices=train_indices)

        # Build the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=b_sampler_train,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,# <--- NEW: Keeps workers alive between epochs
            prefetch_factor=4

        )
        # Build the test sampler (not sure it's useful having two different samplers)
        b_sampler_test = get_batch_weighted_smart_sampler(base_dataset=base_dataset,batch_size=batch_size,len_dataset=len_dataset_valid, indices=valid_indices)
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_sampler=b_sampler_test,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,# <--- NEW: Keeps workers alive between epochs
            prefetch_factor=4
        )

    else:
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
            batch_size=batch_size,
            persistent_workers=True,# <--- NEW: Keeps workers alive between epochs
            prefetch_factor=4
        )

    
    input_size = tuple(train_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes



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

def get_test_dataloaders(config, use_cuda, input_transform = None, tmp_testpath=None): 

    data_config = config['data']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    
    if tmp_testpath:
        test_path = tmp_testpath
    else:
        test_path = data_config['testpath']
        
    batch_size = data_config["batch_size"]

    if not input_transform:
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

if __name__ == "__main__":
    one_image_per_class()

import os
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def get_or_create_split(targets, split_file):
    """
    Génère ou charge le split Train/Val de manière déterministe.
    Prend directement la liste des 'targets' pour éviter de re-scanner 1 million de fichiers.
    """
    if os.path.exists(split_file):
        print(f"🔄 Chargement du split statique depuis {split_file}")
        with open(split_file, 'r') as f:
            return json.load(f)
            
    print(f"🔨 Création d'un nouveau split stratifié statique ({len(targets)} images)...")
    
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.15,
        shuffle=True,
        stratify=targets,
        random_state=42  # Reproductibilité stricte
    )
    
    split_dict = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist()
    }
    
    with open(split_file, 'w') as f:
        json.dump(split_dict, f)
        
    return split_dict

def compute_class_weights(dataset):
    targets = dataset.targets
    counts = np.bincount(targets)
    counts = np.maximum(counts, 1)
    max_count = np.max(counts)
    weights = (max_count / counts) ** 0.25
    return torch.FloatTensor(weights)

def evaluate_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
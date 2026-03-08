# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn

import timm



class PretrainedModel(nn.Module):
    """
    I leave it here but, for now, this is useless 
    
    pretrained_model : the pretrained model 
    num_classes : the number of possible classes in the dataset 
    linear_input_size : the size of the tensor outputted by the pretrained model
    """
    def __init__(self, pretrained_model,num_classes:int, linear_input_size):
        self.pretrained_model = pretrained_model 
        self.num_classses = num_classes
        self.linear = nn.Linear(linear_input_size,num_classes)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.linear(x)
        return x





class PlanktonMobileNet_class(nn.Module):
    def __init__(self, num_classes=86):
        super(PlanktonMobileNet_class, self).__init__()
        
        # 1. Extracteur de caractéristiques (MobileNetV2 avec width multiplier 1.4)
        # num_classes=0 retire la tête de classification d'origine.
        # pretrained=False because we load fine-tuned weights from checkpoint immediately after
        self.feature_extractor = timm.create_model('mobilenetv2_140', pretrained=True, num_classes=0)
        
        # Récupération dynamique de la dimension du vecteur
        in_features = self.feature_extractor.num_features # 1792
        
        # 2. Classifieur hybride (Auteur + BatchNorm)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),                  # Premier Dropout de l'autrice
            nn.Linear(in_features, 600),        # Transformation affine
            nn.BatchNorm1d(600),                # <-- AJOUT : Normalisation des pré-activations
            nn.ReLU(inplace=True),              # Activation non-linéaire
            nn.Dropout(p=0.2),                  # Second Dropout de l'autrice
            nn.Linear(600, num_classes)         # Couche de classification finale
        )

    def forward(self, x, extract_features=False):
        """
        Passe avant du réseau.
        Si extract_features=True, renvoie le vecteur profond de dimension 1792.
        """

        
        # Extraction des features : [Batch, 3, 224, 224] -> [Batch, 1792]
        deep_features = self.feature_extractor(x)
        
        if extract_features:
            return deep_features
            
        # Passage dans le classifieur final : [Batch, 1792] -> [Batch, num_classes]
        logits = self.classifier(deep_features)
        return logits


def PlanktonMobileNet(cfg, input_size, num_classes):
    """
    Wrapper function for PlanktonMobileNet class to match the expected interface.
    This allows the model to be instantiated from config like other models.
    
    Args:
        cfg: Model configuration dict (may contain model-specific parameters)
        input_size: Input image size tuple (not used, but required for interface compatibility)
        num_classes: Number of output classes
    
    Returns:
        PlanktonMobileNet_class instance
    """
    return PlanktonMobileNet_class(num_classes=num_classes)
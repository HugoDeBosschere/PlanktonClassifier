# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn
import timm

class SimpleClassifier(nn.Module):
    def __init__(self, pretrained_path, num_classes=86, **kwargs):
        super().__init__()
        
        # Conserve la tête de classification standard de timm
        self.backbone = timm.create_model(pretrained_path, pretrained=True, num_classes=num_classes)
        self.pretrained_cfg = self.backbone.pretrained_cfg

    def forward(self, x, extract_features=False):
        if extract_features:
            # Extrait le tenseur avant la couche linéaire finale (pre_logits)
            features = self.backbone.forward_features(x)
            return self.backbone.forward_head(features, pre_logits=True)
        return self.backbone(x)

    def get_classifier(self):
        return self.backbone.get_classifier()

    def get_backbone(self):
        return self.backbone


class PlanktonClassifier(nn.Module):
    def __init__(self, pretrained_path, num_classes=86, **kwargs):
        super().__init__()
        
        # 1. Extracteur de caractéristiques
        self.feature_extractor = timm.create_model(pretrained_path, pretrained=True, num_classes=0)
        self.pretrained_cfg = self.feature_extractor.pretrained_cfg
        
        in_features = self.feature_extractor.num_features 
        
        # 2. Classifieur hybride (Auteur + BatchNorm)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),                  
            nn.Linear(in_features, 600),        
            nn.BatchNorm1d(600),                
            nn.ReLU(inplace=True),              
            nn.Dropout(p=0.2),                  
            nn.Linear(600, num_classes)         
        )

    def forward(self, x, extract_features=False):
        deep_features = self.feature_extractor(x)
        if extract_features:
            return deep_features
        logits = self.classifier(deep_features)
        return logits 

    def get_classifier(self):
        return self.classifier

    def get_backbone(self):
        return self.feature_extractor
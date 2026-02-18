# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn



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

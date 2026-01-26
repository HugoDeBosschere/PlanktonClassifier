# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def VanillaCNN(cfg, input_size, num_classes):

    layers = []
    cin = input_size[0]
    cout = 16
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)

def PollenNet(cfg, input_size, num_classes):

    layers = []
    cin = input_size[0]
    for k in range(1,5):
        cout = 2**(k + 4)
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(nn.MaxPool2d())
        if k != 1:
            layers.extend(nn.Dropout(0.5))

        cin = cout 
    
    layers.extend(nn.Flatten())
    layers.extend(nn.Linear(512, 512))
    layers.extend(nn.ReLU())
    layers.extend(nn.Droupout(0.5))
    layers.extend(nn.Linear(512, num_classes))
    
    return nn.Sequential(*layers)
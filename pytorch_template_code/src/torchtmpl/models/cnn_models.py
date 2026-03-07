# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn

# Local imports
from .pretrained_models import PlanktonMobileNet


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]

def conv_relu_maxpool_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        nn.BatchNorm2d(cout),
    ]

def conv_relu_maxpool_dropout_bn(cin, cout,prob):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2),
        nn.BatchNorm2d(cout),
        nn.Dropout(prob),
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

def PollenNetAbuse(cfg, input_size, num_classes):
    """
    Modèle PollenNet. 
    Nombres de paramètres : 
    Couche convolutive1 : 9 * 1 * (2**(5) + 1)
    Couche convolutive : kernel_size**2 * cin * (cout + 1) -> 9 * somme_(2<= k <=4) (2**(k +4))*(2**(k + 5) + 1) 
    Batch norm : 2 * cout : sum_(1<= k <= 5) 2 * (2**(k+4)) = 2 ** (k+5)
    Couche linéaire1 : num_features * (256  + 1) (num_features = 400 000)
    Couche linéaire2 : 256 * (86 + 1) 
    sans couche linéaire1 : res = 1575593
    """

    layers = []
    cin = input_size[0]
    print(f"cin : {cin}")
    prob_dropout = cfg["prob_droupout"]
    for k in range(1,5):
        cout = 2**(k + 4)
        if k == 1:
            layers.extend(conv_relu_maxpool_bn(cin,cout))
        if k != 1:
            layers.extend(conv_relu_maxpool_dropout_bn(cin,cout,prob_dropout))
        cin = cout 

    conv_model = nn.Sequential(*layers)
    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1) 
    print(f"nombre de features {num_features}")
    
    out_layers = []

    out_layers.append(nn.Flatten(start_dim=1))
    out_layers.append(nn.Linear(num_features, 256))
    out_layers.append(nn.ReLU())
    out_layers.append(nn.Dropout(0.5))
    out_layers.append(nn.Linear(256, num_classes))
    
    return nn.Sequential(conv_model,*out_layers)

def PollenNet(cfg, input_size, num_classes):
    """
    cfg : config fu model 
    input_size : size of the inputs 
    num_layers : number of convolutionnal layers
    """

    layers = []
    cin = input_size[0]
    print(f"cin : {cin}")
    prob_dropout = cfg["prob_droupout"]
    size_linear = cfg["size_linear"]
    num_layers = cfg["num_layers"]
    for k in range(num_layers):
        cout = 2**(k + 5)
        if k == 0:
            layers.extend(conv_relu_maxpool_bn(cin,cout))
        if k != 0:
            layers.extend(conv_relu_maxpool_dropout_bn(cin,cout,prob_dropout))
        cin = cout 


    layers.append(nn.AdaptiveAvgPool2d(1))
    conv_model = nn.Sequential(*layers)
    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1) 
    print(f"nombre de features {num_features}")
    
    out_layers = []
    
    out_layers.append(nn.Flatten(start_dim=1))
    out_layers.append(nn.Linear(num_features, size_linear))
    out_layers.append(nn.ReLU())
    out_layers.append(nn.Dropout(0.5))
    out_layers.append(nn.Linear(size_linear, num_classes))
    
    return nn.Sequential(conv_model,*out_layers)



def dummydumdum(cfg, input_size,num_classes):
    layers = []
    cin = input_size[0]
    cout = 32 
    layers.extend(conv_relu_bn(cin,32))
    conv_model = nn.Sequential(*layers)
    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1) 
    out_layers = []

    out_layers.append(nn.Flatten(start_dim=1))
    out_layers.append(nn.Linear(num_features, num_classes))

    return nn.Sequential(conv_model,*out_layers)




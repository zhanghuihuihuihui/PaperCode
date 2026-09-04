import torch
import torch.nn as nn
import torch.nn.functional as F

# 'bec','becWith','binary_cross','margin',
# 'logistic','focal','cross','marginRank',
# 'self_adversarial'


def activation_function_(act_fun,x):
    # 'sigmoid','ReLU','softmax','softmin','ELU','tanh','leaky_relu','rrelu','softsign'
    # 'PReLU','softplus','swish'
    
    if act_fun == 'sigmoid':
        x = F.sigmoid(x)
    elif act_fun == 'ReLU':
        x = F.relu(x)
    elif act_fun == 'softmax':
        x = F.softmax(x)
    elif act_fun == 'softmin':
        x = F.softmin(x)
    elif act_fun == 'ELU':
        x = F.elu(x)
    elif act_fun == 'tanh':
        x = F.tanh(x)
    elif act_fun == 'leaky_relu':
        x = F.leaky_relu(x)
    elif act_fun == 'rrelu':
        x = F.rrelu(x)
    elif act_fun == 'softsign':
        x = F.softsign(x)
    return x

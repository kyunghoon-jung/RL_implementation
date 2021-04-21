import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

class QNetwork(nn.Module):
    def __init__(self, in_dim, action_dim):
        super(QNetwork_1dim, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        
        # 각 Action에 대한 mu와 std를 구해주는 Network을 정의 
        self.mu_layer = nn.Linear(128, out_dim)     
        self.log_std_layer = nn.Linear(128, out_dim)   

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))

        # 각 Action에 대한 mu와 std를 이용하여 분포를 구하고, 그 분포에서 각 action을 sampling 함. 
        mu = torch.tanh(self.mu_layer(x)) * 2       # -2 ~ 2 로 scaling. 경험적인 매개변수.
        log_std = F.softplus(self.log_std_layer(x)) # softplus로 log_std를 구함
        std = torch.exp(log_std) 
        
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)
        
        return value
    
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

# The network structure is the only difference.
class QNetwork(nn.Module):

    def __init__(self, 
                 input_feature: ("int: input state dimension"), 
                 action_dim: ("output: action dimensions"),
                 n_atoms: ("int: The number of atoms") # Categorical Variable 
        ):

        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms

        self.linear1 = nn.Linear(input_feature, 512)
        self.linear2 = nn.Linear(512, 512) 
        self.linear3 = nn.Linear(512, action_dim*n_atoms)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.linear1(x))
        x = self.linear3(self.relu(self.linear2(x)))
        x = x.view(-1, self.action_dim, self.n_atoms)
        return F.softmax(x, dim=-1) # Shape: (batch_size, action_dim, n_atoms)
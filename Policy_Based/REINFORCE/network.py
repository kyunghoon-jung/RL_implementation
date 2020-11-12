import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dim=4, action_dim=2, hidden_dim=16, device_num=0):
        super(Policy, self).__init__()
        self.device = device_num
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
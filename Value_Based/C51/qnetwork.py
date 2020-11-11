import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

# The network structure is the only difference.
class QNetwork(nn.Module):
    
    def __init__(self, input_dim, action_dim, rand_seed=False,
                conv_channel_1=32, conv_channel_2=64, conv_channel_3=128,
                kernel_1=3, kernel_2=3, kernel_3=3, 
                stride_1=2, stride_2=2, stride_3=1,
                linear_1=512, linear_2=512, n_atoms=51):

        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        # self.seed = torch.manual_seed(rand_seed)
        self.Conv1 = nn.Conv2d(input_dim[0], conv_channel_1, (kernel_1,kernel_1), stride=stride_1)
        self.Conv2 = nn.Conv2d(conv_channel_1, conv_channel_2, (kernel_2,kernel_2), stride=stride_2)
        self.Conv3 = nn.Conv2d(conv_channel_2, conv_channel_3, (kernel_3,kernel_3), stride=stride_3)

        def calculate_conv2d_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        w, h = input_dim[1], input_dim[2]
        convw = calculate_conv2d_size(calculate_conv2d_size(calculate_conv2d_size(w,kernel_1,stride_1),
                                                            kernel_2,stride_2),
                                      kernel_3,stride_3)
        convh = calculate_conv2d_size(calculate_conv2d_size(calculate_conv2d_size(h,kernel_1,stride_1),
                                                            kernel_2,stride_2),
                                      kernel_3,stride_3)
        linear_input_size = convw * convh * conv_channel_3

        self.V_fc1 = nn.Linear(linear_input_size, linear_1)
        self.V_fc2 = nn.Linear(linear_1, n_atoms)            # Dist. of Values
        self.A_fc1 = nn.Linear(linear_input_size, linear_2) 
        self.A_fc2 = nn.Linear(linear_2, action_dim*n_atoms) # Dist. of Action-Values
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv1(x)) 
        x = self.relu(self.Conv2(x)) 
        x = self.relu(self.Conv3(x)) 
        x = x.reshape(x.shape[0], -1) 
        V = self.V_fc2(self.relu(self.V_fc1(x))).view(-1, 1, self.n_atoms) 
        A = self.A_fc2(self.relu(self.A_fc1(x))).view(-1, self.action_dim, self.n_atoms)
        Q = V + A - A.mean(dim=1, keepdim=True) 
        return F.log_softmax(Q, dim=2) # Shape: (batch_size, action_dim, n_atoms), Implementing with log-softmax for stability reason.

# Test
if __name__ == '__main__':
    state_size = (4, 84, 84)
    action_size = 10
    net = QNetwork(state_size, action_size, 
                conv_channel_1=32, conv_channel_2=64, conv_channel_3=64)
    test = torch.randn(size=(64, 4, 84, 84))
    print(net(test).shape) 
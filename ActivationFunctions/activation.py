'''

net = wx+b
o = f(net)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# Option 1: Create nn modules
class NN1(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # others
        # nn.Sigmoid
        # nn.Softmax
        # nn.TanH
        # nn.LeakyReLU


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# Option 2: use the functions directly in the forward pass
class NN2(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(NN2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))

        # others
        # torch.sigmoid
        # torch.tanH
        # F.relu
        # F.leaky_relu

        return out
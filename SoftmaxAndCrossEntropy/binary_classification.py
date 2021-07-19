'''
Linear and Sigmoid Activation function
Use BCE loss and implement sigmoid at the end

'''

import torch
import torch.nn as nn

# Binary Classification
class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) # last layer has o/p size of 1

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NN(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
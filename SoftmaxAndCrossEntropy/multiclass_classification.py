from typing import NoReturn
import torch
import torch.nn as nn


# Multiclass problem
class NN(nn.Module):
    def __init__(self, input_size, hidding_size, num_classes):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidding_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidding_size, num_classes)


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NN(input_size=28*28, hidding_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (applies softmax)

'''
Softmax = Squash the o/p between 0 and 1
        = e^(y[i])/SUM(e^(y[i])) 
        = Helps getting the probability
'''
import torch
import torch.nn as nn
import numpy as np

def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

# Numpy
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)

print(f'softmax numpy: {np.round(outputs, decimals=3)}')

# PyTorch
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)

print(f'softmax pytorch: {outputs}')

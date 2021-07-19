'''
Cross-entropy
    - The better the prediction, the lower the loss
'''

import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])


# y must be one-hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]

#  Using Numpy
Y = np.array([1, 0, 0])

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(f'Loss1 numpy: {l1:.4}')
print(f'Loss2 numpy: {l2:.4}')



# Using PyTorch
'''
n.CrossEntropyLoss applies nn.LogSoftmax + nn.LLLoss (negative log lokeligood loss)
    -> No softmax in the last layer

Y has class labels, not one-hot encoded
Y_pred has raw scores (logits), no Softmax!

'''
loss = nn.CrossEntropyLoss()

# class 0 is the correct prediction, not one-hot encoded a using np, size = n_samples x n_classes; ex: 1 x 3 (arr of arrays)
# Y = torch.tensor([0])

# size = n_samples x n_classes; ex: 3 x 3 (arr of arrays)
Y = torch.tensor([2, 0, 1]) # values here are the index, 2: index 2 (starting from 0) must hv highest value for good classification
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 2.1], [0.1, 1.0, 0.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

# To get the actual predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(f'Loss1 PyTorch: {l1.item():.4}')
print(f'Loss2 PyTorch: {l2.item():.4}')

print(f'Prediction1 PyTorch: {predictions1}')
print(f'Prediction2 PyTorch: {predictions2}')

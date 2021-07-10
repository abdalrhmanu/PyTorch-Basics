'''
Backpropagation in PyTorch

As at the end of the computational graph "nodes", we will have a loss function that we want to minize \
    with applying the gradient to this loss to the input dLoss/dx..

    The operation is using chain rule to reach the input variable.

    1) Forward pass: To compute the loss
    2) Compute local gradient at each node
    3) Backward pass: To comoute the dLoss / dWeights using the chain rule

    The loss is the error : (y_predicted-y)^2, to minimize it, we look for dLoss/dW on the backward pass

'''

import torch

# Tensors
x = torch.tensor(1.0)
y = torch.tensor(2.0)

# Weights
w = torch.tensor(1.0, requires_grad=True)


# Forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# Backward pass
loss.backward()
print(w.grad)

# Update weights
# Next forward and backward pass
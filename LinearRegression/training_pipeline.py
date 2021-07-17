# Training Pipeline
# 1. Design Model ( i/p & o/p size, forward pass)
# 2. Construct the loss and optimizer
# 3. Training loop
#   3.1 Forward pass: compute prediction
#   3.2 Backward pass: Calc gradients
#   3.3 Update weights 


"""
Linear Regression and Gradient Descent

- Prediction: Manually
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter Updates: PyTorch Optimizer

"""

import torch
import torch.nn as nn
 

# Linear regression: f = w * x (no bias)
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initializing weights
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model Prediction
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradients -> make gradients not accumulate
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

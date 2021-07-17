# PyTorch Pipeline
# 1. Design Model ( i/p & o/p size, forward pass)
# 2. Construct the loss and optimizer
# 3. Training loop
#   3.1 Forward pass: compute prediction
#   3.2 Backward pass: Calc gradients
#   3.3 Update weights


"""
Linear Regression and Gradient Descent

- Prediction: PyTorch Model
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter Updates: PyTorch Optimizer

"""

import torch
import torch.nn as nn


# Linear regression: f = w * x (no bias)
# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)

# Custom model as above
class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([model.parameters()], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradients -> make gradients not accumulate
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

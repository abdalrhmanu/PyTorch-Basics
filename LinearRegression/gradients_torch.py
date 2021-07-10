"""
Linear Regression and Gradient Descent

- Prediction: Manually
- Gradients Computation: Automatic
- Loss Computation: Manually
- Parameter Updates: Manually

"""

import torch

# Linear regression: f = w * x (no bias)
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initializing weights
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model Prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients -> make gradients not accumulate
    w.grad.zero()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

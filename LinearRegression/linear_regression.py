# PyTorch Pipeline
# 1. Design Model ( i/p & o/p size, forward pass)
# 2. Construct the loss and optimizer
# 3. Training loop
#   3.1 Forward pass: compute prediction
#   3.2 Backward pass: Calc gradients
#   3.3 Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

print("GG")

# 0) Prepare Data
# Generating regression dataset
X_numpy, Y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise= 20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))

y = torch.from_numpy(Y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)


n_samples, n_features = X.shape

# 1) Model
# 1 Layer model
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass 
    loss.backward()
    
    # update
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss={loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 0. Prepare Date
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features) 569 30

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features: features will have 0 mean, good for logistic regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# conv data to torch tensors: By default it is double
X_train = torch.from_numpy(X_train.astype(np.float32)) 
X_test = torch.from_numpy(X_test.astype(np.float32)) 
y_train = torch.from_numpy(y_train.astype(np.float32)) 
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshaping data
# will have only 1 row, making it a column vector, each value will be in one row, with only one column
y_train = y_train.view(y_train.shape[0], 1) 
y_test = y_test.view(y_test.shape[0], 1) 

# 1. Model
# Linear compination of weights and bias
# f = wx + b, sigmoid at the end


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()

        # defining layer, only one
        # takes parameters as input size and output size
        # output is logistic, only one
        self.linear = nn.Linear(n_input_features, 1)

    # x: data
    def forward(self, x):
        # liner layer
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

# will be of size 30 i/p feature x 1 o/p feature
model = LogisticRegression(n_features)

# 2. Loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() #Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # forward pass and loss calculation
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # empty the gradient as the backward function will add all gradients into the .grad attribute
    optimizer.zero_grad()

    if(epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss:{loss.item():.4}')

# 4. Evaluate model
# Evaluation shouldn't be a part of the computation graph where we want to track the history, the comp graph is the model nural architecture design

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() # Threshold here is 0.5

    # calc accuracy
    # y_test.shape[0] will return the number of test samples
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])

    print(f'accuracy: {acc:.4}')


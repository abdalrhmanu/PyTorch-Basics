'''
epoch = 1 forward and backward pass of all training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size = 20, 100/20 = 5 iterations for 1 epoch

'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# Implementing Custom Dataset

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) # all samples, skipping first col, and taking others
        self.y = torch.from_numpy(xy[:, [0]])  # all samples, only first col, [n_samples, 1]
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
# first_data = dataset[0]
# features , labels = first_data
# print(features, labels)

# We can use the data loader directly
dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True)

# Conv the data loader to iterator
# datatiter = iter(dataloader)
# data = datatiter.next()
# features, labels = data
# print(features, labels) # as batch_size = 4, we will have 4 feature vectors and class labels


# Dummy training
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)  # 4 is batch_size
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    # looping over the train loader
    for i, (input, labels) in enumerate(dataloader):
        # forward, backward, update
        if(i+1) % 5  == 0:
            print(
                f'epoch: {epoch}/{num_epochs}, step: {i+1}/{n_iterations}, input{input.shape}')

# PyTorch built-in dataset
torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco

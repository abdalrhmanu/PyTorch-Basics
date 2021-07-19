# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model Evalutation
# GPU Support


from matplotlib import image
import torch
import torch.nn as nn
import torchvision # for the dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784 # images have size of 28 x 28, which will be flatten to 1D tensor
hidden_size = 100
num_classes = 10 # 10 digits [0:9]
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root = './data', train=True,
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
    shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
    shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
# print(samples.shape, labels.shape)

'''
torch.Size([100, 1, 28, 28]) torch.Size([100])

batch size = 100
color channels = 1
image size arr = 28 x 28
labels = 100 (torch.Size[100]), so for each class label we have 1 value

'''
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

# Neural Network
class NeuralNet(nn.Module):
    # num_classes = output_size
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # Creating layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax activation function as we will use crossentropy loss later
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshaping images
        # shape : 100, 1, 28 28
        # desired: 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(
                f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

# Testing and Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        # Reshaping
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Will return value and index, we remove first value by _, 
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')


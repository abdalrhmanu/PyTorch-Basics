# Calculating gradiant using autograd. Important for model optimization
import torch

x = torch.randn(3, requires_grad=True)
# y = x + 2 # it will create the backpropagation grad function automatically
# z = y*y*2
# z = z.mean()

# # To calculate the gradient z with respect to x (dz/dx)
# z.backward() # calculates the gradient
# print(x.grad)


## Prevent PyTorch from tracking the history
# Option 1
# x.requires_grad_(False)


# Option 2
# y = x.detach()

# Option 3
# with torch.no_grad():
#     y = x + 2;


## Dummy training example
# weights = torch.ones(4, requires_grad = True)

# for epoch in range(10):
#     model_output = (weights*3).sum()
    
#     # grad values will be accumulated and written in grad attribute
#     model_output.backward() 
#     print(weights.grad)

#     # To avoid accumulation, it is important to empty the gradient 
#     weights.grad.zero_()

# Optimizers
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()

# print(x)
# print(y)
# print(z)

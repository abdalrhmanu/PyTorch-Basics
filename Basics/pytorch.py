
import torch
import numpy as np

## PyTorch deals with tensors
# Scalar value
# x = torch.empty(1)

# 1D vector with 3 elements
# x = torch.empty(3)

# 2D tensor with 3 elements each row
# x = torch.empty(2, 3)

# Tensor with random numbers
# x = torch.rand(2,2)

# Tensor with zeros
# x = torch.zeros(2, 2)

# Tensor with ones
# x = torch.ones(2, 2)

# Set with specific datatype
# x = torch.ones(2, 2, dtype=torch.int)

# Tensor from a python list
# x = torch.tensor([2.5, 0.1])


## Operations
# x = torch.rand(2, 2)
# y = torch.rand(2, 2)

# Element wise addition
# x + y
# torch.add(x, y)

# In place addition
# y.add_(x)


## Slicing
# x = torch.rand(5, 3)
# x = x[:,0]
# x = x[1, 1].item() # to get the value from the tensor, tensor must contain only one value


## Reshaping 
# x = torch.rand(4, 4)
# y = x.view(16)
# y = x.view(-1, 8)


## Converting from numpy to torch tensor and vice versa
# If you are running on CPU, both will allocate the same memory location; adjusting either will adjust the
# other automatically
# x = torch.ones(5)
# y = x.numpy()

# x = np.ones(5)
# y = torch.from_numpy(x)


## To create a tensor in GPU and apply the above operation with different memory allocation, you must have cuda toolkit for GPU installed 
# if torch.cuda.is_available():
#     device = torch.device()
#     x = torch.ones(5, device = device); # create tensor on GPU

#     # numpy can only handle cpu tensors so you can't convert torch tensor to numpy as above, must move to cpu first
#     y = x.to("cpu") # now it is on the cpu


# To be able to calculate gradient on a tensor, must make requires_grad = True, by default it is fale
# x = torch.ones(5, requires_grad= True)

# print(x)
# print(y) 

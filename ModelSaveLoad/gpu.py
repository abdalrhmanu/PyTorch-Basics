import torch
import torch.nn as nn

# Save on GPU, load on CPU
device = torch.device("cuda")
model.to(device)
model.save(model.state_dict(), PATH)

device = torch.device("cpu")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# Save on GPU, load on GPU
device = torch.device("cuda")
model.to(device)
model.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)


# Save on CPU, load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0")) # choose whatever GPU device number to load on
model.to(device)
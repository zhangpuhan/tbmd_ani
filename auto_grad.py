import torch

torch.manual_seed(7)
train_data = torch.rand((10000, 3), requires_grad=True)
add_noise = torch.randn((10000, 1))/500.0
print(train_data)
# with torch.no_grad():
train_y = torch.reshape(torch.sum(train_data**2, dim=1)**0.5, (-1, 1)) + add_noise

v = torch.reshape(torch.ones(10000), (-1, 1))
train_y.backward(v)
print(train_data.grad)
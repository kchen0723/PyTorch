import torch

dtype = torch.float
device = torch.device("cpu")

a = torch.randn(2, 3, device=device, dtype=dtype)
b = torch.randn(2, 3, device=device, dtype=dtype)

print(a)
print(b)

print(a*b)

print(a.sum())
print(a[1, 2])

print(a.max())
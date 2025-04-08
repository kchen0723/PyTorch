import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# x = torch.randn(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
print(out)

out.backward()
print(x.grad)
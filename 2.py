import torch

a = torch.zeros(2, 3)
print(a)

b = torch.ones(2, 3)
print(b)

c = torch.randn(2, 3)
print(c)

import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_numpy = torch.from_numpy(numpy_array)
print(tensor_numpy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device = device)
print(d)

e = torch.randn(2, 3)
f = torch.randn(2, 3)
print(e + f)
print("e * f")
print(e * f)

g = torch.randn(3, 2)
print(g)
print(g.t())
print(g.shape)

tensor_grad = torch.tensor([1.0], requires_grad=True)
tensor_result = tensor_grad * 2
tensor_result.backward()
print(tensor_grad.grad)
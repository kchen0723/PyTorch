import torch

a = torch.zeros(2, 3)
print(a)

b = torch.ones(2, 3)
print(b)

c = torch.randn(2, 3)

import numpy as np
np_array = np.array([[1, 2], [3, 4]])
tensor_numpy = torch.from_numpy(np_array)
print(tensor_numpy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)
print(torch.cuda.is_available())
print(d.shape)
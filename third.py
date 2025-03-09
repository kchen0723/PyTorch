import torch

grad = torch.tensor([1.0], requires_grad=True)
print(grad)
print(grad.device)
print(grad.is_cuda)
result = grad * 2
result.backward()
print(grad.grad)
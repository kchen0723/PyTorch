import torch
import numpy as np

torch.manual_seed(42)

x=torch.randn(100, 2)
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
y = x @ true_w + true_b + torch.randn(100) * 0.1
print(x[:5])
print(y[:5])

import torch.nn as nn
class LRM(nn.Module):
    def __init__(self):
        super(LRM, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = LRM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    preditions = model(x)
    loss = criterion(predictions.squeeze(), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f'{epoch + 1}, {loss.item():.4f}')
        
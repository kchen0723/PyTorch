import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
# print(model)

# x = torch.randn(1, 2)
# x = torch.tensor([[1.0, -2.0]])
# print(x)
# output = model(x)
# print("output is ")
# print(output)

criterion = nn.MSELoss()
# target = torch.tensor([[1.0]])
# loss = criterion(output, target)
# print(loss)

optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(10, 2)
y = torch.randn(10, 1)
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if(epoch % 10 == 0):
        print(f"{epoch}, loss:{loss.item():.4f}")
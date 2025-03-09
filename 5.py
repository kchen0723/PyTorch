import torch
import torch.nn as nn
import torch.optim as optim

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
# output = model(x)
# print(output)

criterion = nn.MSELoss()
# target = torch.randn(1, 1)
# loss = criterion(output, target)
# print(loss)

optimizer = optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(10, 2)
y = torch.randn(10, 1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if(epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], loss: {loss.item():.4f}')

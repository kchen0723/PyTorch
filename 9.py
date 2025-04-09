import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(100, 2)
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1
print(X[:5])
print(Y[:5])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch % 100 == 0):
        print(f"{epoch}, {loss.item():.4f}")
        print(f"now the parameters: {model.linear.weight.data.numpy()}, {model.linear.bias.data.numpy()}")
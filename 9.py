import torch
import torch.nn as nn

n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], 
                  [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid()
)

criterion = torch.nn.MSELoss()
optimizor = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(f'{loss.item():.4f}')

    optimizor.zero_grad()
    loss.backward()
    optimizor.step()

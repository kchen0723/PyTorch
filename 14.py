import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(model_dim, output_dim)
    
    def forward(self, src, target):
        src_seq_length, target_seq_length = src.size(1), target.size(1)
        src = self.embedding(src) + self.position_encoding[:, :src_seq_length]
        target = self.embedding(target) + self.position_encoding[:, :target_seq_length]
        output = self.transformer(src, target)
        output = self.fc(output)
        return output
    
input_dim = 10000
model_dim = 512
num_heads = 8
num_layers = 6
output_dim = 10000

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

src = torch.randint(0, input_dim, (10, 32))
target = torch.randint(0, input_dim, (10, 32))

output = model(src, target)
loss = criterion(output.view(-1, output_dim), target.view(-1))

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
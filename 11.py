import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

char_set = list("hello")
char_to_index = {c: i for i, c in enumerate(char_set)}
index_to_char = {i : c for i, c in enumerate(char_set)}

input_str = "hello"
target_str = "elloh"
input_data = [char_to_index[c] for c in input_str]
target_data = [char_to_index[c] for c in target_str]

input_one_hot = np.eye(len(char_set))[input_data]

inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)

input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 200
learning_rate = 0.1

class RNNModle(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModle, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.rc(out)
        return out, hidden

model = RNNModle(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

losses = []
hidden = None
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    hidden = hidden.detach()

    loss = criterion(outputs.view(-1, output_size), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if(epoch % 20 == 0):
        print(f"{epoch}, {loss.item():.4f}")

with torch.no_grad():
    test_hidden = None
    test_output, _ = model(inputs.unsqueeze(0), test_hidden)
    predicted = torch.argmax(test_output, dim=2).squeeze().numpy()
    print(f"input is {''.join(index_to_char[i] for i in input_data)}")
    print(f"predicted is: {''.join(index_to_char[i] for i in predicted)}")

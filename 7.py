import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.X_data[index], dtype=torch.float32)
        y = torch.tensor(self.Y_data[index], dtype=torch.float32)
        return x, y
    
X_data = [[1, 2],[3, 4],[5, 6],[7, 8]]
Y_data = [1, 0, 1, 0]
dataset = MyDataset(X_data, Y_data)
print(dataset[1])

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for epoch in range(1):
    for batch_index, (inputs, labels) in enumerate(dataloader):
        print(f"{batch_index}, {inputs}, {labels}")
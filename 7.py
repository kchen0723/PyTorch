import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x_data[index], dtype=torch.float32)
        y = torch.tensor(self.y_data[index], dtype=torch.float32)
        return x, y
    
x_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
y_data = [1, 0, 1, 0]
dataset = MyDataSet(x_data, y_data)

from torch.utils.data import DataLoader
dataLoader = DataLoader(dataset, batch_size = 2, shuffle = True)
for epoch in range(1):
    for batch_index, (inputs, labels) in enumerate(dataLoader):
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')
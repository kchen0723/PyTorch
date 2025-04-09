import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)
        return features, label
    
dataset = CSVDataset("runoob_pytorch_data.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for features, label in dataloader:
    print(f"features: {features}, label: {label}")

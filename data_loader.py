import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

class IndustrialIOTDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, window_size=10, train=True, transform=None):
        self.data = pd.read_csv(data_file)
        self.window_size = window_size
        self.transform = transform
        if train:
            # Assuming the last 20% of data is for testing/validation
            split_index = int(len(self.data) * 0.8)
            self.train_data = self.data.iloc[:split_index]
            self.test_data = self.data.iloc[split_index:]
        else:
            self.train_data = self.data

    def __len__(self):
        return len(self.train_data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.train_data.iloc[idx:idx+self.window_size].values
        if self.transform:
            window = self.transform(window)
        return torch.tensor(window, dtype=torch.float32)

    def get_test_data(self):
        test_windows = []
        for i in range(len(self.test_data) - self.window_size + 1):
            window = self.test_data.iloc[i:i+self.window_size].values
            if self.transform:
                window = self.transform(window)
            test_windows.append(torch.tensor(window, dtype=torch.float32))
        return test_windows

def create_dataloaders(data_file, batch_size=32, window_size=10):
    dataset = IndustrialIOTDataset(data_file, window_size=window_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Example usage:
if __name__ == "__main__":
    data_file = 'industrial_iot_data.csv'
    train_loader, test_loader = create_dataloaders(data_file)
    
    for batch in train_loader:
        print(batch)
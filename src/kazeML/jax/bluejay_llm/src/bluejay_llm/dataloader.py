from torch.utils.data import Dataset
import torch
import numpy as np
class ThePileDataset(Dataset):
    def __init__(self, path: str, max_length: int = 1024):
        self.max_length = max_length
        data = np.memmap(path, dtype=np.int16, mode="r")
        self.data = torch.tensor(data, dtype=torch.int16)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]
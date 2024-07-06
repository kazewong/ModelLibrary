from torch.utils.data import Dataset
import torch
import numpy as np


class ThePileDataset(Dataset):
    def __init__(self, path: str, max_length: int = 1024, process_id: int = 0, num_processes: int = 1):
        self.max_length = max_length
        data = np.memmap(path, dtype=np.uint16, mode="r").astype(np.int32)
        total_length = len(data)
        data = data[process_id:total_length:num_processes]
        self.data = torch.tensor(data, dtype=torch.int32)
        
    def __len__(self):
        return len(self.data)-self.max_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index:index+self.max_length], self.data[index+1:index+self.max_length+1]
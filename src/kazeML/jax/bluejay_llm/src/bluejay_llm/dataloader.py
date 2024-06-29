from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import tiktoken
import json

class ThePileDataset(Dataset):
    def __init__(self, path: str, tokenizer: tiktoken.Encoding, max_length: int = 1024):
        file = open(path, "r").readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = list(map(lambda x: self.tokenizer.encode(json.loads(x)['text']),file))
        # self.data = torch.stack(list(map(lambda x: F.pad(torch.tensor(x), pad=(0, max_length-len(x)), value=self.tokenizer.eot_token), data)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.data[index])
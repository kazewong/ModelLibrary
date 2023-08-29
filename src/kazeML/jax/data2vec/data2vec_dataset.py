from torch.utils.data import Dataset
import h5py
from jaxtyping import Array, Float

class Data2VecDataset(Dataset):
    def __init__(self, path: str, transform=None, n_example: int = 1):
        h5_file = h5py.File(path, "r")
        self.transform = transform

        self.data: h5py.Dataset = h5_file["data"]  # type: ignore

        assert isinstance(self.data, h5py.Dataset), "Data is not a dataset"

        self.n_dim = len(self.data.shape[2:])
        self.n_example = 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Float[Array, "batch"], Float[Array, "batch n_example"]]:
        sample = self.data[index]
        if self.transform != None:
            for f in self.transform:
                sample = f(sample)
        
        return sample

    @staticmethod
    def make_mask(n_mask: int) -> Array:
        raise NotImplementedError
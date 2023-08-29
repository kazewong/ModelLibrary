from torch.utils.data import Dataset
import torch
import h5py
from jaxtyping import Array, Float


class Data2VecDataset(Dataset):
    def __init__(
        self,
        path: str,
        data_length: int,
        transform=None,
        n_example: int = 1,
        mask_fraction: float = 0.1,
        mask_length: float = 0.1,
        min_masks: int = 1,
        seed: int = 0,
    ):
        h5_file = h5py.File(path, "r")
        self.transform = transform

        self.data: h5py.Dataset = h5_file["data"]  # type: ignore

        assert isinstance(self.data, h5py.Dataset), "Data is not a dataset"

        self.n_dim = len(self.data.shape[2:])
        self.n_example = n_example
        self.data_length = data_length
        self.mask_fraction = mask_fraction
        self.mask_length = mask_length
        self.min_masks = min_masks

        torch.manual_seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[
        Float[torch.Tensor, "n_channel n_size"], Float[torch.Tensor, "n_example n_size"]
    ]:
        sample = self.data[index]
        if self.transform != None:
            for f in self.transform:
                sample = f(sample)

        mask = torch.zeros((self.n_example, self.data_length))
        for i in range(self.n_example):
            mask[i] = self.make_one_mask()
        return sample, mask

    def make_one_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.data_length))

        if self.n_dim == 1:
            num_mask = int(self.mask_fraction / self.mask_length + torch.rand(1))
            num_mask = max(num_mask, self.min_masks)
            mask_length = int(self.mask_length * self.data_length)
            center_indices = torch.randint(
                0, self.data_length - mask_length, (num_mask,)
            )
            for i in range(num_mask):
                mask[center_indices[i]-mask_length//2:center_indices[i]+mask_length//2] = 1
        elif self.n_dim == 2:
            num_mask = int(self.mask_fraction / self.mask_length**2 + torch.rand(1))
            num_mask = max(num_mask, self.min_masks)
            mask_length = int(self.mask_length * self.data_length)
            center_indices = torch.randint(
                0, self.data_length - mask_length, (num_mask, 2)
            )
            for i in range(num_mask):
                mask[center_indices[i, 0]-mask_length//2:center_indices[i, 0]+mask_length//2,
                     center_indices[i, 1]-mask_length//2:center_indices[i, 1]+mask_length//2] = 1
        elif self.n_dim == 3:
            num_mask = int(self.mask_fraction / self.mask_length**3 + torch.rand(1))
            num_mask = max(num_mask, self.min_masks)
            mask_length = int(self.mask_length * self.data_length)
            center_indices = torch.randint(
                0, self.data_length - mask_length, (num_mask, 3)
            )
            for i in range(num_mask):
                mask[center_indices[i, 0]-mask_length//2:center_indices[i, 0]+mask_length//2,
                     center_indices[i, 1]-mask_length//2:center_indices[i, 1]+mask_length//2,
                     center_indices[i, 2]-mask_length//2:center_indices[i, 2]+mask_length//2] = 1
        else:
            print("Dimension not supported")
            raise NotImplementedError

        return mask
